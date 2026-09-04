from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np

from modal_gaussians.data.sequence import read_binary_mask, validate_image_mask_sequence
from modal_gaussians.data.stabilization import (
    StabilizationSettings,
    stabilize_image_sequence,
    write_stabilized_sequence,
)
from modal_gaussians.flow.artifact import (
    FlowAnalysisArtifact,
    FlowAnalysisArrays,
    publish_flow_analysis_artifact,
)
from modal_gaussians.flow.estimation import (
    FARNEBACK_PARAMETERS,
    compute_farneback_pair,
    make_gaussian_smoother,
)
from modal_gaussians.flow.spectrum import temporal_rfft
from modal_gaussians.flow.storage import BLOCK_BYTES, create_array
from modal_gaussians.progress import Progress, report_progress


SMOOTHING_METHODS = ("none", "weighted-gaussian")


@dataclass(frozen=True)
class FlowAnalysisConfig:
    stabilize: bool = False
    stabilization_settings: StabilizationSettings = field(
        default_factory=StabilizationSettings
    )
    smoothing: str = "none"
    sigma_b_px: float = 3.0
    sigma_c_px: float = 0.0
    gradient_pyramid_weights: tuple[float, ...] = (0.5, 0.3, 0.2)
    smoothing_epsilon: float = 1e-6
    fft_block_width: int = 128

    def parameters(self) -> dict[str, Any]:
        stabilization: dict[str, Any] = {"method": "none"}
        if self.stabilize:
            stabilization = {
                "method": "reference-background-lk-ransac-homography",
                "implementation_version": 1,
                "opencv_version": cv2.__version__,
                "settings": asdict(self.stabilization_settings),
            }
        smoothing: dict[str, Any] = {"method": self.smoothing}
        if self.smoothing == "weighted-gaussian":
            smoothing.update(
                {
                    "sigma_b_px": float(self.sigma_b_px),
                    "sigma_c_px": float(self.sigma_c_px),
                    "gradient_pyramid_weights": [
                        float(value) for value in self.gradient_pyramid_weights
                    ],
                    "epsilon": float(self.smoothing_epsilon),
                    "contrast_source": "processed-reference-frame",
                    "component_weights": "absolute-multiscale-sobel",
                    "implementation_version": 1,
                    "opencv_version": cv2.__version__,
                }
            )
        return {
            "stabilization": stabilization,
            "mask_union": {"method": "union-of-processed-frame-masks"},
            "flow": {
                "method": "opencv-farneback-reference-to-frame",
                "implementation_version": 1,
                "opencv_version": cv2.__version__,
                "parameters": dict(FARNEBACK_PARAMETERS),
            },
            "smoothing": smoothing,
            "fft": {
                "detrend": "temporal-mean",
                "window": "hann-symmetric",
                "transform": "numpy-rfft",
                "implementation_version": 1,
                "numpy_version": np.__version__,
                "block_width": int(self.fft_block_width),
            },
        }


def _validate_config(config: FlowAnalysisConfig) -> None:
    if config.smoothing not in SMOOTHING_METHODS:
        raise ValueError(
            f"Unknown smoothing method {config.smoothing!r}; "
            f"expected one of {SMOOTHING_METHODS}"
        )
    if config.fft_block_width <= 0:
        raise ValueError("fft_block_width must be positive")
    if config.smoothing == "weighted-gaussian":
        if not np.isfinite(config.sigma_b_px) or config.sigma_b_px <= 0.0:
            raise ValueError("sigma_b_px must be finite and positive")
        if not np.isfinite(config.sigma_c_px) or config.sigma_c_px < 0.0:
            raise ValueError("sigma_c_px must be finite and non-negative")
        weights = np.asarray(config.gradient_pyramid_weights, dtype=np.float64)
        if (
            weights.ndim != 1
            or weights.size == 0
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError(
                "gradient_pyramid_weights must be a non-empty, finite, "
                "non-negative vector with positive sum"
            )
        if not np.isfinite(config.smoothing_epsilon) or config.smoothing_epsilon <= 0.0:
            raise ValueError("smoothing_epsilon must be finite and positive")


def run_flow_analysis(
    *,
    image_dir: str | Path,
    mask_dir: str | Path,
    fps_hz: float,
    reference_frame_name: str,
    output_dir: str | Path,
    config: FlowAnalysisConfig | None = None,
    command: Sequence[str] = (),
) -> FlowAnalysisArtifact:
    """Generate compressed flow and full rFFT without retaining the video in RAM."""
    settings = config or FlowAnalysisConfig()
    _validate_config(settings)
    output_path = Path(output_dir).expanduser()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Flow artifact target already exists: {output_path}")
    report_progress("flow: validating image/mask sequence")
    sequence = validate_image_mask_sequence(
        image_dir=image_dir,
        mask_dir=mask_dir,
        fps_hz=fps_hz,
        reference_frame_name=reference_frame_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".flow-stabilization-", dir=output_path.parent) as scratch:
        stabilization_result = None
        if settings.stabilize:
            report_progress("flow: streaming stabilization to temporary compressed arrays")
            stabilization_result = stabilize_image_sequence(
                sequence, settings.stabilization_settings, Path(scratch)
            )

        shape_hw = (sequence.height, sequence.width)
        mask_union = np.zeros(shape_hw, dtype=bool)
        valid_intersection = np.ones(shape_hw, dtype=bool)
        for index in range(sequence.frame_count):
            if stabilization_result is None:
                mask_union |= read_binary_mask(sequence.mask_paths[index], shape_hw)
            else:
                mask_union |= stabilization_result.masks[index]
                valid_intersection &= stabilization_result.valid_mask[index]
        mask_union &= valid_intersection
        if not np.any(mask_union):
            raise ValueError("The union of all processed masks is empty")

        def read_gray(index: int) -> np.ndarray:
            """Read just one original or stabilized grayscale frame."""

            if stabilization_result is None:
                return sequence.read_frame(index)[0]
            return np.asarray(stabilization_result.frames_gray[index])

        reference = read_gray(sequence.reference_frame_index)
        smooth = None
        if settings.smoothing == "weighted-gaussian":
            smooth = make_gaussian_smoother(
                reference, mask_union, sigma_b_px=settings.sigma_b_px,
                sigma_c_px=settings.sigma_c_px,
                gradient_pyramid_weights=settings.gradient_pyramid_weights,
                epsilon=settings.smoothing_epsilon,
            )

        def build_arrays(directory: Path) -> FlowAnalysisArrays:
            """Estimate/smooth small frame batches, then write full-frequency FFT tiles."""

            shape = (sequence.frame_count, *shape_hw, 2)
            flow = create_array(directory / "flow.zarr", shape, np.float32)
            batch_size = max(1, min(32, BLOCK_BYTES // (sequence.height * sequence.width * 8)))
            progress = Progress("flow Farneback + smoothing", sequence.frame_count, unit="frames")
            for start in range(0, sequence.frame_count, batch_size):
                stop = min(sequence.frame_count, start + batch_size)
                batch = np.empty((stop - start, *shape_hw, 2), dtype=np.float32)
                for index in range(start, stop):
                    frame_flow = (
                        np.zeros((*shape_hw, 2), dtype=np.float32)
                        if index == sequence.reference_frame_index
                        else compute_farneback_pair(reference, read_gray(index))
                    )
                    batch[index - start] = frame_flow if smooth is None else smooth(frame_flow)
                    del frame_flow
                flow[start:stop] = batch
                progress.update(stop)
                del batch
            spectrum = create_array(
                directory / "spectrum.zarr",
                (sequence.frame_count // 2 + 1, *shape_hw, 2), np.complex64,
            )
            temporal_rfft(flow, block_width=settings.fft_block_width, output=spectrum)
            return FlowAnalysisArrays(flow=flow, mask_union=mask_union, spectrum=spectrum)

        write_stabilized = None
        if stabilization_result is not None:
            write_stabilized = lambda directory: write_stabilized_sequence(
                directory, sequence, stabilization_result
            )
        return publish_flow_analysis_artifact(
            output_path, arrays=build_arrays,
            inputs={"sequence": {
                "image_directory": str(sequence.image_dir),
                "mask_directory": str(sequence.mask_dir),
            }},
            parameters=settings.parameters(), frame_names=list(sequence.frame_names),
            reference_frame_name=sequence.reference_frame_name,
            reference_frame_index=sequence.reference_frame_index,
            fps_hz=sequence.fps_hz, command=list(command),
            write_stabilized=write_stabilized,
        )
