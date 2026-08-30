from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from modal_gaussians.data.sequence import validate_image_mask_sequence
from modal_gaussians.data.stabilization import (
    StabilizationSettings,
    stabilize_to_reference,
    write_stabilized_sequence,
)
from modal_gaussians.flow.artifact import (
    FlowAnalysisArtifact,
    FlowAnalysisArrays,
    publish_flow_analysis_artifact,
)
from modal_gaussians.flow.estimation import (
    FARNEBACK_PARAMETERS,
    compute_reference_to_frame_flow,
    weighted_gaussian_smooth,
)
from modal_gaussians.flow.spectrum import temporal_rfft


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
    settings = config or FlowAnalysisConfig()
    _validate_config(settings)
    output_path = Path(output_dir).expanduser()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Flow artifact target already exists: {output_path}")
    sequence = validate_image_mask_sequence(
        image_dir=image_dir,
        mask_dir=mask_dir,
        fps_hz=fps_hz,
        reference_frame_name=reference_frame_name,
    )
    frames_gray, masks = sequence.read_arrays()
    stabilization_result = None
    if settings.stabilize:
        stabilization_result = stabilize_to_reference(
            frames_gray,
            masks,
            sequence.reference_frame_index,
            settings.stabilization_settings,
        )
        processed_gray = stabilization_result.frames_gray
        processed_masks = stabilization_result.masks
    else:
        processed_gray = frames_gray
        processed_masks = masks

    mask_union = np.any(processed_masks, axis=0)
    if stabilization_result is not None:
        mask_union &= np.all(stabilization_result.valid_mask, axis=0)
    if not np.any(mask_union):
        raise ValueError("The union of all processed masks is empty")

    flow = compute_reference_to_frame_flow(
        processed_gray, sequence.reference_frame_index
    )
    if settings.smoothing == "weighted-gaussian":
        flow = weighted_gaussian_smooth(
            flow,
            processed_gray[sequence.reference_frame_index],
            mask_union,
            sigma_b_px=settings.sigma_b_px,
            sigma_c_px=settings.sigma_c_px,
            gradient_pyramid_weights=settings.gradient_pyramid_weights,
            epsilon=settings.smoothing_epsilon,
        )

    spectrum = temporal_rfft(
        flow,
        block_width=settings.fft_block_width,
    )
    arrays = FlowAnalysisArrays(
        flow=flow,
        mask_union=mask_union,
        spectrum=spectrum,
    )
    inputs = {
        "sequence": {
            "image_directory": str(sequence.image_dir),
            "mask_directory": str(sequence.mask_dir),
        }
    }

    write_stabilized = None
    if stabilization_result is not None:
        write_stabilized = lambda directory: write_stabilized_sequence(
            directory, sequence, stabilization_result
        )
    return publish_flow_analysis_artifact(
        output_path,
        arrays=arrays,
        inputs=inputs,
        parameters=settings.parameters(),
        frame_names=list(sequence.frame_names),
        reference_frame_name=sequence.reference_frame_name,
        reference_frame_index=sequence.reference_frame_index,
        fps_hz=sequence.fps_hz,
        command=list(command),
        write_stabilized=write_stabilized,
    )
