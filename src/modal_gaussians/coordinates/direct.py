"""Ridge initialization of independent per-recording modal coefficients."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from scipy.linalg import solve_triangular
from modal_gaussians.flow.storage import DenseArray, read_pixels
from modal_gaussians.common.progress import Progress
SOLVER_CONVENTION = {
    "solver": "rendered_projection_ridge_v1",
    "view_coupling": "independent_per_view",
    "observation": "reference_to_frame_optical_flow_at_rendered_design_pixels",
    "reference_subtraction": "subtract_reference_flow_and_reference_coordinate",
    "gauge": "per_view_temporal_mean_zero",
    "coordinate_packing": "[real(q_0),imag(q_0),real(q_1),imag(q_1),...]",
    "design_packing": "[real(J_phi_k),-imag(J_phi_k)]",
    "mode_pair_normalization": "rms_over_two_pixel_components",
    "ridge": "ridge_relative_times_identity_in_normalized_mode_pair_space",
    "frame_solve": "independent_with_one_reused_gram_cholesky_per_view",
}
SAMPLE_BLOCK_SIZE = 4096


def mode_pair_scales(diagonal, sample_count):
    """Pixel-space RMS shared by ridge initialization and zero-start refinement."""
    diagonal = np.asarray(diagonal, np.float64)
    if diagonal.ndim != 1 or len(diagonal) % 2 or not len(diagonal) or sample_count < 1 or np.any(diagonal < 0):
        raise ValueError('Invalid modal design diagonal')
    scales = np.sqrt((diagonal[0::2]+diagonal[1::2]) / (2*sample_count))
    if not np.isfinite(scales).all():
        raise ValueError('Direct-coordinate mode-pair scales are non-finite')
    scales[scales <= np.finfo(np.float64).eps] = 1.
    return scales

DIRECT_COORDINATES_FORMAT = "modal_gaussians.direct_modal_coordinates"
@dataclass(frozen=True)
class DirectCoordinateConfig:
    """Hold the accepted relative ridge and frame batching settings."""

    ridge_relative: float = 1.0e-4
    frame_chunk_size: int = 64

    def validate(self) -> None:
        """Reject non-positive ridge or frame batching values."""

        if not math.isfinite(self.ridge_relative) or self.ridge_relative <= 0.0:
            raise ValueError("Direct-coordinate ridge_relative must be positive")
        if (
            isinstance(self.frame_chunk_size, bool)
            or not isinstance(self.frame_chunk_size, int)
            or self.frame_chunk_size <= 0
        ):
            raise ValueError("Direct-coordinate frame_chunk_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the only numerical settings of the direct solve."""

        return {
            "ridge_relative": self.ridge_relative,
            "frame_chunk_size": self.frame_chunk_size,
        }

@dataclass(frozen=True)
class DirectCoordinateViewInput:
    """Bind one ordered rendered-design view to its exact flow artifact."""

    label: str
    flow_artifact: Path

@dataclass(frozen=True)
class DirectModalCoordinatesArtifact:
    """Represent one validated mmap-backed direct-coordinate artifact."""

    path: Path
    manifest: dict[str, Any]
    coordinates: np.ndarray
    diagnostics: dict[str, np.ndarray]

def _canonical_json(value: Any) -> bytes:
    """Encode one path-independent identity payload deterministically."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")

def _sha256_file(path: Path) -> str:
    """Hash one artifact file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _validate_modes(modes: Any) -> None:
    """Validate the ordered greedy mode prefix and frequencies."""

    if not isinstance(modes, list) or not modes:
        raise ValueError("Direct coordinates must contain ordered modes")
    candidates: list[int] = []
    for slot, mode in enumerate(modes):
        if not isinstance(mode, dict) or mode.get("mode_slot") != slot:
            raise ValueError("Direct-coordinate mode slots must be contiguous")
        candidate = mode.get("candidate_index")
        frequency = float(mode.get("frequency_hz", np.nan))
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate < 0
            or not math.isfinite(frequency)
            or frequency <= 0.0
        ):
            raise ValueError("Direct-coordinate mode metadata is invalid")
        candidates.append(candidate)
    if len(set(candidates)) != len(candidates):
        raise ValueError("Direct-coordinate candidate indices must be unique")

def _flow_matrix(
    flow: DenseArray,
    pixels: np.ndarray,
    start: int,
    end: int,
    reference_index: int,
) -> np.ndarray:
    """Sample reference-relative `(u,v)` flow as `[2P,B]` float64."""

    values = read_pixels(flow, slice(start, end), pixels).astype(np.float64)
    reference = read_pixels(flow, reference_index, pixels).astype(np.float64)
    values -= reference[None]
    if not np.isfinite(values).all():
        raise ValueError("Flow contains NaN or Inf at rendered-design pixels")
    return values.transpose(1, 2, 0).reshape(2 * len(pixels), end - start)

def solve_direct_coordinates_view(
    *,
    design: np.ndarray,
    pixels_xy: np.ndarray,
    flow: DenseArray,
    reference_frame_index: int,
    fps_hz: float,
    frequencies_hz: np.ndarray,
    config: DirectCoordinateConfig | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    """Fit one view with one normalized Gram/Cholesky reused across all frames."""

    settings = config or DirectCoordinateConfig()
    settings.validate()
    design_value = np.asarray(design)
    pixels = np.asarray(pixels_xy, dtype=np.int64)
    flow_value = flow
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if design_value.ndim != 3 or design_value.shape[1] != 2:
        raise ValueError("Direct-coordinate design must be [P,2,2K]")
    sample_count, _, column_count = design_value.shape
    if column_count < 2 or column_count % 2:
        raise ValueError("Direct-coordinate design must contain mode pairs")
    mode_count = column_count // 2
    if pixels.shape != (sample_count, 2):
        raise ValueError("Direct-coordinate pixels do not match design samples")
    if (
        flow_value.ndim != 4
        or flow_value.shape[-1] != 2
        or flow_value.shape[0] < 3
        or reference_frame_index < 0
        or reference_frame_index >= flow_value.shape[0]
    ):
        raise ValueError("Direct-coordinate flow or reference index is invalid")
    if frequencies.shape != (mode_count,) or not np.all(
        np.isfinite(frequencies) & (frequencies > 0.0)
    ):
        raise ValueError("Direct-coordinate frequencies are invalid")
    if not math.isfinite(fps_hz) or fps_hz <= 0.0:
        raise ValueError("Direct-coordinate FPS must be positive")
    height, width = flow_value.shape[1:3]
    if (
        np.any(pixels[:, 0] < 0)
        or np.any(pixels[:, 0] >= width)
        or np.any(pixels[:, 1] < 0)
        or np.any(pixels[:, 1] >= height)
    ):
        raise ValueError("Direct-coordinate pixels leave the flow image")

    normalizer = float(2 * sample_count)
    raw_gram = np.zeros((column_count, column_count), dtype=np.float64)
    for lower in range(0, sample_count, SAMPLE_BLOCK_SIZE):
        block = np.asarray(
            design_value[lower : lower + SAMPLE_BLOCK_SIZE], dtype=np.float64
        ).reshape(-1, column_count)
        if not np.isfinite(block).all():
            raise ValueError("Direct-coordinate design contains NaN or Inf")
        raw_gram += block.T @ block
    pair_scales = mode_pair_scales(np.diag(raw_gram), sample_count)
    column_scales = np.repeat(pair_scales, 2)
    normalized_gram = raw_gram / normalizer
    normalized_gram /= column_scales[:, None]
    normalized_gram /= column_scales[None, :]
    system = normalized_gram + settings.ridge_relative * np.eye(column_count)
    cholesky = np.linalg.cholesky(system)
    frame_count = flow_value.shape[0]
    relative = np.empty((frame_count, mode_count), dtype=np.complex128)
    progress = Progress("direct coordinate solve", frame_count, unit="frames")
    for start in range(0, frame_count, settings.frame_chunk_size):
        end = min(start + settings.frame_chunk_size, frame_count)
        rhs = np.zeros((column_count, end - start), dtype=np.float64)
        for lower in range(0, sample_count, SAMPLE_BLOCK_SIZE):
            upper = min(lower + SAMPLE_BLOCK_SIZE, sample_count)
            block = np.asarray(design_value[lower:upper], dtype=np.float64).reshape(
                -1, column_count
            )
            block /= column_scales[None]
            observed = _flow_matrix(
                flow_value,
                pixels[lower:upper],
                start,
                end,
                reference_frame_index,
            )
            rhs += block.T @ observed
        rhs /= normalizer
        intermediate = solve_triangular(
            cholesky, rhs, lower=True, check_finite=False
        )
        scaled = solve_triangular(
            cholesky.T, intermediate, lower=False, check_finite=False
        )
        solution = scaled / column_scales[:, None]
        relative[start:end] = solution[0::2].T + 1j * solution[1::2].T
        progress.update(end)
    if not np.isfinite(relative).all():
        raise ValueError("Direct-coordinate solve produced NaN or Inf")
    relative -= relative[reference_frame_index : reference_frame_index + 1]
    coordinates = relative - np.mean(relative, axis=0, keepdims=True)
    coordinates = coordinates.astype(np.complex64).astype(np.complex128)

    return coordinates.astype(np.complex64), {"mode_pair_scales": pair_scales}

def load_direct_modal_coordinates(path):
    from modal_gaussians.coordinates.preparation import load_initial_coordinates
    return load_initial_coordinates(path)
