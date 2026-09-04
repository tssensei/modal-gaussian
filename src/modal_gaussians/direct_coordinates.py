"""Per-view direct modal coordinates fitted to rendered-design optical flow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from modal_gaussians.progress import Progress, report_progress

from modal_gaussians.numpy_io import save_named_arrays

from modal_gaussians import __version__
from modal_gaussians.flow.storage import DenseArray, read_pixels
from modal_gaussians.flow.artifact import (
    FlowAnalysisArtifact,
    flow_artifact_identity,
    load_flow_analysis_artifact,
)
from modal_gaussians.rendered_design import (
    RenderedModalDesignArtifact,
    load_rendered_modal_design,
)


DIRECT_COORDINATES_FORMAT = "modal_gaussians.direct_modal_coordinates"
DIRECT_COORDINATES_VERSION = 1
COORDINATES_FILENAME = "coordinates.npy"
DIAGNOSTICS_FILENAME = "diagnostics.npz"
COORDINATES_DTYPE = np.dtype(np.complex64)
SAMPLE_BLOCK_SIZE = 65_536

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

DIAGNOSTIC_DTYPES = {
    "frame_view_index": np.dtype(np.int64),
    "frame_local_index": np.dtype(np.int64),
    "frame_times_sec": np.dtype(np.float64),
    "reference_local_index": np.dtype(np.int64),
    "mode_pair_scales": np.dtype(np.float64),
    "singular_values": np.dtype(np.float64),
    "numerical_rank": np.dtype(np.int64),
    "condition_number": np.dtype(np.float64),
    "ridge_condition_number": np.dtype(np.float64),
    "per_frame_flow_rmse": np.dtype(np.float64),
    "per_frame_relative_residual": np.dtype(np.float64),
    "per_frame_flow_r2": np.dtype(np.float64),
    "view_residual_sum_squares": np.dtype(np.float64),
    "view_flow_sum_squares": np.dtype(np.float64),
    "view_flow_rmse": np.dtype(np.float64),
    "view_relative_residual": np.dtype(np.float64),
    "view_flow_r2": np.dtype(np.float64),
    "reference_residual_norm": np.dtype(np.float64),
    "coordinate_rms": np.dtype(np.float64),
    "coordinate_max": np.dtype(np.float64),
    "coordinate_temporal_mean_abs": np.dtype(np.float64),
    "dominant_signed_frequency_hz": np.dtype(np.float64),
    "assigned_frequency_energy_ratio": np.dtype(np.float64),
    "cross_mode_correlation": np.dtype(np.float64),
}


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


def _arrays_identity(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash diagnostic names, dtypes, shapes, and values canonically."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select the scientific inputs, solver, frame map, and output identities."""

    return {
        "format": DIRECT_COORDINATES_FORMAT,
        "version": DIRECT_COORDINATES_VERSION,
        "rendered_design_identity": manifest["rendered_design_identity"],
        "completed_modes_identity": manifest["completed_modes_identity"],
        "modes": manifest["modes"],
        "views": manifest["views"],
        "solver": manifest["solver"],
        "settings": manifest["settings"],
        "quality_gate": manifest["quality_gate"],
        "counts": manifest["counts"],
        "coordinates": manifest["coordinates"],
        "diagnostics": {
            "arrays": manifest["diagnostics"]["arrays"],
            "arrays_identity": manifest["diagnostics"]["arrays_identity"],
        },
        "overall": manifest["overall"],
    }


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


def _expected_diagnostic_shapes(
    view_count: int, frame_count: int, mode_count: int
) -> dict[str, tuple[int, ...]]:
    """Return the fixed C17 diagnostic shapes for one artifact."""

    return {
        "frame_view_index": (frame_count,),
        "frame_local_index": (frame_count,),
        "frame_times_sec": (frame_count,),
        "reference_local_index": (view_count,),
        "mode_pair_scales": (view_count, mode_count),
        "singular_values": (view_count, 2 * mode_count),
        "numerical_rank": (view_count,),
        "condition_number": (view_count,),
        "ridge_condition_number": (view_count,),
        "per_frame_flow_rmse": (frame_count,),
        "per_frame_relative_residual": (frame_count,),
        "per_frame_flow_r2": (frame_count,),
        "view_residual_sum_squares": (view_count,),
        "view_flow_sum_squares": (view_count,),
        "view_flow_rmse": (view_count,),
        "view_relative_residual": (view_count,),
        "view_flow_r2": (view_count,),
        "reference_residual_norm": (view_count,),
        "coordinate_rms": (view_count, mode_count),
        "coordinate_max": (view_count, mode_count),
        "coordinate_temporal_mean_abs": (view_count, mode_count),
        "dominant_signed_frequency_hz": (view_count, mode_count),
        "assigned_frequency_energy_ratio": (view_count, mode_count),
        "cross_mode_correlation": (view_count, mode_count, mode_count),
    }


def _validate_diagnostics(
    arrays: Mapping[str, np.ndarray],
    *,
    views: Sequence[Mapping[str, Any]],
    frame_count: int,
    mode_count: int,
) -> None:
    """Validate frame partitions, numerical diagnostics, and temporal gauge."""

    if set(arrays) != set(DIAGNOSTIC_DTYPES):
        raise ValueError("Direct-coordinate diagnostic fields are invalid")
    shapes = _expected_diagnostic_shapes(len(views), frame_count, mode_count)
    for name, dtype in DIAGNOSTIC_DTYPES.items():
        value = arrays[name]
        if value.dtype != dtype or value.shape != shapes[name]:
            raise ValueError(f"Direct-coordinate diagnostic {name} is invalid")
        if value.dtype.kind == "f":
            if name == "condition_number":
                if np.isnan(value).any():
                    raise ValueError("Direct-coordinate condition numbers contain NaN")
            elif not np.isfinite(value).all():
                raise ValueError(f"Direct-coordinate diagnostic {name} is non-finite")
    frame_views = arrays["frame_view_index"]
    frame_local = arrays["frame_local_index"]
    frame_times = arrays["frame_times_sec"]
    references = arrays["reference_local_index"]
    offset = 0
    for index, view in enumerate(views):
        count = int(view["frame_count"])
        if view.get("frame_offset") != offset or count < 3:
            raise ValueError("Direct-coordinate view frame partition is invalid")
        rows = slice(offset, offset + count)
        fps = float(view["fps_hz"])
        reference = int(view["reference_frame_index"])
        if (
            not np.all(frame_views[rows] == index)
            or not np.array_equal(frame_local[rows], np.arange(count))
            or not np.allclose(
                frame_times[rows], np.arange(count, dtype=np.float64) / fps
            )
            or references[index] != reference
        ):
            raise ValueError("Direct-coordinate frame metadata is inconsistent")
        offset += count
    if offset != frame_count:
        raise ValueError("Direct-coordinate views do not cover all frames")
    if np.any(arrays["mode_pair_scales"] <= 0.0):
        raise ValueError("Direct-coordinate mode-pair scales must be positive")
    if np.any(arrays["assigned_frequency_energy_ratio"] < -1.0e-12) or np.any(
        arrays["assigned_frequency_energy_ratio"] > 1.0 + 1.0e-12
    ):
        raise ValueError("Assigned-frequency energy ratios leave [0,1]")


def load_direct_modal_coordinates(
    path: str | Path,
) -> DirectModalCoordinatesArtifact:
    """Load and strictly validate one direct modal-coordinate artifact."""

    root = Path(path).expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    coordinates_path = root / COORDINATES_FILENAME
    diagnostics_path = root / DIAGNOSTICS_FILENAME
    if (
        not manifest_path.is_file()
        or not coordinates_path.is_file()
        or not diagnostics_path.is_file()
    ):
        raise FileNotFoundError(f"Incomplete direct-coordinate artifact: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != DIRECT_COORDINATES_FORMAT:
        raise ValueError("Unsupported direct-coordinate format")
    if manifest.get("version") != DIRECT_COORDINATES_VERSION:
        raise ValueError("Unsupported direct-coordinate version")
    if manifest.get("solver") != SOLVER_CONVENTION:
        raise ValueError("Direct-coordinate solver convention is unsupported")
    if manifest.get("quality_gate") != {
        "required": True,
        "status": "direct_coordinates_candidate_unapproved",
        "inherited_from": "rendered_design_candidate_unapproved",
    }:
        raise ValueError("Direct coordinates must inherit the design quality gate")
    for name in ("rendered_design_identity", "completed_modes_identity"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"Direct-coordinate {name} is invalid")
    _validate_modes(manifest.get("modes"))
    settings = manifest.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Direct-coordinate settings are invalid")
    config = DirectCoordinateConfig(
        ridge_relative=float(settings["ridge_relative"]),
        frame_chunk_size=int(settings["frame_chunk_size"]),
    )
    config.validate()
    if settings != config.to_dict():
        raise ValueError("Direct-coordinate settings contain unsupported fields")
    views = manifest.get("views")
    counts = manifest.get("counts")
    if not isinstance(views, list) or not views or not isinstance(counts, dict):
        raise ValueError("Direct-coordinate views or counts are invalid")
    view_count = int(counts.get("views", -1))
    frame_count = int(counts.get("frames", -1))
    mode_count = int(counts.get("modes", -1))
    if view_count != len(views) or mode_count != len(manifest["modes"]) or frame_count < 3:
        raise ValueError("Direct-coordinate counts disagree with metadata")
    labels: list[str] = []
    for index, view in enumerate(views):
        if not isinstance(view, dict) or view.get("index") != index:
            raise ValueError("Direct-coordinate views must be ordered")
        label = view.get("label")
        names = view.get("frame_names")
        reference = view.get("reference_frame_index")
        raw_fps = view.get("fps_hz")
        fps = (
            float(raw_fps)
            if isinstance(raw_fps, (int, float)) and not isinstance(raw_fps, bool)
            else float("nan")
        )
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(names, list)
            or len(names) != view.get("frame_count")
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)
            or isinstance(reference, bool)
            or not isinstance(reference, int)
            or reference < 0
            or reference >= len(names)
            or not math.isfinite(fps)
            or fps <= 0.0
        ):
            raise ValueError("Direct-coordinate view frame metadata is invalid")
        if names[reference] != view.get("reference_frame_name"):
            raise ValueError("Direct-coordinate reference frame metadata differs")
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise ValueError("Direct-coordinate view labels must be unique")
    coordinate_record = manifest.get("coordinates")
    diagnostic_record = manifest.get("diagnostics")
    if not isinstance(coordinate_record, dict) or not isinstance(diagnostic_record, dict):
        raise ValueError("Direct-coordinate file metadata is invalid")
    if (
        coordinate_record.get("file") != COORDINATES_FILENAME
        or coordinate_record.get("dtype") != COORDINATES_DTYPE.name
        or coordinate_record.get("shape") != [frame_count, mode_count]
        or coordinate_record.get("sha256") != _sha256_file(coordinates_path)
        or diagnostic_record.get("file") != DIAGNOSTICS_FILENAME
        or diagnostic_record.get("sha256") != _sha256_file(diagnostics_path)
    ):
        raise ValueError("Direct-coordinate file metadata or SHA-256 is invalid")
    coordinates = np.load(coordinates_path, mmap_mode="r", allow_pickle=False)
    if (
        coordinates.dtype != COORDINATES_DTYPE
        or coordinates.shape != (frame_count, mode_count)
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("Direct-coordinate array is invalid")
    with np.load(diagnostics_path, allow_pickle=False) as archive:
        diagnostics = {name: archive[name] for name in archive.files}
    metadata = {
        name: {"dtype": value.dtype.name, "shape": list(value.shape)}
        for name, value in diagnostics.items()
    }
    if diagnostic_record.get("arrays") != metadata:
        raise ValueError("Direct-coordinate diagnostic metadata differs")
    if diagnostic_record.get("arrays_identity") != _arrays_identity(diagnostics):
        raise ValueError("Direct-coordinate diagnostic identity differs")
    _validate_diagnostics(
        diagnostics,
        views=views,
        frame_count=frame_count,
        mode_count=mode_count,
    )
    for view in views:
        lower = int(view["frame_offset"])
        upper = lower + int(view["frame_count"])
        mean = np.abs(np.mean(coordinates[lower:upper].astype(np.complex128), axis=0))
        scale = np.maximum(
            np.sqrt(np.mean(np.abs(coordinates[lower:upper]) ** 2, axis=0)), 1.0
        )
        if np.any(mean > 5.0e-6 * scale):
            raise ValueError("Direct-coordinate temporal mean-zero gauge is violated")
    expected_identity = hashlib.sha256(
        _canonical_json(_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("direct_coordinates_identity") != expected_identity:
        raise ValueError("Direct-coordinate identity differs from its contents")
    return DirectModalCoordinatesArtifact(root, manifest, coordinates, diagnostics)


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


def _spectral_diagnostics(
    coordinates: np.ndarray,
    fps_hz: float,
    assigned_frequencies_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure recovered coordinate frequency assignment and cross-mode leakage."""

    frame_count, mode_count = coordinates.shape
    spectrum = np.fft.fft(coordinates, axis=0)
    frequencies = np.fft.fftfreq(frame_count, d=1.0 / fps_hz)
    energy = np.abs(spectrum) ** 2
    energy[0] = 0.0
    dominant = np.zeros(mode_count, dtype=np.float64)
    assigned_ratio = np.zeros(mode_count, dtype=np.float64)
    for slot, assigned in enumerate(assigned_frequencies_hz):
        total = float(np.sum(energy[:, slot]))
        if total <= np.finfo(np.float64).eps:
            continue
        dominant[slot] = float(frequencies[int(np.argmax(energy[:, slot]))])
        positive = int(np.argmin(np.abs(frequencies - assigned)))
        negative = int(np.argmin(np.abs(frequencies + assigned)))
        bins = np.unique(np.asarray([positive, negative], dtype=np.int64))
        assigned_ratio[slot] = float(np.sum(energy[bins, slot]) / total)
    norms = np.sqrt(np.sum(np.abs(coordinates) ** 2, axis=0))
    denominator = norms[:, None] * norms[None, :]
    correlation = np.zeros((mode_count, mode_count), dtype=np.float64)
    valid = denominator > np.finfo(np.float64).eps
    gram = coordinates.conj().T @ coordinates
    correlation[valid] = np.abs(gram[valid]) / denominator[valid]
    return dominant, assigned_ratio, correlation


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
    pair_scales = np.sqrt(
        (
            np.diag(raw_gram)[0::2]
            + np.diag(raw_gram)[1::2]
        )
        / normalizer
    )
    if not np.isfinite(pair_scales).all():
        raise ValueError("Direct-coordinate mode-pair scales are non-finite")
    pair_scales[pair_scales <= np.finfo(np.float64).eps] = 1.0
    column_scales = np.repeat(pair_scales, 2)
    normalized_gram = raw_gram / normalizer
    normalized_gram /= column_scales[:, None]
    normalized_gram /= column_scales[None, :]
    system = normalized_gram + settings.ridge_relative * np.eye(column_count)
    cholesky = np.linalg.cholesky(system)
    scipy_linalg = importlib.import_module("scipy.linalg")
    solve_triangular = getattr(scipy_linalg, "solve_triangular")

    eigenvalues = np.linalg.eigvalsh(normalized_gram)
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))[::-1]
    tolerance = (
        np.finfo(np.float64).eps
        * max(2 * sample_count, column_count)
        * float(singular_values[0])
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == column_count and singular_values[-1] > 0.0
        else float("inf")
    )
    ridge_condition = float(
        (float(eigenvalues[-1]) + settings.ridge_relative)
        / (max(float(eigenvalues[0]), 0.0) + settings.ridge_relative)
    )

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

    residual_sq = np.zeros(frame_count, dtype=np.float64)
    flow_sq = np.zeros(frame_count, dtype=np.float64)
    reference_coordinates = coordinates[reference_frame_index]
    progress = Progress("direct coordinate evaluation", frame_count, unit="frames")
    for start in range(0, frame_count, settings.frame_chunk_size):
        end = min(start + settings.frame_chunk_size, frame_count)
        q = coordinates[start:end] - reference_coordinates[None]
        packed = np.empty((column_count, end - start), dtype=np.float64)
        packed[0::2] = q.real.T
        packed[1::2] = q.imag.T
        for lower in range(0, sample_count, SAMPLE_BLOCK_SIZE):
            upper = min(lower + SAMPLE_BLOCK_SIZE, sample_count)
            block = np.asarray(design_value[lower:upper], dtype=np.float64).reshape(
                -1, column_count
            )
            observed = _flow_matrix(
                flow_value,
                pixels[lower:upper],
                start,
                end,
                reference_frame_index,
            )
            residual = block @ packed - observed
            residual_sq[start:end] += np.sum(residual * residual, axis=0)
            flow_sq[start:end] += np.sum(observed * observed, axis=0)
        progress.update(end)
    per_frame_rmse = np.sqrt(residual_sq / normalizer)
    per_frame_relative = np.zeros(frame_count, dtype=np.float64)
    per_frame_r2 = np.ones(frame_count, dtype=np.float64)
    positive = flow_sq > np.finfo(np.float64).eps
    per_frame_relative[positive] = np.sqrt(residual_sq[positive] / flow_sq[positive])
    per_frame_r2[positive] = 1.0 - residual_sq[positive] / flow_sq[positive]
    if np.any((~positive) & (residual_sq > np.finfo(np.float64).eps)):
        raise ValueError("Direct solve predicts motion for a zero reference-relative frame")
    total_residual = float(np.sum(residual_sq))
    total_flow = float(np.sum(flow_sq))
    view_rmse = math.sqrt(total_residual / (normalizer * frame_count))
    if total_flow <= np.finfo(np.float64).eps:
        view_relative, view_r2 = 0.0, 1.0
    else:
        view_relative = math.sqrt(total_residual / total_flow)
        view_r2 = 1.0 - total_residual / total_flow
    dominant, assigned_ratio, correlation = _spectral_diagnostics(
        coordinates, fps_hz, frequencies
    )
    absolute = np.abs(coordinates)
    diagnostics: dict[str, np.ndarray | float | int] = {
        "mode_pair_scales": pair_scales,
        "singular_values": singular_values,
        "numerical_rank": rank,
        "condition_number": condition,
        "ridge_condition_number": ridge_condition,
        "per_frame_flow_rmse": per_frame_rmse,
        "per_frame_relative_residual": per_frame_relative,
        "per_frame_flow_r2": per_frame_r2,
        "residual_sum_squares": total_residual,
        "flow_sum_squares": total_flow,
        "flow_rmse": view_rmse,
        "relative_residual": view_relative,
        "flow_r2": view_r2,
        "reference_residual_norm": math.sqrt(residual_sq[reference_frame_index]),
        "coordinate_rms": np.sqrt(np.mean(absolute * absolute, axis=0)),
        "coordinate_max": np.max(absolute, axis=0),
        "coordinate_temporal_mean_abs": np.abs(np.mean(coordinates, axis=0)),
        "dominant_signed_frequency_hz": dominant,
        "assigned_frequency_energy_ratio": assigned_ratio,
        "cross_mode_correlation": correlation,
    }
    return coordinates.astype(np.complex64), diagnostics


def _load_sources(
    design_dir: str | Path,
    views: Sequence[DirectCoordinateViewInput],
) -> tuple[
    RenderedModalDesignArtifact,
    tuple[FlowAnalysisArtifact, ...],
    list[dict[str, Any]],
]:
    """Load C16 and require the exact ordered flow artifacts used to build it."""

    design = load_rendered_modal_design(design_dir)
    if not views:
        raise ValueError("At least one direct-coordinate view is required")
    labels = [view.label.strip() for view in views]
    design_views = design.manifest["views"]
    if len(views) != len(design_views):
        raise ValueError("Direct-coordinate and rendered-design view counts differ")
    flows: list[FlowAnalysisArtifact] = []
    records: list[dict[str, Any]] = []
    frame_offset = 0
    for index, (label, source, design_view) in enumerate(
        zip(labels, views, design_views)
    ):
        if not label or design_view.get("index") != index or design_view.get("label") != label:
            raise ValueError("Direct-coordinate view order differs from rendered design")
        flow = load_flow_analysis_artifact(
            Path(source.flow_artifact).expanduser().resolve(strict=True)
        )
        identity = flow_artifact_identity(flow)
        if identity != design_view.get("flow_identity"):
            raise ValueError(f"Flow identity for {label!r} differs from rendered design")
        shape = list(flow.arrays.flow.shape[1:3])
        if shape != design_view.get("shape_hw"):
            raise ValueError(f"Flow shape for {label!r} differs from rendered design")
        if (
            int(flow.arrays.flow.shape[0]) != design_view.get("frame_count")
            or float(flow.manifest["fps_hz"]) != design_view.get("fps_hz")
            or int(flow.manifest["reference_frame_index"])
            != design_view.get("flow_reference_frame_index")
            or flow.manifest["reference_frame_name"]
            != design_view.get("flow_reference_frame_name")
        ):
            raise ValueError(f"Flow temporal metadata for {label!r} differs")
        pixels = design.samples["sample_pixels_xy"]
        lower = int(design_view["sample_offset"])
        upper = lower + int(design_view["sample_count"])
        selected = pixels[lower:upper]
        if not np.all(flow.arrays.mask_union[selected[:, 1], selected[:, 0]]):
            raise ValueError(f"Rendered-design pixels for {label!r} leave its flow mask")
        frame_count = int(flow.arrays.flow.shape[0])
        record = {
            "index": index,
            "label": label,
            "flow_identity": identity,
            "shape_hw": shape,
            "fps_hz": float(flow.manifest["fps_hz"]),
            "frame_offset": frame_offset,
            "frame_count": frame_count,
            "frame_names": list(flow.manifest["frame_names"]),
            "reference_frame_name": flow.manifest["reference_frame_name"],
            "reference_frame_index": int(flow.manifest["reference_frame_index"]),
            "sample_count": int(design_view["sample_count"]),
        }
        frame_offset += frame_count
        flows.append(flow)
        records.append(record)
    if len(set(labels)) != len(labels):
        raise ValueError("Direct-coordinate view labels must be unique")
    return design, tuple(flows), records


def _json_number(value: float) -> float | None:
    """Represent infinite condition numbers as JSON null."""

    number = float(value)
    return number if math.isfinite(number) else None


def build_direct_modal_coordinates_artifact(
    *,
    rendered_design_dir: str | Path,
    views: Sequence[DirectCoordinateViewInput],
    output_dir: str | Path,
    config: DirectCoordinateConfig | None = None,
    command: Sequence[str] = (),
) -> DirectModalCoordinatesArtifact:
    """Fit every asynchronous view and atomically publish the direct C17 artifact."""

    settings = config or DirectCoordinateConfig()
    settings.validate()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Direct-coordinate output already exists: {destination}")
    design, flows, view_records = _load_sources(rendered_design_dir, views)
    modes = [dict(mode) for mode in design.manifest["modes"]]
    _validate_modes(modes)
    frequencies = np.asarray(
        [mode["frequency_hz"] for mode in modes], dtype=np.float64
    )
    mode_count = len(modes)
    coordinates_by_view: list[np.ndarray] = []
    diagnostics_by_view: list[dict[str, np.ndarray | float | int]] = []
    offsets = design.samples["view_sample_offsets"]
    pixels = design.samples["sample_pixels_xy"]
    for index, (flow, record) in enumerate(zip(flows, view_records)):
        report_progress(f"direct coordinates: view={record['label']} ({index + 1}/{len(view_records)})")
        lower, upper = int(offsets[index]), int(offsets[index + 1])
        coordinate, diagnostic = solve_direct_coordinates_view(
            design=design.design[lower:upper],
            pixels_xy=pixels[lower:upper],
            flow=flow.arrays.flow,
            reference_frame_index=int(record["reference_frame_index"]),
            fps_hz=float(record["fps_hz"]),
            frequencies_hz=frequencies,
            config=settings,
        )
        record["diagnostics"] = {
            "numerical_rank": int(diagnostic["numerical_rank"]),
            "condition_number": _json_number(float(diagnostic["condition_number"])),
            "ridge_condition_number": float(diagnostic["ridge_condition_number"]),
            "flow_rmse": float(diagnostic["flow_rmse"]),
            "relative_residual": float(diagnostic["relative_residual"]),
            "flow_r2": float(diagnostic["flow_r2"]),
            "reference_residual_norm": float(
                diagnostic["reference_residual_norm"]
            ),
        }
        coordinates_by_view.append(coordinate)
        diagnostics_by_view.append(diagnostic)
    coordinates = np.concatenate(coordinates_by_view, axis=0).astype(np.complex64)
    frame_count = len(coordinates)
    frame_view = np.concatenate(
        [
            np.full(record["frame_count"], index, dtype=np.int64)
            for index, record in enumerate(view_records)
        ]
    )
    frame_local = np.concatenate(
        [np.arange(record["frame_count"], dtype=np.int64) for record in view_records]
    )
    frame_times = np.concatenate(
        [
            np.arange(record["frame_count"], dtype=np.float64) / record["fps_hz"]
            for record in view_records
        ]
    )
    diagnostics = {
        "frame_view_index": frame_view,
        "frame_local_index": frame_local,
        "frame_times_sec": frame_times,
        "reference_local_index": np.asarray(
            [record["reference_frame_index"] for record in view_records],
            dtype=np.int64,
        ),
        "mode_pair_scales": np.stack(
            [value["mode_pair_scales"] for value in diagnostics_by_view]
        ).astype(np.float64),
        "singular_values": np.stack(
            [value["singular_values"] for value in diagnostics_by_view]
        ).astype(np.float64),
        "numerical_rank": np.asarray(
            [value["numerical_rank"] for value in diagnostics_by_view],
            dtype=np.int64,
        ),
        "condition_number": np.asarray(
            [value["condition_number"] for value in diagnostics_by_view],
            dtype=np.float64,
        ),
        "ridge_condition_number": np.asarray(
            [value["ridge_condition_number"] for value in diagnostics_by_view],
            dtype=np.float64,
        ),
        "per_frame_flow_rmse": np.concatenate(
            [value["per_frame_flow_rmse"] for value in diagnostics_by_view]
        ).astype(np.float64),
        "per_frame_relative_residual": np.concatenate(
            [value["per_frame_relative_residual"] for value in diagnostics_by_view]
        ).astype(np.float64),
        "per_frame_flow_r2": np.concatenate(
            [value["per_frame_flow_r2"] for value in diagnostics_by_view]
        ).astype(np.float64),
        "view_residual_sum_squares": np.asarray(
            [value["residual_sum_squares"] for value in diagnostics_by_view]
        ),
        "view_flow_sum_squares": np.asarray(
            [value["flow_sum_squares"] for value in diagnostics_by_view]
        ),
        "view_flow_rmse": np.asarray(
            [value["flow_rmse"] for value in diagnostics_by_view]
        ),
        "view_relative_residual": np.asarray(
            [value["relative_residual"] for value in diagnostics_by_view]
        ),
        "view_flow_r2": np.asarray(
            [value["flow_r2"] for value in diagnostics_by_view]
        ),
        "reference_residual_norm": np.asarray(
            [value["reference_residual_norm"] for value in diagnostics_by_view]
        ),
        "coordinate_rms": np.stack(
            [value["coordinate_rms"] for value in diagnostics_by_view]
        ),
        "coordinate_max": np.stack(
            [value["coordinate_max"] for value in diagnostics_by_view]
        ),
        "coordinate_temporal_mean_abs": np.stack(
            [value["coordinate_temporal_mean_abs"] for value in diagnostics_by_view]
        ),
        "dominant_signed_frequency_hz": np.stack(
            [value["dominant_signed_frequency_hz"] for value in diagnostics_by_view]
        ),
        "assigned_frequency_energy_ratio": np.stack(
            [value["assigned_frequency_energy_ratio"] for value in diagnostics_by_view]
        ),
        "cross_mode_correlation": np.stack(
            [value["cross_mode_correlation"] for value in diagnostics_by_view]
        ),
    }
    diagnostics = {
        name: np.ascontiguousarray(value, dtype=DIAGNOSTIC_DTYPES[name])
        for name, value in diagnostics.items()
    }
    _validate_diagnostics(
        diagnostics,
        views=view_records,
        frame_count=frame_count,
        mode_count=mode_count,
    )
    total_residual = float(np.sum(diagnostics["view_residual_sum_squares"]))
    total_flow = float(np.sum(diagnostics["view_flow_sum_squares"]))
    total_observations = sum(
        2 * record["sample_count"] * record["frame_count"]
        for record in view_records
    )
    overall = {
        "flow_rmse": math.sqrt(total_residual / total_observations),
        "relative_residual": (
            math.sqrt(total_residual / total_flow) if total_flow > 0.0 else 0.0
        ),
        "flow_r2": 1.0 - total_residual / total_flow if total_flow > 0.0 else 1.0,
        "residual_sum_squares": total_residual,
        "flow_sum_squares": total_flow,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        coordinates_path = temporary / COORDINATES_FILENAME
        diagnostics_path = temporary / DIAGNOSTICS_FILENAME
        np.save(coordinates_path, coordinates, allow_pickle=False)
        save_named_arrays(diagnostics_path, diagnostics)
        coordinate_record = {
            "file": COORDINATES_FILENAME,
            "dtype": COORDINATES_DTYPE.name,
            "shape": [frame_count, mode_count],
            "sha256": _sha256_file(coordinates_path),
        }
        diagnostic_record = {
            "file": DIAGNOSTICS_FILENAME,
            "arrays": {
                name: {"dtype": value.dtype.name, "shape": list(value.shape)}
                for name, value in diagnostics.items()
            },
            "arrays_identity": _arrays_identity(diagnostics),
            "sha256": _sha256_file(diagnostics_path),
        }
        manifest = {
            "format": DIRECT_COORDINATES_FORMAT,
            "version": DIRECT_COORDINATES_VERSION,
            "producer": {
                "project_version": __version__,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": list(command),
            },
            "rendered_design": str(design.path),
            "rendered_design_identity": design.manifest["rendered_design_identity"],
            "completed_modes_identity": design.manifest["completed_modes_identity"],
            "flow_artifacts": [str(flow.path.resolve()) for flow in flows],
            "modes": modes,
            "views": view_records,
            "solver": dict(SOLVER_CONVENTION),
            "settings": settings.to_dict(),
            "quality_gate": {
                "required": True,
                "status": "direct_coordinates_candidate_unapproved",
                "inherited_from": "rendered_design_candidate_unapproved",
            },
            "counts": {
                "views": len(view_records),
                "frames": frame_count,
                "modes": mode_count,
            },
            "coordinates": coordinate_record,
            "diagnostics": diagnostic_record,
            "overall": overall,
        }
        manifest["direct_coordinates_identity"] = hashlib.sha256(
            _canonical_json(_identity_payload(manifest))
        ).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_direct_modal_coordinates(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Direct-coordinate output already exists: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_direct_modal_coordinates(destination)


__all__ = [
    "DirectCoordinateConfig",
    "DirectCoordinateViewInput",
    "DirectModalCoordinatesArtifact",
    "build_direct_modal_coordinates_artifact",
    "load_direct_modal_coordinates",
    "solve_direct_coordinates_view",
]
