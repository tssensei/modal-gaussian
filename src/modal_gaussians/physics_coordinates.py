"""Damped-oscillator post-fit for direct per-view modal coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence, cast

import numpy as np
from modal_gaussians.progress import Progress
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, onenormest, splu

from modal_gaussians import __version__
from modal_gaussians.flow.storage import DenseArray, read_pixels
from modal_gaussians.direct_coordinates import (
    DirectModalCoordinatesArtifact,
    load_direct_modal_coordinates,
)
from modal_gaussians.flow.artifact import (
    FlowAnalysisArtifact,
    flow_artifact_identity,
    load_flow_analysis_artifact,
)
from modal_gaussians.numpy_io import save_named_arrays
from modal_gaussians.rendered_design import (
    RenderedModalDesignArtifact,
    load_rendered_modal_design,
)


PHYSICS_COORDINATES_FORMAT = "modal_gaussians.physics_modal_coordinates"
PHYSICS_COORDINATES_VERSION = 1
COORDINATES_FILENAME = "coordinates.npy"
DIAGNOSTICS_FILENAME = "diagnostics.npz"
COORDINATES_DTYPE = np.dtype(np.complex64)
SAMPLE_BLOCK_SIZE = 65_536

# SciPy's public sparse API is more permissive than its current type stubs.
_SPARSE = cast(Any, sparse)
_LINEAR_OPERATOR = cast(Any, LinearOperator)
_ONE_NORM_ESTIMATE = cast(Any, onenormest)
_SPARSE_LU = cast(Any, splu)

SOLVER_CONVENTION = {
    "solver": "latent_force_oscillator_postfit_v1",
    "source": "direct_modal_coordinates",
    "dynamics_operator": "D2 + 2*zeta*omega*D1 + omega^2*I",
    "derivatives": "actual_time_three_point_second_order",
    "forcing_difference": "adjacent_sample_difference",
    "complex_channels": "shared_operator_independent_real_imaginary_rhs",
    "gauge": "per_view_temporal_mean_zero",
    "flow_evaluation": "same_rendered_design_and_reference_relative_flow_as_direct",
}

MODE_DIAGNOSTIC_NAMES = (
    "input_coordinate_rms",
    "output_coordinate_rms",
    "input_coordinate_p90",
    "output_coordinate_p90",
    "input_coordinate_p99",
    "output_coordinate_p99",
    "input_coordinate_max",
    "output_coordinate_max",
    "fidelity_nrmse",
    "rms_retention",
    "p90_retention",
    "p99_retention",
    "input_first_derivative_rms",
    "output_first_derivative_rms",
    "input_second_derivative_rms",
    "output_second_derivative_rms",
    "input_forcing_normalized_rms",
    "output_forcing_normalized_rms",
    "input_forcing_difference_normalized_rms",
    "output_forcing_difference_normalized_rms",
    "system_condition_estimate",
    "input_assigned_frequency_energy_ratio",
    "output_assigned_frequency_energy_ratio",
    "input_dominant_signed_frequency_hz",
    "output_dominant_signed_frequency_hz",
)

FLOW_FRAME_DIAGNOSTIC_NAMES = (
    "input_per_frame_flow_rmse",
    "output_per_frame_flow_rmse",
    "input_per_frame_relative_residual",
    "output_per_frame_relative_residual",
    "input_per_frame_flow_r2",
    "output_per_frame_flow_r2",
)

FLOW_VIEW_DIAGNOSTIC_NAMES = (
    "input_view_flow_rmse",
    "output_view_flow_rmse",
    "input_view_relative_residual",
    "output_view_relative_residual",
    "input_view_flow_r2",
    "output_view_flow_r2",
    "input_view_strong_motion_flow_r2",
    "output_view_strong_motion_flow_r2",
)

DIAGNOSTIC_DTYPES = {
    **{name: np.dtype(np.float64) for name in MODE_DIAGNOSTIC_NAMES},
    "input_cross_mode_correlation": np.dtype(np.float64),
    "output_cross_mode_correlation": np.dtype(np.float64),
    **{name: np.dtype(np.float64) for name in FLOW_FRAME_DIAGNOSTIC_NAMES},
    **{name: np.dtype(np.float64) for name in FLOW_VIEW_DIAGNOSTIC_NAMES},
}


@dataclass(frozen=True)
class PhysicsCoordinateConfig:
    """Hold the accepted damped-oscillator post-fit settings."""

    damping_ratio: float = 0.05
    forcing_weight: float = 0.1
    forcing_difference_weight: float = 0.0
    assigned_band_half_width_hz: float = 0.1
    frame_chunk_size: int = 64

    def validate(self) -> None:
        """Reject non-finite weights, invalid bands, and invalid batching."""

        for name, value in (
            ("damping_ratio", self.damping_ratio),
            ("forcing_weight", self.forcing_weight),
            ("forcing_difference_weight", self.forcing_difference_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Physics-coordinate {name} must be non-negative")
        if (
            not math.isfinite(self.assigned_band_half_width_hz)
            or self.assigned_band_half_width_hz <= 0.0
        ):
            raise ValueError(
                "Physics-coordinate assigned_band_half_width_hz must be positive"
            )
        if (
            isinstance(self.frame_chunk_size, bool)
            or not isinstance(self.frame_chunk_size, int)
            or self.frame_chunk_size <= 0
        ):
            raise ValueError("Physics-coordinate frame_chunk_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete scientific and evaluation configuration."""

        return {
            "damping_ratio": self.damping_ratio,
            "forcing_weight": self.forcing_weight,
            "forcing_difference_weight": self.forcing_difference_weight,
            "assigned_band_half_width_hz": self.assigned_band_half_width_hz,
            "frame_chunk_size": self.frame_chunk_size,
        }


@dataclass(frozen=True)
class PhysicsModalCoordinatesArtifact:
    """Represent one validated mmap-backed physics-coordinate artifact."""

    path: Path
    manifest: dict[str, Any]
    coordinates: np.ndarray
    diagnostics: dict[str, np.ndarray]


def _canonical_json(value: Any) -> bytes:
    """Encode identity fields deterministically without non-JSON floats."""

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
    """Select path-independent scientific inputs, settings, and outputs."""

    return {
        "format": PHYSICS_COORDINATES_FORMAT,
        "version": PHYSICS_COORDINATES_VERSION,
        "direct_coordinates_identity": manifest["direct_coordinates_identity"],
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
    """Validate the ordered greedy mode prefix and positive frequencies."""

    if not isinstance(modes, list) or not modes:
        raise ValueError("Physics coordinates must contain ordered modes")
    candidates: list[int] = []
    for slot, mode in enumerate(modes):
        if not isinstance(mode, dict) or mode.get("mode_slot") != slot:
            raise ValueError("Physics-coordinate mode slots must be contiguous")
        candidate = mode.get("candidate_index")
        frequency = float(mode.get("frequency_hz", np.nan))
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate < 0
            or not math.isfinite(frequency)
            or frequency <= 0.0
        ):
            raise ValueError("Physics-coordinate mode metadata is invalid")
        candidates.append(candidate)
    if len(set(candidates)) != len(candidates):
        raise ValueError("Physics-coordinate candidate indices must be unique")


def _expected_diagnostic_shapes(
    view_count: int, frame_count: int, mode_count: int
) -> dict[str, tuple[int, ...]]:
    """Return every fixed diagnostic shape for one physics artifact."""

    shapes: dict[str, tuple[int, ...]] = {
        name: (view_count, mode_count) for name in MODE_DIAGNOSTIC_NAMES
    }
    shapes.update(
        {
            "input_cross_mode_correlation": (view_count, mode_count, mode_count),
            "output_cross_mode_correlation": (view_count, mode_count, mode_count),
        }
    )
    shapes.update({name: (frame_count,) for name in FLOW_FRAME_DIAGNOSTIC_NAMES})
    shapes.update({name: (view_count,) for name in FLOW_VIEW_DIAGNOSTIC_NAMES})
    return shapes


def _validate_diagnostics(
    arrays: Mapping[str, np.ndarray],
    *,
    view_count: int,
    frame_count: int,
    mode_count: int,
) -> None:
    """Validate fixed shapes, dtypes, finiteness, ratios, and conditions."""

    if set(arrays) != set(DIAGNOSTIC_DTYPES):
        raise ValueError("Physics-coordinate diagnostic fields are invalid")
    shapes = _expected_diagnostic_shapes(view_count, frame_count, mode_count)
    for name, dtype in DIAGNOSTIC_DTYPES.items():
        value = arrays[name]
        if value.dtype != dtype or value.shape != shapes[name]:
            raise ValueError(f"Physics-coordinate diagnostic {name} is invalid")
        if not np.isfinite(value).all():
            raise ValueError(f"Physics-coordinate diagnostic {name} is non-finite")
    for name in (
        "input_assigned_frequency_energy_ratio",
        "output_assigned_frequency_energy_ratio",
        "input_cross_mode_correlation",
        "output_cross_mode_correlation",
    ):
        if np.any(arrays[name] < -1.0e-12) or np.any(arrays[name] > 1.0 + 1.0e-12):
            raise ValueError(f"Physics-coordinate diagnostic {name} leaves [0,1]")
    if np.any(arrays["system_condition_estimate"] < 1.0):
        raise ValueError("Physics-coordinate condition estimates must be at least one")


def load_physics_modal_coordinates(
    path: str | Path,
) -> PhysicsModalCoordinatesArtifact:
    """Load and strictly validate one physics-coordinate artifact."""

    root = Path(path).expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    coordinates_path = root / COORDINATES_FILENAME
    diagnostics_path = root / DIAGNOSTICS_FILENAME
    if not all(value.is_file() for value in (manifest_path, coordinates_path, diagnostics_path)):
        raise FileNotFoundError(f"Incomplete physics-coordinate artifact: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != PHYSICS_COORDINATES_FORMAT:
        raise ValueError("Unsupported physics-coordinate format")
    if manifest.get("version") != PHYSICS_COORDINATES_VERSION:
        raise ValueError("Unsupported physics-coordinate version")
    if manifest.get("solver") != SOLVER_CONVENTION:
        raise ValueError("Physics-coordinate solver convention is unsupported")
    if manifest.get("quality_gate") != {
        "required": True,
        "status": "physics_coordinates_candidate_unapproved",
        "inherited_from": "direct_coordinates_candidate_unapproved",
    }:
        raise ValueError("Physics coordinates must inherit the direct quality gate")
    for name in (
        "direct_coordinates_identity",
        "rendered_design_identity",
        "completed_modes_identity",
    ):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"Physics-coordinate {name} is invalid")
    _validate_modes(manifest.get("modes"))
    settings = manifest.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Physics-coordinate settings are invalid")
    config = PhysicsCoordinateConfig(
        damping_ratio=float(settings["damping_ratio"]),
        forcing_weight=float(settings["forcing_weight"]),
        forcing_difference_weight=float(settings["forcing_difference_weight"]),
        assigned_band_half_width_hz=float(settings["assigned_band_half_width_hz"]),
        frame_chunk_size=int(settings["frame_chunk_size"]),
    )
    config.validate()
    if settings != config.to_dict():
        raise ValueError("Physics-coordinate settings contain unsupported fields")
    views = manifest.get("views")
    counts = manifest.get("counts")
    if not isinstance(views, list) or not views or not isinstance(counts, dict):
        raise ValueError("Physics-coordinate views or counts are invalid")
    view_count = int(counts.get("views", -1))
    frame_count = int(counts.get("frames", -1))
    mode_count = int(counts.get("modes", -1))
    if (
        view_count != len(views)
        or mode_count != len(manifest["modes"])
        or frame_count < 3
    ):
        raise ValueError("Physics-coordinate counts disagree with metadata")
    labels: list[str] = []
    offset = 0
    for index, view in enumerate(views):
        if not isinstance(view, dict) or view.get("index") != index:
            raise ValueError("Physics-coordinate views must be ordered")
        label = view.get("label")
        count = view.get("frame_count")
        fps = view.get("fps_hz")
        reference = view.get("reference_frame_index")
        names = view.get("frame_names")
        if (
            not isinstance(label, str)
            or not label
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 3
            or not isinstance(fps, (int, float))
            or isinstance(fps, bool)
            or not math.isfinite(float(fps))
            or float(fps) <= 0.0
            or isinstance(reference, bool)
            or not isinstance(reference, int)
            or reference < 0
            or reference >= count
            or not isinstance(names, list)
            or len(names) != count
            or view.get("frame_offset") != offset
        ):
            raise ValueError("Physics-coordinate view metadata is invalid")
        if names[reference] != view.get("reference_frame_name"):
            raise ValueError("Physics-coordinate reference frame metadata differs")
        labels.append(label)
        offset += count
    if offset != frame_count or len(set(labels)) != len(labels):
        raise ValueError("Physics-coordinate frame partition or labels are invalid")
    coordinate_record = manifest.get("coordinates")
    diagnostic_record = manifest.get("diagnostics")
    if not isinstance(coordinate_record, dict) or not isinstance(diagnostic_record, dict):
        raise ValueError("Physics-coordinate file metadata is invalid")
    if (
        coordinate_record.get("file") != COORDINATES_FILENAME
        or coordinate_record.get("dtype") != COORDINATES_DTYPE.name
        or coordinate_record.get("shape") != [frame_count, mode_count]
        or coordinate_record.get("sha256") != _sha256_file(coordinates_path)
        or diagnostic_record.get("file") != DIAGNOSTICS_FILENAME
        or diagnostic_record.get("sha256") != _sha256_file(diagnostics_path)
    ):
        raise ValueError("Physics-coordinate file metadata or SHA-256 is invalid")
    coordinates = np.load(coordinates_path, mmap_mode="r", allow_pickle=False)
    if (
        coordinates.dtype != COORDINATES_DTYPE
        or coordinates.shape != (frame_count, mode_count)
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("Physics-coordinate array is invalid")
    with np.load(diagnostics_path, allow_pickle=False) as archive:
        diagnostics = {name: archive[name] for name in archive.files}
    metadata = {
        name: {"dtype": value.dtype.name, "shape": list(value.shape)}
        for name, value in diagnostics.items()
    }
    if diagnostic_record.get("arrays") != metadata:
        raise ValueError("Physics-coordinate diagnostic metadata differs")
    if diagnostic_record.get("arrays_identity") != _arrays_identity(diagnostics):
        raise ValueError("Physics-coordinate diagnostic identity differs")
    _validate_diagnostics(
        diagnostics,
        view_count=view_count,
        frame_count=frame_count,
        mode_count=mode_count,
    )
    for view in views:
        lower = int(view["frame_offset"])
        upper = lower + int(view["frame_count"])
        values = coordinates[lower:upper].astype(np.complex128)
        mean = np.abs(np.mean(values, axis=0))
        scale = np.maximum(np.sqrt(np.mean(np.abs(values) ** 2, axis=0)), 1.0)
        if np.any(mean > 5.0e-6 * scale):
            raise ValueError("Physics-coordinate temporal mean-zero gauge is violated")
    expected_identity = hashlib.sha256(
        _canonical_json(_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("physics_coordinates_identity") != expected_identity:
        raise ValueError("Physics-coordinate identity differs from its contents")
    return PhysicsModalCoordinatesArtifact(root, manifest, coordinates, diagnostics)


def _finite_difference_weights(
    sample_times: np.ndarray,
    evaluation_time: float,
    derivative_order: int,
) -> np.ndarray:
    """Return exact local polynomial finite-difference weights."""

    offsets = np.asarray(sample_times, dtype=np.float64) - float(evaluation_time)
    powers = np.arange(offsets.size, dtype=np.int64)[:, None]
    system = offsets[None, :] ** powers
    right_hand_side = np.zeros(offsets.size, dtype=np.float64)
    right_hand_side[derivative_order] = math.factorial(derivative_order)
    return np.linalg.solve(system, right_hand_side)


def finite_difference_matrix(
    times_sec: np.ndarray, derivative_order: int
) -> Any:
    """Build the accepted actual-time first- or second-derivative matrix."""

    times = np.asarray(times_sec, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("Finite-difference times must be a non-empty vector")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("Finite-difference times must be finite and increasing")
    if derivative_order not in (1, 2):
        raise ValueError("Only first and second derivatives are supported")
    count = times.size
    if count == 1 or (count == 2 and derivative_order == 2):
        return _SPARSE.csr_matrix((count, count), dtype=np.float64)
    if count == 2:
        step = float(times[1] - times[0])
        return _SPARSE.csr_matrix(
            np.asarray([[-1.0 / step, 1.0 / step]] * 2, dtype=np.float64)
        )
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for row in range(count):
        if row == 0:
            stencil = np.asarray([0, 1, 2], dtype=np.int64)
        elif row == count - 1:
            stencil = np.asarray([count - 3, count - 2, count - 1], dtype=np.int64)
        else:
            stencil = np.asarray([row - 1, row, row + 1], dtype=np.int64)
        weights = _finite_difference_weights(
            times[stencil], float(times[row]), derivative_order
        )
        rows.extend([row] * len(stencil))
        columns.extend(stencil.tolist())
        data.extend(weights.tolist())
    return _SPARSE.coo_matrix(
        (data, (rows, columns)), shape=(count, count), dtype=np.float64
    ).tocsr()


def _forcing_difference_matrix(count: int) -> Any:
    """Build the adjacent latent-force difference operator."""

    if count <= 1:
        return _SPARSE.csr_matrix((0, count), dtype=np.float64)
    return _SPARSE.diags(
        diagonals=(-np.ones(count - 1), np.ones(count - 1)),
        offsets=(0, 1),
        shape=(count - 1, count),
        format="csr",
        dtype=np.float64,
    )


def _condition_estimate(system: Any, factor: Any) -> float:
    """Estimate the sparse one-norm condition number without densifying."""

    system_norm = float(_ONE_NORM_ESTIMATE(system))

    def solve(value: np.ndarray) -> np.ndarray:
        return np.asarray(factor.solve(np.asarray(value, dtype=np.float64)))

    inverse = _LINEAR_OPERATOR(
        system.shape,
        matvec=solve,
        rmatvec=solve,
        matmat=solve,
        rmatmat=solve,
        dtype=np.float64,
    )
    result = system_norm * float(_ONE_NORM_ESTIMATE(inverse))
    if not math.isfinite(result) or result < 1.0:
        raise ValueError("Physics-coordinate condition estimate is invalid")
    return result


def _rms(values: np.ndarray) -> float:
    """Return the real RMS magnitude of a real or complex array."""

    array = np.asarray(values)
    return float(np.sqrt(np.mean(np.abs(array) ** 2)))


def _spectral_metrics(
    coordinates: np.ndarray,
    times_sec: np.ndarray,
    assigned_frequency_hz: float,
    band_half_width_hz: float,
) -> tuple[float, float]:
    """Measure dominant signed frequency and assigned-band energy ratio."""

    values = np.asarray(coordinates, dtype=np.complex128)
    times = np.asarray(times_sec, dtype=np.float64)
    if values.ndim != 1 or times.shape != values.shape:
        raise ValueError("Spectral coordinate inputs must be matching vectors")
    if len(values) < 2:
        return 0.0, 0.0
    steps = np.diff(times)
    step = float(np.median(steps))
    if not np.allclose(steps, step, rtol=1.0e-6, atol=1.0e-9):
        raise ValueError("Spectral diagnostics require uniform view timestamps")
    centered = values - np.mean(values)
    energy = np.abs(np.fft.fft(centered)) ** 2
    frequencies = np.fft.fftfreq(len(values), d=step)
    energy[np.isclose(frequencies, 0.0, rtol=0.0, atol=1.0e-15)] = 0.0
    total = float(np.sum(energy))
    if total <= np.finfo(np.float64).eps:
        return 0.0, 0.0
    dominant = float(frequencies[int(np.argmax(energy))])
    band = np.abs(np.abs(frequencies) - assigned_frequency_hz) <= band_half_width_hz
    ratio = float(np.clip(np.sum(energy[band]) / total, 0.0, 1.0))
    return dominant, ratio


def _complex_correlation(coordinates: np.ndarray) -> np.ndarray:
    """Return absolute normalized complex cross-mode correlation."""

    values = np.asarray(coordinates, dtype=np.complex128)
    centered = values - np.mean(values, axis=0, keepdims=True)
    norms = np.sqrt(np.sum(np.abs(centered) ** 2, axis=0))
    denominator = norms[:, None] * norms[None, :]
    correlation = np.zeros(denominator.shape, dtype=np.float64)
    valid = denominator > np.finfo(np.float64).eps
    gram = centered.conj().T @ centered
    correlation[valid] = np.abs(gram[valid]) / denominator[valid]
    return np.clip(correlation, 0.0, 1.0)


def solve_physics_coordinate_mode(
    *,
    source: np.ndarray,
    times_sec: np.ndarray,
    frequency_hz: float,
    config: PhysicsCoordinateConfig | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit one complex mode in one view with a mean-zero damped oscillator."""

    settings = config or PhysicsCoordinateConfig()
    settings.validate()
    values = np.asarray(source, dtype=np.complex128)
    times = np.asarray(times_sec, dtype=np.float64)
    if values.ndim != 1 or times.shape != values.shape or len(values) < 3:
        raise ValueError("Physics mode source and timestamps must be matching vectors")
    if not np.isfinite(values).all():
        raise ValueError("Physics mode source contains NaN or Inf")
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("Physics mode frequency must be positive")
    first = finite_difference_matrix(times, 1)
    second = finite_difference_matrix(times, 2)
    count = len(values)
    omega = 2.0 * np.pi * float(frequency_hz)
    oscillator = (
        second
        + (2.0 * settings.damping_ratio * omega) * first
        + (omega * omega) * _SPARSE.eye(count, format="csr", dtype=np.float64)
    ).tocsr()
    force_difference = _forcing_difference_matrix(count)
    if settings.forcing_weight == 0.0 and settings.forcing_difference_weight == 0.0:
        fitted = values.copy()
        condition = 1.0
    else:
        system: Any = _SPARSE.eye(count, format="csr", dtype=np.float64) / count
        if settings.forcing_weight > 0.0:
            system = system + (
                settings.forcing_weight
                * (oscillator.T @ oscillator)
                / (count * omega**4)
            )
        if settings.forcing_difference_weight > 0.0:
            difference_oscillator = force_difference @ oscillator
            system = system + (
                settings.forcing_difference_weight
                * (difference_oscillator.T @ difference_oscillator)
                / ((count - 1) * omega**4)
            )
        csc_system = system.tocsc()
        factor = _SPARSE_LU(csc_system)
        right_hand_side = np.column_stack(
            (
                np.real(values) / count,
                np.imag(values) / count,
                np.ones(count, dtype=np.float64),
            )
        )
        solved = np.asarray(factor.solve(right_hand_side), dtype=np.float64)
        constraint_direction = solved[:, 2]
        denominator = float(np.sum(constraint_direction))
        if not math.isfinite(denominator) or abs(denominator) <= np.finfo(np.float64).eps:
            raise ValueError("Physics-coordinate mean-zero constraint is singular")
        fitted_real = solved[:, 0] - constraint_direction * (
            float(np.sum(solved[:, 0])) / denominator
        )
        fitted_imag = solved[:, 1] - constraint_direction * (
            float(np.sum(solved[:, 1])) / denominator
        )
        fitted = fitted_real + 1j * fitted_imag
        condition = _condition_estimate(csc_system, factor)
    input_force = oscillator @ values
    output_force = oscillator @ fitted
    input_force_difference = force_difference @ input_force
    output_force_difference = force_difference @ output_force
    scale = max(_rms(values), 1.0e-12)
    input_magnitude = np.abs(values)
    output_magnitude = np.abs(fitted)
    input_p90 = float(np.percentile(input_magnitude, 90.0))
    output_p90 = float(np.percentile(output_magnitude, 90.0))
    input_p99 = float(np.percentile(input_magnitude, 99.0))
    output_p99 = float(np.percentile(output_magnitude, 99.0))
    diagnostics: dict[str, float] = {
        "input_coordinate_rms": _rms(values),
        "output_coordinate_rms": _rms(fitted),
        "input_coordinate_p90": input_p90,
        "output_coordinate_p90": output_p90,
        "input_coordinate_p99": input_p99,
        "output_coordinate_p99": output_p99,
        "input_coordinate_max": float(np.max(input_magnitude)),
        "output_coordinate_max": float(np.max(output_magnitude)),
        "fidelity_nrmse": _rms(fitted - values) / scale,
        "rms_retention": _rms(fitted) / scale,
        "p90_retention": output_p90 / max(input_p90, 1.0e-12),
        "p99_retention": output_p99 / max(input_p99, 1.0e-12),
        "input_first_derivative_rms": _rms(first @ values),
        "output_first_derivative_rms": _rms(first @ fitted),
        "input_second_derivative_rms": _rms(second @ values),
        "output_second_derivative_rms": _rms(second @ fitted),
        "input_forcing_normalized_rms": _rms(input_force) / (omega * omega * scale),
        "output_forcing_normalized_rms": _rms(output_force) / (omega * omega * scale),
        "input_forcing_difference_normalized_rms": (
            _rms(input_force_difference) / (omega * omega * scale)
            if input_force_difference.size
            else 0.0
        ),
        "output_forcing_difference_normalized_rms": (
            _rms(output_force_difference) / (omega * omega * scale)
            if output_force_difference.size
            else 0.0
        ),
        "system_condition_estimate": condition,
    }
    if not np.isfinite(fitted).all() or not all(
        math.isfinite(value) for value in diagnostics.values()
    ):
        raise FloatingPointError("Physics-coordinate solve produced NaN or Inf")
    return fitted, diagnostics


def _flow_matrix(
    flow: DenseArray,
    pixels: np.ndarray,
    start: int,
    end: int,
    reference_index: int,
) -> np.ndarray:
    """Sample one block of reference-relative flow as `[2P,B]`."""

    values = read_pixels(flow, slice(start, end), pixels).astype(np.float64)
    reference = read_pixels(flow, reference_index, pixels).astype(np.float64)
    values -= reference[None]
    if not np.isfinite(values).all():
        raise ValueError("Flow contains NaN or Inf at rendered-design pixels")
    return values.transpose(1, 2, 0).reshape(2 * len(pixels), end - start)


def _evaluate_coordinate_sets_view(
    *,
    design: np.ndarray,
    pixels_xy: np.ndarray,
    flow: DenseArray,
    reference_frame_index: int,
    coordinate_sets: Mapping[str, np.ndarray],
    frame_chunk_size: int,
) -> dict[str, dict[str, np.ndarray | float]]:
    """Evaluate input/output coordinates through the exact rendered design."""

    design_value = np.asarray(design)
    pixels = np.asarray(pixels_xy, dtype=np.int64)
    flow_value = flow
    if design_value.ndim != 3 or design_value.shape[1] != 2:
        raise ValueError("Physics flow evaluation design must be [P,2,2K]")
    sample_count, _, column_count = design_value.shape
    mode_count = column_count // 2
    if column_count < 2 or column_count % 2 or pixels.shape != (sample_count, 2):
        raise ValueError("Physics flow evaluation design samples are invalid")
    if (
        flow_value.ndim != 4
        or flow_value.shape[-1] != 2
        or flow_value.shape[0] < 3
        or reference_frame_index < 0
        or reference_frame_index >= flow_value.shape[0]
    ):
        raise ValueError("Physics flow evaluation source is invalid")
    frame_count = flow_value.shape[0]
    values_by_label: dict[str, np.ndarray] = {}
    for label, coordinates in coordinate_sets.items():
        value = np.asarray(coordinates, dtype=np.complex128)
        if not label or value.shape != (frame_count, mode_count) or not np.isfinite(value).all():
            raise ValueError(f"Physics coordinate set {label!r} is invalid")
        values_by_label[label] = value
    if not values_by_label:
        raise ValueError("Physics flow evaluation requires coordinate sets")
    normalizer = float(2 * sample_count)
    flow_energy = np.zeros(frame_count, dtype=np.float64)
    residual_energy = {
        label: np.zeros(frame_count, dtype=np.float64) for label in values_by_label
    }
    references = {
        label: values[reference_frame_index] for label, values in values_by_label.items()
    }
    for start in range(0, frame_count, frame_chunk_size):
        end = min(start + frame_chunk_size, frame_count)
        packed_by_label: dict[str, np.ndarray] = {}
        for label, values in values_by_label.items():
            relative = values[start:end] - references[label][None]
            packed = np.empty((column_count, end - start), dtype=np.float64)
            packed[0::2] = np.real(relative).T
            packed[1::2] = np.imag(relative).T
            packed_by_label[label] = packed
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
            flow_energy[start:end] += np.sum(observed * observed, axis=0)
            for label, packed in packed_by_label.items():
                residual = block @ packed - observed
                residual_energy[label][start:end] += np.sum(
                    residual * residual, axis=0
                )
    positive = flow_energy > np.finfo(np.float64).eps
    strong_count = max(1, int(np.ceil(0.1 * frame_count)))
    strong_rows = np.argpartition(flow_energy, -strong_count)[-strong_count:]
    results: dict[str, dict[str, np.ndarray | float]] = {}
    for label, residual in residual_energy.items():
        per_frame_rmse = np.sqrt(residual / normalizer)
        per_frame_relative = np.zeros(frame_count, dtype=np.float64)
        per_frame_r2 = np.ones(frame_count, dtype=np.float64)
        per_frame_relative[positive] = np.sqrt(residual[positive] / flow_energy[positive])
        per_frame_r2[positive] = 1.0 - residual[positive] / flow_energy[positive]
        if np.any((~positive) & (residual > np.finfo(np.float64).eps)):
            raise ValueError(
                f"Physics coordinate set {label!r} predicts motion for zero flow"
            )
        residual_sum = float(np.sum(residual))
        flow_sum = float(np.sum(flow_energy))
        strong_flow = float(np.sum(flow_energy[strong_rows]))
        strong_residual = float(np.sum(residual[strong_rows]))
        results[label] = {
            "per_frame_flow_rmse": per_frame_rmse,
            "per_frame_relative_residual": per_frame_relative,
            "per_frame_flow_r2": per_frame_r2,
            "flow_rmse": math.sqrt(residual_sum / (normalizer * frame_count)),
            "relative_residual": (
                math.sqrt(residual_sum / flow_sum) if flow_sum > 0.0 else 0.0
            ),
            "flow_r2": 1.0 - residual_sum / flow_sum if flow_sum > 0.0 else 1.0,
            "strong_motion_flow_r2": (
                1.0 - strong_residual / strong_flow if strong_flow > 0.0 else 1.0
            ),
            "residual_sum_squares": residual_sum,
            "flow_sum_squares": flow_sum,
            "observation_count": normalizer * frame_count,
        }
    return results


def _load_sources(
    direct_coordinates_dir: str | Path,
) -> tuple[
    DirectModalCoordinatesArtifact,
    RenderedModalDesignArtifact,
    tuple[FlowAnalysisArtifact, ...],
]:
    """Load the direct artifact and revalidate its exact design/flow chain."""

    direct = load_direct_modal_coordinates(direct_coordinates_dir)
    design_source = direct.manifest.get("rendered_design")
    if not isinstance(design_source, str) or not design_source:
        raise ValueError("Direct coordinates do not name their rendered design")
    design = load_rendered_modal_design(design_source)
    if design.manifest["rendered_design_identity"] != direct.manifest["rendered_design_identity"]:
        raise ValueError("Direct-coordinate rendered-design identity differs")
    if design.manifest["completed_modes_identity"] != direct.manifest["completed_modes_identity"]:
        raise ValueError("Direct-coordinate completed-mode identity differs")
    if design.manifest["modes"] != direct.manifest["modes"]:
        raise ValueError("Direct-coordinate and rendered-design mode order differs")
    flow_sources = direct.manifest.get("flow_artifacts")
    views = direct.manifest["views"]
    design_views = design.manifest["views"]
    if (
        not isinstance(flow_sources, list)
        or len(flow_sources) != len(views)
        or len(design_views) != len(views)
    ):
        raise ValueError("Direct-coordinate flow source count differs")
    flows: list[FlowAnalysisArtifact] = []
    for index, (source, view, design_view) in enumerate(
        zip(flow_sources, views, design_views)
    ):
        if not isinstance(source, str) or not source:
            raise ValueError("Direct-coordinate flow source path is invalid")
        flow = load_flow_analysis_artifact(source)
        identity = flow_artifact_identity(flow)
        if (
            view.get("index") != index
            or design_view.get("index") != index
            or view.get("label") != design_view.get("label")
            or identity != view.get("flow_identity")
            or identity != design_view.get("flow_identity")
        ):
            raise ValueError("Physics-coordinate ordered flow identity differs")
        flows.append(flow)
    return direct, design, tuple(flows)


def _compare_direct_evaluation(
    direct: DirectModalCoordinatesArtifact,
    diagnostics: Mapping[str, np.ndarray],
    overall: Mapping[str, float],
) -> None:
    """Require recomputed input flow metrics to match the direct artifact."""

    comparisons = {
        "input_per_frame_flow_rmse": direct.diagnostics["per_frame_flow_rmse"],
        "input_per_frame_relative_residual": direct.diagnostics[
            "per_frame_relative_residual"
        ],
        "input_per_frame_flow_r2": direct.diagnostics["per_frame_flow_r2"],
        "input_view_flow_rmse": direct.diagnostics["view_flow_rmse"],
        "input_view_relative_residual": direct.diagnostics["view_relative_residual"],
        "input_view_flow_r2": direct.diagnostics["view_flow_r2"],
    }
    for name, expected in comparisons.items():
        if not np.allclose(diagnostics[name], expected, rtol=5.0e-7, atol=1.0e-9):
            raise RuntimeError(f"Physics input evaluation differs from direct {name}")
    expected_overall = direct.manifest["overall"]
    for name in ("flow_rmse", "relative_residual", "flow_r2"):
        if not math.isclose(
            float(overall[f"input_{name}"]),
            float(expected_overall[name]),
            rel_tol=5.0e-7,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(f"Physics input overall {name} differs from direct")


def build_physics_modal_coordinates_artifact(
    *,
    direct_coordinates_dir: str | Path,
    output_dir: str | Path,
    config: PhysicsCoordinateConfig | None = None,
    command: Sequence[str] = (),
) -> PhysicsModalCoordinatesArtifact:
    """Post-fit direct coordinates and atomically publish the physics artifact."""

    settings = config or PhysicsCoordinateConfig()
    settings.validate()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Physics-coordinate output already exists: {destination}")
    total_start = time.perf_counter()
    direct, design, flows = _load_sources(direct_coordinates_dir)
    modes = [dict(mode) for mode in direct.manifest["modes"]]
    _validate_modes(modes)
    frequencies = np.asarray(
        [mode["frequency_hz"] for mode in modes], dtype=np.float64
    )
    views = direct.manifest["views"]
    view_count = len(views)
    frame_count, mode_count = direct.coordinates.shape
    input_values = np.asarray(direct.coordinates, dtype=np.complex128)
    output_values = np.empty_like(input_values)
    diagnostics: dict[str, np.ndarray] = {
        name: np.empty((view_count, mode_count), dtype=np.float64)
        for name in MODE_DIAGNOSTIC_NAMES
    }
    diagnostics["input_cross_mode_correlation"] = np.empty(
        (view_count, mode_count, mode_count), dtype=np.float64
    )
    diagnostics["output_cross_mode_correlation"] = np.empty(
        (view_count, mode_count, mode_count), dtype=np.float64
    )
    for name in FLOW_FRAME_DIAGNOSTIC_NAMES:
        diagnostics[name] = np.empty(frame_count, dtype=np.float64)
    for name in FLOW_VIEW_DIAGNOSTIC_NAMES:
        diagnostics[name] = np.empty(view_count, dtype=np.float64)

    solve_start = time.perf_counter()
    progress = Progress("physics coordinate fit", view_count * mode_count, unit="view-modes")
    for view_index, view in enumerate(views):
        lower = int(view["frame_offset"])
        upper = lower + int(view["frame_count"])
        times = np.arange(upper - lower, dtype=np.float64) / float(view["fps_hz"])
        view_input = input_values[lower:upper]
        view_output = np.empty_like(view_input)
        for mode_slot, frequency in enumerate(frequencies):
            fitted, mode_diagnostics = solve_physics_coordinate_mode(
                source=view_input[:, mode_slot],
                times_sec=times,
                frequency_hz=float(frequency),
                config=settings,
            )
            view_output[:, mode_slot] = fitted
            for name, value in mode_diagnostics.items():
                diagnostics[name][view_index, mode_slot] = value
            input_dominant, input_ratio = _spectral_metrics(
                view_input[:, mode_slot],
                times,
                float(frequency),
                settings.assigned_band_half_width_hz,
            )
            output_dominant, output_ratio = _spectral_metrics(
                fitted,
                times,
                float(frequency),
                settings.assigned_band_half_width_hz,
            )
            diagnostics["input_dominant_signed_frequency_hz"][view_index, mode_slot] = input_dominant
            diagnostics["output_dominant_signed_frequency_hz"][view_index, mode_slot] = output_dominant
            diagnostics["input_assigned_frequency_energy_ratio"][view_index, mode_slot] = input_ratio
            diagnostics["output_assigned_frequency_energy_ratio"][view_index, mode_slot] = output_ratio
            progress.update(
                view_index * mode_count + mode_slot + 1,
                f"view={view['label']} mode_slot={mode_slot} hz={frequency:.6g}",
            )
        if settings.forcing_weight > 0.0 or settings.forcing_difference_weight > 0.0:
            view_output -= np.mean(view_output, axis=0, keepdims=True)
        if not np.isfinite(view_output).all():
            raise FloatingPointError(f"Physics coordinates for view {view['label']!r} are non-finite")
        output_values[lower:upper] = view_output
        diagnostics["input_cross_mode_correlation"][view_index] = _complex_correlation(view_input)
        diagnostics["output_cross_mode_correlation"][view_index] = _complex_correlation(view_output)
    solve_seconds = float(time.perf_counter() - solve_start)
    output_values = output_values.astype(np.complex64).astype(np.complex128)

    total_by_label = {
        "input": {"residual": 0.0, "flow": 0.0, "observations": 0.0},
        "output": {"residual": 0.0, "flow": 0.0, "observations": 0.0},
    }
    sample_offsets = design.samples["view_sample_offsets"]
    pixels = design.samples["sample_pixels_xy"]
    view_records: list[dict[str, Any]] = []
    for view_index, (view, flow) in enumerate(zip(views, flows)):
        frame_lower = int(view["frame_offset"])
        frame_upper = frame_lower + int(view["frame_count"])
        sample_lower = int(sample_offsets[view_index])
        sample_upper = int(sample_offsets[view_index + 1])
        evaluation = _evaluate_coordinate_sets_view(
            design=design.design[sample_lower:sample_upper],
            pixels_xy=pixels[sample_lower:sample_upper],
            flow=flow.arrays.flow,
            reference_frame_index=int(view["reference_frame_index"]),
            coordinate_sets={
                "input": input_values[frame_lower:frame_upper],
                "output": output_values[frame_lower:frame_upper],
            },
            frame_chunk_size=settings.frame_chunk_size,
        )
        for label in ("input", "output"):
            result = evaluation[label]
            diagnostics[f"{label}_per_frame_flow_rmse"][frame_lower:frame_upper] = result[
                "per_frame_flow_rmse"
            ]
            diagnostics[f"{label}_per_frame_relative_residual"][frame_lower:frame_upper] = result[
                "per_frame_relative_residual"
            ]
            diagnostics[f"{label}_per_frame_flow_r2"][frame_lower:frame_upper] = result[
                "per_frame_flow_r2"
            ]
            for name in (
                "flow_rmse",
                "relative_residual",
                "flow_r2",
                "strong_motion_flow_r2",
            ):
                diagnostics[f"{label}_view_{name}"][view_index] = float(result[name])
            total_by_label[label]["residual"] += float(result["residual_sum_squares"])
            total_by_label[label]["flow"] += float(result["flow_sum_squares"])
            total_by_label[label]["observations"] += float(result["observation_count"])
        view_record = {
            name: value
            for name, value in view.items()
            if name != "diagnostics"
        }
        view_record["input_flow_r2"] = float(evaluation["input"]["flow_r2"])
        view_record["output_flow_r2"] = float(evaluation["output"]["flow_r2"])
        view_record["input_strong_motion_flow_r2"] = float(
            evaluation["input"]["strong_motion_flow_r2"]
        )
        view_record["output_strong_motion_flow_r2"] = float(
            evaluation["output"]["strong_motion_flow_r2"]
        )
        view_records.append(view_record)

    diagnostics = {
        name: np.ascontiguousarray(value, dtype=DIAGNOSTIC_DTYPES[name])
        for name, value in diagnostics.items()
    }
    _validate_diagnostics(
        diagnostics,
        view_count=view_count,
        frame_count=frame_count,
        mode_count=mode_count,
    )
    overall: dict[str, float] = {}
    for label, totals in total_by_label.items():
        residual = totals["residual"]
        flow = totals["flow"]
        observations = totals["observations"]
        overall[f"{label}_flow_rmse"] = math.sqrt(residual / observations)
        overall[f"{label}_relative_residual"] = (
            math.sqrt(residual / flow) if flow > 0.0 else 0.0
        )
        overall[f"{label}_flow_r2"] = 1.0 - residual / flow if flow > 0.0 else 1.0
    overall.update(
        {
            "flow_r2_delta": overall["output_flow_r2"] - overall["input_flow_r2"],
            "mean_fidelity_nrmse": float(np.mean(diagnostics["fidelity_nrmse"])),
            "mean_rms_retention": float(np.mean(diagnostics["rms_retention"])),
            "mean_p90_retention": float(np.mean(diagnostics["p90_retention"])),
            "mean_p99_retention": float(np.mean(diagnostics["p99_retention"])),
            "mean_input_assigned_frequency_energy_ratio": float(
                np.mean(diagnostics["input_assigned_frequency_energy_ratio"])
            ),
            "mean_output_assigned_frequency_energy_ratio": float(
                np.mean(diagnostics["output_assigned_frequency_energy_ratio"])
            ),
            "mean_input_forcing_normalized_rms": float(
                np.mean(diagnostics["input_forcing_normalized_rms"])
            ),
            "mean_output_forcing_normalized_rms": float(
                np.mean(diagnostics["output_forcing_normalized_rms"])
            ),
        }
    )
    _compare_direct_evaluation(direct, diagnostics, overall)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        coordinates_path = temporary / COORDINATES_FILENAME
        diagnostics_path = temporary / DIAGNOSTICS_FILENAME
        np.save(
            coordinates_path,
            output_values.astype(COORDINATES_DTYPE),
            allow_pickle=False,
        )
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
            "format": PHYSICS_COORDINATES_FORMAT,
            "version": PHYSICS_COORDINATES_VERSION,
            "producer": {
                "project_version": __version__,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": list(command),
            },
            "direct_coordinates": str(direct.path),
            "direct_coordinates_identity": direct.manifest[
                "direct_coordinates_identity"
            ],
            "rendered_design": str(design.path),
            "rendered_design_identity": direct.manifest["rendered_design_identity"],
            "completed_modes_identity": direct.manifest["completed_modes_identity"],
            "flow_artifacts": [str(flow.path.resolve()) for flow in flows],
            "modes": modes,
            "views": view_records,
            "solver": dict(SOLVER_CONVENTION),
            "settings": settings.to_dict(),
            "quality_gate": {
                "required": True,
                "status": "physics_coordinates_candidate_unapproved",
                "inherited_from": "direct_coordinates_candidate_unapproved",
            },
            "counts": {
                "views": view_count,
                "frames": frame_count,
                "modes": mode_count,
            },
            "coordinates": coordinate_record,
            "diagnostics": diagnostic_record,
            "overall": overall,
            "timings_seconds": {
                "physics_solve": solve_seconds,
                "total": float(time.perf_counter() - total_start),
            },
        }
        manifest["physics_coordinates_identity"] = hashlib.sha256(
            _canonical_json(_identity_payload(manifest))
        ).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_physics_modal_coordinates(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Physics-coordinate output already exists: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_physics_modal_coordinates(destination)


__all__ = [
    "PhysicsCoordinateConfig",
    "PhysicsModalCoordinatesArtifact",
    "build_physics_modal_coordinates_artifact",
    "finite_difference_matrix",
    "load_physics_modal_coordinates",
    "solve_physics_coordinate_mode",
]
