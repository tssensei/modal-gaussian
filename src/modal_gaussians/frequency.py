"""Shared exact-DFT greedy frequency selection from fixed-view optical flow."""

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
from typing import Any, Mapping, Sequence

import numpy as np
from modal_gaussians.progress import Progress

from modal_gaussians.numpy_io import save_named_arrays

from modal_gaussians import __version__
from modal_gaussians.flow.artifact import (
    FlowAnalysisArtifact,
    flow_artifact_identity,
    load_flow_analysis_artifact,
)
from modal_gaussians.flow.spectrum import exact_dft_basis
from modal_gaussians.flow.storage import read_pixels
from modal_gaussians.topology import load_observation_topology


FREQUENCY_FORMAT = "modal_gaussians.frequency_selection"
FREQUENCY_VERSION = 1
ARRAY_FILENAME = "selection.npz"
PIXEL_CHUNK_SIZE = 1024
GREEDY_TIE_TOLERANCE = 1e-12

ARRAY_DTYPES = {
    "candidate_frequencies_hz": np.dtype(np.float64),
    "selected_candidate_index": np.dtype(np.int64),
    "selected_frequencies_hz": np.dtype(np.float64),
    "marginal_macro_r2_gain": np.dtype(np.float64),
    "view_r2": np.dtype(np.float64),
    "view_sse": np.dtype(np.float64),
    "pooled_r2": np.dtype(np.float64),
    "macro_r2": np.dtype(np.float64),
    "worst_view_r2": np.dtype(np.float64),
    "numerical_rank": np.dtype(np.int64),
    "observable_condition": np.dtype(np.float64),
    "flow_energy": np.dtype(np.float64),
    "candidate_pixel_count": np.dtype(np.int64),
}


@dataclass(frozen=True)
class FrequencySelectionConfig:
    """Define one inclusive candidate grid and the requested greedy prefix."""

    minimum_hz: float
    maximum_hz: float
    step_hz: float
    count: int

    def frequency_grid(self) -> np.ndarray:
        """Build the accepted inclusive grid and reject misaligned endpoints."""

        limits = (self.minimum_hz, self.maximum_hz, self.step_hz)
        if not all(math.isfinite(float(value)) for value in limits):
            raise ValueError("Frequency limits and step must be finite")
        if not 0.0 < self.minimum_hz < self.maximum_hz or self.step_hz <= 0.0:
            raise ValueError("Frequencies must satisfy 0 < min < max and step > 0")
        intervals_float = (self.maximum_hz - self.minimum_hz) / self.step_hz
        intervals = int(round(intervals_float))
        tolerance = 1e-10 * max(1.0, abs(intervals_float))
        if abs(intervals_float - intervals) > tolerance:
            raise ValueError(
                "Frequency endpoints must align to an integral number of steps"
            )
        frequencies = self.minimum_hz + self.step_hz * np.arange(
            intervals + 1, dtype=np.float64
        )
        frequencies[-1] = self.maximum_hz
        if self.count < 1 or self.count > len(frequencies):
            raise ValueError(
                f"count must be between 1 and {len(frequencies)}, got {self.count}"
            )
        return frequencies

    def to_dict(self) -> dict[str, Any]:
        """Record the complete scientific selection convention."""

        return {
            "minimum_hz": float(self.minimum_hz),
            "maximum_hz": float(self.maximum_hz),
            "step_hz": float(self.step_hz),
            "count": int(self.count),
            "objective": "equal_view_macro_r2_grouped_complex_pair",
            "candidate_transform": "exact_dft_at_requested_frequencies",
            "detrend": "temporal_mean",
            "window": "hann_symmetric",
            "time_axis": "local_frame_index_divided_by_fps",
            "target": "reference_relative_raw_flow",
            "tie_break": "lowest_frequency_within_1e-12",
        }


@dataclass(frozen=True)
class FrequencyViewInput:
    """Bind one topology view label to its exact flow artifact."""

    label: str
    flow_artifact: Path


@dataclass(frozen=True)
class FrequencySelectionArrays:
    """Store the selected prefix and compact per-prefix diagnostics."""

    candidate_frequencies_hz: np.ndarray
    selected_candidate_index: np.ndarray
    selected_frequencies_hz: np.ndarray
    marginal_macro_r2_gain: np.ndarray
    view_r2: np.ndarray
    view_sse: np.ndarray
    pooled_r2: np.ndarray
    macro_r2: np.ndarray
    worst_view_r2: np.ndarray
    numerical_rank: np.ndarray
    observable_condition: np.ndarray
    flow_energy: np.ndarray
    candidate_pixel_count: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return all arrays in stable field order and canonical dtypes."""

        return {
            name: np.ascontiguousarray(getattr(self, name), dtype=dtype)
            for name, dtype in ARRAY_DTYPES.items()
        }


@dataclass(frozen=True)
class FrequencySelectionArtifact:
    """Represent one fully validated automatic frequency-selection artifact."""

    path: Path
    manifest: dict[str, Any]
    arrays: FrequencySelectionArrays


@dataclass(frozen=True)
class _ViewStatistics:
    """Hold sufficient statistics for every candidate pair in one view."""

    label: str
    pixel_count: int
    gram: np.ndarray
    cross_flow: np.ndarray
    energy_per_frame: np.ndarray
    flow_energy: float
    pair_scales: np.ndarray


@dataclass(frozen=True)
class _Fit:
    """Summarize one grouped least-squares fit without retaining coefficients."""

    sse: float
    r2: float
    numerical_rank: int
    observable_condition: float


def _canonical_json(value: Any) -> bytes:
    """Encode identity fields deterministically and reject NaN."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    """Hash one file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    """Hash one array's dtype, shape, and C-order contents."""

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _arrays_identity(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash every selection array in stable semantic order."""

    digest = hashlib.sha256()
    for name in ARRAY_DTYPES:
        digest.update(name.encode("utf-8"))
        digest.update(_sha256_array(arrays[name]).encode("ascii"))
    return digest.hexdigest()


def _accumulate_view_statistics(
    label: str,
    artifact: FlowAnalysisArtifact,
    pixels_xy: np.ndarray,
    frequencies_hz: np.ndarray,
) -> _ViewStatistics:
    """Reduce one view's sampled flow into exact-DFT normal-equation statistics."""

    flow = artifact.arrays.flow
    frame_count = flow.shape[0]
    reference_index = int(artifact.manifest["reference_frame_index"])
    basis, window = exact_dft_basis(
        frame_count, float(artifact.manifest["fps_hz"]), frequencies_hz
    )
    column_count = 2 * len(frequencies_hz)
    gram = np.zeros((column_count, column_count), dtype=np.float64)
    cross_flow = np.zeros((column_count, frame_count), dtype=np.float64)
    energy_per_frame = np.zeros(frame_count, dtype=np.float64)

    progress = Progress(f"frequency exact DFT {label}", len(pixels_xy), unit="pixels")
    for lower in range(0, len(pixels_xy), PIXEL_CHUNK_SIZE):
        current = pixels_xy[lower : lower + PIXEL_CHUNK_SIZE]
        raw_u = read_pixels(flow, slice(None), current, 0)
        raw_v = read_pixels(flow, slice(None), current, 1)
        if not np.isfinite(raw_u).all() or not np.isfinite(raw_v).all():
            raise ValueError(f"Flow artifact for {label!r} contains NaN or Inf")

        target_u = raw_u.astype(np.float64)
        target_v = raw_v.astype(np.float64)
        target_u -= target_u[reference_index : reference_index + 1]
        target_v -= target_v[reference_index : reference_index + 1]
        target = np.empty((2 * len(current), frame_count), dtype=np.float64)
        target[0::2] = target_u.T
        target[1::2] = target_v.T

        processed_u = raw_u - raw_u.mean(axis=0, keepdims=True, dtype=np.float32)
        processed_v = raw_v - raw_v.mean(axis=0, keepdims=True, dtype=np.float32)
        mode_u = basis @ (processed_u * window[:, None])
        mode_v = basis @ (processed_v * window[:, None])
        design = np.empty((2 * len(current), column_count), dtype=np.float64)
        design[0::2, 0::2] = mode_u.T.real
        design[1::2, 0::2] = mode_v.T.real
        design[0::2, 1::2] = -mode_u.T.imag
        design[1::2, 1::2] = -mode_v.T.imag
        if not np.isfinite(design).all():
            raise ValueError(f"Exact-DFT design for {label!r} contains NaN or Inf")
        gram += design.T @ design
        cross_flow += design.T @ target
        energy_per_frame += np.sum(target * target, axis=0)
        progress.update(lower + len(current))

    flow_energy = float(np.sum(energy_per_frame))
    if not math.isfinite(flow_energy) or flow_energy <= np.finfo(np.float64).eps:
        raise ValueError(f"Topology-sampled flow for {label!r} has zero energy")
    diagonal = np.diag(gram)
    pair_scales = np.sqrt(
        (diagonal[0::2] + diagonal[1::2]) / float(2 * len(pixels_xy))
    )
    if not np.isfinite(pair_scales).all():
        raise ValueError(f"Exact-DFT pair scales for {label!r} are non-finite")
    pair_scales[pair_scales <= np.finfo(np.float64).eps] = 1.0
    return _ViewStatistics(
        label=label,
        pixel_count=len(pixels_xy),
        gram=gram,
        cross_flow=cross_flow,
        energy_per_frame=energy_per_frame,
        flow_energy=flow_energy,
        pair_scales=pair_scales,
    )


def _selected_columns(selected: Sequence[int]) -> np.ndarray:
    """Expand frequency-group indices into paired real/imaginary columns."""

    columns = np.empty(2 * len(selected), dtype=np.int64)
    columns[0::2] = 2 * np.asarray(selected, dtype=np.int64)
    columns[1::2] = columns[0::2] + 1
    return columns


def _fit(statistics: _ViewStatistics, selected: Sequence[int]) -> _Fit:
    """Fit selected complex pairs with a rank-aware minimum-norm solve."""

    columns = _selected_columns(selected)
    scales = np.repeat(
        statistics.pair_scales[np.asarray(selected, dtype=np.int64)], 2
    )
    gram = statistics.gram[np.ix_(columns, columns)] / (
        scales[:, None] * scales[None, :]
    )
    gram = (gram + gram.T) * 0.5
    cross = statistics.cross_flow[columns] / scales[:, None]
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = max(float(eigenvalues[-1]), 0.0)
    tolerance = (
        np.finfo(np.float64).eps
        * max(len(columns), 2 * statistics.pixel_count)
        * largest
    )
    observable = eigenvalues > tolerance
    rank = int(np.count_nonzero(observable))
    if rank:
        vectors = eigenvectors[:, observable]
        scaled_coefficients = vectors @ (
            (vectors.T @ cross) / eigenvalues[observable, None]
        )
        condition = float(
            np.sqrt(eigenvalues[observable][-1] / eigenvalues[observable][0])
        )
    else:
        scaled_coefficients = np.zeros_like(cross)
        condition = 1.0
    captured_per_frame = np.sum(cross * scaled_coefficients, axis=0)
    frame_sse = statistics.energy_per_frame - captured_per_frame
    negative_tolerance = 1e-9 * max(statistics.flow_energy, 1.0)
    if float(np.min(frame_sse)) < -negative_tolerance:
        raise ValueError("Modal projection captured more energy than the flow contains")
    sse = float(np.sum(np.maximum(frame_sse, 0.0)))
    return _Fit(
        sse=sse,
        r2=float(1.0 - sse / statistics.flow_energy),
        numerical_rank=rank,
        observable_condition=condition,
    )


def _greedy_select(
    view_statistics: Sequence[_ViewStatistics],
    frequencies_hz: np.ndarray,
    count: int,
) -> FrequencySelectionArrays:
    """Choose a shared ordered prefix by equal-view macro-R2 improvement."""

    view_count = len(view_statistics)
    selected: list[int] = []
    available = set(range(len(frequencies_hz)))
    indices = np.empty(count, dtype=np.int64)
    view_r2 = np.empty((view_count, count), dtype=np.float64)
    view_sse = np.empty((view_count, count), dtype=np.float64)
    ranks = np.empty((view_count, count), dtype=np.int64)
    conditions = np.empty((view_count, count), dtype=np.float64)
    gains = np.empty(count, dtype=np.float64)
    previous_macro = 0.0

    progress = Progress("greedy selection", count, unit="frequencies")
    for step in range(count):
        best_candidate: int | None = None
        best_macro = -np.inf
        best_fits: list[_Fit] | None = None
        for candidate in sorted(available):
            fits = [
                _fit(statistics, [*selected, candidate])
                for statistics in view_statistics
            ]
            macro = float(np.mean([fit.r2 for fit in fits]))
            if macro > best_macro + GREEDY_TIE_TOLERANCE:
                best_candidate, best_macro, best_fits = candidate, macro, fits
        if best_candidate is None or best_fits is None:
            raise RuntimeError(f"Greedy selection failed at step {step + 1}")
        gain = best_macro - previous_macro
        if gain < -1e-9:
            raise ValueError("Greedy macro R2 decreased after adding a frequency")
        indices[step] = best_candidate
        for view_index, fit in enumerate(best_fits):
            view_r2[view_index, step] = fit.r2
            view_sse[view_index, step] = fit.sse
            ranks[view_index, step] = fit.numerical_rank
            conditions[view_index, step] = fit.observable_condition
        gains[step] = max(gain, 0.0)
        selected.append(best_candidate)
        available.remove(best_candidate)
        previous_macro = best_macro
        progress.update(
            step + 1,
            f"selected_hz={frequencies_hz[best_candidate]:.6g} "
            f"macro_r2={best_macro:.6f} gain={gains[step]:.6f}",
            force=True,
        )

    energy = np.asarray(
        [statistics.flow_energy for statistics in view_statistics], dtype=np.float64
    )
    macro_r2 = np.mean(view_r2, axis=0)
    return FrequencySelectionArrays(
        candidate_frequencies_hz=frequencies_hz,
        selected_candidate_index=indices,
        selected_frequencies_hz=frequencies_hz[indices],
        marginal_macro_r2_gain=gains,
        view_r2=view_r2,
        view_sse=view_sse,
        pooled_r2=1.0 - np.sum(view_sse, axis=0) / float(np.sum(energy)),
        macro_r2=macro_r2,
        worst_view_r2=np.min(view_r2, axis=0),
        numerical_rank=ranks,
        observable_condition=conditions,
        flow_energy=energy,
        candidate_pixel_count=np.asarray(
            [statistics.pixel_count for statistics in view_statistics], dtype=np.int64
        ),
    )


def _validate_arrays(
    arrays: Mapping[str, np.ndarray], *, view_count: int, count: int
) -> None:
    """Validate shapes, dtypes, ordering, and all derived prefix metrics."""

    if set(arrays) != set(ARRAY_DTYPES):
        raise ValueError("Frequency selection array inventory is incomplete")
    for name, dtype in ARRAY_DTYPES.items():
        if arrays[name].dtype != dtype:
            raise ValueError(f"Frequency array {name} must use dtype {dtype.name}")
    candidates = arrays["candidate_frequencies_hz"]
    indices = arrays["selected_candidate_index"]
    selected = arrays["selected_frequencies_hz"]
    if (
        candidates.ndim != 1
        or len(candidates) < count
        or not np.isfinite(candidates).all()
        or np.any(candidates <= 0.0)
        or np.any(np.diff(candidates) <= 0.0)
    ):
        raise ValueError("Candidate frequency grid is invalid")
    if (
        indices.shape != (count,)
        or len(np.unique(indices)) != count
        or np.any(indices < 0)
        or np.any(indices >= len(candidates))
        or selected.shape != (count,)
        or not np.array_equal(selected, candidates[indices])
    ):
        raise ValueError("Selected frequency prefix is invalid")
    per_prefix = (
        "marginal_macro_r2_gain",
        "pooled_r2",
        "macro_r2",
        "worst_view_r2",
    )
    per_view = ("view_r2", "view_sse", "numerical_rank", "observable_condition")
    if any(arrays[name].shape != (count,) for name in per_prefix):
        raise ValueError("Frequency prefix metric shapes are invalid")
    if any(arrays[name].shape != (view_count, count) for name in per_view):
        raise ValueError("Frequency per-view metric shapes are invalid")
    if arrays["flow_energy"].shape != (view_count,) or arrays[
        "candidate_pixel_count"
    ].shape != (view_count,):
        raise ValueError("Frequency view summary shapes are invalid")
    float_names = [name for name in ARRAY_DTYPES if ARRAY_DTYPES[name].kind == "f"]
    if any(not np.isfinite(arrays[name]).all() for name in float_names):
        raise ValueError("Frequency selection contains NaN or Inf")
    if np.any(arrays["flow_energy"] <= 0.0) or np.any(
        arrays["candidate_pixel_count"] <= 0
    ):
        raise ValueError("Frequency selection view energy or pixel count is invalid")
    if not np.allclose(arrays["macro_r2"], np.mean(arrays["view_r2"], axis=0)):
        raise ValueError("macro R2 is inconsistent with per-view R2")
    if not np.allclose(arrays["worst_view_r2"], np.min(arrays["view_r2"], axis=0)):
        raise ValueError("worst-view R2 is inconsistent with per-view R2")
    expected_pooled = 1.0 - np.sum(arrays["view_sse"], axis=0) / float(
        np.sum(arrays["flow_energy"])
    )
    if not np.allclose(arrays["pooled_r2"], expected_pooled):
        raise ValueError("pooled R2 is inconsistent with per-view SSE")
    expected_gain = np.diff(np.concatenate([[0.0], arrays["macro_r2"]]))
    if np.any(expected_gain < -1e-9) or not np.allclose(
        arrays["marginal_macro_r2_gain"], np.maximum(expected_gain, 0.0)
    ):
        raise ValueError("Marginal macro R2 gains are inconsistent")


def _identity_payload(
    manifest: Mapping[str, Any], arrays_identity: str
) -> dict[str, Any]:
    """Select only path-independent scientific fields for artifact identity."""

    return {
        "format": FREQUENCY_FORMAT,
        "version": FREQUENCY_VERSION,
        "topology_identity": manifest["topology_identity"],
        "views": [
            {"label": view["label"], "flow_identity": view["flow_identity"]}
            for view in manifest["views"]
        ],
        "parameters": manifest["parameters"],
        "arrays_identity": arrays_identity,
    }


def load_frequency_selection(path: str | Path) -> FrequencySelectionArtifact:
    """Load and fully validate one automatic frequency-selection directory."""

    root = Path(path).expanduser().resolve(strict=True)
    manifest_path, arrays_path = root / "manifest.json", root / ARRAY_FILENAME
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"Incomplete frequency selection artifact: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != FREQUENCY_FORMAT:
        raise ValueError(f"Unsupported frequency selection artifact: {manifest_path}")
    if manifest.get("version") != FREQUENCY_VERSION:
        raise ValueError("Unsupported frequency selection artifact version")
    views = manifest.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("Frequency manifest must contain ordered views")
    labels = [view.get("label") for view in views if isinstance(view, dict)]
    if len(labels) != len(views) or any(
        not isinstance(label, str) or not label for label in labels
    ) or len(set(labels)) != len(labels):
        raise ValueError("Frequency manifest view labels are invalid")
    if manifest.get("arrays_file_sha256") != _sha256_file(arrays_path):
        raise ValueError("Frequency NPZ SHA-256 does not match manifest")
    with np.load(arrays_path, allow_pickle=False) as archive:
        loaded = {name: archive[name] for name in archive.files}
    count = int(manifest["parameters"]["count"])
    _validate_arrays(loaded, view_count=len(views), count=count)
    arrays_identity = _arrays_identity(loaded)
    if manifest.get("arrays_identity") != arrays_identity:
        raise ValueError("Frequency array identity does not match manifest")
    expected_identity = hashlib.sha256(
        _canonical_json(_identity_payload(manifest, arrays_identity))
    ).hexdigest()
    if manifest.get("frequency_selection_identity") != expected_identity:
        raise ValueError("Frequency selection identity does not match its contents")
    if manifest.get("selected_frequencies_hz") != loaded[
        "selected_frequencies_hz"
    ].tolist():
        raise ValueError("Manifest selected frequencies do not match selection.npz")
    return FrequencySelectionArtifact(
        root, manifest, FrequencySelectionArrays(**loaded)
    )


def _publish_selection(
    destination: Path,
    *,
    arrays: FrequencySelectionArrays,
    manifest: dict[str, Any],
) -> FrequencySelectionArtifact:
    """Atomically publish and reload the minimal two-file selection artifact."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Frequency selection output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        values = arrays.as_dict()
        arrays_path = temporary / ARRAY_FILENAME
        save_named_arrays(arrays_path, values)
        arrays_identity = _arrays_identity(values)
        manifest["arrays"] = {
            name: {"dtype": value.dtype.name, "shape": list(value.shape)}
            for name, value in values.items()
        }
        manifest["arrays_file"] = ARRAY_FILENAME
        manifest["arrays_file_sha256"] = _sha256_file(arrays_path)
        manifest["arrays_identity"] = arrays_identity
        manifest["selected_frequencies_hz"] = values[
            "selected_frequencies_hz"
        ].tolist()
        manifest["frequency_selection_identity"] = hashlib.sha256(
            _canonical_json(_identity_payload(manifest, arrays_identity))
        ).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_frequency_selection(temporary)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_frequency_selection(destination)


def build_frequency_selection_artifact(
    *,
    topology_dir: str | Path,
    views: Sequence[FrequencyViewInput],
    output_dir: str | Path,
    config: FrequencySelectionConfig,
    command: Sequence[str] = (),
) -> FrequencySelectionArtifact:
    """Validate upstream identities, select the first K frequencies, and publish."""

    frequencies_hz = config.frequency_grid()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Frequency selection output already exists: {destination}")
    topology = load_observation_topology(topology_dir)
    expected_labels = [view["label"] for view in topology.manifest["views"]]
    labels = [view.label.strip() for view in views]
    if labels != expected_labels:
        raise ValueError(
            f"Frequency --view labels must follow topology order {expected_labels}, "
            f"got {labels}"
        )

    statistics: list[_ViewStatistics] = []
    view_records: list[dict[str, Any]] = []
    sample_views = topology.arrays.sample_view_index
    sample_pixels = topology.arrays.sample_pixels_xy
    for view_index, (specification, topology_view) in enumerate(
        zip(views, topology.manifest["views"])
    ):
        artifact = load_flow_analysis_artifact(specification.flow_artifact)
        identity = flow_artifact_identity(artifact)
        if identity != topology_view["flow_identity"]:
            raise ValueError(
                f"Flow artifact for {specification.label!r} does not match topology"
            )
        frame_count, height, width, _ = artifact.arrays.flow.shape
        if [height, width] != topology_view["shape_hw"]:
            raise ValueError(
                f"Flow shape for {specification.label!r} does not match topology"
            )
        if float(frequencies_hz[-1]) > 0.5 * float(artifact.manifest["fps_hz"]) + 1e-12:
            raise ValueError(
                f"Maximum frequency exceeds Nyquist for {specification.label!r}"
            )
        fft = artifact.manifest["parameters"].get("fft", {})
        if fft.get("detrend") != "temporal-mean" or fft.get("window") != "hann-symmetric":
            raise ValueError(
                f"Flow FFT convention for {specification.label!r} is unsupported"
            )
        pixels = np.asarray(
            sample_pixels[sample_views == view_index], dtype=np.int64
        )
        if not len(pixels):
            raise ValueError(f"Topology view {specification.label!r} has no samples")
        print(
            f"Exact DFT {specification.label}: pixels={len(pixels)}, "
            f"frames={frame_count}, candidates={len(frequencies_hz)}",
            flush=True,
        )
        statistics.append(
            _accumulate_view_statistics(
                specification.label, artifact, pixels, frequencies_hz
            )
        )
        view_records.append(
            {
                "index": view_index,
                "label": specification.label,
                "flow_artifact": str(artifact.path.resolve()),
                "flow_identity": identity,
                "frame_count": frame_count,
                "fps_hz": float(artifact.manifest["fps_hz"]),
                "reference_frame_name": artifact.manifest["reference_frame_name"],
                "candidate_pixel_count": len(pixels),
            }
        )

    print(f"Selecting {config.count} shared frequencies", flush=True)
    arrays = _greedy_select(statistics, frequencies_hz, config.count)
    _validate_arrays(arrays.as_dict(), view_count=len(views), count=config.count)
    manifest = {
        "format": FREQUENCY_FORMAT,
        "version": FREQUENCY_VERSION,
        "producer": {
            "project_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": list(command),
        },
        "topology_artifact": str(topology.path),
        "topology_identity": topology.manifest["topology_identity"],
        "views": view_records,
        "parameters": config.to_dict(),
    }
    return _publish_selection(destination, arrays=arrays, manifest=manifest)


__all__ = [
    "FrequencySelectionArtifact",
    "FrequencySelectionArrays",
    "FrequencySelectionConfig",
    "FrequencyViewInput",
    "build_frequency_selection_artifact",
    "load_frequency_selection",
]
