"""Dense complex 2D modal fields at the greedily selected frequencies."""

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

from modal_gaussians import __version__
from modal_gaussians.flow.artifact import (
    FlowAnalysisArtifact,
    flow_artifact_identity,
    load_flow_analysis_artifact,
)
from modal_gaussians.flow.spectrum import exact_dft_basis
from modal_gaussians.flow.storage import spatial_blocks
from modal_gaussians.frequency import load_frequency_selection
from modal_gaussians.progress import Progress
from modal_gaussians.iteration_cache import DEFAULT_CACHE, load_entry, put_entry


MODES_FORMAT = "modal_gaussians.complex_2d_modes"
MODES_VERSION = 1
DENSE_DFT_BLOCK_WIDTH = 32
MODES_DTYPE = np.dtype(np.complex64)

TRANSFORM_CONVENTION = {
    "method": "exact_dft_at_selected_frequencies",
    "exponent": "exp(-2j*pi*frequency_hz*frame_index/fps_hz)",
    "detrend": "temporal_mean",
    "window": "hann_symmetric",
    "normalization": "none",
    "amplitude_clamp": "none",
    "mask_application": "none",
}


@dataclass(frozen=True)
class ComplexModeViewInput:
    """Bind one ordered selection view label to its exact flow artifact."""

    label: str
    flow_artifact: Path


@dataclass(frozen=True)
class Complex2DModesArtifact:
    """Represent validated per-view dense complex mode memmaps."""

    path: Path
    manifest: dict[str, Any]
    view_modes: tuple[np.ndarray, ...]


def _canonical_json(value: Any) -> bytes:
    """Encode path-independent identity fields deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    """Hash one potentially large mode array as a byte stream."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select scientific fields that define one dense-mode artifact."""

    return {
        "format": MODES_FORMAT,
        "version": MODES_VERSION,
        "frequency_selection_identity": manifest["frequency_selection_identity"],
        "topology_identity": manifest["topology_identity"],
        "modes": manifest["modes"],
        "transform": manifest["transform"],
        "views": [
            {
                "index": view["index"],
                "label": view["label"],
                "flow_identity": view["flow_identity"],
                "frame_count": view["frame_count"],
                "fps_hz": view["fps_hz"],
                "shape_hw": view["shape_hw"],
                "reference_frame_name": view["reference_frame_name"],
                "reference_frame_index": view["reference_frame_index"],
                "modes_dtype": view["modes_dtype"],
                "modes_shape": view["modes_shape"],
                "modes_file_sha256": view["modes_file_sha256"],
            }
            for view in manifest["views"]
        ],
    }


def _validate_mode_records(manifest: Mapping[str, Any]) -> np.ndarray:
    """Validate the ordered greedy slots and return their frequencies."""

    modes = manifest.get("modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError("Complex mode manifest must contain ordered modes")
    frequencies = np.empty(len(modes), dtype=np.float64)
    candidates: list[int] = []
    for expected_slot, mode in enumerate(modes):
        if not isinstance(mode, dict) or mode.get("mode_slot") != expected_slot:
            raise ValueError("Complex mode slots must be contiguous and ordered")
        candidate = mode.get("candidate_index")
        if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 0:
            raise ValueError("Complex mode candidate indices must be non-negative integers")
        frequency = float(mode.get("frequency_hz", np.nan))
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("Complex mode frequencies must be finite and positive")
        candidates.append(candidate)
        frequencies[expected_slot] = frequency
    if len(set(candidates)) != len(candidates):
        raise ValueError("Complex mode candidate indices must be unique")
    return frequencies


def _validate_finite_modes(modes: np.ndarray, label: str) -> None:
    """Scan one memmap in bounded blocks for NaN or Inf values."""

    width = modes.shape[2]
    for lower in range(0, width, DENSE_DFT_BLOCK_WIDTH):
        block = modes[:, :, lower : lower + DENSE_DFT_BLOCK_WIDTH, :]
        if not np.isfinite(block).all():
            raise ValueError(f"Dense complex modes for {label!r} contain NaN or Inf")


def load_complex_2d_modes(path: str | Path) -> Complex2DModesArtifact:
    """Load a dense-mode directory and retain each view as an mmap array."""

    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError(f"Complex mode artifact is not a directory: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Complex mode artifact is missing manifest.json: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != MODES_FORMAT:
        raise ValueError(f"Unsupported complex mode artifact: {manifest_path}")
    if manifest.get("version") != MODES_VERSION:
        raise ValueError("Unsupported complex mode artifact version")
    if manifest.get("transform") != TRANSFORM_CONVENTION:
        raise ValueError("Complex mode transform convention is unsupported")
    for name in ("frequency_selection_identity", "topology_identity"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"Complex mode manifest {name} is invalid")
    frequencies = _validate_mode_records(manifest)
    views = manifest.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("Complex mode manifest must contain ordered views")
    labels: list[str] = []
    loaded: list[np.ndarray] = []
    for expected_index, view in enumerate(views):
        if not isinstance(view, dict) or view.get("index") != expected_index:
            raise ValueError("Complex mode view indices must be contiguous and ordered")
        label = view.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("Complex mode view label is invalid")
        labels.append(label)
        filename = f"view_{expected_index:03d}.npy"
        if view.get("modes_file") != filename:
            raise ValueError(f"Complex mode filename for {label!r} is invalid")
        modes_path = root / filename
        if not modes_path.is_file():
            raise FileNotFoundError(f"Complex mode array is missing: {modes_path}")
        if view.get("modes_file_sha256") != _sha256_file(modes_path):
            raise ValueError(f"Complex mode SHA-256 for {label!r} does not match")
        modes = np.load(modes_path, mmap_mode="r", allow_pickle=False)
        if not isinstance(modes, np.ndarray) or modes.dtype != MODES_DTYPE:
            raise ValueError(f"Complex modes for {label!r} must be complex64")
        expected_shape = tuple(view.get("modes_shape", ()))
        if (
            expected_shape != modes.shape
            or modes.shape[0] != len(frequencies)
            or modes.ndim != 4
            or modes.shape[-1] != 2
            or view.get("modes_dtype") != MODES_DTYPE.name
            or view.get("shape_hw") != [modes.shape[1], modes.shape[2]]
        ):
            raise ValueError(f"Complex mode shape metadata for {label!r} is invalid")
        frame_count = view.get("frame_count")
        fps_hz = float(view.get("fps_hz", np.nan))
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count < 3
            or not math.isfinite(fps_hz)
            or fps_hz <= 0.0
            or float(frequencies[-1]) > 0.5 * fps_hz + 1e-12
        ):
            raise ValueError(f"Complex mode temporal metadata for {label!r} is invalid")
        _validate_finite_modes(modes, label)
        loaded.append(modes)
    if len(set(labels)) != len(labels):
        raise ValueError("Complex mode view labels must be unique")
    expected_identity = hashlib.sha256(
        _canonical_json(_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("complex_2d_modes_identity") != expected_identity:
        raise ValueError("Complex 2D mode identity does not match its contents")
    return Complex2DModesArtifact(root, manifest, tuple(loaded))


def dense_cache_contract(flow_identity: str, frequencies_hz: np.ndarray) -> dict[str, Any]:
    return {"implementation": "dense_exact_dft_v1", "flow_identity": flow_identity,
            "frequencies_hz": np.asarray(frequencies_hz, dtype=np.float64).tolist(),
            "transform": TRANSFORM_CONVENTION, "dtype": MODES_DTYPE.name}


def _write_dense_exact_dft(
    artifact: FlowAnalysisArtifact,
    frequencies_hz: np.ndarray,
    destination: Path,
    cache_dir: str | Path = DEFAULT_CACHE,
) -> None:
    """Write one view's full-image selected-frequency DFT directly to a memmap."""

    flow = artifact.arrays.flow
    contract = dense_cache_contract(flow_artifact_identity(artifact), frequencies_hz)
    cache_root = Path(cache_dir) / "dense_dft"
    previous = load_entry(cache_root, contract)
    if previous is not None:
        np.save(destination, previous["modes"], allow_pickle=False)
        return
    frame_count, height, width, _ = flow.shape
    basis, window = exact_dft_basis(
        frame_count,
        float(artifact.manifest["fps_hz"]),
        frequencies_hz,
    )
    modes = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=MODES_DTYPE,
        shape=(len(frequencies_hz), height, width, 2),
    )
    try:
        progress = Progress(f"dense exact DFT {destination.name}", height * width, unit="pixels")
        completed = 0
        for rows, columns in spatial_blocks(flow.shape, DENSE_DFT_BLOCK_WIDTH):
            block_height, block_width = rows.stop - rows.start, columns.stop - columns.start
            for component in range(2):
                values = np.asarray(
                    flow[:, rows, columns, component], dtype=np.float32
                )
                if not np.isfinite(values).all():
                    raise ValueError("Flow block contains NaN or Inf")
                values = values - values.mean(
                    axis=0, keepdims=True, dtype=np.float32
                )
                values *= window[:, None, None]
                transformed = basis @ values.reshape(frame_count, -1)
                if not np.isfinite(transformed).all():
                    raise ValueError("Dense exact-DFT block contains NaN or Inf")
                modes[:, rows, columns, component] = transformed.reshape(
                    len(frequencies_hz), block_height, block_width
                )
            completed += block_height * block_width
            progress.update(completed)
        modes.flush()
    finally:
        del modes
    put_entry(cache_root, contract, {"modes": np.load(destination, mmap_mode="r", allow_pickle=False)})


def build_complex_2d_modes_artifact(
    *,
    selection_dir: str | Path,
    views: Sequence[ComplexModeViewInput],
    output_dir: str | Path,
    command: Sequence[str] = (),
) -> Complex2DModesArtifact:
    """Validate selection inputs and publish every selected dense complex field."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Complex mode output already exists: {destination}")
    selection = load_frequency_selection(selection_dir)
    expected_labels = [view["label"] for view in selection.manifest["views"]]
    labels = [view.label.strip() for view in views]
    if labels != expected_labels:
        raise ValueError(
            f"Mode --view labels must follow selection order {expected_labels}, got {labels}"
        )
    frequencies = np.asarray(
        selection.arrays.selected_frequencies_hz, dtype=np.float64
    )
    candidates = np.asarray(
        selection.arrays.selected_candidate_index, dtype=np.int64
    )
    mode_records = [
        {
            "mode_slot": slot,
            "candidate_index": int(candidate),
            "frequency_hz": float(frequency),
        }
        for slot, (candidate, frequency) in enumerate(zip(candidates, frequencies))
    ]

    validated_views: list[tuple[str, FlowAnalysisArtifact, str]] = []
    for specification, selection_view in zip(views, selection.manifest["views"]):
        artifact = load_flow_analysis_artifact(specification.flow_artifact)
        identity = flow_artifact_identity(artifact)
        if identity != selection_view["flow_identity"]:
            raise ValueError(
                f"Flow artifact for {specification.label!r} does not match selection"
            )
        fft = artifact.manifest["parameters"].get("fft", {})
        if fft.get("detrend") != "temporal-mean" or fft.get("window") != "hann-symmetric":
            raise ValueError(
                f"Flow FFT convention for {specification.label!r} is unsupported"
            )
        exact_dft_basis(
            artifact.arrays.flow.shape[0],
            float(artifact.manifest["fps_hz"]),
            frequencies,
        )
        validated_views.append((specification.label, artifact, identity))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        view_records: list[dict[str, Any]] = []
        for index, (label, artifact, identity) in enumerate(validated_views):
            frame_count, height, width, _ = artifact.arrays.flow.shape
            filename = f"view_{index:03d}.npy"
            modes_path = temporary / filename
            print(
                f"Dense exact DFT {label}: modes={len(frequencies)}, "
                f"frames={frame_count}, shape={height}x{width}",
                flush=True,
            )
            _write_dense_exact_dft(artifact, frequencies, modes_path)
            view_records.append(
                {
                    "index": index,
                    "label": label,
                    "flow_artifact": str(artifact.path.resolve()),
                    "flow_identity": identity,
                    "frame_count": frame_count,
                    "fps_hz": float(artifact.manifest["fps_hz"]),
                    "shape_hw": [height, width],
                    "reference_frame_name": artifact.manifest["reference_frame_name"],
                    "reference_frame_index": int(
                        artifact.manifest["reference_frame_index"]
                    ),
                    "modes_file": filename,
                    "modes_dtype": MODES_DTYPE.name,
                    "modes_shape": [len(frequencies), height, width, 2],
                    "modes_file_sha256": _sha256_file(modes_path),
                }
            )
        manifest = {
            "format": MODES_FORMAT,
            "version": MODES_VERSION,
            "producer": {
                "project_version": __version__,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": list(command),
            },
            "frequency_selection": str(selection.path),
            "frequency_selection_identity": selection.manifest[
                "frequency_selection_identity"
            ],
            "topology_identity": selection.manifest["topology_identity"],
            "modes": mode_records,
            "transform": dict(TRANSFORM_CONVENTION),
            "views": view_records,
        }
        manifest["complex_2d_modes_identity"] = hashlib.sha256(
            _canonical_json(_identity_payload(manifest))
        ).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        validated = load_complex_2d_modes(temporary)
        del validated
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Complex mode output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_complex_2d_modes(destination)


__all__ = [
    "Complex2DModesArtifact",
    "ComplexModeViewInput",
    "build_complex_2d_modes_artifact",
    "load_complex_2d_modes",
]
