"""Complex 2D measurements sampled at observation-topology pixels."""

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
from modal_gaussians.modes import Complex2DModesArtifact, load_complex_2d_modes
from modal_gaussians.topology import (
    ObservationTopologyArtifact,
    load_observation_topology,
)


MEASUREMENTS_FORMAT = "modal_gaussians.gaussian_measurements"
MEASUREMENTS_VERSION = 1
MEASUREMENTS_FILENAME = "measurements.npy"
MEASUREMENTS_DTYPE = np.dtype(np.complex64)
SAMPLE_BLOCK_SIZE = 65_536

SAMPLING_CONVENTION = {
    "source": "dense_complex_2d_modes",
    "lookup": "exact_integer_pixel",
    "sample_order": "observation_topology_sample_order",
    "pixel_coordinate_order": "xy",
    "displacement_component_order": "uv",
    "interpolation": "none",
    "mask_application": "none",
    "normalization": "none",
    "amplitude_clamp": "none",
    "contributor_expansion": "none",
}


@dataclass(frozen=True)
class GaussianMeasurementsArtifact:
    """Represent one validated topology-aligned complex measurement bank."""

    path: Path
    manifest: dict[str, Any]
    measurements: np.ndarray


def _canonical_json(value: Any) -> bytes:
    """Encode path-independent identity fields deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    """Hash one potentially large array without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select scientific fields that define one measurement artifact."""

    return {
        "format": MEASUREMENTS_FORMAT,
        "version": MEASUREMENTS_VERSION,
        "topology_identity": manifest["topology_identity"],
        "complex_2d_modes_identity": manifest["complex_2d_modes_identity"],
        "modes": manifest["modes"],
        "views": manifest["views"],
        "sampling": manifest["sampling"],
        "measurements_dtype": manifest["measurements_dtype"],
        "measurements_shape": manifest["measurements_shape"],
        "measurements_file_sha256": manifest["measurements_file_sha256"],
    }


def _validate_modes(modes: Any) -> None:
    """Validate ordered mode slots, candidates, and frequencies."""

    if not isinstance(modes, list) or not modes:
        raise ValueError("Measurement manifest must contain ordered modes")
    candidates: list[int] = []
    for expected_slot, mode in enumerate(modes):
        if not isinstance(mode, dict) or mode.get("mode_slot") != expected_slot:
            raise ValueError("Measurement mode slots must be contiguous and ordered")
        candidate = mode.get("candidate_index")
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate < 0
        ):
            raise ValueError(
                "Measurement candidate indices must be non-negative integers"
            )
        frequency = float(mode.get("frequency_hz", np.nan))
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("Measurement frequencies must be finite and positive")
        candidates.append(candidate)
    if len(set(candidates)) != len(candidates):
        raise ValueError("Measurement candidate indices must be unique")


def _validate_views(views: Any, sample_count: int) -> None:
    """Validate ordered view metadata and its partition of topology samples."""

    if not isinstance(views, list) or not views:
        raise ValueError("Measurement manifest must contain ordered views")
    labels: list[str] = []
    total = 0
    for expected_index, view in enumerate(views):
        if not isinstance(view, dict) or view.get("index") != expected_index:
            raise ValueError("Measurement view indices must be contiguous and ordered")
        label = view.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("Measurement view label is invalid")
        labels.append(label)
        flow_identity = view.get("flow_identity")
        if not isinstance(flow_identity, str) or not flow_identity:
            raise ValueError(f"Measurement flow identity for {label!r} is invalid")
        shape = view.get("shape_hw")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in shape
            )
            or any(value <= 2 for value in shape)
        ):
            raise ValueError(f"Measurement view shape for {label!r} is invalid")
        count = view.get("sample_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Measurement sample count for {label!r} is invalid")
        total += count
    if len(set(labels)) != len(labels):
        raise ValueError("Measurement view labels must be unique")
    if total != sample_count:
        raise ValueError("Measurement per-view sample counts do not match array shape")


def _validate_finite(measurements: np.ndarray) -> None:
    """Scan a measurement memmap in bounded sample blocks for NaN or Inf."""

    sample_count = measurements.shape[1]
    for lower in range(0, sample_count, SAMPLE_BLOCK_SIZE):
        if not np.isfinite(
            measurements[:, lower : lower + SAMPLE_BLOCK_SIZE, :]
        ).all():
            raise ValueError("Gaussian measurements contain NaN or Inf")


def load_gaussian_measurements(path: str | Path) -> GaussianMeasurementsArtifact:
    """Load and fully validate a topology-aligned measurement bank."""

    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError(f"Measurement artifact is not a directory: {root}")
    manifest_path = root / "manifest.json"
    measurements_path = root / MEASUREMENTS_FILENAME
    if not manifest_path.is_file() or not measurements_path.is_file():
        raise FileNotFoundError(f"Incomplete measurement artifact: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != MEASUREMENTS_FORMAT:
        raise ValueError(f"Unsupported measurement artifact: {manifest_path}")
    if manifest.get("version") != MEASUREMENTS_VERSION:
        raise ValueError("Unsupported measurement artifact version")
    if manifest.get("sampling") != SAMPLING_CONVENTION:
        raise ValueError("Measurement sampling convention is unsupported")
    for name in ("topology_identity", "complex_2d_modes_identity"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"Measurement manifest {name} is invalid")
    _validate_modes(manifest.get("modes"))
    if manifest.get("measurements_file") != MEASUREMENTS_FILENAME:
        raise ValueError("Measurement array filename is invalid")
    if manifest.get("measurements_file_sha256") != _sha256_file(measurements_path):
        raise ValueError("Measurement array SHA-256 does not match manifest")
    measurements = np.load(measurements_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(measurements, np.ndarray) or measurements.dtype != MEASUREMENTS_DTYPE:
        raise ValueError("Gaussian measurements must be complex64")
    shape_metadata = manifest.get("measurements_shape")
    if (
        not isinstance(shape_metadata, list)
        or len(shape_metadata) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in shape_metadata
        )
    ):
        raise ValueError("Measurement array shape metadata is invalid")
    expected_shape = tuple(shape_metadata)
    if (
        measurements.ndim != 3
        or measurements.shape[-1] != 2
        or measurements.shape[0] != len(manifest["modes"])
        or measurements.shape[1] <= 0
        or measurements.shape != expected_shape
        or manifest.get("measurements_dtype") != MEASUREMENTS_DTYPE.name
    ):
        raise ValueError("Measurement array shape metadata is invalid")
    _validate_views(manifest.get("views"), measurements.shape[1])
    _validate_finite(measurements)
    expected_identity = hashlib.sha256(
        _canonical_json(_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("gaussian_measurements_identity") != expected_identity:
        raise ValueError("Gaussian measurement identity does not match its contents")
    return GaussianMeasurementsArtifact(root, manifest, measurements)


def _validate_source_alignment(
    topology: ObservationTopologyArtifact,
    dense_modes: Complex2DModesArtifact,
) -> list[dict[str, Any]]:
    """Require exact topology identity, view order, shapes, and flow identities."""

    topology_identity = topology.manifest["topology_identity"]
    if dense_modes.manifest["topology_identity"] != topology_identity:
        raise ValueError("Dense modes were not produced for this observation topology")
    topology_views = topology.manifest["views"]
    mode_views = dense_modes.manifest["views"]
    if len(topology_views) != len(mode_views):
        raise ValueError("Topology and dense modes have different view counts")
    records: list[dict[str, Any]] = []
    sample_views = topology.arrays.sample_view_index
    for index, (topology_view, mode_view) in enumerate(
        zip(topology_views, mode_views)
    ):
        label = topology_view["label"]
        if mode_view["index"] != index or mode_view["label"] != label:
            raise ValueError("Topology and dense-mode view order does not match")
        shape = [int(value) for value in topology.arrays.view_shapes_hw[index]]
        if topology_view["shape_hw"] != shape or mode_view["shape_hw"] != shape:
            raise ValueError(f"View shape for {label!r} does not match")
        if topology_view["flow_identity"] != mode_view["flow_identity"]:
            raise ValueError(f"Flow identity for {label!r} does not match")
        sample_count = int(np.count_nonzero(sample_views == index))
        if sample_count <= 0:
            raise ValueError(f"Topology view {label!r} contains no samples")
        records.append(
            {
                "index": index,
                "label": label,
                "flow_identity": topology_view["flow_identity"],
                "shape_hw": shape,
                "sample_count": sample_count,
            }
        )
    return records


def _write_measurements(
    topology: ObservationTopologyArtifact,
    dense_modes: Complex2DModesArtifact,
    destination: Path,
) -> None:
    """Sample every dense view at topology pixels while preserving sample order."""

    sample_views = topology.arrays.sample_view_index
    sample_pixels = topology.arrays.sample_pixels_xy
    mode_count = len(dense_modes.manifest["modes"])
    sample_count = len(sample_views)
    output = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=MEASUREMENTS_DTYPE,
        shape=(mode_count, sample_count, 2),
    )
    try:
        for view_index, dense_view in enumerate(dense_modes.view_modes):
            rows = np.flatnonzero(sample_views == view_index)
            for lower in range(0, len(rows), SAMPLE_BLOCK_SIZE):
                block_rows = rows[lower : lower + SAMPLE_BLOCK_SIZE]
                pixels = sample_pixels[block_rows]
                output[:, block_rows, :] = dense_view[
                    :, pixels[:, 1], pixels[:, 0], :
                ]
        output.flush()
    finally:
        del output


def build_gaussian_measurements_artifact(
    *,
    topology_dir: str | Path,
    modes_dir: str | Path,
    output_dir: str | Path,
    command: Sequence[str] = (),
) -> GaussianMeasurementsArtifact:
    """Sample dense complex fields and atomically publish the measurement bank."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Measurement output already exists: {destination}")
    topology = load_observation_topology(topology_dir)
    dense_modes = load_complex_2d_modes(modes_dir)
    view_records = _validate_source_alignment(topology, dense_modes)
    mode_records = [
        {
            "mode_slot": int(mode["mode_slot"]),
            "candidate_index": int(mode["candidate_index"]),
            "frequency_hz": float(mode["frequency_hz"]),
        }
        for mode in dense_modes.manifest["modes"]
    ]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        measurements_path = temporary / MEASUREMENTS_FILENAME
        _write_measurements(topology, dense_modes, measurements_path)
        shape = [
            len(mode_records),
            len(topology.arrays.sample_view_index),
            2,
        ]
        manifest = {
            "format": MEASUREMENTS_FORMAT,
            "version": MEASUREMENTS_VERSION,
            "producer": {
                "project_version": __version__,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": list(command),
            },
            "topology": str(topology.path),
            "topology_identity": topology.manifest["topology_identity"],
            "complex_2d_modes": str(dense_modes.path),
            "complex_2d_modes_identity": dense_modes.manifest[
                "complex_2d_modes_identity"
            ],
            "modes": mode_records,
            "views": view_records,
            "sampling": dict(SAMPLING_CONVENTION),
            "measurements_file": MEASUREMENTS_FILENAME,
            "measurements_dtype": MEASUREMENTS_DTYPE.name,
            "measurements_shape": shape,
            "measurements_file_sha256": _sha256_file(measurements_path),
        }
        manifest["gaussian_measurements_identity"] = hashlib.sha256(
            _canonical_json(_identity_payload(manifest))
        ).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        validated = load_gaussian_measurements(temporary)
        del validated
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Measurement output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_gaussian_measurements(destination)


__all__ = [
    "GaussianMeasurementsArtifact",
    "build_gaussian_measurements_artifact",
    "load_gaussian_measurements",
]
