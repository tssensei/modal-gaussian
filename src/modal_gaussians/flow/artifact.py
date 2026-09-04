from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping

import numpy as np

from modal_gaussians import __version__
from modal_gaussians.flow.storage import (
    DenseArray, copy_array, create_array, open_array, storage_sha256, validate_finite,
)


ARTIFACT_FORMAT = "modal_gaussians.flow_analysis"
ARTIFACT_VERSION = 7

ARRAY_FILES = {
    "flow": "flow.zarr",
    "mask_union": "mask_union.npy",
    "spectrum": "spectrum.zarr",
}
LEGACY_ARRAY_FILES = {**ARRAY_FILES, "flow": "flow.npy", "spectrum": "spectrum.npy"}

ARRAY_DTYPES = {
    "flow": np.dtype(np.float32),
    "mask_union": np.dtype(bool),
    "spectrum": np.dtype(np.complex64),
}


@dataclass(frozen=True)
class FlowAnalysisArrays:
    flow: DenseArray
    mask_union: np.ndarray
    spectrum: DenseArray

    def as_dict(self) -> dict[str, DenseArray]:
        """Expose lazy arrays without materializing their full contents."""
        return {
            name: getattr(self, name)
            for name in ARRAY_FILES
        }


@dataclass(frozen=True)
class FlowAnalysisArtifact:
    path: Path
    manifest: dict[str, Any]
    arrays: FlowAnalysisArrays


def flow_artifact_identity(artifact: FlowAnalysisArtifact) -> str:
    """Bind consumers to exact flow, mask, spectrum, and analysis settings."""

    manifest = artifact.manifest
    scientific_manifest = {
        "format": manifest["format"],
        "version": manifest["version"],
        "parameters": manifest["parameters"],
        "frame_names": manifest["frame_names"],
        "fps_hz": manifest["fps_hz"],
        "reference_frame_name": manifest["reference_frame_name"],
        "reference_frame_index": manifest["reference_frame_index"],
    }
    encoded = json.dumps(
        scientific_manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded)
    for name in ARRAY_FILES:
        filename = str(manifest["arrays"][name]["file"])
        digest.update(name.encode("utf-8"))
        digest.update(storage_sha256(artifact.path / filename).encode("ascii"))
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Flow artifact is missing manifest.json: {path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Flow artifact manifest root must be an object")
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"Unsupported flow artifact format: {manifest.get('format')!r}")
    if manifest.get("version") not in (6, ARTIFACT_VERSION):
        raise ValueError(f"Unsupported flow artifact version: {manifest.get('version')!r}")
    inputs = manifest.get("inputs")
    parameters = manifest.get("parameters")
    if not isinstance(inputs, dict) or not isinstance(parameters, dict):
        raise ValueError("Flow artifact inputs and parameters must be objects")
    sequence_input = inputs.get("sequence")
    if not isinstance(sequence_input, dict):
        raise ValueError("Flow artifact sequence input record is invalid")
    frame_names = manifest.get("frame_names")
    if (
        not isinstance(frame_names, list)
        or len(frame_names) < 3
        or not all(isinstance(value, str) and value for value in frame_names)
        or len(set(frame_names)) != len(frame_names)
    ):
        raise ValueError("Flow artifact frame_names must be unique strings")
    reference_name = manifest.get("reference_frame_name")
    reference_index = manifest.get("reference_frame_index")
    if (
        reference_name not in frame_names
        or not isinstance(reference_index, int)
        or reference_index < 0
        or reference_index >= len(frame_names)
        or frame_names[reference_index] != reference_name
    ):
        raise ValueError("Flow artifact reference frame name/index is inconsistent")
    fps = float(manifest.get("fps_hz", np.nan))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("Flow artifact FPS must be finite and positive")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or set(arrays) != set(ARRAY_FILES):
        raise ValueError("Flow artifact array inventory is incomplete")
    files = LEGACY_ARRAY_FILES if manifest["version"] == 6 else ARRAY_FILES
    for name, filename in files.items():
        record = arrays[name]
        if not isinstance(record, dict):
            raise ValueError(f"Flow artifact arrays.{name} must be an object")
        if record.get("file") != filename:
            raise ValueError(f"Flow artifact arrays.{name}.file is invalid")
        if record.get("dtype") != ARRAY_DTYPES[name].name:
            raise ValueError(f"Flow artifact arrays.{name}.dtype is invalid")
        if not isinstance(record.get("shape"), list):
            raise ValueError(f"Flow artifact arrays.{name}.shape is invalid")
        if manifest["version"] == ARTIFACT_VERSION:
            expected_storage = "npy" if name == "mask_union" else "zarr_v3_zstd"
            if record.get("storage") != expected_storage:
                raise ValueError(f"Flow artifact arrays.{name}.storage is invalid")
            checksum = record.get("sha256")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError(f"Flow artifact arrays.{name}.sha256 is invalid")
    stabilization = parameters.get("stabilization")
    if not isinstance(stabilization, dict) or not isinstance(
        stabilization.get("method"), str
    ):
        raise ValueError("Flow artifact stabilization parameters are invalid")
    stabilized = manifest.get("stabilized_sequence")
    if stabilization["method"] == "none":
        if stabilized is not None:
            raise ValueError("Unstabilized flow artifact contains a stabilized sequence")
    elif (
        not isinstance(stabilized, dict)
        or stabilized.get("path") != "stabilized_sequence"
    ):
        raise ValueError("Stabilized flow artifact has an invalid sequence record")


def _validate_arrays(
    arrays: Mapping[str, DenseArray], manifest: Mapping[str, Any]
) -> None:
    for name in ARRAY_FILES:
        array = arrays[name]
        if array.dtype != ARRAY_DTYPES[name]:
            raise ValueError(
                f"Flow artifact {name} dtype {array.dtype} != {ARRAY_DTYPES[name]}"
            )
        if list(array.shape) != manifest["arrays"][name]["shape"]:
            raise ValueError(f"Flow artifact {name} shape does not match manifest")

    flow = arrays["flow"]
    mask_union = np.asarray(arrays["mask_union"])
    spectrum = arrays["spectrum"]
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError("Flow artifact flow must be [T,H,W,2]")
    frame_count, height, width, _ = flow.shape
    if frame_count != len(manifest["frame_names"]) or height < 1 or width < 1:
        raise ValueError("Flow artifact frame count or image dimensions are invalid")
    frequency_count = frame_count // 2 + 1
    if mask_union.shape != (height, width) or not np.any(mask_union):
        raise ValueError("Flow artifact mask_union is invalid or empty")
    if spectrum.shape != (frequency_count, height, width, 2):
        raise ValueError("Flow artifact spectrum shape is invalid")
    for name in ("flow", "spectrum"):
        validate_finite(arrays[name], f"Flow artifact {name}")
    reference_index = int(manifest["reference_frame_index"])
    if not np.array_equal(flow[reference_index], np.zeros_like(flow[reference_index])):
        raise ValueError("Flow artifact reference-to-reference flow is not exactly zero")


def _validate_stabilized_sequence(
    root: Path,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, DenseArray],
) -> None:
    record = manifest.get("stabilized_sequence")
    if record is None:
        return
    sequence_root = root / "stabilized_sequence"
    sequence_manifest_path = sequence_root / "manifest.json"
    if not sequence_manifest_path.is_file():
        raise FileNotFoundError("Stabilized sequence is missing manifest.json")
    with sequence_manifest_path.open("r", encoding="utf-8") as stream:
        stabilized = json.load(stream)
    if not isinstance(stabilized, dict):
        raise ValueError("Stabilized sequence manifest root must be an object")
    if (
        stabilized.get("format")
        != "modal_gaussians.stabilized_image_mask_sequence"
        or stabilized.get("version") != 2
    ):
        raise ValueError("Unsupported stabilized sequence format or version")
    if stabilized.get("settings") != manifest["parameters"]["stabilization"].get(
        "settings"
    ):
        raise ValueError("Stabilized sequence settings do not match flow parameters")
    flow = arrays["flow"]
    frame_count, height, width, _ = flow.shape
    if (
        stabilized.get("fps_hz") != manifest["fps_hz"]
        or stabilized.get("reference_frame") != manifest["reference_frame_name"]
        or stabilized.get("height") != height
        or stabilized.get("width") != width
    ):
        raise ValueError("Stabilized sequence dimensions, FPS, or reference are invalid")
    frames = stabilized.get("frames")
    if (
        not isinstance(frames, list)
        or not all(isinstance(name, str) for name in frames)
        or frames != manifest["frame_names"]
    ):
        raise ValueError("Stabilized sequence frame inventory is invalid")
    for name in frames:
        for label, relative_path in (
            ("image", Path("images") / f"{name}.png"),
            ("mask", Path("masks") / f"{name}.png"),
        ):
            path = sequence_root / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Stabilized sequence is missing {label}: {path}")
    homographies = stabilized.get("homographies")
    if (
        not isinstance(homographies, dict)
        or homographies.get("file") != "homographies_frame_to_reference.npy"
        or homographies.get("dtype") != "float64"
        or homographies.get("shape") != [frame_count, 3, 3]
    ):
        raise ValueError("Stabilized sequence homography record is invalid")
    homography_path = sequence_root / homographies["file"]
    if not homography_path.is_file():
        raise FileNotFoundError("Stabilized sequence is missing homographies")
    homography_values = np.load(homography_path, mmap_mode="r", allow_pickle=False)
    if (
        homography_values.dtype != np.float64
        or homography_values.shape != (frame_count, 3, 3)
        or not np.isfinite(homography_values).all()
    ):
        raise ValueError("Stabilized sequence homography array is invalid")
    statistics = stabilized.get("statistics")
    if not isinstance(statistics, dict) or statistics.get("file") != "statistics.json":
        raise ValueError("Stabilized sequence statistics record is invalid")
    statistics_path = sequence_root / statistics["file"]
    if not statistics_path.is_file():
        raise FileNotFoundError("Stabilized sequence is missing statistics")
    with statistics_path.open("r", encoding="utf-8") as stream:
        if not isinstance(json.load(stream), dict):
            raise ValueError("Stabilized sequence statistics root must be an object")


def load_flow_analysis_artifact(path: str | Path) -> FlowAnalysisArtifact:
    root = Path(path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Flow artifact directory does not exist: {root}")
    manifest = _load_manifest(root)
    _validate_manifest(manifest)
    loaded: dict[str, Any] = {}
    files = LEGACY_ARRAY_FILES if manifest["version"] == 6 else ARRAY_FILES
    for name, filename in files.items():
        array_path = root / filename
        if not array_path.exists():
            raise FileNotFoundError(f"Flow artifact is missing {filename}")
        if manifest["version"] == ARTIFACT_VERSION:
            if storage_sha256(array_path) != manifest["arrays"][name]["sha256"]:
                raise ValueError(f"Flow artifact {filename} SHA-256 differs")
        if name != "mask_union" and manifest["version"] == ARTIFACT_VERSION:
            array = open_array(array_path)
            record = manifest["arrays"][name]
            if list(array.chunks) != record.get("chunks") or list(array.shards or ()) != record.get("shards"):
                raise ValueError(f"Flow artifact {filename} chunk layout differs")
        else:
            array = np.load(
                array_path, mmap_mode=None if name == "mask_union" else "r",
                allow_pickle=False,
            )
            if not isinstance(array, np.ndarray):
                raise ValueError(f"Flow artifact {filename} must contain one array")
        loaded[name] = array
    _validate_arrays(loaded, manifest)
    _validate_stabilized_sequence(root, manifest, loaded)
    return FlowAnalysisArtifact(
        path=root,
        manifest=manifest,
        arrays=FlowAnalysisArrays(**loaded),
    )


def publish_flow_analysis_artifact(
    target: str | Path,
    *,
    arrays: FlowAnalysisArrays | Callable[[Path], FlowAnalysisArrays],
    inputs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    frame_names: list[str],
    reference_frame_name: str,
    reference_frame_index: int,
    fps_hz: float,
    command: list[str],
    write_stabilized: Callable[[Path], Mapping[str, Any]] | None = None,
) -> FlowAnalysisArtifact:
    destination = Path(target).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Flow artifact target already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        generated = callable(arrays)
        array_values = (arrays(temporary) if callable(arrays) else arrays).as_dict()
        array_records: dict[str, Any] = {}
        for name, filename in ARRAY_FILES.items():
            path = temporary / filename
            value = array_values[name]
            if np.dtype(value.dtype) != ARRAY_DTYPES[name]:
                raise ValueError(f"Flow artifact {name} has an invalid dtype")
            if name == "mask_union":
                np.save(path, np.asarray(value), allow_pickle=False)
                layout = {"storage": "npy"}
            else:
                if not generated:
                    stored = create_array(path, value.shape, value.dtype)
                    copy_array(value, stored)
                stored = open_array(path)
                layout = {
                    "storage": "zarr_v3_zstd",
                    "chunks": list(stored.chunks),
                    "shards": list(stored.shards or ()),
                }
            array_records[name] = {
                "file": filename,
                "dtype": ARRAY_DTYPES[name].name,
                "shape": list(value.shape),
                "sha256": storage_sha256(path),
                **layout,
            }
        stabilized_record = None
        if write_stabilized is not None:
            stabilized_directory = temporary / "stabilized_sequence"
            stabilized_directory.mkdir()
            write_stabilized(stabilized_directory)
            stabilized_record = {"path": "stabilized_sequence"}
        manifest = {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "producer": {
                "project_version": __version__,
                "command": list(command),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
            "inputs": dict(inputs),
            "parameters": dict(parameters),
            "frame_names": list(frame_names),
            "fps_hz": float(fps_hz),
            "reference_frame_name": reference_frame_name,
            "reference_frame_index": int(reference_frame_index),
            "arrays": array_records,
            "stabilized_sequence": stabilized_record,
        }
        with (temporary / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        load_flow_analysis_artifact(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Flow artifact target already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_flow_analysis_artifact(destination)
