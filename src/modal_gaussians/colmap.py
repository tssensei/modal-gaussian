"""Sample frames, run joint COLMAP, and export cameras and a sparse point cloud."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class ReferenceInput:
    """One fixed-view reference RGB image and its foreground mask."""

    label: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class _Camera:
    """The intrinsics stored in one COLMAP cameras.txt record."""

    model: str
    width: int
    height: int
    parameters: tuple[float, ...]


@dataclass(frozen=True)
class _RegisteredImage:
    """The extrinsics stored in one COLMAP images.txt record."""

    image_id: int
    camera_id: int
    qvec_wxyz: tuple[float, float, float, float]
    tvec: tuple[float, float, float]


def _resolve_executable(command: str) -> str:
    """Resolve an executable name or an explicit filesystem path."""

    candidate = Path(command).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(
            f"COLMAP executable {command!r} was not found on PATH"
        )
    return resolved


def _natural_name_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    """Sort numeric filename fragments by value instead of lexicographically."""

    return tuple(
        (0, int(fragment)) if fragment.isdigit() else (1, fragment.casefold())
        for fragment in re.split(r"([0-9]+)", path.name)
    )


def _ordered_pngs(directory: Path, label: str) -> list[Path]:
    """Return one flat PNG sequence in filename order."""

    if not directory.is_dir():
        raise FileNotFoundError(directory)
    entries = sorted(directory.iterdir(), key=_natural_name_key)
    invalid = [path.name for path in entries if not path.is_file()]
    invalid.extend(path.name for path in entries if path.suffix.lower() != ".png")
    if invalid:
        raise ValueError(f"{label} must contain only PNG files: {invalid[:5]}")
    if not entries:
        raise ValueError(f"{label} contains no PNG frames: {directory}")
    return entries


def _validate_inputs(
    frames_dir: Path,
    frame_masks_dir: Path,
    references: Sequence[ReferenceInput],
    sample_stride: int,
    output_dir: Path,
) -> tuple[list[Path], dict[str, Path], tuple[ReferenceInput, ...]]:
    """Validate paths and select the regularly sampled sweep frames."""

    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    frames = _ordered_pngs(frames_dir, "frames_dir")
    masks = _ordered_pngs(frame_masks_dir, "frame_masks_dir")
    masks_by_name = {path.name: path for path in masks}
    frame_names = {path.name for path in frames}
    if frame_names != set(masks_by_name):
        missing = sorted(frame_names - set(masks_by_name))
        extra = sorted(set(masks_by_name) - frame_names)
        raise ValueError(
            "Frame/mask filenames differ: "
            f"missing_masks={missing[:5]}, extra_masks={extra[:5]}"
        )
    resolved_references = tuple(
        ReferenceInput(
            label=reference.label,
            image_path=reference.image_path.expanduser().resolve(strict=True),
            mask_path=reference.mask_path.expanduser().resolve(strict=True),
        )
        for reference in references
    )
    if not resolved_references:
        raise ValueError("At least one reference image is required")
    labels = [reference.label for reference in resolved_references]
    if any(not LABEL_PATTERN.fullmatch(label) for label in labels):
        raise ValueError(
            "Reference labels must contain only letters, digits, underscores, "
            "or hyphens"
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"Reference labels must be unique: {labels}")
    for reference in resolved_references:
        if not reference.image_path.is_file():
            raise FileNotFoundError(reference.image_path)
        if not reference.mask_path.is_file():
            raise FileNotFoundError(reference.mask_path)
        if reference.image_path.suffix.lower() != ".png":
            raise ValueError(f"Reference RGB must be PNG: {reference.image_path}")
        if reference.mask_path.suffix.lower() != ".png":
            raise ValueError(f"Reference mask must be PNG: {reference.mask_path}")
    return frames[::sample_stride], masks_by_name, resolved_references


def _copy_input(
    image_source: Path,
    mask_source: Path,
    relative_name: Path,
    image_root: Path,
    mask_root: Path,
) -> None:
    """Copy one RGB/mask pair into the joint reconstruction output."""

    image_destination = image_root / relative_name
    mask_destination = mask_root / relative_name
    image_destination.parent.mkdir(parents=True, exist_ok=True)
    mask_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_source, image_destination)
    shutil.copy2(mask_source, mask_destination)


def _run_command(command: Sequence[str], log_path: Path) -> None:
    """Run one COLMAP command and append its complete output to a log."""

    process = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = process.stdout or ""
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"$ {shlex.join(command)}\n")
        stream.write(output)
        if output and not output.endswith("\n"):
            stream.write("\n")
    if process.returncode != 0:
        tail = "\n".join(output.splitlines()[-20:])
        raise RuntimeError(
            f"COLMAP command failed with exit code {process.returncode}: "
            f"{shlex.join(command)}\n{tail}"
        )


def _run_reconstruction(
    colmap: str,
    image_dir: Path,
    workspace: Path,
    log_path: Path,
) -> Path:
    """Run feature extraction, exhaustive matching, and sparse mapping."""

    database_path = workspace / "database.db"
    sparse_path = workspace / "sparse"
    feature_command = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
        "--ImageReader.single_camera_per_folder",
        "1",
    ]
    _run_command(feature_command, log_path)
    _run_command(
        [
            colmap,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
        ],
        log_path,
    )
    sparse_path.mkdir()
    _run_command(
        [
            colmap,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_path),
        ],
        log_path,
    )
    return sparse_path


def _parse_cameras_txt(path: Path) -> dict[int, _Camera]:
    """Read camera models and parameters from COLMAP cameras.txt."""

    cameras: dict[int, _Camera] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.split()
            camera_id = int(values[0])
            cameras[camera_id] = _Camera(
                model=values[1],
                width=int(values[2]),
                height=int(values[3]),
                parameters=tuple(float(value) for value in values[4:]),
            )
    if not cameras:
        raise ValueError(f"COLMAP model contains no cameras: {path}")
    return cameras


def _parse_images_txt(path: Path) -> dict[str, _RegisteredImage]:
    """Read image names and poses while skipping 2D feature observations."""

    images: dict[str, _RegisteredImage] = {}
    with path.open("r", encoding="utf-8") as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.split(maxsplit=9)
            if len(values) != 10:
                raise ValueError(f"Invalid COLMAP image record: {stripped}")
            name = values[9].replace("\\", "/")
            if name in images:
                raise ValueError(f"Duplicate COLMAP image name: {name}")
            images[name] = _RegisteredImage(
                image_id=int(values[0]),
                camera_id=int(values[8]),
                qvec_wxyz=tuple(float(value) for value in values[1:5]),
                tvec=tuple(float(value) for value in values[5:8]),
            )
            if not stream.readline():
                raise ValueError(f"Missing points2D line after COLMAP image {name}")
    return images


def _convert_model_to_text(
    colmap: str,
    model_dir: Path,
    text_dir: Path,
    log_path: Path,
) -> dict[str, _RegisteredImage]:
    """Convert one binary sparse component to TXT and return registered images."""

    text_dir.mkdir(parents=True)
    _run_command(
        [
            colmap,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(text_dir),
            "--output_type",
            "TXT",
        ],
        log_path,
    )
    return _parse_images_txt(text_dir / "images.txt")


def _load_complete_model(
    colmap: str,
    sparse_root: Path,
    expected_names: set[str],
    workspace: Path,
    log_path: Path,
) -> tuple[Path, Path, dict[str, _RegisteredImage]]:
    """Load sparse/0 and require it to contain every staged input image."""

    model_dir = sparse_root / "0"
    if not model_dir.is_dir():
        raise RuntimeError("COLMAP mapper did not produce sparse/0")
    text_dir = workspace / "text_model"
    images = _convert_model_to_text(colmap, model_dir, text_dir, log_path)
    missing = sorted(expected_names - set(images))
    if missing:
        raise RuntimeError(
            "COLMAP sparse/0 did not register every sampled frame and "
            f"reference. Missing: {missing}"
        )
    return model_dir, text_dir, images


def _camera_matrix(camera: _Camera) -> np.ndarray:
    """Build a 3x3 K matrix while retaining distortion separately."""

    parameters = camera.parameters
    if camera.model != "SIMPLE_RADIAL" or len(parameters) != 4:
        raise ValueError(
            f"Expected SIMPLE_RADIAL with four parameters, got {camera.model} "
            f"with {len(parameters)}"
        )
    focal_x = focal_y = parameters[0]
    principal_x, principal_y = parameters[1:3]
    return np.asarray(
        [
            [focal_x, 0.0, principal_x],
            [0.0, focal_y, principal_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _qvec_to_rotation(qvec_wxyz: Sequence[float]) -> np.ndarray:
    """Convert a COLMAP qw,qx,qy,qz quaternion to a rotation matrix."""

    qvec = np.asarray(qvec_wxyz, dtype=np.float64)
    qvec /= np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _camera_record(
    name: str,
    metadata: dict[str, Any],
    image: _RegisteredImage,
    camera: _Camera,
) -> dict[str, Any]:
    """Combine source metadata, intrinsics, and raw COLMAP extrinsics."""

    rotation = _qvec_to_rotation(image.qvec_wxyz)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = image.tvec
    camera_to_world = np.linalg.inv(world_to_camera)
    return {
        **metadata,
        "image_name": name,
        "image_id": image.image_id,
        "camera_id": image.camera_id,
        "camera_model": camera.model,
        "image_width": camera.width,
        "image_height": camera.height,
        "camera_parameters": list(camera.parameters),
        "K": _camera_matrix(camera).tolist(),
        "qvec_wxyz": list(image.qvec_wxyz),
        "tvec": list(image.tvec),
        "world_to_camera": world_to_camera.tolist(),
        "camera_to_world": camera_to_world.tolist(),
    }


def prepare_colmap(
    *,
    frames_dir: str | Path,
    frame_masks_dir: str | Path,
    references: Sequence[ReferenceInput],
    sample_stride: int,
    output_dir: str | Path,
    colmap_command: str = "colmap",
) -> Path:
    """Run the complete minimal joint-COLMAP preparation pipeline."""

    frames_dir = Path(frames_dir).expanduser().resolve()
    frame_masks_dir = Path(frame_masks_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    sampled_frames, masks_by_name, references = _validate_inputs(
        frames_dir,
        frame_masks_dir,
        references,
        sample_stride,
        output_dir,
    )
    colmap = _resolve_executable(colmap_command)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    result_dir = temporary_root / "result"
    workspace = temporary_root / "workspace"
    log_path = workspace / "colmap.log"
    image_root = result_dir / "images"
    mask_root = result_dir / "masks"
    metadata_by_name: dict[str, dict[str, Any]] = {}
    try:
        workspace.mkdir()
        result_dir.mkdir()
        for sample_index, frame in enumerate(sampled_frames):
            relative_name = Path("sweep") / frame.name
            _copy_input(
                frame,
                masks_by_name[frame.name],
                relative_name,
                image_root,
                mask_root,
            )
            metadata_by_name[relative_name.as_posix()] = {
                "role": "sweep",
                "source_frame_name": frame.name,
                "source_index": sample_index * sample_stride,
                "sample_index": sample_index,
                "source_image": str(frame),
                "source_mask": str(masks_by_name[frame.name]),
            }
        for reference in references:
            relative_name = Path("references") / f"{reference.label}.png"
            _copy_input(
                reference.image_path,
                reference.mask_path,
                relative_name,
                image_root,
                mask_root,
            )
            metadata_by_name[relative_name.as_posix()] = {
                "role": "reference",
                "label": reference.label,
                "source_image": str(reference.image_path),
                "source_mask": str(reference.mask_path),
            }
        sparse_root = _run_reconstruction(
            colmap,
            image_root,
            workspace,
            log_path,
        )
        selected_model, text_model, registered_images = _load_complete_model(
            colmap,
            sparse_root,
            set(metadata_by_name),
            workspace,
            log_path,
        )
        cameras = _parse_cameras_txt(text_model / "cameras.txt")
        records = []
        for name, metadata in metadata_by_name.items():
            image = registered_images[name]
            if image.camera_id not in cameras:
                raise ValueError(
                    f"Image {name} references missing camera {image.camera_id}"
                )
            records.append(
                _camera_record(name, metadata, image, cameras[image.camera_id])
            )
        shutil.copytree(selected_model, result_dir / "sparse" / "0")
        _run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(selected_model),
                "--output_path",
                str(result_dir / "point_cloud.ply"),
                "--output_type",
                "PLY",
            ],
            log_path,
        )
        payload = {
            "format": "modal_gaussians_colmap",
            "version": 1,
            "sample_stride": sample_stride,
            "camera_model": "SIMPLE_RADIAL",
            "camera_grouping": "sweep_and_references",
            "colmap_uses_foreground_masks": False,
            "pose_convention": "x_camera = world_to_camera @ x_world_homogeneous",
            "world_coordinates": "raw COLMAP coordinates with arbitrary scale",
            "frames": [record for record in records if record["role"] == "sweep"],
            "references": [
                record for record in records if record["role"] == "reference"
            ],
        }
        (result_dir / "cameras.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copy2(log_path, result_dir / "colmap.log")
        os.replace(result_dir, output_dir)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return output_dir


__all__ = ["ReferenceInput", "prepare_colmap"]
