"""RGB refinement of direct coordinates with immutable spatial mode fields."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from modal_gaussians.common.scene_store import resolve_path
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

from modal_gaussians import __version__
from modal_gaussians.coordinates.direct import _canonical_json, _sha256_file, _validate_modes
from modal_gaussians.common.progress import Progress
from modal_gaussians.common.cache import publish_directory
from modal_gaussians.coordinates.fitting import RGBFitConfig, solve_rgb_coordinates_view
from modal_gaussians.coordinates.rendering import make_rgb_renderer, resize_rgb
from modal_gaussians.geometry.scene import cameras_from_scene_manifest


RGB_COORDINATES_FORMAT = "modal_gaussians.rgb_modal_coordinates"
SOLVER_CONVENTION = {
    "solver": "fixed_modes_rgb_v2",
    "parameters": "independent_per_view_per_frame_complex_coordinates",
    "initialization": "direct_q_minus_reference_then_shared_rgb_offset",
    "gauge": "free_reference_and_temporal_mean",
    "normalization": "p=direct_mode_pair_scale*q; shared_real_imag_scale",
    "deformation": "means_static+real(sum(q*phi)); exp(real(sum(q*rotation)))*quat_static",
    "frozen": ["phi", "rotation", "static_scene", "appearance", "cameras"],
    "rgb_loss": "0.8*L1+0.2*(1-SSIM); common_valid_pixels; fully_valid_SSIM_windows",
    "anchor": "mean_k(abs(p-p_initial_rgb_offset)^2); linearly_decaying_weight",
    "temporal_regularization": "none",
}
QUALITY_GATE = {"status": "rgb_coordinates_candidate_unapproved"}


@dataclass(frozen=True)
class RGBModalCoordinatesArtifact:
    path: Path
    manifest: dict[str, Any]
    coordinates: np.ndarray


def load_rgb_frame(path, shape_hw, expected_sha256=None):
    """Decode the recorded PNG grid and optionally verify its immutable bytes."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"RGB source changed since fitting: {path}")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or list(bgr.shape[:2]) != list(shape_hw):
        raise ValueError(f"RGB frame cannot be decoded at the recorded shape: {path}")
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).float() / 255., digest


def _identity(manifest: Mapping[str, Any]) -> str:
    # Paths and command lines may change when the same immutable artifacts move.
    fields = ("format", "version", "direct_coordinates_identity", "rendered_design_identity",
              "completed_modes_identity", "modes", "views", "counts", "coordinates",
              "settings", "solver", "quality_gate", "training")
    payload = {name: manifest[name] for name in fields}
    payload["images"] = [{"label": record["label"], "files": record["files"], 'validity_sha256': record['validity']['sha256']}
                         for record in manifest["images"]]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def load_rgb_modal_coordinates(path: str | Path) -> RGBModalCoordinatesArtifact:
    """Load coefficients and their small provenance record, without rendering."""
    root = resolve_path(path, strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("format") != RGB_COORDINATES_FORMAT
            or manifest.get("version") != 2):
        raise ValueError("Unsupported RGB-coordinate artifact")
    if manifest.get("solver") != SOLVER_CONVENTION or manifest.get("quality_gate") != QUALITY_GATE:
        raise ValueError("RGB-coordinate solver or quality gate differs")
    _validate_modes(manifest.get("modes"))
    settings = RGBFitConfig(**manifest["settings"])
    settings.validate()
    views = manifest["views"]
    if not views or len({view["label"] for view in views}) != len(views):
        raise ValueError("RGB-coordinate views must be nonempty and unique")
    offset = 0
    for index, view in enumerate(views):
        count, reference = view["frame_count"], view["reference_frame_index"]
        names = view["frame_names"]
        if (view["index"] != index or view["frame_offset"] != offset
                or not isinstance(count, int) or count < 1 or len(names) != count
                or len(set(names)) != count or not 0 <= reference < count
                or names[reference] != view["reference_frame_name"]
                or not np.isfinite(view["fps_hz"]) or view["fps_hz"] <= 0):
            raise ValueError("RGB-coordinate frame metadata is inconsistent")
        offset += count
    expected_shape = (offset, len(manifest["modes"]))
    if manifest["counts"] != {"views": len(views), "frames": offset, "modes": expected_shape[1]}:
        raise ValueError("RGB-coordinate counts differ")
    record = manifest["coordinates"]
    if (record.get("file") != "coordinates.npy" or record.get("dtype") != "complex64"
            or record.get("shape") != list(expected_shape)
            or record.get("sha256") != _sha256_file(root / "coordinates.npy")):
        raise ValueError("RGB-coordinate array record differs")
    coordinates = np.load(root / "coordinates.npy", allow_pickle=False)
    if (coordinates.dtype != np.complex64 or coordinates.shape != expected_shape
            or not np.isfinite(coordinates).all()):
        raise ValueError("RGB coordinates must be finite complex64 [sum(T), K]")
    if _identity(manifest) != manifest.get("rgb_coordinates_identity"):
        raise ValueError("RGB-coordinate identity differs")
    if len(views) != len(manifest['images']):
        raise ValueError('RGB image view count differs')
    for view, record in zip(views, manifest['images']):
        if record['label'] != view['label'] or 'validity' not in record:
            raise ValueError('RGB view/support binding differs')
        load_valid_mask(record, view['shape_hw'])
    return RGBModalCoordinatesArtifact(root, manifest, coordinates)


def load_valid_mask(record, shape_hw, scale=1., device='cpu'):
    """Load a recorded common support, conservatively downsample it for RGB losses."""
    support = record.get('validity')
    if support is None:
        return None
    path = resolve_path(support['path'], strict=True)
    if _sha256_file(path) != support['sha256']:
        raise ValueError('RGB validity mask checksum differs')
    mask = np.load(path, allow_pickle=False)
    if mask.dtype != bool or list(mask.shape) != list(shape_hw) or not mask.any():
        raise ValueError('Invalid RGB validity mask')
    result = torch.as_tensor(mask, device=device)
    if scale != 1:
        from torch.nn import functional as F
        height, width = int(shape_hw[0]*scale), int(shape_hw[1]*scale)
        result = F.interpolate(result.float()[None, None], size=(height, width), mode='area')[0, 0] >= 1-1e-6
    if not result.any():
        raise ValueError('No valid RGB pixels at requested scale')
    return result


def _image_directory(flow_dir, view):
    from modal_gaussians.coordinates.sources import load_coordinate_flow
    flow = load_coordinate_flow(flow_dir)
    try:
        if flow.identity != view['flow_identity']:
            raise ValueError('RGB SEA-RAFT identity differs from initial coordinates')
        for name in ('frame_names', 'fps_hz', 'reference_frame_name', 'reference_frame_index'):
            if flow.manifest[name] != view[name]:
                raise ValueError(f'RGB SEA-RAFT {name} differs')
        from modal_gaussians.preprocessing.reference import load_reference
        ref = load_reference(flow.manifest['stabilization_source'])
        return flow.image_directory, dict(path=str(ref.path/'valid_mask.npy'), sha256=ref.manifest['valid_sha256'])
    finally:
        flow.arrays.flow.store.close()


def build_rgb_modal_coordinates_artifact(
    *,
    scene_dir: str | Path,
    completed_modes_dir: str | Path,
    direct_coordinates_dir: str | Path,
    output_dir: str | Path,
    config: RGBFitConfig | None = None,
    image_directories: Mapping[str, str | Path] | None = None,
    view_label: str | None = None,
    device: str = "cuda",
    command: Sequence[str] = (),
) -> RGBModalCoordinatesArtifact:
    """Fit one selected recording or all recordings into a new coefficient artifact."""
    # Local import avoids the result loader's RGB artifact dispatch cycle.
    from modal_gaussians.results.artifact import _load_sources, _select_views

    settings = config or RGBFitConfig()
    settings.validate()
    destination = resolve_path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Choose a new RGB-coordinate output directory: {destination}")
    for source in (scene_dir, completed_modes_dir, direct_coordinates_dir):
        if destination.is_relative_to(resolve_path(source)):
            raise ValueError("RGB output must not be inside an input artifact")
    _, scene, completed, kind, direct, design, _, views = _load_sources(
        scene_dir=scene_dir, completed_modes_dir=completed_modes_dir,
        coordinates_dir=direct_coordinates_dir,
    )
    if kind != "direct":
        raise ValueError("RGB fitting requires direct coordinates as initialization")
    if view_label is not None:
        available = [view["label"] for view in views]
        if view_label not in available:
            raise ValueError(f"Unknown view {view_label!r}; available: {', '.join(available)}")
        views = [view for view in views if view["label"] == view_label]
    overrides = dict(image_directories or {})
    if set(overrides) - {view["label"] for view in views}:
        raise ValueError("RGB image override contains an unknown or unselected view label")
    cameras = {camera.name: camera for camera in cameras_from_scene_manifest(scene.manifest)}
    flow_paths = direct.manifest["flow_artifacts"]
    if len(flow_paths) != len(direct.manifest["views"]):
        raise ValueError("Direct source flow count differs from its views")
    sources = []
    for view in views:
        label = view["label"]
        directory, validity = _image_directory(flow_paths[view["index"]], view)
        if label in overrides:
            directory = resolve_path(overrides[label], strict=True)
        if not directory.is_dir() or destination.is_relative_to(directory):
            raise ValueError("RGB source must be an image directory disjoint from output")
        paths = [directory / f"{name}.png" for name in view["frame_names"]]
        if any(path.parent != directory or not path.is_file() for path in paths):
            raise FileNotFoundError(f"Missing or invalid recorded PNG frames in {directory}")
        camera = cameras[view["camera_name"]]
        if [camera.height, camera.width] != view["shape_hw"]:
            raise ValueError(f"RGB camera shape differs for {label!r}")
        sources.append((directory, paths, camera, validity))

    coordinates_by_view, image_records, training = [], [], []
    for view, (directory, paths, camera, validity) in zip(views, sources):
        hashes: dict[str, str] = {}
        supports = {scale:load_valid_mask({'validity':validity}, view['shape_hw'], scale, device) for scale in settings.scales}

        # ponytail: keep only a few CPU frames; use a disk pyramid if PNG decoding dominates.
        @lru_cache(maxsize=8)
        def target(frame: int, scale: float) -> torch.Tensor:
            path = paths[frame]
            rgb, digest = load_rgb_frame(path, view["shape_hw"], hashes.get(path.name))
            hashes[path.name] = digest
            return resize_rgb(rgb, scale)

        render = make_rgb_renderer(scene, camera, completed.arrays["phi"], completed.rotation, device)
        progress = Progress(f"RGB coordinates {view['label']}",
                            settings.offset_steps + len(settings.scales) * settings.epochs_per_scale,
                            unit="offset steps/epochs")

        def on_progress(row: dict[str, Any]) -> None:
            count = row["step"] if row["phase"] == "offset" else settings.offset_steps + row["epoch"]
            progress.update(count, f"{row['phase']} scale={row['scale']:g} rgb_loss={row['rgb_loss']:.6g}")

        start, count = view["frame_offset"], view["frame_count"]
        coordinates, summary = solve_rgb_coordinates_view(
            direct.coordinates[start:start + count], direct.diagnostics["mode_pair_scales"][view["index"]],
            view["reference_frame_index"], render, target, settings, device,
            on_progress=on_progress, valid_mask=supports.__getitem__,
        )
        coordinates_by_view.append(coordinates)
        training.append({"label": view["label"], **summary})
        image_records.append({"label": view["label"], "directory": str(directory), 'validity':validity,
                              "files": [{"name": path.name, "sha256": hashes[path.name]} for path in paths]})
        target.cache_clear()
        del render

    coordinates = np.concatenate(coordinates_by_view).astype(np.complex64)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary:
        work = Path(temporary)
        np.save(work / "coordinates.npy", coordinates, allow_pickle=False)
        manifest = {
            "format": RGB_COORDINATES_FORMAT, "version": 2,
            "producer": {"project_version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
                         "command": list(command)},
            "direct_coordinates": str(direct.path),
            "direct_coordinates_identity": direct.manifest["direct_coordinates_identity"],
            "rendered_design": str(design.path),
            "rendered_design_identity": design.manifest["rendered_design_identity"],
            "completed_modes_identity": completed.manifest["completed_modes_identity"],
            "flow_artifacts": [flow_paths[view["index"]] for view in views],
            "modes": completed.manifest["modes"],
            "views": [{key: value for key, value in view.items() if key != "diagnostics"}
                      for view in _select_views(direct.manifest["views"], [v["label"] for v in views])],
            "counts": {"views": len(views), "frames": len(coordinates), "modes": coordinates.shape[1]},
            "coordinates": {"file": "coordinates.npy", "dtype": "complex64", "shape": list(coordinates.shape),
                            "sha256": _sha256_file(work / "coordinates.npy")},
            "images": image_records, "settings": settings.to_dict(), "solver": SOLVER_CONVENTION,
            "quality_gate": QUALITY_GATE, "training": training,
        }
        manifest["rgb_coordinates_identity"] = _identity(manifest)
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"RGB-coordinate output appeared during fitting: {destination}")
        publish_directory(work, destination)
    # Do not rerender, replay fields, or run an evaluation/readback scan after fitting.
    return RGBModalCoordinatesArtifact(destination, manifest, coordinates)
