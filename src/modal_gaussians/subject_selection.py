"""Shared IO and box math for saved full-scene Gaussian selections."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import viser.transforms as vtf


def _box_values(position, wxyz, dimensions):
    position, wxyz, dimensions = (np.asarray(value, dtype=np.float64).copy()
                                 for value in (position, wxyz, dimensions))
    if (position.shape != (3,) or wxyz.shape != (4,) or dimensions.shape != (3,)
            or not all(np.isfinite(value).all() for value in (position, wxyz, dimensions))
            or np.any(dimensions <= 0)):
        raise ValueError("Box requires finite position [3], quaternion [4], and positive dimensions [3]")
    norm = np.linalg.norm(wxyz)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Box quaternion must have a nonzero finite norm")
    return position, wxyz / norm, dimensions


def points_in_box(points, position, wxyz, dimensions):
    """Select centers in box-local coordinates, including its boundary."""
    position, wxyz, dimensions = _box_values(position, wxyz, dimensions)
    local = (np.asarray(points) - position) @ vtf.SO3(wxyz).as_matrix()
    return (np.abs(local) <= dimensions / 2 + 1e-8 * dimensions.max()).all(axis=1)


def projected_box_pixels(camera, position, wxyz, dimensions):
    """Return image pixels whose forward camera rays intersect the selected OBB."""
    from modal_gaussians.camera_geometry import undistort_normalized

    position, wxyz, dimensions = _box_values(position, wxyz, dimensions)
    K = camera.K.detach().cpu().numpy().astype(np.float64)
    c2w = np.linalg.inv(camera.world_to_camera.detach().cpu().numpy().astype(np.float64))
    rotation = vtf.SO3(wxyz).as_matrix()
    origin = (c2w[:3, 3] - position) @ rotation
    camera_to_box = c2w[:3, :3].T @ rotation
    half = dimensions / 2
    radial_k = camera.radial_distortion
    pixels = []
    count = camera.height * camera.width
    for start in range(0, count, 65_536):
        y, x = np.divmod(np.arange(start, min(start + 65_536, count)), camera.width)
        xy = np.column_stack(((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1]))
        xy = undistort_normalized(xy, radial_k)
        directions = np.column_stack((xy, np.ones(len(x)))) @ camera_to_box
        # Camera ray z is one: its parameter is camera-space depth, so the
        # positive lower bound also clips boxes crossing/behind the near plane.
        enter, leave = np.full(len(x), 1e-8), np.full(len(x), np.inf)
        valid = np.ones(len(x), dtype=bool)
        for axis in range(3):
            direction = directions[:, axis]
            parallel = np.abs(direction) < 1e-12
            valid &= ~(parallel & (abs(origin[axis]) > half[axis]))
            first = np.divide(-half[axis] - origin[axis], direction,
                              out=np.full(len(x), -np.inf), where=~parallel)
            last = np.divide(half[axis] - origin[axis], direction,
                             out=np.full(len(x), np.inf), where=~parallel)
            enter = np.maximum(enter, np.minimum(first, last))
            leave = np.minimum(leave, np.maximum(first, last))
        hit = valid & (leave >= enter)
        pixels.append(np.column_stack((x[hit], y[hit])))
    return np.concatenate(pixels).astype(np.int64, copy=False) if pixels else np.empty((0, 2), dtype=np.int64)


def save_subject_selection(work_dir, scene_path, scene, points, position, wxyz, dimensions):
    """Publish a new selection without rewriting scene tensors or prior selections."""
    position, wxyz, dimensions = _box_values(position, wxyz, dimensions)
    indices = np.flatnonzero(points_in_box(points, position, wxyz, dimensions))
    if not len(indices):
        raise ValueError("The box contains no Gaussian centers; adjust it before saving")
    manifest = {
        "format": "modal_gaussians.subject_selection", "version": 1,
        "source_scene": str(Path(scene_path).resolve()),
        "static_scene_identity": scene.manifest["static_scene_identity"],
        "foreground_identity": scene.manifest["foreground_identity"],
        "background_identity": scene.manifest["background_identity"],
        "source_counts": {"foreground": scene.foreground.count, "background": scene.background.count},
        "index_order": "foreground_then_background", "selection_rule": "gaussian_center_in_oriented_box",
        "selected_count": len(indices),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    directory = Path(work_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Unique filenames keep each explicit Save, including concurrent sessions.
        prefix = f"subject-selection-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
        with tempfile.NamedTemporaryFile(dir=directory, prefix=prefix, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, manifest_json=np.asarray(json.dumps(manifest)),
                                selected_indices=indices, box_position=position,
                                box_wxyz=wxyz, box_dimensions=dimensions)
        output = temporary.with_suffix(".npz")
        if output.exists():
            raise FileExistsError(output)
        os.rename(temporary, output)
        return output
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_subject_selection(path, scene):
    """Read saved indices and box, checking their original full-scene indexing."""
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        indices = archive["selected_indices"]
        if (manifest.get("format") != "modal_gaussians.subject_selection"
                or manifest.get("version") != 1
                or manifest.get("index_order") != "foreground_then_background"
                or manifest.get("selection_rule") != "gaussian_center_in_oriented_box"):
            raise ValueError("Unsupported subject selection")
        counts = {"foreground": scene.foreground.count, "background": scene.background.count}
        if (manifest.get("source_counts") != counts or any(
                manifest.get(key) != scene.manifest[key] for key in
                ("static_scene_identity", "foreground_identity", "background_identity"))):
            raise ValueError("Subject selection belongs to a different static scene")
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or not len(indices) or np.any(indices < 0) or np.any(indices >= scene.count)
                or np.any(indices[1:] <= indices[:-1]) or manifest.get("selected_count") != len(indices)):
            raise ValueError("Invalid selected Gaussian indices")
        box = _box_values(archive["box_position"], archive["box_wxyz"], archive["box_dimensions"])
        return manifest, indices.astype(np.int64, copy=True), box


def load_subject_selection(path, scene):
    """Restore the Viewer box without changing saved selection semantics."""
    return read_subject_selection(path, scene)[2]
