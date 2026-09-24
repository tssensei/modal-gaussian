"""Normalize saved viewer inputs without publishing another model artifact."""
from dataclasses import dataclass, field
from functools import cached_property
import json
from pathlib import Path
from typing import Any

import numpy as np

from modal_gaussians.motion.common.completed_modes import CompletedModesArtifact, load_completed_modes
from modal_gaussians.coordinates.design import load_rendered_modal_design
from modal_gaussians.results.artifact import _load_coordinate_artifact
from modal_gaussians.common.scene_store import library_root, resolve_path
from modal_gaussians.geometry.scene import cameras_from_scene_manifest, load_static_scene


def read_json(path):
    return json.loads(resolve_path(path, strict=True).read_text(encoding="utf-8"))


@dataclass
class ViewerMode:
    artifact: Any
    slot: int
    design: Any = None
    design_slot: int = 0
    settings: dict = field(default_factory=dict)
    prepared: Path | None = None

    @property
    def key(self):
        return self.artifact.manifest["completed_modes_identity"], self.slot

    @property
    def frequency(self):
        return self.artifact.manifest["modes"][self.slot]["frequency_hz"]

    @cached_property
    def modal_views(self):
        model = self.artifact.manifest
        dense = read_json(resolve_path(model["complex_2d_modes"]) / "manifest.json")
        if dense["complex_2d_modes_identity"] != model["complex_2d_modes_identity"]:
            raise ValueError("Saved modal image source identity differs")
        return {v["label"]: v for v in dense["views"]}

    @cached_property
    def exports(self):
        return {label: read_json(resolve_path(v["selected_source"]["path"]) / "manifest.json")
                for label, v in self.modal_views.items()}


@dataclass
class ViewerInput:
    path: Path
    scene: Any
    modes: list[ViewerMode]
    views: list[dict]
    coordinates: Any = None
    coordinate_columns: tuple = ()

    @property
    def coordinate_views(self):
        return {} if self.coordinates is None else {
            v["label"]: v for v in self.coordinates.manifest["views"]}


def _model_keys(manifest):
    if manifest.get("version") == 17:
        return [(s["identity"], s["slot"]) for s in manifest["sources"]]
    return [(manifest["completed_modes_identity"], k) for k in range(len(manifest["modes"]))]


def _check_design(design, artifact):
    m, source = design.manifest, artifact.manifest
    for key in ("static_scene_identity", "foreground_identity", "completed_modes_identity"):
        if m[key] != source[key]:
            raise ValueError(f"Viewer projection {key} differs from its model")
    if m["modes"] != source["modes"]:
        raise ValueError("Viewer projection mode order differs")
    by_label = {v["label"]: v for v in source["views"]}
    for view in m["views"]:
        original = by_label.get(view["label"], {})
        for key in ("camera_identity", "shape_hw", "motion_reference"):
            if view.get(key) != original.get(key):
                raise ValueError(f"Viewer projection {key} differs")


def load_viewer_input(path, *, coordinates=None):
    """Read explicit sources only; old paths and identities stay unchanged."""
    root = resolve_path(path, strict=True)
    header = read_json(root if root.is_file() else root / "manifest.json")
    models, designs = {}, {}

    def model(path):
        physical = resolve_path(path, strict=True)
        if physical not in models:
            loaded = load_completed_modes(physical)
            if loaded.manifest.get("version") == 18:
                # Release training/interpolation arrays before loading the next frequency.
                names = {"phi", "g_points", "support_class", "observation_view_mask", "alphas",
                         "alpha_identifiable_mask", "g_edge_index", "g_edge_weight", "g_component_index",
                         "c_control_point_index", "c_positions", "t_host_gaussian_index"}
                loaded = CompletedModesArtifact(loaded.path, loaded.manifest,
                    {k: v for k, v in loaded.arrays.items() if k in names},
                    loaded.rotation, loaded.control_displacement)
            models[physical] = loaded
        return models[physical]

    def design(path, artifact):
        physical = resolve_path(path, strict=True)
        if physical not in designs:
            designs[physical] = load_rendered_modal_design(physical)
        value = designs[physical]
        _check_design(value, artifact)
        return value

    def expand(artifact, projection=None, slots=None, settings=None):
        m = artifact.manifest
        selected = range(len(m["modes"])) if slots is None else slots
        result = []
        for k in selected:
            original, slot = artifact, k
            if m.get("version") == 17:
                source = m["sources"][k]
                original, slot = model(source["path"]), source["slot"]
                if (type(slot) is not int or not 0 <= slot < len(original.manifest["modes"])
                        or original.manifest["completed_modes_identity"] != source["identity"]
                        or original.manifest["modes"][slot]["frequency_hz"] != m["modes"][k]["frequency_hz"]
                        or any(original.manifest[key] != m[key] for key in
                               ("static_scene_identity", "foreground_identity"))):
                    raise ValueError("Mode-bank source identity, slot or frequency differs")
            result.append(ViewerMode(original, slot, projection, k,
                dict(projection.manifest["settings"] if projection is not None else settings or {})))
        return result

    indexed = isinstance(header, list)
    if indexed:
        records = header
    elif isinstance(header, dict):
        format_name = header.get("format")
        if format_name == "modal_gaussians.completed_modes":
            records = [{"completed_modes": str(root)}]
        elif format_name == "modal_gaussians.modal_result":
            if header.get("version") != 1:
                raise ValueError("Unsupported result version")
            if coordinates is not None:
                raise ValueError("A result already binds coordinates; --coordinates cannot override them")
            sources = header["sources"]
            record = {}
            for name in ("completed_modes", "rendered_design"):
                source = sources[name]
                if source["identity_name"] != name + "_identity":
                    raise ValueError(f"Result {name} identity name differs")
                record[name], record[name + "_identity"] = source["path"], source["identity"]
            records = [record]
            coordinates = sources["coordinates"]["path"]
        else:
            raise ValueError("Viewer input must be a model, result index or result")
    else:
        raise ValueError("Viewer input manifest must be an object or result index")

    def record_path(value):
        path = Path(value)
        return resolve_path(library_root() / path if indexed and not path.is_absolute() else path)

    modes = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Viewer result index entries must be objects")
        if not record.get("completed_modes") and not record.get("completed_modes_identity"):
            continue
        if not record.get("completed_modes") or (indexed and not record.get("completed_modes_identity")):
            raise ValueError("Published entry has an incomplete model binding")
        artifact = model(record_path(record["completed_modes"]))
        if record.get("completed_modes_identity", artifact.manifest["completed_modes_identity"]) != artifact.manifest["completed_modes_identity"]:
            raise ValueError("Published model identity differs")
        slots, settings, projection = None, {}, None
        if indexed:
            slots = [k for k, m in enumerate(artifact.manifest["modes"])
                     if abs(m["frequency_hz"] - record["frequency_hz"]) < 1e-9]
            if len(slots) != 1:
                raise ValueError("Indexed frequency must identify one model slot")
        elif "modes" in header and header["modes"] != artifact.manifest["modes"]:
            raise ValueError("Viewer wrapper mode order differs from its model")
        design_path = record.get("rendered_design")
        if record.get("experiment"):
            experiment = record_path(record["experiment"])
            if design_path is None and (experiment / "rendered_design" / "manifest.json").is_file():
                design_path = experiment / "rendered_design"
            if (experiment / "iteration.json").is_file():
                settings = read_json(experiment / "iteration.json").get("config", {}).get("design", {})
        if design_path is not None:
            projection = design(record_path(design_path), artifact)
            if record.get("rendered_design_identity", projection.manifest["rendered_design_identity"]) != projection.manifest["rendered_design_identity"]:
                raise ValueError("Published projection identity differs")
        entries = expand(artifact, projection, slots, settings)
        if record.get("prepared"):
            for entry in entries:
                entry.prepared = record_path(record["prepared"])
        modes.extend(entries)
    if not modes:
        raise ValueError("Viewer input contains no published modes")
    modes.sort(key=lambda m: m.frequency)
    if (not np.isfinite([m.frequency for m in modes]).all()
            or len({m.key for m in modes}) != len(modes)
            or np.any(np.diff([m.frequency for m in modes]) < 1e-9)):
        raise ValueError("Duplicate modes/frequencies: select an unambiguous experiment index")

    first = modes[0].artifact.manifest
    scene = load_static_scene(first["static_scene"], "cpu")
    if isinstance(header, dict):
        for key in ("static_scene_identity", "foreground_identity"):
            if key in header and header[key] != scene.manifest[key]:
                raise ValueError(f"Viewer wrapper {key} differs")
        if header.get("format") == "modal_gaussians.modal_result":
            saved_scene = header["sources"].get("static_scene")
            if saved_scene and saved_scene["identity"] != scene.manifest["static_scene_identity"]:
                raise ValueError("Result scene identity differs")
    points = scene.foreground.active()["means"].detach().cpu().numpy()
    cameras = {c.label: c for c in cameras_from_scene_manifest(scene.manifest) if c.role == "reference"}
    views = []
    for source in first["views"]:
        camera = cameras.get(source["label"])
        if camera is None or camera.to_manifest_record()["camera_identity"] != source["camera_identity"]:
            raise ValueError("Viewer camera differs from the saved mode")
        view = {**source, "camera_name": camera.name}
        if "selected_modal_supervision" in first:
            exported = modes[0].exports[source["label"]]
            view.setdefault("fps_hz", exported["fps_hz"])
            view.setdefault("flow_reference_frame_name", exported["reference_frame_name"])
            view.setdefault("flow_reference_frame_index", exported["reference_frame_index"])
        views.append(view)
    if len({v["label"] for v in views}) != len(views):
        raise ValueError("Viewer camera labels must be unique")
    checked = set()
    for mode in modes:
        artifact, m = mode.artifact, mode.artifact.manifest
        if artifact.path in checked:
            continue
        checked.add(artifact.path)
        for key in ("static_scene_identity", "foreground_identity"):
            if m[key] != scene.manifest[key]:
                raise ValueError(f"Viewer model {key} differs")
        if m["views"] != first["views"]:
            raise ValueError("Viewer model view/reference bindings differ")
        if "g_points" in artifact.arrays and not np.array_equal(artifact.arrays["g_points"], points):
            raise ValueError("Viewer Gaussian order differs from the scene")
        if artifact.arrays["phi"].shape != (len(m["modes"]), len(points), 3):
            raise ValueError("Viewer model Gaussian domain differs")

    result = ViewerInput(root, scene, modes, views)
    if coordinates is not None:
        _, fitted = _load_coordinate_artifact(coordinates)
        cm = fitted.manifest
        projection = load_rendered_modal_design(cm["rendered_design"])
        source_manifest = read_json(resolve_path(projection.manifest["completed_modes"]) / "manifest.json")
        if (projection.manifest["rendered_design_identity"] != cm["rendered_design_identity"]
                or source_manifest["completed_modes_identity"] != cm["completed_modes_identity"]
                or projection.manifest["completed_modes_identity"] != cm["completed_modes_identity"]
                or source_manifest["modes"] != cm["modes"]):
            raise ValueError("Coefficient model/projection binding differs")
        if any(source_manifest[k] != scene.manifest[k] or projection.manifest[k] != scene.manifest[k]
               for k in ("static_scene_identity", "foreground_identity")):
            raise ValueError("Coefficient scene/foreground binding differs")
        keys = _model_keys(source_manifest)
        if len(set(keys)) != len(keys) or set(keys) != {m.key for m in modes}:
            raise ValueError("Coefficient source models/slots differ from Viewer modes")
        available = {v["label"]: v for v in views}
        projected = {v["label"]: v for v in projection.manifest["views"]}
        for view in cm["views"]:
            label = view["label"]
            if label not in available or label not in projected:
                raise ValueError("Coefficient view is absent from the model")
            expected, saved = available[label], projected[label]
            if any(expected.get(k) != saved.get(k) for k in ("camera_identity", "shape_hw", "motion_reference")):
                raise ValueError("Coefficient camera/reference binding differs")
            ref = expected.get("motion_reference", {
                "reference_frame_name": saved["flow_reference_frame_name"],
                "reference_frame_index": saved["flow_reference_frame_index"]})
            if (view["shape_hw"] != expected["shape_hw"] or view["fps_hz"] != saved["fps_hz"]
                    or view["flow_identity"] != saved["flow_identity"]
                    or ("frame_count" in saved and view["frame_count"] != saved["frame_count"])
                    or any(view[k] != ref[k] for k in ("reference_frame_name", "reference_frame_index"))):
                raise ValueError("Coefficient view timing/reference differs")
        result.coordinates = fitted
        result.coordinate_columns = tuple(keys.index(m.key) for m in modes)
        if isinstance(header, dict) and header.get("format") == "modal_gaussians.modal_result":
            source = header["sources"]["coordinates"]
            if cm[source["identity_name"]] != source["identity"]:
                raise ValueError("Result coefficient identity differs")
    return result
