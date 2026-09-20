"""Fixed mode banks and SEA-RAFT initialization, stored inside one scene experiment."""
from contextlib import contextmanager
import json
import os
import sys
from pathlib import Path
import tempfile

import numpy as np

from modal_gaussians.coefficient_sources import load_coordinate_flow
from modal_gaussians.direct_coordinates import (
    DIRECT_COORDINATES_FORMAT, SOLVER_CONVENTION, DirectCoordinateConfig,
    DirectModalCoordinatesArtifact, solve_direct_coordinates_view, _validate_modes,
)
from modal_gaussians.iteration_cache import atomic_json, identity, sha256, module_revision
from modal_gaussians.motion.common.completed_modes import (
    COMPLETED_MODES_FORMAT, CompletedModesArtifact, load_completed_modes,
)
from modal_gaussians.numpy_io import save_named_arrays
from modal_gaussians.progress import Progress
from modal_gaussians.rendered_design import (
    RenderedDesignConfig, RenderedDesignViewInput,
    build_rendered_modal_design_artifact, load_rendered_modal_design,
)
from modal_gaussians.scene_store import asset_path, library_root, resolve_path
from modal_gaussians.static import cameras_from_scene_manifest, load_static_scene


BANK_METHOD = "fixed_frequency_mode_bank"
INITIAL_SOLVER = {**SOLVER_CONVENTION, "evaluation": "not_run"}
INITIAL_GATE = {"required": True, "status": "direct_coordinates_candidate_unapproved",
                "inherited_from": "rendered_design_candidate_unapproved"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def _publish(destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as directory:
        work = Path(directory)
        yield work
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Artifact appeared during preparation: {destination}")
        os.rename(work, destination)


def _bank_identity(m):
    return identity({k: m[k] for k in ("format", "version", "completion_method", "contract",
                    "static_scene_identity", "foreground_identity", "modes", "views", "counts", "files")})


def load_mode_bank(path):
    root = resolve_path(path, strict=True)
    m = _json(root / "manifest.json")
    if (m.get("format") != COMPLETED_MODES_FORMAT or m.get("version") != 17
            or m.get("completion_method") != BANK_METHOD
            or m.get("completed_modes_identity") != _bank_identity(m)):
        raise ValueError("Unsupported or inconsistent mode bank")
    _validate_modes(m["modes"])
    arrays = {}
    for name in ("phi", "rotation"):
        if m["files"][name]["file"] != f"{name}.npy":
            raise ValueError("Unexpected mode-bank array path")
        value = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if value.dtype != np.complex64 or list(value.shape) != m["files"][name]["shape"]:
            raise ValueError("Mode-bank array header differs")
        arrays[name] = value
    with np.load(root / "support.npz", allow_pickle=False) as archive:
        arrays.update({name: archive[name] for name in archive.files})
    shape = (len(m["modes"]), m["counts"]["foreground_gaussians"], 3)
    if (arrays["phi"].shape != shape or arrays["rotation"].shape != shape
            or arrays["g_points"].shape != shape[1:]
            or arrays["support_class"].shape != shape[:2]
            or arrays["observation_view_mask"].shape != (*shape[:2], len(m["views"]))
            or arrays["support_class"].dtype != np.int8
            or arrays["observation_view_mask"].dtype != np.bool_
            or m["counts"]["modes"] != shape[0] or m["counts"]["views"] != len(m["views"])):
        raise ValueError("Mode-bank domains differ")
    return CompletedModesArtifact(root, m, arrays, arrays.pop("rotation"))


def _mode_sources(index_path, status, expected_modes):
    records = _json(index_path)
    if not isinstance(records, list):
        raise ValueError("Result index must be a list")
    selected = sorted((r for r in records if r.get("status") == status), key=lambda r: r["frequency_hz"])
    if len(selected) != expected_modes or len({r["frequency_hz"] for r in selected}) != expected_modes:
        raise ValueError(f"Expected exactly {expected_modes} distinct completed frequencies, found {len(selected)}")
    sources = []
    for record in selected:
        path = resolve_path(library_root() / record["completed_modes"], strict=True)
        header = _json(path / "manifest.json")
        if (header.get("version") != 16 or header.get("format") != COMPLETED_MODES_FORMAT
                or header.get("completed_modes_identity") != record["completed_modes_identity"]):
            raise ValueError("Index must bind an unchanged completed v16 model")
        slots = [i for i, mode in enumerate(header["modes"])
                 if abs(mode["frequency_hz"] - record["frequency_hz"]) < 1e-9]
        if len(slots) != 1:
            raise ValueError("Indexed frequency is not unique in its model")
        sources.append({"path": str(path), "slot": slots[0], "identity": record["completed_modes_identity"],
                        "frequency_hz": record["frequency_hz"], "header": header})
    return sources


def _bank_contract(sources, scene_manifest, flows):
    from modal_gaussians.motion.neural import artifacts, neural_field, component_field
    from modal_gaussians import coefficient_sources
    cameras = {c.label: c for c in cameras_from_scene_manifest(scene_manifest) if c.role == "reference"}
    first = sources[0]["header"]
    for source in sources:
        m = source["header"]
        for key in ("static_scene_identity", "foreground_identity"):
            if m[key] != scene_manifest[key]:
                raise ValueError(f"Mode source {key} differs from static scene")
        if m["views"] != first["views"]:
            raise ValueError("Mode source camera/geometry-reference view bindings differ")
    if len(flows) != len(first["views"]):
        raise ValueError("SEA-RAFT and mode view counts differ")
    views = []
    for view, flow in zip(first["views"], flows):
        camera = cameras[view["label"]]
        if (view["flow_identity"] != flow.reference_identity
                or view["camera_identity"] != camera.to_manifest_record()["camera_identity"]
                or view["shape_hw"] != [camera.height, camera.width]
                or view["shape_hw"] != flow.manifest["flow_shape"][1:3]):
            raise ValueError("SEA-RAFT reference/camera differs from the fixed spatial modes")
        views.append({**view, "flow_identity": flow.identity,
                      "geometry_reference_flow_identity": flow.reference_identity})
    return {"implementation": "fixed_frequency_mode_bank_v1",
            "code_revision": module_revision(sys.modules[__name__], coefficient_sources,
                                             artifacts, neural_field, component_field),
            "static_scene_identity": scene_manifest["static_scene_identity"],
            "foreground_identity": scene_manifest["foreground_identity"],
            "sources": [{k: s[k] for k in ("identity", "slot", "frequency_hz")} for s in sources],
            "views": views}


def build_mode_bank(*, sources, scene_dir, flows, output_dir):
    """Copy saved Phi unchanged; evaluate saved v16 fields once to cache rotations."""
    scene = load_static_scene(scene_dir, "cpu")
    contract = _bank_contract(sources, scene.manifest, flows)
    shape = (len(sources), scene.foreground.count, 3)
    points = scene.foreground.active()["means"].detach().cpu().numpy()
    support = np.empty(shape[:2], dtype=np.int8)
    observations = np.empty((*shape[:2], len(flows)), dtype=bool)
    modes = []
    destination = Path(output_dir)
    with _publish(destination) as work:
        phi = np.lib.format.open_memmap(work / "phi.npy", mode="w+", dtype=np.complex64, shape=shape)
        rotation = np.lib.format.open_memmap(work / "rotation.npy", mode="w+", dtype=np.complex64, shape=shape)
        try:
            progress = Progress("fixed mode bank", len(sources), unit="modes")
            for i, source in enumerate(sources):
                artifact = load_completed_modes(source["path"])
                slot = source["slot"]
                if artifact.manifest["completed_modes_identity"] != source["identity"]:
                    raise ValueError("Spatial mode identity changed during preparation")
                if not np.array_equal(artifact.arrays["g_points"], points):
                    raise ValueError("Spatial modes must use the exact static Gaussian order")
                displacement = artifact.arrays["phi"][slot]
                if artifact.rotation is None:
                    raise ValueError("v16 spatial modes must provide their saved-model rotation field")
                angular = artifact.rotation[slot]
                if any(v.dtype != np.complex64 or v.shape != shape[1:] or not np.isfinite(v).all()
                       for v in (displacement, angular)):
                    raise ValueError("Spatial displacement/rotation fields must be finite complex64 [G,3]")
                phi[i], rotation[i] = displacement, angular
                roles = artifact.arrays["support_class"][slot]
                observed = artifact.arrays["observation_view_mask"][slot]
                if (roles.shape != (shape[1],) or not np.issubdtype(roles.dtype, np.integer)
                        or np.any((roles < 0) | (roles > 3))
                        or observed.shape != (shape[1], len(flows)) or observed.dtype != np.bool_):
                    raise ValueError("Spatial mode support/observation domains differ")
                support[i], observations[i] = roles, observed
                original = artifact.manifest["modes"][slot]
                modes.append({"mode_slot": i, "candidate_index": i, "frequency_hz": original["frequency_hz"],
                              "source_mode_slot": slot, "source_candidate_index": original["candidate_index"]})
                del artifact
                progress.update(i + 1)
            phi.flush()
            rotation.flush()
        finally:
            phi._mmap.close()
            rotation._mmap.close()
        save_named_arrays(work / "support.npz", {"g_points": points, "support_class": support,
                                                 "observation_view_mask": observations})
        m = {"format": COMPLETED_MODES_FORMAT, "version": 17, "completion_method": BANK_METHOD,
             "contract": contract, "static_scene": str(resolve_path(scene_dir)),
             "static_scene_identity": scene.manifest["static_scene_identity"],
             "foreground_identity": scene.manifest["foreground_identity"],
             "sources": [{k: v for k, v in source.items() if k != "header"} for source in sources],
             "flow_artifacts": [str(flow.path) for flow in flows], "views": contract["views"], "modes": modes,
             "counts": {"modes": shape[0], "foreground_gaussians": shape[1], "views": len(flows)},
             "files": {name: {"file": f"{name}.npy", "dtype": "complex64", "shape": list(shape),
                               "sha256": sha256(work / f"{name}.npy")} for name in ("phi", "rotation")},
             "quality_gate": {"status": "fixed_mode_bank_candidate_unapproved"}}
        m["files"]["support"] = {"file": "support.npz", "sha256": sha256(work / "support.npz")}
        m["completed_modes_identity"] = _bank_identity(m)
        atomic_json(work / "manifest.json", m)
    return load_mode_bank(destination)  # mmap headers only; no source/network replay or evaluation.


def _initial_identity(m):
    return identity({k: m[k] for k in ("format", "version", "rendered_design_identity",
                    "completed_modes_identity", "modes", "views", "settings", "solver",
                    "quality_gate", "counts", "coordinates", "diagnostics")})


def load_initial_coordinates(path):
    root = resolve_path(path, strict=True)
    m = _json(root / "manifest.json")
    if (m.get("format") != DIRECT_COORDINATES_FORMAT or m.get("version") != 2
            or m.get("solver") != INITIAL_SOLVER or m.get("quality_gate") != INITIAL_GATE
            or m.get("direct_coordinates_identity") != _initial_identity(m)):
        raise ValueError("Unsupported or inconsistent SEA-RAFT initial coordinates")
    _validate_modes(m["modes"])
    DirectCoordinateConfig(**m["settings"]).validate()
    q = np.load(root / "coordinates.npy", allow_pickle=False)
    with np.load(root / "diagnostics.npz", allow_pickle=False) as data:
        scales = data["mode_pair_scales"]
    count, labels = 0, set()
    for index, view in enumerate(m["views"]):
        names, ref = view["frame_names"], view["reference_frame_index"]
        if (view["index"] != index or view["frame_offset"] != count or view["label"] in labels
                or len(names) != view["frame_count"] or len(set(names)) != len(names)
                or not 0 <= ref < len(names) or names[ref] != view["reference_frame_name"]
                or not np.isfinite(view["fps_hz"]) or view["fps_hz"] <= 0):
            raise ValueError("Initial-coordinate frame map differs")
        labels.add(view["label"])
        count += len(names)
    if (q.shape != (count, len(m["modes"])) or q.dtype != np.complex64 or not np.isfinite(q).all()
            or scales.shape != (len(m["views"]), len(m["modes"]))
            or not np.all(np.isfinite(scales) & (scales > 0))
            or m["coordinates"]["sha256"] != sha256(root / "coordinates.npy")
            or m["diagnostics"]["sha256"] != sha256(root / "diagnostics.npz")):
        raise ValueError("Initial coefficient arrays differ")
    if (m["counts"] != {"modes":q.shape[1], "frames":count, "views":len(m["views"])}
            or m["coordinates"]["shape"] != list(q.shape) or m["coordinates"]["dtype"] != "complex64"
            or m["coordinates"]["file"] != "coordinates.npy" or m["diagnostics"]["file"] != "diagnostics.npz"):
        raise ValueError("Initial coefficient inventory differs")
    return DirectModalCoordinatesArtifact(root, m, q, {"mode_pair_scales": scales})


def build_initial_coordinates(*, design, flows, output_dir, config):
    """Reuse the direct ridge solve without flow reconstruction/spectral evaluation."""
    config.validate()
    views, coordinates, scales = [], [], []
    frequencies = np.array([mode["frequency_hz"] for mode in design.manifest["modes"]])
    if len(flows) != len(design.manifest["views"]):
        raise ValueError("Initial-coordinate flow/view count differs")
    offset = 0
    for view, flow in zip(design.manifest["views"], flows):
        if (flow.identity != view["flow_identity"] or flow.manifest["fps_hz"] != view["fps_hz"]
                or len(flow.manifest["frames"]) != view["frame_count"]
                or flow.manifest["reference_frame_index"] != view["flow_reference_frame_index"]
                or flow.manifest["reference_frame_name"] != view["flow_reference_frame_name"]):
            raise ValueError("SEA-RAFT initialization differs from rendered-design sources")
        lo, n = view["sample_offset"], view["sample_count"]
        q, diagnostics = solve_direct_coordinates_view(
            design=design.design[lo:lo+n], pixels_xy=design.samples["sample_pixels_xy"][lo:lo+n],
            flow=flow.arrays.flow, reference_frame_index=flow.manifest["reference_frame_index"],
            fps_hz=flow.manifest["fps_hz"], frequencies_hz=frequencies, config=config, evaluate=False)
        coordinates.append(q)
        scales.append(diagnostics["mode_pair_scales"])
        views.append({"index": len(views), "label": view["label"], "flow_identity": flow.identity,
                      "shape_hw": view["shape_hw"], "fps_hz": view["fps_hz"], "frame_offset": offset,
                      "frame_count": len(q), "frame_names": flow.manifest["frame_names"],
                      "reference_frame_index": flow.manifest["reference_frame_index"],
                      "reference_frame_name": flow.manifest["reference_frame_name"], "sample_count": n})
        offset += len(q)
    q = np.concatenate(coordinates)
    diagnostics = {"mode_pair_scales": np.stack(scales)}
    destination = Path(output_dir)
    with _publish(destination) as work:
        np.save(work / "coordinates.npy", q, allow_pickle=False)
        save_named_arrays(work / "diagnostics.npz", diagnostics)
        m = {"format": DIRECT_COORDINATES_FORMAT, "version": 2, "solver": INITIAL_SOLVER,
             "rendered_design": str(design.path), "rendered_design_identity": design.manifest["rendered_design_identity"],
             "completed_modes_identity": design.manifest["completed_modes_identity"],
             "modes": design.manifest["modes"], "views": views, "settings": config.to_dict(),
             "flow_artifacts": [str(flow.path) for flow in flows], "quality_gate": INITIAL_GATE,
             "counts": {"modes": q.shape[1], "frames": len(q), "views": len(views)},
             "coordinates": {"file": "coordinates.npy", "shape": list(q.shape), "dtype": "complex64",
                             "sha256": sha256(work / "coordinates.npy")},
             "diagnostics": {"file": "diagnostics.npz", "sha256": sha256(work / "diagnostics.npz")}}
        m["direct_coordinates_identity"] = _initial_identity(m)
        atomic_json(work / "manifest.json", m)
    return DirectModalCoordinatesArtifact(destination, m, q, diagnostics)


def prepare_coefficient_inputs(*, scene, output_dir, expected_modes, status="completed_uniform60",
                               index_path=None, design_config=None, direct_config=None, resume=False):
    """Explicit preparation only: no RGB fitting, retraining, evaluation or Viewer."""
    design_config, direct_config = design_config or RenderedDesignConfig(), direct_config or DirectCoordinateConfig()
    design_config.validate()
    direct_config.validate()
    if isinstance(expected_modes, bool) or not isinstance(expected_modes, int) or expected_modes <= 0:
        raise ValueError("expected_modes must be a positive integer")
    destination = Path(output_dir).expanduser().resolve()
    experiments = asset_path(scene, "experiments")
    if destination.parent != experiments or destination.is_symlink():
        raise ValueError("Choose a new immediate child of this scene's experiments directory")
    existed = destination.exists()
    if existed and not resume:
        raise FileExistsError("Experiment exists; use a new name, or --resume for the identical preparation")
    sources = _mode_sources(index_path or library_root() / scene / "results/index.json", status, expected_modes)
    scene_dir = asset_path(scene, "static")
    labels = [view["label"] for view in sources[0]["header"]["views"]]
    flows = [load_coordinate_flow(asset_path(scene, f"flow{i+1}")) for i in range(len(labels))]
    bank_contract = _bank_contract(sources, _json(scene_dir / "manifest.json"), flows)
    contract = {"format": "modal_gaussians.coefficient_preparation", "version": 1,
                "scene": scene, "bank": bank_contract, "design": design_config.to_dict(),
                "direct": direct_config.to_dict(), "evaluation": "not_run"}
    from modal_gaussians import direct_coordinates, rendered_design
    from modal_gaussians.motion.common import projection
    contract["operator_revision"] = module_revision(direct_coordinates, rendered_design, projection)
    state_path = destination / "preparation.json"
    if existed:
        if not state_path.is_file() or _json(state_path)["contract"] != contract:
            raise ValueError("Existing experiment has a different preparation contract")
    else:
        destination.mkdir(parents=True)
        atomic_json(state_path, {"contract": contract, "completed": []})
    state = {"contract": contract, "completed": []}
    bank_dir, design_dir, direct_dir = (destination / n for n in ("mode_bank", "rendered_design", "direct_coordinates"))
    if bank_dir.exists():
        bank = load_mode_bank(bank_dir)
        if bank.manifest["contract"] != bank_contract:
            raise ValueError("Existing mode bank differs from requested sources")
    else:
        bank = build_mode_bank(sources=sources, scene_dir=scene_dir, flows=flows, output_dir=bank_dir)
    state["completed"].append("mode_bank")
    atomic_json(state_path, state)
    if design_dir.exists():
        design = load_rendered_modal_design(design_dir)
        if (design.manifest["completed_modes_identity"] != bank.manifest["completed_modes_identity"]
                or design.manifest["settings"] != design_config.to_dict()):
            raise ValueError("Existing design differs from mode bank/configuration")
    else:
        by_path = {flow.path: flow for flow in flows}
        design = build_rendered_modal_design_artifact(
            scene_dir=scene_dir, completed_modes_dir=bank_dir,
            views=[RenderedDesignViewInput(label, flow.path) for label, flow in zip(labels, flows)],
            output_dir=design_dir, config=design_config, validated_completed=bank,
            flow_loader=lambda path: by_path[resolve_path(path)])
    if (design.manifest["rendered_design_identity"] != identity(rendered_design._identity_payload(design.manifest))
            or design.manifest["modes"] != bank.manifest["modes"]
            or design.manifest["static_scene_identity"] != bank.manifest["static_scene_identity"]
            or design.manifest["foreground_identity"] != bank.manifest["foreground_identity"]):
        raise ValueError("Rendered design metadata is not bound to this mode bank")
    state["completed"].append("rendered_design")
    atomic_json(state_path, state)
    if direct_dir.exists():
        initial = load_initial_coordinates(direct_dir)
        if (initial.manifest["rendered_design_identity"] != design.manifest["rendered_design_identity"]
                or initial.manifest["settings"] != direct_config.to_dict()):
            raise ValueError("Existing initialization differs from design/configuration")
    else:
        initial = build_initial_coordinates(design=design, flows=flows, output_dir=direct_dir, config=direct_config)
    state["completed"].append("direct_coordinates")
    atomic_json(state_path, state)
    return initial
