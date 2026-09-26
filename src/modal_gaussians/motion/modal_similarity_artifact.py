"""Publish soft modal-similarity graphs from a full KNN candidate cache."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from modal_gaussians.common.scene_store import resolve_path
import tempfile

import numpy as np
from PIL import Image

from modal_gaussians.common.cache import Timings, atomic_json, identity, load_entry, sha256
from modal_gaussians.spectrum.modes import TRANSFORM_CONVENTION
from modal_gaussians.common.numpy_io import save_named_arrays
from modal_gaussians.geometry.scene import cameras_from_scene_manifest
from modal_gaussians.motion.geometry_graph import GeometryGraph
from modal_gaussians.motion.modal_similarity import ModalSimilarityConfig, build_modal_similarity_graph


def _manifest(path):
    return json.loads((resolve_path(path) / "manifest.json").read_text(encoding="utf-8"))


def _modal_view(path, flow, view, frequency, *, read_mask=True, static_scene_identity=None):
    """Bind the experimental dense field to the prepared reference coordinates."""
    root = resolve_path(path, strict=True)
    manifest = _manifest(root)
    frozen = flow["manifest"]
    if (manifest.get("format") != "modal_gaussians.spectrum_selected_frequency"
            or manifest.get("version") != 2 or manifest.get("status") != "complete"
            or manifest.get("transform") != TRANSFORM_CONVENTION
            or manifest.get("flow_direction") != "reference_to_frame"
            or manifest.get("flow_units") != "input_pixels"
            or manifest.get("modes_file") != "modal_image.npy"):
        raise ValueError(f"Unsupported selected-frequency modal image: {root}")
    if (not math.isclose(float(manifest["frequency_hz"]), frequency, rel_tol=0, abs_tol=1e-9)
            or manifest["frames"] != frozen["frame_names"]
            or manifest["fps_hz"] != frozen["fps_hz"]
            or resolve_path(manifest["stabilization_source"]) != resolve_path(flow["path"])
            or resolve_path(manifest["images"]) != resolve_path(frozen["inputs"]["sequence"]["image_directory"])):
        raise ValueError(f"Modal image differs from prepared frequency/reference coordinates: {view['label']}")
    from modal_gaussians.flow.reference_selection import motion_reference
    reference_name, _ = motion_reference(manifest, frozen, view=view, static_scene_identity=static_scene_identity)
    spectrum = manifest.get("spectrum_source")
    if not isinstance(spectrum, dict):
        raise ValueError("Spectrum selection requires its cache source")
    length, index = spectrum.get("fft_length"), spectrum.get("bin_index")
    step, cache_id, cache_path = (spectrum.get("frequency_step_hz"),
                                 spectrum.get("identity"), spectrum.get("path"))
    if (type(length) is not int or length < max(3, len(manifest["frames"]))
            or type(index) is not int or not 0 < index <= length // 2
            or type(step) not in (int, float) or not math.isfinite(step) or step <= 0
            or not math.isclose(step, float(manifest["fps_hz"]) / length, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(float(manifest["frequency_hz"]), index * step, rel_tol=0, abs_tol=1e-9)
            or not isinstance(cache_id, str) or len(cache_id) != 64
            or any(c not in "0123456789abcdef" for c in cache_id)
            or not isinstance(cache_path, str) or not Path(cache_path).is_absolute()):
        raise ValueError("Spectrum selection cache/grid contract is invalid")
    stabilized = frozen.get("stabilized_sequence")
    if stabilized is not None:
        sequence = resolve_path(flow["path"]) / "stabilized_sequence"
        image_dir, mask_dir = sequence / "images", sequence / "masks"
    else:
        sequence = frozen["inputs"]["sequence"]
        image_dir = resolve_path(sequence["image_directory"])
        mask_dir = resolve_path(sequence["mask_directory"]) if read_mask else None
    reference = reference_name + ".png"
    inference_images = manifest.get("inference_images")
    if (inference_images is None or resolve_path(inference_images) != image_dir.resolve()
            or resolve_path(manifest["reference_image"]) != (image_dir / reference).resolve()):
        raise ValueError(f"Modal reference image differs: {view['label']}")
    field = np.load(root / "modal_image.npy", mmap_mode="r", allow_pickle=False)
    expected = (1, *view["shape_hw"], 2)
    if field.shape != expected or field.dtype != np.complex64:
        raise ValueError(f"Modal image must be complex64 {expected}: {root}")
    mask_path = mask_dir / reference if read_mask else None
    if read_mask:
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L")) > 0
        if mask.shape != tuple(view["shape_hw"]):
            raise ValueError(f"Reference mask shape differs: {mask_path}")
    else:
        mask = np.ones(view["shape_hw"], dtype=bool)
    valid = np.load(root/'valid_mask.npy', allow_pickle=False)
    if (valid.dtype != bool or valid.shape != mask.shape or not valid.any()
            or sha256(root/'valid_mask.npy') != manifest['valid_mask_sha256']):
        raise ValueError('Modal image validity differs')
    return field[0], mask & valid, {"label": view["label"], "path": str(root),
        "manifest_identity": identity(manifest), "manifest": manifest,
        "reference_mask": str(mask_path) if read_mask else None, "prepared_flow_identity": flow["identity"]}


def build_modal_similarity_graph_artifact(*, prepared_dir, geometry_graph_dir, views,
                                        frequency_hz, output_dir, config=None, command=None):
    """Publish a graph-only experiment; do not build controls or run training."""
    settings = config or ModalSimilarityConfig()
    settings.validate()
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Frequency must be finite and positive")
    output = resolve_path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite graph experiment: {output}")
    prepared_path = resolve_path(prepared_dir, strict=True)
    graph_path = resolve_path(geometry_graph_dir, strict=True)
    timer = Timings()
    with timer.stage("modal_similarity_inputs"):
        prepared = _manifest(prepared_path)
        if prepared.get("format") != "modal_gaussians.neural_prepared" or prepared.get("version") != 3:
            raise ValueError("Unsupported neural preparation")
        source = prepared["source"]
        graph_manifest = _manifest(graph_path)
        contract = graph_manifest.get("contract", {})
        if (contract.get("implementation") != "neural_geometry_cache_v2"
                or contract.get("config", {}).get("graph_edge_filter") != "none"
                or contract.get("foreground") != source["foreground_identity"]
                or graph_path.name != identity(contract)):
            raise ValueError("Expected an unfiltered KNN cache for the prepared foreground")
        arrays = load_entry(graph_path.parent, contract)
        if arrays is None:
            raise FileNotFoundError(f"Missing geometry graph: {graph_path}")
        graph = GeometryGraph.from_dict(arrays)
        requested = dict(views)
        known_views = {v["label"]: v for v in source["views"]}
        if (not requested or len(requested) != len(views)
                or not requested.keys() <= known_views.keys()):
            raise ValueError("--view labels must be unique labels from the prepared scene")
        # Preserve prepared camera order, independent of command-line order.
        selected = [v for v in source["views"] if v["label"] in requested]
        scene_path = Path(source["static_scene"])
        scene = _manifest(scene_path)
        if (scene.get("foreground_identity") != source["foreground_identity"]
                or scene.get("static_scene_identity") != source["static_scene_identity"]):
            raise ValueError("Prepared scene identity differs")
        visible_subject = scene.get("partition", {}).get("method") == "manual_subject_selection_v3"
        cameras = {c.label: c for c in cameras_from_scene_manifest(scene) if c.role == "reference"}
        flows = {f["identity"]: f for f in prepared["flows"]}
        fields, masks, depths, alphas, selected_cameras, provenance = [], [], [], [], [], []
        with np.load(prepared_path / "arrays.npz", allow_pickle=False) as archive:
            if not np.array_equal(graph.points, archive["o_g_points"]):
                raise ValueError("Geometry cache differs from prepared Gaussian order or positions")
            for view in selected:
                camera = cameras[view["label"]]
                if camera.to_manifest_record()["camera_identity"] != view["camera_identity"]:
                    raise ValueError(f"Prepared camera differs: {view['label']}")
                field, mask, record = _modal_view(requested[view["label"]],
                    flows[view["flow_identity"]], view, frequency_hz, read_mask=not visible_subject,
                    static_scene_identity=source["static_scene_identity"])
                selected_reference = record["manifest"].get("reference_selection", {}).get("identity")
                if selected_reference != view.get("motion_reference", {}).get("selection_identity"):
                    raise ValueError("Graph modal reference differs from preparation; prepare these observations first")
                if visible_subject:
                    key = f"v{view['index']}_subject_mask"
                    if key not in archive.files:
                        raise ValueError("Manual subject graph requires rebuilt visible-subject observations")
                    mask = archive[key]
                    if mask.dtype != bool or mask.shape != tuple(view["shape_hw"]):
                        raise ValueError("Prepared visible-subject mask has an invalid shape or dtype")
                    record.update(mask_source="prepared.visible_subject_contribution", mask_array=key)
                fields.append(field)
                masks.append(mask)
                depths.append(archive[f"v{view['index']}_depth"])
                alphas.append(archive[f"v{view['index']}_alpha"])
                selected_cameras.append(camera)
                provenance.append(record)
        endpoint = np.asarray([source['endpoint_relative_tolerances'][v['index']] for v in selected], dtype=np.float64)
    with timer.stage("modal_similarity_graph"):
        result, diagnostics, summary = build_modal_similarity_graph(graph,
            Ks=np.stack([c.K.cpu().numpy() for c in selected_cameras]),
            world_to_cameras=np.stack([c.world_to_camera.cpu().numpy() for c in selected_cameras]),
            radial_coefficients=np.array([c.radial_distortion for c in selected_cameras]),
            rendered_depths=depths, rendered_alphas=alphas, endpoint_thresholds=endpoint,
            modal_fields=fields, masks=masks, config=settings)
        retained_evidence = diagnostics["candidate_modal_evidence"]
        summary["unknown_edges_retained"] = int(np.all(retained_evidence == 0, axis=1).sum())
        for row, view in zip(summary["views"], selected):
            row["label"] = view["label"]
    manifest = {"format": "modal_gaussians.modal_similarity_graph", "version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": list(command or []),
        "scene_dir": str(scene_path), "foreground_identity": source["foreground_identity"],
        "static_scene_identity": source["static_scene_identity"],
        "graph_file": "graph.npz", "evidence_file": "edge_evidence.npz",
        "frequency_hz": frequency_hz,
        "config": {"graph_neighbors": contract["config"]["graph_neighbors"],
                   "graph_max_distance": contract["config"]["graph_max_distance"],
                   "graph_edge_filter": "modal-similarity", "modal_similarity": asdict(settings)},
        "source": {"prepared": str(prepared_path), "prepared_identity": prepared["prepared_identity"],
                   "geometry_graph": str(graph_path), "geometry_contract": contract,
                   "depth_tolerances_from": "prepared.endpoint_relative_tolerances",
                   "endpoint_relative_tolerances": [float(t) if np.isfinite(t) else None for t in endpoint],
                   "views": provenance},
        "summary": summary, "validation": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with timer.stage("modal_similarity_publish"):
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-writing-", dir=output.parent))
        save_named_arrays(temporary / "graph.npz", result.as_dict())
        save_named_arrays(temporary / "edge_evidence.npz", diagnostics)
        atomic_json(temporary / "manifest.json", manifest)
        from modal_gaussians.common.cache import publish_directory
        publish_directory(temporary, output)
    timer.save(output / "timings.json")
    return manifest
