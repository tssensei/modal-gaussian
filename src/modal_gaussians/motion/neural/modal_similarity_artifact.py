"""Publish soft modal-similarity graphs from a full KNN candidate cache."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from modal_gaussians.iteration_cache import Timings, atomic_json, identity, load_entry
from modal_gaussians.modes import TRANSFORM_CONVENTION
from modal_gaussians.numpy_io import save_named_arrays
from modal_gaussians.static import cameras_from_scene_manifest
from .geometry_graph import GeometryGraph, depth_thresholds_from_manifest
from .modal_similarity import ModalSimilarityConfig, build_modal_similarity_graph


def _manifest(path):
    return json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))


def _modal_view(path, flow, view, frequency, *, read_mask=True):
    """Bind the experimental dense field to the prepared reference coordinates."""
    root = Path(path).expanduser().resolve(strict=True)
    manifest = _manifest(root)
    frozen = flow["manifest"]
    if (manifest.get("format") not in ("modal_gaussians.sea_raft_selected_frequency_experiment",
                                       "modal_gaussians.spectrum_selected_frequency")
            or manifest.get("version") != 1 or manifest.get("status") != "complete"
            or manifest.get("transform") != TRANSFORM_CONVENTION
            or manifest.get("flow_direction") != "reference_to_frame"
            or manifest.get("flow_units") != "input_pixels"
            or manifest.get("modes_file") != "modal_image.npy"):
        raise ValueError(f"Unsupported selected-frequency modal image: {root}")
    if (not math.isclose(float(manifest["frequency_hz"]), frequency, rel_tol=0, abs_tol=1e-9)
            or manifest["frames"] != frozen["frame_names"]
            or manifest["fps_hz"] != frozen["fps_hz"]
            or manifest["reference_frame_name"] != frozen["reference_frame_name"]
            or manifest["reference_frame_index"] != frozen["reference_frame_index"]
            or Path(manifest["stabilization_source"]).resolve() != Path(flow["path"]).resolve()
            or Path(manifest["images"]).resolve() != Path(frozen["inputs"]["sequence"]["image_directory"]).resolve()):
        raise ValueError(f"Modal image differs from prepared frequency/reference coordinates: {view['label']}")
    if manifest["format"] == "modal_gaussians.spectrum_selected_frequency":
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
        sequence = Path(flow["path"]) / stabilized["path"]
        image_dir, mask_dir = sequence / "images", sequence / "masks"
    else:
        sequence = frozen["inputs"]["sequence"]
        image_dir = Path(sequence["image_directory"])
        mask_dir = Path(sequence["mask_directory"]) if read_mask else None
    reference = frozen["reference_frame_name"] + ".png"
    inference_images = manifest.get("inference_images") or manifest.get("stabilized_images")
    if (inference_images is None or Path(inference_images).resolve() != image_dir.resolve()
            or Path(manifest["reference_image"]).resolve() != (image_dir / reference).resolve()):
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
    return field[0], mask, {"label": view["label"], "path": str(root),
        "manifest_identity": identity(manifest), "manifest": manifest,
        "reference_mask": str(mask_path) if read_mask else None, "prepared_flow_identity": flow["identity"]}


def build_modal_similarity_graph_artifact(*, prepared_dir, geometry_graph_dir, views,
                                        frequency_hz, output_dir, config=None, command=None):
    """Publish a graph-only experiment; do not build controls or run training."""
    settings = config or ModalSimilarityConfig()
    settings.validate()
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Frequency must be finite and positive")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite graph experiment: {output}")
    prepared_path = Path(prepared_dir).expanduser().resolve(strict=True)
    graph_path = Path(geometry_graph_dir).expanduser().resolve(strict=True)
    timer = Timings()
    with timer.stage("modal_similarity_inputs"):
        prepared = _manifest(prepared_path)
        if prepared.get("format") != "modal_gaussians.neural_prepared" or prepared.get("version") != 1:
            raise ValueError("Unsupported neural preparation")
        source = prepared["source"]
        graph_manifest = _manifest(graph_path)
        contract = graph_manifest.get("contract", {})
        if (contract.get("implementation") != "neural_geometry_cache_v1"
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
        visible_subject = scene.get("partition", {}).get("method") == "manual_subject_selection_v1"
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
                    flows[view["flow_identity"]], view, frequency_hz, read_mask=not visible_subject)
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
        # Read tolerances only: the old graph's edges and depth-jump test are unused.
        endpoint, _ = depth_thresholds_from_manifest(_manifest(source["observed_structure_graph"]),
                                                    [v["label"] for v in selected])
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
    manifest = {"format": "modal_gaussians.modal_similarity_graph", "version": 1,
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
                   "depth_tolerances_from": source["observed_structure_graph"],
                   "endpoint_relative_tolerances": [float(t) if np.isfinite(t) else None for t in endpoint],
                   "views": provenance},
        "summary": summary, "validation": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with timer.stage("modal_similarity_publish"):
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-writing-", dir=output.parent))
        save_named_arrays(temporary / "graph.npz", result.as_dict())
        save_named_arrays(temporary / "edge_evidence.npz", diagnostics)
        atomic_json(temporary / "manifest.json", manifest)
        os.rename(temporary, output)
    timer.save(output / "timings.json")
    return manifest
