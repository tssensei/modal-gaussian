"""Strict neural artifact decoding and historical format replay, separate from training."""
from __future__ import annotations
import json
from dataclasses import fields
import math
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from . import neural_modes as nm

def _validate_arrays(arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any],
                     *, check_field_geometry: bool = True) -> None:
    """Check domains, support semantics and sparse geometry before loading weights."""
    from modal_gaussians.motion.neural.geometry_graph import GeometryGraph, ControlGraph
    config = nm.NeuralModesConfig.from_dict(manifest["config"])
    training_fill = config.training_fragment_config is not None
    required = {
        "phi", "sample_prediction", "support_class", "alphas", "alpha_identifiable_mask",
        "observation_view_mask", "sample_pixels_xy", "sample_view_index", "view_sample_offsets",
        "sample_confidence", "sample_target", "mode_view_rms", "mode_view_loss_scale",
        "measurement_rms_floor", "amplitude_scale", "scene_scale", "contribution_mass",
        "contribution_threshold", "sample_projection_sensitivity",
    }
    required.update("g_" + f.name for f in fields(GeometryGraph))
    required.update("c_" + f.name for f in fields(ControlGraph))
    if config.rigidity_refinement is not None:
        required.update(("rigidity_edge_factor", "rigidity_edge_coherence", "rigidity_edge_evidence"))
    if training_fill:
        from modal_gaussians.motion.neural.strategies import array_names
        required.update(array_names(config.training_fragment_config))
    selected_slots = None
    if "source_modes" in manifest or "mode_selection" in manifest:
        selected_slots = nm._source_mode_slots(manifest)
        required.update(nm.PREFIX_NORMALIZATION_ARRAYS)
    if set(arrays) != required:
        raise ValueError("Neural array inventory does not match its schema")
    K, G, V = (int(manifest["counts"][name]) for name in ("modes", "foreground_gaussians", "views"))
    if min(K, G, V) <= 0:
        raise ValueError("Neural artifact counts must be positive")
    if config.rigidity_refinement is not None:
        for name in ("rigidity_edge_factor", "rigidity_edge_coherence", "rigidity_edge_evidence"):
            value = arrays[name]
            if value.shape != (K, len(arrays["g_edge_index"])) or value.dtype != np.float32:
                raise ValueError(f"Invalid rigidity refinement array {name}")
            if np.any(value < 0) or np.any(value > 1):
                raise ValueError(f"Rigidity refinement {name} must lie in [0,1]")
    if selected_slots is not None and len(selected_slots) != K:
        raise ValueError("Neural prefix mode count differs from selected slots")
    for name, value in arrays.items():
        if value.dtype.kind not in "biufc" or (value.dtype.kind in "fc" and not np.isfinite(value).all()):
            raise ValueError(f"Neural array {name} must be finite numeric data")
    shapes = {"phi": (K, G, 3), "support_class": (K, G), "observation_view_mask": (K, G, V),
              "alphas": (K, V), "alpha_identifiable_mask": (K, V), "amplitude_scale": (K,),
              "mode_view_rms": (K, V), "mode_view_loss_scale": (K, V),
              "g_points": (G, 3), "g_component_index": (G,), "contribution_mass": (G, V),
              "view_sample_offsets": (V + 1,), "contribution_threshold": (V,)}
    for name, shape in shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError(f"Neural {name} shape must be {shape}")
    if arrays["phi"].dtype != np.complex64 or arrays["alphas"].dtype != np.complex64:
        raise ValueError("Neural complex arrays must be complex64")
    for name in ("sample_target", "sample_prediction"):
        if arrays[name].dtype != np.complex64:
            raise ValueError(f"Neural {name} must be complex64")
    for name in ("observation_view_mask", "alpha_identifiable_mask"):
        if arrays[name].dtype != np.bool_:
            raise ValueError(f"Neural {name} must be boolean")
    if arrays["support_class"].dtype != np.int8 or not np.isin(arrays["support_class"], (0, 1, 2, 3) if training_fill else (0, 1, 2)).all():
        raise ValueError("Neural support classes are invalid")
    if "a_view_visible" in arrays and arrays["a_view_visible"].shape != (G, V):
        raise ValueError("Surface attachment view domain differs from observations")
    if not np.all(arrays["alphas"][:, 0] == 1) or not arrays["alpha_identifiable_mask"][:, 0].all():
        raise ValueError("Neural reference alpha gauge must remain one and identifiable")
    if arrays["scene_scale"].shape != () or arrays["measurement_rms_floor"].shape != ():
        raise ValueError("Neural scalar normalization arrays have invalid shapes")
    if np.any(arrays["amplitude_scale"] <= 0) or float(arrays["scene_scale"]) <= 0:
        raise ValueError("Neural normalization scales must be positive")
    if np.any(arrays["mode_view_loss_scale"] <= 0):
        raise ValueError("Neural observation scales must be positive")
    expected_obs = ((arrays["contribution_mass"] > arrays["contribution_threshold"][None])[None]
                    & arrays["alpha_identifiable_mask"][:, None, :])
    if not np.array_equal(arrays["observation_view_mask"], expected_obs):
        raise ValueError("Neural observation support differs from renderer contribution")
    if training_fill:
        from modal_gaussians.motion.neural.strategies import training_support_roles
        roles, supported = training_support_roles(arrays, expected_obs)
    else:
        roles, supported = nm.support_roles(arrays["g_component_index"], expected_obs)
    if not np.array_equal(roles, arrays["support_class"]):
        raise ValueError("Neural support classes disagree with geometry connectivity")
    if np.any(arrays["phi"][~supported] != 0):
        raise ValueError("Unresolved neural components must have exactly zero motion")
    offsets = arrays["view_sample_offsets"]
    if offsets.dtype != np.int64 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("Neural observation offsets are invalid")
    S = int(offsets[-1])
    for name, shape in {"sample_target": (K, S, 2), "sample_prediction": (K, S, 2),
                        "sample_confidence": (S,), "sample_view_index": (S,),
                        "sample_pixels_xy": (S, 2), "sample_projection_sensitivity": (S,)}.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Neural {name} observation shape differs")
    if np.any(arrays["sample_confidence"] <= 0):
        raise ValueError("Neural sample confidence must be positive")
    if np.any(arrays["sample_confidence"] > 1) or np.any(arrays["contribution_mass"] < 0):
        raise ValueError("Neural alpha confidence or contribution mass is invalid")
    if np.any(arrays["sample_projection_sensitivity"] < 0):
        raise ValueError("Neural projection sensitivity must be nonnegative")
    expected_rms = np.zeros((K, V), dtype=np.float64)
    for view in range(V):
        lower, upper = offsets[view:view + 2]
        if np.any(arrays["sample_view_index"][lower:upper] != view):
            raise ValueError("Neural sample view order differs")
        pixels = arrays["sample_pixels_xy"][lower:upper]
        height, width = manifest["views"][view]["shape_hw"]
        if pixels.dtype != np.int64 or np.any(pixels < 0) or np.any(pixels >= [width, height]):
            raise ValueError("Neural sampled pixels are out of bounds")
        confidence = arrays["sample_confidence"][lower:upper].astype(np.float64)
        energy = np.sum(np.abs(arrays["sample_target"][:, lower:upper].astype(np.complex128)) ** 2, axis=-1)
        expected_rms[:, view] = np.sqrt(np.sum(energy * confidence[None], axis=1) / confidence.sum())
        if np.any(arrays["sample_prediction"][~arrays["alpha_identifiable_mask"][:, view], lower:upper] != 0):
            raise ValueError("Excluded neural views must have explicit zero predictions")
    normalization_rms, normalization_identifiable = expected_rms, arrays["alpha_identifiable_mask"]
    if selected_slots is not None:
        normalization_rms = arrays["normalization_source_mode_view_rms"]
        normalization_identifiable = arrays["normalization_source_alpha_identifiable_mask"]
        original_shape = (len(manifest["source_modes"]), V)
        if (normalization_rms.shape != original_shape or normalization_rms.dtype != np.float64
                or normalization_identifiable.shape != original_shape or normalization_identifiable.dtype != np.bool_
                or np.any(normalization_rms < 0) or not normalization_identifiable[:, 0].all()
                or not np.array_equal(normalization_identifiable[selected_slots], arrays["alpha_identifiable_mask"])
                or not np.allclose(normalization_rms[selected_slots], expected_rms, rtol=1e-10, atol=1e-12)):
            raise ValueError("Neural prefix normalization differs from original source mode scope")
    positive = normalization_rms[normalization_identifiable & (normalization_rms > 0)]
    expected_floor = max(config.energy_floor_fraction * (float(np.median(positive)) if len(positive) else 1.0), 1e-12)
    if (not np.allclose(arrays["mode_view_rms"], expected_rms, rtol=1e-10, atol=1e-12)
            or not math.isclose(float(arrays["measurement_rms_floor"]), expected_floor, rel_tol=1e-10)
            or not np.allclose(arrays["mode_view_loss_scale"], np.maximum(expected_rms, expected_floor), rtol=1e-10, atol=1e-12)):
        raise ValueError("Neural observation normalization differs from fixed measurements")
    graph = GeometryGraph.from_dict({name[2:]: value for name, value in arrays.items() if name.startswith("g_")}, validate=True)
    if config.graph_edge_filter == "none":
        from modal_gaussians.motion.neural.geometry_graph import EVIDENCE_SPATIAL_PRIOR, _mutual_knn
        expected_edges, _ = _mutual_knn(graph.points.astype(np.float64), nm._geometry_config(config))
        expected_weights = 1.0 / np.sqrt(graph.degree[expected_edges[:, 0]].astype(np.float64) * graph.degree[expected_edges[:, 1]])
        if (not np.array_equal(graph.edge_index, expected_edges)
                or not np.array_equal(graph.candidate_edge_index, expected_edges)
                or not np.all(graph.edge_evidence_kind == EVIDENCE_SPATIAL_PRIOR)
                or np.any(graph.edge_view_evidence) or np.any(graph.candidate_view_evidence)
                or np.any(graph.node_visible_view_mask)
                or not np.allclose(graph.edge_weight, expected_weights, rtol=1e-12, atol=1e-14)):
            raise ValueError("Neural KNN-only graph differs from its complete spatial candidate contract")
    elif np.any(graph.edge_evidence_kind == 2):
        raise ValueError("Depth-filtered neural configuration cannot contain unfiltered spatial-prior edges")
    control = ControlGraph.from_dict({name[2:]: value for name, value in arrays.items() if name.startswith("c_")}, validate=True)
    control_graph, control_count = graph, G
    if training_fill:
        from modal_gaussians.motion.neural.strategies import validate_training_controls
        from modal_gaussians.motion.common.graph_ops import host_subgraph
        validate_training_controls(arrays, graph, geometry_config=nm._geometry_config(config),
                                   fragment_config=config.training_fragment_config,
                                   scene_scale=float(arrays["scene_scale"]))
        control_graph = host_subgraph(graph, arrays["t_host_gaussian_index"])
        control_count = len(control_graph.points)
    if (len(control.interpolation_indptr) != control_count + 1 or len(control.owner) != control_count
            or np.any(control.control_point_index < 0) or np.any(control.control_point_index >= control_count)):
        raise ValueError("Neural control and Gaussian index domains differ")
    if not np.array_equal(control.positions, control_graph.points[control.control_point_index]):
        raise ValueError("Neural control locations differ from foreground indices")
    if not np.array_equal(control.component_index, control_graph.component_index[control.control_point_index]):
        raise ValueError("Neural control components differ from Gaussian geometry")
    interpolation_rows = np.repeat(np.arange(control_count), np.diff(control.interpolation_indptr))
    if np.any(control_graph.component_index[interpolation_rows] != control.component_index[control.interpolation_indices]):
        raise ValueError("Neural interpolation crosses disconnected geometry")
    if not math.isclose(float(control.scene_scale), float(arrays["scene_scale"]), rel_tol=1e-7):
        raise ValueError("Neural geometry length scales differ")
    view_sensitivity = np.array([
        np.average(arrays["sample_projection_sensitivity"][offsets[v]:offsets[v + 1]],
                   weights=arrays["sample_confidence"][offsets[v]:offsets[v + 1]]) for v in range(V)
    ])
    for mode in range(K):
        valid = arrays["alpha_identifiable_mask"][mode]
        denominator = float(np.sum(np.abs(arrays["alphas"][mode, valid].astype(np.complex128)) ** 2 * view_sensitivity[valid]))
        if denominator <= 0 or not np.any(expected_obs[mode].any(axis=1) & supported[mode]):
            raise ValueError("Neural mode has no effective observation sensitivity")
        expected_scale = max(1e-6 * float(arrays["scene_scale"]), math.sqrt(float(np.sum(expected_rms[mode, valid] ** 2)) / denominator))
        if not math.isclose(float(arrays["amplitude_scale"][mode]), expected_scale, rel_tol=1e-7):
            raise ValueError("Neural amplitude scale differs from fixed projection sensitivity")
        if check_field_geometry:
            nm._field_geometry(arrays, mode)


def _check_persisted_sources(manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    source, scene, _, _, _, alignment, dense = nm._load_sources(
        scene_dir=manifest["static_scene"], topology_dir=manifest["topology"],
        measurements_dir=manifest["measurements"], graph_dir=manifest["observed_structure_graph"],
        alignment_from=manifest["alignment_from"],
    )
    if nm._source_identity(source) != manifest["source_identity"]:
        raise ValueError("Neural artifact source identities differ")
    slots = nm._source_mode_slots(manifest)
    if "source_modes" in manifest and manifest["source_modes"] != source["modes"]:
        raise ValueError("Neural prefix original source modes differ")
    if "source_modes" not in manifest and manifest["modes"] != source["modes"]:
        raise ValueError("Neural complete source modes differ")
    for name in ("alphas", "alpha_identifiable_mask"):
        if not np.array_equal(arrays[name], alignment.arrays[name][slots]):
            raise ValueError(f"Neural fixed {name} differs from alignment source")
    if "source_modes" in manifest and not np.array_equal(
            arrays["normalization_source_alpha_identifiable_mask"], alignment.arrays["alpha_identifiable_mask"]):
        raise ValueError("Neural prefix normalization identifiability differs from alignment source")
    points = scene.foreground.active()["means"].detach().cpu().numpy()
    if not np.array_equal(points, arrays["g_points"]):
        raise ValueError("Neural foreground point order differs from static scene")
    if "a_covariance" in arrays:
        from modal_gaussians.motion.legacy.neural.surface_attachments import scene_covariance
        # Float32 CPU/CUDA activations differ by a few ulps. Off-diagonal
        # covariance entries can nearly cancel, so scale error per matrix,
        # not relative to those near-zero entries.
        expected_covariance = scene_covariance(scene)
        covariance_error = np.linalg.norm(expected_covariance - arrays["a_covariance"], axis=(1, 2))
        covariance_scale = np.linalg.norm(expected_covariance, axis=(1, 2))
        if np.any(covariance_error > 2e-6 * covariance_scale + 1e-15):
            raise ValueError("Surface attachment covariance differs from static scene")
        visible = arrays["contribution_mass"] > arrays["contribution_threshold"][None]
        if np.any(arrays["a_view_visible"] & ~visible):
            raise ValueError("Surface attachment visibility has no observed contribution")
    for view, modes in enumerate(dense.view_modes):
        lo, hi = arrays["view_sample_offsets"][view:view + 2]
        pixels = arrays["sample_pixels_xy"][lo:hi]
        target = np.asarray(modes[:, pixels[:, 1], pixels[:, 0], :])
        if not np.array_equal(target[slots], arrays["sample_target"][:, lo:hi]):
            raise ValueError("Neural target differs from dense modal image source")
        if "source_modes" in manifest:
            confidence = arrays["sample_confidence"][lo:hi].astype(np.float64)
            energy = np.sum(np.abs(target.astype(np.complex128)) ** 2, axis=-1)
            rms = np.sqrt(np.sum(energy * confidence[None], axis=1) / confidence.sum())
            if not np.allclose(rms, arrays["normalization_source_mode_view_rms"][:, view], rtol=1e-10, atol=1e-12):
                raise ValueError("Neural prefix normalization RMS differs from full dense modal source")


def load_neural_completed_modes(path: str | Path, *, validate: bool = False) -> nm.NeuralModesArtifact:
    """Read saved modes; exhaustive consistency checks are explicit diagnostics only."""
    from modal_gaussians.motion.neural.neural_field import evaluate_model
    root = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != nm.COMPLETED_MODES_FORMAT or type(manifest.get("version")) is not int
            or manifest["version"] not in (8, 10, 11, 12, 14, 16)):
        raise ValueError("Unsupported neural completed-mode artifact")
    config = nm.NeuralModesConfig.from_dict(manifest["config"])
    from modal_gaussians.motion.neural.strategies import artifact_contract
    training_fill = config.training_fragment_config is not None
    version, method = artifact_contract(config.training_fragment_config)
    if manifest["version"] != version or manifest.get("completion_method") != method:
        raise ValueError("Unsupported neural artifact version/method/configuration combination")
    if validate:
        if manifest.get("semantics") != nm._semantics(config) or manifest.get("quality_gate") != nm.QUALITY_GATE:
            raise ValueError("Neural completed-mode semantics differ")
        if manifest.get("arrays_file") != nm.ARRAYS_FILENAME or manifest.get("networks_file") != nm.MODELS_FILENAME:
            raise ValueError("Neural artifact file names differ")
        if nm._sha256(root / nm.ARRAYS_FILENAME) != manifest["arrays_file_sha256"] or nm._sha256(root / nm.MODELS_FILENAME) != manifest["networks_sha256"]:
            raise ValueError("Neural artifact file checksum differs")
    with np.load(root / nm.ARRAYS_FILENAME, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if validate:
        if manifest.get("arrays") != {name: {"dtype": a.dtype.name, "shape": list(a.shape)} for name, a in arrays.items()}:
            raise ValueError("Neural array inventory differs")
        if nm._arrays_identity(arrays) != manifest["arrays_identity"]:
            raise ValueError("Neural array identity differs")
        if manifest["source_identity"] != nm._source_identity(manifest):
            raise ValueError("Neural source metadata differs")
        if nm._identity(nm._artifact_identity_payload(manifest)) != manifest.get("completed_modes_identity"):
            raise ValueError("Neural completed-mode identity differs")
        fixed_arrays = {name: value for name, value in arrays.items() if name not in ("phi", "sample_prediction")}
        slots = nm._source_mode_slots(manifest)
        if len(slots) != manifest["counts"]["modes"]:
            raise ValueError("Neural mode inventory differs from artifact count")
        run_contract = {"format": "modal_gaussians.neural_modes_work", "version": 1,
                        "source_identity": manifest["source_identity"], "config": manifest["config"],
                        "runtime": manifest["runtime"], "geometry_graph": manifest["geometry_graph"],
                        "fixed_arrays_identity": nm._arrays_identity(fixed_arrays)}
        if "mode_selection" in manifest:
            selection = manifest["mode_selection"]
            parent_contract = {**run_contract, "fixed_arrays_identity": selection["parent_fixed_arrays_identity"]}
            if nm._identity(parent_contract) != selection["parent_run_identity"]:
                raise ValueError("Neural prefix parent run identity differs from original configuration and sources")
            run_contract["mode_selection"] = selection
        expected_run = nm._identity(run_contract)
        if expected_run != manifest["run_identity"]:
            raise ValueError("Neural run identity differs from resolved inputs and configuration")
        # Network replay constructs and validates each field geometry below.
        nm._validate_arrays(arrays, manifest, check_field_geometry=False)
        nm._check_persisted_sources(manifest, arrays)
    else:
        # ponytail: local experiment artifacts are trusted; request validate=True for forensic checks.
        slots = np.asarray(manifest.get("mode_selection", {}).get(
            "source_mode_slots", list(range(len(manifest["modes"])))), dtype=np.int64)
    if manifest["version"] != 16 and not validate:
        return nm.NeuralModesArtifact(root, manifest, arrays)
    networks = torch.load(root / nm.MODELS_FILENAME, map_location="cpu", weights_only=True)
    if validate and (networks.get("run_identity") != manifest["run_identity"] or len(networks.get("model_states", [])) != len(manifest["modes"])):
        raise ValueError("Neural network inventory/run identity differs")
    if validate and "mode_selection" in manifest and networks.get("source_mode_slots") != slots.tolist():
        raise ValueError("Neural network source mode order differs from completed prefix")
    if len(networks["model_states"]) != len(arrays["phi"]) or len(slots) != len(arrays["phi"]):
        raise ValueError("Neural network/mode count differs from phi")
    # Derived at load time; the historical disk schema and identities stay unchanged.
    rotation = np.empty_like(arrays["phi"]) if manifest["version"] == 16 else None
    for mode, state in enumerate(networks["model_states"]):
        evaluated = evaluate_model(state, nm._field_geometry(arrays, mode), length_scale=float(arrays["scene_scale"]),
                                   amplitude_scale=float(arrays["amplitude_scale"][mode]), config=nm._field_config(config, int(slots[mode])))
        reproduced = evaluated[0].detach().cpu().numpy()
        if validate and not np.allclose(reproduced, arrays["phi"][mode], rtol=3e-5, atol=1e-7 * float(arrays["amplitude_scale"][mode])):
            raise ValueError(f"Neural baked phi differs from network for mode {mode}")
        if rotation is not None:
            rotation[mode] = evaluated[1].detach().cpu().numpy()
    return nm.NeuralModesArtifact(root, manifest, arrays, rotation)

