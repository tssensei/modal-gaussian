"""Immutable observation snapshots; reusable geometry is separate from training."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from modal_gaussians.iteration_cache import (
    DEFAULT_CACHE, Timings, atomic_json, cached, identity, put_entry, sha256, module_revision,
)
from modal_gaussians.numpy_io import save_named_arrays
from modal_gaussians.flow.artifact import FlowAnalysisArtifact, FlowAnalysisArrays, flow_artifact_identity, load_flow_analysis_artifact
from modal_gaussians.motion.neural import neural_modes as nm

FORMAT = "modal_gaussians.neural_prepared"
OBSERVATION_FIELDS = ("pixel_sample_stride", "alpha_minimum", "mask_erosion_iterations", "energy_floor_fraction")
GRAPH_FIELDS = ("graph_neighbors", "graph_max_distance", "unknown_max_distance", "unknown_edge_weight", "graph_edge_filter", "alpha_minimum")
CONTROL_FIELDS = ("control_radius_fraction", "max_controls")


@dataclass(frozen=True)
class ArrayMetadata:
    """A shape declaration, explicitly incapable of reading a dense array."""
    shape: tuple[int, ...]
    dtype: np.dtype

    def __getitem__(self, selection):
        raise RuntimeError("Prepared observations do not contain full flow/spectrum; request full evaluation explicitly")


@dataclass
class PreparedNeuralInputs:
    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def source(self):
        return self.manifest["source"]

    @property
    def cache_dir(self):
        return Path(self.manifest["cache_dir"])

    def flow(self, path: str | Path) -> FlowAnalysisArtifact:
        requested = Path(path).expanduser().resolve()
        for index, record in enumerate(self.manifest["flows"]):
            if requested != Path(record["path"]):
                continue
            manifest = record["manifest"]
            def meta(name):
                a = manifest["arrays"][name]
                return ArrayMetadata(tuple(a["shape"]), np.dtype(a["dtype"]))
            flow = FlowAnalysisArtifact(requested, manifest,
                FlowAnalysisArrays(meta("flow"), self.arrays[f"v{index}_mask"], meta("spectrum")),
                dict(record["hashes"]))
            if flow_artifact_identity(flow) != record["identity"]:
                raise ValueError("Prepared flow identity differs")
            return flow
        raise ValueError(f"Flow is not part of prepared observations: {requested}")

    def validate_sources(self, source, config):
        if nm._source_identity(source) != self.manifest["source_identity"]:
            raise ValueError("Prepared source identity differs; build a new preparation")
        baseline = nm.NeuralModesConfig.from_dict(self.manifest["defaults"]["neural"])
        for name in OBSERVATION_FIELDS:
            if getattr(config, name) != getattr(baseline, name):
                raise ValueError(f"Observation setting {name} changed; build a new preparation")

    def training_inputs(self, source, scene, old_graph, config, device, timer):
        from modal_gaussians.static import cameras_from_scene_manifest
        from modal_gaussians.motion.neural.geometry_graph import (
            GeometryGraph, build_geometry_graph_arrays, build_control_graph, depth_thresholds_from_manifest,
        )
        self.validate_sources(source, config)
        arrays = {k[2:]: v.copy() for k, v in self.arrays.items() if k.startswith("o_")}
        with timer.stage("observation_cache"):
            scene.to(device).eval()
            for parameter in scene.parameters():
                parameter.requires_grad_(False)
            by_label = {c.label: c for c in cameras_from_scene_manifest(scene.manifest) if c.role == "reference"}
            cameras, projectors, depths, alphas = [], [], [], []
            for index, view in enumerate(source["views"]):
                camera = by_label[view["label"]].to(device)
                if camera.to_manifest_record()["camera_identity"] != view["camera_identity"]:
                    raise ValueError("Prepared camera differs")
                lo, hi = arrays["view_sample_offsets"][index:index + 2]
                projectors.append(nm.FrozenModalProjector(scene, camera,
                    torch.as_tensor(self.arrays[f"v{index}_jacobian"], device=device),
                    arrays["sample_pixels_xy"][lo:hi], torch.as_tensor(arrays["sample_confidence"][lo:hi], device=device)))
                cameras.append(camera)
                depths.append(self.arrays[f"v{index}_depth"])
                alphas.append(self.arrays[f"v{index}_alpha"])
        timer.records[-1]["cache_hit"] = True
        endpoint, jump = depth_thresholds_from_manifest(old_graph.manifest, [v["label"] for v in source["views"]])
        settings = nm._geometry_config(config)
        from modal_gaussians.motion.neural import geometry_graph as geometry_module
        geometry_revision = module_revision(geometry_module)
        graph_contract = {"implementation": "neural_geometry_cache_v1", "foreground": source["foreground_identity"],
                          "code": geometry_revision,
                          "config": {k: getattr(config, k) for k in GRAPH_FIELDS}}
        if config.graph_edge_filter == "depth":
            graph_contract["depth_source"] = nm._arrays_identity({
                "Ks": np.stack([c.K.cpu().numpy() for c in cameras]),
                "radial": np.array([c.radial_distortion for c in cameras]),
                "world_to_camera": np.stack([c.world_to_camera.cpu().numpy() for c in cameras]),
                "endpoint": endpoint, "jump": jump,
                **{f"depth_{i}": value for i, value in enumerate(depths)},
                **{f"alpha_{i}": value for i, value in enumerate(alphas)},
            })
        graph_arrays = cached(self.cache_dir / "geometry", graph_contract, lambda: build_geometry_graph_arrays(
            foreground_means=arrays["g_points"], Ks=np.stack([c.K.cpu().numpy() for c in cameras]),
            radial_coefficients=np.array([c.radial_distortion for c in cameras]),
            world_to_cameras=np.stack([c.world_to_camera.cpu().numpy() for c in cameras]),
            rendered_depths=depths, rendered_alphas=alphas, endpoint_thresholds=endpoint,
            depth_jump_thresholds=jump, config=settings).as_dict(), timer, "geometry_cache")
        graph = GeometryGraph.from_dict(graph_arrays)
        control_contract = {"implementation": "neural_controls_cache_v1", "graph": nm._arrays_identity(graph_arrays),
                            "code": geometry_revision,
                            "scene_scale": float(arrays["scene_scale"]),
                            "config": {k: getattr(config, k) for k in CONTROL_FIELDS}}
        arrays.update({"g_" + k: v for k, v in graph_arrays.items()})
        if config.training_fragment_config is not None:
            from . import training_fragments, fragment_propagation, surface_attachments, pointwise_attachments, guarded_attachments
            attachment_inputs = None
            if config.training_fragment_config.get("strategy") == "surface":
                attachment_inputs = surface_attachments.observation_inputs(scene, cameras, depths, alphas, endpoint,
                    arrays["contribution_mass"], arrays["contribution_threshold"])
                control_contract["surface_inputs"] = nm._arrays_identity(attachment_inputs)
            if config.training_fragment_config.get("strategy") == "pointwise":
                attachment_inputs = {"p_observation_view_mask": arrays["observation_view_mask"]}
                control_contract["pointwise_inputs"] = nm._arrays_identity(attachment_inputs)
                control_contract["pointwise_code"] = module_revision(pointwise_attachments)
            if config.training_fragment_config.get("strategy") == "guarded":
                attachment_inputs = guarded_attachments.observation_inputs(arrays["g_points"], cameras, depths,
                                                                           alphas, endpoint, config.alpha_minimum)
                attachment_inputs.update(observation_view_mask=arrays["observation_view_mask"],
                                         contribution_mass=arrays["contribution_mass"])
                control_contract["guarded_inputs"] = nm._arrays_identity(attachment_inputs)
                control_contract["guarded_code"] = module_revision(guarded_attachments, pointwise_attachments)
            if config.training_fragment_config.get("strategy") == "component_field":
                from . import component_field
                attachment_inputs = component_field.observation_inputs(arrays["g_points"], cameras, depths,
                                                                        alphas, endpoint, config.alpha_minimum)
                attachment_inputs.update(observation_view_mask=arrays["observation_view_mask"],
                                         contribution_mass=arrays["contribution_mass"])
                control_contract["component_inputs"] = nm._arrays_identity(attachment_inputs)
                control_contract["component_code"] = module_revision(component_field, guarded_attachments, pointwise_attachments)
            control_contract.update(implementation="host_controls_with_training_fill_v1",
                fragment_config=config.training_fragment_config,
                attachment_code=module_revision(training_fragments, fragment_propagation, surface_attachments))
            control_arrays = cached(self.cache_dir / "controls", control_contract,
                lambda: training_fragments.build_training_controls(graph, geometry_config=settings,
                    fragment_config=config.training_fragment_config, scene_scale=float(arrays["scene_scale"]),
                    attachment_inputs=attachment_inputs),
                timer, "control_cache")
            arrays.update(control_arrays)
        else:
            control_arrays = cached(self.cache_dir / "controls", control_contract,
                lambda: build_control_graph(graph, config=settings, scene_scale=float(arrays["scene_scale"])).as_dict(),
                timer, "control_cache")
            arrays.update({"c_" + k: v for k, v in control_arrays.items()})
        nm._set_support_roles(arrays)
        return arrays, projectors, cameras, depths, alphas


def load_prepared(path: str | Path) -> PreparedNeuralInputs:
    root = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("version") != 1:
        raise ValueError("Unsupported neural preparation")
    expected = identity({k: v for k, v in manifest.items() if k != "prepared_identity"})
    if expected != manifest.get("prepared_identity") or sha256(root / "arrays.npz") != manifest.get("arrays_sha256"):
        raise ValueError("Neural preparation checksum/identity differs")
    with np.load(root / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if nm._arrays_identity(arrays) != manifest.get("arrays_identity"):
        raise ValueError("Neural preparation arrays differ")
    if nm._source_identity(manifest["source"]) != manifest["source_identity"]:
        raise ValueError("Neural preparation source metadata differs")
    result = PreparedNeuralInputs(root, manifest, arrays)
    if len(manifest["flows"]) != len(manifest["source"]["views"]):
        raise ValueError("Neural preparation flow/view counts differ")
    for view, record in zip(manifest["source"]["views"], manifest["flows"]):
        flow = result.flow(record["path"])
        if flow_artifact_identity(flow) != view["flow_identity"]:
            raise ValueError("Neural preparation flow/view source differs")
    return result


def prepare_neural(*, output_dir, cache_dir=DEFAULT_CACHE, from_result=None,
                   scene_dir=None, topology_dir=None, measurements_dir=None, graph_dir=None,
                   alignment_from=None, config=None, config_overrides=None, timer=None):
    from modal_gaussians.result import load_modal_result
    from modal_gaussians.vis.spectrum import _read_reference_rgb
    from modal_gaussians.modes import dense_cache_contract
    from modal_gaussians.motion.neural.fragment_propagation import FragmentPropagationConfig
    from modal_gaussians.motion.neural.surface_attachments import SurfaceAttachmentConfig
    from modal_gaussians.motion.common.projection import RenderedDesignConfig
    timer = timer or Timings()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    cache_dir = Path(cache_dir).expanduser().resolve()
    frozen = None
    legacy_controls_requested = (config_overrides or {}).get("training_fragment_config", "default") is None
    with timer.stage("source_validation"):
        fragment_config = (FragmentPropagationConfig() if legacy_controls_requested else SurfaceAttachmentConfig()).to_dict()
        design_config = RenderedDesignConfig().to_dict()
        if from_result is not None:
            if any(v is not None for v in (scene_dir, topology_dir, measurements_dir, graph_dir, alignment_from)):
                raise ValueError("Use --from-result or the five explicit sources, not both")
            result = load_modal_result(from_result)
            completed = result.completed_modes
            if completed.manifest["version"] == 9:
                fragment_config = completed.manifest["fragment_propagation"]
                completed = nm.load_neural_completed_modes(completed.manifest["parent_completed_modes"])
            if completed.manifest["version"] not in (8, 10, 11, 12, 14, 16) or "source_modes" in completed.manifest:
                raise ValueError("Preparation import requires a full v8/v9/v10/v11/v12/v14 neural source")
            m = completed.manifest
            scene_dir, topology_dir, measurements_dir, graph_dir, alignment_from = (
                m[k] for k in ("static_scene", "topology", "measurements", "observed_structure_graph", "alignment_from"))
            baseline_config = nm.NeuralModesConfig.from_dict(m["config"])
            if baseline_config.training_fragment_config is not None:
                fragment_config = baseline_config.training_fragment_config
            config = config or baseline_config
            if config_overrides:
                config = nm.NeuralModesConfig(**{**config.to_dict(), **config_overrides})
                config_overrides = None
            if all(getattr(config, k) == getattr(baseline_config, k) for k in OBSERVATION_FIELDS):
                frozen = {k: v for k, v in completed.arrays.items() if k not in ("phi", "sample_prediction")}
            design_config = result.rendered_design.manifest["settings"]
        config = config or nm.NeuralModesConfig()
        if config_overrides:
            config = nm.NeuralModesConfig(**{**config.to_dict(), **config_overrides})
        if config.training_fragment_config is None and not legacy_controls_requested:
            config = nm.NeuralModesConfig(**{**config.to_dict(), "training_fragment_config": fragment_config})
        if config.training_fragment_config is not None:
            fragment_config = config.training_fragment_config
        config.validate()
        if any(v is None for v in (scene_dir, topology_dir, measurements_dir, graph_dir, alignment_from)):
            raise ValueError("Preparation needs scene, topology, measurements, graph and alignment-from")
        source, scene, _, _, old_graph, alignment, dense = nm._load_sources(
            scene_dir=scene_dir, topology_dir=topology_dir, measurements_dir=measurements_dir,
            graph_dir=graph_dir, alignment_from=alignment_from)
    arrays, flows, loaded_flows = {}, [], {}
    with timer.stage("flow_validation_and_snapshot"):
        frequencies = np.array([m["frequency_hz"] for m in source["modes"]])
        for index, view in enumerate(dense.manifest["views"]):
            flow = load_flow_analysis_artifact(view["flow_artifact"], cache_dir=cache_dir)
            flow_id = flow_artifact_identity(flow)
            if flow_id != view["flow_identity"]:
                raise ValueError("Preparation flow differs from dense source")
            loaded_flows[str(flow.path.resolve())] = flow
            arrays[f"v{index}_mask"] = flow.arrays.mask_union.copy()
            arrays[f"v{index}_rgb"] = _read_reference_rgb(flow)
            flows.append({"path": str(flow.path.resolve()), "manifest": flow.manifest,
                          "hashes": flow.verified_hashes, "identity": flow_id})
            put_entry(cache_dir / "dense_dft", dense_cache_contract(flow_id, frequencies), {"modes": dense.view_modes[index]})
    with timer.stage("observation_preparation"):
        observed, projectors, _, depths, alpha_images = nm._prepare_observation_arrays(
            scene, source, dense, config, torch.device("cuda"), alignment.arrays["alphas"],
            alignment.arrays["alpha_identifiable_mask"], frozen_arrays=frozen,
            flow_loader=lambda path: loaded_flows[str(Path(path).resolve())])
        for key, value in observed.items():
            if (not key.startswith(("g_", "c_", "t_", "f_", "a_", "p_", "h_", "u_")) or key == "g_points") and key != "support_class":
                arrays["o_" + key] = value
        arrays["o_g_points"] = scene.foreground.active()["means"].detach().cpu().numpy()
        for index, projector in enumerate(projectors):
            arrays[f"v{index}_jacobian"] = projector.jacobian.cpu().numpy()
            arrays[f"v{index}_depth"] = depths[index]
            arrays[f"v{index}_alpha"] = alpha_images[index]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    save_named_arrays(temporary / "arrays.npz", arrays)
    manifest = {"format": FORMAT, "version": 1, "source": source, "source_identity": nm._source_identity(source),
                "flows": flows, "cache_dir": str(cache_dir), "arrays_sha256": sha256(temporary / "arrays.npz"),
                "arrays_identity": nm._arrays_identity(arrays),
                "defaults": {"neural": config.to_dict(), "fragment": fragment_config, "design": design_config}}
    manifest["prepared_identity"] = identity(manifest)
    atomic_json(temporary / "manifest.json", manifest)
    load_prepared(temporary)
    os.rename(temporary, destination)
    prepared = load_prepared(destination)
    # Populate geometry/control caches now; training starts with warm geometry.
    with timer.stage("geometry_preparation"):
        prepared.training_inputs(source, scene, old_graph, config, torch.device("cuda"), timer)
    timer.save(destination / "timings.json")
    return prepared
