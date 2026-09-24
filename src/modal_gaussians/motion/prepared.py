"""Immutable observation snapshots; reusable geometry is separate from training."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from modal_gaussians.common.scene_store import resolve_path, logical_path, scene_cache
import tempfile
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from modal_gaussians.common.cache import DEFAULT_CACHE, Timings, atomic_json, cached, identity, sha256, module_revision
from modal_gaussians.common.numpy_io import save_named_arrays
from modal_gaussians.preprocessing.reference import SequenceReference, reference_identity, load_reference
from modal_gaussians.motion import training as nm

FORMAT = "modal_gaussians.neural_prepared"
OBSERVATION_FIELDS = ("pixel_sample_stride", "alpha_minimum", "mask_erosion_iterations", "energy_floor_fraction")
GRAPH_FIELDS = ("graph_neighbors", "graph_max_distance", "graph_edge_filter")
CONTROL_FIELDS = ("control_radius_fraction", "max_controls")


@dataclass
class PreparedNeuralInputs:
    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    external_geometry_graph: dict[str, np.ndarray] | None = None
    external_geometry_contract: dict[str, Any] | None = None
    propagation_backend: str = "cupy"

    @property
    def source(self):
        return self.manifest["source"]

    @property
    def cache_dir(self):
        return scene_cache(self.source, self.manifest["cache_dir"])

    def attach_geometry_graph(self, path):
        from modal_gaussians.motion.geometry_graph import GeometryGraph
        root = resolve_path(path, strict=True)
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if (manifest.get('format') != 'modal_gaussians.modal_similarity_graph'
                or manifest.get('version') != 2 or manifest.get('graph_file') != 'graph.npz'
                or manifest.get('foreground_identity') != self.source['foreground_identity']
                or manifest.get('static_scene_identity') != self.source['static_scene_identity']
                or manifest['source']['prepared_identity'] != self.manifest['prepared_identity']):
            raise ValueError('Modal graph does not belong to this prepared frequency/scene')
        with np.load(root / 'graph.npz', allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        graph = GeometryGraph.from_dict(arrays, validate=True)
        if (not np.array_equal(graph.points, self.arrays['o_g_points'])
                or not np.array_equal(graph.node_gaussian_index, np.arange(len(graph.points), dtype=np.int64))
                or graph.edge_propagation_length is None):
            raise ValueError('Modal graph order, positions or propagation lengths differ')
        self.external_geometry_graph = arrays
        self.external_geometry_contract = {
            'path': str(logical_path(root)), 'format': manifest['format'], 'manifest_identity': identity(manifest),
            'arrays_identity': nm._arrays_identity(arrays), 'frequency_hz': manifest['frequency_hz'],
            'config': manifest['config'], 'propagation': 'saved_edge_length_over_modal_factor'}

    def flow(self, path):
        requested = resolve_path(path)
        for index, record in enumerate(self.manifest['flows']):
            if requested == resolve_path(record['path']):
                reference = SequenceReference(requested, record['manifest'],
                    SimpleNamespace(mask_union=self.arrays[f'v{index}_mask']))
                if reference_identity(reference) != record['identity']:
                    raise ValueError('Prepared reference identity differs')
                return reference
        raise ValueError(f'Reference is not part of prepared observations: {requested}')

    def validate_sources(self, source, config):
        if nm._source_identity(source) != self.manifest["source_identity"]:
            raise ValueError("Prepared source identity differs; build a new preparation")
        baseline = nm.NeuralModesConfig.from_dict(self.manifest["defaults"]["neural"])
        for name in OBSERVATION_FIELDS:
            if getattr(config, name) != getattr(baseline, name):
                raise ValueError(f"Observation setting {name} changed; build a new preparation")

    def training_inputs(self, source, scene, config, device, timer, *, frozen_arrays=None):
        from modal_gaussians.geometry.scene import cameras_from_scene_manifest
        from modal_gaussians.motion.geometry_graph import GeometryGraph
        self.validate_sources(source, config)
        arrays = (frozen_arrays if frozen_arrays is not None else
                  {k[2:]: v.copy() for k, v in self.arrays.items() if k.startswith("o_")})
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
                    arrays["sample_pixels_xy"][lo:hi], torch.as_tensor(arrays["sample_confidence"][lo:hi], device=device),
                    backend=config.modal_projection_backend))
                cameras.append(camera)
                depths.append(self.arrays[f"v{index}_depth"])
                alphas.append(self.arrays[f"v{index}_alpha"])
        timer.records[-1]["cache_hit"] = True
        if frozen_arrays is not None:
            return arrays, projectors, cameras, depths, alphas
        if self.external_geometry_graph is None:
            raise ValueError('Attach the frequency-specific modal-similarity graph before training')
        from modal_gaussians.motion import component_field, control_propagation, shared_controls
        from modal_gaussians.motion.common import graph_ops, point_transfer, visibility
        endpoint = np.asarray(source['endpoint_relative_tolerances'], dtype=np.float64)
        settings = nm._geometry_config(config)
        graph_arrays = self.external_geometry_graph
        graph = GeometryGraph.from_dict(graph_arrays)
        arrays.update({'g_' + k: v for k, v in graph_arrays.items()})
        inputs = component_field.observation_inputs(arrays['g_points'], cameras, depths, alphas,
                                                    endpoint, config.alpha_minimum)
        inputs.update(observation_view_mask=arrays['observation_view_mask'], contribution_mass=arrays['contribution_mass'])
        contract = {'implementation': 'component_controls_v2', 'graph': nm._arrays_identity(graph_arrays),
            'inputs': nm._arrays_identity(inputs), 'scene_scale': float(arrays['scene_scale']),
            'config': settings.to_dict(), 'component': config.training_fragment_config,
            'code': module_revision(component_field, shared_controls, control_propagation, graph_ops, point_transfer, visibility),
            'propagation': control_propagation.backend_identity(self.propagation_backend)}
        def build_controls():
            geometry = shared_controls.weighted_geometry(graph, geometry_config=settings,
                fragment_config=config.training_fragment_config, scene_scale=float(arrays['scene_scale']),
                cache_dir=self.cache_dir, timer=timer, backend=self.propagation_backend)
            return component_field.build_component_controls(graph, geometry_config=settings,
                fragment_config=config.training_fragment_config, scene_scale=float(arrays['scene_scale']),
                attachment_inputs=inputs, geometry_arrays=geometry)
        arrays.update(cached(self.cache_dir / 'controls', contract, build_controls, timer, 'control_cache'))
        nm._set_support_roles(arrays)
        return arrays, projectors, cameras, depths, alphas


def load_prepared(path: str | Path, *, validate: bool = False) -> PreparedNeuralInputs:
    root = resolve_path(path, strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("version") != 2:
        raise ValueError("Unsupported neural preparation")
    if validate:
        expected = identity({k: v for k, v in manifest.items() if k != "prepared_identity"})
        if expected != manifest.get("prepared_identity") or sha256(root / "arrays.npz") != manifest.get("arrays_sha256"):
            raise ValueError("Neural preparation checksum/identity differs")
    with np.load(root / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if validate:
        if nm._arrays_identity(arrays) != manifest.get("arrays_identity"):
            raise ValueError("Neural preparation arrays differ")
        if nm._source_identity(manifest["source"]) != manifest["source_identity"]:
            raise ValueError("Neural preparation source metadata differs")
    result = PreparedNeuralInputs(root, manifest, arrays)
    if len(manifest["flows"]) != len(manifest["source"]["views"]):
        raise ValueError("Neural preparation flow/view counts differ")
    if validate:
        for view, record in zip(manifest["source"]["views"], manifest["flows"]):
            flow = result.flow(record["path"])
            if reference_identity(flow) != view["flow_identity"]:
                raise ValueError("Neural preparation flow/view source differs")
    return result


def _read_reference_rgb(flow: SequenceReference) -> np.ndarray:
    """Read the geometry reference without importing Viewer code."""
    from modal_gaussians.preprocessing.frames import read_rgb
    images = resolve_path(flow.manifest["inputs"]["sequence"]["image_directory"], strict=True)
    rgb = read_rgb(images / f"{flow.manifest['reference_frame_name']}.png")
    if rgb.shape[:2] != flow.arrays.mask_union.shape:
        raise ValueError("Flow reference image shape differs from flow arrays")
    return rgb


def prepare_neural(*, scene_dir, views, output_dir, cache_dir=DEFAULT_CACHE, config=None, timer=None):
    """views contains (camera label, sequence reference, relative depth tolerance)."""
    from modal_gaussians.geometry.scene import load_static_scene, cameras_from_scene_manifest
    from modal_gaussians.spectrum.modes import Complex2DModesArtifact
    from modal_gaussians.motion import geometry_graph
    from modal_gaussians.motion.observations.alpha import AlphaSyncConfig
    timer = timer or Timings()
    destination = resolve_path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    config = config or nm.NeuralModesConfig()
    config.validate()
    labels = [v[0] for v in views]
    tolerances = [float(v[2]) for v in views]
    if (not views or len(set(labels)) != len(labels)
            or not np.isfinite(tolerances).all() or np.any(np.asarray(tolerances) < 0)):
        raise ValueError('Supply unique views with finite nonnegative depth tolerances')
    cache_dir = resolve_path(cache_dir)
    scene_path = resolve_path(scene_dir, strict=True)
    scene = load_static_scene(scene_path, 'cuda').eval()
    cameras = {c.label: c for c in cameras_from_scene_manifest(scene.manifest) if c.role == 'reference'}
    records, arrays, sources = [], {}, []
    for index, (label, path, _) in enumerate(views):
        flow = load_reference(path)
        camera = cameras.get(label)
        if camera is None or flow.manifest['shape_hw'] != [camera.height, camera.width]:
            raise ValueError(f'Reference camera/grid differs: {label}')
        geometry_image = resolve_path(flow.manifest['inputs']['sequence']['image_directory']) / (flow.manifest['reference_frame_name'] + '.png')
        if sha256(geometry_image) != camera.image_sha256:
            raise ValueError(f'Geometry reference image differs from the registered camera: {label}')
        fid = reference_identity(flow)
        records.append({'path': str(flow.path), 'manifest': flow.manifest, 'identity': fid})
        arrays[f'v{index}_mask'] = flow.arrays.mask_union.copy()
        arrays[f'v{index}_rgb'] = _read_reference_rgb(flow)
        sources.append({'index': index, 'label': label, 'camera_identity': camera.to_manifest_record()['camera_identity'],
            'shape_hw': flow.manifest['shape_hw'], 'flow_artifact': str(flow.path), 'flow_identity': fid,
            'flow_reference_frame_name': flow.manifest['reference_frame_name']})
    source = {'static_scene': str(scene_path), 'static_scene_identity': scene.manifest['static_scene_identity'],
        'foreground_identity': scene.manifest['foreground_identity'], 'modes': [], 'views': sources,
        'endpoint_relative_tolerances': tolerances, 'alignment_identity': None, 'complex_2d_modes_identity': None}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.', dir=destination.parent))
    parent = SimpleNamespace(manifest={'flows': records})
    try:
        with timer.stage('geometry_observations'):
            _, renders = _observation_topology(parent, source, scene, temporary, destination, config)
            dense = Complex2DModesArtifact(temporary, {'views': sources},
                tuple(np.empty((0, *v['shape_hw'], 2), np.complex64) for v in sources))
            observed, projectors, _, depths, alphas = nm._prepare_observation_arrays(
                scene, source, dense, config, torch.device('cuda'),
                np.empty((0, len(views)), np.complex64), np.empty((0, len(views)), bool),
                observation_renders=renders)
            arrays.update({'o_' + key: value for key, value in observed.items()})
            arrays['o_g_points'] = scene.foreground.active()['means'].detach().cpu().numpy()
            for index, projector in enumerate(projectors):
                arrays[f'v{index}_jacobian'] = projector.jacobian.cpu().numpy()
                arrays[f'v{index}_depth'] = depths[index]
                arrays[f'v{index}_alpha'] = alphas[index]
                arrays[f'v{index}_subject_mask'] = alphas[index] >= config.alpha_minimum
        contract = {'implementation': 'neural_geometry_cache_v2', 'foreground': source['foreground_identity'],
            'view_count': len(views), 'code': module_revision(geometry_graph),
            'config': {k: getattr(config, k) for k in GRAPH_FIELDS}}
        cached(cache_dir / 'geometry', contract,
            lambda: geometry_graph.build_geometry_graph_arrays(foreground_means=arrays['o_g_points'],
                view_count=len(views), config=nm._geometry_config(config)).as_dict(), timer, 'geometry_cache')
        save_named_arrays(temporary / 'arrays.npz', arrays)
        manifest = {'format': FORMAT, 'version': 2, 'source': source, 'source_identity': nm._source_identity(source),
            'flows': records, 'cache_dir': str(cache_dir), 'arrays_sha256': sha256(temporary / 'arrays.npz'),
            'arrays_identity': nm._arrays_identity(arrays),
            'geometry_graph': str(cache_dir / 'geometry' / identity(contract)),
            'defaults': {'neural': config.to_dict(), 'fragment': config.training_fragment_config}, 'alpha_config': AlphaSyncConfig().to_dict()}
        manifest['prepared_identity'] = identity(manifest)
        atomic_json(temporary / 'manifest.json', manifest)
        timer.save(temporary / 'timings.json')
        os.rename(temporary, destination)
    except BaseException:
        if temporary.resolve().parent != destination.parent.resolve() or not temporary.name.startswith(f".{destination.name}."):
            raise RuntimeError("Temporary publication directory escaped its output parent")
        shutil.rmtree(temporary)
        raise
    return PreparedNeuralInputs(destination, manifest, arrays)


def _observation_topology(parent, source, scene, output_dir, final_dir, config):
    """Rebuild correspondences from the selected subject's visible contribution."""
    from modal_gaussians.motion.observations import topology as tp
    from modal_gaussians.motion.common.projection import render_observation_geometry, uses_visible_subject
    from modal_gaussians.geometry.scene import cameras_from_scene_manifest

    visible_subject = uses_visible_subject(scene)
    source["observation_region"] = "visible_subject" if visible_subject else "foreground_mask"
    active = scene.foreground.active()
    references = {c.label: c for c in cameras_from_scene_manifest(scene.manifest) if c.role == "reference"}
    cameras, depths, alphas, masks, view_records, renders = [], [], [], [], [], {}
    with torch.no_grad():
        for view, flow in zip(source["views"], parent.manifest["flows"]):
            camera = references.get(view["label"])
            if camera is None or camera.to_manifest_record()["camera_identity"] != view["camera_identity"]:
                raise ValueError("Selected subject reference camera differs from prepared geometry")
            camera = camera.to(active["means"].device)
            rendered = render_observation_geometry(scene, camera)
            alpha = rendered["alpha"].detach().cpu().numpy().astype(np.float32)
            mask = (alpha >= config.alpha_minimum if visible_subject
                    else load_reference(flow["path"]).arrays.mask_union)
            cameras.append(camera)
            depths.append(rendered["expected_depth"].detach().cpu().numpy().astype(np.float32))
            alphas.append(alpha)
            masks.append(mask)
            renders[view["label"]] = rendered
            view_records.append({**view, "flow_artifact": flow["path"],
                "flow_reference_frame_name": flow["manifest"]["reference_frame_name"],
                "mask_union_sha256": tp._sha256_array(mask),
                "mask_role": source["observation_region"]})
    settings = tp.TopologyConfig(pixel_sample_stride=config.pixel_sample_stride,
        foreground_alpha_minimum=config.alpha_minimum,
        mask_erosion_iterations=0 if visible_subject else config.mask_erosion_iterations)
    topology, counts = tp.build_topology_arrays(
        foreground_means=active["means"].detach().cpu().numpy(),
        foreground_scales=active["scales"].detach().cpu().numpy(),
        foreground_quaternions=active["quaternions"].detach().cpu().numpy(),
        foreground_opacities=active["opacities"].detach().cpu().numpy(),
        cameras=cameras, masks=masks, rendered_depths=depths, rendered_alphas=alphas, config=settings)
    for index, record in enumerate(view_records):
        record.update(sample_count=counts["samples_per_view"][index],
                      contributor_count=counts["contributors_per_view"][index])
    manifest = {"format": tp.TOPOLOGY_FORMAT, "version": tp.TOPOLOGY_VERSION,
        "static_scene": source["static_scene"], "static_scene_identity": source["static_scene_identity"],
        "foreground_identity": source["foreground_identity"], "views": view_records,
        "parameters": {**settings.to_dict(), "observation_region": source["observation_region"],
            "occlusion": "full_scene_transmittance" if visible_subject else "foreground_transmittance",
            **({"projection_jacobian": "d_simple_radial_pixel_d_normalized_world_point"}
               if any(c.distortion_applied for c in cameras) else {})},
        "counts": {"views": len(cameras), "foreground_gaussians": scene.foreground.count,
            "samples": len(topology.sample_view_index), "contributors": len(topology.contributor_gaussian_index)}}
    published = tp._publish_topology(output_dir / "topology", arrays=topology, manifest=manifest, validate=False)
    source.update(topology=str(final_dir / "topology"), topology_identity=published.manifest["topology_identity"])
    return topology, renders
