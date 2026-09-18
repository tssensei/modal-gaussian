"""Reuse frozen geometry with independently computed selected modal observations."""
from __future__ import annotations

from dataclasses import fields
import copy
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from modal_gaussians.iteration_cache import Timings, atomic_json, identity, sha256
from modal_gaussians.modes import Complex2DModesArtifact, TRANSFORM_CONVENTION, _validate_mode_records
from modal_gaussians.numpy_io import save_named_arrays
from modal_gaussians.progress import report_progress
from .modal_similarity_artifact import _manifest, _modal_view
from .prepared import FORMAT as PREPARED_FORMAT, PreparedNeuralInputs, load_prepared
from . import neural_modes as nm

FORMAT = "modal_gaussians.selected_complex_2d_modes"
ALIGNMENT_FORMAT = "modal_gaussians.selected_modal_alignment"


def load_selected_modal_bundle(path, manifest=None):
    """Open selected fields without scanning arrays or loading the flow sequences."""
    root = Path(path).expanduser().resolve(strict=True)
    manifest = manifest if manifest is not None else _manifest(root)
    if (manifest.get("format") != FORMAT or manifest.get("version") != 1
            or manifest.get("transform") != TRANSFORM_CONVENTION
            or manifest.get("full_spectrum_available") is not False):
        raise ValueError("Unsupported selected modal bundle")
    modes = _validate_mode_records(manifest)
    if len(modes) != 1 or not manifest.get("views"):
        raise ValueError("Selected modal bundles require one frequency and nonempty views")
    values, labels = [], []
    for index, view in enumerate(manifest["views"]):
        if (view["index"] != index or view.get("flow_role") != "geometry_reference_only"
                or not view.get("label")):
            raise ValueError("Selected modal view order/reference role differs")
        value = np.load(view["modes_file"], mmap_mode="r", allow_pickle=False)
        if value.dtype != np.complex64 or value.shape != (1, *view["shape_hw"], 2):
            raise ValueError("Selected modal field must be complex64 [1,H,W,2]")
        labels.append(view["label"])
        values.append(value)
    if len(set(labels)) != len(labels):
        raise ValueError("Selected modal view labels must be unique")
    return Complex2DModesArtifact(root, manifest, tuple(values))


def _sample_fields(modal_fields, pixels, view_indices):
    pixels, view_indices = np.asarray(pixels), np.asarray(view_indices)
    if pixels.shape != (len(view_indices), 2) or pixels.dtype.kind not in "iu":
        raise ValueError("Modal sampling requires integer [P,2] pixel coordinates")
    if np.any(view_indices < 0) or np.any(view_indices >= len(modal_fields)):
        raise ValueError("Modal sampling view index is out of bounds")
    target = np.empty((1, len(pixels), 2), dtype=np.complex64)
    for index, field in enumerate(modal_fields):
        rows = np.flatnonzero(view_indices == index)
        xy = pixels[rows]
        if np.any(xy < 0) or np.any(xy >= [field.shape[1], field.shape[0]]):
            raise ValueError("Modal sampling pixel is out of bounds")
        target[0, rows] = field[xy[:, 1], xy[:, 0]]
    if not np.isfinite(target).all():
        raise ValueError("Selected sampled modal observations contain NaN or Inf")
    return target


def _replace_observations(arrays, target, alphas, identifiable, energy_floor_fraction):
    """Apply the existing neural observation normalization to a new modal source."""
    result = dict(arrays)
    confidence = arrays["o_sample_confidence"].astype(np.float64)
    offsets = arrays["o_view_sample_offsets"]
    view_count = len(offsets) - 1
    target = np.asarray(target, dtype=np.complex64)
    alphas = np.asarray(alphas, dtype=np.complex64)
    identifiable = np.asarray(identifiable, dtype=bool)
    if (target.shape != (1, len(confidence), 2) or alphas.shape != (1, view_count)
            or identifiable.shape != alphas.shape or not np.isfinite(target).all()
            or not np.isfinite(alphas).all() or alphas[0, 0] != 1 or not identifiable[0, 0]):
        raise ValueError("Selected targets/alignment do not match the prepared sample domain")
    rms = np.empty((1, view_count), dtype=np.float64)
    sensitivity = np.empty(view_count, dtype=np.float64)
    for view in range(view_count):
        lo, hi = offsets[view:view + 2]
        weights = confidence[lo:hi]
        energy = np.sum(np.abs(target[:, lo:hi].astype(np.complex128)) ** 2, axis=-1)
        rms[:, view] = np.sqrt(np.sum(energy * weights[None], axis=1) / weights.sum())
        sensitivity[view] = np.average(arrays["o_sample_projection_sensitivity"][lo:hi], weights=weights)
    positive = rms[identifiable & (rms > 0)]
    floor = max(energy_floor_fraction * (float(np.median(positive)) if len(positive) else 1.), 1e-12)
    valid = identifiable[0]
    denominator = float(np.sum(np.abs(alphas[0, valid].astype(np.complex128)) ** 2 * sensitivity[valid]))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Selected modal alignment has no positive projection sensitivity")
    scale = max(1e-6 * float(arrays["o_scene_scale"]), math.sqrt(float(np.sum(rms[0, valid] ** 2)) / denominator))
    visible = arrays["o_contribution_mass"] > arrays["o_contribution_threshold"][None]
    result.update({"o_sample_target": target, "o_alphas": alphas,
        "o_alpha_identifiable_mask": identifiable,
        "o_observation_view_mask": visible[None] & identifiable[:, None, :],
        "o_mode_view_rms": rms, "o_mode_view_loss_scale": np.maximum(rms, floor),
        "o_measurement_rms_floor": np.asarray(floor, dtype=np.float64),
        "o_amplitude_scale": np.asarray([scale], dtype=np.float64)})
    for name in nm.PREFIX_NORMALIZATION_ARRAYS:
        result.pop("o_" + name, None)
    return result


def _subject_topology(parent, source, scene, output_dir, final_dir, config):
    """Rebuild correspondences from the selected subject's visible contribution."""
    from modal_gaussians import topology as tp
    from modal_gaussians.motion.common.projection import render_observation_geometry, uses_visible_subject
    from modal_gaussians.static import cameras_from_scene_manifest

    if not uses_visible_subject(scene):
        raise ValueError("Scene override requires an applied manual subject selection")
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
            mask = alpha >= config.alpha_minimum
            cameras.append(camera)
            depths.append(rendered["expected_depth"].detach().cpu().numpy().astype(np.float32))
            alphas.append(alpha)
            masks.append(mask)
            renders[view["label"]] = rendered
            view_records.append({**view, "flow_artifact": flow["path"],
                "flow_reference_frame_name": flow["manifest"]["reference_frame_name"],
                "mask_union_sha256": tp._sha256_array(mask),
                "mask_role": "visible_subject_contribution_not_flow_mask"})
    settings = tp.TopologyConfig(pixel_sample_stride=config.pixel_sample_stride,
        foreground_alpha_minimum=config.alpha_minimum, mask_erosion_iterations=0)
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
        "parameters": {**settings.to_dict(), "observation_region": "visible_subject",
            "occlusion": "full_scene_transmittance",
            **({"projection_jacobian": "d_simple_radial_pixel_d_normalized_world_point"}
               if any(c.distortion_applied for c in cameras) else {})},
        "counts": {"views": len(cameras), "foreground_gaussians": scene.foreground.count,
            "samples": len(topology.sample_view_index), "contributors": len(topology.contributor_gaussian_index)}}
    published = tp._publish_topology(output_dir / "topology", arrays=topology, manifest=manifest, validate=False)
    source.update(topology=str(final_dir / "topology"), topology_identity=published.manifest["topology_identity"])
    return topology, renders


def prepare_selected_modal(*, prepared_dir, views, frequency_hz, output_dir, scene_dir=None):
    """Publish one new observation snapshot, with no graph build or training."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-writing-", dir=destination.parent))
    try:
        return _prepare_selected_modal(prepared_dir=prepared_dir, views=views, frequency_hz=frequency_hz,
                                       destination=destination, temporary=temporary, scene_dir=scene_dir)
    finally:
        if temporary.exists():
            resolved = temporary.resolve()
            if resolved.parent != destination.parent.resolve() or not resolved.name.startswith(f".{destination.name}-writing-"):
                raise RuntimeError("Refusing to remove a preparation temporary directory outside its output parent")
            shutil.rmtree(resolved)


def _prepare_selected_modal(*, prepared_dir, views, frequency_hz, destination, temporary, scene_dir):
    from modal_gaussians.synchronization import AlphaSyncConfig, prepare_observations, solve_alpha_sync
    from modal_gaussians.topology import TopologyArrays, ARRAY_FILENAME, ARRAY_DTYPES

    timer = Timings()
    with timer.stage("selected_modal_sources"):
        parent = load_prepared(prepared_dir)
        source = copy.deepcopy(parent.source)
        visible_subject = (scene_dir is not None or
            source.get("selected_modal_supervision", {}).get("observation_region") == "visible_subject")
        requested = dict(views)
        labels = [v["label"] for v in source["views"]]
        if len(requested) != len(views) or set(requested) != set(labels):
            raise ValueError("Selected modal sources must provide every prepared view exactly once")
        selected = [m for m in source["modes"] if math.isclose(m["frequency_hz"], frequency_hz, rel_tol=0, abs_tol=1e-9)]
        if len(selected) != 1:
            raise ValueError("Selected frequency must match exactly one prepared mode")
        mode = {**selected[0], "mode_slot": 0}
        modal_fields, records = [], []
        for view, flow in zip(source["views"], parent.manifest["flows"]):
            field, _, record = _modal_view(requested[view["label"]], flow, view, frequency_hz,
                                         read_mask=not visible_subject)
            modal_fields.append(field)
            records.append({**view, "flow_artifact": flow["path"], "flow_role": "geometry_reference_only",
                "frame_count": len(flow["manifest"]["frame_names"]), "fps_hz": flow["manifest"]["fps_hz"],
                "reference_frame_name": flow["manifest"]["reference_frame_name"],
                "reference_frame_index": flow["manifest"]["reference_frame_index"],
                "modes_file": str(Path(record["path"]) / "modal_image.npy"),
                "modes_file_sha256": sha256(Path(record["path"]) / "modal_image.npy"),
                "modes_dtype": "complex64", "modes_shape": [1, *view["shape_hw"], 2],
                "selected_source": record})
        scene, observation_renders = None, None
        config = nm.NeuralModesConfig.from_dict(parent.manifest["defaults"]["neural"])
        if scene_dir is None:
            topology_path = Path(source["topology"])
            topology_manifest = _manifest(topology_path)
            if (topology_manifest["topology_identity"] != source["topology_identity"]
                    or topology_manifest["foreground_identity"] != source["foreground_identity"]):
                raise ValueError("Inherited topology does not belong to the prepared geometry")
            with np.load(topology_path / ARRAY_FILENAME, allow_pickle=False) as archive:
                topology = TopologyArrays(**{name: archive[name] for name in ARRAY_DTYPES})
            points = parent.arrays["o_g_points"]
        else:
            from modal_gaussians.static import load_static_scene
            scene_path = Path(scene_dir).expanduser().resolve(strict=True)
            scene = load_static_scene(scene_path, "cuda").eval()
            if (scene.manifest["static_scene_identity"] != parent.source["static_scene_identity"]
                    and scene.manifest.get("partition", {}).get("source_static_scene_identity")
                    != parent.source["static_scene_identity"]):
                raise ValueError("Selected subject scene does not derive from the prepared static scene")
            for parameter in scene.parameters():
                parameter.requires_grad_(False)
            source.update(static_scene=str(scene_path), static_scene_identity=scene.manifest["static_scene_identity"],
                          foreground_identity=scene.manifest["foreground_identity"])
            report_progress("selected modal: rebuilding topology from visible selected subject")
            topology, observation_renders = _subject_topology(parent, source, scene, temporary, destination, config)
            points = scene.foreground.active()["means"].detach().cpu().numpy()
        bundle = {"format": FORMAT, "version": 1, "modes": [mode], "views": records,
            "topology_identity": source["topology_identity"], "transform": TRANSFORM_CONVENTION,
            "full_spectrum_available": False,
            "frequency_selection_identity": identity({"parent_prepared": parent.manifest["prepared_identity"], "mode": mode})}
        bundle["complex_2d_modes_identity"] = identity(bundle)
        previous_alignment = _manifest(source["alignment_from"])
        old_alpha_config = previous_alignment.get("alpha_sync", {})
        alpha_config = AlphaSyncConfig(**{f.name: old_alpha_config[f.name] for f in fields(AlphaSyncConfig)
                                         if f.name in old_alpha_config})
    with timer.stage("selected_modal_alignment"):
        report_progress("selected modal: estimating cross-view complex gains from SEA-RAFT observations")
        observations = prepare_observations(points=points, topology=topology,
            sample_measurements=_sample_fields(modal_fields, topology.sample_pixels_xy, topology.sample_view_index)[0],
            view_labels=tuple(labels))
        alpha = solve_alpha_sync(observations, alpha_config)
        alpha_arrays = {f.name: np.asarray(getattr(alpha, f.name)) for f in fields(alpha)}
        alignment = {"format": ALIGNMENT_FORMAT, "version": 1, "modes": [mode], "views": source["views"],
            "complex_2d_modes_identity": bundle["complex_2d_modes_identity"],
            "topology_identity": source["topology_identity"], "alpha_sync": alpha_config.to_dict(),
            "arrays_file": "arrays.npz", "arrays_identity": nm._arrays_identity(alpha_arrays)}
        alignment["alignment_identity"] = identity(alignment)
        for label, gain, usable, reason in zip(labels, alpha.alphas, alpha.identifiable_mask, alpha.exclusion_reason):
            report_progress(f"selected modal alignment {label}: alpha={gain} supervised={bool(usable)} reason={reason}")
    with timer.stage("selected_modal_observations"):
        if scene is None:
            target = _sample_fields(modal_fields, parent.arrays["o_sample_pixels_xy"], parent.arrays["o_sample_view_index"])
            arrays = _replace_observations(parent.arrays, target, alpha.alphas[None], alpha.identifiable_mask[None],
                                          config.energy_floor_fraction)
        else:
            dense = Complex2DModesArtifact(destination / "selected_modes", bundle, tuple(f[None] for f in modal_fields))
            observed, projectors, _, depths, alphas = nm._prepare_observation_arrays(
                scene, source, dense, config, scene.foreground.active()["means"].device,
                alpha.alphas[None], alpha.identifiable_mask[None], flow_loader=parent.flow,
                observation_renders=observation_renders)
            arrays = {"o_" + key: value for key, value in observed.items()}
            arrays["o_g_points"] = points
            for index, projector in enumerate(projectors):
                # Kept only for the immutable flow identity; subject sampling ignores this legacy mask.
                arrays[f"v{index}_mask"] = parent.arrays[f"v{index}_mask"].copy()
                arrays[f"v{index}_rgb"] = parent.arrays[f"v{index}_rgb"].copy()
                arrays[f"v{index}_jacobian"] = projector.jacobian.cpu().numpy()
                arrays[f"v{index}_depth"] = depths[index]
                arrays[f"v{index}_alpha"] = alphas[index]
                arrays[f"v{index}_subject_mask"] = alphas[index] >= config.alpha_minimum
        source.update(modes=[mode], complex_2d_modes=str(destination / "selected_modes"),
            complex_2d_modes_identity=bundle["complex_2d_modes_identity"],
            alignment_from=str(destination / "alignment"), alignment_identity=alignment["alignment_identity"],
            selected_modal_supervision={"parent_prepared": str(parent.path),
                "parent_prepared_identity": parent.manifest["prepared_identity"],
                "complex_2d_modes_identity": bundle["complex_2d_modes_identity"],
                "alignment_identity": alignment["alignment_identity"],
                "inherited_source_roles": {"topology": ("rebuilt_for_selected_subject" if scene is not None else
                                                        "static_pixel_gaussian_correspondence"),
                    "measurements": "historical_provenance_not_training_targets",
                    "flows": "reference_geometry_and_timing_only",
                    "observed_structure_graph": "historical_depth_tolerances_only"}})
        if visible_subject:
            source["selected_modal_supervision"]["observation_region"] = "visible_subject"
    with timer.stage("selected_modal_publish"):
        atomic_json(temporary / "selected_modes" / "manifest.json", bundle)
        atomic_json(temporary / "alignment" / "manifest.json", alignment)
        save_named_arrays(temporary / "alignment" / "arrays.npz", alpha_arrays)
        save_named_arrays(temporary / "arrays.npz", arrays)
        manifest = {"format": PREPARED_FORMAT, "version": 1, "source": source,
            "source_identity": nm._source_identity(source), "flows": parent.manifest["flows"],
            "cache_dir": str(parent.cache_dir), "arrays_sha256": sha256(temporary / "arrays.npz"),
            "arrays_identity": nm._arrays_identity(arrays), "defaults": parent.manifest["defaults"]}
        manifest["prepared_identity"] = identity(manifest)
        atomic_json(temporary / "manifest.json", manifest)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
    timer.save(destination / "timings.json")
    return PreparedNeuralInputs(destination, manifest, arrays)
