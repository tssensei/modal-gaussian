"""Reuse frozen geometry with independently computed selected modal observations."""
from __future__ import annotations

from dataclasses import fields
import copy
import math
from pathlib import Path
from modal_gaussians.common.scene_store import resolve_path
import shutil
import tempfile

import numpy as np

from modal_gaussians.common.cache import Timings, atomic_json, identity, sha256, publish_directory as _publish_prepared
from modal_gaussians.spectrum.modes import Complex2DModesArtifact, TRANSFORM_CONVENTION, _validate_mode_records
from modal_gaussians.common.numpy_io import save_named_arrays
from modal_gaussians.common.progress import report_progress
from modal_gaussians.motion.modal_similarity_artifact import _manifest, _modal_view
from modal_gaussians.motion.prepared import FORMAT as PREPARED_FORMAT, PreparedNeuralInputs, load_prepared
from modal_gaussians.motion import training as nm

FORMAT = "modal_gaussians.selected_complex_2d_modes"
ALIGNMENT_FORMAT = "modal_gaussians.selected_modal_alignment"


def load_selected_modal_bundle(path, manifest=None):
    """Open selected fields without scanning arrays or loading the flow sequences."""
    root = resolve_path(path, strict=True)
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
        value = np.load(resolve_path(view["modes_file"]), mmap_mode="r", allow_pickle=False)
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
    return result


def prepare_selected_modal(*, prepared_dir, views, frequency_hz, output_dir,
                           alpha_backend="cupy", alpha_workspace=None):
    """Publish one new observation snapshot, with no graph build or training."""
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Selected modal frequency must be finite and positive")
    destination = resolve_path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-writing-", dir=destination.parent))
    owned = alpha_backend == "cupy" and alpha_workspace is None
    try:
        if owned:
            from modal_gaussians.motion.observations.alpha_gpu import Workspace
            alpha_workspace = Workspace()
        if alpha_workspace is not None:
            alpha_workspace.stats = {"cold_initialization_seconds":
                0. if getattr(alpha_workspace, "initialization_reported", False) else alpha_workspace.initialization_seconds}
            alpha_workspace.initialization_reported = True
        return _prepare_selected_modal(prepared_dir=prepared_dir, views=views, frequency_hz=frequency_hz,
                                       destination=destination, temporary=temporary,
                                       alpha_backend=alpha_backend, alpha_workspace=alpha_workspace)
    finally:
        if owned and alpha_workspace is not None:
            alpha_workspace.close()
        if temporary.exists():
            resolved = temporary.resolve()
            if resolved.parent != destination.parent.resolve() or not resolved.name.startswith(f".{destination.name}-writing-"):
                raise RuntimeError("Refusing to remove a preparation temporary directory outside its output parent")
            shutil.rmtree(resolved)


def _prepare_selected_modal(*, prepared_dir, views, frequency_hz, destination, temporary,
                            alpha_backend, alpha_workspace):
    from modal_gaussians.motion.observations.alpha import AlphaSyncConfig, prepare_observations, solve_alpha_sync, backend_identity
    from modal_gaussians.motion.observations.topology import TopologyArrays, ARRAY_FILENAME, ARRAY_DTYPES

    timer = Timings()
    alpha_contract = backend_identity(alpha_backend)
    with timer.stage("selected_modal_sources"):
        parent = load_prepared(prepared_dir)
        source = copy.deepcopy(parent.source)
        visible_subject = source["observation_region"] == "visible_subject"
        requested = dict(views)
        labels = [v["label"] for v in source["views"]]
        if len(requested) != len(views) or set(requested) != set(labels):
            raise ValueError("Selected modal sources must provide every prepared view exactly once")
        selected = [m for m in source["modes"] if math.isclose(m["frequency_hz"], frequency_hz, rel_tol=0, abs_tol=1e-9)]
        candidate_index = selected[0]["candidate_index"] if len(selected) == 1 else 0
        modal_fields, records = [], []
        for view, flow in zip(source["views"], parent.manifest["flows"]):
            field, _, record = _modal_view(requested[view["label"]], flow, view, frequency_hz,
                                         read_mask=not visible_subject,
                                         static_scene_identity=source["static_scene_identity"])
            exported = record["manifest"]
            view["motion_reference"] = {"reference_frame_name": exported["reference_frame_name"],
                "reference_frame_index": exported["reference_frame_index"],
                "selection_identity": exported["reference_selection"]["identity"]}
            modal_fields.append(field)
            records.append({**view, "flow_artifact": flow["path"], "flow_role": "geometry_reference_only",
                "frame_count": len(flow["manifest"]["frame_names"]), "fps_hz": flow["manifest"]["fps_hz"],
                "reference_frame_name": exported["reference_frame_name"],
                "reference_frame_index": exported["reference_frame_index"],
                "modes_file": str(Path(record["path"]) / "modal_image.npy"),
                "modes_file_sha256": sha256(Path(record["path"]) / "modal_image.npy"),
                "modes_dtype": "complex64", "modes_shape": [1, *view["shape_hw"], 2],
                "selected_source": record})
        spectrum_sources = [r["selected_source"]["manifest"]["spectrum_source"] for r in records
                            if r["selected_source"].get("manifest", {}).get("format")
                            == "modal_gaussians.spectrum_selected_frequency"]
        if spectrum_sources:
            grid = spectrum_sources[0]
            if any(s["fft_length"] != grid["fft_length"] or s["bin_index"] != grid["bin_index"]
                   or not math.isclose(s["frequency_step_hz"], grid["frequency_step_hz"], rel_tol=0, abs_tol=1e-12)
                   for s in spectrum_sources[1:]):
                raise ValueError("Selected spectrum views must share the same FFT grid and bin")
            candidate_index = grid["bin_index"]
            frequency_hz = candidate_index * grid["frequency_step_hz"]
        mode = {"mode_slot": 0, "candidate_index": candidate_index, "frequency_hz": float(frequency_hz)}
        config = nm.NeuralModesConfig.from_dict(parent.manifest["defaults"]["neural"])
        topology_path = resolve_path(source["topology"])
        topology_manifest = _manifest(topology_path)
        if (topology_manifest["topology_identity"] != source["topology_identity"]
                or topology_manifest["foreground_identity"] != source["foreground_identity"]):
            raise ValueError("Inherited topology does not belong to the prepared geometry")
        with np.load(topology_path / ARRAY_FILENAME, allow_pickle=False) as archive:
            topology = TopologyArrays(**{name: archive[name] for name in ARRAY_DTYPES})
        points = parent.arrays["o_g_points"]
        bundle = {"format": FORMAT, "version": 1, "modes": [mode], "views": records,
            "topology_identity": source["topology_identity"], "transform": TRANSFORM_CONVENTION,
            "full_spectrum_available": False,
            "frequency_selection_identity": identity({"parent_prepared": parent.manifest["prepared_identity"],
                "mode": mode, **({"spectrum_sources": spectrum_sources} if spectrum_sources else {})})}
        bundle["complex_2d_modes_identity"] = identity(bundle)
        alpha_config = AlphaSyncConfig(**{f.name: parent.manifest['alpha_config'][f.name]
                                         for f in fields(AlphaSyncConfig)})
    with timer.stage("selected_modal_alignment"):
        report_progress("selected modal: estimating cross-view complex gains from selected observations")
        observations = prepare_observations(points=points, topology=topology,
            sample_measurements=_sample_fields(modal_fields, topology.sample_pixels_xy, topology.sample_view_index)[0],
            view_labels=tuple(labels), workspace=alpha_workspace,
            topology_identity=source["topology_identity"], cache_dir=parent.cache_dir)
        alpha = solve_alpha_sync(observations, alpha_config, backend=alpha_backend, workspace=alpha_workspace)
        alpha_arrays = {f.name: np.asarray(getattr(alpha, f.name)) for f in fields(alpha)}
        alignment = {"format": ALIGNMENT_FORMAT, "version": 1, "modes": [mode], "views": source["views"],
            "complex_2d_modes_identity": bundle["complex_2d_modes_identity"],
            "topology_identity": source["topology_identity"], "alpha_sync": alpha_config.to_dict(),
            "alpha_backend": alpha_contract,
            "arrays_file": "arrays.npz", "arrays_identity": nm._arrays_identity(alpha_arrays)}
        alignment["alignment_identity"] = identity(alignment)
        for label, gain, usable, reason in zip(labels, alpha.alphas, alpha.identifiable_mask, alpha.exclusion_reason):
            report_progress(f"selected modal alignment {label}: alpha={gain} supervised={bool(usable)} reason={reason}")
    with timer.stage("selected_modal_observations"):
        target = _sample_fields(modal_fields, parent.arrays["o_sample_pixels_xy"], parent.arrays["o_sample_view_index"])
        arrays = _replace_observations(parent.arrays, target, alpha.alphas[None], alpha.identifiable_mask[None],
                                      config.energy_floor_fraction)
        source.update(modes=[mode], complex_2d_modes=str(destination / "selected_modes"),
            complex_2d_modes_identity=bundle["complex_2d_modes_identity"],
            alignment_from=str(destination / "alignment"), alignment_identity=alignment["alignment_identity"],
            selected_modal_supervision={"parent_prepared": str(parent.path),
                "parent_prepared_identity": parent.manifest["prepared_identity"],
                "complex_2d_modes_identity": bundle["complex_2d_modes_identity"],
                "alignment_identity": alignment["alignment_identity"],
                "alpha_backend": alpha_contract,
                'observation_region': source['observation_region']})
    with timer.stage("selected_modal_publish"):
        atomic_json(temporary / "selected_modes" / "manifest.json", bundle)
        atomic_json(temporary / "alignment" / "manifest.json", alignment)
        save_named_arrays(temporary / "alignment" / "arrays.npz", alpha_arrays)
        save_named_arrays(temporary / "arrays.npz", arrays)
        manifest = {"format": PREPARED_FORMAT, "version": 2, "source": source,
            "source_identity": nm._source_identity(source), "flows": parent.manifest["flows"],
            "cache_dir": str(parent.cache_dir), "arrays_sha256": sha256(temporary / "arrays.npz"),
            "arrays_identity": nm._arrays_identity(arrays), "defaults": parent.manifest["defaults"],
            "geometry_graph": parent.manifest["geometry_graph"], "alpha_config": parent.manifest["alpha_config"]}
        manifest["prepared_identity"] = identity(manifest)
        atomic_json(temporary / "manifest.json", manifest)
        _publish_prepared(temporary, destination)
    timer.save(destination / "timings.json")
    if alpha_workspace is not None:
        alpha_workspace.stats["prepared_publish_seconds"] = timer.records[-1]["seconds"]
        atomic_json(destination / "alpha_timings.json", alpha_workspace.stats)
    return PreparedNeuralInputs(destination, manifest, arrays)
