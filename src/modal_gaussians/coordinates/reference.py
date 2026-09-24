"""Publish a fixed motion reference independent of refinement recording selection."""
import numpy as np
import torch
from pathlib import Path
import sys
from modal_gaussians.common.cache import sha256, module_revision
from modal_gaussians.common.progress import Progress, report_progress
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.geometry.scene import load_static_scene
from modal_gaussians.motion.common.completed_modes import load_completed_modes
from modal_gaussians.motion import training, network, reference_field
from modal_gaussians.motion.reference_field import ReferenceField, prepare_paths
from .preparation import _publish

REFERENCE_FORMAT = "modal_gaussians.fixed_motion_reference"


def build_reference(scene, bank, *, source_loader=load_completed_modes, path_backend="cupy"):
    if bank.manifest.get("version") != 17 or bank.manifest['static_scene_identity'] != scene.manifest['static_scene_identity']:
        raise ValueError("Reference requires a matching fixed mode bank and scene")
    points = scene.foreground.params["means"].detach().cpu().numpy()
    source_arrays, controls, modes, checks = [], None, [], []
    if len(bank.manifest["sources"]) != len(bank.manifest["modes"]):
        raise ValueError("Mode-bank source count differs from its frequency order")
    progress = Progress('Reference: source validation and network replay', len(bank.manifest['sources']), unit='modes')
    for k, source in enumerate(bank.manifest["sources"]):
        artifact = source_loader(source["path"], validate=True)
        m, a, slot = artifact.manifest, artifact.arrays, source["slot"]
        if (m["version"] not in (16, 18) or m["completed_modes_identity"] != source["identity"]
                or m["static_scene_identity"] != scene.manifest["static_scene_identity"]
                or m["foreground_identity"] != scene.manifest["foreground_identity"]
                or type(slot) is not int or not 0 <= slot < len(m["modes"])
                or m["modes"][slot]["frequency_hz"] != bank.manifest["modes"][k]["frequency_hz"]
                or not np.array_equal(a["g_points"], points)):
            raise ValueError("Original model/scene/order differs; rebuild upstream inputs")
        binding_keys = ("label", "camera_identity", "shape_hw", "motion_reference")
        if ([{key:v.get(key) for key in binding_keys} for v in m["views"]]
                != [{key:v.get(key) for key in binding_keys} for v in bank.manifest["views"]]):
            raise ValueError("Original model cameras or motion references differ")
        ids = a["t_host_gaussian_index"][a["c_control_point_index"]]
        if controls is None:
            controls = ids
        elif not np.array_equal(controls, ids):
            raise ValueError("Frequencies do not share the same original control layout")
        if not np.array_equal(a["c_positions"], points[ids]):
            raise ValueError("Control positions differ from their protected Gaussians")
        if source_arrays:
            for key in ("g_edge_index", "g_edge_length", "c_coverage_radius"):
                if not np.array_equal(a[key], source_arrays[0][key]):
                    raise ValueError(f"Frequencies disagree on immutable geometry: {key}")
        state = torch.load(artifact.path / training.MODELS_FILENAME, map_location="cpu", weights_only=True)
        if (state["run_identity"] != m["run_identity"] or len(state['model_states']) != len(m['modes'])
                or state.get('source_mode_slots') != m.get('mode_selection', {}).get('source_mode_slots')):
            raise ValueError("Saved network run identity differs")
        amplitude, scene_scale = float(a['amplitude_scale'][slot]), float(a['scene_scale'])
        if not np.isfinite([amplitude, scene_scale]).all() or min(amplitude, scene_scale) <= 0:
            raise ValueError('Reference amplitude and scene scales must be positive')
        with torch.no_grad():
            fields = network.evaluate_model(state["model_states"][slot], training._field_geometry(a, slot),
                length_scale=float(a["scene_scale"]), amplitude_scale=float(a["amplitude_scale"][slot]),
                config=network.NeuralFieldConfig(**{key: m["config"][key] for key in ("hidden_dim", "message_layers", "local_feature_dim")} ))
        phi, omega, displacement, angular = [v.detach().numpy() for v in fields]
        amplitude = float(a["amplitude_scale"][slot])
        angular_scale = amplitude / float(a["scene_scale"])
        errors = []
        for actual, saved, norm in ((phi, bank.arrays["phi"][k], amplitude), (omega, bank.rotation[k], angular_scale)):
            error = float(np.max(np.abs(actual-saved)) / norm)
            errors.append(error)
            if not np.isfinite(error) or error > 1e-5:
                raise ValueError(f"Frequency {bank.manifest['modes'][k]['frequency_hz']:g} Hz: network field error {error:g} exceeds 1e-5")
        checks.append(dict(frequency_hz=bank.manifest["modes"][k]["frequency_hz"], network_errors=errors))
        source_arrays.append({**{key: a[key] for key in ("g_edge_index", "g_edge_length", "c_coverage_radius")},
            "propagation": a.get("g_edge_propagation_length", a["g_edge_length"]),
            "own": a["u_own_field_mask"][slot], "donor_ids": a["u_neighbor_index"][slot],
            "donor_weights": a["u_neighbor_weight"][slot], "displacement": displacement,
            "angular": angular, "control_valid": a["u_own_field_mask"][slot, ids],
            "alphas": a["alphas"][slot], "alpha_identifiable_mask": a["alpha_identifiable_mask"][slot],
            "amplitude_scale": a["amplitude_scale"][slot], "angular_scale": np.asarray(angular_scale)})
        modes.append({"path": str(artifact.path), "identity": source["identity"], "slot": slot,
                      "complex_2d_modes": m["complex_2d_modes"],
                      "complex_2d_modes_identity": m["complex_2d_modes_identity"],
                      "source_identity": m.get('source_identity'), "source_config": m['config'],
                      "arrays_sha256": m.get('arrays_file_sha256'), "networks_sha256": m.get('networks_sha256')})
        progress.update(k + 1, detail=f"{checks[-1]['frequency_hz']:g} Hz, errors={errors}", force=True)
    first = source_arrays[0]
    arrays = prepare_paths(points, first["g_edge_index"], first["g_edge_length"],
        np.stack([a["propagation"] for a in source_arrays]), controls,
        2 * float(first["c_coverage_radius"]), backend=path_backend)
    for name in ("own", "donor_ids", "donor_weights", "displacement", "angular", "control_valid",
                 "alphas", "alpha_identifiable_mask", "amplitude_scale", "angular_scale"):
        arrays[name] = np.stack([a[name] for a in source_arrays])
    arrays.update(support_class=bank.arrays["support_class"],
                  observation_view_mask=bank.arrays["observation_view_mask"])
    field = ReferenceField(arrays)
    progress = Progress('Reference: full-field path equivalence', len(modes), unit='modes')
    with torch.no_grad():
        for k in range(len(modes)):
            phi, omega, valid = field.mode(torch.from_numpy(points), np.arange(len(points)), k)
            checks[k]["query_errors"] = []
            if not bool(valid.all()):
                raise ValueError("Original geometry has an unsupported reference query")
            for actual, saved, norm in ((phi.numpy(), bank.arrays["phi"][k], arrays["amplitude_scale"][k]),
                                         (omega.numpy(), bank.rotation[k], arrays["angular_scale"][k])):
                error = float(np.max(np.abs(actual-saved)) / norm)
                checks[k]["query_errors"].append(error)
                if not np.isfinite(error) or error > 1e-5:
                    raise ValueError(f"Frequency {bank.manifest['modes'][k]['frequency_hz']:g} Hz: reference query error {error:g} exceeds 1e-5")
            progress.update(k + 1, detail=f"{checks[k]['frequency_hz']:g} Hz, errors={checks[k]['query_errors']}", force=True)
    return arrays, modes, checks


def publish_reference(*, scene_dir, completed_modes_dir, output_dir, source_loader=load_completed_modes, path_backend="cupy"):
    from .refinement_artifacts import write_manifest
    destination = resolve_path(output_dir)
    report_progress('Reference: validating static scene and mode-bank files')
    scene = load_static_scene(scene_dir, validate=True)
    bank = load_completed_modes(completed_modes_dir, validate=True)
    protected = [resolve_path(scene_dir), bank.path] + [resolve_path(s['path']) for s in bank.manifest['sources']]
    if destination.exists() or any(destination.is_relative_to(p) for p in protected):
        raise ValueError("Reference output must be new and outside sources")
    arrays, sources, checks = build_reference(scene, bank, source_loader=source_loader, path_backend=path_backend)
    with _publish(destination) as work:
        report_progress('Reference: publishing verified reference arrays and manifest')
        np.savez(work/'reference.npz', **arrays)
        write_manifest(work, dict(format=REFERENCE_FORMAT, version=1,
            static_scene_identity=scene.manifest['static_scene_identity'],
            foreground_identity=scene.manifest['foreground_identity'],
            completed_modes_identity=bank.manifest['completed_modes_identity'],
            modes=bank.manifest['modes'], original_views=bank.manifest['views'], mode_sources=sources,
            validation=checks, implementation=module_revision(sys.modules[__name__], network, reference_field, sys.modules[source_loader.__module__]),
            checksums={'reference.npz':sha256(work/'reference.npz')}), 'reference_identity')
    return destination


def load_reference(path, scene, bank):
    from .refinement_artifacts import read_manifest, _archive
    root, m = read_manifest(path, REFERENCE_FORMAT, 1, 'reference_identity')
    for key, expected in [('static_scene_identity',scene.manifest['static_scene_identity']),
                          ('foreground_identity',scene.manifest['foreground_identity']),
                          ('completed_modes_identity',bank.manifest['completed_modes_identity']),
                          ('modes',bank.manifest['modes']), ('original_views',bank.manifest['views'])]:
        if m[key] != expected: raise ValueError(f"Fixed reference {key} differs")
    if set(m['checksums']) != {'reference.npz'}: raise ValueError('Reference checksum inventory differs')
    a = _archive(root/'reference.npz')
    if not np.array_equal(a['points'],scene.foreground.params['means'].detach().cpu().numpy()):
        raise ValueError('Reference Gaussian order differs')
    return a, m['mode_sources'], m['validation'], m['reference_identity']
