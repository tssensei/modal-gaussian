"""Immutable preparation and publication for fixed-reference scene refinement."""
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from modal_gaussians.common.cache import atomic_json, identity, module_revision, sha256
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.geometry.scene import (PROJECTION_CONVENTION,
    cameras_from_scene_manifest, load_static_scene, tensor_dictionary_identity)
from modal_gaussians.motion.common.completed_modes import CompletedModesArtifact, load_completed_modes
from modal_gaussians.motion.reference_field import ReferenceField, prepare_paths
from .preparation import _publish
from .rgb import RGBModalCoordinatesArtifact
from .sequences import bind_fixed_view, reindex_views, validate_sequences

PREPARED_FORMAT = "modal_gaussians.refinement_preparation"
COORDINATES_FORMAT = "modal_gaussians.refined_rgb_coordinates"


def manifest_identity(m, name):
    return identity({k: v for k, v in m.items() if k != name})


def write_manifest(path, manifest, name):
    manifest[name] = manifest_identity(manifest, name)
    atomic_json(Path(path) / "manifest.json", manifest)


def read_manifest(path, format_name, version, name):
    root = resolve_path(path, strict=True)
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (m.get("format") != format_name or m.get("version") != version
            or m.get(name) != manifest_identity(m, name)):
        raise ValueError(f"Invalid {format_name} manifest or identity")
    for filename, digest in m.get("checksums", {}).items():
        if Path(filename).name != filename or sha256(root / filename) != digest:
            raise ValueError(f"Artifact checksum differs: {filename}")
    return root, m


def _archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def prepare_refinement(*, scene_dir, completed_modes_dir, coordinates_dirs, output_dir,
                       path_backend="cupy", view_labels=None, sweep_coordinates_dir=None, reference_dir=None):
    """Validate all recordings before graph propagation; never launch RGB fitting."""
    from modal_gaussians.results.artifact import _load_sources
    from modal_gaussians.motion import training, network, reference_field
    started = time.perf_counter()
    destination = resolve_path(output_dir)
    paths = [resolve_path(p, strict=True) for p in (scene_dir, completed_modes_dir, *coordinates_dirs)]
    extra_paths = [resolve_path(p, strict=True) for p in (reference_dir, sweep_coordinates_dir) if p is not None]
    if len(paths) < 3 or any(destination.is_relative_to(p) for p in paths + extra_paths):
        raise ValueError("Preparation needs RGB inputs and a separate output directory")
    if destination.exists():
        raise FileExistsError(destination)
    scene = load_static_scene(paths[0], validate=True)
    bank = load_completed_modes(paths[1], validate=True)
    if bank.manifest.get("version") != 17:
        raise ValueError("Refinement preparation requires a current fixed-frequency mode bank")
    by_label, rgb_sources = {}, []
    for path in paths[2:]:
        _, _, _, kind, rgb, _, direct, views = _load_sources(
            scene_dir=paths[0], completed_modes_dir=paths[1], coordinates_dir=path)
        if kind != "rgb":
            raise ValueError("Refinement requires completed fixed-mode RGB fitting")
        source_index = {v["label"]: i for i, v in enumerate(direct.manifest["views"])}
        images = {v["label"]: v for v in rgb.manifest["images"]}
        for v in views:
            label = v["label"]
            if label in by_label:
                raise ValueError(f"Duplicate RGB recording: {label}")
            lo, count = v["frame_offset"], v["frame_count"]
            by_label[label] = (v, rgb.coordinates[lo:lo+count], images[label],
                              direct.diagnostics["mode_pair_scales"][source_index[label]])
        rgb_sources.append({"path": str(path), "identity": rgb.manifest["rgb_coordinates_identity"]})
    available = [v["label"] for v in bank.manifest["views"]]
    labels = available if view_labels is None else list(view_labels)
    if not labels or len(set(labels)) != len(labels) or not set(labels) <= set(available):
        raise ValueError("Selected refinement views must be unique mode-bank recordings")
    if not set(labels) <= set(by_label):
        raise ValueError(f"RGB recording coverage differs; missing={sorted(set(labels)-set(by_label))}, "
                         f"extra={sorted(set(by_label)-set(labels))}. Run fit-rgb explicitly first.")
    views, qs, images, scales, offset = [], [], [], [], 0
    for i, label in enumerate(labels):
        v, q, image, scale = by_label[label]
        directory = resolve_path(image["directory"], strict=True)
        if destination.is_relative_to(directory):
            raise ValueError('Preparation output must be outside immutable input frames')
        if [f["name"] for f in image["files"]] != [f"{name}.png" for name in v["frame_names"]]:
            raise ValueError(f"RGB image ordering differs for {label}")
        missing = [f["name"] for f in image["files"]
                   if Path(f["name"]).name != f["name"] or not (directory / f["name"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing refinement PNGs for {label}: {missing[:8]}")
        views.append(bind_fixed_view({**v, "index": i, "frame_offset": offset}, image))
        offset += v["frame_count"]
        qs.append(q); images.append(image); scales.append(scale)
    if sweep_coordinates_dir is not None:
        from .sweep import load_sweep_coordinates
        sweep = load_sweep_coordinates(sweep_coordinates_dir)
        if sweep.manifest['views'][0]['fps_hz'] != 30:
            raise ValueError('Refinement requires 30 FPS sweep; publish a downsample-sweep artifact first')
        if (sweep.manifest['static_scene_identity'] != scene.manifest['static_scene_identity']
                or sweep.manifest['completed_modes_identity'] != bank.manifest['completed_modes_identity']
                or sweep.manifest['modes'] != bank.manifest['modes']):
            raise ValueError('Sweep initialization scene/modes differ')
        views.extend(sweep.manifest['views']); qs.append(sweep.coordinates)
        images.extend(sweep.manifest['images']); scales.append(sweep.manifest['pair_scales'])
        rgb_sources.append(dict(path=str(sweep.path), identity=sweep.manifest['sweep_coordinates_identity']))
    views = reindex_views(views)
    if any(destination.is_relative_to(resolve_path(image['directory'])) for image in images):
        raise ValueError('Preparation output must be outside immutable input frames')
    validate_sequences(views, images, scene.manifest, frame_count=sum(len(q) for q in qs))
    from .reference import build_reference, load_reference
    from . import reference, sequences
    if reference_dir is None:
        arrays, modes, checks = build_reference(scene, bank, source_loader=load_completed_modes, path_backend=path_backend)
        reference_source_identity = None
    else:
        arrays, modes, checks, reference_source_identity = load_reference(reference_dir, scene, bank)
    scales = np.asarray(scales, np.float32)
    if scales.shape != (len(views), len(modes)) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid frozen coefficient normalization")
    with _publish(destination) as work:
        np.savez(work / "reference.npz", **arrays)
        np.savez(work / "initial.npz", coordinates=np.concatenate(qs), scales=scales)
        m = dict(format=PREPARED_FORMAT, version=2, static_scene=str(paths[0]),
            static_scene_identity=scene.manifest["static_scene_identity"],
            mode_bank=str(paths[1]), completed_modes_identity=bank.manifest["completed_modes_identity"],
            rgb_sources=rgb_sources, mode_sources=modes, modes=bank.manifest["modes"], views=views,
            images=images, original_views=bank.manifest["views"],
            reference_source=None if reference_dir is None else str(resolve_path(reference_dir)),
            reference_source_identity=reference_source_identity, reference_validation=checks,
            implementation=module_revision(sys.modules[__name__], reference_field, network, reference, sequences),
            preparation_seconds=time.perf_counter()-started,
            checksums={f: sha256(work / f) for f in ("reference.npz", "initial.npz")})
        m["reference_identity"] = m["checksums"]["reference.npz"]
        write_manifest(work, m, "preparation_identity")
    return destination


def load_prepared(path):
    root, m = read_manifest(path, PREPARED_FORMAT, 2, "preparation_identity")
    if set(m["checksums"]) != {"reference.npz", "initial.npz"} or m["reference_identity"] != m["checksums"]["reference.npz"]:
        raise ValueError("Prepared reference file contract differs")
    scene = load_static_scene(m["static_scene"], validate=True)
    if scene.manifest["static_scene_identity"] != m["static_scene_identity"]:
        raise ValueError("Prepared parent scene changed")
    arrays, initial = _archive(root / "reference.npz"), _archive(root / "initial.npz")
    if not np.array_equal(arrays["points"], scene.foreground.params["means"].detach().numpy()):
        raise ValueError("Prepared reference graph order differs from parent")
    validate_sequences(m["views"], m["images"], scene.manifest, frame_count=len(initial["coordinates"]))
    return root, m, scene, arrays, initial


def load_refined_coordinates(path):
    root, m = read_manifest(path, COORDINATES_FORMAT, 2, "refined_coordinates_identity")
    q = np.load(root / "coordinates.npy", allow_pickle=False)
    from .direct import _validate_modes
    _validate_modes(m["modes"])
    offset = 0
    for i, v in enumerate(m["views"]):
        if (v["index"] != i or v["frame_offset"] != offset or v["frame_count"] < 1
                or len(v["frame_names"]) != v["frame_count"] or not np.isfinite(v["fps_hz"]) or v["fps_hz"] <= 0
                or len(set(v["frame_names"])) != v["frame_count"]
                or len(v["frames"]) != v["frame_count"]):
            raise ValueError("Invalid refined recording layout")
        offset += v["frame_count"]
    if (q.dtype != np.complex64 or q.shape != (offset, len(m["modes"])) or not np.isfinite(q).all()
            or not m["views"] or len({v["label"] for v in m["views"]}) != len(m["views"])
            or m["counts"] != {"frames": offset, "views": len(m["views"]), "modes": q.shape[1]}
            or set(m["checksums"]) != {"coordinates.npy"}):
        raise ValueError("Invalid refined coefficients")
    images = m["images"]
    if [v["label"] for v in images] != [v["label"] for v in m["views"]]:
        raise ValueError("Refined image-source order differs")
    for view, image in zip(m["views"], images):
        if [f["name"] for f in image["files"]] != [f"{name}.png" for name in view["frame_names"]]:
            raise ValueError("Refined image frames differ")
    return RGBModalCoordinatesArtifact(root, m, q)


def load_refined_modes(path):
    root, m = read_manifest(path, "modal_gaussians.completed_modes", 19, "completed_modes_identity")
    a = _archive(root / "support.npz")
    from .direct import _validate_modes
    _validate_modes(m["modes"])
    if set(m["checksums"]) != {"phi.npy", "rotation.npy", "support.npz"}:
        raise ValueError("Refined mode file inventory differs")
    for name in ("phi", "rotation"):
        a[name] = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if (a[name].dtype != np.complex64 or a[name].shape !=
                (len(m["modes"]), m["counts"]["foreground_gaussians"], 3) or not np.isfinite(a[name]).all()):
            raise ValueError("Invalid refined field array")
    if a["g_points"].shape != a["phi"].shape[1:] or a["root_id"].shape != a["phi"].shape[1:2]:
        raise ValueError("Refined field identity map differs")
    k, g, _ = a["phi"].shape
    v, n = len(m["views"]), len(a["reference_points"])
    if (a["support_class"].shape != (k,g) or a["support_class"].dtype != np.int8
            or np.any((a["support_class"] < 0) | (a["support_class"] > 3))
            or a["observation_view_mask"].shape != (k,g,v) or a["observation_view_mask"].dtype != bool
            or a["alphas"].shape != (k,v) or a["alpha_identifiable_mask"].shape != (k,v)
            or a["reference_points"].shape != (n,3) or a["reference_edges"].ndim != 2
            or a["reference_edges"].shape[1] != 2 or np.any(a["reference_edges"] < 0) or np.any(a["reference_edges"] >= n)
            or a["root_id"].dtype != np.int64 or np.any(a["root_id"] < 0) or np.any(a["root_id"] >= n)
            or len(np.unique(a["uid"])) != g or any(a[key].shape != (g,) for key in ("uid", "protected", "birth_step"))
            or a["protected"].dtype != bool or len(m["mode_sources"]) != k
            or any(not np.isfinite(value).all() for value in a.values() if value.dtype.kind in 'fc')):
        raise ValueError("Invalid refined support/reference domains")
    return CompletedModesArtifact(root, m, a, a.pop("rotation"), a.get("control_displacement"))


def publish_refinement(destination, prepared, scene, field, mapping, coordinates, *, run_identity, settings, baked_fields):
    """Publish the final frozen basis and all three linked artifacts atomically."""
    destination = resolve_path(destination)
    root, source = prepared
    if not bool(field.shape_valid(scene.foreground.params['means'], mapping['root_id'],
                                 settings['shape_radius_fraction']).all()):
        raise ValueError('Final positions violate the fixed graph shape constraint')
    tensors = {f"{part}.{name}": value.detach().cpu().contiguous()
               for part in ("foreground", "background") for name, value in getattr(scene, part).params.items()}
    with _publish(destination) as work:
        for name in ("scene", "mode_bank", "coordinates"):
            (work / name).mkdir()
        scene_path = work / "scene"
        torch.save(tensors, scene_path / "tensors.pt")
        np.savez(scene_path / "identity_map.npz", **mapping)
        m = copy.deepcopy(scene.manifest)
        for name in ("partition", "partition_identity", "partition_source", "partition_source_path", "partition_files"):
            m.pop(name, None)
        m.update(version=4, tensors_sha256=sha256(scene_path / "tensors.pt"),
            foreground_identity=tensor_dictionary_identity(tensors, "foreground."),
            background_identity=tensor_dictionary_identity(tensors, "background."),
            refinement={"parent_scene": source["static_scene"], "parent_scene_identity": source["static_scene_identity"],
                "prepared": str(root), "preparation_identity": source["preparation_identity"],
                "reference_identity": source["reference_identity"], "run_identity": run_identity,
                "reference_count": len(field.a["points"]),
                "birth_step_clock": "geometry_update",
                "parent_partition_identity": scene.manifest.get("partition_identity"),
                "observation_region": "visible_subject" if scene.manifest.get("partition", {}).get("method") == "manual_subject_selection_v1" else "mask_union",
                "identity_map_sha256": sha256(scene_path / "identity_map.npz")})
        m["counts"] = {**m.get("counts", {}), "foreground": scene.foreground.count,
                       "background": scene.background.count, "total": scene.foreground.count+scene.background.count}
        fg, bg = scene.foreground.count, scene.background.count
        m["representation"].update(foreground_local_index_domain=[0, fg], background_local_index_domain=[0, bg],
            combined_foreground_index_domain=[0, fg], combined_background_index_domain=[fg, fg+bg],
            foreground_role="refined_motion_subject", background_role="fixed_parent_background")
        payload = dict(dataset_identity=m["dataset"]["dataset_identity"], foreground_identity=m["foreground_identity"],
            background_identity=m["background_identity"], normalization=m["scene_normalization"],
            representation="vanilla_3dgs_direct_rgb", projection_convention=PROJECTION_CONVENTION,
            camera_identities=[c.to_manifest_record()["camera_identity"] for c in cameras_from_scene_manifest(m)],
            refinement=m["refinement"])
        m["static_scene_identity"] = identity(payload)
        atomic_json(scene_path / "manifest.json", m)
        x = scene.foreground.params["means"].detach()
        shape = (field.mode_count, len(x), 3)
        bank_path = work / "mode_bank"
        phi = np.lib.format.open_memmap(bank_path / "phi.npy", mode="w+", dtype=np.complex64, shape=shape)
        omega = np.lib.format.open_memmap(bank_path / "rotation.npy", mode="w+", dtype=np.complex64, shape=shape)
        try:
            if len(baked_fields) != 2:
                raise ValueError('Final displacement and angular fields are required')
            for target, values in zip((phi, omega), baked_fields):
                if values.shape != shape or values.dtype != torch.complex64 or not torch.isfinite(values).all():
                    raise ValueError('Final frozen basis differs from live Gaussian order')
                target[:] = values.detach().cpu().numpy()
            phi.flush(); omega.flush()
        finally:
            phi._mmap.close(); omega._mmap.close()
        a, roots = field.a, mapping["root_id"]
        control_rows = np.asarray([np.flatnonzero((roots == c) & mapping["protected"])[0] for c in a["controls"]])
        np.savez(bank_path / "support.npz", g_points=x.cpu().numpy(), **mapping,
            support_class=a["support_class"][:, roots], observation_view_mask=a["observation_view_mask"][:, roots],
            alphas=a["alphas"], alpha_identifiable_mask=a["alpha_identifiable_mask"],
            reference_points=a["points"], reference_edges=a["edges"], reference_controls=a["controls"],
            reference_weights=a["lengths"][None] / a["propagation"],
            c_control_point_index=control_rows, c_positions=a["points"][a["controls"]],
            control_displacement=a["displacement"].astype(np.complex64))
        b = dict(format="modal_gaussians.completed_modes", version=19, completion_method="fixed_reference_refinement",
            static_scene=str(destination / "scene"), static_scene_identity=m["static_scene_identity"],
            foreground_identity=m["foreground_identity"], reference_identity=source["reference_identity"],
            prepared=str(root), preparation_identity=source["preparation_identity"], run_identity=run_identity,
            modes=source["modes"], views=source["original_views"], mode_sources=source["mode_sources"],
            semantics={"observation_roles": "inherited_mode_sources", "motion": "local_extension_of_fixed_reference_field"},
            counts={"modes": shape[0], "foreground_gaussians": shape[1], "views": len(source["original_views"])},
            checksums={f: sha256(bank_path / f) for f in ("phi.npy", "rotation.npy", "support.npz")},
            quality_gate={"status": "refined_scene_candidate_unapproved"})
        write_manifest(bank_path, b, "completed_modes_identity")
        coord_path = work / "coordinates"
        np.save(coord_path / "coordinates.npy", np.asarray(coordinates, np.complex64), allow_pickle=False)
        c = dict(format=COORDINATES_FORMAT, version=2, static_scene_identity=m["static_scene_identity"],
            completed_modes=str(destination / "mode_bank"), completed_modes_identity=b["completed_modes_identity"],
            modes=source["modes"], views=source["views"], images=source["images"], settings=settings,
            initialization=source["rgb_sources"], preparation_identity=source["preparation_identity"],
            run_identity=run_identity, quality_gate={"status": "refined_rgb_candidate_unapproved"},
            counts={"views": len(source["views"]), "frames": len(coordinates), "modes": shape[0]},
            checksums={"coordinates.npy": sha256(coord_path / "coordinates.npy")})
        write_manifest(coord_path, c, "refined_coordinates_identity")
        write_manifest(work, dict(format="modal_gaussians.refined_scene", version=1,
            static_scene_identity=m["static_scene_identity"], completed_modes_identity=b["completed_modes_identity"],
            refined_coordinates_identity=c["refined_coordinates_identity"], preparation_identity=source["preparation_identity"],
            run_identity=run_identity), "refinement_identity")
    return destination
