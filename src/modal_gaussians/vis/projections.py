"""Per-view, per-mode projections; existing designs are read without repacking."""
from dataclasses import dataclass, fields
import tempfile
import threading

import numpy as np

from modal_gaussians.common.cache import atomic_json, identity, module_revision, publish_directory
from modal_gaussians.motion.common import projection
from modal_gaussians.common.scene_store import resolve_path, scene_cache
from modal_gaussians.geometry.scene import cameras_from_scene_manifest
from modal_gaussians.vis.inputs import read_json


@dataclass(frozen=True)
class ModalProjection:
    pixels: np.ndarray
    values: np.ndarray


class ViewerProjections:
    def __init__(self, result, work_dir=None):
        self.result = result
        self.gpu_lock = threading.RLock()
        self._samples = {}
        self._prepared = {}
        self.views = result.observation_views
        fallback = work_dir or (result.path.parent if result.path.is_file() else result.path)
        self.cache_dir = scene_cache(result.scene.manifest, resolve_path(fallback) / "cache") / "viewer_projection"

    def _contract(self, k, label):
        mode = self.result.modes[k]
        model, slot = mode.artifact.manifest, mode.slot
        view = next(v for v in self.views if v["label"] == label)
        dense = mode.modal_views[label]
        settings = projection.RenderedDesignConfig(**{f.name: mode.settings[f.name]
            for f in fields(projection.RenderedDesignConfig) if f.name in mode.settings})
        settings.validate()
        return {"implementation": "viewer_modal_projection_v1", "revision": module_revision(projection),
            "model_identity": model["completed_modes_identity"], "slot": slot,
            "static_scene_identity": model["static_scene_identity"],
            "foreground_identity": model["foreground_identity"],
            "view": {key: view.get(key) for key in ("label", "camera_identity", "shape_hw", "motion_reference")},
            "sampling_source": {key: dense.get(key) for key in ("flow_identity", "flow_artifact", "flow_role")},
            "settings": settings.to_dict()}, settings

    @staticmethod
    def _check(value, shape):
        pixels, values = value.pixels, value.values
        if (pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) == 0
                or not np.issubdtype(pixels.dtype, np.integer)
                or np.any(pixels < 0) or np.any(pixels >= [shape[1], shape[0]])
                or values.shape != (len(pixels), 2) or values.dtype != np.complex64):
            raise ValueError("Viewer projection pixel/value domain differs")
        return value

    def get(self, k, label, *, compute=False):
        key = k, label
        if key in self._samples:
            return self._samples[key]
        view = next(v for v in self.views if v["label"] == label)
        mode = self.result.modes[k]
        design, slot = mode.design, mode.design_slot
        if design is not None:
            by_label = {v["label"]: i for i, v in enumerate(design.manifest["views"])}
            if label in by_label:
                v = by_label[label]
                lo, hi = design.samples["view_sample_offsets"][v:v+2]
                matrix = design.design[lo:hi]
                value = ModalProjection(np.asarray(design.samples["sample_pixels_xy"][lo:hi]),
                    np.asarray(matrix[:, :, 2*slot] - 1j * matrix[:, :, 2*slot+1], np.complex64))
                self._samples[key] = self._check(value, view["shape_hw"])
                return self._samples[key]
        contract, settings = self._contract(k, label)
        destination = self.cache_dir / identity(contract)
        if destination.exists():
            if read_json(destination / "manifest.json") != contract:
                raise ValueError("Viewer projection cache contract differs")
            with np.load(destination / "projection.npz", allow_pickle=False) as saved:
                value = ModalProjection(saved["pixels"], saved["values"])
        elif compute:
            value = self._compute(k, label, settings)
            self._check(value, view["shape_hw"])
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".projection-", dir=self.cache_dir) as temporary:
                work = resolve_path(temporary)
                np.savez(work / "projection.npz", pixels=value.pixels, values=value.values)
                atomic_json(work / "manifest.json", contract)
                try:
                    publish_directory(work, destination)
                except FileExistsError:
                    if read_json(destination / "manifest.json") != contract:
                        raise ValueError("Concurrent projection cache contract differs")
        else:
            return None
        self._samples[key] = self._check(value, view["shape_hw"])
        return self._samples[key]

    def _compute(self, k, label, settings):
        from modal_gaussians.preprocessing.reference import load_reference, reference_identity
        mode = self.result.modes[k]
        scene = self.result.scene
        camera = next(c for c in cameras_from_scene_manifest(scene.manifest) if c.label == label)
        device = scene.foreground.active()["means"].device
        camera = camera.to(device)
        dense = mode.modal_views[label]
        mask = None
        if not projection.uses_visible_subject(scene):
            if mode.prepared is not None:
                from modal_gaussians.motion.prepared import load_prepared
                if mode.prepared not in self._prepared:
                    prepared = load_prepared(mode.prepared)
                    prepared.arrays = {f"v{i}_mask": prepared.arrays[f"v{i}_mask"]
                                       for i in range(len(prepared.manifest["flows"]))}
                    self._prepared[mode.prepared] = prepared
                flow = self._prepared[mode.prepared].flow(dense["flow_artifact"])
            else:
                flow = load_reference(resolve_path(dense["flow_artifact"], strict=True))
            if reference_identity(flow) != dense["flow_identity"]:
                raise ValueError("Viewer sampling flow identity differs")
            mask = flow.arrays.mask_union
        with self.gpu_lock:
            pixels, alpha, jacobian, _ = projection.prepare_modal_projection(scene, camera, mask, settings)
            packed = projection.project_modal_features(scene, camera,
                mode.artifact.arrays["phi"][mode.slot:mode.slot+1], pixels, alpha, jacobian)
        return ModalProjection(pixels, np.asarray(packed[:, :, 0] - 1j * packed[:, :, 1], np.complex64))
