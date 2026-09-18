"""Compare fixed-camera modal projections without fitting away shape changes."""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading

import numpy as np
import torch

from modal_gaussians.motion.neural.neural_modes import FrozenModalProjector
from modal_gaussians.vis.spectrum import PREVIEW_PERCENTILE, modal_image_overlay


class RGBModalComparison:
    def __init__(self, data):
        self.data = data
        self.path = Path(data.manifest["prepared"]).expanduser().resolve(strict=True)
        self.prepared = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        source = self.prepared["source"]
        if (self.prepared.get("format") != "modal_gaussians.neural_prepared"
                or self.prepared.get("version") != 1
                or self.prepared["prepared_identity"] != data.manifest["prepared_identity"]
                or source["static_scene_identity"] != data.manifest["static_scene_identity"]):
            raise ValueError("RGB modal comparison preparation differs from the result")
        self.slots = []
        for mode in data.manifest["modes"]:
            slot = mode["source_mode_slot"]
            if (type(slot) is not int or not 0 <= slot < len(source["modes"])
                    or slot in self.slots or not math.isclose(mode["frequency_hz"],
                        source["modes"][slot]["frequency_hz"], rel_tol=0, abs_tol=1e-9)):
                raise ValueError("RGB mode does not match its prepared source slot")
            self.slots.append(slot)
        self.view_indices = {view["label"]: index for index, view in enumerate(source["views"])}
        self.cameras = {camera.label: camera.camera for camera in data.cameras}
        if (len(self.view_indices) != len(source["views"])
                or len(self.cameras) != len(data.cameras)
                or set(self.view_indices) != set(self.cameras)):
            raise ValueError("RGB and prepared reference view labels differ")
        self._views, self._modes = {}, {}
        self._lock = threading.Lock()

    def _view(self, label):
        if label not in self._views:
            index = self.view_indices[label]
            with np.load(self.path / "arrays.npz", allow_pickle=False) as arrays:
                lo, hi = arrays["o_view_sample_offsets"][index:index + 2]
                pixels = arrays["o_sample_pixels_xy"][lo:hi]
                confidence = arrays["o_sample_confidence"][lo:hi]
                targets = arrays["o_sample_target"][self.slots, lo:hi, :]
                excluded = ~arrays["o_alpha_identifiable_mask"][self.slots, index]
                rgb = arrays[f"v{index}_rgb"]
                jacobian = arrays[f"v{index}_jacobian"]
            camera = self.cameras[label].to(self.data.device)
            if (pixels.ndim != 2 or pixels.shape[1:] != (2,) or not len(pixels)
                    or not np.issubdtype(pixels.dtype, np.integer)
                    or confidence.shape != (len(pixels),) or np.any(confidence <= 0)
                    or not np.isfinite(confidence).all()
                    or targets.shape != (len(self.slots), len(pixels), 2) or not np.isfinite(targets).all()
                    or rgb.shape != (camera.height, camera.width, 3) or rgb.dtype != np.uint8
                    or jacobian.shape != (self.data.phi.shape[1], 2, 3)
                    or np.any(pixels < 0) or np.any(pixels[:, 0] >= camera.width)
                    or np.any(pixels[:, 1] >= camera.height)):
                raise ValueError("Prepared modal samples differ from the reference projection domain")
            projector = FrozenModalProjector(self.data.scene, camera,
                torch.as_tensor(jacobian, device=self.data.device), pixels,
                torch.as_tensor(confidence, device=self.data.device))
            self._views[label] = dict(rgb=rgb, pixels=pixels, targets=targets,
                                      excluded=excluded, projector=projector)
        return self._views[label]

    @torch.inference_mode()
    def compare(self, view_label: str, mode_index: int, component_index: int):
        """Return input / initial / refined panels with one fixed display alignment."""
        if view_label not in self.view_indices:
            raise ValueError(f"Unknown modal comparison view: {view_label}")
        if type(mode_index) is not int or not 0 <= mode_index < len(self.slots):
            raise IndexError("Modal comparison mode index is outside its source domain")
        if type(component_index) is not int or component_index not in (0, 1):
            raise ValueError("Modal comparison component must be U (0) or V (1)")
        with self._lock:
            view = self._view(view_label)
            key = (view_label, mode_index)
            if key not in self._modes:
                target = np.asarray(view["targets"][mode_index], dtype=np.complex128)
                initial = view["projector"](self.data.phi0[mode_index]).cpu().numpy().astype(np.complex128)
                if initial.shape != target.shape or not np.isfinite(initial).all():
                    raise ValueError("Initial modal projection is non-finite or has different samples")
                denominator = float(np.vdot(initial, initial).real)
                alpha, refined = None, None
                if denominator > np.finfo(np.float64).tiny and math.isfinite(denominator):
                    alpha = np.vdot(initial, target) / denominator
                    refined = view["projector"](self.data.phi[mode_index]).cpu().numpy().astype(np.complex128)
                    if refined.shape != target.shape or not np.isfinite(refined).all() or not np.isfinite(alpha):
                        raise ValueError("Refined modal projection or display alignment is non-finite")
                    initial, refined = alpha * initial, alpha * refined
                else:
                    initial = None
                self._modes[key] = (target, initial, refined, alpha)
            target, initial, refined, alpha = self._modes[key]
            values = [value[:, component_index] if value is not None else None
                      for value in (target, initial, refined)]
            magnitude = float(np.percentile(np.concatenate([np.abs(v) for v in values if v is not None]),
                                              PREVIEW_PERCENTILE))
            if not math.isfinite(magnitude) or magnitude <= 0:
                magnitude = 1.0
            images = [modal_image_overlay(view["rgb"], view["pixels"], v, magnitude) if v is not None
                      else np.full_like(view["rgb"], 72) for v in values]
            alignment = ("Unavailable: initial projection has no usable image-plane energy; both "
                         "aligned reconstructions are unavailable (gray)." if alpha is None else
                         f"Display alpha = {alpha.real:.6g}{alpha.imag:+.6g}j, fitted from initial shape "
                         "using U and V together; held fixed for both reconstructions.")
            frequency = self.data.manifest["modes"][mode_index]["frequency_hz"]
            status = (f"**View:** {view_label} | **Frequency:** {frequency:.6f} Hz | "
                      f"**Component:** {'UV'[component_index]}\n\n"
                      f"**Training sampled pixels:** {len(view['pixels']):,} (sparse supervision snapshot).\n\n"
                      "**Columns:** Input | Initial | Refined\n\n"
                      f"**Shared brightness:** p99 = {magnitude:.6g}. {alignment}\n\n"
                      f"**Original modal supervision:** {'excluded' if view['excluded'][mode_index] else 'participated'}. "
                      "This status is independent of display alignment.\n\n"
                      "Canonical, frozen-geometry linear center-displacement projection; "
                      "no time coefficients, manual gain/scale or ellipsoid rotation.")
            return np.concatenate(images, axis=1), status
