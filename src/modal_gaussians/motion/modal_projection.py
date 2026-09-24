"""Optional in-memory gsplat preparation cache for fixed modal supervision.

Pixel compositing and its feature backward still use the installed renderer.
Geometry training must use the dynamic path; this cache never owns motion IDs.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from modal_gaussians.common.camera_rendering import _warp_grid, rasterize_cameras
from modal_gaussians.geometry.scene import _load_gsplat_rasterization
from modal_gaussians.motion.common.projection import uses_visible_subject


class CachedModalRasterizer:
    """Cache one frozen scene/camera's projection and sorted tile intersections."""

    def __init__(self, scene, camera, pixels):
        self.scene, self.camera = scene, camera
        self.subject = uses_visible_subject(scene)
        self.stamp = self._stamp()
        with torch.no_grad():
            active = scene._active_for("all" if self.subject else "foreground")
            means = active["means"]
            device, dtype = means.device, means.dtype
            if device.type != "cuda":
                raise ValueError("Cached modal projection requires CUDA")
            # Obtain metadata from the very same high-level path/defaults as
            # render_features, rather than duplicating gsplat projection rules.
            _, _, meta = rasterize_cameras(_load_gsplat_rasterization(), [camera],
                means=means, quats=active["quaternions"], scales=active["scales"],
                opacities=active["opacities"], colors=means.new_zeros((len(means), 1)),
                viewmats=camera.world_to_camera.to(device)[None], Ks=camera.K.to(device)[None],
                width=camera.width, height=camera.height, packed=False,
                backgrounds=means.new_zeros((1, 1)), render_mode="RGB",
                rasterize_mode="classic", camera_model="pinhole")
            self.geometry = {key: meta[key] for key in (
                "means2d", "conics", "opacities", "isect_offsets", "flatten_ids")}
            self.width, self.height, self.tile_size = meta["width"], meta["height"], meta["tile_size"]
            self.count = len(means)
            self.foreground_count = scene.foreground.count
            self.x = torch.as_tensor(pixels[:, 0], device=device, dtype=torch.long)
            self.y = torch.as_tensor(pixels[:, 1], device=device, dtype=torch.long)
            self.grid = None
            if camera.radial_distortion:
                u, v, _ = _warp_grid(camera.width, camera.height,
                    tuple(camera.K.detach().cpu().double().numpy().ravel()), camera.radial_distortion)
                left, top, _, _ = meta["radial_render_padding"]
                # Same arithmetic/float32 rounding as camera_rendering; gather
                # only supervised pixels after building the fixed warp grid.
                grid = np.stack((2 * (u + left + .5) / self.width - 1,
                                 2 * (v + top + .5) / self.height - 1), axis=-1)
                grid = grid[pixels[:, 1], pixels[:, 0]]
                self.grid = torch.as_tensor(grid, device=device, dtype=dtype)[None, None]

    def _stamp(self):
        parts = (self.scene.foreground, self.scene.background) if self.subject else (self.scene.foreground,)
        tensors = [part.params[name] for part in parts
                   for name in ("means", "quaternions", "log_scales", "opacity_logits")]
        tensors += [self.camera.K, self.camera.world_to_camera]
        if any(t.requires_grad for t in tensors):
            raise RuntimeError("Cached modal projection requires frozen geometry/cameras; use dynamic rendering for geometry gradients")
        # Tensor version/storage guards catch optimizer/in-place updates and
        # parameter replacement without scanning arrays or synchronizing CUDA.
        return (uses_visible_subject(self.scene), self.camera.width, self.camera.height,
                self.camera.camera_model, self.camera.camera_parameters, self.camera.distortion_applied,
                tuple((id(t), t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device) for t in tensors))

    def __call__(self, features):
        if self._stamp() != self.stamp:
            raise RuntimeError("Cached modal projection geometry/camera changed; rebuild prepared observations and projector")
        if features.ndim != 2 or features.shape[0] != self.foreground_count or features.shape[1] < 1:
            raise ValueError("Motion features must have shape [foreground_gaussians, channels]")
        reference = self.geometry["means2d"]
        values = features.to(device=reference.device, dtype=reference.dtype).contiguous()
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("features contain non-finite values")
        if self.subject:
            # Match the visible-subject path's channel count/padding as well as
            # zero-valued background features. Background still occludes.
            values = torch.cat((values, values.new_ones((len(values), 1))), dim=-1)
            values = torch.cat((values, values.new_zeros((self.count-len(values), values.shape[1]))))
        from gsplat.cuda._wrapper import rasterize_to_pixels
        image, _ = rasterize_to_pixels(**self.geometry, colors=values[None],
            image_width=self.width, image_height=self.height, tile_size=self.tile_size,
            backgrounds=values.new_zeros((1, values.shape[1])), packed=False)
        if self.grid is None:
            sampled = image[0, self.y, self.x]
        else:
            sampled = F.grid_sample(image.permute(0, 3, 1, 2), self.grid,
                mode="bilinear", padding_mode="zeros", align_corners=False)[0, :, 0].T
        return sampled[:, :-1] if self.subject else sampled
