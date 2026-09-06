"""Differentiable radial image warp around the classic gsplat renderer.

The installed gsplat UT projection has no geometry backward. Rendering an
overscanned pinhole image and resampling rays preserves geometry/feature gradients
and the original image, mask and optical-flow coordinate system. The only added
approximation is bilinear image resampling, not ignoring radial distortion.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from modal_gaussians.camera_geometry import undistort_normalized


@lru_cache(maxsize=4)
def _warp_grid(width: int, height: int, intrinsics: tuple[float, ...], k: float):
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    yy, xx = np.mgrid[:height, :width]
    # gsplat samples at (column + 0.5, row + 0.5), not integer pixel corners.
    normalized = np.stack(((xx + 0.5 - K[0, 2]) / K[0, 0],
                           (yy + 0.5 - K[1, 2]) / K[1, 1]), axis=-1)
    xy = undistort_normalized(normalized, k)
    u = xy[..., 0] * K[0, 0] + K[0, 2] - 0.5
    v = xy[..., 1] * K[1, 1] + K[1, 2] - 0.5
    left, top = max(0, int(np.ceil(-u.min())) + 2), max(0, int(np.ceil(-v.min())) + 2)
    right = max(0, int(np.ceil(u.max() - (width - 1))) + 2)
    bottom = max(0, int(np.ceil(v.max() - (height - 1))) + 2)
    return u.astype(np.float32), v.astype(np.float32), (left, top, right, bottom)


def rasterize_cameras(rasterization, cameras, **kwargs: Any):
    """Keep one batched gsplat call and means2d gradients for density control."""
    ks = [camera.radial_distortion for camera in cameras]
    if not any(ks):
        return rasterization(**kwargs)
    width, height = kwargs["width"], kwargs["height"]
    maps = [_warp_grid(width, height, tuple(c.K.detach().cpu().double().numpy().ravel()), k)
            for c, k in zip(cameras, ks)]
    padding = np.max([m[2] for m in maps], axis=0)
    left, top, right, bottom = map(int, padding)
    padded_w, padded_h = width + left + right, height + top + bottom
    if padded_w * padded_h > 4 * width * height or max(padded_w, padded_h) > 32768:
        raise ValueError("Radial render overscan exceeds 4x image area; check COLMAP calibration")
    device, dtype = kwargs["means"].device, kwargs["means"].dtype
    grid = np.stack([np.stack((2 * (u + left + 0.5) / padded_w - 1,
                              2 * (v + top + 0.5) / padded_h - 1), axis=-1)
                     for u, v, _ in maps])
    grid = torch.as_tensor(grid, device=device, dtype=dtype)
    K = kwargs["Ks"].clone()
    K[:, 0, 2] += left
    K[:, 1, 2] += top
    image, alpha, info = rasterization(**{**kwargs, "Ks": K, "width": padded_w, "height": padded_h})

    def sample(value):
        return F.grid_sample(value.permute(0, 3, 1, 2), grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False).permute(0, 2, 3, 1)

    mapped_alpha = sample(alpha)
    if kwargs.get("render_mode", "RGB") == "RGB+ED":
        # Expected depth must be resampled as an alpha-weighted moment.
        rgb = sample(image[..., :-1])
        numerator = sample(image[..., -1:] * alpha)
        depth = torch.where(mapped_alpha > 1e-8, numerator / mapped_alpha.clamp_min(1e-8), 0)
        mapped_image = torch.cat((rgb, depth), dim=-1)
    else:
        mapped_image = sample(image)
    info["radial_render_padding"] = (left, top, right, bottom)
    return mapped_image, mapped_alpha, info
