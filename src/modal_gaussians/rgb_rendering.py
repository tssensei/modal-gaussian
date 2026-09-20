"""Differentiable RGB rendering with immutable spatial modes and scene parameters."""
from dataclasses import replace
import math

import numpy as np
import torch
from torch.nn import functional as F


def _pyramid_shape(height, width, scale):
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("RGB pyramid scale must be finite and in (0, 1]")
    shape = math.floor(height * scale), math.floor(width * scale)
    if min(shape) < 2:
        raise ValueError("RGB pyramid dimensions must remain at least two pixels")
    return shape


def resize_rgb(image, scale):
    """Resize HWC float RGB with one isotropic scale, including odd image sizes."""
    if image.ndim != 3 or image.shape[-1] != 3 or not image.is_floating_point():
        raise ValueError("RGB target must be a floating-point [H,W,3] tensor")
    _pyramid_shape(*image.shape[:2], scale)
    if scale == 1:
        return image
    return F.interpolate(image.permute(2, 0, 1)[None], scale_factor=scale,
                         mode="bilinear", align_corners=False,
                         recompute_scale_factor=False)[0].permute(1, 2, 0)


def make_rgb_renderer(scene, camera, phi, rotation=None, device="cuda"):
    """Return render(q[K] complex, scale) -> RGB[H,W,3], differentiable only in q."""
    scene = scene.to(device).eval()
    scene.requires_grad_(False)
    camera = camera.to(device)
    active = scene.foreground.active()
    means, base = active["means"].detach(), active["quaternions"].detach()

    def frozen_basis(value):
        value = np.asarray(value)
        if (value.ndim != 3 or value.shape[1:] != tuple(means.shape)
                or value.shape[0] < 1 or not np.iscomplexobj(value)
                or not np.isfinite(value).all()):
            raise ValueError("RGB spatial modes must be finite complex [K,G,3]")
        return torch.tensor(value, dtype=torch.complex64, device=device)

    modes = frozen_basis(phi)
    rotations = None if rotation is None else frozen_basis(rotation)
    if rotations is not None and rotations.shape != modes.shape:
        raise ValueError("RGB displacement and rotation mode shapes differ")
    cameras = {}

    def render(q, scale):
        if q.shape != (len(modes),) or not q.is_complex():
            raise ValueError("RGB coefficients must be complex [K]")
        height, width = _pyramid_shape(camera.height, camera.width, scale)
        if scale not in cameras:
            K = camera.K.clone()
            K[:2] *= scale
            parameters = camera.camera_parameters
            if camera.camera_model == "SIMPLE_RADIAL":
                parameters = tuple(value * scale for value in parameters[:3]) + parameters[3:]
            cameras[scale] = replace(camera, K=K, height=height, width=width,
                                     camera_parameters=parameters)
        coefficients = q.to(device=device, dtype=modes.dtype)
        deformed = means + torch.einsum("k,kgc->gc", coefficients, modes).real
        orientations = None
        if rotations is not None:
            vectors = torch.einsum("k,kgc->gc", coefficients, rotations).real
            angles = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
            scalar = torch.cos(angles / 2)
            vector = 0.5 * torch.sinc(angles / (2 * math.pi)) * vectors
            real, imaginary = base[:, :1], base[:, 1:]
            orientations = F.normalize(torch.cat((
                scalar * real - (vector * imaginary).sum(dim=-1, keepdim=True),
                scalar * imaginary + real * vector
                + torch.linalg.cross(vector, imaginary, dim=-1)), dim=-1), dim=-1)
        return scene.render_deformed(cameras[scale], deformed,
                                     foreground_quaternions=orientations)["rgb"]

    return render
