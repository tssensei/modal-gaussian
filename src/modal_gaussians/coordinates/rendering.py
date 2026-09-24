"""Differentiable RGB rendering with immutable spatial modes and scene parameters."""
from dataclasses import replace
import math

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


def scaled_camera(camera, scale):
    height, width = _pyramid_shape(camera.height, camera.width, scale)
    K = camera.K.clone()
    K[:2] *= scale
    parameters = camera.camera_parameters
    if camera.camera_model == "SIMPLE_RADIAL":
        parameters = tuple(value * scale for value in parameters[:3]) + parameters[3:]
    return replace(camera, K=K, height=height, width=width, camera_parameters=parameters)


def apply_angular_rotation(base, vectors):
    """Left-compose world-space exponential-map rotation with canonical wxyz."""
    angles = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    scalar = torch.cos(angles / 2)
    vector = 0.5 * torch.sinc(angles / (2 * math.pi)) * vectors
    real, imaginary = base[:, :1], base[:, 1:]
    return F.normalize(torch.cat((
        scalar * real - (vector * imaginary).sum(dim=-1, keepdim=True),
        scalar * imaginary + real * vector
        + torch.linalg.cross(vector, imaginary, dim=-1)), dim=-1), dim=-1)


def make_rgb_renderer(scene, camera, phi, rotation=None, device="cuda"):
    """Return render(q[K] complex, scale) -> RGB[H,W,3], differentiable only in q."""
    render = make_sequence_renderer(scene, [camera], phi, rotation, device)
    return lambda q, scale: render(0, q, scale)


def deform_baked(means, base, q, modes, rotations=None):
    """Apply a resident frozen basis, differentiating only the supplied tensors."""
    deformed = means + torch.einsum("k,kgc->gc", q, modes).real
    orientations = None if rotations is None else apply_angular_rotation(
        base, torch.einsum("k,kgc->gc", q, rotations).real)
    return deformed, orientations


def make_sequence_renderer(scene, frame_cameras, phi, rotation=None, device="cuda"):
    """One resident basis and scene, with a camera selected for each input frame."""
    scene = scene.to(device).eval()
    scene.requires_grad_(False)
    active = scene.foreground.active()
    means, base = active["means"].detach(), active["quaternions"].detach()

    def frozen_basis(value):
        value = torch.as_tensor(value, device=device).detach()
        if (value.ndim != 3 or value.shape[1:] != tuple(means.shape)
                or value.shape[0] < 1 or not value.is_complex()
                or not torch.isfinite(value).all()):
            raise ValueError("RGB spatial modes must be finite complex [K,G,3]")
        return value.to(dtype=torch.complex64)

    modes = frozen_basis(phi)
    rotations = None if rotation is None else frozen_basis(rotation)
    if rotations is not None and rotations.shape != modes.shape:
        raise ValueError("RGB displacement and rotation mode shapes differ")
    cameras = {}

    def render(frame, q, scale):
        if q.shape != (len(modes),) or not q.is_complex():
            raise ValueError("RGB coefficients must be complex [K]")
        camera = frame_cameras[frame]
        key = camera.name, scale
        if key not in cameras:
            cameras[key] = scaled_camera(camera.to(device), scale)
        coefficients = q.to(device=device, dtype=modes.dtype)
        deformed, orientations = deform_baked(means, base, coefficients, modes, rotations)
        return scene.render_deformed(cameras[key], deformed,
                                     foreground_quaternions=orientations)["rgb"]

    return render
