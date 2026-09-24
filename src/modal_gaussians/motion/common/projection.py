"""Shared frozen-camera projection and foreground feature-sampling policy."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any
import cv2
import numpy as np
import torch
from modal_gaussians.geometry.scene import Camera, ForegroundBackgroundScene

@dataclass(frozen=True)
class RenderedDesignConfig:
    """Hold the accepted sampling and feature-render batch settings."""

    pixel_sample_stride: int = 2
    alpha_minimum: float = 0.05
    mask_erosion_iterations: int = 1
    modes_per_batch: int = 8

    def validate(self) -> None:
        """Reject settings outside the rendered-design data contract."""

        for name, value in (
            ("pixel_sample_stride", self.pixel_sample_stride),
            ("modes_per_batch", self.modes_per_batch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Rendered-design {name} must be positive")
        if (
            isinstance(self.mask_erosion_iterations, bool)
            or not isinstance(self.mask_erosion_iterations, int)
            or self.mask_erosion_iterations < 0
        ):
            raise ValueError(
                "Rendered-design mask_erosion_iterations must be non-negative"
            )
        if not math.isfinite(self.alpha_minimum) or not (
            0.0 < self.alpha_minimum <= 1.0
        ):
            raise ValueError("Rendered-design alpha_minimum must lie in (0,1]")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fixed grid, alpha, erosion, and batching policy."""

        return {
            "pixel_sample_stride": self.pixel_sample_stride,
            "grid_origin_xy": [1, 1],
            "internal_image_border_pixels": 1,
            "alpha_minimum": self.alpha_minimum,
            "mask_erosion_iterations": self.mask_erosion_iterations,
            "mask_source": "flow_artifact.mask_union",
            "modes_per_batch": self.modes_per_batch,
        }


@torch.no_grad()
def prepare_modal_projection(scene, camera, mask, config):
    """Prepare one view using the shared sampling and projection convention."""
    means = scene.foreground.active()["means"]
    dummy = means.new_zeros((len(means), 1))
    _, alpha = render_motion_features(scene, camera, dummy)
    pixels, sampled_alpha = candidate_observation_pixels(
        scene, mask, alpha.detach().cpu().float().numpy(), config)
    jacobian, visible = projection_jacobian(
        means.detach().cpu().numpy().astype(np.float32),
        camera.K.detach().cpu().numpy().astype(np.float64),
        camera.world_to_camera.detach().cpu().numpy().astype(np.float64),
        camera.radial_distortion)
    if not np.any(visible):
        raise ValueError(f"All foreground Gaussians lie behind view {camera.label!r}")
    return pixels, sampled_alpha, jacobian, int(np.count_nonzero(visible))


@torch.no_grad()
def project_modal_features(scene, camera, phi, pixels, sampled_alpha, jacobian):
    """Return [pixels, UV, real/-imag pairs] before saved view alignment."""
    device = camera.K.device
    jacobian = torch.as_tensor(jacobian, device=device, dtype=torch.float32)
    real = torch.as_tensor(np.ascontiguousarray(np.real(phi)), device=device, dtype=torch.float32)
    imag = torch.as_tensor(np.ascontiguousarray(np.imag(phi)), device=device, dtype=torch.float32)
    real = torch.einsum("gij,kgj->kgi", jacobian, real)
    imag = torch.einsum("gij,kgj->kgi", jacobian, imag)
    features = torch.stack((real[..., 0], real[..., 1], -imag[..., 0], -imag[..., 1]),
                           dim=-1).permute(1, 0, 2).reshape(phi.shape[1], -1).contiguous()
    sampled = sample_feature_render(scene, camera, features, pixels, sampled_alpha)
    return sampled.reshape(len(pixels), len(phi), 2, 2).transpose(0, 3, 1, 2).reshape(len(pixels), 2, 2 * len(phi))


def projection_jacobian(
    points: np.ndarray, K: np.ndarray, world_to_camera: np.ndarray, radial_k: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d(pixel xy)/d(world xyz), with zero rows behind the camera."""

    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [values, np.ones((len(values), 1), dtype=np.float64)], axis=1
    )
    w2c = np.asarray(world_to_camera, dtype=np.float64)
    camera_points = (homogeneous @ w2c.T)[:, :3]
    if not np.isfinite(camera_points).all():
        raise ValueError("Foreground Gaussian camera coordinates are non-finite")
    visible = camera_points[:, 2] > 1.0e-8
    if radial_k < 0:
        from modal_gaussians.common.camera_geometry import radial_domain
        visible &= radial_domain(camera_points, radial_k)
    jacobian = np.zeros((len(values), 2, 3), dtype=np.float64)
    if np.any(visible):
        x = camera_points[visible, 0]
        y = camera_points[visible, 1]
        z = camera_points[visible, 2]
        camera_jacobian = np.zeros((len(z), 2, 3), dtype=np.float64)
        camera_jacobian[:, 0, 0] = float(K[0, 0]) / z
        camera_jacobian[:, 0, 2] = -float(K[0, 0]) * x / (z * z)
        camera_jacobian[:, 1, 1] = float(K[1, 1]) / z
        camera_jacobian[:, 1, 2] = -float(K[1, 1]) * y / (z * z)
        if radial_k:
            from modal_gaussians.common.camera_geometry import camera_jacobian as radial_jacobian
            camera_jacobian = radial_jacobian(camera_points[visible], K, radial_k)
        jacobian[visible] = np.einsum(
            "nij,jk->nik", camera_jacobian, w2c[:3, :3]
        )
    return jacobian.astype(np.float32), visible


def candidate_pixels(
    mask: np.ndarray,
    alpha: np.ndarray,
    config: RenderedDesignConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the eroded semantic mask, alpha gate, and internal stride grid."""

    mask_value = np.asarray(mask, dtype=bool)
    alpha_value = np.asarray(alpha, dtype=np.float32)
    if mask_value.shape != alpha_value.shape or mask_value.ndim != 2:
        raise ValueError("Rendered-design mask and alpha shapes differ")
    if config.mask_erosion_iterations:
        mask_value = cv2.erode(
            mask_value.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=config.mask_erosion_iterations,
        ).astype(bool)
    candidate = mask_value & np.isfinite(alpha_value) & (
        alpha_value >= config.alpha_minimum
    )
    grid = np.zeros(candidate.shape, dtype=bool)
    grid[1:-1:config.pixel_sample_stride, 1:-1:config.pixel_sample_stride] = True
    y, x = np.where(candidate & grid)
    if len(x) == 0:
        raise ValueError("Rendered-design view has no valid sampled pixels")
    pixels = np.stack([x, y], axis=1).astype(np.int64)
    return pixels, alpha_value[y, x].astype(np.float32)


def uses_visible_subject(scene: ForegroundBackgroundScene) -> bool:
    """Manual 3D selections use visibility, without inherited image masks."""

    return (getattr(scene, "manifest", None) or {}).get("partition", {}).get(
        "method"
    ) == "manual_subject_selection_v1"


def render_motion_features(
    scene: ForegroundBackgroundScene, camera: Camera, features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return premultiplied subject features and their matching visible mass."""

    if not uses_visible_subject(scene):
        return scene.render_features(camera, features, composition="foreground")
    if features.ndim != 2 or features.shape[0] != scene.foreground.count or features.shape[1] < 1:
        raise ValueError("Motion features must have shape [foreground_gaussians, channels]")
    subject_features = torch.cat((features, features.new_ones((len(features), 1))), dim=-1)
    all_features = torch.cat((subject_features, features.new_zeros(
        (scene.background.count, subject_features.shape[1])
    )), dim=0)
    image, _ = scene.render_features(camera, all_features, composition="all")
    # The renderer's alpha includes background; the extra channel counts only
    # foreground contributions after all foreground/background occlusion.
    return image[..., :-1], image[..., -1]


def render_observation_geometry(
    scene: ForegroundBackgroundScene, camera: Camera,
) -> dict[str, torch.Tensor]:
    """Render visible subject mass and subject-conditional camera-space depth."""

    if not uses_visible_subject(scene):
        return scene.render(camera, composition="foreground", outputs=("alpha", "expected_depth"))
    means = scene.foreground.active()["means"]
    w2c = camera.world_to_camera.to(device=means.device, dtype=means.dtype)
    z = means @ w2c[2, :3] + w2c[2, 3]
    image, alpha = render_motion_features(scene, camera, z[:, None])
    depth = torch.where(alpha > 1e-8, image[..., 0] / alpha.clamp_min(1e-8),
                        torch.zeros_like(alpha))
    return {"alpha": alpha, "expected_depth": depth}


def candidate_observation_pixels(
    scene: ForegroundBackgroundScene, mask: np.ndarray, alpha: np.ndarray,
    config: RenderedDesignConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Manual selections retain the alpha/stride gate without mask erosion."""

    if uses_visible_subject(scene):
        return candidate_pixels(np.ones_like(alpha, dtype=bool), alpha,
                                replace(config, mask_erosion_iterations=0))
    return candidate_pixels(mask, alpha, config)


def sample_feature_render(
    scene: ForegroundBackgroundScene,
    camera: Camera,
    features: torch.Tensor,
    pixels: np.ndarray,
    expected_alpha: np.ndarray,
) -> np.ndarray:
    """Render all foreground contributors, sample pixels, and alpha-normalize."""

    image, alpha = render_motion_features(scene, camera, features)
    x = torch.as_tensor(pixels[:, 0], device=image.device, dtype=torch.long)
    y = torch.as_tensor(pixels[:, 1], device=image.device, dtype=torch.long)
    rendered_alpha = alpha[y, x].detach().cpu().float().numpy()
    if not np.allclose(rendered_alpha, expected_alpha, rtol=2.0e-5, atol=2.0e-6):
        raise RuntimeError("Feature rendering changed foreground alpha")
    values = image[y, x].detach().cpu().float().numpy()
    return values / expected_alpha[:, None]
