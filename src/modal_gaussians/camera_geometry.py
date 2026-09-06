"""COLMAP SIMPLE_RADIAL geometry, in the original (distorted) image domain.

The optional scalar k defaults to zero only for explicit legacy pinhole cameras.
No image, mask or flow vector is undistorted by these functions.
"""
from __future__ import annotations

import numpy as np

PROJECTION_CONVENTION = "simple_radial_image_warp_v1"


def distort_normalized(xy: np.ndarray, k: float = 0.0) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    return xy * (1.0 + k * np.sum(xy * xy, axis=-1, keepdims=True))


def undistort_normalized(xy: np.ndarray, k: float = 0.0) -> np.ndarray:
    """Invert the central, monotonic radial branch; reject noninvertible FoV."""
    xy = np.asarray(xy, dtype=np.float64)
    if not np.isfinite(xy).all() or not np.isfinite(k):
        raise ValueError("Camera inverse projection requires finite coordinates and k")
    if k == 0:
        return xy.copy()
    rd = np.linalg.norm(xy, axis=-1)
    lo = np.zeros_like(rd)
    if k < 0:
        turning = np.sqrt(-1.0 / (3.0 * k))
        if np.any(rd >= turning * (2.0 / 3.0)):
            raise ValueError("SIMPLE_RADIAL is not invertible over the requested image FoV")
        hi = np.full_like(rd, turning)
    else:
        hi = rd.copy()
    # Bracketed Newton retains the central branch even for negative k.
    r = np.minimum(rd, hi)
    for _ in range(55):
        residual = r * (1 + k * r * r) - rd
        converged = np.abs(residual) <= 1e-12
        if np.all(converged):
            break
        lo = np.where(residual < 0, r, lo)
        hi = np.where(residual >= 0, r, hi)
        proposal = r - residual / np.maximum(1 + 3 * k * r * r, 1e-15)
        updated = np.where((proposal > lo) & (proposal < hi), proposal, (lo + hi) * 0.5)
        r = np.where(converged, r, updated)
    if not np.allclose(r * (1 + k * r * r), rd, atol=1e-10, rtol=1e-10):
        raise ValueError("SIMPLE_RADIAL inverse projection did not converge")
    return xy * np.divide(r, rd, out=np.ones_like(r), where=rd > 0)[..., None]


def project_camera(camera_points: np.ndarray, K: np.ndarray, k: float = 0.0) -> np.ndarray:
    xyz = np.asarray(camera_points, dtype=np.float64)
    xy = distort_normalized(xyz[..., :2] / xyz[..., 2:3], k)
    pixels = xy * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    if k < 0:
        pixels = np.where(radial_domain(xyz, k)[..., None], pixels, np.nan)
    return pixels


def radial_domain(camera_points: np.ndarray, k: float) -> np.ndarray:
    """Exclude folded rays that the inverse image warp cannot represent."""
    xyz = np.asarray(camera_points, dtype=np.float64)
    if k >= 0:
        return np.isfinite(xyz).all(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.sum(xyz[..., :2] ** 2, axis=-1) / xyz[..., 2] ** 2
    return np.isfinite(r2) & (1 + 3 * k * r2 > 0)


def camera_jacobian(camera_points: np.ndarray, K: np.ndarray, k: float = 0.0) -> np.ndarray:
    """Analytic d(distorted pixel)/d(camera XYZ), without a small-k approximation."""
    xyz = np.asarray(camera_points, dtype=np.float64)
    x, y, z = xyz.T
    qx, qy = x / z, y / z
    pinhole = np.zeros((len(xyz), 2, 3), dtype=np.float64)
    pinhole[:, 0, 0], pinhole[:, 0, 2] = 1 / z, -qx / z
    pinhole[:, 1, 1], pinhole[:, 1, 2] = 1 / z, -qy / z
    s = 1 + k * (qx * qx + qy * qy)
    radial = np.empty((len(xyz), 2, 2), dtype=np.float64)
    radial[:, 0, 0] = K[0, 0] * (s + 2 * k * qx * qx)
    radial[:, 0, 1] = K[0, 0] * (2 * k * qx * qy)
    radial[:, 1, 0] = K[1, 1] * (2 * k * qx * qy)
    radial[:, 1, 1] = K[1, 1] * (s + 2 * k * qy * qy)
    return radial @ pinhole


def validate_radial_views(values: np.ndarray | None, count: int) -> np.ndarray:
    result = np.zeros(count) if values is None else np.asarray(values, dtype=np.float64)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError("Camera radial coefficients must be finite [V]")
    return result


def radial_pixel_path(start: np.ndarray, end: np.ndarray, fractions: np.ndarray,
                      K: np.ndarray, k: float) -> np.ndarray:
    """A straight projected 3D segment curves under radial distortion."""
    focal = np.array([K[0, 0], K[1, 1]])
    center = np.array([K[0, 2], K[1, 2]])
    a = undistort_normalized((start - center) / focal, k)
    b = undistort_normalized((end - center) / focal, k)
    xy = a[:, None] * (1 - fractions[None, :, None]) + b[:, None] * fractions[None, :, None]
    return distort_normalized(xy, k) * focal + center


def radial_path_length_bound(start: np.ndarray, end: np.ndarray, K: np.ndarray, k: float) -> np.ndarray:
    """Bound maximum pixel speed over a unit segment to choose safe sample spacing."""
    focal = np.array([K[0, 0], K[1, 1]])
    center = np.array([K[0, 2], K[1, 2]])
    a = undistort_normalized((start - center) / focal, k)
    b = undistort_normalized((end - center) / focal, k)
    r2 = np.maximum(np.sum(a * a, axis=-1), np.sum(b * b, axis=-1))
    # On the invertible negative-k branch both eigenvalues lie in [0,1].
    norm_bound = np.maximum(1.0, 1.0 + 3 * k * r2)
    return np.linalg.norm(b - a, axis=-1) * focal.max() * norm_bound
