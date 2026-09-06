"""Low-level geometry sampling, retaining each pipeline's original precision.

The original rigid sampler rounds weights to float32; the neural depth sampler
preserves float64. They remain explicit separate functions for compatibility.
"""
from __future__ import annotations

import numpy as np

def project_points(
    points: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project normalized-world points and return image xy plus camera z."""

    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [values, np.ones((len(values), 1), dtype=np.float64)], axis=1
    )
    camera = homogeneous @ np.asarray(world_to_camera, dtype=np.float64).T
    z = camera[:, 2]
    x = float(K[0, 0]) * camera[:, 0] / z + float(K[0, 2])
    y = float(K[1, 1]) * camera[:, 1] / z + float(K[1, 2])
    return np.stack([x, y], axis=1).astype(np.float32), z.astype(np.float32)


def bilinear_valid(pixels: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return whether floating xy coordinates have a full bilinear footprint."""

    values = np.asarray(pixels, dtype=np.float64)
    return (
        np.isfinite(values).all(axis=-1)
        & (values[..., 0] >= 0.0)
        & (values[..., 1] >= 0.0)
        & (values[..., 0] < width - 1)
        & (values[..., 1] < height - 1)
    )


def bilinear_sample(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Sample one 2D image at already validated floating xy coordinates."""

    values = np.asarray(pixels, dtype=np.float64)
    x0 = np.floor(values[:, 0]).astype(np.int64)
    y0 = np.floor(values[:, 1]).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx = (values[:, 0] - x0).astype(np.float32)
    wy = (values[:, 1] - y0).astype(np.float32)
    source = np.asarray(image)
    return (
        (1.0 - wx) * (1.0 - wy) * source[y0, x0]
        + wx * (1.0 - wy) * source[y0, x1]
        + (1.0 - wx) * wy * source[y1, x0]
        + wx * wy * source[y1, x1]
    )


def sample_valid(image: np.ndarray, pixels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Sample valid pixels and fill invalid rows with NaN."""

    result = np.full(valid.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        result[valid] = bilinear_sample(image, pixels[valid]).astype(np.float32)
    return result


def pixel_valid(pixels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return (np.isfinite(pixels).all(axis=-1) & (pixels[..., 0] >= 0)
            & (pixels[..., 1] >= 0) & (pixels[..., 0] < width - 1)
            & (pixels[..., 1] < height - 1))


def bilinear_sample_float64(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    x0 = np.floor(pixels[:, 0]).astype(np.int64)
    y0 = np.floor(pixels[:, 1]).astype(np.int64)
    x, y = pixels[:, 0] - x0, pixels[:, 1] - y0
    return ((1 - x) * (1 - y) * image[y0, x0]
            + x * (1 - y) * image[y0, x0 + 1]
            + (1 - x) * y * image[y0 + 1, x0]
            + x * y * image[y0 + 1, x0 + 1])
