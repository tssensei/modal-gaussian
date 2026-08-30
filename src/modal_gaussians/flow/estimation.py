from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


FARNEBACK_PARAMETERS = {
    "pyr_scale": 0.5,
    "levels": 4,
    "winsize": 15,
    "iterations": 4,
    "poly_n": 5,
    "poly_sigma": 1.1,
    "flags": 0,
}


def compute_farneback_pair(
    reference_gray: np.ndarray, current_gray: np.ndarray
) -> np.ndarray:
    if reference_gray.ndim != 2 or current_gray.ndim != 2:
        raise ValueError("Farneback inputs must both be [H,W]")
    if reference_gray.shape != current_gray.shape:
        raise ValueError("Farneback inputs must have identical shapes")
    if not np.isfinite(reference_gray).all() or not np.isfinite(current_gray).all():
        raise ValueError("Farneback inputs contain NaN or Inf")
    reference = np.clip(reference_gray * 255.0, 0.0, 255.0).astype(np.uint8)
    current = np.clip(current_gray * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(
        reference, current, None, **FARNEBACK_PARAMETERS
    ).astype(np.float32, copy=False)


def compute_reference_to_frame_flow(
    frames_gray: np.ndarray, reference_frame_index: int
) -> np.ndarray:
    if frames_gray.ndim != 3 or frames_gray.shape[0] < 3:
        raise ValueError("frames_gray must be [T,H,W] with at least three frames")
    if not np.isfinite(frames_gray).all():
        raise ValueError("frames_gray contains NaN or Inf")
    if reference_frame_index < 0 or reference_frame_index >= frames_gray.shape[0]:
        raise ValueError("reference_frame_index is outside the frame sequence")

    frame_count, height, width = frames_gray.shape
    flow = np.empty((frame_count, height, width, 2), dtype=np.float32)
    reference = frames_gray[reference_frame_index]
    for frame_index in range(frame_count):
        if frame_index == reference_frame_index:
            flow[frame_index] = 0.0
        else:
            flow[frame_index] = compute_farneback_pair(
                reference, frames_gray[frame_index]
            )
    return flow


def _pyramid_sobel(frame: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    image = frame.astype(np.float32, copy=False)
    for _ in range(level):
        image = cv2.pyrDown(image)
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    if level:
        height, width = frame.shape
        gradient_x = cv2.resize(
            gradient_x, (width, height), interpolation=cv2.INTER_LINEAR
        ) * float(2**level)
        gradient_y = cv2.resize(
            gradient_y, (width, height), interpolation=cv2.INTER_LINEAR
        ) * float(2**level)
    return gradient_x, gradient_y


def _weighted_pyramid_sobel(
    frame: np.ndarray, pyramid_weights: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(pyramid_weights, dtype=np.float32)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("pyramid_weights must be a non-empty vector")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("pyramid_weights must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("pyramid_weights must contain a positive value")
    weights /= total
    gradient_x = np.zeros_like(frame, dtype=np.float32)
    gradient_y = np.zeros_like(frame, dtype=np.float32)
    for level, weight in enumerate(weights):
        if weight == 0.0:
            continue
        current_x, current_y = _pyramid_sobel(frame, level)
        gradient_x += float(weight) * current_x
        gradient_y += float(weight) * current_y
    return gradient_x, gradient_y


def weighted_gaussian_smooth(
    flow: np.ndarray,
    reference_gray: np.ndarray,
    mask_union: np.ndarray,
    *,
    sigma_b_px: float = 3.0,
    sigma_c_px: float = 0.0,
    gradient_pyramid_weights: Sequence[float] = (0.5, 0.3, 0.2),
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Davis-inspired contrast-weighted spatial flow smoothing.

    This preserves the accepted legacy approximation: horizontal and vertical
    displacement are weighted by the corresponding reference-image Sobel
    gradient, spatially blurred, and normalized by the blurred weights.
    """
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError("flow must be [T,H,W,2]")
    if reference_gray.shape != flow.shape[1:3]:
        raise ValueError("reference_gray does not spatially match flow")
    if mask_union.shape != reference_gray.shape or mask_union.dtype != np.bool_:
        raise ValueError("mask_union must be bool [H,W]")
    if not np.isfinite(flow).all() or not np.isfinite(reference_gray).all():
        raise ValueError("Smoothing inputs contain NaN or Inf")
    if not np.isfinite(sigma_b_px) or sigma_b_px <= 0.0:
        raise ValueError("sigma_b_px must be finite and positive")
    if not np.isfinite(sigma_c_px) or sigma_c_px < 0.0:
        raise ValueError("sigma_c_px must be finite and non-negative")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    contrast_image = reference_gray.astype(np.float32, copy=False)
    if sigma_c_px > 0.0:
        contrast_kernel = int(round(6.0 * float(sigma_c_px) + 1.0))
        if contrast_kernel % 2 == 0:
            contrast_kernel += 1
        contrast_image = cv2.GaussianBlur(
            contrast_image,
            (contrast_kernel, contrast_kernel),
            sigmaX=float(sigma_c_px),
            sigmaY=float(sigma_c_px),
            borderType=cv2.BORDER_REFLECT,
        )
    gradient_x, gradient_y = _weighted_pyramid_sobel(
        contrast_image, gradient_pyramid_weights
    )
    mask_float = mask_union.astype(np.float32)
    weight_x = np.abs(gradient_x).astype(np.float32) * mask_float
    weight_y = np.abs(gradient_y).astype(np.float32) * mask_float
    displacement_kernel = int(round(6.0 * float(sigma_b_px) + 1.0))
    if displacement_kernel % 2 == 0:
        displacement_kernel += 1
    denominator_x = cv2.GaussianBlur(
        weight_x,
        (displacement_kernel, displacement_kernel),
        sigmaX=float(sigma_b_px),
        sigmaY=float(sigma_b_px),
        borderType=cv2.BORDER_REFLECT,
    ) + float(epsilon)
    denominator_y = cv2.GaussianBlur(
        weight_y,
        (displacement_kernel, displacement_kernel),
        sigmaX=float(sigma_b_px),
        sigmaY=float(sigma_b_px),
        borderType=cv2.BORDER_REFLECT,
    ) + float(epsilon)

    output = np.empty_like(flow, dtype=np.float32)
    for frame_index in range(flow.shape[0]):
        numerator_x = cv2.GaussianBlur(
            flow[frame_index, :, :, 0] * weight_x,
            (displacement_kernel, displacement_kernel),
            sigmaX=float(sigma_b_px),
            sigmaY=float(sigma_b_px),
            borderType=cv2.BORDER_REFLECT,
        )
        numerator_y = cv2.GaussianBlur(
            flow[frame_index, :, :, 1] * weight_y,
            (displacement_kernel, displacement_kernel),
            sigmaX=float(sigma_b_px),
            sigmaY=float(sigma_b_px),
            borderType=cv2.BORDER_REFLECT,
        )
        output[frame_index, :, :, 0] = numerator_x / denominator_x
        output[frame_index, :, :, 1] = numerator_y / denominator_y
    output *= mask_float[None, :, :, None]
    if not np.isfinite(output).all():
        raise ValueError("Davis-style smoothing produced NaN or Inf")
    return output
