from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from modal_gaussians.data.sequence import (
    ImageMaskSequence,
    read_binary_mask,
    read_color_image,
)


@dataclass(frozen=True)
class StabilizationSettings:
    """Store every parameter used by reference-anchored stabilization.

    Construct this object from values such as feature counts, LK thresholds,
    and the RANSAC threshold, then pass it to
    ``stabilize_to_reference(..., settings=...)``.
    """

    mask_dilate_iterations: int = 7
    border_margin_px: int = 12
    max_corners: int = 1000
    feature_quality: float = 0.01
    feature_min_distance_px: int = 10
    feature_block_size: int = 7
    lk_window_px: int = 21
    lk_max_level: int = 3
    lk_max_iterations: int = 60
    lk_epsilon: float = 0.005
    pass1_forward_backward_threshold_px: float = 1.0
    pass2_forward_backward_threshold_px: float = 2.0
    minimum_track_success_fraction: float = 0.80
    maximum_average_forward_backward_error_px: float = 1.5
    minimum_stable_points: int = 25
    minimum_matches: int = 12
    minimum_inliers: int = 10
    ransac_threshold_px: float = 3.0
    temporal_smoothing_radius: int = 10
    maximum_fallback_fraction: float = 0.20


@dataclass(frozen=True)
class StabilizationResult:
    """Store every output from one stabilization run.

    The result contains stabilized grayscale frames, masks, valid pixels,
    frame-to-reference homographies, diagnostics, and the applied settings.
    """

    frames_gray: np.ndarray
    masks: np.ndarray
    valid_mask: np.ndarray
    homographies_frame_to_reference: np.ndarray
    statistics: dict[str, Any]
    settings: StabilizationSettings


def _validate_settings(settings: StabilizationSettings) -> None:
    """Validate the stabilization settings.

    Args:
        settings: A ``StabilizationSettings`` object.
    Returns:
        Nothing when valid. Negative, non-finite, or inconsistent values raise
        ``ValueError``.
    """
    if settings.mask_dilate_iterations < 0:
        raise ValueError("mask_dilate_iterations must be non-negative")
    if settings.border_margin_px < 0:
        raise ValueError("border_margin_px must be non-negative")
    if settings.max_corners < 4:
        raise ValueError("max_corners must be at least four")
    if not np.isfinite(settings.feature_quality) or not 0.0 < settings.feature_quality < 1.0:
        raise ValueError("feature_quality must be in (0,1)")
    if settings.feature_min_distance_px <= 0:
        raise ValueError("feature_min_distance_px must be positive")
    if settings.feature_block_size < 3 or settings.feature_block_size % 2 == 0:
        raise ValueError("feature_block_size must be an odd integer at least three")
    if settings.lk_window_px < 3 or settings.lk_window_px % 2 == 0:
        raise ValueError("lk_window_px must be an odd integer at least three")
    if settings.lk_max_level < 0 or settings.lk_max_iterations <= 0:
        raise ValueError("LK levels must be non-negative and iterations positive")
    if not np.isfinite(settings.lk_epsilon) or settings.lk_epsilon <= 0.0:
        raise ValueError("lk_epsilon must be positive")
    if (
        not np.isfinite(settings.pass1_forward_backward_threshold_px)
        or settings.pass1_forward_backward_threshold_px < 0.0
        or not np.isfinite(settings.pass2_forward_backward_threshold_px)
        or settings.pass2_forward_backward_threshold_px < 0.0
    ):
        raise ValueError("forward-backward thresholds must be finite and non-negative")
    if (
        not np.isfinite(settings.minimum_track_success_fraction)
        or not 0.0 < settings.minimum_track_success_fraction <= 1.0
    ):
        raise ValueError("minimum_track_success_fraction must be in (0,1]")
    if (
        not np.isfinite(settings.maximum_average_forward_backward_error_px)
        or settings.maximum_average_forward_backward_error_px < 0.0
    ):
        raise ValueError(
            "maximum_average_forward_backward_error_px must be finite and non-negative"
        )
    if (
        not np.isfinite(settings.maximum_fallback_fraction)
        or not 0.0 <= settings.maximum_fallback_fraction <= 1.0
    ):
        raise ValueError("maximum_fallback_fraction must be in [0,1]")
    if settings.minimum_stable_points < 4:
        raise ValueError("minimum_stable_points must be at least four")
    if settings.minimum_matches < 4 or settings.minimum_inliers < 4:
        raise ValueError("minimum_matches and minimum_inliers must be at least four")
    if (
        settings.minimum_stable_points > settings.max_corners
        or settings.minimum_matches > settings.max_corners
        or settings.minimum_inliers > settings.max_corners
    ):
        raise ValueError("point-count thresholds cannot exceed max_corners")
    if not np.isfinite(settings.ransac_threshold_px) or settings.ransac_threshold_px <= 0.0:
        raise ValueError("ransac_threshold_px must be finite and positive")
    if settings.temporal_smoothing_radius < 0:
        raise ValueError("temporal_smoothing_radius must be non-negative")


def _dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Expand foreground to avoid selecting features near moving-object edges.

    Args:
        mask: A ``bool [H,W]`` foreground mask.
        iterations: Number of 3x3 dilation passes; zero disables dilation.
    Returns:
        A dilated ``bool [H,W]`` mask with the same shape.
    """
    binary = np.asarray(mask, dtype=bool)
    if iterations == 0:
        return binary
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(
        binary.astype(np.uint8), kernel, iterations=int(iterations)
    ) > 0


def _detect_reference_features(
    reference_u8: np.ndarray,
    exclude_mask: np.ndarray,
    settings: StabilizationSettings,
) -> np.ndarray:
    """Detect Shi-Tomasi feature points on the reference background.

    Args:
        reference_u8: Reference grayscale image as ``uint8 [H,W]``.
        exclude_mask: ``bool [H,W]`` mask where ``True`` forbids detection.
        settings: Feature-detection parameters.
    Returns:
        ``float32 [N,1,2]`` feature points with ``(x, y)`` coordinates.
    """
    detection_mask = (~exclude_mask).astype(np.uint8) * 255
    margin = int(settings.border_margin_px)
    if margin:
        if 2 * margin >= min(reference_u8.shape):
            raise ValueError("border_margin_px leaves no valid feature region")
        detection_mask[:margin] = 0
        detection_mask[-margin:] = 0
        detection_mask[:, :margin] = 0
        detection_mask[:, -margin:] = 0
    points = cv2.goodFeaturesToTrack(
        reference_u8,
        maxCorners=int(settings.max_corners),
        qualityLevel=float(settings.feature_quality),
        minDistance=float(settings.feature_min_distance_px),
        blockSize=int(settings.feature_block_size),
        useHarrisDetector=False,
        mask=detection_mask,
    )
    if points is None or len(points) < settings.minimum_stable_points:
        count = 0 if points is None else len(points)
        raise ValueError(
            f"Only {count} reference background features were detected; "
            f"need at least {settings.minimum_stable_points}"
        )
    return points.astype(np.float32)


def _points_inside(points_xy: np.ndarray, height: int, width: int) -> np.ndarray:
    """Test whether 2D points are finite and inside an image.

    Args:
        points_xy: Point coordinates as ``[N,2]``.
        height: Image height.
        width: Image width.
    Returns:
        ``bool [N]`` with ``True`` for each finite, in-bounds point.
    """
    return (
        np.isfinite(points_xy).all(axis=1)
        & (points_xy[:, 0] >= 0.0)
        & (points_xy[:, 0] <= width - 1.0)
        & (points_xy[:, 1] >= 0.0)
        & (points_xy[:, 1] <= height - 1.0)
    )


def _points_on_background(points_xy: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Test whether tracked points remain on the current-frame background.

    Args:
        points_xy: Current-frame coordinates as ``float [N,2]``.
        foreground: Current foreground mask as ``bool [H,W]``.
    Returns:
        ``bool [N]`` with ``True`` for in-bounds points outside foreground.
    """
    height, width = foreground.shape
    inside = _points_inside(points_xy, height, width)
    keep = np.zeros(points_xy.shape[0], dtype=bool)
    if not np.any(inside):
        return keep
    indices = np.flatnonzero(inside)
    x = np.rint(points_xy[indices, 0]).astype(np.int64)
    y = np.rint(points_xy[indices, 1]).astype(np.int64)
    keep[indices] = ~foreground[y, x]
    return keep


def _track_reference_points(
    reference_u8: np.ndarray,
    current_u8: np.ndarray,
    reference_points: np.ndarray,
    current_exclude_mask: np.ndarray,
    settings: StabilizationSettings,
    forward_backward_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track reference features with forward-backward Lucas-Kanade checks.

    Args:
        reference_u8: Reference image as ``uint8 [H,W]``.
        current_u8: Current image as ``uint8 [H,W]``.
        reference_points: Reference features as ``float32 [N,1,2]``.
        current_exclude_mask: Current foreground area that rejects tracks.
        settings: Lucas-Kanade parameters.
        forward_backward_threshold_px: Maximum round-trip error in pixels.
    Returns:
        ``current_xy``: Current coordinates as ``float32 [N,2]``.
        ``good``: ``bool [N]`` reliable background-track mask.
        ``error``: Forward-backward error as ``float32 [N]``.
    """
    lk_parameters = {
        "winSize": (int(settings.lk_window_px), int(settings.lk_window_px)),
        "maxLevel": int(settings.lk_max_level),
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(settings.lk_max_iterations),
            float(settings.lk_epsilon),
        ),
    }
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        reference_u8, current_u8, reference_points, None, **lk_parameters
    )
    count = len(reference_points)
    if forward is None or forward_status is None:
        return (
            np.zeros((count, 2), dtype=np.float32),
            np.zeros(count, dtype=bool),
            np.full(count, np.inf, dtype=np.float32),
        )
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_u8, reference_u8, forward, None, **lk_parameters
    )
    if backward is None or backward_status is None:
        return (
            forward.reshape(-1, 2).astype(np.float32),
            np.zeros(count, dtype=bool),
            np.full(count, np.inf, dtype=np.float32),
        )

    reference_xy = reference_points.reshape(-1, 2)
    current_xy = forward.reshape(-1, 2)
    backward_xy = backward.reshape(-1, 2)
    error = np.linalg.norm(reference_xy - backward_xy, axis=1)
    good = (
        (forward_status.reshape(-1) == 1)
        & (backward_status.reshape(-1) == 1)
        & np.isfinite(error)
        & (error <= float(forward_backward_threshold_px))
        & _points_inside(current_xy, current_u8.shape[0], current_u8.shape[1])
        & _points_on_background(current_xy, current_exclude_mask)
    )
    return current_xy.astype(np.float32), good, error.astype(np.float32)


def _estimate_homography_to_reference(
    reference_xy: np.ndarray,
    current_xy: np.ndarray,
    good: np.ndarray,
    settings: StabilizationSettings,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Estimate a current-to-reference RANSAC homography from reliable matches.

    Args:
        reference_xy: Reference coordinates as ``[N,2]``.
        current_xy: Matching current-frame coordinates as ``[N,2]``.
        good: ``bool [N]`` mask selecting reliable matches.
        settings: Minimum match/inlier counts and the RANSAC threshold.
    Returns:
        A ``float64 [3,3]`` frame-to-reference matrix and statistics on
        success. The matrix is ``None`` on failure and statistics explain why.
    """
    source = current_xy[good]
    target = reference_xy[good]
    stats: dict[str, Any] = {
        "tracked": int(np.count_nonzero(good)),
        "inliers": 0,
        "valid": False,
    }
    if len(source) < settings.minimum_matches:
        stats["reason"] = f"tracked<{settings.minimum_matches}"
        return None, stats
    matrix, inlier_mask = cv2.findHomography(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(settings.ransac_threshold_px),
        maxIters=3000,
        confidence=0.995,
    )
    if (
        matrix is None
        or not np.isfinite(matrix).all()
        or abs(float(matrix[2, 2])) < 1e-12
    ):
        stats["reason"] = "findHomography_failed"
        return None, stats
    inliers = (
        np.ones(len(source), dtype=bool)
        if inlier_mask is None
        else inlier_mask.reshape(-1).astype(bool)
    )
    stats["inliers"] = int(np.count_nonzero(inliers))
    if stats["inliers"] < settings.minimum_inliers:
        stats["reason"] = f"inliers<{settings.minimum_inliers}"
        return None, stats
    matrix = matrix.astype(np.float64) / float(matrix[2, 2])
    reprojected = cv2.perspectiveTransform(
        source[inliers, None, :].astype(np.float64), matrix
    )[:, 0]
    reprojection_error = np.linalg.norm(reprojected - target[inliers], axis=1)
    stats.update(
        {
            "valid": True,
            "reason": "ok",
            "mean_reprojection_error_px": float(np.mean(reprojection_error)),
            "median_reprojection_error_px": float(np.median(reprojection_error)),
        }
    )
    return matrix, stats


def _fill_missing_homographies(
    homographies: list[np.ndarray | None], maximum_fallback_fraction: float
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Fill a limited number of failed homographies from the nearest valid frame.

    Args:
        homographies: One ``[3,3]`` matrix or ``None`` per frame.
        maximum_fallback_fraction: Maximum allowed fraction of failed frames.
    Returns:
        A matrix list without ``None`` and records describing which frames
        were filled. Too many failures raise an error.
    """
    valid_indices = [index for index, value in enumerate(homographies) if value is not None]
    if not valid_indices:
        raise ValueError("Stabilization produced no valid homographies")
    missing_count = len(homographies) - len(valid_indices)
    fraction = missing_count / len(homographies)
    if fraction > maximum_fallback_fraction:
        raise ValueError(
            f"Homography fallback fraction {fraction:.3f} exceeds "
            f"{maximum_fallback_fraction:.3f}"
        )
    filled: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for index, value in enumerate(homographies):
        source_index = index
        if value is None:
            source_index = min(valid_indices, key=lambda candidate: abs(candidate - index))
            value = homographies[source_index]
        assert value is not None
        filled.append(value.copy())
        records.append(
            {
                "frame_index": index,
                "filled": source_index != index,
                "source_index": source_index,
            }
        )
    return filled, records


def _smooth_homographies(
    homographies: list[np.ndarray], radius: int
) -> list[np.ndarray]:
    """Temporally smooth the eight free homography parameters.

    Args:
        homographies: Time-ordered ``[3,3]`` frame-to-reference matrices.
        radius: Number of neighboring frames on each side; zero disables
            smoothing.
    Returns:
        Time-ordered ``float64 [3,3]`` matrices with unchanged list length.
    """
    if radius == 0:
        return [value.astype(np.float64, copy=True) for value in homographies]
    parameters = np.asarray(
        [
            [
                matrix[0, 0],
                matrix[0, 1],
                matrix[0, 2],
                matrix[1, 0],
                matrix[1, 1],
                matrix[1, 2],
                matrix[2, 0],
                matrix[2, 1],
            ]
            for matrix in homographies
        ],
        dtype=np.float64,
    )
    output = parameters.copy()
    sigma = max(float(radius) / 3.0, 1e-6)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    for index in range(len(parameters)):
        lower = max(0, index - radius)
        upper = min(len(parameters), index + radius + 1)
        kernel_lower = lower - (index - radius)
        weights = kernel[kernel_lower : kernel_lower + (upper - lower)]
        weights /= np.sum(weights)
        output[index] = np.sum(parameters[lower:upper] * weights[:, None], axis=0)
    return [
        np.asarray(
            [
                [value[0], value[1], value[2]],
                [value[3], value[4], value[5]],
                [value[6], value[7], 1.0],
            ],
            dtype=np.float64,
        )
        for value in output
    ]


def stabilize_to_reference(
    frames_gray: np.ndarray,
    masks: np.ndarray,
    reference_index: int,
    settings: StabilizationSettings | None = None,
) -> StabilizationResult:
    """Estimate camera motion and align all grayscale frames to the reference.

    Args:
        frames_gray: Grayscale sequence as ``float [T,H,W]`` in 0--1.
        masks: Foreground masks as ``bool [T,H,W]``.
        reference_index: Temporal index of the reference frame.
        settings: Optional settings; ``None`` uses the accepted defaults.
    Returns:
        A ``StabilizationResult`` containing aligned frames/masks, valid areas,
        ``float64 [T,3,3]`` homographies, and tracking diagnostics.
    """
    config = settings or StabilizationSettings()
    _validate_settings(config)
    if frames_gray.ndim != 3 or frames_gray.shape[0] < 3:
        raise ValueError("frames_gray must be [T,H,W] with at least three frames")
    if masks.shape != frames_gray.shape or masks.dtype != np.bool_:
        raise ValueError("masks must be bool [T,H,W] matching frames_gray")
    if not np.isfinite(frames_gray).all():
        raise ValueError("frames_gray contains NaN or Inf")
    if np.any(frames_gray < 0.0) or np.any(frames_gray > 1.0):
        raise ValueError("frames_gray intensities must be in [0,1]")
    if reference_index < 0 or reference_index >= frames_gray.shape[0]:
        raise ValueError("reference_index is outside the frame sequence")

    frame_count, height, width = frames_gray.shape
    frames_u8 = np.clip(frames_gray * 255.0, 0.0, 255.0).astype(np.uint8)
    dilated_masks = np.stack(
        [_dilate_mask(mask, config.mask_dilate_iterations) for mask in masks], axis=0
    )
    reference_u8 = frames_u8[reference_index]
    candidate_points = _detect_reference_features(
        reference_u8, dilated_masks[reference_index], config
    )

    success_count = np.zeros(len(candidate_points), dtype=np.int64)
    error_sum = np.zeros(len(candidate_points), dtype=np.float64)
    for frame_index in range(frame_count):
        if frame_index == reference_index:
            success_count += 1
            continue
        _, good, error = _track_reference_points(
            reference_u8,
            frames_u8[frame_index],
            candidate_points,
            dilated_masks[frame_index],
            config,
            config.pass1_forward_backward_threshold_px,
        )
        success_count += good.astype(np.int64)
        error_sum += np.where(good, error, 0.0)
    success_fraction = success_count / frame_count
    average_error = np.divide(
        error_sum,
        success_count,
        out=np.full(len(candidate_points), np.inf, dtype=np.float64),
        where=success_count > 0,
    )
    stable = (
        (success_fraction >= config.minimum_track_success_fraction)
        & (average_error <= config.maximum_average_forward_backward_error_px)
    )
    if int(np.count_nonzero(stable)) < config.minimum_stable_points:
        raise ValueError(
            f"Only {int(np.count_nonzero(stable))} stable background features remain; "
            f"need at least {config.minimum_stable_points}"
        )
    stable_reference_points = candidate_points[stable]
    stable_reference_xy = stable_reference_points.reshape(-1, 2)

    raw_homographies: list[np.ndarray | None] = []
    per_frame_stats: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        if frame_index == reference_index:
            raw_homographies.append(np.eye(3, dtype=np.float64))
            per_frame_stats.append(
                {
                    "frame_index": frame_index,
                    "tracked": len(stable_reference_points),
                    "inliers": len(stable_reference_points),
                    "valid": True,
                    "reason": "reference",
                    "mean_forward_backward_error_px": 0.0,
                }
            )
            continue
        current_xy, good, error = _track_reference_points(
            reference_u8,
            frames_u8[frame_index],
            stable_reference_points,
            dilated_masks[frame_index],
            config,
            config.pass2_forward_backward_threshold_px,
        )
        matrix, stats = _estimate_homography_to_reference(
            stable_reference_xy, current_xy, good, config
        )
        stats["frame_index"] = frame_index
        stats["mean_forward_backward_error_px"] = (
            float(np.mean(error[good])) if np.any(good) else None
        )
        raw_homographies.append(matrix)
        per_frame_stats.append(stats)

    filled, fill_records = _fill_missing_homographies(
        raw_homographies, config.maximum_fallback_fraction
    )
    smoothed = _smooth_homographies(
        filled, config.temporal_smoothing_radius
    )
    homographies = np.stack(smoothed, axis=0)

    stabilized_gray = np.empty_like(frames_gray, dtype=np.float32)
    stabilized_masks = np.empty_like(masks, dtype=bool)
    valid_mask = np.empty_like(masks, dtype=bool)
    ones = np.ones((height, width), dtype=np.uint8)
    for frame_index, matrix in enumerate(homographies):
        stabilized_gray[frame_index] = cv2.warpPerspective(
            frames_gray[frame_index],
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        stabilized_masks[frame_index] = cv2.warpPerspective(
            masks[frame_index].astype(np.uint8),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        valid_mask[frame_index] = cv2.warpPerspective(
            ones,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0

    statistics = {
        "detected_reference_points": int(len(candidate_points)),
        "stable_reference_points": int(len(stable_reference_points)),
        "minimum_stable_success_fraction": float(np.min(success_fraction[stable])),
        "maximum_stable_average_forward_backward_error_px": float(
            np.max(average_error[stable])
        ),
        "valid_homographies": int(sum(value is not None for value in raw_homographies)),
        "filled_homographies": int(sum(value is None for value in raw_homographies)),
        "per_frame": per_frame_stats,
        "fill": fill_records,
    }
    return StabilizationResult(
        frames_gray=stabilized_gray,
        masks=stabilized_masks,
        valid_mask=valid_mask,
        homographies_frame_to_reference=homographies,
        statistics=statistics,
        settings=config,
    )


def write_stabilized_sequence(
    directory: Path,
    sequence: ImageMaskSequence,
    result: StabilizationResult,
) -> dict[str, Any]:
    """Write stabilization results as a structured derived image/mask sequence.

    Args:
        directory: New output directory, such as
            ``artifact/stabilized_sequence``.
        sequence: Source ``ImageMaskSequence`` providing names and paths.
        result: Output from ``stabilize_to_reference``.
    Returns:
        The same dictionary written to ``manifest.json`` after writing images,
        masks, homographies, and diagnostics.
    """
    images_directory = directory / "images"
    masks_directory = directory / "masks"
    images_directory.mkdir(parents=True)
    masks_directory.mkdir(parents=True)
    height, width = sequence.height, sequence.width
    for frame_index, frame_name in enumerate(sequence.frame_names):
        color = read_color_image(sequence.image_paths[frame_index])
        stabilized_color = cv2.warpPerspective(
            color,
            result.homographies_frame_to_reference[frame_index],
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        image_path = images_directory / f"{frame_name}.png"
        mask_path = masks_directory / f"{frame_name}.png"
        if not cv2.imwrite(str(image_path), stabilized_color):
            raise OSError(f"Failed to write stabilized image: {image_path}")
        if not cv2.imwrite(
            str(mask_path), result.masks[frame_index].astype(np.uint8) * 255
        ):
            raise OSError(f"Failed to write stabilized mask: {mask_path}")
        read_binary_mask(mask_path, (height, width))

    homographies_path = directory / "homographies_frame_to_reference.npy"
    np.save(
        homographies_path,
        result.homographies_frame_to_reference,
        allow_pickle=False,
    )
    statistics_path = directory / "statistics.json"
    with statistics_path.open("w", encoding="utf-8") as stream:
        json.dump(result.statistics, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    manifest = {
        "format": "modal_gaussians.stabilized_image_mask_sequence",
        "version": 2,
        "fps_hz": sequence.fps_hz,
        "reference_frame": sequence.reference_frame_name,
        "width": width,
        "height": height,
        "settings": asdict(result.settings),
        "frames": list(sequence.frame_names),
        "homographies": {
            "file": homographies_path.name,
            "dtype": result.homographies_frame_to_reference.dtype.name,
            "shape": list(result.homographies_frame_to_reference.shape),
        },
        "statistics": {
            "file": statistics_path.name,
        },
    }
    with (directory / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest
