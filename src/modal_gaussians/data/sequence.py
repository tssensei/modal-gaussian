from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _list_pngs(directory: Path, label: str) -> tuple[Path, ...]:
    """List every PNG in lexicographic filename order.

    Args:
        directory: Image or mask directory, for example ``Path("images")``.
        label: Category name used in error messages, such as ``"Image"``.
    Returns:
        Ordered paths, for example ``(Path("images/frame_000001.png"), ...)``.
        Zero-padded filenames make this order equal to temporal order.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    paths: list[Path] = []
    unsupported: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".png":
            unsupported.append(path.name)
            continue
        paths.append(path)
    if unsupported:
        raise ValueError(
            f"{label} directory contains non-PNG files: {unsupported[:5]}"
        )
    if not paths:
        raise ValueError(f"{label} directory contains no PNG files: {directory}")
    stems = [path.stem for path in paths]
    if len(set(stems)) != len(stems):
        raise ValueError(f"{label} directory contains duplicate PNG stems")
    return tuple(paths)


def read_color_image(path: Path) -> np.ndarray:
    """Read and validate one three-channel color PNG.

    Args:
        path: Image path, for example ``Path("images/frame_0001.png")``.
    Returns:
        A ``uint8 [H,W,3]`` array in OpenCV BGR order with values from
        0 through 255.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode RGB frame: {path}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"RGB frame must be uint8 [H,W,3], got {image.dtype} {image.shape}: {path}"
        )
    return image


def read_binary_mask(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    """Read a mask and validate its size and binary single-channel format.

    Args:
        path: Path to the mask PNG.
        expected_hw: Matching image size as ``(height, width)``, for example
            ``(1080, 1920)``.
    Returns:
        A ``bool [H,W]`` array with ``True`` for foreground and ``False``
        for background.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to decode mask: {path}")
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError(
            f"Mask must be a single-channel uint8 PNG, got {mask.dtype} {mask.shape}: {path}"
        )
    if mask.shape != expected_hw:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {expected_hw}: {path}"
        )
    values = np.unique(mask)
    if values.size > 2 or (values.size == 2 and int(values[0]) != 0):
        raise ValueError(
            f"Mask must contain at most one nonzero foreground value, "
            f"got {values.tolist()}: {path}"
        )
    return mask > 0


@dataclass(frozen=True)
class ImageMaskSequence:
    """Describe a validated, ordered image and mask sequence.

    The object stores lexicographically ordered paths, FPS, reference frame,
    and dimensions. The pipeline decodes pixels one frame at a time.
    """

    image_dir: Path
    mask_dir: Path
    frame_names: tuple[str, ...]
    image_paths: tuple[Path, ...]
    mask_paths: tuple[Path, ...]
    fps_hz: float
    reference_frame_name: str
    reference_frame_index: int
    height: int
    width: int

    @property
    def frame_count(self) -> int:
        """Return the number of ordered frames; 120 images return ``120``."""
        return len(self.frame_names)

    def read_frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Decode one grayscale frame and mask without allocating the video stack."""

        image = read_color_image(self.image_paths[index])
        if image.shape[:2] != (self.height, self.width):
            raise ValueError(f"Image dimensions changed: {self.image_paths[index]}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return gray, read_binary_mask(self.mask_paths[index], (self.height, self.width))

    def read_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Load every image and mask in the discovered filename order.

        Returns:
            ``frames_gray``: ``float32 [T,H,W]`` grayscale values in 0--1.
            ``masks``: ``bool [T,H,W]`` with ``True`` for foreground.
        """
        gray = np.empty(
            (self.frame_count, self.height, self.width), dtype=np.float32
        )
        masks = np.empty(
            (self.frame_count, self.height, self.width), dtype=bool
        )
        for index in range(self.frame_count):
            gray[index], masks[index] = self.read_frame(index)
        return gray, masks

def validate_image_mask_sequence(
    *,
    image_dir: str | Path,
    mask_dir: str | Path,
    fps_hz: float,
    reference_frame_name: str,
) -> ImageMaskSequence:
    """Validate image/mask inputs and build the sequence used downstream.

    Args:
        image_dir: Directory containing PNG frames.
        mask_dir: Directory containing binary PNG masks with matching stems.
        fps_hz: Sampling rate, for example ``30.0``.
        reference_frame_name: Reference stem, for example ``"frame_0060"``.
    Returns:
        An ``ImageMaskSequence`` containing ordered paths, dimensions,
        FPS, and the selected reference index.
    """
    fps = float(fps_hz)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("FPS must be finite and positive")

    images_root = Path(image_dir).expanduser().resolve()
    masks_root = Path(mask_dir).expanduser().resolve()
    image_paths = _list_pngs(images_root, "Image")
    discovered_mask_paths = _list_pngs(masks_root, "Mask")
    frame_names = tuple(path.stem for path in image_paths)
    if len(frame_names) < 3:
        raise ValueError("At least three ordered frames are required for modal analysis")
    masks_by_stem = {path.stem: path for path in discovered_mask_paths}
    missing_masks = sorted(set(frame_names) - set(masks_by_stem))
    extra_masks = sorted(set(masks_by_stem) - set(frame_names))
    if missing_masks or extra_masks:
        raise ValueError(
            "Mask stems must exactly match image stems: "
            f"missing={missing_masks[:5]}, extra={extra_masks[:5]}"
        )
    mask_paths = tuple(masks_by_stem[name] for name in frame_names)
    if reference_frame_name not in frame_names:
        raise ValueError(
            f"Reference frame {reference_frame_name!r} is not present in the image directory"
        )

    first = read_color_image(image_paths[0])
    height, width = first.shape[:2]
    for image_path, mask_path in zip(image_paths, mask_paths):
        image = read_color_image(image_path)
        if image.shape[:2] != (height, width):
            raise ValueError(
                f"Image shape mismatch: {image_path} has {image.shape[:2]}, "
                f"expected {(height, width)}"
            )
        read_binary_mask(mask_path, (height, width))

    return ImageMaskSequence(
        image_dir=images_root,
        mask_dir=masks_root,
        frame_names=frame_names,
        image_paths=image_paths,
        mask_paths=mask_paths,
        fps_hz=fps,
        reference_frame_name=reference_frame_name,
        reference_frame_index=frame_names.index(reference_frame_name),
        height=height,
        width=width,
    )
