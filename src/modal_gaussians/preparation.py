"""Optional video/SAM/XMem preparation; core analysis consumes only its PNGs."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from threading import Event
from typing import Any
import warnings

import cv2
import numpy as np

from modal_gaussians.data.sequence import (
    _list_pngs,
    read_color_image,
    validate_image_mask_sequence,
)
from modal_gaussians.video_color import resolve_video_color, tone_map_rgb16

CHECKPOINTS = {
    "sam": "sam_vit_h_4b8939.pth",
    "xmem": "XMem-s012.pth",
}
XMEM_CONFIG = {
    "max_mid_term_frames": 10,
    "min_mid_term_frames": 5,
    "max_long_term_elements": 1000,
    "num_prototypes": 128,
    "top_k": 30,
    "query_chunk_size": 256,
    "mem_every": 5,
    "deep_update_every": -1,
    "enable_long_term": True,
    "enable_long_term_count_usage": True,
}
Progress = Callable[[int, int, str], None]


class PreparationCancelled(RuntimeError):
    """A cooperative cancellation before publication, preserving old outputs."""


class _RecoveryRequired(RuntimeError):
    """Keep the staging directory when the filesystem prevents rollback."""


def _check_cancel(cancel: Event | None) -> None:
    """Stop between frames, never during a model call or publication."""
    if cancel is not None and cancel.is_set():
        raise PreparationCancelled("Cancelled; previously published results are unchanged.")


def _positive(value: float, name: str) -> float:
    """Reject missing, non-finite, or non-positive sampling parameters."""
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _sequence_name(name: str) -> str:
    """Allow one portable directory name, never paths or Windows device names."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Sequence name must contain only letters, digits, '-' or '_'")
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", name, re.IGNORECASE):
        raise ValueError("Reserved Windows sequence name")
    return name


def _overlap(left: Path, right: Path) -> bool:
    """Detect either direction of containment before overwriting owned outputs."""
    a, b = left.resolve(), right.resolve()
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def _assert_owned(path: Path, root: Path) -> None:
    """Refuse recursive operations outside root or through links/junctions."""
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Not an owned preparation path: {path}")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink() or (part.exists() and part.resolve() != part):
            raise ValueError(f"Preparation outputs cannot use links/junctions: {part}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Preparation path escapes its root: {path}")


def _remove_owned(path: Path, root: Path) -> None:
    """Delete only one validated output or temporary subtree."""
    _assert_owned(path, root)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


@contextmanager
def _staging(root: Path) -> Iterator[Path]:
    """Serialize publishers across processes and clean failed temporary output."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".prepare.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(
            f"Preparation root is busy ({lock}). After a crashed process, inspect "
            "any .prepare-* recovery directories before removing this lock."
        ) from error
    stage: Path | None = None
    keep_recovery = False
    try:
        handle.write(str(os.getpid()))
        handle.flush()
        stage = Path(tempfile.mkdtemp(prefix=".prepare-", dir=root))
        yield stage
    except _RecoveryRequired:
        keep_recovery = True
        raise
    finally:
        handle.close()
        if not keep_recovery:
            try:
                if stage is not None:
                    _remove_owned(stage, root)
            except OSError as error:
                warnings.warn(f"Temporary preparation files remain at {stage}: {error}")
            lock.unlink()


def _publish(root: Path, stage: Path, changes: list[tuple[Path, Path | None]]) -> None:
    """Replace whole directories with rollback; None invalidates old masks.

    Windows cannot atomically replace a nonempty directory. Existing targets
    are first renamed into this same-filesystem stage, then new targets are
    renamed into place. A caught failure restores every old target. The caller
    deletes rollback copies only after successful publication.
    """
    for target, source in changes:
        _assert_owned(target, root)
        if source is not None:
            _assert_owned(source, stage)
        target.parent.mkdir(parents=True, exist_ok=True)
    journal = []
    for index, (target, source) in enumerate(changes):
        journal.append({"target": str(target), "source": str(source) if source else None,
                        "backup": str(stage / f"previous-{index}")})
    _write_json(stage / "publication.json", journal)
    applied: list[tuple[Path, Path, bool]] = []
    try:
        for index, (target, source) in enumerate(changes):
            backup = stage / f"previous-{index}"
            existed = target.exists()
            applied.append((target, backup, existed))
            if existed:
                target.rename(backup)
            if source is not None:
                source.rename(target)
    except BaseException:
        try:
            for target, backup, existed in reversed(applied):
                if backup.exists():
                    _remove_owned(target, root)
                    backup.rename(target)
                elif not existed:
                    _remove_owned(target, root)
        except BaseException as error:
            raise _RecoveryRequired(
                f"Publication rollback needs manual recovery; originals and journal "
                f"are retained at {stage}. Do not delete this directory."
            ) from error
        raise


def _write_json(path: Path, value: Any) -> None:
    """Write preparation metadata only outside image/mask PNG directories."""
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _snapshot(paths: tuple[Path, ...]) -> tuple[tuple[str, int, int], ...]:
    """Detect source edits during an interactive session without caching frames."""
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


@dataclass(frozen=True)
class PreparedImages:
    """An ordered, fixed-resolution PNG input; no pixel/video stack is retained."""
    directory: Path
    paths: tuple[Path, ...]
    height: int
    width: int
    snapshot: tuple[tuple[str, int, int], ...]

    def assert_unchanged(self) -> None:
        """Reject a stale prompt after external replacement or editing of images."""
        paths = _list_pngs(self.directory, "Image")
        if _snapshot(paths) != self.snapshot:
            raise ValueError("Images changed; load the sequence again and create a new prompt")


def inspect_images(directory: str | Path, cancel: Event | None = None) -> PreparedImages:
    """Validate at least three same-size uint8 RGB PNGs, one frame at a time."""
    path = Path(directory).expanduser().resolve()
    paths = _list_pngs(path, "Image")
    if len(paths) < 3:
        raise ValueError("At least three PNG frames are required")
    before = _snapshot(paths)
    height, width = read_color_image(paths[0]).shape[:2]
    for frame in paths:
        _check_cancel(cancel)
        if read_color_image(frame).shape[:2] != (height, width):
            raise ValueError(f"Frame dimensions differ: {frame}")
    if _snapshot(paths) != before:
        raise ValueError("Image files changed during validation")
    return PreparedImages(path, paths, height, width, before)


def read_rgb(path: Path) -> np.ndarray:
    """Convert validated OpenCV BGR input to SAM/XMem's RGB convention."""
    return cv2.cvtColor(read_color_image(path), cv2.COLOR_BGR2RGB)


def media_executable(name: str) -> str:
    """Find FFmpeg/ffprobe on PATH or in this Python's Conda environment."""
    found = shutil.which(name)
    if found:
        return found
    suffix = ".exe" if os.name == "nt" else ""
    for folder in (Path(sys.prefix) / "Library" / "bin", Path(sys.prefix) / "bin"):
        candidate = folder / f"{name}{suffix}"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"{name} not found; install conda-forge::ffmpeg in this environment")


def _process_options() -> dict[str, Any]:
    """Keep Windows helpers hidden while using ordinary pipes on Linux."""
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def probe_video(path: Path) -> dict[str, Any]:
    """Read video timing, rotation, and HDR metadata without decoding the full file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    result = subprocess.run(
        [media_executable("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
        **_process_options(),
    )
    if result.returncode:
        raise RuntimeError(f"ffprobe failed: {result.stderr[-2000:]}")
    info = json.loads(result.stdout)
    if not info.get("streams"):
        raise ValueError("Input has no video stream")
    return info


class PreparationWorkspace:
    """Own only images/name, masks/name and metadata/name.json under one root."""

    def __init__(self, root: str | Path):
        """Resolve the explicit output root, leaving all source inputs in place."""
        self.root = Path(root).expanduser().resolve()

    def paths(self, sequence: str) -> tuple[Path, Path, Path]:
        """Return checked output targets for one sequence."""
        name = _sequence_name(sequence)
        targets = (self.root / "images" / name, self.root / "masks" / name,
                   self.root / "metadata" / f"{name}.json")
        for target in targets:
            _assert_owned(target, self.root)
        return targets

    def extract(
        self, video: str | Path, sequence: str, *, fps: float, start: float = 0,
        end: float | None = None, height: int | None = None,
        color_mode: str = "auto",
        cancel: Event | None = None, progress: Progress | None = None,
    ) -> dict[str, Any]:
        """Extract/validate a complete clip, then replace frames and clear masks."""
        fps = _positive(fps, "Output FPS")
        if not math.isfinite(start) or start < 0:
            raise ValueError("Start time must be finite and non-negative")
        if end is not None and (not math.isfinite(end) or end <= start):
            raise ValueError("End time must be finite and greater than start")
        if height is not None and (height <= 0 or int(height) != height):
            raise ValueError("Height must be a positive integer or blank")
        source = Path(video).expanduser().resolve()
        images, masks, metadata = self.paths(sequence)
        if any(_overlap(source, target) for target in (images, masks, metadata)):
            raise ValueError("Source video cannot be inside an overwritten output")
        info = probe_video(source)
        color = resolve_video_color(info, color_mode)
        source_stat = source.stat()
        executable = media_executable("ffmpeg")
        with _staging(self.root) as stage:
            staged_images = stage / "images"
            staged_images.mkdir()
            filters = color.decode_filters(int(height) if height is not None else None, fps)
            command = [executable, "-hide_banner", "-loglevel", "error", "-nostdin",
                       "-ss", f"{start:.12g}", "-i", str(source)]
            if end is not None:
                command += ["-t", f"{end-start:.12g}"]
            command += ["-map", "0:v:0", "-an", "-vf", ",".join(filters),
                        "-start_number", "1", str(staged_images / "%05d.png")]
            _check_cancel(cancel)
            with (stage / "ffmpeg.log").open("wb") as log:
                process = subprocess.Popen(command, stdout=log, stderr=log, **_process_options())
                try:
                    while process.poll() is None:
                        _check_cancel(cancel)
                        if progress is not None:
                            progress(0, 0, "Extracting video frames…")
                        (cancel or Event()).wait(0.2)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
            if process.returncode:
                message = (stage / "ffmpeg.log").read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(f"FFmpeg failed: {message[-2000:]}")
            if color.transfer is not None:
                frames = _list_pngs(staged_images, "HDR frame")
                for index, frame in enumerate(frames):
                    _check_cancel(cancel)
                    image = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise ValueError(f"Cannot read decoded HDR frame: {frame}")
                    converted = tone_map_rgb16(image[..., ::-1], color)
                    if not cv2.imwrite(str(frame), converted[..., ::-1]):
                        raise OSError(f"Cannot write tone-mapped frame: {frame}")
                    if progress is not None:
                        progress(index + 1, len(frames), "Tone mapping HDR to SDR / sRGB…")
            prepared = inspect_images(staged_images, cancel)
            if len(prepared.paths) > 99999:
                raise ValueError("Clip exceeds five-digit frame numbering; use a shorter clip")
            current_stat = source.stat()
            if (current_stat.st_size, current_stat.st_mtime_ns) != (
                source_stat.st_size, source_stat.st_mtime_ns
            ):
                raise ValueError("Source video changed during extraction")
            record = {
                "format": "modal-gaussians-preparation", "version": 1,
                "sequence": sequence, "source_video": str(source),
                "source_video_size": source_stat.st_size,
                "source_video_mtime_ns": source_stat.st_mtime_ns,
                "clip_start_seconds": start, "clip_end_seconds": end,
                "fps_hz": fps, "requested_height": height,
                "width": prepared.width, "height": prepared.height,
                "frame_count": len(prepared.paths),
                "images": str(images), "masks": str(masks),
                "mask_status": "pending", "prompt_frame": None,
                "rotation_policy": "ffmpeg_default_autorotate", "video_probe": info,
                "color_processing": color.metadata(), "extraction_filters": filters,
            }
            _write_json(stage / "metadata.json", record)
            _check_cancel(cancel)
            # Invalidate masks before exposing new images, in one rollback transaction.
            _publish(self.root, stage, [(masks, None), (images, staged_images),
                                       (metadata, stage / "metadata.json")])
        return record

    def track(
        self, images: PreparedImages, sequence: str, *, fps: float,
        prompt_index: int, prompt_mask: np.ndarray, tracker: Any,
        checkpoint_info: dict[str, Any], cancel: Event | None = None,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        """Stream forward/backward XMem inference, validate and replace masks."""
        fps = _positive(fps, "Sequence FPS")
        _, masks, metadata = self.paths(sequence)
        if _overlap(images.directory, masks) or _overlap(images.directory, metadata):
            raise ValueError("External input images overlap an overwritten output")
        images.assert_unchanged()
        if not 0 <= prompt_index < len(images.paths):
            raise ValueError("Prompt frame is outside the sequence")
        if prompt_mask.shape != (images.height, images.width) or not np.any(prompt_mask):
            raise ValueError("Prompt must be a nonempty foreground mask matching the image")
        prompt = (prompt_mask > 0).astype(np.uint8)
        record: dict[str, Any] = {}
        if metadata.is_file():
            old = json.loads(metadata.read_text(encoding="utf-8"))
            if old.get("images") == str(images.directory):
                record = old
                if record.get("source_video") and float(record["fps_hz"]) != fps:
                    raise ValueError("Sequence FPS differs from extraction metadata")
        record.update({
            "format": "modal-gaussians-preparation", "version": 1,
            "sequence": sequence, "images": str(images.directory), "masks": str(masks),
            "fps_hz": fps, "frame_count": len(images.paths),
            "height": images.height, "width": images.width,
            "mask_status": "ready", "prompt_frame": images.paths[prompt_index].stem,
            "prompt_frame_index": prompt_index, "checkpoints": checkpoint_info,
            "tracker": "XMem-s012", "tracker_config": XMEM_CONFIG,
            "mask_values": {"background": 0, "foreground": 255},
        })
        with _staging(self.root) as stage:
            staged_masks = stage / "masks"
            staged_masks.mkdir()
            count = 0
            try:
                for direction, indices in (
                    ("forward", range(prompt_index, len(images.paths))),
                    ("backward", range(prompt_index - 1, -1, -1)),
                ):
                    tracker.clear_memory()
                    if direction == "backward" and prompt_index > 0:
                        _check_cancel(cancel)
                        tracker.track(read_rgb(images.paths[prompt_index]), prompt)
                    for index in indices:
                        _check_cancel(cancel)
                        frame = read_rgb(images.paths[index])
                        mask = tracker.track(frame, prompt if index == prompt_index else None)
                        if mask.shape != prompt.shape or not np.isfinite(mask).all():
                            raise ValueError(f"Invalid tracked mask at frame {index}")
                        output = np.asarray(mask > 0, dtype=np.uint8) * 255
                        if not cv2.imwrite(str(staged_masks / images.paths[index].name), output):
                            raise OSError(f"Could not save mask for {images.paths[index].name}")
                        count += 1
                        if progress is not None:
                            progress(count, len(images.paths), f"Tracking {direction}: {count}/{len(images.paths)}")
            finally:
                tracker.clear_memory()
            _check_cancel(cancel)
            validate_image_mask_sequence(
                image_dir=images.directory, mask_dir=staged_masks, fps_hz=fps,
                reference_frame_name=images.paths[prompt_index].stem,
            )
            images.assert_unchanged()
            _write_json(stage / "metadata.json", record)
            _check_cancel(cancel)
            _publish(self.root, stage, [(masks, staged_masks), (metadata, stage / "metadata.json")])
        return record


def checkpoint_identity(directory: str | Path) -> dict[str, Any]:
    """Require explicit official weight filenames and record their SHA-256."""
    root = Path(directory).expanduser().resolve()
    paths = {name: root / filename for name, filename in CHECKPOINTS.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing model weight: {path}; see README mask setup")
    result = {}
    for name, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result[name] = {"path": str(path), "sha256": digest.hexdigest()}
    return result


class XMemTracker:
    """Binary wrapper preserving the accepted full-resolution inference path."""

    def __init__(self, checkpoint: Path, device: str):
        """Load the full checkpoint strictly; never fetch redundant ResNet weights."""
        import torch
        from torchvision import transforms
        from modal_gaussians._vendor.xmem.inference.inference_core import InferenceCore
        from modal_gaussians._vendor.xmem.model.network import XMem
        from modal_gaussians._vendor.xmem.util.mask_mapper import MaskMapper
        from modal_gaussians._vendor.xmem.util.range_transform import im_normalization

        self.torch = torch
        self.device = device
        config = dict(XMEM_CONFIG)
        self.network = XMem(config, str(checkpoint)).to(device).eval()
        self.core = InferenceCore(self.network, config)
        self.mapper = MaskMapper()
        self.transform = transforms.Compose([transforms.ToTensor(), im_normalization])

    def track(self, frame: np.ndarray, annotation: np.ndarray | None = None) -> np.ndarray:
        """Return uint8 foreground labels for one RGB frame without resizing."""
        with self.torch.inference_mode():
            mask, labels = None, None
            if annotation is not None:
                mask, labels = self.mapper.convert_mask(annotation)
                mask = mask.to(self.device)
                self.core.set_all_labels(list(self.mapper.remappings.values()))
            frame_tensor = self.torch.as_tensor(self.transform(frame), device=self.device)
            probabilities, _ = self.core.step(frame_tensor, mask, labels)
            indices = probabilities.argmax(dim=0).cpu().numpy().astype(np.uint8)
            return self.mapper.remap_index_mask(indices)

    def clear_memory(self) -> None:
        """Separate forward/backward passes and successive video sessions."""
        self.core.clear_memory()
        self.mapper.clear_labels()


class MaskPrompt:
    """One prompting image, positive/negative clicks and a union of foregrounds."""

    def __init__(self, checkpoint_directory: str | Path, device: str = "cuda"):
        """Keep weights lazy; no network downloads occur during GUI interaction."""
        self.checkpoint_directory = Path(checkpoint_directory).expanduser().resolve()
        self.device = device
        self.predictor: Any = None
        self.image: np.ndarray | None = None
        self.accepted: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.logits: np.ndarray | None = None
        self.points: list[tuple[int, int]] = []
        self.labels: list[int] = []

    def set_image(self, image: np.ndarray) -> None:
        """Changing frame/sequence invalidates every prompt and SAM embedding."""
        self.image = image
        self.accepted = np.zeros(image.shape[:2], dtype=bool)
        self.clear_points()
        if self.predictor is not None:
            self.predictor.reset_image()

    def features(self) -> None:
        """Load SAM ViT-H once and encode the currently selected image."""
        if self.image is None:
            raise ValueError("Load a sequence and select a prompt frame first")
        if self.predictor is None:
            from segment_anything import SamPredictor, sam_model_registry
            path = self.checkpoint_directory / CHECKPOINTS["sam"]
            if not path.is_file():
                raise FileNotFoundError(path)
            model = sam_model_registry["vit_h"](checkpoint=str(path))
            self.predictor = SamPredictor(model.to(self.device).eval())
        self.predictor.set_image(self.image)

    def add_point(self, x: int, y: int, positive: bool) -> np.ndarray:
        """Use SAM's highest-score mask and previous logits, as in the old GUI."""
        if self.predictor is None or not self.predictor.is_image_set or self.image is None:
            raise ValueError("Click Get SAM features first")
        if not (0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]):
            raise ValueError("Point is outside the image")
        self.points.append((x, y))
        self.labels.append(int(positive))
        masks, scores, logits = self.predictor.predict(
            point_coords=np.asarray(self.points), point_labels=np.asarray(self.labels),
            mask_input=None if self.logits is None else self.logits[None],
            multimask_output=True,
        )
        selected = int(np.argmax(scores))
        self.current, self.logits = masks[selected], logits[selected]
        return self.preview()

    def clear_points(self) -> None:
        """Discard current object prompts but retain previously added foregrounds."""
        self.points = []
        self.labels = []
        self.current = None
        self.logits = None

    def add_foreground(self) -> None:
        """Accept the current foreground into the binary union and start another."""
        if self.current is None or self.accepted is None:
            raise ValueError("Select a foreground mask first")
        self.accepted |= self.current
        self.clear_points()

    def mask(self) -> np.ndarray:
        """Union accepted and current masks; do not create multi-object labels."""
        if self.accepted is None:
            raise ValueError("No prompting image loaded")
        return self.accepted | self.current if self.current is not None else self.accepted.copy()

    def preview(self) -> np.ndarray:
        """Show a single-frame mask overlay and click markers, not a video preview."""
        if self.image is None:
            raise ValueError("No prompting image loaded")
        output = self.image.copy()
        mask = self.mask()
        output[mask] = (0.55 * output[mask] + 0.45 * np.array([40, 220, 120])).astype(np.uint8)
        for (x, y), label in zip(self.points, self.labels):
            cv2.circle(output, (x, y), 5, (30, 240, 60) if label else (255, 50, 50), -1)
        return output

    def release(self) -> None:
        """Release SAM before XMem tracking, preserving the small prompt arrays."""
        self.predictor = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
