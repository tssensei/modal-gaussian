"""Explicit offline export of one RGB-fitted recording and its reconstruction."""
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
import torch

from modal_gaussians.preparation import media_executable, _process_options
from modal_gaussians.progress import Progress
from modal_gaussians.result import load_modal_result
from modal_gaussians.rgb_rendering import make_rgb_renderer
from modal_gaussians.scene_store import resolve_path
from modal_gaussians.static import cameras_from_scene_manifest


def export_result_video(*, result_dir, view_label, output_dir, device="cuda") -> Path:
    """Stream original-above-reconstruction MP4 into a new, atomic output directory."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Video output already exists: {destination}")
    result = load_modal_result(result_dir)
    if result.manifest["coordinate_source"]["kind"] != "rgb":
        raise ValueError("Video comparison requires an RGB-fitted result with recorded input images")
    views = {view["label"]: view for view in result.manifest["views"]}
    if view_label not in views:
        raise ValueError(f"Unknown view {view_label!r}; available: {', '.join(views)}")
    view = views[view_label]
    sources = [item for item in result.coordinates.manifest["images"] if item["label"] == view_label]
    if len(sources) != 1:
        raise ValueError("RGB result must record exactly one image source for the selected view")
    source = sources[0]
    directory = resolve_path(source["directory"], strict=True)
    protected = [result.path, directory] + [
        resolve_path(item["path"], strict=True) for item in result.manifest["sources"].values()
    ]
    if any(destination.is_relative_to(path) for path in protected):
        raise ValueError("Video output must be outside immutable input artifacts and image directories")
    files = source["files"]
    count, offset = view["frame_count"], view["frame_offset"]
    if count < 1 or len(files) != count or len(view["frame_names"]) != count:
        raise ValueError("RGB source frame count differs from the selected view")
    for frame, record in zip(view["frame_names"], files):
        name = record["name"]
        if name != f"{frame}.png" or Path(name).name != name or not (directory / name).is_file():
            raise ValueError(f"Missing or mismatched recorded RGB frame: {name!r}")
    fps = float(view["fps_hz"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS must be finite and positive")
    cameras = {camera.name: camera for camera in cameras_from_scene_manifest(result.scene.manifest)}
    camera = cameras[view["camera_name"]]
    height, width = view["shape_hw"]
    if [camera.height, camera.width] != [height, width]:
        raise ValueError("Video camera dimensions differ from the recorded RGB frames")
    ffmpeg = media_executable("ffmpeg")
    render = make_rgb_renderer(result.scene, camera, result.completed_modes.arrays["phi"],
                               result.completed_modes.rotation, device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary:
        work = Path(temporary)
        video = work / "comparison.mp4"
        # Pad only the right edge for odd widths; both panels retain their original pixels.
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{2 * height}",
                   "-framerate", str(fps), "-i", "pipe:0", "-an",
                   "-vf", "pad=ceil(iw/2)*2:ih", "-c:v", "libx264", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
        with (work / "encode.log").open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=log, **_process_options())
            try:
                progress = Progress(f"Export video {view_label}", count, unit="frames")
                with torch.inference_mode():
                    for index, record in enumerate(files):
                        raw = (directory / record["name"]).read_bytes()
                        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                            raise ValueError(f"RGB source changed since fitting: {record['name']}")
                        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                        if bgr is None or bgr.shape != (height, width, 3):
                            raise ValueError(f"RGB source has invalid dimensions: {record['name']}")
                        original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        # Fitted q already includes the pose offset; never re-zero or amplify it.
                        q = torch.tensor(result.coordinates.coordinates[offset + index],
                                         dtype=torch.complex64, device=device)
                        prediction = render(q, 1.0)
                        if tuple(prediction.shape) != (height, width, 3) or not torch.isfinite(prediction).all():
                            raise ValueError(f"Invalid reconstructed RGB frame: {record['name']}")
                        reconstructed = prediction.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
                        process.stdin.write(np.concatenate((original, reconstructed), axis=0).tobytes())
                        progress.update(index + 1)
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("FFmpeg failed")
            except BaseException as error:
                if process.poll() is None:
                    process.kill()
                process.wait()
                if isinstance(error, (BrokenPipeError, RuntimeError)):
                    log.flush()
                    detail = (work / "encode.log").read_text(encoding="utf-8", errors="replace")[-4000:]
                    if detail:
                        raise RuntimeError(f"Video export failed: {error}\n{detail}") from error
                raise
            finally:
                with suppress(OSError):
                    process.stdin.close()
        if not video.is_file() or not video.stat().st_size:
            raise RuntimeError("FFmpeg produced no video")
        manifest = {
            "format": "modal_gaussians.result_video", "version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "result": str(result.path), "modal_result_identity": result.manifest["modal_result_identity"],
            "coordinate_source": result.manifest["coordinate_source"],
            "view": view, "images": source,
            "video": {"file": "comparison.mp4", "layout": "original_top_reconstruction_bottom",
                      "fps_hz": fps, "frame_count": count, "shape_hw": [2 * height, width + width % 2],
                      "panel_shape_hw": [height, width], "right_padding_pixels": width % 2,
                      "codec": "libx264", "crf": 18, "pixel_format": "yuv420p", "audio": False},
        }
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"Video output appeared during export: {destination}")
        os.rename(work, destination)
    return destination / "comparison.mp4"
