"""SEA-RAFT reference-to-frame inference, separate from the shared FFT cache."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from modal_gaussians.scene_store import resolve_path
import sys
import time

import cv2
import numpy as np
import torch

from modal_gaussians.flow.storage import create_array
from modal_gaussians.iteration_cache import atomic_json
from modal_gaussians.modes import TRANSFORM_CONVENTION
from modal_gaussians.progress import Progress, report_progress

FORMAT = "modal_gaussians.sea_raft_flow"


def read_image(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float()


def load_model(repo, weights):
    repo, weights = resolve_path(repo), resolve_path(weights)
    if not (weights / "model.safetensors").is_file():
        raise FileNotFoundError(weights / "model.safetensors")
    config = argparse.Namespace(**json.loads(
        (repo / "config/eval/spring-M.json").read_text(encoding="utf-8")))
    sys.path.insert(0, str(repo / "core"))
    from raft import RAFT

    report_progress("Loading local SEA-RAFT M weights")
    return RAFT.from_pretrained(str(weights), args=config).cuda().eval()


def compute_flow(*, images, reuse_stabilization, output_dir, sea_raft_repo, model_dir, command=None):
    """Reuse timing/reference metadata and stabilized RGBs, never old flow or masks.

    The source is the existing geometry preparation's flow manifest. Its arrays
    are not read; this keeps the same camera/reference contract as saved prepared
    artifacts without running the legacy estimator or a frequency transform.
    """
    started = time.perf_counter()
    previous, images = resolve_path(reuse_stabilization), resolve_path(images)
    output = resolve_path(output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new flow output directory: {output}")
    source = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
    if source.get("format") != "modal_gaussians.flow_analysis" or source.get("version") not in (6, 7):
        raise ValueError("Expected an existing geometry preparation flow manifest")
    if images != resolve_path(source["inputs"]["sequence"]["image_directory"]):
        raise ValueError("Images differ from the recorded sequence")
    names, fps = source["frame_names"], float(source["fps_hz"])
    reference_index = source["reference_frame_index"]
    if (len(names) < 3 or not np.isfinite(fps) or fps <= 0
            or not isinstance(reference_index, int) or not 0 <= reference_index < len(names)
            or names[reference_index] != source["reference_frame_name"]
            or sorted(p.stem for p in images.glob("*.png")) != names):
        raise ValueError("Invalid sequence timing, reference, or frame names")
    image_dir = images
    stable_record = source.get("stabilized_sequence")
    if stable_record is not None:
        stable_root = (previous / stable_record["path"]).resolve()
        if not stable_root.is_relative_to(previous):
            raise ValueError("Stabilized sequence must be inside its source artifact")
        stable = json.loads((stable_root / "manifest.json").read_text(encoding="utf-8"))
        if (names != stable["frames"] or stable["reference_frame"] != names[reference_index]
                or float(stable["fps_hz"]) != fps):
            raise ValueError("Stabilized sequence metadata differs from its source")
        image_dir = stable_root / "images"
    reference_path = image_dir / (names[reference_index] + ".png")
    reference_cpu = read_image(reference_path)
    height, width = reference_cpu.shape[-2:]
    if [height, width] != source["arrays"]["flow"]["shape"][1:3]:
        raise ValueError("Reference dimensions differ from source metadata")
    if not torch.cuda.is_available():
        raise RuntimeError("SEA-RAFT inference requires CUDA")
    model = load_model(sea_raft_repo, model_dir)
    shape = (len(names), height, width, 2)
    manifest = {
        "format": FORMAT, "version": 1, "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "images": str(images), "stabilization_source": str(previous),
        "stabilized_images": str(image_dir) if stable_record is not None else None,
        "inference_images": str(image_dir), "reference_image": str(reference_path),
        "reference_frame_name": names[reference_index], "reference_frame_index": reference_index,
        "fps_hz": fps, "frames": names, "flow_file": "flow.zarr", "flow_shape": list(shape),
        "flow_dtype": "float32", "flow_units": "input_pixels", "flow_direction": "reference_to_frame",
        "smoothing": "none", "full_spectrum": False, "transform": TRANSFORM_CONVENTION,
        "completed_frames": 0, "validation": False,
        "model": {"weights": str(resolve_path(model_dir) / "model.safetensors"),
                  "repository": str(resolve_path(sea_raft_repo)),
                  "config": vars(model.args), "native_resolution": True},
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "manifest.json", manifest)
    flow = None
    try:
        reference = reference_cpu.cuda()
        flow = create_array(output / "flow.zarr", shape, np.float32)
        batch_size = int(flow.shards[0])
        progress = Progress("SEA-RAFT flow", len(names), unit="frames")
        with torch.inference_mode():
            for start in range(0, len(names), batch_size):
                stop = min(start + batch_size, len(names))
                batch = np.empty((stop - start, height, width, 2), dtype=np.float32)
                for index in range(start, stop):
                    if index == reference_index:
                        frame_flow = np.zeros((height, width, 2), dtype=np.float32)
                    else:
                        current = read_image(image_dir / (names[index] + ".png"))
                        if current.shape != reference_cpu.shape:
                            raise ValueError(f"Frame dimensions changed: {names[index]}")
                        result = model(reference, current.cuda(), iters=model.args.iters, test_mode=True)
                        frame_flow = result["final"][0].permute(1, 2, 0).float().cpu().numpy()
                        del result, current
                    if frame_flow.shape != (height, width, 2) or not np.isfinite(frame_flow).all():
                        raise RuntimeError(f"Invalid model output: {names[index]}")
                    batch[index - start] = frame_flow
                    progress.update(index + 1)
                flow[start:stop] = batch
                del batch
                manifest["completed_frames"] = stop
                atomic_json(output / "manifest.json", manifest)
        flow.store.close()
        flow = None
        manifest.update(status="complete", elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / "manifest.json", manifest)
        report_progress(f"SEA-RAFT flow saved: {output} ({manifest['elapsed_seconds']:.1f}s)")
        return output
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        atomic_json(output / "manifest.json", manifest)
        raise
    finally:
        if flow is not None:
            flow.store.close()
