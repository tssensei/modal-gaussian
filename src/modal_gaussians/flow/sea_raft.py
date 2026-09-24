"""SEA-RAFT reference-to-frame inference, separate from the shared FFT cache."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from modal_gaussians.common.scene_store import resolve_path
import sys
import time

import cv2
import numpy as np
import torch

from modal_gaussians.flow.storage import create_array
from modal_gaussians.common.cache import atomic_json
from modal_gaussians.spectrum.modes import TRANSFORM_CONVENTION
from modal_gaussians.common.progress import Progress, report_progress

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


def compute_flow(*, images, reference, output_dir, sea_raft_repo, model_dir, command=None,
                 reference_selection):
    """Compute flow using the selected reference on the prepared pixel grid."""
    started = time.perf_counter()
    from modal_gaussians.flow.reference_selection import sequence_metadata, reference_binding, motion_reference

    output = resolve_path(output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new flow output directory: {output}")
    seq = sequence_metadata(reference, images)
    source, previous, images = seq["source"], seq["root"], seq["images"]
    names, fps, reference_index = seq["names"], seq["fps"], seq["reference"]
    image_dir = seq["image_dir"]
    stable_record = source.get("stabilized_sequence")
    binding = None
    selection_path = resolve_path(reference_selection, strict=True)
    selected = json.loads((selection_path / "manifest.json").read_text(encoding="utf-8"))
    binding = reference_binding(selected)
    contract = binding["contract"]
    _, reference_index = motion_reference({
        "reference_frame_name": contract["reference_frame_name"],
        "reference_frame_index": contract["reference_frame_index"],
        "reference_selection": binding}, source)
    reference_path = image_dir / (names[reference_index] + ".png")
    reference_cpu = read_image(reference_path)
    height, width = reference_cpu.shape[-2:]
    if [height, width] != source["shape_hw"]:
        raise ValueError("Reference dimensions differ from source metadata")
    if not torch.cuda.is_available():
        raise RuntimeError("SEA-RAFT inference requires CUDA")
    model = load_model(sea_raft_repo, model_dir)
    shape = (len(names), height, width, 2)
    manifest = {
        "format": FORMAT, "version": 2, "status": "running",
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
    manifest["reference_selection"] = binding
    manifest["reference_selection_path"] = str(selection_path)
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
