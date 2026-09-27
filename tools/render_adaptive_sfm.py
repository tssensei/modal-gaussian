"""Render native Adaptive checkpoints on the immutable common SfM test cameras.

Run in the Adaptive Python environment. Float32 RGB is preserved for independent
metrics; PNGs are clipped previews. Frozen times are diagnostics, not synchronized
reconstructions or an imported static Gaussian scene.
"""
from __future__ import annotations

import argparse
from itertools import chain
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time


def validate_mapping(mapping: list[dict], split: dict) -> None:
    """Keep the model clock separate from the native loader's i/N clock."""
    if len(mapping) < 2:
        raise ValueError("At least two timed frames are required")
    expected = {key: [] for key in ("train_indices", "test_indices")}
    for index, row in enumerate(mapping):
        if row["index"] != index or row["name"] != f"{index:06d}.png":
            raise ValueError("Frame mapping order/name differs")
        if not math.isclose(row["normalized_time"], index / (len(mapping) - 1), abs_tol=1e-12, rel_tol=0):
            raise ValueError("Model time must be frame_index / (frame_count - 1)")
        if not math.isclose(row["adaptive_raw_time"], index / len(mapping), abs_tol=1e-12, rel_tol=0):
            raise ValueError("Native loader time differs")
        if row["split"] not in {"train", "test"}:
            raise ValueError("Unknown frame split")
        expected[row["split"] + "_indices"].append(index)
    if any(split[key] != values for key, values in expected.items()) or not expected["test_indices"]:
        raise ValueError("Frame mapping and test split differ")


def frozen_time(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Frozen time must be finite and in [0, 1]")
    return number


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import numpy as np
    from PIL import Image
    from modal_gaussians.common.cache import atomic_json, identity, publish_directory, sha256
    from modal_gaussians.common.scene_store import resolve_path

    root = resolve_path(args.input, strict=True)
    repo = resolve_path(args.adaptive_repo, strict=True)
    model_root = resolve_path(args.model, strict=True)
    output = resolve_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if any(output.is_relative_to(parent) for parent in (root, model_root, repo)):
        raise ValueError("Output must be outside immutable data/model/source directories")
    if len(set(args.frozen_times)) != len(args.frozen_times):
        raise ValueError("Frozen times must be unique")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["identity"] != identity({key: value for key, value in manifest.items() if key != "identity"}):
        raise ValueError("Input manifest identity differs")
    mapping = json.loads((root / "frame_mapping.json").read_text(encoding="utf-8"))
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    validate_mapping(mapping, split)
    if split != manifest["split"]:
        raise ValueError("Manifest and split differ")
    dataset = root / "dataset_adaptive"
    # read_scene_info would create this PLY if absent: never allow source writes.
    if not (dataset / "sparse/0/points3D.ply").is_file():
        raise ValueError("Phase 1 must publish points3D.ply before rendering")
    consumed = ["frame_mapping.json", "split.json"] + [
        name for name in manifest["files"] if name.startswith("dataset_adaptive/")
    ]
    for name in consumed:
        if sha256(root / name) != manifest["files"][name]:
            raise ValueError(f"Input file changed: {name}")
    checkpoint = model_root / "checkpoints" / args.checkpoint / "gaussian_model.pth"
    checkpoint_hash = sha256(checkpoint)
    training_args_hash = sha256(model_root / "train_args.txt")

    sys.path.insert(0, str(repo))
    import torch
    from dynamic_gaussians.checkpoint_loading import get_training_parameters, load_checkpoint
    from dynamic_gaussians.gaussian_renderer import render
    from dynamic_gaussians.scene import Scene

    parameters = get_training_parameters(str(model_root))
    data_parameters = parameters.dataset_parameters
    if (resolve_path(data_parameters.source_path, strict=True) != dataset
            or not data_parameters.eval or data_parameters.llffhold != split["llffhold"]
            or data_parameters.test_set_segment_length != split["segment_length"]
            or data_parameters.resolution_scale != 1 or data_parameters.images != "images"):
        raise ValueError("Training arguments do not match the Phase 1 dataset/split/resolution")
    data_parameters.data_device = "cpu"
    data_parameters.cache_images = False
    scene = Scene(data_parameters)
    cameras = sorted(chain.from_iterable(scene.test_cameras.values()), key=lambda camera: camera.image_name)
    if [camera.image_name for camera in cameras] != [f"{i:06d}" for i in split["test_indices"]]:
        raise ValueError("Native loader and common test-camera order differ")
    gaussian_model = load_checkpoint(str(model_root), args.checkpoint, parameters.optimization_parameters)
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - started
    width, height = manifest["derivation"]["resolution"]
    variants = [("dynamic", None)] + [("frozen_" + f"{value:g}".replace(".", "p"), value) for value in args.frozen_times]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    records = []
    try:
        with torch.no_grad():
            for variant, fixed_time in variants:
                (temporary / variant).mkdir()
                for camera in cameras:
                    index = int(camera.image_name)
                    row = mapping[index]
                    K = np.asarray(row["K"])
                    if (camera.width != width or camera.height != height
                            or not np.allclose(camera.world_view_matrix.cpu().numpy(), row["raw_world_to_camera"], rtol=0, atol=1e-6)
                            or not np.allclose([width / (2 * math.tan(camera.fov_x / 2)), height / (2 * math.tan(camera.fov_y / 2))], K[[0, 1], [0, 1]], rtol=0, atol=5e-5)
                            or not np.allclose(K[:2, 2], [width / 2, height / 2], rtol=0, atol=1e-8)
                            or not math.isclose(camera.time, row["adaptive_raw_time"], rel_tol=0, abs_tol=1e-12)):
                        raise ValueError(f"Native camera identity differs: {camera.image_name}")
                    query_time = row["normalized_time"] if fixed_time is None else fixed_time
                    torch.cuda.synchronize()
                    render_started = time.perf_counter()
                    point_cloud = gaussian_model.point_cloud_at_time(query_time)
                    color = render(camera, point_cloud, velocities=None, log_se3_translations=None,
                                   log_se3_rotations=None, pipeline_parameters=parameters.pipeline_parameters,
                                   bg_color=scene.background)["color"]
                    torch.cuda.synchronize()
                    render_seconds = time.perf_counter() - render_started
                    rgb = color.detach().permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32, copy=False)
                    if rgb.shape != (height, width, 3) or not np.isfinite(rgb).all():
                        raise ValueError(f"Invalid RGB at {variant}/{index}")
                    npy = f"{variant}/{index:06d}.npy"
                    png = f"{variant}/{index:06d}.png"
                    np.save(temporary / npy, rgb, allow_pickle=False)
                    Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(temporary / png)
                    records.append({
                        "variant": variant, "index": index, "name": row["name"], "split": "test",
                        "source_index": row["source_index"], "timestamp_seconds": row["timestamp_seconds"],
                        "normalized_time": row["normalized_time"], "render_time": query_time,
                        "render_effective_frame": round(query_time * (len(mapping) - 1)),
                        "loader_time": camera.time, "image_sha256": row["image_sha256"],
                        "camera_identity": identity({"K": row["K"], "raw_world_to_camera": row["raw_world_to_camera"], "resolution": [width, height]}),
                        "K": row["K"], "raw_world_to_camera": row["raw_world_to_camera"],
                        "npy": npy, "png": png, "npy_sha256": sha256(temporary / npy),
                        "png_sha256": sha256(temporary / png), "render_seconds": render_seconds,
                        "rgb_min": float(rgb.min()), "rgb_max": float(rgb.max()),
                    })
                print(f"Rendered {variant}: {len(cameras)} frames", flush=True)
        if sha256(checkpoint) != checkpoint_hash or sha256(model_root / "train_args.txt") != training_args_hash:
            raise ValueError("Model changed while rendering")
        for name in consumed:
            if sha256(root / name) != manifest["files"][name]:
                raise ValueError(f"Input changed while rendering: {name}")
        source_files = ["dynamic_gaussians/checkpoint_loading.py", "dynamic_gaussians/gaussian_renderer/__init__.py",
                        "dynamic_gaussians/scene/cameras.py", "dynamic_gaussians/scene/dataset_readers.py",
                        "dynamic_gaussians/models/keyframe_gaussian_model.py", "dynamic_gaussians/models/keyframe_gaussian_model_taichi.py"]
        report = {
            "format": "modal_gaussians.adaptive_sfm_renders", "version": 1, "method": "adaptive",
            "input": str(root), "data_identity": manifest["identity"], "model": str(model_root),
            "checkpoint": str(checkpoint), "checkpoint_name": args.checkpoint, "checkpoint_sha256": checkpoint_hash,
            "training_args_sha256": training_args_hash, "worker_sha256": sha256(Path(__file__)),
            "adaptive_source_sha256": {name: sha256(repo / name) for name in source_files},
            "gaussian_count": gaussian_model.number_of_gaussians, "test_indices": split["test_indices"],
            "background_rgb": scene.background.cpu().tolist(),
            "native_pipeline": {name: getattr(parameters.pipeline_parameters, name) for name in
                                ("convert_SHs_python", "compute_cov3D_python", "debug")},
            "sh_degree": point_cloud.sh_degree,
            "resolution": [width, height], "array_format": "float32 HWC RGB; unquantized, unclamped",
            "preview_format": "uint8 RGB; clamp [0,1], round 255*x",
            "time_convention": "dynamic = frame_index / (frame_count - 1); frozen variants use constant model time",
            "frozen_diagnostic_scope": "same held-out cameras, fixed model time; not synchronized reconstruction or static-scene import",
            "initialization_seconds": initialization_seconds,
            "render_seconds": sum(row["render_seconds"] for row in records),
            "end_to_end_seconds": time.perf_counter() - started,
            "frames": records,
            "files": {row[kind]: row[kind + "_sha256"] for row in records for kind in ("npy", "png")},
        }
        report["identity"] = identity(report)
        atomic_json(temporary / "render_manifest.json", report)
        publish_directory(temporary, output)
        return report
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Phase 1 data directory")
    parser.add_argument("--adaptive-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("time_300", "final"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-times", type=frozen_time, nargs="*", default=[])
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("identity", "checkpoint_name", "gaussian_count", "render_seconds")}, indent=2))


if __name__ == "__main__":
    main()
