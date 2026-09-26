"""Posed DA3 depth preparation and immutable targets for static RGB-D training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import torch

from modal_gaussians.common.cache import atomic_json, identity, publish_directory, sha256
from modal_gaussians.common.camera_geometry import distort_normalized, undistort_normalized
from modal_gaussians.common.progress import report_progress
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.geometry.scene import Camera, StaticDataset, load_static_dataset


@dataclass(frozen=True)
class DepthConfig:
    process_res: int = 1008
    chunk_size: int = 16
    confidence_percentile: float = 10.0

    def validate(self):
        if (not isinstance(self.process_res, int) or not isinstance(self.chunk_size, int)
                or self.process_res < 14 or self.chunk_size < 3):
            raise ValueError("Depth process_res >= 14 and chunk_size >= 3 are required")
        if not 0 <= self.confidence_percentile < 100:
            raise ValueError("Depth confidence percentile must be in [0,100)")


def _pixel_rays(camera: Camera) -> np.ndarray:
    y, x = np.mgrid[:camera.height, :camera.width]
    K = camera.K.cpu().numpy()
    return (np.stack([x, y], -1) + .5 - K[:2, 2]) / np.diag(K)[:2]


def undistort_input(rgb: np.ndarray, camera: Camera):
    """Use COLMAP half-pixel centers, but expose OpenCV integer centers to DA3."""
    K = camera.K.cpu().numpy().astype(np.float64)
    pixels = distort_normalized(_pixel_rays(camera), camera.radial_distortion) * np.diag(K)[:2] + K[:2, 2] - .5
    x, y = pixels[..., 0].astype(np.float32), pixels[..., 1].astype(np.float32)
    valid = (x >= 0) & (x <= camera.width-1) & (y >= 0) & (y <= camera.height-1)
    image = cv2.remap(rgb, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    K[:2, 2] -= .5
    return image, valid, K


def restore_depth_grid(depth, confidence, processed_K, camera, input_valid, percentile):
    """Map camera-Z depth from the returned pinhole grid to original distorted PNGs."""
    depth, confidence, K = np.asarray(depth), np.asarray(confidence), np.asarray(processed_K)
    if (depth.ndim != 2 or confidence.shape != depth.shape or K.shape != (3, 3)
            or not np.isfinite(K).all() or np.any(np.diag(K)[:2] <= 0)
            or not np.allclose(K[2], [0, 0, 1]) or not np.allclose(K[[0, 1], [1, 0]], 0)):
        raise ValueError("Invalid DA3 depth/confidence/intrinsics")
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 0)
    original_K = camera.K.cpu().numpy()
    py, px = np.mgrid[:depth.shape[0], :depth.shape[1]]
    processed_rays = (np.stack([px, py], -1) - K[:2, 2]) / np.diag(K)[:2]
    source_pixels = processed_rays * np.diag(original_K)[:2] + original_K[:2, 2] - .5
    processed_support = cv2.remap(input_valid.astype(np.float32), source_pixels[..., 0].astype(np.float32),
        source_pixels[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid &= processed_support >= 1 - 1e-6
    if not valid.any():
        raise ValueError(f"No valid DA3 depth for {camera.name}")
    threshold = float(np.percentile(confidence[valid], percentile))
    valid &= confidence >= threshold
    rays = undistort_normalized(_pixel_rays(camera), camera.radial_distortion)
    pixels = rays * np.diag(K)[:2] + K[:2, 2]
    x, y = pixels[..., 0].astype(np.float32), pixels[..., 1].astype(np.float32)
    remap = lambda array: cv2.remap(array.astype(np.float32), x, y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
    supported = remap(valid) >= 1 - 1e-6
    # Exclude every bilinear footprint touching undistortion padding.
    pinhole_pixels = rays * np.diag(original_K)[:2] + original_K[:2, 2] - .5
    supported &= cv2.remap(input_valid.astype(np.float32),
        pinhole_pixels[..., 0].astype(np.float32), pinhole_pixels[..., 1].astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) >= 1 - 1e-6
    target = remap(np.where(valid, depth, 0))
    conf = remap(np.where(valid, confidence, 0))
    supported &= np.isfinite(target) & (target > 0) & np.isfinite(conf) & (conf > 0)
    if not supported.any():
        raise ValueError(f"No supported DA3 pixels on training grid: {camera.name}")
    return np.where(supported, target, 0).astype(np.float32), np.where(supported, conf, 0).astype(np.float32), supported, threshold


def inference_groups(cameras, chunk_size):
    """Bound memory; include three spatially spread cameras in every chunk."""
    centers = np.stack([np.linalg.inv(c.raw_world_to_camera.cpu().numpy())[:3, 3] for c in cameras])
    if len(centers) < 3 or not np.isfinite(centers).all():
        raise ValueError("Posed DA3 requires at least three valid COLMAP cameras")
    anchors = [0]
    for _ in range(2):
        distance = np.min(np.linalg.norm(centers[:, None] - centers[anchors], axis=-1), axis=1)
        if distance.max() <= 1e-6:
            raise ValueError("DA3 scale alignment requires three distinct camera centers")
        anchors.append(int(distance.argmax()))
    groups = []
    # chunk_size is the number of target frames; anchors add at most three views.
    for start in range(0, len(cameras), chunk_size):
        targets = list(range(start, min(start+chunk_size, len(cameras))))
        groups.append({"targets": targets, "context": sorted(set(targets + anchors))})
    return groups


def model_files(model: Path):
    files = sorted(p for p in model.rglob("*") if p.is_file() and ".cache" not in p.relative_to(model).parts)
    if not (model / "config.json").is_file() or not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("--model must be a local DA3 pretrained snapshot with config.json and safetensors")
    return {p.relative_to(model).as_posix(): sha256(p) for p in files}


def prepare_depth(*, input_dir, output_dir, model_dir, python, config=DepthConfig()):
    """Run inference in an isolated DA3 Python, then publish validated native-grid targets."""
    config.validate()
    dataset = load_static_dataset(input_dir)
    model = resolve_path(model_dir, strict=True)
    weights = model_files(model)
    groups = inference_groups(dataset.cameras, config.chunk_size)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        scratch = temporary / "inference"
        scratch.mkdir()
        inputs = []
        for index, camera in enumerate(dataset.cameras):
            rgb = dataset.load_rgb(camera, as_uint8=True).numpy()
            image, _, K = undistort_input(rgb, camera)
            path = scratch / f"{index:06d}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise OSError(path)
            inputs.append({"image": str(path), "K": K.tolist(),
                           "world_to_camera": camera.raw_world_to_camera.tolist()})
        request = {"model": str(model), "inputs": inputs, "groups": groups,
                   "process_res": config.process_res, "output": str(scratch)}
        atomic_json(scratch / "request.json", request)
        worker = Path(__file__).with_name("_da3_worker.py")
        report_progress(f"depth: DA3 on {len(inputs)} COLMAP images in {len(groups)} groups")
        with (temporary / "inference.log").open("w", encoding="utf-8") as log:
            try:
                subprocess.run([str(python), str(worker), str(scratch / "request.json")],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                log.flush()
                raise RuntimeError("DA3 failed; no depth artifact published:\n" +
                    (temporary / "inference.log").read_text(encoding="utf-8", errors="replace")[-8000:]) from error
        records = []
        for index, camera in enumerate(dataset.cameras):
            _, input_valid, _ = undistort_input(np.zeros((camera.height, camera.width, 3), np.uint8), camera)
            with np.load(scratch / f"{index:06d}.npz", allow_pickle=False) as raw:
                expected_pose = camera.raw_world_to_camera.numpy()[:3]
                if not np.allclose(raw["extrinsics"], expected_pose, atol=1e-5, rtol=1e-5):
                    raise ValueError(f"DA3 changed supplied camera: {camera.name}")
                depth, confidence, valid, threshold = restore_depth_grid(raw["depth"], raw["confidence"],
                    raw["K"], camera, input_valid, config.confidence_percentile)
                processed_K = raw["K"].tolist()
                processed_shape = list(raw["depth"].shape)
            # DA3 has already aligned to RAW input translation scale. Convert exactly once.
            depth /= dataset.normalization.scale
            if not np.isfinite(depth).all():
                raise ValueError(f"Non-finite normalized depth: {camera.name}")
            file = temporary / f"{index:06d}.npz"
            np.savez_compressed(file, depth=depth, confidence=confidence, valid=valid)
            records.append({"camera": camera.to_manifest_record(), "file": file.name,
                "sha256": sha256(file), "processed_K": processed_K, "processed_shape": processed_shape,
                "confidence_threshold": threshold, "valid_fraction": float(valid.mean())})
        if weights != model_files(model):
            raise ValueError("DA3 model snapshot changed during inference")
        if load_static_dataset(input_dir).dataset_identity != dataset.dataset_identity:
            raise ValueError("COLMAP inputs changed during depth preparation")
        runtime = json.loads((scratch / "runtime.json").read_text(encoding="utf-8"))
        manifest = {"format": "modal_gaussians.static_depth", "version": 1,
            "dataset_identity": dataset.dataset_identity, "input_root": str(dataset.root),
            "normalization": dataset.normalization.to_dict(), "depth_convention": "camera_z_normalized_world",
            "config": asdict(config), "model_files": weights, "model_path": str(model),
            "runtime": runtime, "groups": groups, "records": records,
            "implementation": {"depth.py": sha256(Path(__file__)), "worker": sha256(worker),
                "scene.py": sha256(Path(__file__).with_name("scene.py")),
                "camera_geometry.py": sha256(Path(__file__).parents[1] / "common" / "camera_geometry.py")},
            "preparation_runtime": {"numpy": np.__version__, "opencv": cv2.__version__, "torch": torch.__version__}}
        manifest["depth_identity"] = identity(manifest)
        shutil.rmtree(scratch)
        atomic_json(temporary / "manifest.json", manifest)
        DepthTargets(temporary, dataset)
        publish_directory(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


class DepthTargets:
    """Validate once; keep only the current training batch's depth maps in memory."""
    def __init__(self, root, dataset: StaticDataset):
        self.root = resolve_path(root, strict=True)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.identity = manifest.pop("depth_identity", None)
        if (manifest.get("format") != "modal_gaussians.static_depth" or manifest.get("version") != 1
                or identity(manifest) != self.identity):
            raise ValueError("Invalid static depth manifest identity/version")
        if (manifest["dataset_identity"] != dataset.dataset_identity
                or manifest["normalization"] != dataset.normalization.to_dict()
                or manifest["depth_convention"] != "camera_z_normalized_world"):
            raise ValueError("Static depth dataset, normalization or depth convention mismatch")
        expected = {c.name: c.to_manifest_record() for c in dataset.cameras}
        self.records = {}
        for record in manifest["records"]:
            name = record["camera"]["name"]
            if name in self.records or record["camera"] != expected.get(name):
                raise ValueError(f"Static depth camera binding mismatch: {name}")
            file = self.root / record["file"]
            if file.resolve().parent != self.root or sha256(file) != record["sha256"]:
                raise ValueError(f"Static depth file identity mismatch: {name}")
            self.records[name] = record
            self.load(name)
        if set(self.records) != set(expected):
            raise ValueError("Static depth must cover every training camera")

    def load(self, name):
        record = self.records[name]
        with np.load(self.root / record["file"], allow_pickle=False) as data:
            depth, valid, confidence = data["depth"], data["valid"], data["confidence"]
        shape = (record["camera"]["height"], record["camera"]["width"])
        if (depth.shape != shape or valid.shape != shape or confidence.shape != shape
                or depth.dtype != np.float32 or valid.dtype != np.bool_ or confidence.dtype != np.float32
                or not np.isfinite(depth).all() or not np.isfinite(confidence).all()
                or not valid.any() or np.any(depth[valid] <= 0) or np.any(confidence[valid] <= 0)
                or np.any(depth[~valid] != 0) or np.any(confidence[~valid] != 0)):
            raise ValueError(f"Invalid static depth arrays: {name}")
        return torch.from_numpy(depth), torch.from_numpy(valid)


def depth_l2(prediction, target, valid):
    """Ambient-style L2, averaged within each image then equally across images."""
    if prediction.shape != target.shape or valid.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Depth loss requires matching [B,H,W] tensors")
    count = valid.sum(dim=(-2, -1))
    if bool((count == 0).any()):
        raise ValueError("Depth loss has an empty supervised image")
    residual = torch.where(valid, prediction - target, 0)
    return (residual.square().sum(dim=(-2, -1)) / count).mean()
