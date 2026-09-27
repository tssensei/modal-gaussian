"""Check a common SfM dataset through both projects' real CPU data loaders.

Run with the Modal Python; --adaptive-python selects the independent Python 3.12
environment. This loads inputs and checks projections, but never trains a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


def array_hash(value: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def adaptive_probe(dataset: Path, repo: Path, output: Path) -> None:
    sys.path.insert(0, str(repo))
    from dynamic_gaussians.scene.dataset_readers import read_scene_info

    if not (dataset / "sparse/0/points3D.ply").is_file():
        raise ValueError("Publish points3D.ply first; validation must not modify inputs")
    scene = read_scene_info(SimpleNamespace(
        source_path=str(dataset), images="images", eval=True, llffhold=8,
        test_set_segment_length=4, resolution_scale=1.0, data_device="cpu",
        cache_images=False,
    ))
    if scene.point_cloud is None or len(scene.point_cloud.positions) == 0:
        raise ValueError("Adaptive loader did not load the SfM point cloud")
    records = []
    for split, groups in (("train", scene.train_cameras), ("test", scene.test_cameras)):
        for cameras in groups.values():
            for camera in cameras:
                image = camera.image.cpu().numpy().transpose(1, 2, 0)
                if image.shape != (camera.height, camera.width, 3):
                    raise ValueError(f"Unexpected RGB shape: {camera.image_name}")
                if not np.isfinite(image).all():
                    raise ValueError(f"Nonfinite RGB: {camera.image_name}")
                records.append({
                    "name": camera.image_name, "split": split, "time": camera.time,
                    "width": camera.width, "height": camera.height,
                    "focal_x": camera.width / (2 * np.tan(camera.fov_x / 2)),
                    "focal_y": camera.height / (2 * np.tan(camera.fov_y / 2)),
                    "world_to_camera": camera.world_view_matrix.tolist(),
                    "projection_matrix": camera.projection_matrix.tolist(),
                    "rgb_uint8_sha256": array_hash(np.rint(image * 255), "u1"),
                })
    output.write_text(json.dumps({
        "cameras": sorted(records, key=lambda row: row["name"]),
        "point_count": len(scene.point_cloud.positions),
        "points_f32_sha256": array_hash(scene.point_cloud.positions.cpu().numpy(), "<f4"),
        "colors_u8_sha256": array_hash(np.rint(scene.point_cloud.colors.cpu().numpy() * 255), "u1"),
        "camera_extent": float(scene.nerf_normalization["radius"]),
    }, indent=2) + "\n", encoding="utf-8")


def verify(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import cv2
    from modal_gaussians.common.cache import identity, sha256
    from modal_gaussians.common.scene_store import resolve_path
    from modal_gaussians.geometry.scene import load_static_dataset, _read_points3d_binary

    root = resolve_path(args.input, strict=True)
    source = resolve_path(args.source, strict=True)
    repo = resolve_path(args.adaptive_repo, strict=True)
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    probe = output / "adaptive_loader.json"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with (output / "adaptive_loader.log").open("w", encoding="utf-8") as log:
        subprocess.run([
            str(args.adaptive_python), str(Path(__file__).resolve()),
            "--adaptive-probe", "--input", str(root / "dataset_adaptive"),
            "--adaptive-repo", str(repo), "--output", str(probe),
        ], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    adaptive = json.loads(probe.read_text(encoding="utf-8"))
    static = load_static_dataset(root / "dataset_static_train")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "frame_mapping.json").read_text(encoding="utf-8"))
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    source_payload = json.loads((source / "cameras.json").read_text(encoding="utf-8"))
    source_frames = sorted(source_payload["frames"], key=lambda row: row["timestamp_seconds"])
    points, colors = _read_points3d_binary(source / "sparse/0/points3D.bin")
    checks: dict[str, bool] = {}

    def check(name: str, passed: bool) -> None:
        checks[name] = bool(passed)
        if not passed:
            raise ValueError(f"Dataset verification failed: {name}")

    check("manifest_identity", manifest["identity"] == identity({k: v for k, v in manifest.items() if k != "identity"}))
    check("published_files_unchanged", all(sha256(root / name) == digest for name, digest in manifest["files"].items()))
    check("mapping_identity", sha256(root / "frame_mapping.json") == manifest["files"]["frame_mapping.json"])
    check("split_identity", sha256(root / "split.json") == manifest["files"]["split.json"])
    check("source_root", resolve_path(manifest["source"]["path"], strict=True) == source)
    check("source_files_unchanged", all(sha256(source / name) == digest for name, digest in manifest["source"]["files"].items()))
    check("source_points_preserved_static", np.array_equal(points, static.raw_points))
    check("source_colors_preserved_static", np.array_equal(colors, static.point_colors))
    check("source_points_preserved_adaptive", array_hash(points, "<f4") == adaptive["points_f32_sha256"])
    check("source_colors_preserved_adaptive", array_hash(np.rint(colors * 255), "u1") == adaptive["colors_u8_sha256"])
    check("matching_point_count", adaptive["point_count"] == len(points))
    cameras = adaptive["cameras"]
    check("all_source_sweep_frames_present", len(cameras) == len(source_frames))
    check("all_mapping_frames_present", len(mapping) == len(cameras))
    check("nonempty_time_interval", len(cameras) >= 2)
    # Independently enforce the requested native split, not just agreement
    # between two exports that could share the same mistaken frame selection.
    check("native_segment4_split", all(
        row["split"] == ("test" if index // 4 % 8 == 7 and len(cameras) - index > 3 else "train")
        for index, row in enumerate(cameras)
    ))
    check("split_metadata", (
        split["llffhold"] == 8 and split["segment_length"] == 4 and split["final_three_train"] is True
        and split["train_indices"] == [i for i, row in enumerate(cameras) if row["split"] == "train"]
        and split["test_indices"] == [i for i, row in enumerate(cameras) if row["split"] == "test"]
        and split == manifest["split"]
    ))
    train_names = {row["name"] for row in cameras if row["split"] == "train"}
    static_by_name = {Path(camera.name).stem: camera for camera in static.cameras}
    check("matching_train_domain", train_names == set(static_by_name))
    check("references_excluded", all(camera.role == "sweep" for camera in static.cameras))
    time_values = np.array([row["timestamp_seconds"] for row in source_frames])
    check("finite_source_times", np.isfinite(time_values).all())
    check("real_uniform_30_fps", np.allclose(np.diff(time_values), 1 / 30, rtol=0, atol=1e-8))
    check("adaptive_time_matches_source", np.allclose(
        [row["time"] for row in cameras],
        (time_values - time_values[0]) / (time_values[-1] - time_values[0]) * (len(cameras) - 1) / len(cameras),
        rtol=0, atol=1e-8,
    ))
    max_projection_error = 0.0
    max_normalization_error = 0.0
    for index, (camera, source_camera) in enumerate(zip(cameras, source_frames)):
        mapped = mapping[index]
        check(f"name_{index}", camera["name"] == f"{index:06d}")
        check(f"mapping_{index}", (
            mapped["index"] == index and mapped["name"] == camera["name"] + ".png"
            and mapped["split"] == camera["split"]
            and mapped["source_index"] == source_camera["source_index"]
            and mapped["source_frame_name"] == source_camera["source_frame_name"]
            and mapped["source_image_name"] == source_camera["image_name"]
            and mapped["image_id"] == source_camera["image_id"]
            and mapped["source_camera_id"] == source_camera["camera_id"]
            and mapped["timestamp_seconds"] == source_camera["timestamp_seconds"]
            and mapped["adaptive_raw_time"] == camera["time"]
            and np.isclose(mapped["normalized_time"], index / (len(cameras) - 1), rtol=0, atol=1e-12)
            and mapped["source_K"] == source_camera["K"]
            and mapped["source_camera_parameters"] == source_camera["camera_parameters"]
            and mapped["raw_world_to_camera"] == source_camera["world_to_camera"]
        ))
        check(f"source_image_identity_{index}", mapped["source_image_sha256"] == manifest["source"]["files"]["images/" + source_camera["image_name"]])
        check(f"source_mask_identity_{index}", mapped["source_mask_sha256"] == manifest["source"]["files"]["masks/" + source_camera["image_name"]])
        rgb_path = root / "dataset_adaptive/images" / mapped["name"]
        check(f"output_image_identity_{index}", sha256(rgb_path) == mapped["image_sha256"] == manifest["files"]["dataset_adaptive/images/" + mapped["name"]])
        decoded = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        check(f"output_rgb_format_{index}", decoded is not None and decoded.dtype == np.uint8 and decoded.shape == (camera["height"], camera["width"], 3))
        check(f"adaptive_rgb_{index}", array_hash(decoded[..., ::-1], "u1") == camera["rgb_uint8_sha256"])
        mapped_K = np.asarray(mapped["K"])
        check(f"mapping_intrinsics_{index}", np.allclose(mapped_K, [
            [camera["focal_x"], 0, camera["width"] / 2],
            [0, camera["focal_y"], camera["height"] / 2], [0, 0, 1]], rtol=0, atol=5e-5))
        raw_pose = np.array(source_camera["world_to_camera"])
        pose = np.array(camera["world_to_camera"])
        check(f"source_pose_{index}", np.allclose(raw_pose, pose, rtol=0, atol=1e-6))
        if camera["name"] not in static_by_name:
            continue
        ours = static_by_name[camera["name"]]
        check(f"static_image_identity_{index}", ours.image_sha256 == mapped["image_sha256"])
        check(f"static_mask_identity_{index}", ours.mask_sha256 == mapped["mask_sha256"])
        K = ours.K.numpy().astype(np.float64)
        check(f"camera_{index}", (
            ours.width == camera["width"] and ours.height == camera["height"]
            and np.allclose(K[[0, 1], [0, 1]], [camera["focal_x"], camera["focal_y"]], rtol=0, atol=5e-5)
            and np.allclose(K[:2, 2], [ours.width / 2, ours.height / 2], rtol=0, atol=1e-8)
            and ours.camera_parameters[3] == 0
            and np.allclose(ours.raw_world_to_camera.numpy(), pose, rtol=0, atol=1e-6)
        ))
        rgb = static.load_rgb(ours, as_uint8=True).numpy()
        check(f"rgb_{index}", array_hash(rgb, "u1") == camera["rgb_uint8_sha256"])
        # Compare actual Adaptive projection with Modal's half-pixel camera domain.
        camera_xyz = points[::97].astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]
        camera_xyz = camera_xyz[camera_xyz[:, 2] > 0.1]
        uv = camera_xyz[:, :2] / camera_xyz[:, 2:]
        uv = uv * K[[0, 1], [0, 1]] + K[:2, 2]
        in_frame = ((uv > 0) & (uv < [ours.width, ours.height])).all(axis=1)
        camera_xyz, uv = camera_xyz[in_frame], uv[in_frame]
        check(f"visible_sparse_points_{index}", len(uv) > 0)
        clip = np.column_stack([camera_xyz, np.ones(len(camera_xyz))]) @ np.array(camera["projection_matrix"]).T
        ndc = clip[:, :2] / clip[:, 3:]
        adaptive_uv = ((ndc + 1) * [ours.width, ours.height] - 1) / 2
        max_projection_error = max(max_projection_error, float(np.max(np.abs(uv - 0.5 - adaptive_uv))))
        xyz = points[::97].astype(np.float64)
        normalized = static.normalization.normalize_points(xyz)
        normalized_camera = normalized @ ours.world_to_camera.numpy()[:3, :3].T + ours.world_to_camera.numpy()[:3, 3]
        raw_camera = xyz @ ours.raw_world_to_camera.numpy()[:3, :3].T + ours.raw_world_to_camera.numpy()[:3, 3]
        max_normalization_error = max(max_normalization_error, float(np.max(np.abs(normalized_camera - raw_camera / static.normalization.scale))))
    check("cross_renderer_projection", max_projection_error < 1e-3)
    check("static_normalization", max_normalization_error < 1e-4)
    report = {
        "format": "modal_gaussians.adaptive_sfm_loader_validation", "version": 1,
        "status": "passed", "source": str(source), "dataset": str(root),
        "static_dataset_identity": static.dataset_identity,
        "frame_count": len(cameras), "train_count": len(train_names),
        "test_count": len(cameras) - len(train_names), "point_count": len(points),
        "max_projection_error_pixels": max_projection_error,
        "max_normalized_camera_coordinate_error": max_normalization_error,
        "checks_passed": len(checks), "static_normalization": static.normalization.to_dict(),
        "adaptive_camera_extent": adaptive["camera_extent"],
        "time_conventions": {"loader": "frame_index / frame_count",
                             "model": "frame_index / (frame_count - 1)",
                             "physical_frame_interval_seconds": 1 / 30},
        "scope": "CPU input loading and camera/point/RGB equivalence; no training or quality evaluation",
        "input_identity": manifest["identity"],
        "verifier_sha256": sha256(Path(__file__)),
        "adaptive_loader_sha256": sha256(repo / "dynamic_gaussians/scene/dataset_readers.py"),
        "adaptive_camera_sha256": sha256(repo / "dynamic_gaussians/scene/cameras.py"),
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--adaptive-repo", type=Path, required=True)
    parser.add_argument("--adaptive-python", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adaptive-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.adaptive_probe:
        adaptive_probe(args.input, args.adaptive_repo, args.output)
    else:
        if args.source is None or args.adaptive_python is None:
            parser.error("--source and --adaptive-python are required")
        print(json.dumps(verify(args), indent=2))


if __name__ == "__main__":
    main()
