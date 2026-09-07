"""Post-training mask classification using full-scene visibility contributions.

No Gaussian is deleted or optimized. Only confidently supported subject points
enter the derived scene's foreground; background AND uncertain points remain
static. Evidence uses fixed-geometry feature gradients, including depth-ordered
transmittance and the same distortion warp as RGB rendering.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from modal_gaussians.camera_geometry import PROJECTION_CONVENTION
from modal_gaussians.progress import Progress
from modal_gaussians.static import (
    GAUSSIAN_FIELDS, Camera, ForegroundBackgroundScene, _sha256_file, _sha256_json,
    cameras_from_scene_manifest, load_static_scene, tensor_dictionary_identity,
)

UNCERTAIN, SUBJECT, BACKGROUND = 0, 1, 2
LABELS = {"0": "uncertain", "1": "subject", "2": "background"}
REASONS = {"0": "resolved", "1": "insufficient_visibility",
           "2": "mixed_mask_evidence", "3": "conflicting_views"}
LEGACY_METHOD = "full_scene_visible_mask_contribution_v1"
METHOD = "full_scene_visible_mask_contribution_v2"


@dataclass(frozen=True)
class PartitionConfig:
    mask_dilation_pixels: int = 10
    minimum_visible_mass: float = 0.5
    minimum_visible_groups: int = 2
    class_fraction: float = 0.8
    view_angle_degrees: float = 10.0
    view_position_fraction: float = 0.05

    def validate(self) -> None:
        for name, minimum in (("mask_dilation_pixels", 0), ("minimum_visible_groups", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("minimum_visible_mass", "view_angle_degrees", "view_position_fraction"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.class_fraction) or not 0.5 < self.class_fraction <= 1:
            raise ValueError("class_fraction must be in (0.5,1]")
        if self.view_angle_degrees > 180 or self.view_position_fraction > 1:
            raise ValueError("View grouping requires angle <=180 and position fraction <=1")


def dilated_mask_regions(mask: np.ndarray, dilation: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Dilate the foreground once; its exact complement is the background.

    The square kernel extends the mask by ``dilation`` original-image pixels
    along each axis. There is no erosion or ignored silhouette/image-border band.
    """
    values = np.asarray(mask)
    if values.dtype != np.bool_ or values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("mask must be a nonempty two-dimensional bool array")
    if isinstance(dilation, bool) or not isinstance(dilation, int) or dilation < 0:
        raise ValueError("mask dilation must be a nonnegative integer")
    kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)
    foreground = cv2.dilate(values.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                           borderValue=0).astype(bool)
    return foreground, ~foreground


def group_camera_views(cameras: Sequence[Camera], config: PartitionConfig) -> list[list[int]]:
    """Deterministic pose grouping; repeated nearby frames are one evidence group.

    A view joins the first representative close in BOTH center and optical axis.
    Distances use the camera trajectory's robust extent, not the contaminated FG.
    Every camera is still rendered, and any strong view conflict is retained.
    """
    config.validate()
    if not cameras:
        raise ValueError("Repartition requires at least one camera")
    poses = np.stack([c.world_to_camera.detach().cpu().numpy() for c in cameras]).astype(np.float64)
    inverse = np.linalg.inv(poses)
    centers, axes = inverse[:, :3, 3], inverse[:, :3, 2]
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    extent = float(np.linalg.norm(np.quantile(centers, .99, axis=0) - np.quantile(centers, .01, axis=0)))
    distance_limit = max(extent * config.view_position_fraction, 1e-9)
    cosine_limit = math.cos(math.radians(config.view_angle_degrees))
    groups: list[list[int]] = []
    for index in range(len(cameras)):
        for group in groups:
            representative = group[0]
            if (np.linalg.norm(centers[index] - centers[representative]) <= distance_limit
                    and np.dot(axes[index], axes[representative]) >= cosine_limit - 1e-12):
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


def visible_mask_mass(scene: ForegroundBackgroundScene, camera: Camera,
                      foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Return [all_G,2] sums of T*alpha in supplied FG/BG pixels.

    Two dummy feature channels obtain both adjoints in one backward pass. No
    per-Gaussian one-hot images, nearest-depth approximation, or alpha division.
    Full-scene occluders remain present regardless of their old semantic label.
    """
    if foreground.shape != (camera.height, camera.width) or background.shape != foreground.shape:
        raise ValueError("Mask dimensions must match the original calibrated camera")
    if foreground.dtype != np.bool_ or background.dtype != np.bool_ or np.any(foreground & background):
        raise ValueError("Foreground/background regions must be disjoint bool masks")
    parameters = list(scene.parameters())
    requires_grad = [p.requires_grad for p in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        device = scene.foreground.params["means"].device
        with torch.enable_grad():
            dummy = torch.zeros((scene.count, 2), device=device, dtype=torch.float32, requires_grad=True)
            features, _ = scene.render_features(camera, dummy, composition="all")
            regions = torch.as_tensor(np.stack((foreground, background), axis=-1),
                                      device=device, dtype=features.dtype)
            mass, = torch.autograd.grad((features * regions).sum(), dummy)
        result = mass.detach().cpu().numpy().astype(np.float64)
        if result.shape != (scene.count, 2) or not np.isfinite(result).all() or np.any(result < -1e-6):
            raise FloatingPointError("Invalid full-scene mask contribution")
        return np.maximum(result, 0)
    finally:
        for parameter, flag in zip(parameters, requires_grad):
            parameter.requires_grad_(flag)


class EvidenceAccumulator:
    """Streaming O(G) evidence; memory does not grow with sweep frame count."""

    def __init__(self, count: int, config: PartitionConfig):
        config.validate()
        if count < 1:
            raise ValueError("Evidence requires at least one Gaussian")
        self.config = config
        self.arrays = {name: np.zeros(count, dtype=dtype) for name, dtype in {
            "foreground_mass": np.float64, "background_mass": np.float64,
            "group_fraction_sum": np.float64, "visible_group_count": np.int32,
            "visible_view_count": np.int32, "foreground_view_count": np.int32,
            "background_view_count": np.int32, "mixed_view_count": np.int32,
        }.items()}
        self.start_group()

    def start_group(self) -> None:
        count = len(self.arrays["foreground_mass"])
        self.group_sum = np.zeros(count, np.float64)
        self.group_count = np.zeros(count, np.int32)

    def add_view(self, mass: np.ndarray) -> None:
        values = np.asarray(mass, dtype=np.float64)
        if values.shape != (len(self.group_sum), 2) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("View mass must be finite nonnegative [G,2]")
        total = values.sum(axis=1)
        visible = total >= self.config.minimum_visible_mass
        fraction = np.divide(values[:, 0], total, out=np.zeros_like(total), where=total > 0)
        positive = visible & (fraction >= self.config.class_fraction - 1e-10)
        negative = visible & (fraction <= 1 - self.config.class_fraction + 1e-10)
        self.arrays["foreground_mass"] += values[:, 0]
        self.arrays["background_mass"] += values[:, 1]
        self.arrays["visible_view_count"] += visible
        self.arrays["foreground_view_count"] += positive
        self.arrays["background_view_count"] += negative
        self.arrays["mixed_view_count"] += visible & ~positive & ~negative
        self.group_sum += np.where(visible, fraction, 0)
        self.group_count += visible

    def finish_group(self) -> None:
        valid = self.group_count > 0
        self.arrays["group_fraction_sum"] += np.divide(
            self.group_sum, self.group_count, out=np.zeros_like(self.group_sum), where=valid)
        self.arrays["visible_group_count"] += valid
        self.start_group()

    def finalize(self) -> dict[str, np.ndarray]:
        if np.any(self.group_count):
            raise ValueError("Finish the current evidence group before classification")
        arrays = {name: value.copy() for name, value in self.arrays.items()}
        arrays["label"], arrays["reason"] = classify_evidence(arrays, self.config)
        arrays["subject_fraction"] = np.divide(
            arrays["group_fraction_sum"], arrays["visible_group_count"],
            out=np.zeros_like(arrays["group_fraction_sum"]), where=arrays["visible_group_count"] > 0)
        return arrays


def classify_evidence(arrays: dict[str, np.ndarray], config: PartitionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Only consistently visible, sufficiently supported points become subject.

    A reliable mixed silhouette footprint or a FG/BG view disagreement remains
    uncertain even if a majority vote could hide that disagreement.
    """
    groups = arrays["visible_group_count"]
    fraction = np.divide(arrays["group_fraction_sum"], groups,
                         out=np.zeros_like(arrays["group_fraction_sum"]), where=groups > 0)
    conflict = (arrays["foreground_view_count"] > 0) & (arrays["background_view_count"] > 0)
    mixed = arrays["mixed_view_count"] > 0
    enough = groups >= config.minimum_visible_groups
    eligible = enough & ~conflict & ~mixed
    labels = np.full(len(groups), UNCERTAIN, np.uint8)
    labels[eligible & (fraction >= config.class_fraction - 1e-10)] = SUBJECT
    labels[eligible & (fraction <= 1 - config.class_fraction + 1e-10)] = BACKGROUND
    reasons = np.zeros(len(groups), np.uint8)
    reasons[labels == UNCERTAIN] = 1
    reasons[mixed] = 2
    reasons[conflict] = 3
    return labels, reasons


def _read_camera_mask(root: Path, camera: Camera) -> np.ndarray:
    path = (root / camera.mask_relative_path).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Camera mask path escapes the dataset root")
    if _sha256_file(path) != camera.mask_sha256:
        raise ValueError(f"Camera mask identity mismatch: {camera.name}")
    values = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if values is None:
        raise ValueError(f"Unreadable mask: {path}")
    if values.ndim == 3:
        if not np.all(values == values[..., :1]):
            raise ValueError(f"Mask channels disagree: {camera.name}")
        values = values[..., 0]
    if values.shape != (camera.height, camera.width):
        raise ValueError(f"Mask dimensions disagree: {camera.name}")
    unique = np.unique(values)
    if np.any(unique < 0) or np.count_nonzero(unique > 0) > 1:
        raise ValueError(f"Mask must contain zero and at most one foreground value: {camera.name}")
    return values > 0


def _partition_counts(labels: np.ndarray) -> dict[str, int]:
    return {name: int(np.count_nonzero(labels == int(value))) for value, name in LABELS.items()}


def export_repartitioned_scene(scene: ForegroundBackgroundScene, output_dir: str | Path,
                               arrays: dict[str, np.ndarray], config: PartitionConfig,
                               camera_groups: list[list[int]], source_path: str = "") -> Path:
    """Publish a v3 static scene and independently verifiable three-way evidence."""
    config.validate()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    cameras = cameras_from_scene_manifest(scene.manifest)
    if not all(c.distortion_applied for c in cameras):
        raise ValueError("Repartition requires a distortion-aware static scene; legacy source remains unchanged")
    labels, reasons = classify_evidence(arrays, config)
    if not np.array_equal(labels, arrays["label"]) or not np.array_equal(reasons, arrays["reason"]):
        raise ValueError("Partition labels disagree with visibility evidence")
    subject = np.flatnonzero(labels == SUBJECT)
    stationary = np.flatnonzero(labels != SUBJECT)
    if len(subject) < 2 or len(stationary) < 2:
        raise ValueError(f"Repartition cannot form valid Gaussian sets: {_partition_counts(labels)}")
    if len(labels) != scene.count or camera_groups != group_camera_views(cameras, config):
        raise ValueError("Partition source count or camera groups disagree")
    order = np.concatenate((subject, stationary)).astype(np.int64)
    arrays = {**arrays, "new_to_source_index": order}
    source_tensors = scene.tensor_dictionary()
    tensors = {}
    for field in GAUSSIAN_FIELDS:
        combined = torch.cat((source_tensors[f"foreground.{field}"], source_tensors[f"background.{field}"]))
        tensors[f"foreground.{field}"] = combined[torch.from_numpy(subject)].contiguous()
        tensors[f"background.{field}"] = combined[torch.from_numpy(stationary)].contiguous()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        np.savez_compressed(temporary / "partition.npz", **arrays)
        partition = {
            "method": METHOD, "config": asdict(config), "labels": LABELS, "reasons": REASONS,
            "source_static_version": scene.manifest["version"],
            "source_partition_identity": scene.manifest.get("partition_identity"),
            "source_static_scene_identity": scene.manifest["static_scene_identity"],
            "source_dataset_identity": scene.manifest["dataset"]["dataset_identity"],
            "source_foreground_identity": scene.manifest["foreground_identity"],
            "source_background_identity": scene.manifest["background_identity"],
            "source_foreground_count": scene.foreground.count,
            "source_background_count": scene.background.count,
            "camera_identities": [c.to_manifest_record()["camera_identity"] for c in cameras],
            "mask_sha256": [c.mask_sha256 for c in cameras], "camera_groups": camera_groups,
            "counts": _partition_counts(labels), "evidence_file": "partition.npz",
            "evidence_sha256": _sha256_file(temporary / "partition.npz"),
            "motion_label": SUBJECT, "stationary_labels": [UNCERTAIN, BACKGROUND],
        }
        partition_identity = _sha256_json(partition)
        torch.save(tensors, temporary / "tensors.pt")
        manifest = deepcopy(scene.manifest)
        manifest.update(version=3, partition=partition, partition_identity=partition_identity,
                        partition_source_path=source_path,
                        foreground_identity=tensor_dictionary_identity(tensors, "foreground."),
                        background_identity=tensor_dictionary_identity(tensors, "background."),
                        tensors_sha256=_sha256_file(temporary / "tensors.pt"),
                        counts={"foreground": len(subject), "background": len(stationary)})
        manifest["representation"].update(
            foreground_local_index_domain=[0, len(subject)], background_local_index_domain=[0, len(stationary)],
            combined_foreground_index_domain=[0, len(subject)],
            combined_background_index_domain=[len(subject), len(labels)],
            background_role="static_background_and_uncertain", foreground_role="confident_motion_subject")
        identity_payload = {
            "dataset_identity": manifest["dataset"]["dataset_identity"],
            "foreground_identity": manifest["foreground_identity"],
            "background_identity": manifest["background_identity"], "normalization": manifest["scene_normalization"],
            "representation": "vanilla_3dgs_direct_rgb", "camera_identities": partition["camera_identities"],
            "projection_convention": PROJECTION_CONVENTION, "partition_identity": partition_identity,
        }
        manifest["static_scene_identity"] = _sha256_json(identity_payload)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summary = {
            "counts": partition["counts"], "source_counts": scene.manifest["counts"],
            "transition_counts": {
                old: _partition_counts(labels[lo:hi]) for old, lo, hi in (
                    ("old_foreground", 0, scene.foreground.count),
                    ("old_background", scene.foreground.count, scene.count))},
            "uncertain_reasons": {name: int(np.count_nonzero(reasons == int(code)))
                                  for code, name in REASONS.items() if code != "0"},
            "camera_count": len(cameras), "camera_group_count": len(camera_groups),
            "static_scene_identity": manifest["static_scene_identity"], "partition_identity": partition_identity,
            "gaussian_parameters_changed": False, "uncertain_in_motion_subject": False,
            "downstream_artifacts_require_rebuild": True,
        }
        subject_points = tensors["foreground.means"].numpy().astype(np.float64)
        summary["subject_bounds"] = [subject_points.min(0).tolist(), subject_points.max(0).tolist()]
        summary["subject_bbox_diagonal"] = float(np.linalg.norm(np.ptp(subject_points, axis=0)))
        summary["subject_quantile_01_99_diagonal"] = float(np.linalg.norm(
            np.quantile(subject_points, .99, axis=0) - np.quantile(subject_points, .01, axis=0)))
        (temporary / "partition-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        load_static_scene(temporary)
        os.replace(temporary, output)
    except BaseException:
        # Verify the resolved target before recursive cleanup on Windows.
        if temporary.resolve().parent != output.parent.resolve() or not temporary.name.startswith(f".{output.name}.tmp-"):
            raise RuntimeError("Refusing to clean an unexpected temporary path")
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def validate_partition_bundle(path: Path, manifest: dict[str, Any], tensors: dict[str, torch.Tensor]) -> None:
    """Validate evidence, decisions and a lossless mapping back to source tensors."""
    partition = manifest.get("partition", {})
    if partition.get("method") not in (METHOD, LEGACY_METHOD) or _sha256_json(partition) != manifest.get("partition_identity"):
        raise ValueError("Static partition identity mismatch")
    if (partition.get("labels") != LABELS or partition.get("reasons") != REASONS
            or partition.get("motion_label") != SUBJECT or partition.get("stationary_labels") != [UNCERTAIN, BACKGROUND]
            or partition.get("evidence_file") != "partition.npz"):
        raise ValueError("Static partition semantic contract mismatch")
    saved_config = dict(partition["config"])
    if partition["method"] == LEGACY_METHOD:
        # Old evidence was computed with erosion. Validate its recorded setting,
        # then reuse only the unchanged classification/pose settings below.
        # Loading never recomputes masks or relabels an existing artifact.
        margin = saved_config.pop("mask_margin_pixels", None)
        if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0 or "mask_dilation_pixels" in saved_config:
            raise ValueError("Invalid legacy partition mask configuration")
        saved_config["mask_dilation_pixels"] = 0
    config = PartitionConfig(**saved_config)
    config.validate()
    cameras = cameras_from_scene_manifest(manifest)
    if (partition["camera_identities"] != [c.to_manifest_record()["camera_identity"] for c in cameras]
            or partition["mask_sha256"] != [c.mask_sha256 for c in cameras]
            or partition["camera_groups"] != group_camera_views(cameras, config)
            or partition["source_dataset_identity"] != manifest["dataset"]["dataset_identity"]):
        raise ValueError("Partition camera/mask source mismatch")
    source_identity_payload = {
        "dataset_identity": partition["source_dataset_identity"],
        "foreground_identity": partition["source_foreground_identity"],
        "background_identity": partition["source_background_identity"],
        "normalization": manifest["scene_normalization"], "representation": "vanilla_3dgs_direct_rgb",
        "camera_identities": partition["camera_identities"], "projection_convention": PROJECTION_CONVENTION,
    }
    if partition.get("source_static_version") == 3:
        source_identity_payload["partition_identity"] = partition["source_partition_identity"]
    elif partition.get("source_static_version") != 2 or partition.get("source_partition_identity") is not None:
        raise ValueError("Unsupported partition source version")
    if _sha256_json(source_identity_payload) != partition["source_static_scene_identity"]:
        raise ValueError("Partition source scene identity mismatch")
    evidence_path = path / "partition.npz"
    if _sha256_file(evidence_path) != partition["evidence_sha256"]:
        raise ValueError("Partition evidence checksum mismatch")
    with np.load(evidence_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    count = partition["source_foreground_count"] + partition["source_background_count"]
    expected = EvidenceAccumulator(count, config).finalize()
    if set(arrays) != {*expected, "new_to_source_index"}:
        raise ValueError("Partition evidence schema mismatch")
    for name, reference in expected.items():
        value = arrays[name]
        if value.shape != (count,) or value.dtype != reference.dtype or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"Invalid partition evidence: {name}")
    visible = arrays["visible_view_count"]
    groups = arrays["visible_group_count"]
    if (np.any(visible > len(cameras)) or np.any(groups > len(partition["camera_groups"]))
            or np.any(groups > visible) or np.any(arrays["group_fraction_sum"] > groups + 1e-9)
            or not np.array_equal(visible, arrays["foreground_view_count"] + arrays["background_view_count"] + arrays["mixed_view_count"])):
        raise ValueError("Inconsistent partition evidence counts")
    labels, reasons = classify_evidence(arrays, config)
    fraction = np.divide(arrays["group_fraction_sum"], groups,
                         out=np.zeros(count, np.float64), where=groups > 0)
    if (not np.array_equal(labels, arrays["label"]) or not np.array_equal(reasons, arrays["reason"])
            or not np.array_equal(fraction, arrays["subject_fraction"])
            or partition["counts"] != _partition_counts(labels)):
        raise ValueError("Partition decisions disagree with evidence")
    order = np.concatenate((np.flatnonzero(labels == SUBJECT), np.flatnonzero(labels != SUBJECT)))
    if arrays["new_to_source_index"].dtype != np.int64 or not np.array_equal(order, arrays["new_to_source_index"]):
        raise ValueError("Partition index mapping mismatch")
    fg_count = int(np.count_nonzero(labels == SUBJECT))
    if manifest["counts"] != {"foreground": fg_count, "background": count - fg_count}:
        raise ValueError("Partition scene counts mismatch")
    reconstructed = {}
    split = partition["source_foreground_count"]
    for field in GAUSSIAN_FIELDS:
        if tensors[f"foreground.{field}"].shape[0] != fg_count or tensors[f"background.{field}"].shape[0] != count - fg_count:
            raise ValueError("Partition tensor domain mismatch")
        current = torch.cat((tensors[f"foreground.{field}"], tensors[f"background.{field}"]))
        restored = current[torch.from_numpy(np.argsort(order))]
        reconstructed[f"foreground.{field}"] = restored[:split]
        reconstructed[f"background.{field}"] = restored[split:]
    for part in ("foreground", "background"):
        if tensor_dictionary_identity(reconstructed, part + ".") != partition[f"source_{part}_identity"]:
            raise ValueError("Partition changed source Gaussian parameters")


def repartition_static_scene(*, scene_dir: str | Path, output_dir: str | Path,
                             config: PartitionConfig | None = None, device: str = "auto",
                             dataset_root: str | Path | None = None) -> Path:
    """Classify an existing static scene; no training, control graph, PNG or Viewer."""
    settings = config or PartitionConfig()
    settings.validate()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    scene = load_static_scene(scene_dir, device=selected_device).eval()
    cameras = cameras_from_scene_manifest(scene.manifest)
    if not all(c.distortion_applied for c in cameras):
        raise ValueError("Repartition requires a distortion-aware static scene")
    root = Path(dataset_root or scene.manifest["dataset"]["input_root"]).expanduser().resolve(strict=True)
    groups = group_camera_views(cameras, settings)
    evidence = EvidenceAccumulator(scene.count, settings)
    progress = Progress("static visibility repartition", len(cameras), unit="views")
    completed = 0
    for group_index, group in enumerate(groups):
        for index in group:
            camera = cameras[index]
            regions = dilated_mask_regions(_read_camera_mask(root, camera), settings.mask_dilation_pixels)
            evidence.add_view(visible_mask_mass(scene, camera, *regions))
            completed += 1
            progress.update(completed, f"group={group_index + 1}/{len(groups)} camera={camera.name}")
        evidence.finish_group()
    return export_repartitioned_scene(scene, output, evidence.finalize(), settings, groups,
                                      str(Path(scene_dir).expanduser().resolve()))
