"""Mode-independent pixel-to-foreground-Gaussian observation topology."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from modal_gaussians.common.scene_store import resolve_path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from modal_gaussians.common.numpy_io import save_named_arrays
from modal_gaussians.common.cache import publish_directory
from modal_gaussians.geometry.scene import Camera


TOPOLOGY_FORMAT = "modal_gaussians.observation_topology"
TOPOLOGY_VERSION = 1
ARRAY_FILENAME = "topology.npz"

ARRAY_DTYPES = {
    "view_shapes_hw": np.dtype(np.int64),
    "sample_view_index": np.dtype(np.int64),
    "sample_pixels_xy": np.dtype(np.int64),
    "sample_surface_points": np.dtype(np.float32),
    "sample_rendered_depth": np.dtype(np.float32),
    "sample_foreground_alpha": np.dtype(np.float32),
    "sample_offsets": np.dtype(np.int64),
    "contributor_gaussian_index": np.dtype(np.int64),
    "contributor_weight": np.dtype(np.float32),
    "contributor_jacobian": np.dtype(np.float32),
}


@dataclass(frozen=True)
class TopologyConfig:
    """Hold the accepted pixel sampling and Gaussian contribution settings."""

    pixel_sample_stride: int = 4
    pixel_candidate_count: int = 4
    pixel_preselect_count: int = 32
    foreground_alpha_minimum: float = 0.05
    minimum_contribution: float = 1e-12
    mask_erosion_iterations: int = 1

    def validate(self) -> None:
        """Reject settings that cannot define a valid observation topology."""

        if self.pixel_sample_stride < 1:
            raise ValueError("pixel_sample_stride must be at least one")
        if self.pixel_candidate_count < 1:
            raise ValueError("pixel_candidate_count must be at least one")
        if self.pixel_preselect_count < self.pixel_candidate_count:
            raise ValueError(
                "pixel_preselect_count must be at least pixel_candidate_count"
            )
        if not math.isfinite(self.foreground_alpha_minimum) or not (
            0.0 <= self.foreground_alpha_minimum <= 1.0
        ):
            raise ValueError("foreground_alpha_minimum must be finite and in [0, 1]")
        if not math.isfinite(self.minimum_contribution) or (
            self.minimum_contribution < 0.0
        ):
            raise ValueError("minimum_contribution must be finite and non-negative")
        if self.mask_erosion_iterations < 0:
            raise ValueError("mask_erosion_iterations must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the exact scientific settings included in the identity."""

        return {
            "pixel_sample_stride": self.pixel_sample_stride,
            "pixel_candidate_count": self.pixel_candidate_count,
            "pixel_preselect_count": self.pixel_preselect_count,
            "foreground_alpha_minimum": self.foreground_alpha_minimum,
            "minimum_contribution": self.minimum_contribution,
            "mask_erosion_iterations": self.mask_erosion_iterations,
            "candidate_method": "rendered_depth_opacity_mahalanobis",
            "contributor_normalization": "positive_scores_sum_to_one_per_pixel",
            "projection_jacobian": "d_pinhole_pixel_d_normalized_world_point",
        }




@dataclass(frozen=True)
class TopologyArrays:
    """Store the compact ragged pixel-to-Gaussian topology arrays."""

    view_shapes_hw: np.ndarray
    sample_view_index: np.ndarray
    sample_pixels_xy: np.ndarray
    sample_surface_points: np.ndarray
    sample_rendered_depth: np.ndarray
    sample_foreground_alpha: np.ndarray
    sample_offsets: np.ndarray
    contributor_gaussian_index: np.ndarray
    contributor_weight: np.ndarray
    contributor_jacobian: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return arrays with canonical dtypes for publication and hashing."""

        return {
            name: np.ascontiguousarray(
                getattr(self, name), dtype=ARRAY_DTYPES[name]
            )
            for name in ARRAY_DTYPES
        }


@dataclass(frozen=True)
class ObservationTopologyArtifact:
    """Represent one validated topology directory."""

    path: Path
    manifest: dict[str, Any]
    arrays: TopologyArrays


def _canonical_json(value: Any) -> bytes:
    """Encode one identity payload deterministically without NaN values."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    """Hash one file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    """Hash one array's dtype, shape, and raw C-order values."""

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _arrays_identity(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash all topology arrays in stable field order."""

    digest = hashlib.sha256()
    for name in ARRAY_DTYPES:
        digest.update(name.encode("utf-8"))
        digest.update(_sha256_array(arrays[name]).encode("ascii"))
    return digest.hexdigest()


def _quaternion_wxyz_to_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    """Convert normalized-or-raw wxyz quaternions into rotation matrices."""

    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"quaternions must have shape [G,4], got {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("quaternions contain zero or non-finite values")
    q = values / norms
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrices = np.empty((len(q), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - w * z)
    matrices[:, 0, 2] = 2.0 * (x * z + w * y)
    matrices[:, 1, 0] = 2.0 * (x * y + w * z)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - w * x)
    matrices[:, 2, 0] = 2.0 * (x * z - w * y)
    matrices[:, 2, 1] = 2.0 * (y * z + w * x)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def _world_to_camera_points(points: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    """Transform normalized-world points into one camera coordinate system."""

    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [values, np.ones((len(values), 1), dtype=np.float64)], axis=1
    )
    return (homogeneous @ np.asarray(world_to_camera, dtype=np.float64).T)[:, :3]


def _unproject_pixels(
    pixels_xy: np.ndarray,
    depths: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
    radial_k: float = 0.0,
) -> np.ndarray:
    """Unproject camera-z depth pixels into normalized world coordinates."""

    pixels = np.asarray(pixels_xy, dtype=np.float64)
    z = np.asarray(depths, dtype=np.float64)
    x = (pixels[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (pixels[:, 1] - K[1, 2]) * z / K[1, 1]
    if radial_k:
        from modal_gaussians.common.camera_geometry import undistort_normalized
        xy = undistort_normalized(np.column_stack(((pixels[:, 0] - K[0, 2]) / K[0, 0],
                                                   (pixels[:, 1] - K[1, 2]) / K[1, 1])), radial_k)
        x, y = (xy * z[:, None]).T
    camera_points = np.stack([x, y, z, np.ones_like(z)], axis=1)
    camera_to_world = np.linalg.inv(np.asarray(world_to_camera, dtype=np.float64))
    return (camera_points @ camera_to_world.T)[:, :3].astype(np.float32)


def _projection_jacobian(
    points: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
    radial_k: float = 0.0,
) -> np.ndarray:
    """Compute d(pixel xy)/d(normalized world xyz) at Gaussian centers."""

    camera_points = _world_to_camera_points(points, world_to_camera)
    x, y, z = camera_points[:, 0], camera_points[:, 1], camera_points[:, 2]
    if np.any(z <= 0.0) or not np.isfinite(camera_points).all():
        raise ValueError("Projection Jacobian requires finite positive camera depth")
    camera_jacobian = np.zeros((len(points), 2, 3), dtype=np.float64)
    camera_jacobian[:, 0, 0] = float(K[0, 0]) / z
    camera_jacobian[:, 0, 2] = -float(K[0, 0]) * x / (z * z)
    camera_jacobian[:, 1, 1] = float(K[1, 1]) / z
    camera_jacobian[:, 1, 2] = -float(K[1, 1]) * y / (z * z)
    if radial_k:
        from modal_gaussians.common.camera_geometry import camera_jacobian as radial_jacobian
        camera_jacobian = radial_jacobian(camera_points, K, radial_k)
    rotation = np.asarray(world_to_camera, dtype=np.float64)[:3, :3]
    return np.einsum("nij,jk->nik", camera_jacobian, rotation).astype(np.float32)


def _require_ckdtree() -> Any:
    """Load SciPy's compiled KD-tree without relying on incomplete type stubs."""

    try:
        spatial = importlib.import_module("scipy.spatial")
    except ImportError as error:
        raise RuntimeError("Observation topology requires scipy") from error
    tree_type = getattr(spatial, "cKDTree", None)
    if tree_type is None:
        raise RuntimeError("scipy.spatial.cKDTree is unavailable")
    return tree_type


def _validate_gaussian_inputs(
    means: np.ndarray,
    scales: np.ndarray,
    quaternions: np.ndarray,
    opacities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate activated foreground Gaussian attributes used for scoring."""

    points = np.asarray(means, dtype=np.float32)
    scale_values = np.asarray(scales, dtype=np.float32)
    quaternion_values = np.asarray(quaternions, dtype=np.float32)
    opacity_values = np.asarray(opacities, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"foreground means must be [G,3] with G>=2, got {points.shape}")
    if scale_values.shape != points.shape:
        raise ValueError(f"foreground scales must have shape {points.shape}")
    if quaternion_values.shape != (len(points), 4):
        raise ValueError(f"foreground quaternions must have shape [{len(points)},4]")
    if opacity_values.shape != (len(points),):
        raise ValueError(f"foreground opacities must have shape [{len(points)}]")
    if not np.isfinite(points).all():
        raise ValueError("foreground means contain NaN or Inf")
    if not np.isfinite(scale_values).all() or np.any(scale_values <= 0.0):
        raise ValueError("foreground scales must be finite and positive")
    if (
        not np.isfinite(opacity_values).all()
        or np.any(opacity_values < 0.0)
        or np.any(opacity_values > 1.0)
    ):
        raise ValueError("foreground opacities must be finite and in [0,1]")
    return points, scale_values, quaternion_values, opacity_values


def build_topology_arrays(
    *,
    foreground_means: np.ndarray,
    foreground_scales: np.ndarray,
    foreground_quaternions: np.ndarray,
    foreground_opacities: np.ndarray,
    cameras: Sequence[Camera],
    masks: Sequence[np.ndarray],
    rendered_depths: Sequence[np.ndarray],
    rendered_alphas: Sequence[np.ndarray],
    config: TopologyConfig | None = None,
) -> tuple[TopologyArrays, dict[str, list[int]]]:
    """Build the accepted rendered-depth and Mahalanobis contributor mapping."""

    settings = config or TopologyConfig()
    settings.validate()
    if not cameras or not (
        len(cameras) == len(masks) == len(rendered_depths) == len(rendered_alphas)
    ):
        raise ValueError("Each topology view requires one camera, mask, depth, and alpha")
    points, scales, quaternions, opacities = _validate_gaussian_inputs(
        foreground_means,
        foreground_scales,
        foreground_quaternions,
        foreground_opacities,
    )
    rotations = _quaternion_wxyz_to_rotation_matrices(quaternions)
    tree = _require_ckdtree()(points.astype(np.float64))

    sample_view_indices: list[int] = []
    sample_pixels: list[list[int]] = []
    sample_surface_points: list[np.ndarray] = []
    sample_depths: list[float] = []
    sample_alphas: list[float] = []
    sample_offsets = [0]
    contributor_indices: list[int] = []
    contributor_weights: list[float] = []
    contributor_jacobians: list[np.ndarray] = []
    sample_counts_per_view: list[int] = []
    contributor_counts_per_view: list[int] = []

    kernel = np.ones((3, 3), dtype=np.uint8)
    preselect_count = min(settings.pixel_preselect_count, len(points))
    for view_index, (camera, mask, depth, alpha) in enumerate(
        zip(cameras, masks, rendered_depths, rendered_alphas)
    ):
        expected_shape = (camera.height, camera.width)
        mask_value = np.asarray(mask, dtype=bool)
        depth_value = np.asarray(depth, dtype=np.float32)
        alpha_value = np.asarray(alpha, dtype=np.float32)
        if mask_value.shape != expected_shape:
            raise ValueError(f"Mask for {camera.label or camera.name} has wrong shape")
        if depth_value.shape != expected_shape or alpha_value.shape != expected_shape:
            raise ValueError(f"Rendered outputs for {camera.label or camera.name} have wrong shape")
        if settings.mask_erosion_iterations:
            mask_value = cv2.erode(
                mask_value.astype(np.uint8),
                kernel,
                iterations=settings.mask_erosion_iterations,
            ).astype(bool)
        visible = (
            mask_value
            & np.isfinite(depth_value)
            & (depth_value > 0.0)
            & np.isfinite(alpha_value)
            & (alpha_value >= settings.foreground_alpha_minimum)
        )
        ys = np.arange(
            1,
            camera.height - 1,
            settings.pixel_sample_stride,
            dtype=np.int64,
        )
        xs = np.arange(
            1,
            camera.width - 1,
            settings.pixel_sample_stride,
            dtype=np.int64,
        )
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        x_values = xx.reshape(-1)
        y_values = yy.reshape(-1)
        keep = visible[y_values, x_values]
        x_values = x_values[keep]
        y_values = y_values[keep]
        if len(x_values) == 0:
            raise ValueError(
                f"No candidate pixels survived mask/depth/alpha checks for "
                f"{camera.label or camera.name}"
            )
        selected_depths = depth_value[y_values, x_values]
        selected_alphas = alpha_value[y_values, x_values]
        pixels_xy = np.stack([x_values, y_values], axis=1).astype(np.float32)
        K = camera.K.detach().cpu().numpy().astype(np.float64)
        world_to_camera = (
            camera.world_to_camera.detach().cpu().numpy().astype(np.float64)
        )
        surface_points = _unproject_pixels(
            pixels_xy, selected_depths, K, world_to_camera, camera.radial_distortion
        )
        finite = np.isfinite(surface_points).all(axis=1)
        x_values = x_values[finite]
        y_values = y_values[finite]
        selected_depths = selected_depths[finite]
        selected_alphas = selected_alphas[finite]
        surface_points = surface_points[finite]
        if len(surface_points) == 0:
            raise ValueError(
                f"No finite unprojected surface points for {camera.label or camera.name}"
            )
        _, preselected = tree.query(
            surface_points.astype(np.float64), k=preselect_count
        )
        if preselect_count == 1:
            preselected = np.asarray(preselected)[:, None]
        camera_points = _world_to_camera_points(points, world_to_camera)
        camera_depths = camera_points[:, 2]
        from modal_gaussians.common.camera_geometry import radial_domain
        projectable = radial_domain(camera_points, camera.radial_distortion)
        samples_before = len(sample_view_indices)
        contributors_before = len(contributor_indices)

        for x, y, surface_point, rendered_depth, rendered_alpha, candidates in zip(
            x_values,
            y_values,
            surface_points,
            selected_depths,
            selected_alphas,
            preselected,
        ):
            candidate_indices = np.asarray(candidates, dtype=np.int64).reshape(-1)
            delta = (
                surface_point.astype(np.float64)[None]
                - points[candidate_indices].astype(np.float64)
            )
            local_delta = np.einsum(
                "nij,nj->ni",
                np.swapaxes(rotations[candidate_indices], 1, 2),
                delta,
            )
            scaled_delta = local_delta / np.maximum(
                scales[candidate_indices].astype(np.float64),
                float(np.float32(1e-8)),
            )
            mahalanobis_squared = np.sum(scaled_delta * scaled_delta, axis=1)
            scores = opacities[candidate_indices].astype(np.float64) * np.exp(
                -0.5 * mahalanobis_squared
            )
            valid_score = np.isfinite(scores) & (
                scores >= settings.minimum_contribution
            )
            candidate_indices = candidate_indices[valid_score]
            scores = scores[valid_score]
            if len(candidate_indices) == 0:
                continue
            order = np.argsort(scores)[::-1]
            candidate_indices = candidate_indices[order][
                : settings.pixel_candidate_count
            ]
            scores = scores[order][: settings.pixel_candidate_count]
            positive_depth = (
                np.isfinite(camera_depths[candidate_indices])
                & (camera_depths[candidate_indices] > 0.0)
                & projectable[candidate_indices]
            )
            candidate_indices = candidate_indices[positive_depth]
            scores = scores[positive_depth]
            score_sum = float(np.sum(scores))
            if len(candidate_indices) == 0 or not math.isfinite(score_sum) or score_sum <= 0.0:
                continue
            weights = (scores / score_sum).astype(np.float32)
            jacobians = _projection_jacobian(
                points[candidate_indices], K, world_to_camera, camera.radial_distortion
            )
            sample_view_indices.append(view_index)
            sample_pixels.append([int(x), int(y)])
            sample_surface_points.append(surface_point.astype(np.float32))
            sample_depths.append(float(rendered_depth))
            sample_alphas.append(float(rendered_alpha))
            contributor_indices.extend(candidate_indices.tolist())
            contributor_weights.extend(weights.tolist())
            contributor_jacobians.extend(jacobians)
            sample_offsets.append(len(contributor_indices))

        view_sample_count = len(sample_view_indices) - samples_before
        view_contributor_count = len(contributor_indices) - contributors_before
        if view_sample_count == 0:
            raise ValueError(
                f"No Gaussian contributors survived for {camera.label or camera.name}"
            )
        sample_counts_per_view.append(view_sample_count)
        contributor_counts_per_view.append(view_contributor_count)

    arrays = TopologyArrays(
        view_shapes_hw=np.asarray(
            [[camera.height, camera.width] for camera in cameras], dtype=np.int64
        ),
        sample_view_index=np.asarray(sample_view_indices, dtype=np.int64),
        sample_pixels_xy=np.asarray(sample_pixels, dtype=np.int64),
        sample_surface_points=np.asarray(sample_surface_points, dtype=np.float32),
        sample_rendered_depth=np.asarray(sample_depths, dtype=np.float32),
        sample_foreground_alpha=np.asarray(sample_alphas, dtype=np.float32),
        sample_offsets=np.asarray(sample_offsets, dtype=np.int64),
        contributor_gaussian_index=np.asarray(contributor_indices, dtype=np.int64),
        contributor_weight=np.asarray(contributor_weights, dtype=np.float32),
        contributor_jacobian=np.asarray(contributor_jacobians, dtype=np.float32),
    )
    return arrays, {
        "samples_per_view": sample_counts_per_view,
        "contributors_per_view": contributor_counts_per_view,
    }


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    view_count: int,
    foreground_count: int,
    config: TopologyConfig,
) -> None:
    """Validate shapes, ragged offsets, index domains, and normalized weights."""

    if set(arrays) != set(ARRAY_DTYPES):
        raise ValueError(f"Topology array inventory is invalid: {sorted(arrays)}")
    for name, dtype in ARRAY_DTYPES.items():
        if arrays[name].dtype != dtype:
            raise ValueError(f"Topology {name} must be {dtype.name}")
    view_shapes = arrays["view_shapes_hw"]
    sample_views = arrays["sample_view_index"]
    sample_pixels = arrays["sample_pixels_xy"]
    surface_points = arrays["sample_surface_points"]
    sample_depth = arrays["sample_rendered_depth"]
    sample_alpha = arrays["sample_foreground_alpha"]
    offsets = arrays["sample_offsets"]
    indices = arrays["contributor_gaussian_index"]
    weights = arrays["contributor_weight"]
    jacobians = arrays["contributor_jacobian"]
    sample_count = len(sample_views)
    contributor_count = len(indices)
    if view_shapes.shape != (view_count, 2) or np.any(view_shapes <= 2):
        raise ValueError("Topology view_shapes_hw is invalid")
    if sample_count == 0 or contributor_count == 0:
        raise ValueError("Topology must contain samples and contributors")
    expected_sample_shapes = {
        "sample_pixels_xy": (sample_count, 2),
        "sample_surface_points": (sample_count, 3),
        "sample_rendered_depth": (sample_count,),
        "sample_foreground_alpha": (sample_count,),
        "sample_offsets": (sample_count + 1,),
    }
    for name, shape in expected_sample_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Topology {name} has shape {arrays[name].shape}, expected {shape}")
    if weights.shape != (contributor_count,) or jacobians.shape != (
        contributor_count,
        2,
        3,
    ):
        raise ValueError("Topology contributor arrays have inconsistent shapes")
    if offsets[0] != 0 or offsets[-1] != contributor_count:
        raise ValueError("Topology sample_offsets endpoints are invalid")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("Every topology sample must have at least one contributor")
    if np.any(sample_views < 0) or np.any(sample_views >= view_count):
        raise ValueError("Topology sample view index is out of range")
    sample_shapes = view_shapes[sample_views]
    if (
        np.any(sample_pixels[:, 0] < 1)
        or np.any(sample_pixels[:, 1] < 1)
        or np.any(sample_pixels[:, 0] >= sample_shapes[:, 1] - 1)
        or np.any(sample_pixels[:, 1] >= sample_shapes[:, 0] - 1)
    ):
        raise ValueError("Topology sample pixel is outside the accepted interior grid")
    if (
        np.any((sample_pixels[:, 0] - 1) % config.pixel_sample_stride != 0)
        or np.any((sample_pixels[:, 1] - 1) % config.pixel_sample_stride != 0)
    ):
        raise ValueError("Topology sample pixel does not follow the configured stride")
    if np.any(indices < 0) or np.any(indices >= foreground_count):
        raise ValueError("Topology contributor index is outside the foreground domain")
    if not np.isfinite(surface_points).all():
        raise ValueError("Topology surface points contain NaN or Inf")
    if not np.isfinite(sample_depth).all() or np.any(sample_depth <= 0.0):
        raise ValueError("Topology rendered depths must be finite and positive")
    if (
        not np.isfinite(sample_alpha).all()
        or np.any(sample_alpha < config.foreground_alpha_minimum)
        or np.any(sample_alpha > 1.0)
    ):
        raise ValueError("Topology foreground alpha values are invalid")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("Topology contributor weights must be finite and positive")
    if not np.isfinite(jacobians).all():
        raise ValueError("Topology contributor Jacobians contain NaN or Inf")
    weight_sums = np.add.reduceat(weights, offsets[:-1])
    if not np.allclose(weight_sums, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Topology contributor weights do not sum to one per sample")


def _identity_payload(
    manifest: Mapping[str, Any], arrays_identity: str
) -> dict[str, Any]:
    """Select path-independent scientific fields for topology identity hashing."""

    views = []
    for view in manifest["views"]:
        views.append(
            {
                "label": view["label"],
                "camera_identity": view["camera_identity"],
                "flow_identity": view["flow_identity"],
                "mask_union_sha256": view["mask_union_sha256"],
                "shape_hw": view["shape_hw"],
            }
        )
    return {
        "format": TOPOLOGY_FORMAT,
        "version": TOPOLOGY_VERSION,
        "static_scene_identity": manifest["static_scene_identity"],
        "foreground_identity": manifest["foreground_identity"],
        "views": views,
        "parameters": manifest["parameters"],
        "arrays_identity": arrays_identity,
    }


def load_observation_topology(path: str | Path) -> ObservationTopologyArtifact:
    """Load and fully validate one compact topology artifact directory."""

    root = resolve_path(path, strict=True)
    manifest_path = root / "manifest.json"
    arrays_path = root / ARRAY_FILENAME
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"Incomplete observation topology artifact: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Topology manifest root must be an object")
    if manifest.get("format") != TOPOLOGY_FORMAT or manifest.get("version") != 1:
        raise ValueError(f"Unsupported topology artifact: {manifest_path}")
    views = manifest.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("Topology manifest must contain ordered views")
    labels = [view.get("label") for view in views if isinstance(view, dict)]
    if len(labels) != len(views) or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Topology view labels are invalid")
    if len(set(labels)) != len(labels):
        raise ValueError("Topology view labels must be unique")
    if manifest.get("arrays_file_sha256") != _sha256_file(arrays_path):
        raise ValueError("Topology NPZ SHA-256 does not match manifest")
    with np.load(arrays_path, allow_pickle=False) as archive:
        loaded = {name: archive[name] for name in archive.files}
    config = TopologyConfig(
        pixel_sample_stride=int(manifest["parameters"]["pixel_sample_stride"]),
        pixel_candidate_count=int(manifest["parameters"]["pixel_candidate_count"]),
        pixel_preselect_count=int(manifest["parameters"]["pixel_preselect_count"]),
        foreground_alpha_minimum=float(
            manifest["parameters"]["foreground_alpha_minimum"]
        ),
        minimum_contribution=float(manifest["parameters"]["minimum_contribution"]),
        mask_erosion_iterations=int(
            manifest["parameters"]["mask_erosion_iterations"]
        ),
    )
    config.validate()
    foreground_count = int(manifest["counts"]["foreground_gaussians"])
    _validate_arrays(
        loaded,
        view_count=len(views),
        foreground_count=foreground_count,
        config=config,
    )
    arrays_identity = _arrays_identity(loaded)
    if manifest.get("arrays_identity") != arrays_identity:
        raise ValueError("Topology array identity does not match manifest")
    expected_identity = hashlib.sha256(
        _canonical_json(_identity_payload(manifest, arrays_identity))
    ).hexdigest()
    if manifest.get("topology_identity") != expected_identity:
        raise ValueError("Topology identity does not match manifest and arrays")
    arrays = TopologyArrays(**loaded)
    return ObservationTopologyArtifact(root, manifest, arrays)


def _publish_topology(
    destination: Path,
    *,
    arrays: TopologyArrays,
    manifest: dict[str, Any],
    validate: bool = True,
) -> ObservationTopologyArtifact:
    """Publish topology atomically; optional replay is for explicit diagnostics."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Topology output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        values = arrays.as_dict()
        arrays_path = temporary / ARRAY_FILENAME
        save_named_arrays(arrays_path, values)
        arrays_identity = _arrays_identity(values)
        manifest["arrays"] = {
            name: {"dtype": value.dtype.name, "shape": list(value.shape)}
            for name, value in values.items()
        }
        manifest["arrays_file"] = ARRAY_FILENAME
        manifest["arrays_file_sha256"] = _sha256_file(arrays_path)
        manifest["arrays_identity"] = arrays_identity
        manifest["topology_identity"] = hashlib.sha256(
            _canonical_json(_identity_payload(manifest, arrays_identity))
        ).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        validated = (load_observation_topology(temporary) if validate else
                     ObservationTopologyArtifact(temporary, manifest, arrays))
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Topology output already exists: {destination}")
        publish_directory(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ObservationTopologyArtifact(destination.resolve(), validated.manifest, validated.arrays)


__all__ = [
    "ObservationTopologyArtifact",
    "TopologyArrays",
    "TopologyConfig",
    "build_observation_topology_artifact",
    "build_topology_arrays",
    "load_observation_topology",
]
