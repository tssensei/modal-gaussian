"""Observed foreground-Gaussian graph for local rigid-component solving."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from modal_gaussians.numpy_io import save_named_arrays
import torch

from modal_gaussians import __version__
from modal_gaussians.static import cameras_from_scene_manifest, load_static_scene
from modal_gaussians.topology import (
    ObservationTopologyArtifact,
    TopologyArrays,
    load_observation_topology,
)


GRAPH_FORMAT = "modal_gaussians.observed_structure_graph"
GRAPH_VERSION = 1
GRAPH_FILENAME = "graph.npz"
MAD_SCALE = 1.4826
PROFILE_BATCH_SIZE = 65_536

GRAPH_SEMANTICS = {
    "node_selection": "positive_topology_contributor",
    "knn_policy": "deterministic_mutual_knn",
    "color_space": "opencv_float_rgb_to_lab",
    "depth_source": "static_foreground_expected_depth",
    "distance_filter": "absolute_maximum",
    "color_filter": "global_median_plus_scaled_mad",
    "depth_filter": "per_view_endpoint_and_profile_median_plus_scaled_mad",
    "component_pruning": "remove_edges_keep_observed_nodes_isolated",
    "edge_weight": "inverse_distance_times_color_gaussian_times_depth_score",
    "quality_gate": "manual_component_inspection_required",
}

ARRAY_DTYPES = {
    "node_gaussian_index": np.dtype(np.int64),
    "node_observed_view_mask": np.dtype(bool),
    "edge_index": np.dtype(np.int64),
    "edge_distance": np.dtype(np.float32),
    "edge_color_distance": np.dtype(np.float32),
    "edge_depth_score": np.dtype(np.float32),
    "edge_weight": np.dtype(np.float32),
    "edge_view_support_mask": np.dtype(bool),
    "edge_endpoint_gap_by_view": np.dtype(np.float32),
    "edge_depth_jump_by_view": np.dtype(np.float32),
    "degree": np.dtype(np.int64),
    "component_index": np.dtype(np.int64),
    "component_size": np.dtype(np.int64),
    "component_pruned_node_mask": np.dtype(bool),
    "isolated_mask": np.dtype(bool),
}


@dataclass(frozen=True)
class ObservedStructureGraphConfig:
    """Hold the accepted rigid-component graph construction settings."""

    max_neighbors: int = 8
    max_distance: float = 0.008
    color_mad_multiplier: float = 3.0
    depth_mad_multiplier: float = 3.0
    depth_samples: int = 5
    min_shared_views: int = 1
    min_component_nodes: int = 4
    min_component_edges: int = 3

    def validate(self, view_count: int) -> None:
        """Reject settings that cannot define the accepted graph."""

        for name, value in (
            ("max_neighbors", self.max_neighbors),
            ("depth_samples", self.depth_samples),
            ("min_shared_views", self.min_shared_views),
            ("min_component_nodes", self.min_component_nodes),
            ("min_component_edges", self.min_component_edges),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Observed graph {name} must be a positive integer")
        if self.depth_samples < 2:
            raise ValueError("Observed graph depth_samples must be at least two")
        if self.min_shared_views > view_count:
            raise ValueError("Observed graph min_shared_views exceeds the view count")
        if not math.isfinite(self.max_distance) or self.max_distance <= 0.0:
            raise ValueError("Observed graph max_distance must be finite and positive")
        for name, value in (
            ("color_mad_multiplier", self.color_mad_multiplier),
            ("depth_mad_multiplier", self.depth_mad_multiplier),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Observed graph {name} must be finite and non-negative")

    def to_dict(self, render_alpha_minimum: float) -> dict[str, Any]:
        """Serialize graph settings plus the topology-owned render threshold."""

        return {
            "max_neighbors": self.max_neighbors,
            "max_distance": self.max_distance,
            "color_mad_multiplier": self.color_mad_multiplier,
            "depth_mad_multiplier": self.depth_mad_multiplier,
            "depth_samples": self.depth_samples,
            "min_shared_views": self.min_shared_views,
            "min_component_nodes": self.min_component_nodes,
            "min_component_edges": self.min_component_edges,
            "render_alpha_minimum": render_alpha_minimum,
            "epsilon": 1.0e-8,
            "mad_scale": MAD_SCALE,
        }


@dataclass(frozen=True)
class StructureGraphArrays:
    """Store the minimal solver-facing observed graph arrays."""

    node_gaussian_index: np.ndarray
    node_observed_view_mask: np.ndarray
    edge_index: np.ndarray
    edge_distance: np.ndarray
    edge_color_distance: np.ndarray
    edge_depth_score: np.ndarray
    edge_weight: np.ndarray
    edge_view_support_mask: np.ndarray
    edge_endpoint_gap_by_view: np.ndarray
    edge_depth_jump_by_view: np.ndarray
    degree: np.ndarray
    component_index: np.ndarray
    component_size: np.ndarray
    component_pruned_node_mask: np.ndarray
    isolated_mask: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return arrays in the canonical artifact dtype and field order."""

        return {
            name: np.ascontiguousarray(getattr(self, name), dtype=dtype)
            for name, dtype in ARRAY_DTYPES.items()
        }


@dataclass(frozen=True)
class ObservedStructureGraphArtifact:
    """Represent one validated graph candidate awaiting manual approval."""

    path: Path
    manifest: dict[str, Any]
    arrays: StructureGraphArrays


def _canonical_json(value: Any) -> bytes:
    """Encode identity fields deterministically without non-JSON floats."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    """Hash one artifact file without loading it entirely into memory."""

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
    """Hash all graph arrays in stable scientific field order."""

    digest = hashlib.sha256()
    for name in ARRAY_DTYPES:
        digest.update(name.encode("utf-8"))
        digest.update(_sha256_array(arrays[name]).encode("ascii"))
    return digest.hexdigest()


def _identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select path-independent fields that define one graph candidate."""

    return {
        "format": GRAPH_FORMAT,
        "version": GRAPH_VERSION,
        "static_scene_identity": manifest["static_scene_identity"],
        "foreground_identity": manifest["foreground_identity"],
        "topology_identity": manifest["topology_identity"],
        "views": manifest["views"],
        "parameters": manifest["parameters"],
        "semantics": manifest["semantics"],
        "thresholds": manifest["thresholds"],
        "counts": manifest["counts"],
        "arrays_identity": manifest["arrays_identity"],
    }


def _require_ckdtree() -> Any:
    """Load SciPy cKDTree without relying on incomplete editor stubs."""

    try:
        spatial = importlib.import_module("scipy.spatial")
    except ImportError as error:
        raise RuntimeError("Observed graph construction requires scipy") from error
    tree_type = getattr(spatial, "cKDTree", None)
    if tree_type is None:
        raise RuntimeError("scipy.spatial.cKDTree is unavailable")
    return tree_type


def _observed_nodes(
    topology: TopologyArrays,
    foreground_count: int,
    view_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive globally indexed observed nodes and their positive view support."""

    contributor_counts = np.diff(topology.sample_offsets)
    contributor_samples = np.repeat(
        np.arange(len(topology.sample_view_index), dtype=np.int64),
        contributor_counts,
    )
    if len(contributor_samples) != len(topology.contributor_gaussian_index):
        raise ValueError("Topology contributor offsets are inconsistent")
    contributor_views = topology.sample_view_index[contributor_samples]
    point_view_mask = np.zeros((foreground_count, view_count), dtype=bool)
    point_view_mask[
        topology.contributor_gaussian_index,
        contributor_views,
    ] = True
    node_indices = np.flatnonzero(point_view_mask.any(axis=1)).astype(np.int64)
    if len(node_indices) == 0:
        raise ValueError("Observation topology contains no observed foreground Gaussian")
    return node_indices, point_view_mask[node_indices]


def _query_deterministic_knn(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Query K nearest neighbors with stable distance/index tie breaking."""

    tree = _require_ckdtree()(np.asarray(points, dtype=np.float64))
    distances, _ = tree.query(points, k=count + 1)
    distances = np.asarray(distances, dtype=np.float64)
    boundaries = np.nextafter(distances[:, -1], np.inf)
    candidate_rows = tree.query_ball_point(points, boundaries, return_sorted=False)
    neighbors = np.empty((len(points), count), dtype=np.int64)
    neighbor_distances = np.empty((len(points), count), dtype=np.float64)
    for point_index, row in enumerate(candidate_rows):
        row_indices = np.asarray(row, dtype=np.int64)
        row_indices = row_indices[row_indices != point_index]
        row_distances = np.linalg.norm(points[row_indices] - points[point_index], axis=1)
        if len(row_indices) < count:
            raise RuntimeError("KNN query returned too few non-self candidates")
        order = np.lexsort((row_indices, row_distances))[:count]
        neighbors[point_index] = row_indices[order]
        neighbor_distances[point_index] = row_distances[order]
    return neighbors, neighbor_distances


def _mutual_knn_edges(
    points: np.ndarray,
    max_neighbors: int,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build lexicographically ordered mutual-KNN edges within max distance."""

    point_count = len(points)
    if point_count < 2:
        empty_edges = np.empty((0, 2), dtype=np.int64)
        empty_distance = np.empty((0,), dtype=np.float32)
        return empty_edges, empty_distance, {
            "knn_directed_candidates": 0,
            "distance_rejected_directed": 0,
            "nonmutual_rejected_pairs": 0,
            "mutual_distance_candidates": 0,
        }
    neighbor_count = min(max_neighbors, point_count - 1)
    neighbors, distances = _query_deterministic_knn(points, neighbor_count)
    source = np.repeat(np.arange(point_count, dtype=np.int64), neighbor_count)
    target = neighbors.reshape(-1)
    distance = distances.reshape(-1)
    directed_total = len(source)
    within = np.isfinite(distance) & (distance <= max_distance)
    source, target, distance = source[within], target[within], distance[within]
    directed_codes = set((source * point_count + target).tolist())
    mutual = np.asarray(
        [target_value * point_count + source_value in directed_codes
         for source_value, target_value in zip(source, target)],
        dtype=bool,
    )
    lower = mutual & (source < target)
    edges = np.column_stack([source[lower], target[lower]]).astype(np.int64)
    edge_distance = distance[lower].astype(np.float32)
    if len(edges):
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges, edge_distance = edges[order], edge_distance[order]
    within_pairs = {
        (min(int(start), int(end)), max(int(start), int(end)))
        for start, end in zip(source, target)
    }
    return edges, edge_distance, {
        "knn_directed_candidates": directed_total,
        "distance_rejected_directed": directed_total - len(source),
        "nonmutual_rejected_pairs": len(within_pairs) - len(edges),
        "mutual_distance_candidates": len(edges),
    }


def _robust_threshold(values: np.ndarray, multiplier: float) -> tuple[float, float, float]:
    """Return median, MAD, and the accepted scaled-MAD upper threshold."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return math.nan, math.nan, math.nan
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median, mad, median + multiplier * MAD_SCALE * mad


def _threshold_pass(values: np.ndarray, threshold: float, epsilon: float) -> np.ndarray:
    """Apply one finite upper threshold with exact-zero handling."""

    values = np.asarray(values, dtype=np.float64)
    if not math.isfinite(threshold):
        return np.zeros(values.shape, dtype=bool)
    if threshold == 0.0:
        return np.isfinite(values) & (values == 0.0)
    return np.isfinite(values) & (values <= threshold + epsilon)


def _soft_weight(values: np.ndarray, threshold: float, epsilon: float) -> np.ndarray:
    """Convert accepted distances to the old Gaussian confidence weight."""

    values = np.asarray(values, dtype=np.float64)
    if threshold == 0.0:
        return np.where(values == 0.0, 1.0, 0.0).astype(np.float32)
    return np.exp(-0.5 * np.square(values / max(threshold, epsilon))).astype(
        np.float32
    )


def _project_points(
    points: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project normalized-world points and return image xy plus camera z."""

    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [values, np.ones((len(values), 1), dtype=np.float64)], axis=1
    )
    camera = homogeneous @ np.asarray(world_to_camera, dtype=np.float64).T
    z = camera[:, 2]
    x = float(K[0, 0]) * camera[:, 0] / z + float(K[0, 2])
    y = float(K[1, 1]) * camera[:, 1] / z + float(K[1, 2])
    return np.stack([x, y], axis=1).astype(np.float32), z.astype(np.float32)


def _bilinear_valid(pixels: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return whether floating xy coordinates have a full bilinear footprint."""

    values = np.asarray(pixels, dtype=np.float64)
    return (
        np.isfinite(values).all(axis=-1)
        & (values[..., 0] >= 0.0)
        & (values[..., 1] >= 0.0)
        & (values[..., 0] < width - 1)
        & (values[..., 1] < height - 1)
    )


def _bilinear_sample(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Sample one 2D image at already validated floating xy coordinates."""

    values = np.asarray(pixels, dtype=np.float64)
    x0 = np.floor(values[:, 0]).astype(np.int64)
    y0 = np.floor(values[:, 1]).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx = (values[:, 0] - x0).astype(np.float32)
    wy = (values[:, 1] - y0).astype(np.float32)
    source = np.asarray(image)
    return (
        (1.0 - wx) * (1.0 - wy) * source[y0, x0]
        + wx * (1.0 - wy) * source[y0, x1]
        + (1.0 - wx) * wy * source[y1, x0]
        + wx * wy * source[y1, x1]
    )


def _sample_valid(image: np.ndarray, pixels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Sample valid pixels and fill invalid rows with NaN."""

    result = np.full(valid.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        result[valid] = _bilinear_sample(image, pixels[valid]).astype(np.float32)
    return result


def _component_data(
    node_count: int,
    edge_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute stable connected components, sizes, degree, and isolation."""

    degree = np.zeros(node_count, dtype=np.int64)
    if len(edge_index):
        np.add.at(degree, edge_index[:, 0], 1)
        np.add.at(degree, edge_index[:, 1], 1)
    parent = np.arange(node_count, dtype=np.int64)

    def find(index: int) -> int:
        """Return and path-compress one disjoint-set root."""

        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for start, end in edge_index:
        root_start, root_end = find(int(start)), find(int(end))
        if root_start != root_end:
            parent[max(root_start, root_end)] = min(root_start, root_end)
    roots = np.asarray([find(index) for index in range(node_count)], dtype=np.int64)
    _, component_index = np.unique(roots, return_inverse=True)
    component_index = component_index.astype(np.int64)
    component_size = np.bincount(component_index).astype(np.int64)
    return degree, component_index, component_size, degree == 0


def build_structure_graph_arrays(
    *,
    foreground_means: np.ndarray,
    foreground_colors: np.ndarray,
    topology: TopologyArrays,
    Ks: np.ndarray,
    world_to_cameras: np.ndarray,
    rendered_depths: Sequence[np.ndarray],
    rendered_alphas: Sequence[np.ndarray],
    render_alpha_minimum: float,
    config: ObservedStructureGraphConfig,
) -> tuple[StructureGraphArrays, dict[str, Any]]:
    """Build the accepted color/depth-filtered mutual-KNN graph in memory."""

    means = np.asarray(foreground_means, dtype=np.float32)
    colors = np.asarray(foreground_colors, dtype=np.float32)
    intrinsics = np.asarray(Ks, dtype=np.float32)
    extrinsics = np.asarray(world_to_cameras, dtype=np.float32)
    view_count = len(rendered_depths)
    config.validate(view_count)
    if means.ndim != 2 or means.shape[1] != 3 or len(means) < 2:
        raise ValueError("Foreground means must be finite [G,3] with G at least two")
    if not np.isfinite(means).all():
        raise ValueError("Foreground means contain NaN or Inf")
    if colors.shape != means.shape or not np.isfinite(colors).all():
        raise ValueError("Foreground colors must be finite [G,3]")
    if np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("Foreground colors must lie in [0,1]")
    if intrinsics.shape != (view_count, 3, 3):
        raise ValueError("Observed graph intrinsics have an invalid shape")
    if extrinsics.shape != (view_count, 4, 4):
        raise ValueError("Observed graph camera poses have an invalid shape")
    if len(rendered_alphas) != view_count or view_count == 0:
        raise ValueError("Observed graph depth/alpha view counts do not match")
    if not math.isfinite(render_alpha_minimum) or not (
        0.0 <= render_alpha_minimum <= 1.0
    ):
        raise ValueError("Observed graph render alpha minimum must lie in [0,1]")

    node_indices, node_view_mask = _observed_nodes(
        topology,
        foreground_count=len(means),
        view_count=view_count,
    )
    node_points = means[node_indices]
    node_colors = colors[node_indices]
    candidate_edges, candidate_distances, knn_counts = _mutual_knn_edges(
        node_points,
        config.max_neighbors,
        config.max_distance,
    )
    candidate_count = len(candidate_edges)

    lab = cv2.cvtColor(
        node_colors.reshape(-1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3)
    color_distances = (
        np.linalg.norm(
            lab[candidate_edges[:, 0]].astype(np.float64)
            - lab[candidate_edges[:, 1]].astype(np.float64),
            axis=1,
        ).astype(np.float32)
        if candidate_count
        else np.empty((0,), dtype=np.float32)
    )
    color_median, color_mad, color_threshold = _robust_threshold(
        color_distances,
        config.color_mad_multiplier,
    )
    color_pass = _threshold_pass(color_distances, color_threshold, 1.0e-8)

    endpoint_gap = np.full(
        (candidate_count, view_count), np.nan, dtype=np.float32
    )
    depth_jump = np.full(
        (candidate_count, view_count), np.nan, dtype=np.float32
    )
    raw_depth_valid = np.zeros((candidate_count, view_count), dtype=bool)
    endpoint_thresholds = np.full((view_count, 3), np.nan, dtype=np.float64)
    jump_thresholds = np.full((view_count, 3), np.nan, dtype=np.float64)
    endpoint_gap_per_node = np.full(
        (len(node_indices), view_count), np.nan, dtype=np.float32
    )
    projected_pixels: list[np.ndarray] = []
    endpoint_surface_valid: list[np.ndarray] = []

    for view_index in range(view_count):
        depth = np.asarray(rendered_depths[view_index], dtype=np.float32)
        alpha = np.asarray(rendered_alphas[view_index], dtype=np.float32)
        if depth.ndim != 2 or alpha.shape != depth.shape:
            raise ValueError("Rendered foreground depth and alpha must be matching 2D arrays")
        if not np.isfinite(depth).all() or not np.isfinite(alpha).all():
            raise ValueError("Rendered foreground depth or alpha contains NaN or Inf")
        pixels, camera_z = _project_points(
            node_points,
            intrinsics[view_index],
            extrinsics[view_index],
        )
        pixel_valid = _bilinear_valid(pixels, depth.shape[0], depth.shape[1])
        sampled_depth = _sample_valid(depth, pixels, pixel_valid)
        sampled_alpha = _sample_valid(alpha, pixels, pixel_valid)
        surface_valid = (
            node_view_mask[:, view_index]
            & pixel_valid
            & np.isfinite(camera_z)
            & (camera_z > 0.0)
            & np.isfinite(sampled_depth)
            & (sampled_depth > 0.0)
            & np.isfinite(sampled_alpha)
            & (sampled_alpha >= render_alpha_minimum)
        )
        gaps = np.full((len(node_indices),), np.nan, dtype=np.float32)
        gaps[surface_valid] = (
            np.abs(camera_z[surface_valid] - sampled_depth[surface_valid])
            / np.maximum(np.abs(sampled_depth[surface_valid]), 1.0e-8)
        ).astype(np.float32)
        endpoint_gap_per_node[:, view_index] = gaps
        projected_pixels.append(pixels)
        endpoint_surface_valid.append(surface_valid)

    line_fraction = np.linspace(
        0.0,
        1.0,
        config.depth_samples,
        dtype=np.float32,
    )
    for view_index in range(view_count):
        if candidate_count == 0:
            continue
        depth = np.asarray(rendered_depths[view_index], dtype=np.float32)
        alpha = np.asarray(rendered_alphas[view_index], dtype=np.float32)
        start, end = candidate_edges[:, 0], candidate_edges[:, 1]
        common = node_view_mask[start, view_index] & node_view_mask[end, view_index]
        endpoint_valid = (
            endpoint_surface_valid[view_index][start]
            & endpoint_surface_valid[view_index][end]
        )
        endpoint_gap[:, view_index] = np.maximum(
            endpoint_gap_per_node[start, view_index],
            endpoint_gap_per_node[end, view_index],
        )
        for batch_start in range(0, candidate_count, PROFILE_BATCH_SIZE):
            batch_end = min(batch_start + PROFILE_BATCH_SIZE, candidate_count)
            rows = np.arange(batch_start, batch_end, dtype=np.int64)
            possible = common[rows] & endpoint_valid[rows]
            if not np.any(possible):
                continue
            possible_rows = rows[possible]
            point_start = projected_pixels[view_index][start[possible_rows]]
            point_end = projected_pixels[view_index][end[possible_rows]]
            line_pixels = (
                point_start[:, None, :] * (1.0 - line_fraction[None, :, None])
                + point_end[:, None, :] * line_fraction[None, :, None]
            )
            profile_valid = _bilinear_valid(
                line_pixels, depth.shape[0], depth.shape[1]
            ).all(axis=1)
            if not np.any(profile_valid):
                continue
            valid_rows = possible_rows[profile_valid]
            valid_pixels = line_pixels[profile_valid]
            sampled_depth = _bilinear_sample(
                depth, valid_pixels.reshape(-1, 2)
            ).reshape(-1, config.depth_samples)
            sampled_alpha = _bilinear_sample(
                alpha, valid_pixels.reshape(-1, 2)
            ).reshape(-1, config.depth_samples)
            samples_valid = (
                np.isfinite(sampled_depth).all(axis=1)
                & (sampled_depth > 0.0).all(axis=1)
                & np.isfinite(sampled_alpha).all(axis=1)
                & (sampled_alpha >= render_alpha_minimum).all(axis=1)
            )
            valid_rows = valid_rows[samples_valid]
            sampled_depth = sampled_depth[samples_valid]
            if len(valid_rows) == 0:
                continue
            denominator = np.maximum(
                0.5
                * (
                    np.abs(sampled_depth[:, :-1])
                    + np.abs(sampled_depth[:, 1:])
                ),
                1.0e-8,
            )
            jumps = np.max(
                np.abs(np.diff(sampled_depth, axis=1)) / denominator,
                axis=1,
            )
            depth_jump[valid_rows, view_index] = jumps.astype(np.float32)
            raw_depth_valid[valid_rows, view_index] = True
        endpoint_thresholds[view_index] = _robust_threshold(
            endpoint_gap[raw_depth_valid[:, view_index], view_index],
            config.depth_mad_multiplier,
        )
        jump_thresholds[view_index] = _robust_threshold(
            depth_jump[raw_depth_valid[:, view_index], view_index],
            config.depth_mad_multiplier,
        )

    view_support = np.zeros((candidate_count, view_count), dtype=bool)
    view_depth_score = np.zeros((candidate_count, view_count), dtype=np.float32)
    endpoint_pass_all = np.zeros((candidate_count, view_count), dtype=bool)
    jump_pass_all = np.zeros((candidate_count, view_count), dtype=bool)
    for view_index in range(view_count):
        endpoint_pass = _threshold_pass(
            endpoint_gap[:, view_index],
            float(endpoint_thresholds[view_index, 2]),
            1.0e-8,
        )
        jump_pass = _threshold_pass(
            depth_jump[:, view_index],
            float(jump_thresholds[view_index, 2]),
            1.0e-8,
        )
        endpoint_pass_all[:, view_index] = endpoint_pass
        jump_pass_all[:, view_index] = jump_pass
        support = raw_depth_valid[:, view_index] & endpoint_pass & jump_pass
        view_support[:, view_index] = support
        if np.any(support):
            endpoint_weight = _soft_weight(
                endpoint_gap[support, view_index],
                float(endpoint_thresholds[view_index, 2]),
                1.0e-8,
            )
            jump_weight = _soft_weight(
                depth_jump[support, view_index],
                float(jump_thresholds[view_index, 2]),
                1.0e-8,
            )
            view_depth_score[support, view_index] = endpoint_weight * jump_weight

    support_count = view_support.sum(axis=1).astype(np.int64)
    depth_pass = support_count >= config.min_shared_views
    retained = color_pass & depth_pass
    final_edges = candidate_edges[retained]
    final_distances = candidate_distances[retained]
    final_colors = color_distances[retained]
    final_support = view_support[retained]
    final_support_count = support_count[retained]
    final_endpoint_gap = endpoint_gap[retained]
    final_depth_jump = depth_jump[retained]
    final_depth_score = np.divide(
        view_depth_score[retained].sum(axis=1),
        final_support_count,
        out=np.zeros(final_support_count.shape, dtype=np.float32),
        where=final_support_count > 0,
    ).astype(np.float32)

    _, pre_component, pre_size, _ = _component_data(len(node_indices), final_edges)
    pre_edge_count = np.zeros(len(pre_size), dtype=np.int64)
    if len(final_edges):
        pre_edge_component = pre_component[final_edges[:, 0]]
        pre_edge_count = np.bincount(
            pre_edge_component, minlength=len(pre_size)
        ).astype(np.int64)
    else:
        pre_edge_component = np.empty((0,), dtype=np.int64)
    connected_component = pre_edge_count > 0
    pruned_component = connected_component & (
        (pre_size < config.min_component_nodes)
        | (pre_edge_count < config.min_component_edges)
    )
    pruned_node_mask = pruned_component[pre_component]
    retained_component_edge = ~pruned_component[pre_edge_component]
    pruned_edge_count = int(np.count_nonzero(~retained_component_edge))
    final_edges = final_edges[retained_component_edge]
    final_distances = final_distances[retained_component_edge]
    final_colors = final_colors[retained_component_edge]
    final_support = final_support[retained_component_edge]
    final_endpoint_gap = final_endpoint_gap[retained_component_edge]
    final_depth_jump = final_depth_jump[retained_component_edge]
    final_depth_score = final_depth_score[retained_component_edge]

    distance_weight = (
        1.0 / np.maximum(final_distances, 1.0e-8)
    ).astype(np.float32)
    color_weight = _soft_weight(final_colors, color_threshold, 1.0e-8)
    combined_weight = (
        distance_weight * color_weight * final_depth_score
    ).astype(np.float32)
    degree, component_index, component_size, isolated = _component_data(
        len(node_indices), final_edges
    )
    if np.any(pruned_node_mask & ~isolated):
        raise RuntimeError("Component pruning did not isolate every rejected node")

    candidate_common_view = (
        node_view_mask[candidate_edges[:, 0]]
        & node_view_mask[candidate_edges[:, 1]]
        if candidate_count
        else np.empty((0, view_count), dtype=bool)
    )
    counts = {
        **knn_counts,
        "foreground_gaussians": len(means),
        "topology_samples": len(topology.sample_view_index),
        "topology_contributors": len(topology.contributor_gaussian_index),
        "node_count": len(node_indices),
        "single_view_nodes": int(np.count_nonzero(node_view_mask.sum(axis=1) == 1)),
        "multi_view_nodes": int(np.count_nonzero(node_view_mask.sum(axis=1) >= 2)),
        "shared_observed_candidate_views": int(np.count_nonzero(candidate_common_view)),
        "raw_depth_valid_candidate_views": int(np.count_nonzero(raw_depth_valid)),
        "endpoint_rejected_candidate_views": int(
            np.count_nonzero(raw_depth_valid & ~endpoint_pass_all)
        ),
        "jump_rejected_candidate_views": int(
            np.count_nonzero(raw_depth_valid & endpoint_pass_all & ~jump_pass_all)
        ),
        "supporting_candidate_views": int(np.count_nonzero(view_support)),
        "color_rejected_edges": int(np.count_nonzero(~color_pass)),
        "depth_rejected_edges": int(np.count_nonzero(color_pass & ~depth_pass)),
        "retained_edges": len(final_edges),
        "components": len(component_size),
        "isolated_nodes": int(np.count_nonzero(isolated)),
        "pruned_components": int(np.count_nonzero(pruned_component)),
        "pruned_nodes": int(np.count_nonzero(pruned_node_mask)),
        "pruned_edges": pruned_edge_count,
    }
    thresholds = {
        "color": {
            "median": color_median,
            "mad": color_mad,
            "threshold": color_threshold,
        },
        "endpoint_gap_by_view": endpoint_thresholds,
        "depth_jump_by_view": jump_thresholds,
    }
    arrays = StructureGraphArrays(
        node_gaussian_index=node_indices,
        node_observed_view_mask=node_view_mask,
        edge_index=final_edges,
        edge_distance=final_distances,
        edge_color_distance=final_colors,
        edge_depth_score=final_depth_score,
        edge_weight=combined_weight,
        edge_view_support_mask=final_support,
        edge_endpoint_gap_by_view=final_endpoint_gap,
        edge_depth_jump_by_view=final_depth_jump,
        degree=degree,
        component_index=component_index,
        component_size=component_size,
        component_pruned_node_mask=pruned_node_mask,
        isolated_mask=isolated,
    )
    return arrays, {"counts": counts, "thresholds": thresholds}


def _json_triplet(values: Sequence[float]) -> dict[str, float | None]:
    """Convert one median/MAD/threshold triplet to strict JSON numbers."""

    names = ("median", "mad", "threshold")
    return {
        name: float(value) if math.isfinite(float(value)) else None
        for name, value in zip(names, values)
    }


def _threshold_manifest(
    diagnostics: Mapping[str, Any],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Serialize adaptive thresholds with explicit per-view labels."""

    thresholds = diagnostics["thresholds"]
    color = thresholds["color"]
    endpoint = np.asarray(thresholds["endpoint_gap_by_view"], dtype=np.float64)
    jump = np.asarray(thresholds["depth_jump_by_view"], dtype=np.float64)
    if endpoint.shape != (len(labels), 3) or jump.shape != (len(labels), 3):
        raise ValueError("Observed graph threshold diagnostics have invalid shapes")
    return {
        "color": _json_triplet(
            [color["median"], color["mad"], color["threshold"]]
        ),
        "views": [
            {
                "index": index,
                "label": label,
                "endpoint_gap": _json_triplet(endpoint[index]),
                "depth_jump": _json_triplet(jump[index]),
            }
            for index, label in enumerate(labels)
        ],
    }


def _validate_threshold_triplet(
    payload: Any,
    multiplier: float,
    name: str,
) -> None:
    """Validate a finite or entirely unavailable robust-threshold triplet."""

    if not isinstance(payload, dict) or set(payload) != {"median", "mad", "threshold"}:
        raise ValueError(f"Observed graph {name} threshold metadata is invalid")
    values = [payload[key] for key in ("median", "mad", "threshold")]
    if all(value is None for value in values):
        return
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        raise ValueError(f"Observed graph {name} thresholds must be finite and non-negative")
    expected = float(values[0]) + multiplier * MAD_SCALE * float(values[1])
    if not math.isclose(float(values[2]), expected, rel_tol=1.0e-6, abs_tol=1.0e-8):
        raise ValueError(f"Observed graph {name} threshold formula does not match")


def _config_from_manifest(manifest: Mapping[str, Any], view_count: int) -> tuple[ObservedStructureGraphConfig, float]:
    """Reconstruct and validate the graph settings stored in a manifest."""

    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Observed graph parameters are missing")
    integer_names = (
        "max_neighbors",
        "depth_samples",
        "min_shared_views",
        "min_component_nodes",
        "min_component_edges",
    )
    if any(
        isinstance(parameters.get(name), bool)
        or not isinstance(parameters.get(name), int)
        for name in integer_names
    ):
        raise ValueError("Observed graph integer parameters are invalid")
    config = ObservedStructureGraphConfig(
        max_neighbors=int(parameters["max_neighbors"]),
        max_distance=float(parameters["max_distance"]),
        color_mad_multiplier=float(parameters["color_mad_multiplier"]),
        depth_mad_multiplier=float(parameters["depth_mad_multiplier"]),
        depth_samples=int(parameters["depth_samples"]),
        min_shared_views=int(parameters["min_shared_views"]),
        min_component_nodes=int(parameters["min_component_nodes"]),
        min_component_edges=int(parameters["min_component_edges"]),
    )
    config.validate(view_count)
    render_alpha_minimum = float(parameters["render_alpha_minimum"])
    if not math.isfinite(render_alpha_minimum) or not (
        0.0 <= render_alpha_minimum <= 1.0
    ):
        raise ValueError("Observed graph render alpha minimum is invalid")
    if float(parameters.get("epsilon", math.nan)) != 1.0e-8:
        raise ValueError("Observed graph epsilon convention is unsupported")
    if float(parameters.get("mad_scale", math.nan)) != MAD_SCALE:
        raise ValueError("Observed graph MAD scale convention is unsupported")
    return config, render_alpha_minimum


def _validate_manifest_thresholds(
    manifest: Mapping[str, Any],
    labels: Sequence[str],
    config: ObservedStructureGraphConfig,
) -> None:
    """Validate color and per-view adaptive threshold diagnostics."""

    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != {"color", "views"}:
        raise ValueError("Observed graph threshold metadata is invalid")
    _validate_threshold_triplet(
        thresholds["color"], config.color_mad_multiplier, "color"
    )
    views = thresholds["views"]
    if not isinstance(views, list) or len(views) != len(labels):
        raise ValueError("Observed graph per-view thresholds are invalid")
    for index, (label, view) in enumerate(zip(labels, views)):
        if (
            not isinstance(view, dict)
            or view.get("index") != index
            or view.get("label") != label
        ):
            raise ValueError("Observed graph threshold view order is invalid")
        _validate_threshold_triplet(
            view.get("endpoint_gap"),
            config.depth_mad_multiplier,
            f"{label} endpoint gap",
        )
        _validate_threshold_triplet(
            view.get("depth_jump"),
            config.depth_mad_multiplier,
            f"{label} depth jump",
        )


def _validate_graph_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    view_count: int,
    foreground_count: int,
    config: ObservedStructureGraphConfig,
    counts: Mapping[str, Any],
) -> None:
    """Validate graph index domains, edge metadata, and component invariants."""

    if set(arrays) != set(ARRAY_DTYPES):
        raise ValueError(f"Observed graph array inventory is invalid: {sorted(arrays)}")
    for name, dtype in ARRAY_DTYPES.items():
        if arrays[name].dtype != dtype:
            raise ValueError(f"Observed graph {name} must be {dtype.name}")
    nodes = arrays["node_gaussian_index"]
    node_views = arrays["node_observed_view_mask"]
    edges = arrays["edge_index"]
    edge_count = len(edges)
    node_count = len(nodes)
    if nodes.ndim != 1 or node_count == 0:
        raise ValueError("Observed graph must contain a 1D non-empty node index")
    if (
        np.any(nodes < 0)
        or np.any(nodes >= foreground_count)
        or not np.array_equal(nodes, np.unique(nodes))
    ):
        raise ValueError("Observed graph node Gaussian indices are invalid")
    if node_views.shape != (node_count, view_count) or not node_views.any(axis=1).all():
        raise ValueError("Observed graph node view support is invalid")
    if edges.shape != (edge_count, 2):
        raise ValueError("Observed graph edge_index must be [E,2]")
    if edge_count:
        if (
            np.any(edges[:, 0] < 0)
            or np.any(edges[:, 1] >= node_count)
            or np.any(edges[:, 0] >= edges[:, 1])
        ):
            raise ValueError("Observed graph edge index domain is invalid")
        expected_order = np.lexsort((edges[:, 1], edges[:, 0]))
        if not np.array_equal(expected_order, np.arange(edge_count)):
            raise ValueError("Observed graph edges are not lexicographically ordered")
        if np.any(np.all(edges[1:] == edges[:-1], axis=1)):
            raise ValueError("Observed graph contains duplicate edges")
    scalar_edge_fields = (
        "edge_distance",
        "edge_color_distance",
        "edge_depth_score",
        "edge_weight",
    )
    for name in scalar_edge_fields:
        if arrays[name].shape != (edge_count,):
            raise ValueError(f"Observed graph {name} must have shape [E]")
    for name in (
        "edge_view_support_mask",
        "edge_endpoint_gap_by_view",
        "edge_depth_jump_by_view",
    ):
        if arrays[name].shape != (edge_count, view_count):
            raise ValueError(f"Observed graph {name} must have shape [E,V]")
    distance = arrays["edge_distance"]
    color = arrays["edge_color_distance"]
    depth_score = arrays["edge_depth_score"]
    weight = arrays["edge_weight"]
    if (
        not np.isfinite(distance).all()
        or np.any(distance < 0.0)
        or np.any(distance > config.max_distance + 1.0e-7)
        or not np.isfinite(color).all()
        or np.any(color < 0.0)
        or not np.isfinite(depth_score).all()
        or np.any(depth_score <= 0.0)
        or np.any(depth_score > 1.0)
        or not np.isfinite(weight).all()
        or np.any(weight <= 0.0)
    ):
        raise ValueError("Observed graph retained edge values are invalid")
    support = arrays["edge_view_support_mask"]
    if np.any(support.sum(axis=1) < config.min_shared_views):
        raise ValueError("Observed graph edge has insufficient shared-view support")
    for name in ("edge_endpoint_gap_by_view", "edge_depth_jump_by_view"):
        values = arrays[name]
        if np.any(~np.isnan(values) & (~np.isfinite(values) | (values < 0.0))):
            raise ValueError(f"Observed graph {name} contains invalid values")
        if np.any(~np.isfinite(values[support])):
            raise ValueError(f"Observed graph {name} is missing a supporting-view value")
    degree, component, size, isolated = _component_data(node_count, edges)
    expected_node_shapes = {
        "degree": (node_count,),
        "component_index": (node_count,),
        "component_pruned_node_mask": (node_count,),
        "isolated_mask": (node_count,),
    }
    for name, shape in expected_node_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Observed graph {name} has an invalid shape")
    if arrays["component_size"].shape != size.shape:
        raise ValueError("Observed graph component_size has an invalid shape")
    if (
        not np.array_equal(arrays["degree"], degree)
        or not np.array_equal(arrays["component_index"], component)
        or not np.array_equal(arrays["component_size"], size)
        or not np.array_equal(arrays["isolated_mask"], isolated)
        or np.any(arrays["component_pruned_node_mask"] & ~isolated)
    ):
        raise ValueError("Observed graph component metadata is inconsistent")
    derived_counts = {
        "foreground_gaussians": foreground_count,
        "node_count": node_count,
        "single_view_nodes": int(np.count_nonzero(node_views.sum(axis=1) == 1)),
        "multi_view_nodes": int(np.count_nonzero(node_views.sum(axis=1) >= 2)),
        "retained_edges": edge_count,
        "components": len(size),
        "isolated_nodes": int(np.count_nonzero(isolated)),
        "pruned_nodes": int(np.count_nonzero(arrays["component_pruned_node_mask"])),
    }
    for name, value in derived_counts.items():
        if counts.get(name) != value:
            raise ValueError(f"Observed graph count {name} does not match arrays")


def load_observed_structure_graph(path: str | Path) -> ObservedStructureGraphArtifact:
    """Load and fully validate one observed structure graph candidate."""

    root = Path(path).expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    graph_path = root / GRAPH_FILENAME
    if not manifest_path.is_file() or not graph_path.is_file():
        raise FileNotFoundError(f"Incomplete observed structure graph: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != GRAPH_FORMAT:
        raise ValueError(f"Unsupported observed structure graph: {manifest_path}")
    if manifest.get("version") != GRAPH_VERSION:
        raise ValueError("Unsupported observed structure graph version")
    if manifest.get("semantics") != GRAPH_SEMANTICS:
        raise ValueError("Observed structure graph semantics are unsupported")
    gate = manifest.get("quality_gate")
    if gate != {"required": True, "status": "candidate_unapproved"}:
        raise ValueError("Observed graph must remain an unapproved immutable candidate")
    views = manifest.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("Observed graph must contain ordered views")
    labels: list[str] = []
    for index, view in enumerate(views):
        if not isinstance(view, dict) or view.get("index") != index:
            raise ValueError("Observed graph view indices are invalid")
        label = view.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("Observed graph view label is invalid")
        if any(
            not isinstance(view.get(name), str) or not view[name]
            for name in ("camera_identity", "flow_identity")
        ):
            raise ValueError(f"Observed graph identity metadata for {label!r} is invalid")
        shape = view.get("shape_hw")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 2
                for value in shape
            )
        ):
            raise ValueError(f"Observed graph shape metadata for {label!r} is invalid")
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise ValueError("Observed graph view labels must be unique")
    config, _ = _config_from_manifest(manifest, len(views))
    _validate_manifest_thresholds(manifest, labels, config)
    if manifest.get("graph_file") != GRAPH_FILENAME:
        raise ValueError("Observed graph filename metadata is invalid")
    if manifest.get("graph_file_sha256") != _sha256_file(graph_path):
        raise ValueError("Observed graph NPZ SHA-256 does not match manifest")
    with np.load(graph_path, allow_pickle=False) as archive:
        loaded = {name: archive[name] for name in archive.files}
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ):
        raise ValueError("Observed graph counts are invalid")
    foreground_count = int(counts.get("foreground_gaussians", -1))
    _validate_graph_arrays(
        loaded,
        view_count=len(views),
        foreground_count=foreground_count,
        config=config,
        counts=counts,
    )
    arrays_identity = _arrays_identity(loaded)
    if manifest.get("arrays_identity") != arrays_identity:
        raise ValueError("Observed graph array identity does not match")
    expected_identity = hashlib.sha256(
        _canonical_json(_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("observed_structure_graph_identity") != expected_identity:
        raise ValueError("Observed structure graph identity does not match its contents")
    return ObservedStructureGraphArtifact(
        root,
        manifest,
        StructureGraphArrays(**loaded),
    )


def _publish_graph(
    destination: Path,
    *,
    arrays: StructureGraphArrays,
    manifest: dict[str, Any],
) -> ObservedStructureGraphArtifact:
    """Atomically publish and reload one graph candidate directory."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Observed graph output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        values = arrays.as_dict()
        graph_path = temporary / GRAPH_FILENAME
        save_named_arrays(graph_path, values)
        manifest["arrays"] = {
            name: {"dtype": value.dtype.name, "shape": list(value.shape)}
            for name, value in values.items()
        }
        manifest["graph_file"] = GRAPH_FILENAME
        manifest["graph_file_sha256"] = _sha256_file(graph_path)
        manifest["arrays_identity"] = _arrays_identity(values)
        manifest["observed_structure_graph_identity"] = hashlib.sha256(
            _canonical_json(_identity_payload(manifest))
        ).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_observed_structure_graph(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Observed graph output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_observed_structure_graph(destination)


def build_observed_structure_graph_artifact(
    *,
    scene_dir: str | Path,
    topology_dir: str | Path,
    output_dir: str | Path,
    config: ObservedStructureGraphConfig | None = None,
    command: Sequence[str] = (),
) -> ObservedStructureGraphArtifact:
    """Render reference depth, build the graph, and publish an unapproved candidate."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Observed graph output already exists: {destination}")
    topology = load_observation_topology(topology_dir)
    settings = config or ObservedStructureGraphConfig()
    view_count = len(topology.manifest["views"])
    settings.validate(view_count)
    render_alpha_minimum = float(
        topology.manifest["parameters"]["foreground_alpha_minimum"]
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Observed graph construction requires CUDA foreground rendering")
    device = torch.device("cuda")
    scene_path = Path(scene_dir).expanduser().resolve(strict=True)
    scene = load_static_scene(scene_path, device)
    if scene.manifest is None:
        raise ValueError("Static scene has no manifest")
    if (
        scene.manifest["static_scene_identity"]
        != topology.manifest["static_scene_identity"]
        or scene.manifest["foreground_identity"]
        != topology.manifest["foreground_identity"]
    ):
        raise ValueError("Static scene and observation topology identities do not match")
    if scene.foreground.count != int(
        topology.manifest["counts"]["foreground_gaussians"]
    ):
        raise ValueError("Static scene foreground count does not match topology")

    reference_cameras = {
        camera.label: camera
        for camera in cameras_from_scene_manifest(scene.manifest)
        if camera.role == "reference" and camera.label is not None
    }
    labels = [view["label"] for view in topology.manifest["views"]]
    if any(label not in reference_cameras for label in labels):
        raise ValueError("Static scene is missing one or more topology reference cameras")
    cameras = []
    view_records: list[dict[str, Any]] = []
    rendered_depths: list[np.ndarray] = []
    rendered_alphas: list[np.ndarray] = []
    scene.eval()
    with torch.no_grad():
        for index, (label, topology_view) in enumerate(
            zip(labels, topology.manifest["views"])
        ):
            camera = reference_cameras[label]
            camera_record = camera.to_manifest_record()
            if camera_record["camera_identity"] != topology_view["camera_identity"]:
                raise ValueError(f"Reference camera identity for {label!r} does not match")
            if topology_view["shape_hw"] != [camera.height, camera.width]:
                raise ValueError(f"Reference camera shape for {label!r} does not match")
            rendered = scene.render(
                camera.to(device),
                composition="foreground",
                outputs=("alpha", "expected_depth"),
            )
            rendered_depths.append(
                rendered["expected_depth"].detach().cpu().numpy().astype(np.float32)
            )
            rendered_alphas.append(
                rendered["alpha"].detach().cpu().numpy().astype(np.float32)
            )
            cameras.append(camera)
            view_records.append(
                {
                    "index": index,
                    "label": label,
                    "camera_identity": topology_view["camera_identity"],
                    "flow_identity": topology_view["flow_identity"],
                    "shape_hw": [camera.height, camera.width],
                }
            )

    active = scene.foreground.active()
    arrays, diagnostics = build_structure_graph_arrays(
        foreground_means=active["means"].detach().cpu().numpy(),
        foreground_colors=active["colors"].detach().cpu().numpy(),
        topology=topology.arrays,
        Ks=np.stack(
            [camera.K.detach().cpu().numpy() for camera in cameras], axis=0
        ),
        world_to_cameras=np.stack(
            [camera.world_to_camera.detach().cpu().numpy() for camera in cameras],
            axis=0,
        ),
        rendered_depths=rendered_depths,
        rendered_alphas=rendered_alphas,
        render_alpha_minimum=render_alpha_minimum,
        config=settings,
    )
    counts = {name: int(value) for name, value in diagnostics["counts"].items()}
    values = arrays.as_dict()
    _validate_graph_arrays(
        values,
        view_count=view_count,
        foreground_count=scene.foreground.count,
        config=settings,
        counts=counts,
    )
    manifest = {
        "format": GRAPH_FORMAT,
        "version": GRAPH_VERSION,
        "producer": {
            "project_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": list(command),
        },
        "static_scene": str(scene_path),
        "static_scene_identity": scene.manifest["static_scene_identity"],
        "foreground_identity": scene.manifest["foreground_identity"],
        "topology": str(topology.path),
        "topology_identity": topology.manifest["topology_identity"],
        "views": view_records,
        "parameters": settings.to_dict(render_alpha_minimum),
        "semantics": dict(GRAPH_SEMANTICS),
        "thresholds": _threshold_manifest(diagnostics, labels),
        "counts": counts,
        "quality_gate": {"required": True, "status": "candidate_unapproved"},
    }
    return _publish_graph(destination, arrays=arrays, manifest=manifest)


__all__ = [
    "ObservedStructureGraphArtifact",
    "ObservedStructureGraphConfig",
    "StructureGraphArrays",
    "build_observed_structure_graph_artifact",
    "build_structure_graph_arrays",
    "load_observed_structure_graph",
]
