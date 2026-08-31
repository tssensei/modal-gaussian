"""Sequential rigid-component promotion and full-foreground motion fill."""

from __future__ import annotations

from collections import deque
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

import numpy as np

from modal_gaussians.numpy_io import save_named_arrays

from modal_gaussians import __version__
from modal_gaussians.measurements import load_gaussian_measurements
from modal_gaussians.rigid import RigidModesArtifact, load_rigid_modes
from modal_gaussians.static import load_static_scene
from modal_gaussians.structure_graph import (
    StructureGraphArrays,
    load_observed_structure_graph,
)
from modal_gaussians.synchronization import PreparedObservations, prepare_observations
from modal_gaussians.topology import load_observation_topology


EPSILON = 1.0e-8
CONVERGED_LSMR_CODES = frozenset({0, 1, 2, 4, 5})
COMPLETED_MODES_FORMAT = "modal_gaussians.completed_modes"
COMPLETED_MODES_VERSION = 1
COMPLETED_MODES_FILENAME = "completed_modes.npz"

SUPPORT_TRUSTED_RIGID = 0
SUPPORT_PROMOTED_RIGID = 1
SUPPORT_POINTWISE_FILL = 2
SUPPORT_UNRESOLVED = 3
SUPPORT_CLASS_NAMES = (
    "trusted_rigid",
    "promoted_single_view_rigid",
    "pointwise_knn_fill",
    "unresolved",
)


@dataclass(frozen=True)
class MotionFillConfig:
    """Hold the accepted sequential rigid motion-fill settings."""

    neighbors: int = 8
    max_distance: float = 0.008
    max_anchor_hops: int = 8
    observable_singular_ratio_minimum: float = 1.0e-2
    ray_direction_minimum_fraction: float = 0.8
    maximum_finite_drift: float = 2.0
    lsmr_atol: float = 1.0e-6
    lsmr_btol: float = 1.0e-6
    lsmr_conlim: float = 1.0e8
    graph_epsilon: float = 1.0e-8

    def validate(self) -> None:
        """Reject graph, component, or sparse-solver settings outside the mainline."""

        for name, value in (
            ("neighbors", self.neighbors),
            ("max_anchor_hops", self.max_anchor_hops),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Motion-fill {name} must be positive")
        for name, value in (
            ("max_distance", self.max_distance),
            ("graph_epsilon", self.graph_epsilon),
            ("lsmr_conlim", self.lsmr_conlim),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Motion-fill {name} must be finite and positive")
        for name, value in (
            ("observable_singular_ratio_minimum", self.observable_singular_ratio_minimum),
            ("ray_direction_minimum_fraction", self.ray_direction_minimum_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Motion-fill {name} must lie in [0,1]")
        if self.observable_singular_ratio_minimum <= 0.0:
            raise ValueError("Observable singular-ratio minimum must be positive")
        if not math.isfinite(self.maximum_finite_drift) or self.maximum_finite_drift < 0.0:
            raise ValueError("Maximum finite drift must be finite and non-negative")
        for name, value in (("lsmr_atol", self.lsmr_atol), ("lsmr_btol", self.lsmr_btol)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Motion-fill {name} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the one accepted sequential completion policy."""

        return {
            "method": "rigid_seed_partial_component_then_independent_gaussian_knn_lsmr",
            "neighbors": self.neighbors,
            "max_distance": self.max_distance,
            "max_anchor_hops": self.max_anchor_hops,
            "observable_singular_ratio_minimum": self.observable_singular_ratio_minimum,
            "ray_direction_minimum_fraction": self.ray_direction_minimum_fraction,
            "maximum_finite_drift": self.maximum_finite_drift,
            "lsmr_atol": self.lsmr_atol,
            "lsmr_btol": self.lsmr_btol,
            "lsmr_conlim": self.lsmr_conlim,
            "graph_epsilon": self.graph_epsilon,
        }


@dataclass(frozen=True)
class KnnGraph:
    """Store the full-foreground distance-pruned union-KNN graph."""

    edge_index: np.ndarray
    edge_distance: np.ndarray
    edge_weight: np.ndarray
    degree: np.ndarray
    component_index: np.ndarray
    component_size: np.ndarray
    isolated_mask: np.ndarray
    point_count: int
    candidate_directed_count: int
    retained_directed_count: int
    pruned_directed_count: int


@dataclass(frozen=True)
class SparseSolveMetadata:
    """Store one real-valued LSMR convergence summary."""

    performed: bool
    converged: bool
    stop_code: int
    iterations: int
    residual_norm: float
    normal_residual_norm: float
    matrix_norm: float
    condition_estimate: float
    solution_norm: float


@dataclass(frozen=True)
class SingleViewComponentResult:
    """Store promoted single-view component motion and diagnostics."""

    phi: np.ndarray
    component_fill_mask: np.ndarray
    component_completion_mask: np.ndarray
    component_anchor_mask: np.ndarray
    component_translation: np.ndarray
    component_rotation: np.ndarray
    component_first_order_relative_max: np.ndarray
    component_observable_rank: np.ndarray
    component_fill_nullity: np.ndarray
    component_ray_dominated_basis_count: np.ndarray
    component_trusted_knn_edge_count: np.ndarray
    component_cross_knn_edge_count: np.ndarray
    component_connected_to_trusted_mask: np.ndarray
    component_postfill_normalized_residual: np.ndarray
    component_ray_motion_rms: np.ndarray
    component_tangent_motion_rms: np.ndarray
    component_ray_motion_ratio: np.ndarray
    component_finite_drift_max: np.ndarray
    component_finite_drift_rejected_mask: np.ndarray
    point_fill_mask: np.ndarray
    real_solver: SparseSolveMetadata
    imaginary_solver: SparseSolveMetadata
    system_row_count: int
    system_column_count: int
    active_edge_count: int


@dataclass(frozen=True)
class SequentialMotionFillResult:
    """Store one mode's completed field, masks, and sparse-solver diagnostics."""

    phi: np.ndarray
    trusted_seed_mask: np.ndarray
    component_anchor_point_mask: np.ndarray
    completion_mask: np.ndarray
    completion_connected_to_anchor: np.ndarray
    unresolved_mask: np.ndarray
    support_class: np.ndarray
    hop_distance: np.ndarray
    point_residual: np.ndarray
    point_residual_valid_mask: np.ndarray
    single_view: SingleViewComponentResult
    pointwise_real_solver: SparseSolveMetadata
    pointwise_imaginary_solver: SparseSolveMetadata
    pointwise_system_row_count: int
    pointwise_system_column_count: int
    pointwise_active_edge_count: int


@dataclass(frozen=True)
class CompletedModesArtifact:
    """Represent one validated full-foreground completed modal-field artifact."""

    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _require_module(name: str) -> Any:
    """Import one SciPy module without relying on incomplete editor stubs."""

    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(f"Motion fill requires {name}") from error


def _stable_components(point_count: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label connected components deterministically by their smallest point index."""

    parent = np.arange(point_count, dtype=np.int64)

    def find(point: int) -> int:
        root = point
        while parent[root] != root:
            root = int(parent[root])
        while parent[point] != point:
            following = int(parent[point])
            parent[point] = root
            point = following
        return root

    for point_i, point_j in edges.tolist():
        root_i, root_j = find(int(point_i)), find(int(point_j))
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)
    roots = np.asarray([find(point) for point in range(point_count)], dtype=np.int64)
    unique = np.unique(roots)
    root_to_component = np.full(point_count, -1, dtype=np.int64)
    root_to_component[unique] = np.arange(len(unique), dtype=np.int64)
    component = root_to_component[roots].astype(np.int32)
    size = np.bincount(component, minlength=len(unique)).astype(np.int32)
    return component, size


def build_motion_fill_graph(
    points: np.ndarray,
    config: MotionFillConfig | None = None,
) -> KnnGraph:
    """Build the accepted deterministic union-KNN graph over all foreground points."""

    settings = config or MotionFillConfig()
    settings.validate()
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("Motion-fill points must be [G,3] with at least two points")
    if not np.isfinite(values).all():
        raise ValueError("Motion-fill points contain NaN or Inf")
    if settings.neighbors >= len(values):
        raise ValueError("Motion-fill neighbors must be smaller than point count")
    spatial = _require_module("scipy.spatial")
    tree_type = getattr(spatial, "cKDTree", None)
    if tree_type is None:
        raise RuntimeError("scipy.spatial.cKDTree is unavailable")
    tree = tree_type(values)
    distances, _ = tree.query(values, k=settings.neighbors + 1)
    boundaries = np.nextafter(np.asarray(distances)[:, -1], np.inf)
    candidate_rows = tree.query_ball_point(values, boundaries, return_sorted=False)
    neighbors = np.empty((len(values), settings.neighbors), dtype=np.int64)
    neighbor_distances = np.empty_like(neighbors, dtype=np.float64)
    for point, row in enumerate(candidate_rows):
        indices = np.asarray(row, dtype=np.int64)
        indices = indices[indices != point]
        row_distances = np.linalg.norm(values[indices] - values[point], axis=1)
        if len(indices) < settings.neighbors:
            raise RuntimeError(f"KNN query returned too few candidates for point {point}")
        order = np.lexsort((indices, row_distances))[: settings.neighbors]
        neighbors[point] = indices[order]
        neighbor_distances[point] = row_distances[order]
    source = np.repeat(np.arange(len(values), dtype=np.int64), settings.neighbors)
    target = neighbors.reshape(-1)
    distance = neighbor_distances.reshape(-1)
    candidate_count = len(source)
    keep = distance <= settings.max_distance
    retained_count = int(np.count_nonzero(keep))
    source, target, distance = source[keep], target[keep], distance[keep]
    if len(source):
        pairs = np.column_stack((np.minimum(source, target), np.maximum(source, target)))
        edge_index, first = np.unique(pairs, axis=0, return_index=True)
        edge_distance = distance[first]
    else:
        edge_index = np.empty((0, 2), dtype=np.int64)
        edge_distance = np.empty(0, dtype=np.float64)
    edge_weight = 1.0 / (edge_distance + settings.graph_epsilon)
    degree = np.zeros(len(values), dtype=np.int32)
    if len(edge_index):
        np.add.at(degree, edge_index[:, 0], 1)
        np.add.at(degree, edge_index[:, 1], 1)
    component, size = _stable_components(len(values), edge_index)
    return KnnGraph(
        edge_index=edge_index.astype(np.int64),
        edge_distance=edge_distance.astype(np.float64),
        edge_weight=edge_weight.astype(np.float64),
        degree=degree,
        component_index=component,
        component_size=size,
        isolated_mask=degree == 0,
        point_count=len(values),
        candidate_directed_count=candidate_count,
        retained_directed_count=retained_count,
        pruned_directed_count=candidate_count - retained_count,
    )


def _anchor_connectivity(graph: KnnGraph, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return graph connectivity and minimum hop distance from any fixed anchor."""

    anchor_mask = np.asarray(anchors, dtype=bool)
    if anchor_mask.shape != (graph.point_count,) or not np.any(anchor_mask):
        raise ValueError("Motion fill requires at least one correctly shaped anchor")
    component_has_anchor = np.zeros(len(graph.component_size), dtype=bool)
    component_has_anchor[graph.component_index[anchor_mask]] = True
    connected = component_has_anchor[graph.component_index]
    adjacency: list[list[int]] = [[] for _ in range(graph.point_count)]
    for point_i, point_j in graph.edge_index.tolist():
        adjacency[int(point_i)].append(int(point_j))
        adjacency[int(point_j)].append(int(point_i))
    hops = np.full(graph.point_count, -1, dtype=np.int32)
    queue: deque[int] = deque()
    for point in np.flatnonzero(anchor_mask).tolist():
        hops[point] = 0
        queue.append(int(point))
    while queue:
        point = queue.popleft()
        for neighbor in adjacency[point]:
            if hops[neighbor] < 0:
                hops[neighbor] = hops[point] + 1
                queue.append(neighbor)
    if not np.array_equal(hops >= 0, connected):
        raise RuntimeError("Motion-fill hop traversal disagrees with graph components")
    return connected, hops


def _run_lsmr(
    matrix: Any,
    target: np.ndarray,
    config: MotionFillConfig,
    *,
    component_stage: bool,
) -> tuple[np.ndarray, SparseSolveMetadata]:
    """Run one accepted real-valued LSMR system and validate its stop code."""

    sparse_linalg = _require_module("scipy.sparse.linalg")
    lsmr = getattr(sparse_linalg, "lsmr", None)
    if lsmr is None:
        raise RuntimeError("scipy.sparse.linalg.lsmr is unavailable")
    maxiter = max(1000, 4 * int(matrix.shape[1])) if component_stage else None
    solved = lsmr(
        matrix,
        np.asarray(target, dtype=np.float64),
        atol=config.lsmr_atol,
        btol=config.lsmr_btol,
        conlim=config.lsmr_conlim,
        maxiter=maxiter,
    )
    metadata = SparseSolveMetadata(
        performed=True,
        converged=int(solved[1]) in CONVERGED_LSMR_CODES,
        stop_code=int(solved[1]),
        iterations=int(solved[2]),
        residual_norm=float(solved[3]),
        normal_residual_norm=float(solved[4]),
        matrix_norm=float(solved[5]),
        condition_estimate=float(solved[6]),
        solution_norm=float(solved[7]),
    )
    if not metadata.converged:
        raise RuntimeError(f"Motion-fill LSMR did not converge: stop={metadata.stop_code}")
    return np.asarray(solved[0], dtype=np.float64), metadata


def _empty_solver(target: np.ndarray) -> SparseSolveMetadata:
    """Return the exact no-variable solver summary."""

    return SparseSolveMetadata(
        performed=False,
        converged=True,
        stop_code=0,
        iterations=0,
        residual_norm=float(np.linalg.norm(target)),
        normal_residual_norm=0.0,
        matrix_norm=0.0,
        condition_estimate=1.0,
        solution_norm=0.0,
    )


def _rigid_point_blocks(points: np.ndarray, centroid: np.ndarray, radius: float) -> np.ndarray:
    """Map one normalized complex twist to per-point 3D displacement."""

    centered = np.asarray(points, dtype=np.float64) - np.asarray(centroid, dtype=np.float64)
    blocks = np.zeros((len(centered), 3, 6), dtype=np.float64)
    blocks[:, :, :3] = np.eye(3, dtype=np.float64)[None]
    if radius > EPSILON:
        skew = np.zeros((len(centered), 3, 3), dtype=np.float64)
        skew[:, 0, 1] = -centered[:, 2]
        skew[:, 0, 2] = centered[:, 1]
        skew[:, 1, 0] = centered[:, 2]
        skew[:, 1, 2] = -centered[:, 0]
        skew[:, 2, 0] = -centered[:, 1]
        skew[:, 2, 1] = centered[:, 0]
        blocks[:, :, 3:] = -skew / radius
    return blocks


def _point_residuals(
    prepared: PreparedObservations,
    alphas: np.ndarray,
    identifiable: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute old-mainline RMS 2D residual per foreground Gaussian."""

    valid = identifiable[prepared.obs_view_index] & (prepared.obs_weights > 0.0)
    rows = np.flatnonzero(valid)
    sums = np.zeros(len(prepared.points), dtype=np.float64)
    counts = np.zeros(len(prepared.points), dtype=np.int32)
    if len(rows):
        projected = np.einsum(
            "oij,oj->oi",
            prepared.obs_jacobian[rows].astype(np.float32),
            phi[prepared.obs_point_index[rows]].astype(np.complex64),
        )
        prediction = alphas[prepared.obs_view_index[rows], None] * projected
        residual = np.linalg.norm(prepared.obs_y[rows] - prediction, axis=1)
        np.add.at(sums, prepared.obs_point_index[rows], residual.astype(np.float64) ** 2)
        np.add.at(counts, prepared.obs_point_index[rows], 1)
    valid_points = counts > 0
    output = np.zeros(len(prepared.points), dtype=np.float32)
    output[valid_points] = np.sqrt(sums[valid_points] / counts[valid_points]).astype(np.float32)
    return output, valid_points


def _component_finite_drift(
    points: np.ndarray,
    phi: np.ndarray,
    global_edges: np.ndarray,
    edge_component: np.ndarray,
    selected_components: np.ndarray,
    phase_samples: int,
    component_count: int,
) -> np.ndarray:
    """Measure maximum additive-playback edge drift for selected components."""

    output = np.zeros(component_count, dtype=np.float64)
    selected_edges = selected_components[edge_component]
    indices = np.flatnonzero(selected_edges)
    phase = np.linspace(0.0, 2.0 * np.pi, phase_samples, endpoint=False)
    cosine, sine = np.cos(phase)[None], np.sin(phase)[None]
    for start in range(0, len(indices), 32_768):
        batch = indices[start : start + 32_768]
        edges = global_edges[batch]
        base = points[edges[:, 1]] - points[edges[:, 0]]
        delta = phi[edges[:, 1]] - phi[edges[:, 0]]
        displacement = (
            np.real(delta)[:, :, None] * cosine[:, None]
            - np.imag(delta)[:, :, None] * sine[:, None]
        )
        base_length = np.linalg.norm(base, axis=1)
        deformed = np.linalg.norm(base[:, :, None] + displacement, axis=1)
        edge_maximum = np.max(
            np.abs(deformed - base_length[:, None])
            / np.maximum(base_length[:, None], EPSILON),
            axis=1,
        )
        np.maximum.at(output, edge_component[batch], edge_maximum)
    return output


def _single_view_component_fill(
    *,
    points: np.ndarray,
    prepared: PreparedObservations,
    alphas: np.ndarray,
    identifiable: np.ndarray,
    trusted_phi: np.ndarray,
    trusted_seed_mask: np.ndarray,
    point_component: np.ndarray,
    component_centroid: np.ndarray,
    component_radius: np.ndarray,
    component_supported_view_count: np.ndarray,
    component_retained_mask: np.ndarray,
    edge_component: np.ndarray,
    observed_graph: StructureGraphArrays,
    fill_graph: KnnGraph,
    first_order_rtol: float,
    phase_samples: int,
    config: MotionFillConfig,
) -> SingleViewComponentResult:
    """Promote finite-safe single-view rigid components through shared twist fill."""

    point_count = len(points)
    component_count = len(component_centroid)
    selected_mask = (component_supported_view_count == 1) & ~component_retained_mask
    selected_components = np.flatnonzero(selected_mask)
    empty_solver = _empty_solver(np.empty(0, dtype=np.float64))
    zero_component = np.zeros(component_count, dtype=np.float32)
    zero_component_bool = np.zeros(component_count, dtype=bool)
    zero_component_int = np.zeros(component_count, dtype=np.int32)
    zero_twist = np.zeros((component_count, 3), dtype=np.complex64)
    if len(selected_components) == 0:
        phi = np.zeros((point_count, 3), dtype=np.complex64)
        phi[trusted_seed_mask] = trusted_phi[trusted_seed_mask]
        return SingleViewComponentResult(
            phi=phi,
            component_fill_mask=zero_component_bool.copy(),
            component_completion_mask=zero_component_bool.copy(),
            component_anchor_mask=zero_component_bool.copy(),
            component_translation=zero_twist.copy(),
            component_rotation=zero_twist.copy(),
            component_first_order_relative_max=zero_component.copy(),
            component_observable_rank=zero_component_int.astype(np.int8),
            component_fill_nullity=zero_component_int.astype(np.int8),
            component_ray_dominated_basis_count=zero_component_int.astype(np.int8),
            component_trusted_knn_edge_count=zero_component_int.copy(),
            component_cross_knn_edge_count=zero_component_int.copy(),
            component_connected_to_trusted_mask=zero_component_bool.copy(),
            component_postfill_normalized_residual=zero_component.copy(),
            component_ray_motion_rms=zero_component.copy(),
            component_tangent_motion_rms=zero_component.copy(),
            component_ray_motion_ratio=zero_component.copy(),
            component_finite_drift_max=zero_component.copy(),
            component_finite_drift_rejected_mask=zero_component_bool.copy(),
            point_fill_mask=np.zeros(point_count, dtype=bool),
            real_solver=empty_solver,
            imaginary_solver=empty_solver,
            system_row_count=0,
            system_column_count=0,
            active_edge_count=0,
        )

    component_to_group = np.full(component_count, -1, dtype=np.int32)
    component_to_group[selected_components] = np.arange(len(selected_components), dtype=np.int32)
    selected_point_mask = (point_component >= 0) & selected_mask[
        np.maximum(point_component, 0)
    ]
    point_group = np.full(point_count, -1, dtype=np.int32)
    point_group[selected_point_mask] = component_to_group[point_component[selected_point_mask]]
    member_points = np.flatnonzero(selected_point_mask)
    member_order = np.argsort(point_group[member_points], kind="stable")
    ordered_members = member_points[member_order]
    group_count = len(selected_components)
    member_counts = np.bincount(point_group[ordered_members], minlength=group_count)
    if np.any(member_counts < 2):
        raise ValueError("A selected single-view component has fewer than two points")
    member_offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(member_counts, dtype=np.int64)]
    )

    usable_mask = (
        (prepared.obs_weights > 0.0)
        & identifiable[prepared.obs_view_index]
        & (point_component[prepared.obs_point_index] >= 0)
    )
    usable_rows = np.flatnonzero(usable_mask)
    usable_components = point_component[prepared.obs_point_index[usable_rows]]
    row_order = np.argsort(usable_components, kind="stable")
    usable_rows = usable_rows[row_order]
    usable_components = usable_components[row_order]
    component_row_count = np.bincount(
        usable_components, minlength=component_count
    ).astype(np.int64)
    row_offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(component_row_count)]
    )

    ray_by_point = np.zeros((point_count, 3), dtype=np.float64)
    usable_ray_mask = np.zeros(point_count, dtype=bool)
    row_points = prepared.obs_point_index[usable_rows]
    unique_points, first_positions = np.unique(row_points, return_index=True)
    first_rows = usable_rows[first_positions]
    first_jacobians = prepared.obs_jacobian[first_rows].astype(np.float64)
    rays = np.cross(first_jacobians[:, 0], first_jacobians[:, 1])
    ray_norm = np.linalg.norm(rays, axis=1)
    if np.any(ray_norm <= EPSILON):
        raise ValueError("An observation projection Jacobian has no viewing ray")
    ray_by_point[unique_points] = rays / ray_norm[:, None]
    usable_ray_mask[unique_points] = True

    point_blocks = np.zeros((point_count, 3, 6), dtype=np.float64)
    observable_twist = np.zeros((component_count, 6), dtype=np.complex128)
    fill_basis = np.zeros((component_count, 6, 6), dtype=np.float64)
    observable_rank = np.zeros(component_count, dtype=np.int8)
    fill_nullity = np.zeros(component_count, dtype=np.int8)
    ray_dominated_count = np.zeros(component_count, dtype=np.int8)
    observable_phi = np.zeros((point_count, 3), dtype=np.complex128)
    observable_phi[trusted_seed_mask] = trusted_phi[trusted_seed_mask]

    for group, component in enumerate(selected_components.tolist()):
        members = ordered_members[member_offsets[group] : member_offsets[group + 1]]
        blocks = _rigid_point_blocks(
            points[members], component_centroid[component], float(component_radius[component])
        )
        point_blocks[members] = blocks
        rows = usable_rows[row_offsets[component] : row_offsets[component + 1]]
        if len(rows) == 0:
            raise ValueError(f"Single-view component {component} has no usable row")
        row_blocks = _rigid_point_blocks(
            points[prepared.obs_point_index[rows]],
            component_centroid[component],
            float(component_radius[component]),
        )
        projected_blocks = np.einsum(
            "rij,rjk->rik", prepared.obs_jacobian[rows].astype(np.float64), row_blocks
        )
        sqrt_weight = np.sqrt(prepared.obs_weights[rows])
        design = (
            sqrt_weight[:, None, None]
            * alphas[prepared.obs_view_index[rows], None, None]
            * projected_blocks
        ).reshape(-1, 6)
        target = (sqrt_weight[:, None] * prepared.obs_y[rows]).reshape(-1)
        gram = design.conj().T @ design
        if np.max(np.abs(gram.imag), initial=0.0) > 1.0e-8 * max(
            1.0, float(np.max(np.abs(gram.real), initial=0.0))
        ):
            raise RuntimeError("Single-view component Gram matrix is not real")
        eigenvalues, eigenvectors = np.linalg.eigh(gram.real)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        right_vectors = eigenvectors[:, order]
        singular = np.sqrt(eigenvalues)
        singular_ratio = singular / max(float(singular[0]), EPSILON)
        normal_rhs = design.conj().T @ target
        mode_coefficients = np.zeros(6, dtype=np.complex128)
        positive = eigenvalues > EPSILON**2
        mode_coefficients[positive] = (
            right_vectors[:, positive].T @ normal_rhs
        ) / eigenvalues[positive]

        induced = np.einsum("nij,jk->nik", blocks, right_vectors)
        ray_member_mask = usable_ray_mask[members]
        if not np.any(ray_member_mask):
            raise ValueError(f"Single-view component {component} has no viewing ray")
        ray_members = members[ray_member_mask]
        ray_induced = induced[ray_member_mask]
        induced_energy = np.sum(np.square(ray_induced), axis=(0, 1))
        radial = np.einsum("ni,nik->nk", ray_by_point[ray_members], ray_induced)
        radial_energy = np.sum(np.square(radial), axis=0)
        radial_fraction = np.sqrt(
            radial_energy / np.maximum(induced_energy, EPSILON**2)
        )
        physical = induced_energy > EPSILON**2
        ray_dominated = physical & (
            radial_fraction >= config.ray_direction_minimum_fraction
        )
        stable = (
            physical
            & (singular_ratio >= config.observable_singular_ratio_minimum)
            & ~ray_dominated
        )
        fill_directions = physical & ~stable
        observable_rank[component] = np.count_nonzero(stable)
        fill_nullity[component] = np.count_nonzero(fill_directions)
        ray_dominated_count[component] = np.count_nonzero(ray_dominated)
        observable_twist[component] = right_vectors[:, stable] @ mode_coefficients[stable]
        dimension = int(fill_nullity[component])
        if dimension:
            fill_basis[component, :, :dimension] = right_vectors[:, fill_directions]
        observable_phi[members] = np.einsum(
            "nij,j->ni", blocks, observable_twist[component]
        )

    edges = fill_graph.edge_index
    endpoint_trusted = trusted_seed_mask[edges]
    endpoint_groups = point_group[edges]
    endpoint_single = endpoint_groups >= 0
    eligible = (
        (endpoint_trusted[:, 0] | endpoint_single[:, 0])
        & (endpoint_trusted[:, 1] | endpoint_single[:, 1])
        & (endpoint_single[:, 0] | endpoint_single[:, 1])
        & ~(
            endpoint_single[:, 0]
            & endpoint_single[:, 1]
            & (endpoint_groups[:, 0] == endpoint_groups[:, 1])
        )
    )
    eligible_indices = np.flatnonzero(eligible)
    group_adjacency: list[list[int]] = [[] for _ in range(group_count)]
    group_hop = np.full(group_count, -1, dtype=np.int32)
    queue: list[int] = []
    trusted_knn_count = np.zeros(component_count, dtype=np.int32)
    cross_knn_count = np.zeros(component_count, dtype=np.int32)
    for edge in eligible_indices.tolist():
        group_i, group_j = int(endpoint_groups[edge, 0]), int(endpoint_groups[edge, 1])
        if group_i >= 0:
            component_i = int(selected_components[group_i])
            cross_knn_count[component_i] += 1
            if endpoint_trusted[edge, 1]:
                trusted_knn_count[component_i] += 1
                if group_hop[group_i] < 0:
                    group_hop[group_i] = 1
                    queue.append(group_i)
        if group_j >= 0:
            component_j = int(selected_components[group_j])
            cross_knn_count[component_j] += 1
            if endpoint_trusted[edge, 0]:
                trusted_knn_count[component_j] += 1
                if group_hop[group_j] < 0:
                    group_hop[group_j] = 1
                    queue.append(group_j)
        if group_i >= 0 and group_j >= 0:
            group_adjacency[group_i].append(group_j)
            group_adjacency[group_j].append(group_i)
    position = 0
    while position < len(queue):
        group = queue[position]
        position += 1
        for neighbor in group_adjacency[group]:
            if group_hop[neighbor] < 0:
                group_hop[neighbor] = group_hop[group] + 1
                queue.append(neighbor)
    connected_groups = group_hop >= 0
    component_connected = np.zeros(component_count, dtype=bool)
    component_connected[selected_components] = connected_groups
    active_edge_mask = eligible.copy()
    for side in (0, 1):
        side_single = endpoint_single[:, side]
        active_edge_mask[side_single] &= connected_groups[
            endpoint_groups[side_single, side]
        ]
    active_indices = np.flatnonzero(active_edge_mask)
    active_edges = edges[active_indices]

    group_dimensions = fill_nullity[selected_components].astype(np.int32)
    active_dimensions = np.where(connected_groups, group_dimensions, 0).astype(np.int64)
    coefficient_offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(active_dimensions, dtype=np.int64)]
    )
    correction_blocks = np.zeros((point_count, 3, 6), dtype=np.float64)
    for group, component in enumerate(selected_components.tolist()):
        dimension = int(group_dimensions[group])
        if dimension:
            members = ordered_members[member_offsets[group] : member_offsets[group + 1]]
            correction_blocks[members, :, :dimension] = np.einsum(
                "nij,jk->nik",
                point_blocks[members],
                fill_basis[component, :, :dimension],
            )
    phi_observable = np.zeros((point_count, 3), dtype=np.complex128)
    phi_observable[trusted_seed_mask] = observable_phi[trusted_seed_mask]
    phi_observable[selected_point_mask] = observable_phi[selected_point_mask]
    sqrt_weight = np.sqrt(fill_graph.edge_weight[active_indices])
    right_hand_side = (
        sqrt_weight[:, None]
        * (phi_observable[active_edges[:, 1]] - phi_observable[active_edges[:, 0]])
    ).reshape(-1)
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    edge_rows = np.arange(len(active_edges), dtype=np.int64)
    coordinate = np.arange(3, dtype=np.int64)
    for side, sign in ((0, 1.0), (1, -1.0)):
        side_points = active_edges[:, side]
        side_groups = point_group[side_points]
        for dimension in range(1, 7):
            selected = (side_groups >= 0) & (
                group_dimensions[np.maximum(side_groups, 0)] == dimension
            )
            if not np.any(selected):
                continue
            rows = edge_rows[selected]
            selected_points = side_points[selected]
            groups = side_groups[selected]
            block = (
                sign
                * sqrt_weight[selected, None, None]
                * correction_blocks[selected_points, :, :dimension]
            )
            row_parts.append(
                np.broadcast_to(
                    rows[:, None, None] * 3 + coordinate[None, :, None], block.shape
                ).reshape(-1)
            )
            column_parts.append(
                np.broadcast_to(
                    coefficient_offsets[groups, None, None]
                    + np.arange(dimension)[None, None],
                    block.shape,
                ).reshape(-1)
            )
            value_parts.append(block.reshape(-1))
    sparse = _require_module("scipy.sparse")
    coo_matrix = getattr(sparse, "coo_matrix", None)
    if coo_matrix is None:
        raise RuntimeError("scipy.sparse.coo_matrix is unavailable")
    row_count = len(active_edges) * 3
    column_count = int(coefficient_offsets[-1])
    if value_parts:
        matrix = coo_matrix(
            (
                np.concatenate(value_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=(row_count, column_count),
            dtype=np.float64,
        ).tocsr()
        matrix.eliminate_zeros()
    else:
        matrix = coo_matrix((row_count, column_count), dtype=np.float64).tocsr()
    if column_count:
        real_coefficients, real_solver = _run_lsmr(
            matrix, right_hand_side.real, config, component_stage=True
        )
        imaginary_coefficients, imaginary_solver = _run_lsmr(
            matrix, right_hand_side.imag, config, component_stage=True
        )
        coefficients = real_coefficients + 1j * imaginary_coefficients
    else:
        coefficients = np.empty(0, dtype=np.complex128)
        real_solver = _empty_solver(right_hand_side.real)
        imaginary_solver = _empty_solver(right_hand_side.imag)

    final_phi = phi_observable.copy()
    final_twist = observable_twist.copy()
    component_translation = np.zeros((component_count, 3), dtype=np.complex128)
    component_rotation = np.zeros((component_count, 3), dtype=np.complex128)
    for group, component in enumerate(selected_components.tolist()):
        dimension = int(group_dimensions[group]) if connected_groups[group] else 0
        if dimension:
            start = int(coefficient_offsets[group])
            final_twist[component] += (
                fill_basis[component, :, :dimension]
                @ coefficients[start : start + dimension]
            )
        members = ordered_members[member_offsets[group] : member_offsets[group + 1]]
        final_phi[members] = np.einsum(
            "nij,j->ni", point_blocks[members], final_twist[component]
        )
        component_translation[component] = final_twist[component, :3]
        if component_radius[component] > EPSILON:
            component_rotation[component] = (
                final_twist[component, 3:] / component_radius[component]
            )
    final_phi[trusted_seed_mask] = trusted_phi[trusted_seed_mask]

    global_edges = observed_graph.node_gaussian_index[observed_graph.edge_index]
    edge_vectors = points[global_edges[:, 1]] - points[global_edges[:, 0]]
    edge_delta = final_phi[global_edges[:, 1]] - final_phi[global_edges[:, 0]]
    axial = np.einsum("ij,ij->i", edge_vectors, edge_delta)
    denominator = np.maximum(np.einsum("ij,ij->i", edge_vectors, edge_vectors), EPSILON)
    edge_relative = np.maximum(np.abs(axial.real), np.abs(axial.imag)) / denominator
    component_first_order = np.zeros(component_count, dtype=np.float64)
    selected_edges = selected_mask[edge_component]
    np.maximum.at(
        component_first_order,
        edge_component[selected_edges],
        edge_relative[selected_edges],
    )
    if np.max(component_first_order[selected_mask], initial=0.0) > first_order_rtol:
        raise RuntimeError("Single-view component fill violated first-order rigidity")

    postfill_residual = np.zeros(component_count, dtype=np.float64)
    ray_motion_rms = np.zeros(component_count, dtype=np.float64)
    tangent_motion_rms = np.zeros(component_count, dtype=np.float64)
    ray_motion_ratio = np.zeros(component_count, dtype=np.float64)
    for group, component in enumerate(selected_components.tolist()):
        rows = usable_rows[row_offsets[component] : row_offsets[component + 1]]
        row_blocks = _rigid_point_blocks(
            points[prepared.obs_point_index[rows]],
            component_centroid[component],
            float(component_radius[component]),
        )
        projected = np.einsum(
            "rij,rjk->rik", prepared.obs_jacobian[rows].astype(np.float64), row_blocks
        )
        sqrt_rows = np.sqrt(prepared.obs_weights[rows])
        design = (
            sqrt_rows[:, None, None]
            * alphas[prepared.obs_view_index[rows], None, None]
            * projected
        ).reshape(-1, 6)
        target = (sqrt_rows[:, None] * prepared.obs_y[rows]).reshape(-1)
        postfill_residual[component] = np.linalg.norm(
            design @ final_twist[component] - target
        ) / max(float(np.linalg.norm(target)), EPSILON)
        members = ordered_members[member_offsets[group] : member_offsets[group + 1]]
        ray_members = members[usable_ray_mask[members]]
        member_phi = final_phi[ray_members]
        radial = np.einsum("ni,ni->n", ray_by_point[ray_members], member_phi)
        tangent = member_phi - radial[:, None] * ray_by_point[ray_members]
        ray_motion_rms[component] = np.sqrt(np.mean(np.abs(radial) ** 2))
        tangent_motion_rms[component] = np.sqrt(
            np.mean(np.sum(np.abs(tangent) ** 2, axis=1))
        )
        ray_motion_ratio[component] = ray_motion_rms[component] / max(
            tangent_motion_rms[component], EPSILON
        )
    finite_drift = _component_finite_drift(
        points,
        final_phi,
        global_edges,
        edge_component,
        selected_mask,
        phase_samples,
        component_count,
    )
    finite_rejected = selected_mask & (finite_drift > config.maximum_finite_drift)
    retained_after_fill = selected_mask & ~finite_rejected
    rejected_points = selected_point_mask & finite_rejected[
        np.maximum(point_component, 0)
    ]
    final_phi[rejected_points] = 0.0
    component_translation[finite_rejected] = 0.0
    component_rotation[finite_rejected] = 0.0
    effective_connected = connected_groups & retained_after_fill[selected_components]
    component_completion = np.zeros(component_count, dtype=bool)
    component_completion[selected_components] = effective_connected
    return SingleViewComponentResult(
        phi=final_phi.astype(np.complex64),
        component_fill_mask=selected_mask,
        component_completion_mask=component_completion,
        component_anchor_mask=retained_after_fill,
        component_translation=component_translation.astype(np.complex64),
        component_rotation=component_rotation.astype(np.complex64),
        component_first_order_relative_max=component_first_order.astype(np.float32),
        component_observable_rank=observable_rank,
        component_fill_nullity=fill_nullity,
        component_ray_dominated_basis_count=ray_dominated_count,
        component_trusted_knn_edge_count=trusted_knn_count,
        component_cross_knn_edge_count=cross_knn_count,
        component_connected_to_trusted_mask=component_connected,
        component_postfill_normalized_residual=postfill_residual.astype(np.float32),
        component_ray_motion_rms=ray_motion_rms.astype(np.float32),
        component_tangent_motion_rms=tangent_motion_rms.astype(np.float32),
        component_ray_motion_ratio=ray_motion_ratio.astype(np.float32),
        component_finite_drift_max=finite_drift.astype(np.float32),
        component_finite_drift_rejected_mask=finite_rejected,
        point_fill_mask=selected_point_mask,
        real_solver=real_solver,
        imaginary_solver=imaginary_solver,
        system_row_count=row_count,
        system_column_count=column_count,
        active_edge_count=len(active_edges),
    )


def _pointwise_fill(
    graph: KnnGraph,
    fixed_phi: np.ndarray,
    fixed_anchor_mask: np.ndarray,
    config: MotionFillConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    SparseSolveMetadata,
    SparseSolveMetadata,
    int,
    int,
    int,
]:
    """Fill independent 3D Gaussian motion within the accepted anchor-hop radius."""

    connected, hops = _anchor_connectivity(graph, fixed_anchor_mask)
    solve_mask = connected & (hops <= config.max_anchor_hops)
    variable_mask = solve_mask & ~fixed_anchor_mask
    variable_points = np.flatnonzero(variable_mask)
    point_to_variable = np.full(graph.point_count, -1, dtype=np.int64)
    point_to_variable[variable_points] = np.arange(len(variable_points), dtype=np.int64)
    active_edge_mask = solve_mask[graph.edge_index[:, 0]] & solve_mask[
        graph.edge_index[:, 1]
    ]
    active_indices = np.flatnonzero(active_edge_mask)
    edges = graph.edge_index[active_indices]
    sqrt_weight = np.sqrt(graph.edge_weight[active_indices])
    right_hand_side = (
        sqrt_weight[:, None] * (fixed_phi[edges[:, 1]] - fixed_phi[edges[:, 0]])
    ).reshape(-1)
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    edge_rows = np.arange(len(edges), dtype=np.int64)
    coordinate = np.arange(3, dtype=np.int64)
    for side, sign in ((0, 1.0), (1, -1.0)):
        points = edges[:, side]
        selected = variable_mask[points]
        if not np.any(selected):
            continue
        selected_rows = edge_rows[selected]
        selected_points = points[selected]
        rows = selected_rows[:, None] * 3 + coordinate[None]
        columns = point_to_variable[selected_points, None] * 3 + coordinate[None]
        values = np.broadcast_to(sign * sqrt_weight[selected, None], rows.shape)
        row_parts.append(rows.reshape(-1))
        column_parts.append(columns.reshape(-1))
        value_parts.append(values.reshape(-1))
    sparse = _require_module("scipy.sparse")
    coo_matrix = getattr(sparse, "coo_matrix", None)
    if coo_matrix is None:
        raise RuntimeError("scipy.sparse.coo_matrix is unavailable")
    row_count = len(edges) * 3
    column_count = len(variable_points) * 3
    if value_parts:
        matrix = coo_matrix(
            (
                np.concatenate(value_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=(row_count, column_count),
            dtype=np.float64,
        ).tocsr()
    else:
        matrix = coo_matrix((row_count, column_count), dtype=np.float64).tocsr()
    if column_count:
        real_coefficients, real_solver = _run_lsmr(
            matrix, np.real(right_hand_side), config, component_stage=False
        )
        imaginary_coefficients, imaginary_solver = _run_lsmr(
            matrix, np.imag(right_hand_side), config, component_stage=False
        )
        coefficients = (real_coefficients + 1j * imaginary_coefficients).reshape(-1, 3)
    else:
        coefficients = np.empty((0, 3), dtype=np.complex128)
        real_solver = _empty_solver(np.real(right_hand_side))
        imaginary_solver = _empty_solver(np.imag(right_hand_side))
    phi = np.asarray(fixed_phi, dtype=np.complex64).copy()
    phi[variable_points] = coefficients.astype(np.complex64)
    phi[fixed_anchor_mask] = fixed_phi[fixed_anchor_mask]
    if not np.array_equal(phi[fixed_anchor_mask], fixed_phi[fixed_anchor_mask]):
        raise RuntimeError("Pointwise motion fill changed a fixed anchor")
    completion_mask = variable_mask
    return (
        phi,
        completion_mask,
        hops,
        real_solver,
        imaginary_solver,
        row_count,
        column_count,
        len(edges),
    )


def apply_sequential_motion_fill(
    *,
    points: np.ndarray,
    prepared: PreparedObservations,
    alphas: np.ndarray,
    identifiable: np.ndarray,
    trusted_phi: np.ndarray,
    trusted_seed_mask: np.ndarray,
    point_component: np.ndarray,
    component_centroid: np.ndarray,
    component_radius: np.ndarray,
    component_supported_view_count: np.ndarray,
    component_retained_mask: np.ndarray,
    edge_component: np.ndarray,
    observed_graph: StructureGraphArrays,
    fill_graph: KnnGraph,
    first_order_rtol: float,
    phase_samples: int,
    config: MotionFillConfig | None = None,
) -> SequentialMotionFillResult:
    """Run single-view rigid promotion followed by independent Gaussian fill."""

    settings = config or MotionFillConfig()
    settings.validate()
    point_values = np.asarray(points, dtype=np.float64)
    point_count = len(point_values)
    trusted = np.asarray(trusted_seed_mask, dtype=bool)
    trusted_values = np.asarray(trusted_phi, dtype=np.complex64)
    if trusted.shape != (point_count,) or trusted_values.shape != (point_count, 3):
        raise ValueError("Trusted rigid arrays do not match foreground point count")
    if not np.any(trusted):
        raise ValueError("Sequential motion fill requires at least one trusted rigid seed")
    if not np.all(trusted_values[~trusted] == 0.0):
        raise ValueError("Untrusted rigid phi values must be zero")
    if fill_graph.point_count != point_count:
        raise ValueError("Motion-fill graph point count differs from static foreground")

    single_view = _single_view_component_fill(
        points=point_values,
        prepared=prepared,
        alphas=np.asarray(alphas, dtype=np.complex64),
        identifiable=np.asarray(identifiable, dtype=bool),
        trusted_phi=trusted_values,
        trusted_seed_mask=trusted,
        point_component=np.asarray(point_component, dtype=np.int64),
        component_centroid=np.asarray(component_centroid, dtype=np.float64),
        component_radius=np.asarray(component_radius, dtype=np.float64),
        component_supported_view_count=np.asarray(
            component_supported_view_count, dtype=np.int32
        ),
        component_retained_mask=np.asarray(component_retained_mask, dtype=bool),
        edge_component=np.asarray(edge_component, dtype=np.int64),
        observed_graph=observed_graph,
        fill_graph=fill_graph,
        first_order_rtol=float(first_order_rtol),
        phase_samples=int(phase_samples),
        config=settings,
    )
    component_anchor_points = np.zeros(point_count, dtype=bool)
    has_component = point_component >= 0
    component_anchor_points[has_component] = single_view.component_anchor_mask[
        point_component[has_component]
    ]
    if np.any(component_anchor_points & trusted):
        raise RuntimeError("Promoted component anchors overlap trusted rigid seeds")
    fixed_anchor = trusted | component_anchor_points
    fixed_phi = np.zeros((point_count, 3), dtype=np.complex64)
    fixed_phi[fixed_anchor] = single_view.phi[fixed_anchor]
    (
        phi,
        pointwise_completion,
        hops,
        real_solver,
        imaginary_solver,
        row_count,
        column_count,
        active_edge_count,
    ) = _pointwise_fill(fill_graph, fixed_phi, fixed_anchor, settings)
    overall_completion = pointwise_completion | component_anchor_points
    completed = trusted | overall_completion
    unresolved = ~completed
    connected, _ = _anchor_connectivity(fill_graph, fixed_anchor)
    support_class = np.full(point_count, SUPPORT_UNRESOLVED, dtype=np.int8)
    support_class[pointwise_completion] = SUPPORT_POINTWISE_FILL
    support_class[component_anchor_points] = SUPPORT_PROMOTED_RIGID
    support_class[trusted] = SUPPORT_TRUSTED_RIGID
    point_residual, residual_valid = _point_residuals(
        prepared,
        np.asarray(alphas, dtype=np.complex64),
        np.asarray(identifiable, dtype=bool),
        phi,
    )
    if not np.isfinite(phi[completed]).all():
        raise FloatingPointError("Completed motion contains NaN or Inf")
    if not np.array_equal(phi[trusted], trusted_values[trusted]):
        raise RuntimeError("Sequential motion fill changed trusted rigid seeds")
    return SequentialMotionFillResult(
        phi=phi,
        trusted_seed_mask=trusted,
        component_anchor_point_mask=component_anchor_points,
        completion_mask=overall_completion,
        completion_connected_to_anchor=connected,
        unresolved_mask=unresolved,
        support_class=support_class,
        hop_distance=hops,
        point_residual=point_residual,
        point_residual_valid_mask=residual_valid,
        single_view=single_view,
        pointwise_real_solver=real_solver,
        pointwise_imaginary_solver=imaginary_solver,
        pointwise_system_row_count=row_count,
        pointwise_system_column_count=column_count,
        pointwise_active_edge_count=active_edge_count,
    )


SOLVER_METADATA_FIELDS = (
    "performed",
    "converged",
    "stop_code",
    "iterations",
    "residual_norm",
    "normal_residual_norm",
    "matrix_norm",
    "condition_estimate",
    "solution_norm",
)

MODE_ARRAY_FIELDS = (
    "phi",
    "trusted_seed_mask",
    "component_anchor_point_mask",
    "completion_mask",
    "completion_connected_to_anchor",
    "unresolved_mask",
    "support_class",
    "hop_distance",
    "point_residual",
    "point_residual_valid_mask",
    "single_view_component_fill_mask",
    "single_view_component_completion_mask",
    "single_view_component_anchor_mask",
    "single_view_component_translation",
    "single_view_component_rotation",
    "single_view_component_first_order_relative_max",
    "single_view_component_observable_rank",
    "single_view_component_fill_nullity",
    "single_view_component_ray_dominated_basis_count",
    "single_view_component_trusted_knn_edge_count",
    "single_view_component_cross_knn_edge_count",
    "single_view_component_connected_to_trusted_mask",
    "single_view_component_postfill_normalized_residual",
    "single_view_component_ray_motion_rms",
    "single_view_component_tangent_motion_rms",
    "single_view_component_ray_motion_ratio",
    "single_view_component_finite_drift_max",
    "single_view_component_finite_drift_rejected_mask",
    "single_view_point_fill_mask",
    "single_view_system_row_count",
    "single_view_system_column_count",
    "single_view_active_edge_count",
    "pointwise_system_row_count",
    "pointwise_system_column_count",
    "pointwise_active_edge_count",
) + tuple(
    f"{stage}_{part}_{field}"
    for stage in ("single_view", "pointwise")
    for part in ("real", "imaginary")
    for field in SOLVER_METADATA_FIELDS
)

GRAPH_ARRAY_FIELDS = (
    "fill_graph_edge_index",
    "fill_graph_edge_distance",
    "fill_graph_edge_weight",
    "fill_graph_degree",
    "fill_graph_component_index",
    "fill_graph_component_size",
    "fill_graph_isolated_mask",
)


def _solver_arrays(prefix: str, metadata: SparseSolveMetadata) -> dict[str, np.ndarray]:
    """Flatten one LSMR metadata record into scalar NumPy fields."""

    return {
        f"{prefix}_performed": np.asarray(metadata.performed, dtype=bool),
        f"{prefix}_converged": np.asarray(metadata.converged, dtype=bool),
        f"{prefix}_stop_code": np.asarray(metadata.stop_code, dtype=np.int32),
        f"{prefix}_iterations": np.asarray(metadata.iterations, dtype=np.int32),
        f"{prefix}_residual_norm": np.asarray(metadata.residual_norm, dtype=np.float64),
        f"{prefix}_normal_residual_norm": np.asarray(
            metadata.normal_residual_norm, dtype=np.float64
        ),
        f"{prefix}_matrix_norm": np.asarray(metadata.matrix_norm, dtype=np.float64),
        f"{prefix}_condition_estimate": np.asarray(
            metadata.condition_estimate, dtype=np.float64
        ),
        f"{prefix}_solution_norm": np.asarray(metadata.solution_norm, dtype=np.float64),
    }


def _mode_arrays(result: SequentialMotionFillResult) -> dict[str, np.ndarray]:
    """Flatten one completed mode into fixed-shape checkpoint arrays."""

    single = result.single_view
    arrays = {
        "phi": result.phi,
        "trusted_seed_mask": result.trusted_seed_mask,
        "component_anchor_point_mask": result.component_anchor_point_mask,
        "completion_mask": result.completion_mask,
        "completion_connected_to_anchor": result.completion_connected_to_anchor,
        "unresolved_mask": result.unresolved_mask,
        "support_class": result.support_class,
        "hop_distance": result.hop_distance,
        "point_residual": result.point_residual,
        "point_residual_valid_mask": result.point_residual_valid_mask,
        "single_view_component_fill_mask": single.component_fill_mask,
        "single_view_component_completion_mask": single.component_completion_mask,
        "single_view_component_anchor_mask": single.component_anchor_mask,
        "single_view_component_translation": single.component_translation,
        "single_view_component_rotation": single.component_rotation,
        "single_view_component_first_order_relative_max": (
            single.component_first_order_relative_max
        ),
        "single_view_component_observable_rank": single.component_observable_rank,
        "single_view_component_fill_nullity": single.component_fill_nullity,
        "single_view_component_ray_dominated_basis_count": (
            single.component_ray_dominated_basis_count
        ),
        "single_view_component_trusted_knn_edge_count": (
            single.component_trusted_knn_edge_count
        ),
        "single_view_component_cross_knn_edge_count": (
            single.component_cross_knn_edge_count
        ),
        "single_view_component_connected_to_trusted_mask": (
            single.component_connected_to_trusted_mask
        ),
        "single_view_component_postfill_normalized_residual": (
            single.component_postfill_normalized_residual
        ),
        "single_view_component_ray_motion_rms": single.component_ray_motion_rms,
        "single_view_component_tangent_motion_rms": single.component_tangent_motion_rms,
        "single_view_component_ray_motion_ratio": single.component_ray_motion_ratio,
        "single_view_component_finite_drift_max": single.component_finite_drift_max,
        "single_view_component_finite_drift_rejected_mask": (
            single.component_finite_drift_rejected_mask
        ),
        "single_view_point_fill_mask": single.point_fill_mask,
        "single_view_system_row_count": np.asarray(single.system_row_count, dtype=np.int64),
        "single_view_system_column_count": np.asarray(
            single.system_column_count, dtype=np.int64
        ),
        "single_view_active_edge_count": np.asarray(single.active_edge_count, dtype=np.int64),
        "pointwise_system_row_count": np.asarray(
            result.pointwise_system_row_count, dtype=np.int64
        ),
        "pointwise_system_column_count": np.asarray(
            result.pointwise_system_column_count, dtype=np.int64
        ),
        "pointwise_active_edge_count": np.asarray(
            result.pointwise_active_edge_count, dtype=np.int64
        ),
    }
    arrays.update(_solver_arrays("single_view_real", single.real_solver))
    arrays.update(_solver_arrays("single_view_imaginary", single.imaginary_solver))
    arrays.update(_solver_arrays("pointwise_real", result.pointwise_real_solver))
    arrays.update(
        _solver_arrays("pointwise_imaginary", result.pointwise_imaginary_solver)
    )
    return arrays


def _graph_arrays(graph: KnnGraph) -> dict[str, np.ndarray]:
    """Return full-foreground graph arrays in final artifact field names."""

    return {
        "fill_graph_edge_index": graph.edge_index,
        "fill_graph_edge_distance": graph.edge_distance,
        "fill_graph_edge_weight": graph.edge_weight,
        "fill_graph_degree": graph.degree,
        "fill_graph_component_index": graph.component_index,
        "fill_graph_component_size": graph.component_size,
        "fill_graph_isolated_mask": graph.isolated_mask,
    }


def _canonical_json(value: Any) -> bytes:
    """Encode one identity payload deterministically."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    """Hash one artifact file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrays_identity(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash array field names, dtypes, shapes, and values canonically."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _run_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select immutable upstream identities, modes, views, graph, and config."""

    return {
        "static_scene_identity": manifest["static_scene_identity"],
        "foreground_identity": manifest["foreground_identity"],
        "topology_identity": manifest["topology_identity"],
        "gaussian_measurements_identity": manifest[
            "gaussian_measurements_identity"
        ],
        "observed_structure_graph_identity": manifest[
            "observed_structure_graph_identity"
        ],
        "rigid_modes_identity": manifest["rigid_modes_identity"],
        "modes": manifest["modes"],
        "views": manifest["views"],
        "motion_fill": manifest["motion_fill"],
        "fill_graph_identity": manifest["fill_graph_identity"],
    }


def _artifact_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select scientific fields defining one completed modal artifact."""

    payload = _run_identity_payload(manifest)
    payload.update(
        {
            "format": COMPLETED_MODES_FORMAT,
            "version": COMPLETED_MODES_VERSION,
            "semantics": manifest["semantics"],
            "quality_gate": manifest["quality_gate"],
            "counts": manifest["counts"],
            "arrays_identity": manifest["arrays_identity"],
        }
    )
    return payload


def _load_sources(
    *,
    scene_dir: str | Path,
    topology_dir: str | Path,
    measurements_dir: str | Path,
    observed_graph_dir: str | Path,
    rigid_modes_dir: str | Path,
    config: MotionFillConfig,
) -> tuple[dict[str, Any], Any, Any, Any, Any, RigidModesArtifact, KnnGraph, np.ndarray]:
    """Load the complete identity chain and build its deterministic fill graph."""

    scene = load_static_scene(scene_dir, "cpu")
    topology = load_observation_topology(topology_dir)
    measurements = load_gaussian_measurements(measurements_dir)
    observed_graph = load_observed_structure_graph(observed_graph_dir)
    rigid = load_rigid_modes(rigid_modes_dir)
    scene_manifest = scene.manifest
    if scene_manifest is None:
        raise ValueError("Static scene has no manifest")
    scene_identity = scene_manifest["static_scene_identity"]
    foreground_identity = scene_manifest["foreground_identity"]
    topology_identity = topology.manifest["topology_identity"]
    measurement_identity = measurements.manifest["gaussian_measurements_identity"]
    observed_graph_identity = observed_graph.manifest[
        "observed_structure_graph_identity"
    ]
    required = {
        "static_scene_identity": scene_identity,
        "foreground_identity": foreground_identity,
        "topology_identity": topology_identity,
        "gaussian_measurements_identity": measurement_identity,
        "observed_structure_graph_identity": observed_graph_identity,
    }
    for name, expected in required.items():
        if rigid.manifest.get(name) != expected:
            raise ValueError(f"Rigid-mode {name} does not match the supplied input")
    if topology.manifest["static_scene_identity"] != scene_identity:
        raise ValueError("Topology static scene identity differs")
    if topology.manifest["foreground_identity"] != foreground_identity:
        raise ValueError("Topology foreground identity differs")
    if measurements.manifest["topology_identity"] != topology_identity:
        raise ValueError("Measurement topology identity differs")
    if observed_graph.manifest["topology_identity"] != topology_identity:
        raise ValueError("Observed graph topology identity differs")
    if observed_graph.manifest["static_scene_identity"] != scene_identity:
        raise ValueError("Observed graph static scene identity differs")

    if rigid.manifest["modes"] != measurements.manifest["modes"]:
        raise ValueError("Rigid and measurement mode order differs")
    topology_views = topology.manifest["views"]
    measurement_views = measurements.manifest["views"]
    observed_views = observed_graph.manifest["views"]
    rigid_views = rigid.manifest["views"]
    if not (
        len(topology_views)
        == len(measurement_views)
        == len(observed_views)
        == len(rigid_views)
    ):
        raise ValueError("Motion-fill inputs have different view counts")
    view_records: list[dict[str, Any]] = []
    for index, views in enumerate(
        zip(topology_views, measurement_views, observed_views, rigid_views)
    ):
        topology_view, measurement_view, observed_view, rigid_view = views
        label = topology_view["label"]
        for source_name, view in (
            ("measurement", measurement_view),
            ("observed graph", observed_view),
            ("rigid", rigid_view),
        ):
            if view["index"] != index or view["label"] != label:
                raise ValueError(f"{source_name} view order differs from topology")
            if view["flow_identity"] != topology_view["flow_identity"]:
                raise ValueError(f"{source_name} flow identity for {label!r} differs")
            if view["shape_hw"] != topology_view["shape_hw"]:
                raise ValueError(f"{source_name} shape for {label!r} differs")
        if observed_view["camera_identity"] != topology_view["camera_identity"]:
            raise ValueError(f"Observed graph camera identity for {label!r} differs")
        view_records.append(dict(rigid_view))

    points = (
        scene.foreground.active()["means"].detach().cpu().numpy().astype(np.float32)
    )
    if len(points) != int(rigid.manifest["counts"]["foreground_gaussians"]):
        raise ValueError("Rigid foreground count differs from static scene")
    fill_graph = build_motion_fill_graph(points, config)
    graph_values = _graph_arrays(fill_graph)
    fill_graph_identity = hashlib.sha256(
        _canonical_json(
            {
                "foreground_identity": foreground_identity,
                "parameters": {
                    "neighbors": config.neighbors,
                    "max_distance": config.max_distance,
                    "epsilon": config.graph_epsilon,
                    "policy": "distance_pruned_union_knn",
                },
                "arrays_identity": _arrays_identity(graph_values),
            }
        )
    ).hexdigest()
    source = {
        "static_scene": str(scene_dir),
        "static_scene_identity": scene_identity,
        "foreground_identity": foreground_identity,
        "topology": str(topology.path),
        "topology_identity": topology_identity,
        "measurements": str(measurements.path),
        "gaussian_measurements_identity": measurement_identity,
        "observed_structure_graph": str(observed_graph.path),
        "observed_structure_graph_identity": observed_graph_identity,
        "rigid_modes": str(rigid.path),
        "rigid_modes_identity": rigid.manifest["rigid_modes_identity"],
        "modes": [dict(mode) for mode in rigid.manifest["modes"]],
        "views": view_records,
        "motion_fill": config.to_dict(),
        "fill_graph_identity": fill_graph_identity,
    }
    source["solver_run_identity"] = hashlib.sha256(
        _canonical_json(_run_identity_payload(source))
    ).hexdigest()
    return (
        source,
        scene,
        topology,
        measurements,
        observed_graph,
        rigid,
        fill_graph,
        points,
    )


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write one per-mode checkpoint through a sibling temporary NPZ."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        save_named_arrays(temporary, arrays)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _prepare_work_dir(work: Path, source: Mapping[str, Any]) -> None:
    """Create or identity-check the resumable completion work directory."""

    manifest_path = work / "manifest.json"
    if work.exists() and not work.is_dir():
        raise FileExistsError(f"Motion-fill work path is not a directory: {work}")
    work.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("solver_run_identity") != source["solver_run_identity"]:
            raise ValueError("Motion-fill work directory belongs to another run")
        return
    if any(work.iterdir()):
        raise FileExistsError("Motion-fill work directory is non-empty without manifest")
    payload = {
        "format": "modal_gaussians.completed_modes_work",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **dict(source),
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)


def _load_mode_checkpoint(
    path: Path, *, mode_slot: int, solver_run_identity: str
) -> dict[str, np.ndarray]:
    """Load one complete identity-bound per-mode completion checkpoint."""

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    expected = set(MODE_ARRAY_FIELDS) | {"mode_slot", "solver_run_identity"}
    if set(arrays) != expected:
        raise ValueError(f"Motion-fill checkpoint fields are invalid: {path}")
    if int(arrays["mode_slot"]) != mode_slot:
        raise ValueError(f"Motion-fill checkpoint mode slot differs: {path}")
    if str(arrays["solver_run_identity"]) != solver_run_identity:
        raise ValueError(f"Motion-fill checkpoint identity differs: {path}")
    for name, value in arrays.items():
        if value.dtype == np.dtype("O"):
            raise ValueError(f"Motion-fill checkpoint {name} uses object dtype")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            allowed = name.endswith("condition_estimate")
            if not allowed or np.isnan(value).any():
                raise ValueError(f"Motion-fill checkpoint {name} is non-finite")
    return arrays


def _validate_final_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    mode_count: int,
    point_count: int,
    rigid_component_count: int,
    fill_edge_count: int,
) -> None:
    """Validate completed fields, masks, support classes, and graph dimensions."""

    if set(arrays) != set(MODE_ARRAY_FIELDS) | set(GRAPH_ARRAY_FIELDS):
        raise ValueError("Completed-mode array fields are invalid")
    expected_shapes = {
        "phi": (mode_count, point_count, 3),
        "trusted_seed_mask": (mode_count, point_count),
        "component_anchor_point_mask": (mode_count, point_count),
        "completion_mask": (mode_count, point_count),
        "completion_connected_to_anchor": (mode_count, point_count),
        "unresolved_mask": (mode_count, point_count),
        "support_class": (mode_count, point_count),
        "hop_distance": (mode_count, point_count),
        "point_residual": (mode_count, point_count),
        "single_view_component_anchor_mask": (mode_count, rigid_component_count),
        "fill_graph_edge_index": (fill_edge_count, 2),
        "fill_graph_edge_distance": (fill_edge_count,),
        "fill_graph_edge_weight": (fill_edge_count,),
        "fill_graph_degree": (point_count,),
        "fill_graph_component_index": (point_count,),
        "fill_graph_isolated_mask": (point_count,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Completed-mode {name} shape is invalid")
    for name in MODE_ARRAY_FIELDS:
        if arrays[name].shape[0] != mode_count:
            raise ValueError(f"Completed-mode {name} lacks the mode axis")
    for name, value in arrays.items():
        if value.dtype == np.dtype("O"):
            raise ValueError(f"Completed-mode {name} uses object dtype")
        if value.dtype.kind in "fc" and np.isnan(value).any():
            raise ValueError(f"Completed-mode {name} contains NaN")
    trusted = arrays["trusted_seed_mask"]
    promoted = arrays["component_anchor_point_mask"]
    completion = arrays["completion_mask"]
    unresolved = arrays["unresolved_mask"]
    if np.any(trusted & promoted):
        raise ValueError("Trusted and promoted completion anchors overlap")
    if not np.array_equal(unresolved, ~(trusted | completion)):
        raise ValueError("Completed-mode unresolved mask is inconsistent")
    if not np.all(arrays["phi"][unresolved] == 0.0):
        raise ValueError("Unresolved Gaussian motion must remain zero")
    support = arrays["support_class"]
    if np.any(support < 0) or np.any(support >= len(SUPPORT_CLASS_NAMES)):
        raise ValueError("Completed-mode support class is invalid")
    expected_support = np.full(support.shape, SUPPORT_UNRESOLVED, dtype=np.int8)
    pointwise = completion & ~promoted
    expected_support[pointwise] = SUPPORT_POINTWISE_FILL
    expected_support[promoted] = SUPPORT_PROMOTED_RIGID
    expected_support[trusted] = SUPPORT_TRUSTED_RIGID
    if not np.array_equal(support, expected_support):
        raise ValueError("Completed-mode support classes disagree with masks")


def load_completed_modes(path: str | Path) -> CompletedModesArtifact:
    """Load and fully validate one completed full-foreground modal artifact."""

    root = Path(path).expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    arrays_path = root / COMPLETED_MODES_FILENAME
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"Incomplete completed-mode artifact: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != COMPLETED_MODES_FORMAT:
        raise ValueError("Unsupported completed-mode format")
    if manifest.get("version") != COMPLETED_MODES_VERSION:
        raise ValueError("Unsupported completed-mode version")
    if manifest.get("quality_gate") != {
        "required": True,
        "status": "completion_candidate_unapproved",
    }:
        raise ValueError("Completed modes must remain an unapproved candidate")
    if manifest.get("arrays_file") != COMPLETED_MODES_FILENAME:
        raise ValueError("Completed-mode array filename is invalid")
    if manifest.get("arrays_file_sha256") != _sha256_file(arrays_path):
        raise ValueError("Completed-mode NPZ SHA-256 differs")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Completed-mode counts are invalid")
    _validate_final_arrays(
        arrays,
        mode_count=int(counts["modes"]),
        point_count=int(counts["foreground_gaussians"]),
        rigid_component_count=int(counts["rigid_components"]),
        fill_edge_count=int(counts["fill_graph_edges"]),
    )
    metadata = {
        name: {"dtype": value.dtype.name, "shape": list(value.shape)}
        for name, value in arrays.items()
    }
    if manifest.get("arrays") != metadata:
        raise ValueError("Completed-mode array metadata differs")
    if manifest.get("arrays_identity") != _arrays_identity(arrays):
        raise ValueError("Completed-mode array identity differs")
    expected_run = hashlib.sha256(
        _canonical_json(_run_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("solver_run_identity") != expected_run:
        raise ValueError("Completed-mode run identity differs")
    expected_artifact = hashlib.sha256(
        _canonical_json(_artifact_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("completed_modes_identity") != expected_artifact:
        raise ValueError("Completed-mode artifact identity differs")
    return CompletedModesArtifact(root, manifest, arrays)


def build_completed_modes_artifact(
    *,
    scene_dir: str | Path,
    topology_dir: str | Path,
    measurements_dir: str | Path,
    observed_graph_dir: str | Path,
    rigid_modes_dir: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
    config: MotionFillConfig | None = None,
    command: Sequence[str] = (),
) -> CompletedModesArtifact:
    """Run resumable sequential fill for every greedy mode and publish atomically."""

    settings = config or MotionFillConfig()
    settings.validate()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Completed-mode output already exists: {destination}")
    work = Path(work_dir).expanduser().resolve()
    (
        source,
        _,
        topology,
        measurements,
        observed_graph,
        rigid,
        fill_graph,
        points,
    ) = _load_sources(
        scene_dir=scene_dir,
        topology_dir=topology_dir,
        measurements_dir=measurements_dir,
        observed_graph_dir=observed_graph_dir,
        rigid_modes_dir=rigid_modes_dir,
        config=settings,
    )
    _prepare_work_dir(work, source)
    rigid_arrays = rigid.arrays
    component_count = len(rigid_arrays["component_centroid"])
    first_order_rtol = float(rigid.manifest["rigid_components"]["first_order_rtol"])
    phase_samples = int(rigid.manifest["rigid_components"]["phase_samples"])
    view_labels = tuple(view["label"] for view in source["views"])
    mode_results: list[dict[str, np.ndarray]] = []
    for mode in source["modes"]:
        slot = int(mode["mode_slot"])
        checkpoint = work / f"mode_{slot:03d}.npz"
        if checkpoint.is_file():
            result = _load_mode_checkpoint(
                checkpoint,
                mode_slot=slot,
                solver_run_identity=source["solver_run_identity"],
            )
        else:
            prepared = prepare_observations(
                points=points,
                topology=topology.arrays,
                sample_measurements=np.asarray(measurements.measurements[slot]),
                view_labels=view_labels,
            )
            completed = apply_sequential_motion_fill(
                points=points,
                prepared=prepared,
                alphas=rigid_arrays["alphas"][slot],
                identifiable=rigid_arrays["alpha_identifiable_mask"][slot],
                trusted_phi=rigid_arrays["trusted_phi"][slot],
                trusted_seed_mask=rigid_arrays["trusted_seed_mask"][slot],
                point_component=rigid_arrays["point_component_index"],
                component_centroid=rigid_arrays["component_centroid"],
                component_radius=rigid_arrays["component_radius"],
                component_supported_view_count=rigid_arrays[
                    "component_supported_valid_view_count"
                ][slot],
                component_retained_mask=rigid_arrays["component_retained_mask"][slot],
                edge_component=rigid_arrays["edge_component_index"],
                observed_graph=observed_graph.arrays,
                fill_graph=fill_graph,
                first_order_rtol=first_order_rtol,
                phase_samples=phase_samples,
                config=settings,
            )
            result = {
                **_mode_arrays(completed),
                "mode_slot": np.asarray(slot, dtype=np.int32),
                "solver_run_identity": np.asarray(
                    source["solver_run_identity"], dtype="<U64"
                ),
            }
            _atomic_savez(checkpoint, result)
            result = _load_mode_checkpoint(
                checkpoint,
                mode_slot=slot,
                solver_run_identity=source["solver_run_identity"],
            )
        mode_results.append(result)

    final_arrays = {
        name: np.stack([result[name] for result in mode_results], axis=0)
        for name in MODE_ARRAY_FIELDS
    }
    final_arrays.update(_graph_arrays(fill_graph))
    _validate_final_arrays(
        final_arrays,
        mode_count=len(source["modes"]),
        point_count=len(points),
        rigid_component_count=component_count,
        fill_edge_count=len(fill_graph.edge_index),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        arrays_path = temporary / COMPLETED_MODES_FILENAME
        save_named_arrays(arrays_path, final_arrays)
        trusted_count = np.count_nonzero(final_arrays["trusted_seed_mask"], axis=1)
        promoted_count = np.count_nonzero(
            final_arrays["component_anchor_point_mask"], axis=1
        )
        pointwise_count = np.count_nonzero(
            final_arrays["completion_mask"]
            & ~final_arrays["component_anchor_point_mask"],
            axis=1,
        )
        unresolved_count = np.count_nonzero(final_arrays["unresolved_mask"], axis=1)
        manifest = {
            "format": COMPLETED_MODES_FORMAT,
            "version": COMPLETED_MODES_VERSION,
            "producer": {
                "project_version": __version__,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": list(command),
            },
            **source,
            "work_dir": str(work),
            "semantics": {
                "pipeline": "single_view_partial_components_then_independent_3d_gaussians",
                "field": "complex_3d_displacement_in_normalized_scene_coordinates",
                "playback": "real(phi * exp(i*phase))",
                "trusted_seed_policy": "preserved_exactly",
                "unresolved_policy": "zero_with_explicit_unresolved_mask",
                "support_class_names": list(SUPPORT_CLASS_NAMES),
                "background_gaussians": "excluded",
            },
            "quality_gate": {
                "required": True,
                "status": "completion_candidate_unapproved",
            },
            "fill_graph": {
                "policy": "distance_pruned_union_knn",
                "candidate_directed_count": fill_graph.candidate_directed_count,
                "retained_directed_count": fill_graph.retained_directed_count,
                "pruned_directed_count": fill_graph.pruned_directed_count,
            },
            "counts": {
                "modes": len(source["modes"]),
                "views": len(source["views"]),
                "foreground_gaussians": len(points),
                "rigid_components": component_count,
                "fill_graph_edges": len(fill_graph.edge_index),
                "fill_graph_components": len(fill_graph.component_size),
                "fill_graph_isolated_gaussians": int(
                    np.count_nonzero(fill_graph.isolated_mask)
                ),
                "trusted_rigid_gaussians_per_mode": trusted_count.astype(int).tolist(),
                "promoted_rigid_gaussians_per_mode": promoted_count.astype(int).tolist(),
                "pointwise_filled_gaussians_per_mode": pointwise_count.astype(int).tolist(),
                "unresolved_gaussians_per_mode": unresolved_count.astype(int).tolist(),
            },
            "arrays_file": COMPLETED_MODES_FILENAME,
            "arrays": {
                name: {"dtype": value.dtype.name, "shape": list(value.shape)}
                for name, value in final_arrays.items()
            },
            "arrays_identity": _arrays_identity(final_arrays),
            "arrays_file_sha256": _sha256_file(arrays_path),
        }
        manifest["completed_modes_identity"] = hashlib.sha256(
            _canonical_json(_artifact_identity_payload(manifest))
        ).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_completed_modes(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Completed-mode output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_completed_modes(destination)


__all__ = [
    "CompletedModesArtifact",
    "KnnGraph",
    "MotionFillConfig",
    "SequentialMotionFillResult",
    "apply_sequential_motion_fill",
    "build_completed_modes_artifact",
    "build_motion_fill_graph",
    "load_completed_modes",
]
