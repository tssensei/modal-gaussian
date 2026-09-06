"""Static geometry and overlapping control graphs without appearance or motion gates.

Depth evidence is three-valued: a view can support a connection, contradict it
when both endpoints are visible, or provide no evidence.  Unknown short edges
are explicitly weaker geometric priors.  These graphs define interpolation and
regularization neighborhoods, not a finite-element elasticity discretization.
"""

from __future__ import annotations

from modal_gaussians.motion.common.geometry_ops import (
    pixel_valid as _pixel_valid,
    bilinear_sample_float64 as _bilinear,
)

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import scipy.spatial as spatial

cKDTree = getattr(spatial, "cKDTree")


EVIDENCE_CONTRADICTED = -1
EVIDENCE_UNKNOWN = 0
EVIDENCE_SUPPORTED = 1
EVIDENCE_SPATIAL_PRIOR = 2  # KNN-only ablation; no claim of depth support.


@dataclass(frozen=True)
class GeometryGraphConfig:
    max_neighbors: int = 8
    max_distance: float = 0.008
    unknown_max_distance: float = 0.004
    unknown_weight: float = 0.1
    alpha_minimum: float = 0.05
    profile_min_samples: int = 5
    profile_max_step_pixels: float = 1.0
    control_radius_fraction: float = 0.03
    max_controls: int = 2048
    edge_filter: str = "depth"

    def validate(self) -> None:
        if self.edge_filter not in ("depth", "none"):
            raise ValueError("Geometry edge_filter must be depth or none")
        for name in ("max_neighbors", "profile_min_samples", "max_controls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Geometry graph {name} must be a positive integer")
        if self.profile_min_samples < 5:
            raise ValueError("Depth profiles require at least five samples")
        for name in ("max_distance", "unknown_max_distance", "profile_max_step_pixels", "control_radius_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Geometry graph {name} must be finite and positive")
        if self.unknown_max_distance > self.max_distance:
            raise ValueError("Unknown-edge distance cannot exceed the candidate distance")
        if self.profile_max_step_pixels > 1.0:
            raise ValueError("Depth-profile spacing cannot exceed one pixel")
        if not math.isfinite(self.unknown_weight) or not 0 < self.unknown_weight <= 1:
            raise ValueError("Unknown-edge weight must lie in (0,1]")
        if not math.isfinite(self.alpha_minimum) or not 0 < self.alpha_minimum <= 1:
            raise ValueError("Geometry alpha minimum must lie in (0,1]")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Preserve the identity/configuration of existing depth-filtered v8 data.
        if self.edge_filter == "depth":
            result.pop("edge_filter")
        return result


@dataclass(frozen=True)
class GeometryGraph:
    points: np.ndarray
    node_gaussian_index: np.ndarray
    edge_index: np.ndarray
    edge_length: np.ndarray
    edge_weight: np.ndarray
    edge_evidence_kind: np.ndarray
    edge_view_evidence: np.ndarray
    node_visible_view_mask: np.ndarray
    degree: np.ndarray
    component_index: np.ndarray
    component_size: np.ndarray
    candidate_edge_index: np.ndarray
    candidate_view_evidence: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {field.name: np.ascontiguousarray(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, arrays: Mapping[str, Any]) -> GeometryGraph:
        names = {field.name for field in fields(cls)}
        if set(arrays) != names:
            raise ValueError("Geometry graph array inventory does not match its schema")
        graph = cls(**{name: np.asarray(arrays[name]) for name in names})
        _validate_geometry_arrays(graph)
        return graph


@dataclass(frozen=True)
class ControlGraph:
    control_point_index: np.ndarray
    positions: np.ndarray
    control_edges: np.ndarray
    control_edge_length: np.ndarray
    control_edge_weight: np.ndarray
    interpolation_indptr: np.ndarray
    interpolation_indices: np.ndarray
    interpolation_weights: np.ndarray
    owner: np.ndarray
    owner_distance: np.ndarray
    component_index: np.ndarray
    coverage_radius: float
    scene_scale: float

    def as_dict(self) -> dict[str, np.ndarray]:
        return {field.name: np.asarray(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, arrays: Mapping[str, Any]) -> ControlGraph:
        names = {field.name for field in fields(cls)}
        if set(arrays) != names:
            raise ValueError("Control graph array inventory does not match its schema")
        values: dict[str, Any] = {name: np.asarray(arrays[name]) for name in names}
        for name in ("coverage_radius", "scene_scale"):
            _require_dtype(name, values[name], np.float64)
            if values[name].shape != ():
                raise ValueError(f"Control graph {name} must be a scalar")
            values[name] = float(values[name])
        graph = cls(**values)
        _validate_control_arrays(graph)
        return graph


def _require_dtype(name: str, value: np.ndarray, dtype: Any) -> None:
    if value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must have dtype {np.dtype(dtype).name}")


def _validate_edges(name: str, edges: np.ndarray, count: int) -> None:
    _require_dtype(name, edges, np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"{name} must have shape [E,2]")
    if np.any(edges < 0) or np.any(edges >= count) or np.any(edges[:, 0] >= edges[:, 1]):
        raise ValueError(f"{name} has invalid endpoint indices")
    if not np.array_equal(edges, np.unique(edges, axis=0)):
        raise ValueError(f"{name} must be sorted and unique")


def _validate_geometry_arrays(graph: GeometryGraph) -> None:
    points = graph.points
    _require_dtype("points", points, np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Geometry points must be nonempty finite [G,3]")
    count = len(points)
    for name in ("node_gaussian_index", "degree", "component_index", "component_size"):
        _require_dtype(name, getattr(graph, name), np.int64)
    if not np.array_equal(graph.node_gaussian_index, np.arange(count, dtype=np.int64)):
        raise ValueError("Geometry node indices must cover all foreground Gaussians in order")
    _validate_edges("edge_index", graph.edge_index, count)
    _validate_edges("candidate_edge_index", graph.candidate_edge_index, count)
    edge_count = len(graph.edge_index)
    for name in ("edge_length", "edge_weight"):
        value = getattr(graph, name)
        _require_dtype(name, value, np.float64)
        if value.shape != (edge_count,) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must be finite nonnegative [E]")
    if np.any(graph.edge_weight <= 0):
        raise ValueError("Retained geometry edge weights must be positive")
    if np.any(graph.edge_length <= 0):
        raise ValueError("Retained geometry edges must have positive length")
    candidate_length = np.linalg.norm(points[graph.candidate_edge_index[:, 0]].astype(np.float64) - points[graph.candidate_edge_index[:, 1]], axis=1)
    if np.any(candidate_length <= 0):
        raise ValueError("Geometry candidates must exclude zero-length edges")
    geometric_length = np.linalg.norm(points[graph.edge_index[:, 0]].astype(np.float64) - points[graph.edge_index[:, 1]], axis=1)
    if not np.allclose(graph.edge_length, geometric_length, atol=1e-7, rtol=1e-5):
        raise ValueError("Geometry edge lengths disagree with point positions")
    _require_dtype("node_visible_view_mask", graph.node_visible_view_mask, bool)
    if graph.node_visible_view_mask.ndim != 2 or graph.node_visible_view_mask.shape[0] != count or not graph.node_visible_view_mask.shape[1]:
        raise ValueError("Geometry visibility must have shape [G,V] with V positive")
    view_count = graph.node_visible_view_mask.shape[1]
    for name, rows in (("edge_view_evidence", edge_count), ("candidate_view_evidence", len(graph.candidate_edge_index))):
        value = getattr(graph, name)
        _require_dtype(name, value, np.int8)
        if value.shape != (rows, view_count) or np.any((value < -1) | (value > 1)):
            raise ValueError(f"{name} must contain three-valued evidence in [E,V]")
    _require_dtype("edge_evidence_kind", graph.edge_evidence_kind, np.int8)
    if graph.edge_evidence_kind.shape != (edge_count,) or not np.isin(graph.edge_evidence_kind, (0, 1, EVIDENCE_SPATIAL_PRIOR)).all():
        raise ValueError("Retained edge evidence kind must be unknown, supported or an explicit spatial prior")
    spatial_only = graph.edge_evidence_kind == EVIDENCE_SPATIAL_PRIOR
    if np.any(spatial_only) and (not np.all(spatial_only) or np.any(graph.edge_view_evidence)
            or np.any(graph.candidate_view_evidence) or np.any(graph.node_visible_view_mask)
            or not np.array_equal(graph.edge_index, graph.candidate_edge_index)):
        raise ValueError("KNN-only graph must retain all candidates without fabricated depth evidence")
    if np.any(graph.edge_view_evidence[~spatial_only] < 0) or not np.array_equal(graph.edge_evidence_kind[~spatial_only], np.any(graph.edge_view_evidence[~spatial_only] == 1, axis=1).astype(np.int8)):
        raise ValueError("Retained edge evidence contradicts its classification")
    candidate_codes = graph.candidate_edge_index[:, 0] * count + graph.candidate_edge_index[:, 1]
    retained_codes = graph.edge_index[:, 0] * count + graph.edge_index[:, 1]
    slots = np.searchsorted(candidate_codes, retained_codes)
    if np.any(slots >= len(candidate_codes)):
        raise ValueError("Retained edge is missing from geometry candidates")
    if not np.array_equal(candidate_codes[slots], retained_codes) or not np.array_equal(graph.candidate_view_evidence[slots], graph.edge_view_evidence):
        raise ValueError("Retained edge evidence differs from its candidate evidence")
    jointly_visible = graph.node_visible_view_mask[graph.candidate_edge_index[:, 0]] & graph.node_visible_view_mask[graph.candidate_edge_index[:, 1]]
    if np.any((graph.candidate_view_evidence != 0) & ~jointly_visible):
        raise ValueError("A view cannot classify an edge with an invisible endpoint")
    degree, components, sizes = _components(count, graph.edge_index)
    if not np.array_equal(graph.degree, degree) or not np.array_equal(graph.component_index, components) or not np.array_equal(graph.component_size, sizes):
        raise ValueError("Geometry component or degree metadata disagrees with edges")


def _validate_control_arrays(graph: ControlGraph) -> None:
    for name in ("control_point_index", "interpolation_indptr", "interpolation_indices", "owner", "component_index"):
        _require_dtype(name, getattr(graph, name), np.int64)
    _require_dtype("positions", graph.positions, np.float32)
    indices = graph.control_point_index
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise ValueError("Control point indices must be nonempty unique nonnegative indices")
    count = len(indices)
    if graph.positions.shape != (count, 3) or not np.isfinite(graph.positions).all():
        raise ValueError("Control positions must be finite [J,3]")
    if not math.isfinite(graph.coverage_radius) or graph.coverage_radius <= 0 or not math.isfinite(graph.scene_scale) or graph.scene_scale <= 0:
        raise ValueError("Control graph scale and coverage radius must be finite positive scalars")
    _validate_edges("control_edges", graph.control_edges, count)
    for name in ("control_edge_length", "control_edge_weight"):
        value = getattr(graph, name)
        _require_dtype(name, value, np.float64)
        if value.shape != (len(graph.control_edges),) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must be finite nonnegative [E]")
    if np.any(graph.control_edge_weight <= 0):
        raise ValueError("Control edge weights must be positive")
    if np.any(graph.control_edge_length <= 0):
        raise ValueError("Control edges must have positive graph-path lengths")
    indptr = graph.interpolation_indptr
    if indptr.ndim != 1 or len(indptr) < 2 or indptr[0] != 0 or np.any(np.diff(indptr) <= 0):
        raise ValueError("Interpolation CSR must have nonempty rows starting at offset zero")
    foreground_count = len(indptr) - 1
    if np.any(indices >= foreground_count):
        raise ValueError("Control point index exceeds the foreground count")
    columns, weights = graph.interpolation_indices, graph.interpolation_weights
    _require_dtype("interpolation_weights", weights, np.float64)
    if columns.shape != (int(indptr[-1]),) or weights.shape != columns.shape or np.any(columns < 0) or np.any(columns >= count):
        raise ValueError("Interpolation CSR indices are invalid")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Interpolation weights must be finite and positive")
    row_sum = np.add.reduceat(weights, indptr[:-1])
    if not np.allclose(row_sum, 1, atol=1e-10, rtol=1e-10):
        raise ValueError("Interpolation rows must sum to one")
    for start, end in zip(indptr[:-1], indptr[1:]):
        if np.any(np.diff(columns[start:end]) <= 0):
            raise ValueError("Interpolation columns must be sorted and unique in every row")
    _require_dtype("owner_distance", graph.owner_distance, np.float64)
    if graph.owner.shape != (foreground_count,) or np.any(graph.owner < 0) or np.any(graph.owner >= count):
        raise ValueError("Control owners must be valid control indices for every Gaussian")
    if graph.owner_distance.shape != (foreground_count,) or not np.isfinite(graph.owner_distance).all() or np.any(graph.owner_distance < 0) or np.any(graph.owner_distance > graph.coverage_radius + 1e-10):
        raise ValueError("Control owner distances violate foreground coverage")
    if graph.component_index.shape != (count,) or np.any(graph.component_index < 0):
        raise ValueError("Control component labels must be nonnegative [J]")
    unique_components = np.unique(graph.component_index)
    if not np.array_equal(unique_components, np.arange(len(unique_components), dtype=np.int64)):
        raise ValueError("Control graph must represent every component with contiguous labels")
    row_components = graph.component_index[graph.owner]
    if np.any(graph.component_index[columns] != np.repeat(row_components, np.diff(indptr))):
        raise ValueError("Interpolation crosses disconnected components")
    if np.any(graph.component_index[graph.control_edges[:, 0]] != graph.component_index[graph.control_edges[:, 1]]):
        raise ValueError("Control edges cross disconnected components")


def depth_thresholds_from_manifest(
    manifest: Mapping[str, Any], labels: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read only fixed depth tolerances from an old graph, never its edges/colors.

    An unavailable tolerance remains NaN and makes that view unknown.  Label
    matching prevents accidentally applying another camera's thresholds.
    """
    views = manifest.get("thresholds", {}).get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("Source graph has no per-view fixed depth thresholds")
    by_label: dict[str, Mapping[str, Any]] = {}
    for view in views:
        if not isinstance(view, Mapping) or not isinstance(view.get("label"), str):
            raise ValueError("Source depth threshold view is invalid")
        label = view["label"]
        if label in by_label:
            raise ValueError("Source depth threshold labels must be unique")
        by_label[label] = view
    order = list(by_label) if labels is None else list(labels)
    if not order or len(set(order)) != len(order) or any(label not in by_label for label in order):
        raise ValueError("Requested camera labels do not match fixed depth thresholds")
    result = []
    for key in ("endpoint_gap", "depth_jump"):
        values = []
        for label in order:
            payload = by_label[label].get(key)
            if not isinstance(payload, Mapping) or "threshold" not in payload:
                raise ValueError(f"Missing fixed {key} threshold for {label}")
            value = payload["threshold"]
            if value is None:
                values.append(math.nan)
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid fixed {key} threshold for {label}")
            else:
                values.append(float(value))
        result.append(np.asarray(values, dtype=np.float64))
    return result[0], result[1]


def _mutual_knn(points: np.ndarray, config: GeometryGraphConfig) -> tuple[np.ndarray, np.ndarray]:
    count = len(points)
    if count < 2:
        return np.empty((0, 2), np.int64), np.empty(0, np.float64)
    k = min(config.max_neighbors, count - 1)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1, workers=1)
    radii = np.nextafter(np.asarray(distances)[:, -1], np.inf)
    rows = tree.query_ball_point(points, radii, return_sorted=False)
    directed: set[tuple[int, int]] = set()
    for i, row in enumerate(rows):
        candidates = np.asarray(row, dtype=np.int64)
        candidates = candidates[candidates != i]
        lengths = np.linalg.norm(points[candidates] - points[i], axis=1)
        order = np.lexsort((candidates, lengths))[:k]
        for target, length in zip(candidates[order], lengths[order]):
            if 0 < length <= config.max_distance:
                directed.add((i, int(target)))
    edges = np.asarray(sorted((i, j) for i, j in directed if i < j and (j, i) in directed), dtype=np.int64).reshape(-1, 2)
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    return edges, lengths


def _components(count: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent = np.arange(count, dtype=np.int64)
    degree = np.bincount(edges.ravel(), minlength=count).astype(np.int64)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for start, end in edges:
        a, b = root(int(start)), root(int(end))
        parent[max(a, b)] = min(a, b)
    roots = np.asarray([root(i) for i in range(count)], dtype=np.int64)
    _, component = np.unique(roots, return_inverse=True)
    component = component.astype(np.int64)
    return degree, component, np.bincount(component).astype(np.int64)


def build_geometry_graph_arrays(
    *, foreground_means: np.ndarray, Ks: np.ndarray, world_to_cameras: np.ndarray,
    rendered_depths: Sequence[np.ndarray], rendered_alphas: Sequence[np.ndarray],
    endpoint_thresholds: np.ndarray, depth_jump_thresholds: np.ndarray,
    config: GeometryGraphConfig | None = None,
) -> GeometryGraph:
    """Build all-foreground mutual KNN with fixed static depth-path evidence.

    No view with an occluded endpoint can contradict an edge.  A jointly visible
    view contradicts a path through invalid foreground coverage or a depth jump.
    Contradiction in any such view vetoes support from another view.
    """
    settings = config or GeometryGraphConfig()
    settings.validate()
    points = np.asarray(foreground_means, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("Geometry points must be nonempty finite [G,3]")
    Ks = np.asarray(Ks, dtype=np.float64)
    poses = np.asarray(world_to_cameras, dtype=np.float64)
    view_count = len(rendered_depths)
    if view_count == 0 or len(rendered_alphas) != view_count:
        raise ValueError("Geometry graph requires matching nonempty depth/alpha views")
    if Ks.shape != (view_count, 3, 3) or poses.shape != (view_count, 4, 4) or not np.isfinite(Ks).all() or not np.isfinite(poses).all():
        raise ValueError("Geometry camera arrays must be finite [V,3,3] and [V,4,4]")
    endpoint = np.asarray(endpoint_thresholds, dtype=np.float64)
    jump = np.asarray(depth_jump_thresholds, dtype=np.float64)
    for name, value in (("endpoint", endpoint), ("depth jump", jump)):
        if value.shape != (view_count,) or np.any(np.isinf(value)) or np.any(value < 0):
            raise ValueError(f"Fixed {name} thresholds must be nonnegative [V], with NaN for unavailable")
    edges, lengths = _mutual_knn(points, settings)
    evidence = np.zeros((len(edges), view_count), dtype=np.int8)
    visibility = np.zeros((len(points), view_count), dtype=bool)
    if settings.edge_filter == "none":
        degree, component, sizes = _components(len(points), edges)
        weights = 1.0 / np.sqrt(degree[edges[:, 0]].astype(np.float64) * degree[edges[:, 1]])
        return GeometryGraph(
            points=points.astype(np.float32), node_gaussian_index=np.arange(len(points), dtype=np.int64),
            edge_index=edges, edge_length=lengths, edge_weight=weights,
            edge_evidence_kind=np.full(len(edges), EVIDENCE_SPATIAL_PRIOR, dtype=np.int8),
            edge_view_evidence=evidence, node_visible_view_mask=visibility,
            degree=degree, component_index=component, component_size=sizes,
            candidate_edge_index=edges.copy(), candidate_view_evidence=evidence.copy(),
        )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    for view in range(view_count):
        depth = np.asarray(rendered_depths[view], dtype=np.float64)
        alpha = np.asarray(rendered_alphas[view], dtype=np.float64)
        if depth.ndim != 2 or min(depth.shape) < 2 or alpha.shape != depth.shape:
            raise ValueError("Rendered depth/alpha must have matching 2D shapes of at least 2x2")
        if not np.isfinite(depth).all() or not np.isfinite(alpha).all() or np.any(depth < 0) or np.any(alpha < 0) or np.any(alpha > 1):
            raise ValueError("Rendered depth/alpha must be finite, depth >=0 and alpha in [0,1]")
        if not np.isfinite(endpoint[view]) or not np.isfinite(jump[view]):
            continue
        camera = homogeneous @ poses[view].T
        z = camera[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = np.column_stack((Ks[view, 0, 0] * camera[:, 0] / z + Ks[view, 0, 2],
                                      Ks[view, 1, 1] * camera[:, 1] / z + Ks[view, 1, 2]))
        in_frame = _pixel_valid(pixels, depth.shape) & (z > 0)
        valid_indices = np.flatnonzero(in_frame)
        sampled_depth = _bilinear(depth, pixels[valid_indices])
        sampled_alpha = _bilinear(alpha, pixels[valid_indices])
        relative_gap = np.abs(z[valid_indices] - sampled_depth) / np.maximum(sampled_depth, 1e-8)
        visible = ((sampled_depth > 0) & (sampled_alpha >= settings.alpha_minimum)
                   & (relative_gap <= endpoint[view] + 1e-8))
        visibility[valid_indices[visible], view] = True
        selected = np.flatnonzero(visibility[edges[:, 0], view] & visibility[edges[:, 1], view])
        if not len(selected):
            continue
        projected_lengths = np.linalg.norm(pixels[edges[selected, 0]] - pixels[edges[selected, 1]], axis=1)
        samples_per_edge = np.maximum(settings.profile_min_samples,
                                      np.ceil(projected_lengths / settings.profile_max_step_pixels).astype(np.int64) + 1)
        for sample_count in np.unique(samples_per_edge):
            matching = selected[samples_per_edge == sample_count]
            fractions = np.linspace(0, 1, int(sample_count), dtype=np.float64)
            # Bound temporary path buffers even for unusually long projected edges.
            batch_size = max(1, min(4096, 1_000_000 // int(sample_count)))
            for offset in range(0, len(matching), batch_size):
                rows = matching[offset:offset + batch_size]
                p0, p1 = pixels[edges[rows, 0]], pixels[edges[rows, 1]]
                path = p0[:, None] * (1 - fractions[None, :, None]) + p1[:, None] * fractions[None, :, None]
                path_depth = _bilinear(depth, path.reshape(-1, 2)).reshape(len(rows), -1)
                path_alpha = _bilinear(alpha, path.reshape(-1, 2)).reshape(len(rows), -1)
                denominator = np.maximum(0.5 * (path_depth[:, 1:] + path_depth[:, :-1]), 1e-8)
                depth_jump = np.max(np.abs(np.diff(path_depth, axis=1)) / denominator, axis=1)
                support = ((path_alpha >= settings.alpha_minimum).all(axis=1)
                           & (path_depth > 0).all(axis=1) & (depth_jump <= jump[view] + 1e-8))
                evidence[rows, view] = np.where(support, EVIDENCE_SUPPORTED, EVIDENCE_CONTRADICTED)
    contradicted = np.any(evidence == EVIDENCE_CONTRADICTED, axis=1)
    supported = np.any(evidence == EVIDENCE_SUPPORTED, axis=1) & ~contradicted
    unknown = ~np.any(evidence != EVIDENCE_UNKNOWN, axis=1)
    retained = supported | (unknown & (lengths <= settings.unknown_max_distance))
    final_edges, final_lengths = edges[retained], lengths[retained]
    kind = np.where(supported[retained], EVIDENCE_SUPPORTED, EVIDENCE_UNKNOWN).astype(np.int8)
    degree, component, sizes = _components(len(points), final_edges)
    factors = np.where(kind == EVIDENCE_SUPPORTED, 1.0, settings.unknown_weight)
    weights = factors / np.sqrt(degree[final_edges[:, 0]].astype(np.float64) * degree[final_edges[:, 1]])
    return GeometryGraph(
        points=points.astype(np.float32), node_gaussian_index=np.arange(len(points), dtype=np.int64),
        edge_index=final_edges, edge_length=final_lengths, edge_weight=weights,
        edge_evidence_kind=kind, edge_view_evidence=evidence[retained],
        node_visible_view_mask=visibility, degree=degree, component_index=component,
        component_size=sizes, candidate_edge_index=edges, candidate_view_evidence=evidence,
    )


def build_control_graph(
    graph: GeometryGraph, config: GeometryGraphConfig | None = None,
    *, scene_scale: float | None = None,
) -> ControlGraph:
    """Graph-distance FPS, all-support Wendland CSR, and Voronoi adjacency.

    Every component is seeded by its smallest Gaussian index.  FPS then selects
    the global farthest node, breaking distance ties by Gaussian index.  A hard
    control budget never silently discards components or relaxes coverage.
    """
    settings = config or GeometryGraphConfig()
    settings.validate()
    points = np.asarray(graph.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Control graph requires nonempty finite [G,3] points")
    scale = float(np.linalg.norm(np.ptp(points, axis=0))) if scene_scale is None else float(scene_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Control graph scene scale L must be positive; foreground extent is degenerate")
    h = settings.control_radius_fraction * scale
    if not math.isfinite(h) or h <= 0:
        raise ValueError("Control coverage radius h must be finite and positive")
    edges = np.asarray(graph.edge_index, dtype=np.int64)
    lengths = np.asarray(graph.edge_length, dtype=np.float64)
    if edges.shape != (len(lengths), 2) or np.any(edges < 0) or np.any(edges >= len(points)) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("Control source graph has invalid edges or lengths")
    _, component, sizes = _components(len(points), edges)
    if not np.array_equal(component, graph.component_index):
        raise ValueError("Control source component labels disagree with its edges")
    if len(sizes) > settings.max_controls:
        raise ValueError(f"Control budget {settings.max_controls} cannot cover {len(sizes)} disconnected components (including isolated nodes)")
    adjacency = coo_matrix((np.concatenate((lengths, lengths)),
                            (np.concatenate((edges[:, 0], edges[:, 1])),
                             np.concatenate((edges[:, 1], edges[:, 0])))), shape=(len(points), len(points))).tocsr()
    nearest = np.full(len(points), np.inf, dtype=np.float64)
    owners = np.full(len(points), -1, dtype=np.int64)
    control_nodes: list[int] = []
    support_rows: list[np.ndarray] = []
    support_columns: list[np.ndarray] = []
    support_values: list[np.ndarray] = []
    control_distances: list[np.ndarray] = []

    def add_control(node: int) -> None:
        slot = len(control_nodes)
        distances = np.asarray(dijkstra(adjacency, directed=False, indices=node), dtype=np.float64)
        control_distances.append(distances[np.asarray(control_nodes, dtype=np.int64)].copy())
        prior_nodes = np.full(len(points), np.iinfo(np.int64).max, dtype=np.int64)
        assigned = owners >= 0
        if control_nodes:
            prior_nodes[assigned] = np.asarray(control_nodes)[owners[assigned]]
        take = (distances < nearest) | ((distances == nearest) & np.isfinite(distances) & (node < prior_nodes))
        nearest[take] = distances[take]
        owners[take] = slot
        control_nodes.append(node)
        rows = np.flatnonzero(distances < 2 * h)
        ratio = distances[rows] / (2 * h)
        weights = (1 - ratio) ** 4 * (4 * ratio + 1)
        support_rows.append(rows)
        support_columns.append(np.full(len(rows), slot, dtype=np.int64))
        support_values.append(weights)

    _, first_indices = np.unique(component, return_index=True)
    for seed in first_indices:
        add_control(int(seed))
    while np.max(nearest) > h:
        if len(control_nodes) >= settings.max_controls:
            uncovered = int(np.count_nonzero(nearest > h))
            raise ValueError(f"Control budget {settings.max_controls} cannot satisfy h={h:.9g}: {uncovered} foreground nodes remain uncovered (max distance {np.max(nearest):.9g})")
        add_control(int(np.argmax(nearest)))
    controls = np.asarray(control_nodes, dtype=np.int64)
    interpolation = coo_matrix((np.concatenate(support_values),
                                (np.concatenate(support_rows), np.concatenate(support_columns))),
                               shape=(len(points), len(controls))).tocsr()
    totals = np.asarray(interpolation.sum(axis=1)).ravel()
    if np.any(totals <= 0) or not np.isfinite(totals).all():
        raise RuntimeError("Wendland interpolation failed to cover every foreground node")
    interpolation.data /= np.repeat(totals, np.diff(interpolation.indptr))
    interpolation.sort_indices()

    boundary: dict[tuple[int, int], float] = {}
    for edge_slot, (a, b) in enumerate(edges):
        owner_a, owner_b = int(owners[a]), int(owners[b])
        if owner_a == owner_b:
            continue
        key = (min(owner_a, owner_b), max(owner_a, owner_b))
        boundary[key] = boundary.get(key, 0.0) + float(graph.edge_weight[edge_slot])
    control_edges = np.asarray(sorted(boundary), dtype=np.int64).reshape(-1, 2)
    control_lengths = np.asarray([control_distances[int(b)][int(a)] for a, b in control_edges], dtype=np.float64)
    control_weights = np.asarray([boundary[tuple(pair)] for pair in control_edges], dtype=np.float64)
    return ControlGraph(
        control_point_index=controls, positions=points[controls].astype(np.float32),
        control_edges=control_edges, control_edge_length=control_lengths, control_edge_weight=control_weights,
        interpolation_indptr=interpolation.indptr.astype(np.int64),
        interpolation_indices=interpolation.indices.astype(np.int64),
        interpolation_weights=interpolation.data.astype(np.float64), owner=owners,
        owner_distance=nearest, component_index=component[controls], coverage_radius=h, scene_scale=scale,
    )


__all__ = ["GeometryGraphConfig", "GeometryGraph", "ControlGraph", "build_geometry_graph_arrays",
           "build_control_graph", "depth_thresholds_from_manifest", "EVIDENCE_SUPPORTED",
           "EVIDENCE_UNKNOWN", "EVIDENCE_CONTRADICTED"]
