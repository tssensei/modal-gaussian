"""Complex first-order SE(3) solve over observed graph components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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
from modal_gaussians.static import load_static_scene
from modal_gaussians.structure_graph import (
    StructureGraphArrays,
    load_observed_structure_graph,
)
from modal_gaussians.synchronization import (
    AlphaSyncConfig,
    AlphaSyncResult,
    PreparedObservations,
    prepare_observations,
    solve_alpha_sync,
)
from modal_gaussians.topology import load_observation_topology
from modal_gaussians.progress import Progress, report_progress


EPSILON = 1.0e-12
FINITE_DRIFT_EDGE_BATCH_SIZE = 32_768
RIGID_MODES_FORMAT = "modal_gaussians.rigid_modes"
RIGID_MODES_VERSION = 1
RIGID_MODES_FILENAME = "rigid_modes.npz"
WORK_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class RigidComponentConfig:
    """Hold the accepted first-order rigid-component solve settings."""

    rcond: float = 1.0e-8
    first_order_rtol: float = 1.0e-6
    phase_samples: int = 64

    def validate(self) -> None:
        """Reject invalid SVD and first-order validation settings."""

        if not math.isfinite(self.rcond) or not 0.0 < self.rcond < 1.0:
            raise ValueError("Rigid rcond must lie in (0,1)")
        if not math.isfinite(self.first_order_rtol) or self.first_order_rtol <= 0.0:
            raise ValueError("Rigid first_order_rtol must be finite and positive")
        if (
            isinstance(self.phase_samples, bool)
            or not isinstance(self.phase_samples, int)
            or self.phase_samples <= 0
        ):
            raise ValueError("Rigid phase_samples must be positive")

    def to_dict(self) -> dict[str, float | int | str]:
        """Serialize the fixed complex infinitesimal-SE(3) convention."""

        return {
            "model": "complex_first_order_se3_per_connected_component",
            "rcond": self.rcond,
            "first_order_rtol": self.first_order_rtol,
            "phase_samples": self.phase_samples,
        }


@dataclass(frozen=True)
class RigidSeedConfig:
    """Hold accepted quality thresholds for trusted rigid seed components."""

    minimum_valid_views: int = 2
    minimum_secondary_view_node_ratio: float = 1.0 / 3.0
    minimum_singular_ratio: float = 1.0e-3
    maximum_finite_drift: float = 2.0

    def validate(self) -> None:
        """Reject invalid trusted-seed selection settings."""

        if (
            isinstance(self.minimum_valid_views, bool)
            or not isinstance(self.minimum_valid_views, int)
            or self.minimum_valid_views <= 0
        ):
            raise ValueError("Rigid minimum_valid_views must be positive")
        for name, value in (
            (
                "minimum_secondary_view_node_ratio",
                self.minimum_secondary_view_node_ratio,
            ),
            ("minimum_singular_ratio", self.minimum_singular_ratio),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Rigid {name} must lie in [0,1]")
        if (
            not math.isfinite(self.maximum_finite_drift)
            or self.maximum_finite_drift < 0.0
        ):
            raise ValueError("Rigid maximum_finite_drift must be non-negative")

    def to_dict(self) -> dict[str, float | int]:
        """Serialize the accepted trusted-component thresholds."""

        return {
            "minimum_valid_views": self.minimum_valid_views,
            "minimum_secondary_view_node_ratio": (
                self.minimum_secondary_view_node_ratio
            ),
            "minimum_singular_ratio": self.minimum_singular_ratio,
            "maximum_finite_drift": self.maximum_finite_drift,
        }


@dataclass(frozen=True)
class RigidComponentResult:
    """Store raw rigid fields and component/edge quality diagnostics."""

    phi: np.ndarray
    rigid_seed_mask: np.ndarray
    observed_mask: np.ndarray
    point_component_index: np.ndarray
    component_graph_index: np.ndarray
    component_node_count: np.ndarray
    component_edge_count: np.ndarray
    component_centroid: np.ndarray
    component_radius: np.ndarray
    component_usable_observation_row_count: np.ndarray
    component_valid_view_node_count: np.ndarray
    component_distinct_valid_view_count: np.ndarray
    component_singular_values: np.ndarray
    component_rank: np.ndarray
    component_condition: np.ndarray
    component_normalized_weighted_residual: np.ndarray
    component_translation: np.ndarray
    component_rotation: np.ndarray
    edge_component_index: np.ndarray
    edge_model_first_order_relative_real: np.ndarray
    edge_model_first_order_relative_imag: np.ndarray
    edge_first_order_relative_real: np.ndarray
    edge_first_order_relative_imag: np.ndarray
    edge_finite_drift_max: np.ndarray
    component_finite_drift_p50: np.ndarray
    component_finite_drift_p90: np.ndarray
    component_finite_drift_max: np.ndarray

    @property
    def component_count(self) -> int:
        """Return the number of connected graph components solved as rigid."""

        return len(self.component_node_count)


@dataclass(frozen=True)
class RigidSeedResult:
    """Store trusted rigid seeds and component rejection diagnostics."""

    phi: np.ndarray
    trusted_seed_mask: np.ndarray
    component_retained_mask: np.ndarray
    component_valid_view_rejected_mask: np.ndarray
    component_singular_rejected_mask: np.ndarray
    component_finite_drift_rejected_mask: np.ndarray
    component_singular_ratio: np.ndarray
    component_supported_valid_view_count: np.ndarray
    component_dominant_valid_view_index: np.ndarray
    component_secondary_view_node_ratio: np.ndarray


def _validate_inputs(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    graph: StructureGraphArrays,
) -> None:
    """Validate solver-facing graph, observation, and alpha index domains."""

    point_count = len(prepared.points)
    node_indices = np.asarray(graph.node_gaussian_index)
    edge_index = np.asarray(graph.edge_index)
    node_count = len(node_indices)
    if (
        node_indices.ndim != 1
        or np.any(node_indices < 0)
        or np.any(node_indices >= point_count)
        or not np.array_equal(node_indices, np.unique(node_indices))
    ):
        raise ValueError("Rigid graph node Gaussian indices are invalid")
    if edge_index.ndim != 2 or edge_index.shape[1:] != (2,):
        raise ValueError("Rigid graph edge_index must be [E,2]")
    if len(edge_index) and (
        np.any(edge_index < 0)
        or np.any(edge_index >= node_count)
        or np.any(edge_index[:, 0] >= edge_index[:, 1])
    ):
        raise ValueError("Rigid graph edge indices are invalid")
    if np.asarray(graph.degree).shape != (node_count,):
        raise ValueError("Rigid graph degree shape is invalid")
    if np.asarray(graph.component_index).shape != (node_count,):
        raise ValueError("Rigid graph component index shape is invalid")
    point_view = np.zeros((point_count, prepared.num_views), dtype=bool)
    positive = prepared.obs_weights > 0.0
    point_view[
        prepared.obs_point_index[positive],
        prepared.obs_view_index[positive],
    ] = True
    if not np.array_equal(
        graph.node_observed_view_mask, point_view[node_indices]
    ):
        raise ValueError("Rigid graph view support does not match observations")
    if alpha.alphas.shape != (prepared.num_views,):
        raise ValueError("Rigid alpha length does not match views")
    if alpha.identifiable_mask.shape != (prepared.num_views,):
        raise ValueError("Rigid alpha identifiability length does not match views")
    identifiable = alpha.alphas[alpha.identifiable_mask]
    if not np.isfinite(identifiable).all():
        raise ValueError("Identifiable rigid alpha values must be finite")


def _batch_skew(vectors: np.ndarray) -> np.ndarray:
    """Return skew matrices used by the infinitesimal rotation blocks."""

    skew = np.zeros((len(vectors), 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]
    return skew


def _minimum_norm_svd(
    design: np.ndarray,
    target: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve one complex twist with deterministic truncated minimum-norm SVD."""

    left, singular, right = np.linalg.svd(design, full_matrices=False)
    if len(singular) == 0 or singular[0] <= EPSILON:
        return np.zeros(design.shape[1], dtype=np.complex128), singular, 0
    rank = int(np.count_nonzero(singular > rcond * singular[0]))
    if rank == 0:
        solution = np.zeros(design.shape[1], dtype=np.complex128)
    else:
        coefficients = (left[:, :rank].conj().T @ target) / singular[:rank]
        solution = right[:rank].conj().T @ coefficients
    return solution.astype(np.complex128), singular, rank


def _finite_drift(
    edge_vectors: np.ndarray,
    edge_delta_phi: np.ndarray,
    phase_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure finite additive playback length drift over sampled phases."""

    edge_count = len(edge_vectors)
    p50 = np.zeros(edge_count, dtype=np.float64)
    p90 = np.zeros(edge_count, dtype=np.float64)
    maximum = np.zeros(edge_count, dtype=np.float64)
    cosine = np.cos(phase_angles)[None]
    sine = np.sin(phase_angles)[None]
    for start in range(0, edge_count, FINITE_DRIFT_EDGE_BATCH_SIZE):
        end = min(start + FINITE_DRIFT_EDGE_BATCH_SIZE, edge_count)
        base = edge_vectors[start:end]
        delta = edge_delta_phi[start:end]
        real, imag = np.real(delta), np.imag(delta)
        base_squared = np.einsum("ij,ij->i", base, base)[:, None]
        base_real = np.einsum("ij,ij->i", base, real)[:, None]
        base_imag = np.einsum("ij,ij->i", base, imag)[:, None]
        real_squared = np.einsum("ij,ij->i", real, real)[:, None]
        imag_squared = np.einsum("ij,ij->i", imag, imag)[:, None]
        real_imag = np.einsum("ij,ij->i", real, imag)[:, None]
        deformed_squared = (
            base_squared
            + 2.0 * (base_real * cosine - base_imag * sine)
            + real_squared * cosine**2
            + imag_squared * sine**2
            - 2.0 * real_imag * cosine * sine
        )
        drift = np.abs(
            np.sqrt(np.maximum(deformed_squared, 0.0))
            - np.sqrt(np.maximum(base_squared, 0.0))
        ) / np.maximum(np.sqrt(np.maximum(base_squared, 0.0)), EPSILON)
        p50[start:end] = np.percentile(drift, 50.0, axis=1)
        p90[start:end] = np.percentile(drift, 90.0, axis=1)
        maximum[start:end] = np.max(drift, axis=1)
    return p50, p90, maximum


def solve_rigid_components(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    graph: StructureGraphArrays,
    config: RigidComponentConfig | None = None,
) -> RigidComponentResult:
    """Solve one complex first-order SE(3) twist per connected graph component."""

    config = config or RigidComponentConfig()
    config.validate()
    _validate_inputs(prepared, alpha, graph)

    points = np.asarray(prepared.points, dtype=np.float64)
    point_count = len(points)
    node_indices = np.asarray(graph.node_gaussian_index, dtype=np.int64)
    degree = np.asarray(graph.degree, dtype=np.int64)
    graph_component = np.asarray(graph.component_index, dtype=np.int64)
    edge_index = np.asarray(graph.edge_index, dtype=np.int64)
    rigid_node_mask = degree > 0
    selected_graph_components = np.unique(graph_component[rigid_node_mask])
    component_count = len(selected_graph_components)
    if component_count == 0:
        raise ValueError("Observed graph has no component with an accepted edge")

    graph_to_rigid = np.full(int(graph_component.max()) + 1, -1, dtype=np.int32)
    graph_to_rigid[selected_graph_components] = np.arange(
        component_count, dtype=np.int32
    )
    node_component = np.full(len(node_indices), -1, dtype=np.int32)
    node_component[rigid_node_mask] = graph_to_rigid[
        graph_component[rigid_node_mask]
    ]
    rigid_point_indices = node_indices[rigid_node_mask]
    rigid_seed_mask = np.zeros(point_count, dtype=bool)
    rigid_seed_mask[rigid_point_indices] = True
    observed_mask = np.zeros(point_count, dtype=bool)
    observed_mask[node_indices] = True
    point_component = np.full(point_count, -1, dtype=np.int32)
    point_component[rigid_point_indices] = node_component[rigid_node_mask]

    component_node_count = np.bincount(
        node_component[rigid_node_mask], minlength=component_count
    ).astype(np.int32)
    if np.any(component_node_count < 2):
        raise RuntimeError("A rigid component contains fewer than two nodes")
    component_centroid = np.zeros((component_count, 3), dtype=np.float64)
    np.add.at(
        component_centroid,
        node_component[rigid_node_mask],
        points[rigid_point_indices],
    )
    component_centroid /= component_node_count[:, None]
    centered_points = (
        points[rigid_point_indices]
        - component_centroid[node_component[rigid_node_mask]]
    )
    squared_radius = np.zeros(component_count, dtype=np.float64)
    np.add.at(
        squared_radius,
        node_component[rigid_node_mask],
        np.einsum("ij,ij->i", centered_points, centered_points),
    )
    component_radius = np.sqrt(squared_radius / component_node_count)

    edge_component = node_component[edge_index[:, 0]]
    if np.any(edge_component < 0) or np.any(
        edge_component != node_component[edge_index[:, 1]]
    ):
        raise RuntimeError("Accepted edges do not map to one rigid component")
    component_edge_count = np.bincount(
        edge_component, minlength=component_count
    ).astype(np.int32)

    usable_mask = (
        (prepared.obs_weights > 0.0)
        & alpha.identifiable_mask[prepared.obs_view_index]
        & rigid_seed_mask[prepared.obs_point_index]
    )
    usable_rows = np.flatnonzero(usable_mask)
    usable_component = point_component[prepared.obs_point_index[usable_rows]]
    order = np.argsort(usable_component, kind="stable")
    usable_rows = usable_rows[order]
    usable_component = usable_component[order]
    component_row_count = np.bincount(
        usable_component, minlength=component_count
    ).astype(np.int32)
    row_offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(component_row_count)]
    )

    usable_points = prepared.obs_point_index[usable_rows].astype(np.int64)
    usable_views = prepared.obs_view_index[usable_rows].astype(np.int64)
    component_view_point_code = (
        (usable_component.astype(np.int64) * prepared.num_views + usable_views)
        * point_count
        + usable_points
    )
    unique_codes = np.unique(component_view_point_code)
    component_view_bins = unique_codes // point_count
    component_view_node_count = np.bincount(
        component_view_bins,
        minlength=component_count * prepared.num_views,
    ).reshape(component_count, prepared.num_views).astype(np.int32)
    component_view_count = np.count_nonzero(
        component_view_node_count, axis=1
    ).astype(np.int32)

    phi = np.zeros((point_count, 3), dtype=np.complex128)
    singular_values = np.zeros((component_count, 6), dtype=np.float64)
    rank = np.zeros(component_count, dtype=np.int8)
    condition = np.full(component_count, np.inf, dtype=np.float64)
    normalized_residual = np.zeros(component_count, dtype=np.float64)
    translation = np.zeros((component_count, 3), dtype=np.complex128)
    rotation = np.zeros((component_count, 3), dtype=np.complex128)

    rigid_order = np.argsort(node_component[rigid_node_mask], kind="stable")
    ordered_rigid_points = rigid_point_indices[rigid_order]
    node_offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(component_node_count)]
    )
    for component in range(component_count):
        rows = usable_rows[row_offsets[component] : row_offsets[component + 1]]
        if len(rows) == 0:
            continue
        row_points = prepared.obs_point_index[rows]
        centered = points[row_points] - component_centroid[component]
        point_blocks = np.zeros((len(rows), 3, 6), dtype=np.float64)
        point_blocks[:, :, :3] = np.eye(3, dtype=np.float64)[None]
        radius = float(component_radius[component])
        if radius > EPSILON:
            point_blocks[:, :, 3:] = -_batch_skew(centered) / radius
        projected = np.einsum(
            "rij,rjk->rik",
            prepared.obs_jacobian[rows].astype(np.float64),
            point_blocks,
        )
        sqrt_weight = np.sqrt(prepared.obs_weights[rows])
        row_alpha = alpha.alphas[prepared.obs_view_index[rows]].astype(
            np.complex128
        )
        design = (
            sqrt_weight[:, None, None]
            * row_alpha[:, None, None]
            * projected.astype(np.complex128)
        ).reshape(-1, 6)
        target = (
            sqrt_weight[:, None]
            * prepared.obs_y[rows].astype(np.complex128)
        ).reshape(-1)
        solution, singular, component_rank = _minimum_norm_svd(
            design, target, config.rcond
        )
        singular_values[component, : len(singular)] = singular
        rank[component] = component_rank
        if component_rank == 6:
            condition[component] = singular[0] / max(singular[5], EPSILON)
        translation[component] = solution[:3]
        if radius > EPSILON:
            rotation[component] = solution[3:] / radius
        residual = design @ solution - target
        normalized_residual[component] = np.linalg.norm(residual) / max(
            np.linalg.norm(target), EPSILON
        )
        component_points = ordered_rigid_points[
            node_offsets[component] : node_offsets[component + 1]
        ]
        phi[component_points] = translation[component] + np.cross(
            rotation[component][None],
            points[component_points] - component_centroid[component],
        )

    if not np.isfinite(phi).all():
        raise FloatingPointError("Rigid complex128 field is non-finite")
    persisted_phi = phi.astype(np.complex64)
    if not np.isfinite(persisted_phi).all():
        raise FloatingPointError("Rigid field overflowed during complex64 conversion")

    global_edges = node_indices[edge_index]
    edge_vectors = points[global_edges[:, 1]] - points[global_edges[:, 0]]
    model_delta = phi[global_edges[:, 1]] - phi[global_edges[:, 0]]
    model_axial = np.einsum("ij,ij->i", edge_vectors, model_delta)
    persisted_delta = (
        persisted_phi[global_edges[:, 1]].astype(np.complex128)
        - persisted_phi[global_edges[:, 0]].astype(np.complex128)
    )
    persisted_axial = np.einsum("ij,ij->i", edge_vectors, persisted_delta)
    squared_length = np.einsum("ij,ij->i", edge_vectors, edge_vectors)
    denominator = np.maximum(squared_length, EPSILON)
    model_relative_real = np.abs(model_axial.real) / denominator
    model_relative_imag = np.abs(model_axial.imag) / denominator
    maximum_model_relative = float(
        max(
            np.max(model_relative_real, initial=0.0),
            np.max(model_relative_imag, initial=0.0),
        )
    )
    if maximum_model_relative > config.first_order_rtol:
        raise RuntimeError(
            "Rigid complex128 field violates first-order edge rigidity: "
            f"{maximum_model_relative:.9g} > {config.first_order_rtol:.9g}"
        )

    cast_error = persisted_phi.astype(np.complex128) - phi
    source_error = cast_error[global_edges[:, 0]]
    target_error = cast_error[global_edges[:, 1]]
    absolute_vectors = np.abs(edge_vectors)
    bound_real = np.einsum(
        "ij,ij->i",
        absolute_vectors,
        np.abs(source_error.real) + np.abs(target_error.real),
    )
    bound_imag = np.einsum(
        "ij,ij->i",
        absolute_vectors,
        np.abs(source_error.imag) + np.abs(target_error.imag),
    )
    float64_epsilon = np.finfo(np.float64).eps
    real_scale = (
        np.abs(model_axial.real)
        + bound_real
        + np.einsum("ij,ij->i", absolute_vectors, np.abs(model_delta.real))
        + np.einsum("ij,ij->i", absolute_vectors, np.abs(persisted_delta.real))
    )
    imag_scale = (
        np.abs(model_axial.imag)
        + bound_imag
        + np.einsum("ij,ij->i", absolute_vectors, np.abs(model_delta.imag))
        + np.einsum("ij,ij->i", absolute_vectors, np.abs(persisted_delta.imag))
    )
    real_limit = (
        np.abs(model_axial.real)
        + bound_real
        + 64.0
        * float64_epsilon
        * np.maximum(real_scale, np.finfo(np.float64).tiny)
    )
    imag_limit = (
        np.abs(model_axial.imag)
        + bound_imag
        + 64.0
        * float64_epsilon
        * np.maximum(imag_scale, np.finfo(np.float64).tiny)
    )
    if np.max(np.abs(persisted_axial.real) - real_limit, initial=0.0) > 0.0 or np.max(
        np.abs(persisted_axial.imag) - imag_limit, initial=0.0
    ) > 0.0:
        raise RuntimeError("Rigid complex64 field exceeds its cast quantization bound")

    edge_relative_real = np.abs(persisted_axial.real) / denominator
    edge_relative_imag = np.abs(persisted_axial.imag) / denominator
    phase_angles = np.linspace(
        0.0, 2.0 * np.pi, config.phase_samples, endpoint=False, dtype=np.float64
    )
    _, _, edge_drift_max = _finite_drift(
        edge_vectors, persisted_delta, phase_angles
    )
    component_drift_p50 = np.zeros(component_count, dtype=np.float64)
    component_drift_p90 = np.zeros(component_count, dtype=np.float64)
    component_drift_max = np.zeros(component_count, dtype=np.float64)
    for component in range(component_count):
        values = edge_drift_max[edge_component == component]
        if len(values) == 0:
            raise RuntimeError("A rigid component unexpectedly has no edge")
        component_drift_p50[component] = np.percentile(values, 50.0)
        component_drift_p90[component] = np.percentile(values, 90.0)
        component_drift_max[component] = np.max(values)

    return RigidComponentResult(
        phi=persisted_phi,
        rigid_seed_mask=rigid_seed_mask,
        observed_mask=observed_mask,
        point_component_index=point_component,
        component_graph_index=selected_graph_components.astype(np.int32),
        component_node_count=component_node_count,
        component_edge_count=component_edge_count,
        component_centroid=component_centroid.astype(np.float32),
        component_radius=component_radius.astype(np.float32),
        component_usable_observation_row_count=component_row_count,
        component_valid_view_node_count=component_view_node_count,
        component_distinct_valid_view_count=component_view_count,
        component_singular_values=singular_values.astype(np.float32),
        component_rank=rank,
        component_condition=condition.astype(np.float32),
        component_normalized_weighted_residual=normalized_residual.astype(np.float32),
        component_translation=translation.astype(np.complex64),
        component_rotation=rotation.astype(np.complex64),
        edge_component_index=edge_component.astype(np.int32),
        edge_model_first_order_relative_real=model_relative_real.astype(np.float32),
        edge_model_first_order_relative_imag=model_relative_imag.astype(np.float32),
        edge_first_order_relative_real=edge_relative_real.astype(np.float32),
        edge_first_order_relative_imag=edge_relative_imag.astype(np.float32),
        edge_finite_drift_max=edge_drift_max.astype(np.float32),
        component_finite_drift_p50=component_drift_p50.astype(np.float32),
        component_finite_drift_p90=component_drift_p90.astype(np.float32),
        component_finite_drift_max=component_drift_max.astype(np.float32),
    )


def select_trusted_rigid_seeds(
    rigid: RigidComponentResult,
    config: RigidSeedConfig | None = None,
) -> RigidSeedResult:
    """Keep only multi-view, full-rank, finite-playback rigid components."""

    config = config or RigidSeedConfig()
    config.validate()
    component_count = rigid.component_count
    singular = np.asarray(rigid.component_singular_values, dtype=np.float64)
    rank = np.asarray(rigid.component_rank)
    if singular.shape != (component_count, 6) or not np.isfinite(singular).all():
        raise ValueError("Rigid component singular values are invalid")
    if np.any(singular < 0.0) or np.any(singular[:, 1:] > singular[:, :-1]):
        raise ValueError("Rigid singular values must be non-negative and descending")
    if rank.shape != (component_count,) or np.any(rank < 0) or np.any(rank > 6):
        raise ValueError("Rigid component rank is invalid")
    view_counts = np.asarray(rigid.component_valid_view_node_count)
    if (
        view_counts.ndim != 2
        or view_counts.shape[0] != component_count
        or view_counts.shape[1] < 1
        or np.any(view_counts < 0)
    ):
        raise ValueError("Rigid component view support is invalid")
    expected_distinct = np.count_nonzero(view_counts, axis=1)
    if not np.array_equal(rigid.component_distinct_valid_view_count, expected_distinct):
        raise ValueError("Rigid distinct view counts are inconsistent")

    dominant_view = np.argmax(view_counts, axis=1).astype(np.int32)
    dominant_count = np.max(view_counts, axis=1)
    supported = (view_counts > 0) & (
        view_counts
        >= dominant_count[:, None] * config.minimum_secondary_view_node_ratio
    )
    supported_count = np.count_nonzero(supported, axis=1).astype(np.int32)
    sorted_counts = np.sort(view_counts, axis=1)
    secondary_count = (
        sorted_counts[:, -2]
        if view_counts.shape[1] > 1
        else np.zeros(component_count, dtype=np.int64)
    )
    secondary_ratio = np.divide(
        secondary_count.astype(np.float64),
        dominant_count.astype(np.float64),
        out=np.zeros(component_count, dtype=np.float64),
        where=dominant_count > 0,
    )
    singular_ratio = np.zeros(component_count, dtype=np.float64)
    full_rank = (rank == 6) & (singular[:, 0] > 0.0)
    singular_ratio[full_rank] = singular[full_rank, 5] / singular[full_rank, 0]
    valid_view_rejected = supported_count < config.minimum_valid_views
    singular_rejected = singular_ratio < config.minimum_singular_ratio
    finite_drift = np.asarray(rigid.component_finite_drift_max, dtype=np.float64)
    if finite_drift.shape != (component_count,) or not np.isfinite(finite_drift).all():
        raise ValueError("Rigid component finite drift is invalid")
    finite_drift_rejected = finite_drift > config.maximum_finite_drift
    retained = ~(valid_view_rejected | singular_rejected | finite_drift_rejected)

    candidate_indices = np.flatnonzero(rigid.rigid_seed_mask)
    candidate_components = rigid.point_component_index[candidate_indices]
    if np.any(candidate_components < 0) or np.any(
        candidate_components >= component_count
    ):
        raise ValueError("Rigid seed component mapping is invalid")
    trusted_mask = np.zeros(rigid.rigid_seed_mask.shape, dtype=bool)
    trusted_mask[candidate_indices] = retained[candidate_components]
    trusted_phi = np.zeros(rigid.phi.shape, dtype=np.complex64)
    trusted_phi[trusted_mask] = rigid.phi[trusted_mask]
    return RigidSeedResult(
        phi=trusted_phi,
        trusted_seed_mask=trusted_mask,
        component_retained_mask=retained,
        component_valid_view_rejected_mask=valid_view_rejected,
        component_singular_rejected_mask=singular_rejected,
        component_finite_drift_rejected_mask=finite_drift_rejected,
        component_singular_ratio=singular_ratio.astype(np.float32),
        component_supported_valid_view_count=supported_count,
        component_dominant_valid_view_index=dominant_view,
        component_secondary_view_node_ratio=secondary_ratio.astype(np.float32),
    )


@dataclass(frozen=True)
class RigidModesArtifact:
    """Represent validated multi-mode alpha and rigid solver candidates."""

    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


MODE_ARRAY_FIELDS = (
    "phi",
    "trusted_phi",
    "trusted_seed_mask",
    "alphas",
    "alpha_identifiable_mask",
    "alpha_reference_connected_mask",
    "alpha_exclusion_reason",
    "alpha_shared_point_count",
    "alpha_edge_point_count",
    "alpha_edge_information",
    "alpha_constraint_count_per_view",
    "alpha_information_matrix",
    "alpha_rank_ratio",
    "alpha_information_ratio",
    "alpha_condition",
    "alpha_consistency_residual",
    "alpha_phase_std",
    "alpha_log_gain_std",
    "alpha_optimizer_success",
    "alpha_optimizer_status",
    "alpha_optimizer_message",
    "alpha_gain_bound_active_mask",
    "alpha_information_kind",
    "component_usable_observation_row_count",
    "component_valid_view_node_count",
    "component_distinct_valid_view_count",
    "component_singular_values",
    "component_rank",
    "component_condition",
    "component_normalized_weighted_residual",
    "component_translation",
    "component_rotation",
    "edge_model_first_order_relative_real",
    "edge_model_first_order_relative_imag",
    "edge_first_order_relative_real",
    "edge_first_order_relative_imag",
    "edge_finite_drift_max",
    "component_finite_drift_p50",
    "component_finite_drift_p90",
    "component_finite_drift_max",
    "component_retained_mask",
    "component_valid_view_rejected_mask",
    "component_singular_rejected_mask",
    "component_finite_drift_rejected_mask",
    "component_singular_ratio",
    "component_supported_valid_view_count",
    "component_dominant_valid_view_index",
    "component_secondary_view_node_ratio",
)

SHARED_ARRAY_FIELDS = (
    "rigid_seed_mask",
    "observed_mask",
    "point_component_index",
    "component_graph_index",
    "component_node_count",
    "component_edge_count",
    "component_centroid",
    "component_radius",
    "edge_component_index",
)


def _canonical_json(value: Any) -> bytes:
    """Encode one solver identity payload deterministically."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    """Hash one solver file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrays_identity(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash field names, dtypes, shapes, and raw array values canonically."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _solver_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select path-independent inputs and settings for work-dir compatibility."""

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
        "modes": manifest["modes"],
        "views": manifest["views"],
        "alpha_sync": manifest["alpha_sync"],
        "rigid_components": manifest["rigid_components"],
        "trusted_seeds": manifest["trusted_seeds"],
    }


def _artifact_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select scientific fields that define a final rigid-mode candidate."""

    payload = _solver_identity_payload(manifest)
    payload.update(
        {
            "format": RIGID_MODES_FORMAT,
            "version": RIGID_MODES_VERSION,
            "semantics": manifest["semantics"],
            "quality_gate": manifest["quality_gate"],
            "counts": manifest["counts"],
            "arrays_identity": manifest["arrays_identity"],
        }
    )
    return payload


def _source_manifest(
    *,
    scene_dir: str | Path,
    topology_dir: str | Path,
    measurements_dir: str | Path,
    graph_dir: str | Path,
    alpha_config: AlphaSyncConfig,
    rigid_config: RigidComponentConfig,
    seed_config: RigidSeedConfig,
) -> tuple[dict[str, Any], Any, Any, Any, Any]:
    """Load all upstream artifacts and enforce their immutable identity chain."""

    scene = load_static_scene(scene_dir, "cpu")
    topology = load_observation_topology(topology_dir)
    measurements = load_gaussian_measurements(measurements_dir)
    graph = load_observed_structure_graph(graph_dir)
    scene_manifest = scene.manifest
    if scene_manifest is None:
        raise ValueError("Static scene has no manifest")
    scene_identity = scene_manifest["static_scene_identity"]
    foreground_identity = scene_manifest["foreground_identity"]
    topology_identity = topology.manifest["topology_identity"]
    for name, value in (
        ("topology static scene", topology.manifest["static_scene_identity"]),
        ("graph static scene", graph.manifest["static_scene_identity"]),
    ):
        if value != scene_identity:
            raise ValueError(f"{name} identity does not match the static scene")
    for name, value in (
        ("topology foreground", topology.manifest["foreground_identity"]),
        ("graph foreground", graph.manifest["foreground_identity"]),
    ):
        if value != foreground_identity:
            raise ValueError(f"{name} identity does not match the static foreground")
    if measurements.manifest["topology_identity"] != topology_identity:
        raise ValueError("Measurements do not belong to the supplied topology")
    if graph.manifest["topology_identity"] != topology_identity:
        raise ValueError("Observed graph does not belong to the supplied topology")

    topology_views = topology.manifest["views"]
    measurement_views = measurements.manifest["views"]
    graph_views = graph.manifest["views"]
    if not (
        len(topology_views) == len(measurement_views) == len(graph_views)
    ):
        raise ValueError("Solver inputs have different view counts")
    views: list[dict[str, Any]] = []
    for index, (topology_view, measurement_view, graph_view) in enumerate(
        zip(topology_views, measurement_views, graph_views)
    ):
        label = topology_view["label"]
        for source, view in (
            ("measurement", measurement_view),
            ("graph", graph_view),
        ):
            if view["index"] != index or view["label"] != label:
                raise ValueError(f"{source} view order does not match topology")
            if view["flow_identity"] != topology_view["flow_identity"]:
                raise ValueError(f"{source} flow identity for {label!r} differs")
            if view["shape_hw"] != topology_view["shape_hw"]:
                raise ValueError(f"{source} shape for {label!r} differs")
        if graph_view["camera_identity"] != topology_view["camera_identity"]:
            raise ValueError(f"Graph camera identity for {label!r} differs")
        views.append(
            {
                "index": index,
                "label": label,
                "shape_hw": list(topology_view["shape_hw"]),
                "camera_identity": topology_view["camera_identity"],
                "flow_identity": topology_view["flow_identity"],
                "sample_count": int(measurement_view["sample_count"]),
            }
        )
    modes = [
        {
            "mode_slot": int(mode["mode_slot"]),
            "candidate_index": int(mode["candidate_index"]),
            "frequency_hz": float(mode["frequency_hz"]),
        }
        for mode in measurements.manifest["modes"]
    ]
    source = {
        "static_scene": str(scene_dir),
        "static_scene_identity": scene_identity,
        "foreground_identity": foreground_identity,
        "topology": str(topology.path),
        "topology_identity": topology_identity,
        "measurements": str(measurements.path),
        "gaussian_measurements_identity": measurements.manifest[
            "gaussian_measurements_identity"
        ],
        "observed_structure_graph": str(graph.path),
        "observed_structure_graph_identity": graph.manifest[
            "observed_structure_graph_identity"
        ],
        "modes": modes,
        "views": views,
        "alpha_sync": alpha_config.to_dict(),
        "rigid_components": rigid_config.to_dict(),
        "trusted_seeds": seed_config.to_dict(),
    }
    source["solver_run_identity"] = hashlib.sha256(
        _canonical_json(_solver_identity_payload(source))
    ).hexdigest()
    return source, scene, topology, measurements, graph


def _mode_arrays(
    alpha: AlphaSyncResult,
    rigid: RigidComponentResult,
    trusted: RigidSeedResult,
) -> dict[str, np.ndarray]:
    """Flatten one mode's fixed-shape scientific result for checkpointing."""

    return {
        "phi": rigid.phi,
        "trusted_phi": trusted.phi,
        "trusted_seed_mask": trusted.trusted_seed_mask,
        "alphas": alpha.alphas,
        "alpha_identifiable_mask": alpha.identifiable_mask,
        "alpha_reference_connected_mask": alpha.reference_connected_mask,
        "alpha_exclusion_reason": alpha.exclusion_reason,
        "alpha_shared_point_count": alpha.shared_point_count,
        "alpha_edge_point_count": alpha.edge_point_count,
        "alpha_edge_information": alpha.edge_information,
        "alpha_constraint_count_per_view": alpha.constraint_count_per_view,
        "alpha_information_matrix": alpha.information_matrix,
        "alpha_rank_ratio": np.asarray(alpha.rank_ratio, dtype=np.float32),
        "alpha_information_ratio": np.asarray(
            alpha.information_ratio, dtype=np.float32
        ),
        "alpha_condition": np.asarray(alpha.condition, dtype=np.float32),
        "alpha_consistency_residual": np.asarray(
            alpha.consistency_residual, dtype=np.float32
        ),
        "alpha_phase_std": alpha.phase_std,
        "alpha_log_gain_std": alpha.log_gain_std,
        "alpha_optimizer_success": np.asarray(alpha.optimizer_success, dtype=bool),
        "alpha_optimizer_status": np.asarray(alpha.optimizer_status, dtype=np.int32),
        "alpha_optimizer_message": np.asarray(alpha.optimizer_message, dtype="<U512"),
        "alpha_gain_bound_active_mask": alpha.gain_bound_active_mask,
        "alpha_information_kind": np.asarray(alpha.information_kind, dtype="<U64"),
        "component_usable_observation_row_count": (
            rigid.component_usable_observation_row_count
        ),
        "component_valid_view_node_count": rigid.component_valid_view_node_count,
        "component_distinct_valid_view_count": (
            rigid.component_distinct_valid_view_count
        ),
        "component_singular_values": rigid.component_singular_values,
        "component_rank": rigid.component_rank,
        "component_condition": rigid.component_condition,
        "component_normalized_weighted_residual": (
            rigid.component_normalized_weighted_residual
        ),
        "component_translation": rigid.component_translation,
        "component_rotation": rigid.component_rotation,
        "edge_model_first_order_relative_real": (
            rigid.edge_model_first_order_relative_real
        ),
        "edge_model_first_order_relative_imag": (
            rigid.edge_model_first_order_relative_imag
        ),
        "edge_first_order_relative_real": rigid.edge_first_order_relative_real,
        "edge_first_order_relative_imag": rigid.edge_first_order_relative_imag,
        "edge_finite_drift_max": rigid.edge_finite_drift_max,
        "component_finite_drift_p50": rigid.component_finite_drift_p50,
        "component_finite_drift_p90": rigid.component_finite_drift_p90,
        "component_finite_drift_max": rigid.component_finite_drift_max,
        "component_retained_mask": trusted.component_retained_mask,
        "component_valid_view_rejected_mask": (
            trusted.component_valid_view_rejected_mask
        ),
        "component_singular_rejected_mask": (
            trusted.component_singular_rejected_mask
        ),
        "component_finite_drift_rejected_mask": (
            trusted.component_finite_drift_rejected_mask
        ),
        "component_singular_ratio": trusted.component_singular_ratio,
        "component_supported_valid_view_count": (
            trusted.component_supported_valid_view_count
        ),
        "component_dominant_valid_view_index": (
            trusted.component_dominant_valid_view_index
        ),
        "component_secondary_view_node_ratio": (
            trusted.component_secondary_view_node_ratio
        ),
    }


def _shared_arrays(rigid: RigidComponentResult) -> dict[str, np.ndarray]:
    """Extract graph-dependent arrays that are identical for every mode."""

    return {name: np.asarray(getattr(rigid, name)) for name in SHARED_ARRAY_FIELDS}


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write one compressed NumPy checkpoint through a sibling temporary file."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        save_named_arrays(temporary, arrays)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_mode_checkpoint(
    path: Path,
    *,
    mode_slot: int,
    solver_run_identity: str,
) -> dict[str, np.ndarray]:
    """Load one complete per-mode checkpoint and reject stale partial results."""

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = {"mode_slot", "solver_run_identity"}
    if set(arrays) != set(MODE_ARRAY_FIELDS) | set(SHARED_ARRAY_FIELDS) | metadata:
        raise ValueError(f"Rigid work checkpoint fields are invalid: {path}")
    if int(arrays["mode_slot"]) != mode_slot:
        raise ValueError(f"Rigid work checkpoint slot is invalid: {path}")
    if str(arrays["solver_run_identity"]) != solver_run_identity:
        raise ValueError(f"Rigid work checkpoint identity differs: {path}")
    for name, value in arrays.items():
        if value.dtype == np.dtype("O"):
            raise ValueError(f"Rigid checkpoint {name} may not use object dtype")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            allowed_inf = name in {
                "alpha_phase_std",
                "alpha_log_gain_std",
                "alpha_condition",
                "component_condition",
            }
            if not allowed_inf or np.isnan(value).any():
                raise ValueError(f"Rigid checkpoint {name} contains invalid values")
    return arrays


def _prepare_work_dir(work_dir: Path, source: Mapping[str, Any]) -> None:
    """Create or validate the immutable run identity for resumable mode files."""

    manifest_path = work_dir / WORK_MANIFEST_FILENAME
    if work_dir.exists() and not work_dir.is_dir():
        raise FileExistsError(f"Rigid work path is not a directory: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("solver_run_identity") != source["solver_run_identity"]:
            raise ValueError("Rigid work directory belongs to different inputs/config")
        return
    if any(work_dir.iterdir()):
        raise FileExistsError(
            f"Rigid work directory is non-empty without a manifest: {work_dir}"
        )
    work_manifest = {
        "format": "modal_gaussians.rigid_modes_work",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **dict(source),
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(work_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)


def _validate_final_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    mode_count: int,
    point_count: int,
    view_count: int,
    component_count: int,
    edge_count: int,
) -> None:
    """Validate field set, primary dimensions, finite values, and mask semantics."""

    if set(arrays) != set(MODE_ARRAY_FIELDS) | set(SHARED_ARRAY_FIELDS):
        raise ValueError("Rigid-mode array fields are invalid")
    expected_shapes = {
        "phi": (mode_count, point_count, 3),
        "trusted_phi": (mode_count, point_count, 3),
        "trusted_seed_mask": (mode_count, point_count),
        "alphas": (mode_count, view_count),
        "alpha_identifiable_mask": (mode_count, view_count),
        "component_translation": (mode_count, component_count, 3),
        "component_rotation": (mode_count, component_count, 3),
        "component_singular_values": (mode_count, component_count, 6),
        "component_valid_view_node_count": (
            mode_count,
            component_count,
            view_count,
        ),
        "edge_finite_drift_max": (mode_count, edge_count),
        "rigid_seed_mask": (point_count,),
        "observed_mask": (point_count,),
        "point_component_index": (point_count,),
        "component_centroid": (component_count, 3),
        "edge_component_index": (edge_count,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Rigid-mode {name} has shape {arrays[name].shape}, expected {shape}")
    for name in MODE_ARRAY_FIELDS:
        if arrays[name].shape[0] != mode_count:
            raise ValueError(f"Rigid-mode field {name} does not start with mode axis")
    for name, value in arrays.items():
        if value.dtype == np.dtype("O"):
            raise ValueError(f"Rigid-mode {name} may not use object dtype")
        if value.dtype.kind in "fc" and np.isnan(value).any():
            raise ValueError(f"Rigid-mode {name} contains NaN")
    if not np.all(arrays["trusted_seed_mask"] <= arrays["rigid_seed_mask"][None]):
        raise ValueError("Trusted rigid seeds are not a subset of rigid candidates")
    if not np.all(arrays["trusted_phi"][~arrays["trusted_seed_mask"]] == 0.0):
        raise ValueError("Untrusted rigid displacement values must remain zero")
    if not np.all(arrays["alphas"][:, 0] == np.complex64(1.0 + 0.0j)):
        raise ValueError("Reference-view alpha must remain exactly one")


def load_rigid_modes(path: str | Path) -> RigidModesArtifact:
    """Load and fully validate a final alpha/rigid solver candidate artifact."""

    root = Path(path).expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    arrays_path = root / RIGID_MODES_FILENAME
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"Incomplete rigid-mode artifact: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != RIGID_MODES_FORMAT:
        raise ValueError("Unsupported rigid-mode artifact format")
    if manifest.get("version") != RIGID_MODES_VERSION:
        raise ValueError("Unsupported rigid-mode artifact version")
    if manifest.get("quality_gate") != {
        "required": True,
        "status": "solver_candidate_unapproved",
    }:
        raise ValueError("Rigid-mode artifact must remain an unapproved candidate")
    if manifest.get("arrays_file") != RIGID_MODES_FILENAME:
        raise ValueError("Rigid-mode array filename is invalid")
    if manifest.get("arrays_file_sha256") != _sha256_file(arrays_path):
        raise ValueError("Rigid-mode NPZ SHA-256 does not match manifest")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Rigid-mode counts are invalid")
    _validate_final_arrays(
        arrays,
        mode_count=int(counts["modes"]),
        point_count=int(counts["foreground_gaussians"]),
        view_count=int(counts["views"]),
        component_count=int(counts["rigid_components"]),
        edge_count=int(counts["accepted_graph_edges"]),
    )
    fields = manifest.get("arrays")
    expected_fields = {
        name: {"dtype": value.dtype.name, "shape": list(value.shape)}
        for name, value in arrays.items()
    }
    if fields != expected_fields:
        raise ValueError("Rigid-mode array metadata does not match NPZ")
    if manifest.get("arrays_identity") != _arrays_identity(arrays):
        raise ValueError("Rigid-mode array identity does not match")
    expected_run = hashlib.sha256(
        _canonical_json(_solver_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("solver_run_identity") != expected_run:
        raise ValueError("Rigid-mode solver run identity does not match")
    expected_artifact = hashlib.sha256(
        _canonical_json(_artifact_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("rigid_modes_identity") != expected_artifact:
        raise ValueError("Rigid-mode artifact identity does not match")
    return RigidModesArtifact(root, manifest, arrays)


def build_rigid_modes_artifact(
    *,
    scene_dir: str | Path,
    topology_dir: str | Path,
    measurements_dir: str | Path,
    graph_dir: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
    alpha_config: AlphaSyncConfig | None = None,
    rigid_config: RigidComponentConfig | None = None,
    seed_config: RigidSeedConfig | None = None,
    command: Sequence[str] = (),
) -> RigidModesArtifact:
    """Synchronize views, solve rigid components, resume per mode, and publish."""

    alpha_settings = alpha_config or AlphaSyncConfig()
    rigid_settings = rigid_config or RigidComponentConfig()
    seed_settings = seed_config or RigidSeedConfig()
    alpha_settings.validate()
    rigid_settings.validate()
    seed_settings.validate()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Rigid-mode output already exists: {destination}")
    work = Path(work_dir).expanduser().resolve()
    source, scene, topology, measurements, graph = _source_manifest(
        scene_dir=scene_dir,
        topology_dir=topology_dir,
        measurements_dir=measurements_dir,
        graph_dir=graph_dir,
        alpha_config=alpha_settings,
        rigid_config=rigid_settings,
        seed_config=seed_settings,
    )
    _prepare_work_dir(work, source)
    points = (
        scene.foreground.active()["means"].detach().cpu().numpy().astype(np.float32)
    )
    view_labels = tuple(view["label"] for view in source["views"])
    mode_results: list[dict[str, np.ndarray]] = []
    progress = Progress("rigid solve", len(source["modes"]), unit="modes")
    for mode in source["modes"]:
        slot = int(mode["mode_slot"])
        checkpoint = work / f"mode_{slot:03d}.npz"
        reused = checkpoint.is_file()
        report_progress(f"rigid: mode_slot={slot} {'loading checkpoint' if reused else 'solving'}")
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
            alpha = solve_alpha_sync(prepared, alpha_settings)
            rigid = solve_rigid_components(
                prepared, alpha, graph.arrays, rigid_settings
            )
            trusted = select_trusted_rigid_seeds(rigid, seed_settings)
            result = {
                **_mode_arrays(alpha, rigid, trusted),
                **_shared_arrays(rigid),
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
        progress.update(len(mode_results), f"mode_slot={slot} reused={reused}", force=True)

    shared = {
        name: np.asarray(mode_results[0][name]) for name in SHARED_ARRAY_FIELDS
    }
    for result in mode_results[1:]:
        for name, expected in shared.items():
            if not np.array_equal(result[name], expected):
                raise RuntimeError(f"Per-mode shared rigid field changed: {name}")
    final_arrays = {
        name: np.stack([result[name] for result in mode_results], axis=0)
        for name in MODE_ARRAY_FIELDS
    }
    final_arrays.update(shared)
    point_count = len(points)
    view_count = len(source["views"])
    component_count = len(shared["component_node_count"])
    edge_count = len(shared["edge_component_index"])
    _validate_final_arrays(
        final_arrays,
        mode_count=len(source["modes"]),
        point_count=point_count,
        view_count=view_count,
        component_count=component_count,
        edge_count=edge_count,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        arrays_path = temporary / RIGID_MODES_FILENAME
        save_named_arrays(arrays_path, final_arrays)
        manifest = {
            "format": RIGID_MODES_FORMAT,
            "version": RIGID_MODES_VERSION,
            "producer": {
                "project_version": __version__,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": list(command),
            },
            **source,
            "work_dir": str(work),
            "semantics": {
                "alpha": "measurement_view = alpha_view * reference_gauge_projection",
                "rigid_model": "one_complex_first_order_se3_twist_per_graph_component",
                "playback": "real(phi * exp(i*phase)); additive, finite drift measured",
                "raw_phi_outside_rigid_seed_mask": "zero_unsolved",
                "trusted_phi_outside_trusted_seed_mask": "zero_untrusted",
                "motion_fill": "not_applied",
            },
            "quality_gate": {
                "required": True,
                "status": "solver_candidate_unapproved",
            },
            "counts": {
                "modes": len(source["modes"]),
                "views": view_count,
                "foreground_gaussians": point_count,
                "observed_gaussians": int(np.count_nonzero(shared["observed_mask"])),
                "rigid_seed_gaussians": int(
                    np.count_nonzero(shared["rigid_seed_mask"])
                ),
                "rigid_components": component_count,
                "accepted_graph_edges": edge_count,
                "trusted_seed_gaussians_per_mode": np.count_nonzero(
                    final_arrays["trusted_seed_mask"], axis=1
                ).astype(int).tolist(),
                "trusted_components_per_mode": np.count_nonzero(
                    final_arrays["component_retained_mask"], axis=1
                ).astype(int).tolist(),
                "identifiable_views_per_mode": np.count_nonzero(
                    final_arrays["alpha_identifiable_mask"], axis=1
                ).astype(int).tolist(),
            },
            "arrays_file": RIGID_MODES_FILENAME,
            "arrays": {
                name: {"dtype": value.dtype.name, "shape": list(value.shape)}
                for name, value in final_arrays.items()
            },
            "arrays_identity": _arrays_identity(final_arrays),
            "arrays_file_sha256": _sha256_file(arrays_path),
        }
        manifest["rigid_modes_identity"] = hashlib.sha256(
            _canonical_json(_artifact_identity_payload(manifest))
        ).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_rigid_modes(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Rigid-mode output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_rigid_modes(destination)


__all__ = [
    "RigidComponentConfig",
    "RigidComponentResult",
    "RigidModesArtifact",
    "RigidSeedConfig",
    "RigidSeedResult",
    "build_rigid_modes_artifact",
    "load_rigid_modes",
    "select_trusted_rigid_seeds",
    "solve_rigid_components",
]
