"""Bounded-complex asynchronous phase/gain synchronization across views."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from modal_gaussians.motion.observations.topology import TopologyArrays


EPSILON = 1.0e-12


@dataclass(frozen=True)
class AlphaSyncConfig:
    """Hold the accepted bounded-complex alpha solver settings."""

    gain_minimum: float = 0.25
    gain_maximum: float = 4.0
    minimum_shared_points: int = 16
    rank_ratio_minimum: float = 1.0e-4
    information_ratio_minimum: float = 1.0e-4
    failure: str = "exclude"

    def validate(self) -> None:
        """Reject alpha settings outside the accepted numerical domain."""

        if not (
            math.isfinite(self.gain_minimum)
            and math.isfinite(self.gain_maximum)
            and 0.0 < self.gain_minimum <= 1.0 <= self.gain_maximum
        ):
            raise ValueError("Alpha gain bounds must satisfy 0 < minimum <= 1 <= maximum")
        if (
            isinstance(self.minimum_shared_points, bool)
            or not isinstance(self.minimum_shared_points, int)
            or self.minimum_shared_points <= 0
        ):
            raise ValueError("Alpha minimum_shared_points must be positive")
        for name, value in (
            ("rank_ratio_minimum", self.rank_ratio_minimum),
            ("information_ratio_minimum", self.information_ratio_minimum),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"Alpha {name} must lie in (0,1]")
        if self.failure not in {"exclude", "error"}:
            raise ValueError("Alpha failure policy must be 'exclude' or 'error'")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fixed bounded-complex synchronization contract."""

        return {
            "model": "bounded_complex_phase_then_log_gain",
            "reference_view_index": 0,
            "gain_minimum": self.gain_minimum,
            "gain_maximum": self.gain_maximum,
            "minimum_shared_points": self.minimum_shared_points,
            "rank_ratio_minimum": self.rank_ratio_minimum,
            "information_ratio_minimum": self.information_ratio_minimum,
            "failure": self.failure,
            "loss": "profiled_exact_block_huber",
            "maximum_function_evaluations": 500,
        }


@dataclass(frozen=True)
class PreparedObservations:
    """Store one mode's contributor-row projection observations."""

    points: np.ndarray
    obs_point_index: np.ndarray
    obs_view_index: np.ndarray
    obs_y: np.ndarray
    obs_jacobian: np.ndarray
    obs_weights: np.ndarray
    rows_by_point: tuple[np.ndarray, ...]
    view_labels: tuple[str, ...]
    geometry_key: str | None = None

    @property
    def num_views(self) -> int:
        """Return the number of fixed views in canonical order."""

        return len(self.view_labels)


@dataclass(frozen=True)
class AlphaCandidateSolve:
    """Store one candidate view subset's bounded-complex optimization result."""

    alphas: np.ndarray
    consistency_residual: float
    singular_values: np.ndarray
    rank_ratio: float
    information_ratio: float
    condition: float
    phase_std: np.ndarray
    log_gain_std: np.ndarray
    constraint_information: np.ndarray
    parameter_information: np.ndarray
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    gain_bound_active_mask: np.ndarray
    information_kind: str


@dataclass(frozen=True)
class AlphaSyncResult:
    """Store final per-view alpha values and identifiability diagnostics."""

    alphas: np.ndarray
    identifiable_mask: np.ndarray
    reference_connected_mask: np.ndarray
    exclusion_reason: np.ndarray
    shared_point_count: np.ndarray
    edge_point_count: np.ndarray
    edge_information: np.ndarray
    constraint_count_per_view: np.ndarray
    information_matrix: np.ndarray
    singular_values: np.ndarray
    rank_ratio: float
    information_ratio: float
    condition: float
    consistency_residual: float
    phase_std: np.ndarray
    log_gain_std: np.ndarray
    parameter_information: np.ndarray
    parameter_view_indices: np.ndarray
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    gain_bound_active_mask: np.ndarray
    information_kind: str


def _rows_by_point(point_count: int, point_index: np.ndarray) -> tuple[np.ndarray, ...]:
    """Group observation row indices by global foreground Gaussian index."""

    rows: list[list[int]] = [[] for _ in range(point_count)]
    for row, point in enumerate(point_index.tolist()):
        rows[int(point)].append(row)
    return tuple(np.asarray(value, dtype=np.int64) for value in rows)


def prepare_observations(
    *,
    points: np.ndarray,
    topology: TopologyArrays,
    sample_measurements: np.ndarray,
    view_labels: tuple[str, ...],
    workspace=None, topology_identity: str | None = None, cache_dir=None,
) -> PreparedObservations:
    """Expand topology samples to contributor rows using pair-normalized contribution weights."""

    point_values = np.asarray(points, dtype=np.float32)
    measurements = np.asarray(sample_measurements, dtype=np.complex64)
    view_count = len(view_labels)
    if point_values.ndim != 2 or point_values.shape[1] != 3:
        raise ValueError("Alpha points must have shape [G,3]")
    if not np.isfinite(point_values).all():
        raise ValueError("Alpha points contain NaN or Inf")
    sample_count = len(topology.sample_view_index)
    if measurements.shape != (sample_count, 2):
        raise ValueError("One mode's measurements must have shape [P,2]")
    if not np.isfinite(measurements).all():
        raise ValueError("Complex measurements contain NaN or Inf")
    if view_count == 0 or len(set(view_labels)) != view_count:
        raise ValueError("Alpha view labels must be non-empty and unique")
    if workspace is not None:
        return workspace.prepare(point_values, topology, measurements, view_labels,
                                 topology_identity=topology_identity, cache_dir=cache_dir)
    contributor_count = np.diff(topology.sample_offsets)
    sample_index = np.repeat(
        np.arange(sample_count, dtype=np.int64), contributor_count
    )
    point_index = np.asarray(topology.contributor_gaussian_index, dtype=np.int64)
    if len(sample_index) != len(point_index):
        raise ValueError("Topology contributor offsets are inconsistent")
    view_index = topology.sample_view_index[sample_index].astype(np.int64)
    if np.any(point_index < 0) or np.any(point_index >= len(point_values)):
        raise ValueError("Observation point index is outside the foreground domain")
    if np.any(view_index < 0) or np.any(view_index >= view_count):
        raise ValueError("Observation view index is outside the fixed-view domain")
    contribution = np.asarray(topology.contributor_weight, dtype=np.float64)
    if not np.isfinite(contribution).all() or np.any(contribution <= 0.0):
        raise ValueError("Observation contribution weights must be finite and positive")
    pair_key = point_index * view_count + view_index
    _, inverse, multiplicity = np.unique(
        pair_key, return_inverse=True, return_counts=True
    )
    weights = contribution / multiplicity[inverse].astype(np.float64)
    jacobian = np.asarray(topology.contributor_jacobian, dtype=np.float32)
    if jacobian.shape != (len(point_index), 2, 3) or not np.isfinite(jacobian).all():
        raise ValueError("Observation Jacobians must be finite [M,2,3]")
    return PreparedObservations(
        points=point_values,
        obs_point_index=point_index,
        obs_view_index=view_index,
        obs_y=measurements[sample_index],
        obs_jacobian=jacobian,
        obs_weights=weights,
        rows_by_point=_rows_by_point(len(point_values), point_index),
        view_labels=view_labels,
    )


def _raw_shared_counts(prepared: PreparedObservations) -> np.ndarray:
    """Count positive shared Gaussians before alpha information filtering."""

    counts = np.zeros((prepared.num_views, prepared.num_views), dtype=np.int32)
    for point_rows in prepared.rows_by_point:
        positive = point_rows[prepared.obs_weights[point_rows] > 0.0]
        views = np.unique(prepared.obs_view_index[positive])
        for local, view_a in enumerate(views.tolist()):
            for view_b in views[local + 1 :].tolist():
                counts[int(view_a), int(view_b)] += 1
                counts[int(view_b), int(view_a)] += 1
    return counts


def _reference_connected(
    adjacency: np.ndarray, allowed_indices: np.ndarray
) -> np.ndarray:
    """Return the allowed views connected to reference view zero."""

    allowed = np.zeros(len(adjacency), dtype=bool)
    allowed[allowed_indices] = True
    connected = np.zeros_like(allowed)
    if not allowed[0]:
        return np.empty((0,), dtype=np.int64)
    queue = [0]
    connected[0] = True
    for view in queue:
        for neighbor in np.where(adjacency[view] & allowed)[0].tolist():
            if not connected[neighbor]:
                connected[neighbor] = True
                queue.append(int(neighbor))
    return np.flatnonzero(connected).astype(np.int64)


def backend_identity(backend):
    if backend != "cupy":
        raise ValueError(f"Unknown alpha backend: {backend}")
    # Hash source without importing CuPy: artifact metadata remains readable without a GPU workspace.
    from pathlib import Path
    from modal_gaussians.common.cache import identity, sha256
    root = Path(__file__).parent
    return {"backend": "cupy", "algorithm": "scipy_1.17.1_trf_exact_2point_qr_v2",
            "revision": identity({name: sha256(root / name) for name in
                                  ("alpha.py", "alpha_gpu.py", "_alpha_trf.py")})}


def solve_alpha_sync(prepared, config=None, *, backend="cupy", workspace=None):
    backend_identity(backend)
    from modal_gaussians.motion.observations.alpha_gpu import Workspace
    owned = workspace is None
    workspace = Workspace() if owned else workspace
    try:
        return _solve_alpha_sync(prepared, config, _engine=workspace)
    finally:
        if owned:
            workspace.close()


def _solve_alpha_sync(
    prepared: PreparedObservations,
    config: AlphaSyncConfig | None = None,
    *, _engine,
) -> AlphaSyncResult:
    """Estimate identifiable per-view complex alpha values for one mode."""

    settings = config or AlphaSyncConfig()
    settings.validate()
    view_count = prepared.num_views
    alphas = np.ones(view_count, dtype=np.complex128)
    identifiable = np.zeros(view_count, dtype=bool)
    reasons = np.full(view_count, "insufficient_information", dtype="<U32")
    phase_std = np.full(view_count, np.inf, dtype=np.float64)
    log_gain_std = np.full(view_count, np.inf, dtype=np.float64)
    gain_bound_active = np.zeros(view_count, dtype=bool)
    reference_usable = bool(
        np.any((prepared.obs_view_index == 0) & (prepared.obs_weights > 0.0))
    )
    if reference_usable:
        identifiable[0] = True
        reasons[0] = "reference"
        phase_std[0] = 0.0
        log_gain_std[0] = 0.0
    else:
        reasons[0] = "empty_reference"

    build = _engine.constraints
    refine = _engine.refine
    graph_constraints = build(prepared)
    edge_counts, edge_information = _engine.edge_statistics(graph_constraints)
    raw_shared_counts = _engine.raw_counts(prepared)
    adjacency = edge_counts >= settings.minimum_shared_points
    np.fill_diagonal(adjacency, False)
    connected_views = _reference_connected(
        adjacency, np.arange(view_count, dtype=np.int64)
    )
    reference_connected_mask = np.zeros(view_count, dtype=bool)
    reference_connected_mask[connected_views] = True
    candidate_views = (
        connected_views.copy()
        if reference_usable
        else np.asarray([0], dtype=np.int64)
    )

    final_constraints = []
    final_information = np.zeros((view_count, view_count), dtype=np.complex128)
    final_singular = np.empty((0,), dtype=np.float64)
    final_rank_ratio = 1.0
    final_information_ratio = 0.0
    final_condition = 1.0
    final_consistency = 0.0
    final_parameter_information = np.zeros((0, 0), dtype=np.float64)
    final_parameter_views = np.empty((0,), dtype=np.int32)
    final_success = reference_usable
    final_status = 0 if reference_usable else -1
    final_message = (
        "reference-only; no relative alpha parameters"
        if reference_usable
        else "reference view has no positive-weight observations"
    )
    final_information_kind = "not_computed"

    while len(candidate_views) > 1:
        connected_candidate = _reference_connected(adjacency, candidate_views)
        unconnected = np.setdiff1d(
            candidate_views, connected_candidate, assume_unique=True
        )
        if len(unconnected):
            reasons[unconnected] = "insufficient_information"
            candidate_views = connected_candidate
            if len(candidate_views) <= 1:
                break
        if len(candidate_views) == view_count:
            constraints = graph_constraints
        else:
            allowed = np.zeros(view_count, dtype=bool)
            allowed[candidate_views] = True
            constraints = build(prepared, allowed)
        if not constraints:
            removed = int(candidate_views[-1])
            reasons[removed] = "insufficient_information"
            candidate_views = candidate_views[candidate_views != removed]
            continue

        solved = refine(prepared, constraints, candidate_views, settings)
        parameter_count = 2 * (len(candidate_views) - 1)
        full_rank = len(solved.singular_values) == parameter_count and (
            parameter_count == 0 or solved.singular_values[-1] > EPSILON
        )
        numerically_identifiable = (
            solved.optimizer_success
            and not bool(np.any(solved.gain_bound_active_mask))
            and full_rank
            and solved.rank_ratio >= settings.rank_ratio_minimum
            and solved.information_ratio >= settings.information_ratio_minimum
        )
        final_constraints = constraints
        final_information.fill(0.0)
        final_information[np.ix_(candidate_views, candidate_views)] = (
            solved.constraint_information
        )
        final_singular = solved.singular_values
        final_rank_ratio = solved.rank_ratio
        final_information_ratio = solved.information_ratio
        final_condition = solved.condition
        final_consistency = solved.consistency_residual
        final_parameter_information = solved.parameter_information
        nonreference = candidate_views[candidate_views != 0].astype(np.int32)
        final_parameter_views = np.concatenate([nonreference, nonreference])
        final_success = solved.optimizer_success
        final_status = solved.optimizer_status
        final_message = solved.optimizer_message
        final_information_kind = solved.information_kind
        gain_bound_active[candidate_views] = solved.gain_bound_active_mask
        if numerically_identifiable:
            alphas[candidate_views] = solved.alphas
            identifiable[candidate_views] = True
            reasons[candidate_views] = "estimated"
            reasons[0] = "reference"
            phase_std[candidate_views] = solved.phase_std
            log_gain_std[candidate_views] = solved.log_gain_std
            break

        active_nonreference = candidate_views[
            (candidate_views != 0) & solved.gain_bound_active_mask
        ]
        if len(active_nonreference):
            weakest = int(active_nonreference[0])
            reason = "gain_bound_limited"
        elif len(candidate_views) == 2:
            weakest = int(candidate_views[candidate_views != 0][0])
            reason = (
                "optimizer_failure"
                if not solved.optimizer_success
                else "insufficient_information"
            )
        else:
            diagonal = np.abs(np.diag(solved.parameter_information))
            count = len(candidate_views) - 1
            strength = np.minimum(diagonal[:count], diagonal[count:])
            nonreference = candidate_views[candidate_views != 0]
            weakest = int(nonreference[int(np.argmin(strength))])
            reason = (
                "optimizer_failure"
                if not solved.optimizer_success
                else "insufficient_information"
            )
        reasons[weakest] = reason
        candidate_views = candidate_views[candidate_views != weakest]

    observed_views = np.zeros(view_count, dtype=bool)
    observed_views[np.unique(prepared.obs_view_index[prepared.obs_weights > 0.0])] = True
    invalid = np.flatnonzero(observed_views & ~identifiable)
    if settings.failure == "error" and len(invalid):
        labels = [prepared.view_labels[index] for index in invalid.tolist()]
        raise ValueError(f"Unidentifiable alpha for observed views: {labels}")
    constraint_count = np.zeros(view_count, dtype=np.int32)
    if len(final_constraints):
        constraint_count = _engine.constraint_counts(final_constraints)
    return AlphaSyncResult(
        alphas=alphas.astype(np.complex64),
        identifiable_mask=identifiable,
        reference_connected_mask=reference_connected_mask,
        exclusion_reason=reasons,
        shared_point_count=raw_shared_counts,
        edge_point_count=edge_counts,
        edge_information=edge_information.astype(np.float32),
        constraint_count_per_view=constraint_count,
        information_matrix=final_information.astype(np.complex64),
        singular_values=final_singular.astype(np.float32),
        rank_ratio=final_rank_ratio,
        information_ratio=final_information_ratio,
        condition=final_condition,
        consistency_residual=final_consistency,
        phase_std=phase_std.astype(np.float32),
        log_gain_std=log_gain_std.astype(np.float32),
        parameter_information=final_parameter_information.astype(np.float32),
        parameter_view_indices=final_parameter_views,
        optimizer_success=final_success,
        optimizer_status=final_status,
        optimizer_message=final_message,
        gain_bound_active_mask=gain_bound_active,
        information_kind=final_information_kind,
    )


__all__ = [
    "AlphaSyncConfig",
    "AlphaSyncResult",
    "PreparedObservations",
    "prepare_observations",
    "solve_alpha_sync",
]
