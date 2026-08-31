"""Bounded-complex asynchronous phase/gain synchronization across views."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from typing import Any

import numpy as np

from modal_gaussians.topology import TopologyArrays


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

    @property
    def num_views(self) -> int:
        """Return the number of fixed views in canonical order."""

        return len(self.view_labels)


@dataclass(frozen=True)
class AlphaConstraint:
    """Store one shared Gaussian's left-null alpha constraint."""

    rows: np.ndarray
    views: np.ndarray
    matrix: np.ndarray
    observation_energy: float


@dataclass(frozen=True)
class ProfiledObservationBatch:
    """Store vectorized per-Gaussian blocks for profiled alpha residuals."""

    row_block_index: np.ndarray
    row_local_view_index: np.ndarray
    weighted_jacobian: np.ndarray
    weighted_observation: np.ndarray
    gram_by_block_view: np.ndarray
    rhs_by_block_view: np.ndarray
    normalization: np.ndarray
    equation_count: np.ndarray


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
) -> PreparedObservations:
    """Expand topology samples to contributor rows using the old weight convention."""

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


def _build_constraints(
    prepared: PreparedObservations,
    allowed_views: np.ndarray | None = None,
) -> list[AlphaConstraint]:
    """Eliminate 3D displacement with each shared Gaussian's left nullspace."""

    constraints: list[AlphaConstraint] = []
    if allowed_views is None:
        allowed_views = np.ones(prepared.num_views, dtype=bool)
    for point_rows in prepared.rows_by_point:
        if len(point_rows) == 0:
            continue
        keep = (
            allowed_views[prepared.obs_view_index[point_rows]]
            & (prepared.obs_weights[point_rows] > 0.0)
        )
        rows = point_rows[keep]
        views = np.unique(prepared.obs_view_index[rows])
        if len(views) < 2:
            continue
        sqrt_weight = np.sqrt(prepared.obs_weights[rows])
        matrix_a = (
            sqrt_weight[:, None, None]
            * prepared.obs_jacobian[rows].astype(np.float64)
        ).reshape(-1, 3)
        matrix_b = np.zeros(
            (len(rows) * 2, prepared.num_views), dtype=np.complex128
        )
        for local_row, observation_row in enumerate(rows.tolist()):
            view = int(prepared.obs_view_index[observation_row])
            matrix_b[2 * local_row : 2 * local_row + 2, view] = (
                sqrt_weight[local_row]
                * prepared.obs_y[observation_row].astype(np.complex128)
            )
        left_vectors, singular, _ = np.linalg.svd(matrix_a, full_matrices=True)
        if len(singular) == 0 or singular[0] <= EPSILON:
            continue
        rank = int(np.count_nonzero(singular > 1.0e-8 * singular[0]))
        matrix_c = left_vectors[:, rank:].conj().T @ matrix_b
        if float(np.linalg.norm(matrix_c)) <= EPSILON:
            continue
        constraints.append(
            AlphaConstraint(
                rows=rows,
                views=views,
                matrix=matrix_c,
                observation_energy=float(np.linalg.norm(matrix_b)),
            )
        )
    return constraints


def _edge_statistics(
    constraints: list[AlphaConstraint], view_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Measure informative shared-point connectivity between view pairs."""

    counts = np.zeros((view_count, view_count), dtype=np.int32)
    information = np.zeros((view_count, view_count), dtype=np.float64)
    for constraint in constraints:
        energy = max(constraint.observation_energy, EPSILON)
        hessian = (
            constraint.matrix.conj().T @ constraint.matrix
        ) / (energy * energy)
        scale = max(float(np.real(np.trace(hessian))), EPSILON)
        for local, view_a in enumerate(constraint.views.tolist()):
            for view_b in constraint.views[local + 1 :].tolist():
                value = float(np.abs(hessian[int(view_a), int(view_b)]))
                if value > 1.0e-12 * scale:
                    counts[int(view_a), int(view_b)] += 1
                    counts[int(view_b), int(view_a)] += 1
                    information[int(view_a), int(view_b)] += value
                    information[int(view_b), int(view_a)] += value
    return counts, information


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


def _information_blocks(
    constraints: list[AlphaConstraint], candidate_views: np.ndarray
) -> np.ndarray:
    """Build normalized complex alpha information per shared Gaussian."""

    blocks = []
    for constraint in constraints:
        matrix = constraint.matrix[:, candidate_views] / max(
            constraint.observation_energy, EPSILON
        )
        blocks.append(matrix.conj().T @ matrix)
    if not blocks:
        return np.zeros(
            (0, len(candidate_views), len(candidate_views)), dtype=np.complex128
        )
    return np.stack(blocks).astype(np.complex128)


def _complex_initialization(
    information: np.ndarray, reference_local: int
) -> np.ndarray:
    """Initialize inverse alpha beta through gauge-fixed complex least squares."""

    beta = np.ones(len(information), dtype=np.complex128)
    unknown = np.asarray(
        [index for index in range(len(beta)) if index != reference_local],
        dtype=np.int64,
    )
    if len(unknown):
        beta[unknown] = np.linalg.lstsq(
            information[np.ix_(unknown, unknown)],
            -information[unknown, reference_local],
            rcond=None,
        )[0]
    beta[reference_local] = 1.0 + 0.0j
    return beta


def _parameterized_beta(
    parameters: np.ndarray, view_count: int, reference_local: int
) -> np.ndarray:
    """Map phase/log-gain parameters to inverse complex alpha values."""

    beta = np.ones(view_count, dtype=np.complex128)
    unknown = np.asarray(
        [index for index in range(view_count) if index != reference_local],
        dtype=np.int64,
    )
    count = len(unknown)
    theta = parameters[:count]
    log_gain = parameters[count:]
    beta[unknown] = np.exp(-log_gain - 1j * theta)
    return beta


def _build_profiled_batch(
    prepared: PreparedObservations,
    constraints: list[AlphaConstraint],
    candidate_views: np.ndarray,
) -> ProfiledObservationBatch:
    """Pack all candidate shared-point observations for vectorized profiling."""

    row_count = np.asarray([len(value.rows) for value in constraints], dtype=np.int64)
    rows = np.concatenate([value.rows for value in constraints])
    row_block = np.repeat(np.arange(len(constraints), dtype=np.int64), row_count)
    global_to_local = np.full(prepared.num_views, -1, dtype=np.int64)
    global_to_local[candidate_views] = np.arange(len(candidate_views), dtype=np.int64)
    row_view = global_to_local[prepared.obs_view_index[rows]]
    if np.any(row_view < 0):
        raise RuntimeError("Alpha constraint contains a view outside its candidate")
    sqrt_weight = np.sqrt(prepared.obs_weights[rows]).astype(np.float64)
    weighted_jacobian = (
        sqrt_weight[:, None, None]
        * prepared.obs_jacobian[rows].astype(np.float64)
    )
    weighted_observation = (
        sqrt_weight[:, None] * prepared.obs_y[rows].astype(np.complex128)
    )
    gram = np.zeros(
        (len(constraints), len(candidate_views), 3, 3), dtype=np.complex128
    )
    rhs = np.zeros(
        (len(constraints), len(candidate_views), 3), dtype=np.complex128
    )
    np.add.at(
        gram,
        (row_block, row_view),
        np.einsum(
            "rki,rkj->rij", np.conj(weighted_jacobian), weighted_jacobian
        ),
    )
    np.add.at(
        rhs,
        (row_block, row_view),
        np.einsum(
            "rki,rk->ri", np.conj(weighted_jacobian), weighted_observation
        ),
    )
    energy = np.bincount(
        row_block,
        weights=np.sum(np.abs(weighted_observation) ** 2, axis=1),
        minlength=len(constraints),
    )
    return ProfiledObservationBatch(
        row_block_index=row_block,
        row_local_view_index=row_view,
        weighted_jacobian=weighted_jacobian,
        weighted_observation=weighted_observation,
        gram_by_block_view=gram,
        rhs_by_block_view=rhs,
        normalization=np.maximum(np.sqrt(energy), EPSILON),
        equation_count=(2 * row_count).astype(np.int64),
    )


def _profiled_residuals(
    batch: ProfiledObservationBatch,
    parameters: np.ndarray,
    reference_local: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Profile out per-Gaussian 3D displacement and return normalized residuals."""

    beta = _parameterized_beta(
        parameters, batch.gram_by_block_view.shape[1], reference_local
    )
    alpha = 1.0 / beta
    gram = np.einsum(
        "v,bvij->bij", np.abs(alpha) ** 2, batch.gram_by_block_view
    )
    rhs = np.einsum("v,bvi->bi", np.conj(alpha), batch.rhs_by_block_view)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = np.maximum(eigenvalues[:, -1], 0.0)
    singular_rcond = np.finfo(np.float64).eps * np.maximum(
        batch.equation_count, 3
    )
    cutoff = singular_rcond**2 * largest
    inverse = np.zeros_like(eigenvalues)
    retained = eigenvalues > cutoff[:, None]
    np.divide(1.0, eigenvalues, out=inverse, where=retained)
    projected_rhs = np.einsum(
        "bji,bj->bi", np.conj(eigenvectors), rhs
    )
    point_phi = np.einsum(
        "bij,bj->bi", eigenvectors, inverse * projected_rhs
    )
    prediction = alpha[batch.row_local_view_index, None] * np.einsum(
        "rki,ri->rk",
        batch.weighted_jacobian,
        point_phi[batch.row_block_index],
    )
    residual = (
        prediction - batch.weighted_observation
    ) / batch.normalization[batch.row_block_index, None]
    squared = np.sum(np.abs(residual) ** 2, axis=1)
    block_squared = np.bincount(
        batch.row_block_index,
        weights=squared,
        minlength=len(batch.normalization),
    )
    return residual, np.sqrt(np.maximum(block_squared, 0.0)).astype(np.float64)


def _block_huber_scale(residual: np.ndarray, scale: float) -> np.ndarray:
    """Return the exact residual scaling that reproduces block Huber loss."""

    result = np.ones_like(residual, dtype=np.float64)
    large = residual > scale
    ratio = scale / np.maximum(residual[large], EPSILON)
    result[large] = np.sqrt(np.maximum(2.0 * ratio - ratio**2, 0.0))
    return result


def _flatten_scaled_residual(
    batch: ProfiledObservationBatch,
    residual: np.ndarray,
    block_scale: np.ndarray,
) -> np.ndarray:
    """Pack scaled complex residuals as real least-squares equations."""

    scaled = block_scale[batch.row_block_index, None] * residual
    return np.concatenate(
        [np.real(scaled).reshape(-1), np.imag(scaled).reshape(-1)]
    ).astype(np.float64)


def _information_summary(
    parameter_information: np.ndarray, constraint_count: int
) -> tuple[np.ndarray, float, float, float]:
    """Summarize gauge-reduced alpha information and conditioning."""

    eigenvalues = (
        np.linalg.eigvalsh(
            0.5 * (parameter_information + parameter_information.T)
        )
        if parameter_information.size
        else np.empty((0,), dtype=np.float64)
    )
    singular = np.sqrt(np.maximum(eigenvalues, 0.0))[::-1]
    if len(singular) == 0:
        return singular, 1.0, 0.0, 1.0
    rank_ratio = float(singular[-1] / max(singular[0], EPSILON))
    information_ratio = float(
        singular[-1] / np.sqrt(max(constraint_count, 1))
    )
    condition = float(singular[0] / max(singular[-1], EPSILON))
    return singular, rank_ratio, information_ratio, condition


def _require_least_squares() -> Any:
    """Load SciPy least_squares without depending on incomplete type stubs."""

    try:
        optimize = importlib.import_module("scipy.optimize")
    except ImportError as error:
        raise RuntimeError("Alpha synchronization requires scipy") from error
    solver = getattr(optimize, "least_squares", None)
    if solver is None:
        raise RuntimeError("scipy.optimize.least_squares is unavailable")
    return solver


def _refine_candidate(
    prepared: PreparedObservations,
    constraints: list[AlphaConstraint],
    candidate_views: np.ndarray,
    config: AlphaSyncConfig,
) -> AlphaCandidateSolve:
    """Optimize bounded phase/gain after profiling per-point 3D displacement."""

    reference_local = int(np.where(candidate_views == 0)[0][0])
    information_blocks = _information_blocks(constraints, candidate_views)
    beta_initial = _complex_initialization(
        np.sum(information_blocks, axis=0), reference_local
    )
    unknown = np.asarray(
        [index for index in range(len(candidate_views)) if index != reference_local],
        dtype=np.int64,
    )
    theta_initial = -np.angle(beta_initial[unknown])
    log_gain_initial = np.clip(
        -np.log(np.maximum(np.abs(beta_initial[unknown]), EPSILON)),
        np.log(config.gain_minimum),
        np.log(config.gain_maximum),
    )
    initial = np.concatenate([theta_initial, log_gain_initial]).astype(np.float64)
    lower = np.concatenate(
        [
            np.full_like(theta_initial, -np.inf),
            np.full_like(log_gain_initial, np.log(config.gain_minimum)),
        ]
    )
    upper = np.concatenate(
        [
            np.full_like(theta_initial, np.inf),
            np.full_like(log_gain_initial, np.log(config.gain_maximum)),
        ]
    )
    batch = _build_profiled_batch(prepared, constraints, candidate_views)
    _, initial_norm = _profiled_residuals(batch, initial, reference_local)
    huber_scale = max(
        float(np.median(initial_norm)) if len(initial_norm) else 1.0,
        1.0e-6,
    )

    def residual_function(parameters: np.ndarray) -> np.ndarray:
        """Return the exact block-Huber residual packed for least_squares."""

        residual, block_residual = _profiled_residuals(
            batch, parameters, reference_local
        )
        return _flatten_scaled_residual(
            batch,
            residual,
            _block_huber_scale(block_residual, huber_scale),
        )

    result = _require_least_squares()(
        residual_function,
        initial,
        bounds=(lower, upper),
        loss="linear",
        max_nfev=500,
    )
    beta = _parameterized_beta(result.x, len(candidate_views), reference_local)
    alpha = 1.0 / beta
    alpha[reference_local] = 1.0 + 0.0j
    _, final_block_residual = _profiled_residuals(
        batch, result.x, reference_local
    )
    robust_weights = np.ones_like(final_block_residual)
    large = final_block_residual > huber_scale
    robust_weights[large] = huber_scale / np.maximum(
        final_block_residual[large], EPSILON
    )
    constraint_information = np.einsum(
        "b,bij->ij", robust_weights, information_blocks
    ).astype(np.complex128)
    jacobian = np.asarray(result.jac, dtype=np.float64)
    parameter_information = jacobian.T @ jacobian
    singular, rank_ratio, information_ratio, condition = _information_summary(
        parameter_information, len(constraints)
    )
    consistency = (
        float(np.sqrt(np.mean(final_block_residual**2)))
        if len(final_block_residual)
        else 0.0
    )
    gain_bound_active = np.zeros(len(candidate_views), dtype=bool)
    gain_bound_active[unknown] = np.asarray(result.active_mask, dtype=np.int8)[
        len(unknown) :
    ] != 0
    phase_std = np.full(len(candidate_views), np.inf, dtype=np.float64)
    log_gain_std = np.full(len(candidate_views), np.inf, dtype=np.float64)
    phase_std[reference_local] = 0.0
    log_gain_std[reference_local] = 0.0
    if parameter_information.size and len(singular) and singular[-1] > EPSILON:
        covariance = np.linalg.pinv(parameter_information)
        degrees = max(jacobian.shape[0] - parameter_information.shape[0], 1.0)
        variance = float(np.sum(np.asarray(result.fun) ** 2)) / degrees
        parameter_std = np.sqrt(
            np.maximum(np.diag(covariance) * variance, 0.0)
        )
        phase_std[unknown] = parameter_std[: len(unknown)]
        log_gain_std[unknown] = parameter_std[len(unknown) :]
        log_gain_std[gain_bound_active] = np.inf
    return AlphaCandidateSolve(
        alphas=alpha,
        consistency_residual=consistency,
        singular_values=singular,
        rank_ratio=rank_ratio,
        information_ratio=information_ratio,
        condition=condition,
        phase_std=phase_std,
        log_gain_std=log_gain_std,
        constraint_information=constraint_information,
        parameter_information=parameter_information,
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        gain_bound_active_mask=gain_bound_active,
        information_kind="profiled_exact_block_huber_gauss_newton",
    )


def solve_alpha_sync(
    prepared: PreparedObservations,
    config: AlphaSyncConfig | None = None,
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

    graph_constraints = _build_constraints(prepared)
    edge_counts, edge_information = _edge_statistics(graph_constraints, view_count)
    raw_shared_counts = _raw_shared_counts(prepared)
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

    final_constraints: list[AlphaConstraint] = []
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
            constraints = _build_constraints(prepared, allowed)
        if not constraints:
            removed = int(candidate_views[-1])
            reasons[removed] = "insufficient_information"
            candidate_views = candidate_views[candidate_views != removed]
            continue

        solved = _refine_candidate(prepared, constraints, candidate_views, settings)
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
    for constraint in final_constraints:
        constraint_count[
            np.linalg.norm(constraint.matrix, axis=0) > EPSILON
        ] += 1
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
