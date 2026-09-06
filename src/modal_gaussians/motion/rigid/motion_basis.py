"""Shared motion-basis blending for a complete foreground modal field.

The rigid solver supplies complex infinitesimal SE(3) fields. By default only
components trusted at every mode are eligible; an explicit experiment policy
can include all solved nontrivial components without changing their trust data.
This module treats those fields as fixed bases and solves one
real, non-negative, frequency-shared simplex weight vector per foreground
Gaussian.  Unlike the earlier rigid completion stage, the data term composes
all contributor Gaussians at each measured pixel before computing a residual.
"""

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
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from modal_gaussians import __version__
from modal_gaussians.measurements import load_gaussian_measurements
from modal_gaussians.numpy_io import save_named_arrays
from modal_gaussians.progress import Progress, report_progress
from modal_gaussians.motion.rigid.rigid import load_rigid_modes
from modal_gaussians.static import load_static_scene
from modal_gaussians.motion.rigid.structure_graph import load_observed_structure_graph
from modal_gaussians.topology import load_observation_topology


COMPLETED_MODES_FORMAT = "modal_gaussians.completed_modes"
COMPLETED_MODES_VERSION = 3
COMPLETED_MODES_FILENAME = "completed_modes.npz"
WORK_MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_FILENAME = "weights_checkpoint.npz"
DESIGN_FILENAME = "pixel_basis_design.npy"
EPSILON = 1.0e-12


@dataclass(frozen=True)
class MotionBasisConfig:
    """Configure the accepted first shared motion-basis experiment."""

    rigid_basis_count: int | None = 6
    local_rigid_basis_count: int = 4
    graph_neighbors: int = 8
    graph_max_distance: float = 0.008
    distance_temperature: float = 0.02
    zero_prior_score: float = 0.05
    smooth_weight: float = 0.01
    prior_weight: float = 0.001
    energy_floor_fraction: float = 0.05
    max_iterations: int = 1000
    relative_tolerance: float = 1.0e-8
    projected_gradient_tolerance: float = 1.0e-6
    convergence_patience: int = 10
    initial_lipschitz: float = 1.0
    backtracking_factor: float = 2.0
    checkpoint_every: int = 25
    mode_chunk_size: int = 4
    sample_chunk_size: int = 16_384
    device: str = "auto"
    basis_selection_policy: str = "trusted_all_modes"

    def validate(self) -> None:
        """Reject ambiguous basis, graph, optimizer, or chunk settings."""

        positive_integers = (
            "local_rigid_basis_count",
            "graph_neighbors",
            "max_iterations",
            "convergence_patience",
            "checkpoint_every",
            "mode_chunk_size",
            "sample_chunk_size",
        )
        for name in positive_integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Motion-basis {name} must be a positive integer")
        if self.basis_selection_policy not in {"trusted_all_modes", "all_rigid_components", "trusted_per_mode"}:
            raise ValueError("Motion-basis basis_selection_policy must be trusted_all_modes, all_rigid_components, or trusted_per_mode")
        if self.basis_selection_policy == "trusted_per_mode":
            if self.rigid_basis_count is not None:
                raise ValueError("trusted_per_mode requires rigid_basis_count=None; the trusted union count is automatic")
        elif (isinstance(self.rigid_basis_count, bool) or not isinstance(self.rigid_basis_count, int)
              or self.rigid_basis_count <= 0):
            raise ValueError("Legacy motion-basis selection requires a positive exact rigid_basis_count")
        if self.rigid_basis_count is not None and self.local_rigid_basis_count > self.rigid_basis_count:
            raise ValueError(
                "Motion-basis local_rigid_basis_count cannot exceed rigid_basis_count"
            )
        if self.rigid_basis_count is not None and self.rigid_basis_count > np.iinfo(np.int8).max + 1:
            raise ValueError("Motion-basis rigid_basis_count exceeds the persisted owner-index capacity (128)")
        positive_floats = (
            "graph_max_distance",
            "distance_temperature",
            "zero_prior_score",
            "initial_lipschitz",
        )
        for name in positive_floats:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Motion-basis {name} must be finite and positive")
        nonnegative_floats = (
            "smooth_weight",
            "prior_weight",
            "relative_tolerance",
            "projected_gradient_tolerance",
        )
        for name in nonnegative_floats:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Motion-basis {name} must be finite and non-negative"
                )
        if (
            not math.isfinite(self.energy_floor_fraction)
            or not 0.0 < self.energy_floor_fraction <= 1.0
        ):
            raise ValueError("Motion-basis energy_floor_fraction must lie in (0,1]")
        if (
            not math.isfinite(self.backtracking_factor)
            or self.backtracking_factor <= 1.0
        ):
            raise ValueError("Motion-basis backtracking_factor must exceed one")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Motion-basis device must be 'auto', 'cpu', or 'cuda'")
        if (
            self.relative_tolerance == 0.0
            and self.projected_gradient_tolerance == 0.0
        ):
            raise ValueError("At least one motion-basis convergence tolerance is required")

    def to_dict(self) -> dict[str, Any]:
        """Serialize every scientific and numerical setting."""

        return {
            # Preserve the established serialized default and hence identities
            # and resumability of existing v3/v4/v5 artifacts. Only an opt-in
            # policy adds a new identity-bound configuration field.
            **({"basis_selection_policy": self.basis_selection_policy}
               if self.basis_selection_policy != "trusted_all_modes" else {}),
            "rigid_basis_count": self.rigid_basis_count,
            "local_rigid_basis_count": self.local_rigid_basis_count,
            "graph_neighbors": self.graph_neighbors,
            "graph_max_distance": self.graph_max_distance,
            "distance_temperature": self.distance_temperature,
            "zero_prior_score": self.zero_prior_score,
            "smooth_weight": self.smooth_weight,
            "prior_weight": self.prior_weight,
            "energy_floor_fraction": self.energy_floor_fraction,
            "max_iterations": self.max_iterations,
            "relative_tolerance": self.relative_tolerance,
            "projected_gradient_tolerance": self.projected_gradient_tolerance,
            "convergence_patience": self.convergence_patience,
            "initial_lipschitz": self.initial_lipschitz,
            "backtracking_factor": self.backtracking_factor,
            "checkpoint_every": self.checkpoint_every,
            "mode_chunk_size": self.mode_chunk_size,
            "sample_chunk_size": self.sample_chunk_size,
            "device": self.device,
            "precision": {
                "optimization": "float64_complex128",
                "persisted_weights": "float32",
                "persisted_field": "complex64",
            },
        }


def _config_from_manifest(payload: Mapping[str, Any]) -> MotionBasisConfig:
    """Read old/new configs without changing the bytes used for their identity."""

    if not isinstance(payload, Mapping):
        raise ValueError("Motion-basis configuration must be a mapping")
    try:
        settings = MotionBasisConfig(**{
            name: (payload.get(name, "trusted_all_modes") if name == "basis_selection_policy" else payload[name])
            for name in MotionBasisConfig.__dataclass_fields__
        })
    except (TypeError, KeyError) as error:
        raise ValueError("Motion-basis configuration fields are invalid") from error
    settings.validate()
    recorded = dict(payload)
    if recorded.pop("resolved_device", None) not in {"cpu", "cuda"}:
        raise ValueError("Motion-basis resolved device is invalid")
    if recorded.get("basis_selection_policy") == "trusted_all_modes":
        recorded.pop("basis_selection_policy")
    if recorded != settings.to_dict():
        raise ValueError("Motion-basis resolved configuration differs")
    return settings


@dataclass(frozen=True)
class ForegroundGraph:
    """Store one full-foreground distance-pruned union-KNN graph."""

    edge_index: np.ndarray
    edge_distance: np.ndarray
    edge_weight: np.ndarray
    degree: np.ndarray
    component_index: np.ndarray
    component_size: np.ndarray
    candidate_directed_count: int
    retained_directed_count: int


@dataclass(frozen=True)
class PixelBasisOperator:
    """Store the bounded-memory pixel-composited linear data operator."""

    design: np.ndarray
    measurements: np.ndarray
    contributor_sample_index: np.ndarray
    contributor_point_index: np.ndarray
    sample_loss_weight: np.ndarray
    mode_chunk_size: int

    @property
    def mode_count(self) -> int:
        """Return the number of complex modes."""

        return int(self.design.shape[0])

    @property
    def sample_count(self) -> int:
        """Return the number of measured topology pixels."""

        return int(self.measurements.shape[1])

    @property
    def basis_count(self) -> int:
        """Return the number of rigid-plus-zero bases."""

        return int(self.design.shape[2])


@dataclass(frozen=True)
class MotionBasisModesArtifact:
    """Represent one validated completed-mode v3 motion-basis artifact."""

    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


ARRAY_DTYPES = {
    "phi": np.dtype(np.complex64),
    "weights": np.dtype(np.float32),
    "spatial_prior_weights": np.dtype(np.float32),
    "basis_component_index": np.dtype(np.int32),
    "basis_owner_index": np.dtype(np.int8),
    "basis_translation": np.dtype(np.complex64),
    "basis_rotation": np.dtype(np.complex64),
    "basis_centroid": np.dtype(np.float32),
    "basis_radius": np.dtype(np.float32),
    "candidate_mask": np.dtype(bool),
    "measurement_supported_mask": np.dtype(bool),
    "graph_propagated_mask": np.dtype(bool),
    "zero_fallback_mask": np.dtype(bool),
    "measurement_contributor_count": np.dtype(np.int32),
    "graph_degree": np.dtype(np.int32),
    "graph_component_index": np.dtype(np.int32),
    "spatial_edge_index": np.dtype(np.int32),
    "spatial_edge_distance": np.dtype(np.float32),
    "spatial_edge_weight": np.dtype(np.float32),
    "weight_entropy": np.dtype(np.float32),
    "dominant_basis_index": np.dtype(np.int16),
    "sample_residual_rms": np.dtype(np.float32),
}


def _canonical_json(value: Any) -> bytes:
    """Encode a deterministic JSON identity payload."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    """Hash one file without reading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrays_identity(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash names, dtypes, shapes, and values in stable order."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _require_ckdtree() -> Any:
    """Import SciPy's exact CPU KNN implementation behind a typed boundary."""

    try:
        spatial = importlib.import_module("scipy.spatial")
    except ImportError as error:
        raise RuntimeError("Motion-basis fitting requires scipy") from error
    tree = getattr(spatial, "cKDTree", None)
    if tree is None:
        raise RuntimeError("scipy.spatial.cKDTree is unavailable")
    return tree


def _stable_components(point_count: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label graph components deterministically by their smallest point."""

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

    for point_i, point_j in np.asarray(edges, dtype=np.int64).tolist():
        root_i, root_j = find(int(point_i)), find(int(point_j))
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)
    roots = np.asarray([find(point) for point in range(point_count)], dtype=np.int64)
    unique = np.unique(roots)
    root_to_component = np.full(point_count, -1, dtype=np.int64)
    root_to_component[unique] = np.arange(len(unique), dtype=np.int64)
    component = root_to_component[roots].astype(np.int32)
    sizes = np.bincount(component, minlength=len(unique)).astype(np.int32)
    return component, sizes


def build_foreground_graph(
    points: np.ndarray, config: MotionBasisConfig
) -> ForegroundGraph:
    """Build the full-FG graph without RGB/depth component boundaries."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("Motion-basis points must be finite [G,3]")
    if not np.isfinite(values).all():
        raise ValueError("Motion-basis points contain NaN or Inf")
    if config.graph_neighbors >= len(values):
        raise ValueError("Motion-basis graph_neighbors must be smaller than G")
    tree = _require_ckdtree()(values)
    distances, _ = tree.query(values, k=config.graph_neighbors + 1, workers=1)
    boundaries = np.nextafter(np.asarray(distances)[:, -1], np.inf)
    candidate_rows = tree.query_ball_point(values, boundaries, return_sorted=False)
    neighbors = np.empty((len(values), config.graph_neighbors), dtype=np.int64)
    neighbor_distances = np.empty_like(neighbors, dtype=np.float64)
    for point, row in enumerate(candidate_rows):
        row_indices = np.asarray(row, dtype=np.int64)
        row_indices = row_indices[row_indices != point]
        row_distances = np.linalg.norm(values[row_indices] - values[point], axis=1)
        if len(row_indices) < config.graph_neighbors:
            raise RuntimeError(f"KNN returned too few non-self points for {point}")
        order = np.lexsort((row_indices, row_distances))[: config.graph_neighbors]
        neighbors[point] = row_indices[order]
        neighbor_distances[point] = row_distances[order]
    source = np.repeat(np.arange(len(values), dtype=np.int64), config.graph_neighbors)
    target = neighbors.reshape(-1)
    directed_count = len(source)
    directed_distance = neighbor_distances.reshape(-1)
    keep = directed_distance <= config.graph_max_distance
    retained_directed = int(np.count_nonzero(keep))
    pairs = np.column_stack(
        (np.minimum(source[keep], target[keep]), np.maximum(source[keep], target[keep]))
    )
    edge_index = (
        np.unique(pairs, axis=0)
        if len(pairs)
        else np.empty((0, 2), dtype=np.int64)
    )
    edge_distance = (
        np.linalg.norm(values[edge_index[:, 0]] - values[edge_index[:, 1]], axis=1)
        if len(edge_index)
        else np.empty(0, dtype=np.float64)
    )
    edge_weight = np.exp(-np.square(edge_distance / config.graph_max_distance))
    degree = np.zeros(len(values), dtype=np.int32)
    if len(edge_index):
        np.add.at(degree, edge_index[:, 0], 1)
        np.add.at(degree, edge_index[:, 1], 1)
    component, component_size = _stable_components(len(values), edge_index)
    return ForegroundGraph(
        edge_index=edge_index.astype(np.int64),
        edge_distance=edge_distance.astype(np.float64),
        edge_weight=edge_weight.astype(np.float64),
        degree=degree,
        component_index=component,
        component_size=component_size,
        candidate_directed_count=directed_count,
        retained_directed_count=retained_directed,
    )


def _select_bases(rigid: Any, config: MotionBasisConfig) -> tuple[np.ndarray, np.ndarray]:
    """Select stable source component indices using the explicit trust policy."""

    config.validate()
    retained = np.asarray(rigid.arrays["component_retained_mask"], dtype=bool)
    if retained.ndim != 2:
        raise ValueError("Rigid component_retained_mask must have shape [K,C]")
    if config.basis_selection_policy == "trusted_all_modes":
        eligible = np.flatnonzero(np.all(retained, axis=0)).astype(np.int32)
        description = "components trusted at every mode"
    elif config.basis_selection_policy == "trusted_per_mode":
        empty_modes = np.flatnonzero(~np.any(retained, axis=1))
        if len(empty_modes):
            descriptions = []
            modes = rigid.manifest.get("modes", [])
            for mode in empty_modes.tolist():
                frequency_hz = modes[mode].get("frequency_hz", "unknown") if mode < len(modes) else "unknown"
                descriptions.append(f"mode slot {mode} (frequency_hz={frequency_hz})")
            raise ValueError("trusted_per_mode found no trusted rigid component for " + ", ".join(descriptions))
        eligible = np.flatnonzero(np.any(retained, axis=0)).astype(np.int32)
        if len(eligible) > np.iinfo(np.int8).max + 1:
            raise ValueError("Trusted component union exceeds the persisted owner-index capacity (128)")
        for name in ("component_translation", "component_rotation"):
            raw = np.asarray(rigid.arrays[name])
            if raw.shape != (*retained.shape, 3) or not np.isfinite(raw[retained]).all():
                raise ValueError(f"trusted_per_mode requires finite trusted raw {name}")
        return eligible, eligible.copy()
    else:
        # Trust values remain untouched. The C axis belongs to the raw rigid
        # solutions, not the graph's isolated nodes nor the trusted subset.
        translation = np.asarray(rigid.arrays["component_translation"])
        rotation = np.asarray(rigid.arrays["component_rotation"])
        node_count = np.asarray(rigid.arrays["component_node_count"])
        edge_count = np.asarray(rigid.arrays["component_edge_count"])
        expected = (*retained.shape, 3)
        if (translation.shape != expected or rotation.shape != expected
                or node_count.shape != retained.shape[1:]
                or edge_count.shape != retained.shape[1:]
                or np.any(node_count < 2) or np.any(edge_count < 1)):
            raise ValueError("All-components basis policy requires raw nontrivial rigid components")
        if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
            raise ValueError("All-components basis policy requires finite raw rigid twists")
        eligible = np.arange(retained.shape[1], dtype=np.int32)
        description = "solved nontrivial rigid components (trust filter disabled)"
    if len(eligible) != config.rigid_basis_count:
        raise ValueError(
            "Motion-basis fitting requires exactly "
            f"{config.rigid_basis_count} {description}; "
            f"found {len(eligible)}"
        )
    selected = np.sort(eligible, kind="stable").astype(np.int32)
    return eligible, selected


def _basis_arrays(
    rigid: Any, selected: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Append one explicit zero basis to the selected raw rigid twists."""

    translation = np.asarray(
        rigid.arrays["component_translation"][:, selected], dtype=np.complex64
    )
    rotation = np.asarray(
        rigid.arrays["component_rotation"][:, selected], dtype=np.complex64
    )
    centroid = np.asarray(rigid.arrays["component_centroid"][selected], dtype=np.float32)
    radius = np.asarray(rigid.arrays["component_radius"][selected], dtype=np.float32)
    mode_count = translation.shape[0]
    translation = np.concatenate(
        [translation, np.zeros((mode_count, 1, 3), dtype=np.complex64)], axis=1
    )
    rotation = np.concatenate(
        [rotation, np.zeros((mode_count, 1, 3), dtype=np.complex64)], axis=1
    )
    centroid = np.concatenate(
        [centroid, np.zeros((1, 3), dtype=np.float32)], axis=0
    )
    radius = np.concatenate([radius, np.zeros(1, dtype=np.float32)])
    component_indices = np.concatenate(
        [selected, np.asarray([-1], dtype=np.int32)]
    )
    return component_indices, translation, rotation, centroid, radius


def _candidate_and_prior_weights(
    points: np.ndarray,
    point_component: np.ndarray,
    selected: np.ndarray,
    config: MotionBasisConfig,
    *, basis_active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign nearest-support candidates and exp(-distance/T) priors.

    Distance is measured to the nearest actual member of each selected rigid
    component, not merely to its centroid.  This prevents a long or curved
    component from receiving an artificially distant prior near its ends.
    """

    point_values = np.asarray(points, dtype=np.float64)
    point_count = len(point_values)
    basis_count = len(selected) + 1
    if basis_active_mask is not None:
        active = np.asarray(basis_active_mask)
        if (active.dtype != np.bool_ or active.ndim != 2 or active.shape[1] != basis_count
                or not np.all(active[:, -1]) or not np.all(np.any(active[:, :-1], axis=1))):
            raise ValueError("Per-mode candidate construction requires active rigid bases and an active zero basis")
        candidates = np.zeros((len(active), point_count, basis_count), dtype=bool)
        priors = np.zeros(candidates.shape, dtype=np.float64)
        for mode, active_row in enumerate(active):
            active_slots = np.flatnonzero(active_row[:-1])
            local_candidate, local_prior = _candidate_and_prior_weights(
                points, point_component, selected[active_slots], config,
            )
            candidates[mode][:, active_slots] = local_candidate[:, :-1]
            priors[mode][:, active_slots] = local_prior[:, :-1]
            candidates[mode, :, -1] = True
            priors[mode, :, -1] = local_prior[:, -1]
        return candidates, priors
    support_distance = np.empty((point_count, len(selected)), dtype=np.float64)
    tree_type = _require_ckdtree()
    for basis, component in enumerate(selected.tolist()):
        members = np.flatnonzero(point_component == component)
        if len(members) < 2:
            raise RuntimeError(
                f"Selected rigid component {component} has too few member points"
            )
        distance, _ = tree_type(point_values[members]).query(
            point_values, k=1, workers=1
        )
        support_distance[:, basis] = np.asarray(distance, dtype=np.float64)
    candidate = np.zeros((point_count, basis_count), dtype=bool)
    prior = np.zeros((point_count, basis_count), dtype=np.float64)
    component_to_basis = {
        int(component): basis for basis, component in enumerate(selected.tolist())
    }
    for lower in range(0, point_count, config.sample_chunk_size):
        upper = min(point_count, lower + config.sample_chunk_size)
        distance = support_distance[lower:upper]
        order = np.argsort(distance, axis=1, kind="stable")
        local = order[:, : config.local_rigid_basis_count]
        rows = np.arange(upper - lower, dtype=np.int64)[:, None]
        chunk_candidate = np.zeros((upper - lower, basis_count), dtype=bool)
        chunk_candidate[rows, local] = True
        for local_row, component in enumerate(point_component[lower:upper].tolist()):
            own_basis = component_to_basis.get(int(component))
            if own_basis is None or chunk_candidate[local_row, own_basis]:
                continue
            selected_slots = np.flatnonzero(chunk_candidate[local_row, :-1])
            farthest = selected_slots[
                np.argmax(distance[local_row, selected_slots])
            ]
            chunk_candidate[local_row, int(farthest)] = False
            chunk_candidate[local_row, own_basis] = True
        chunk_candidate[:, -1] = True
        score = np.exp(-distance / config.distance_temperature)
        score[~chunk_candidate[:, :-1]] = 0.0
        chunk_prior = np.zeros_like(chunk_candidate, dtype=np.float64)
        chunk_prior[:, :-1] = score
        chunk_prior[:, -1] = config.zero_prior_score
        chunk_prior /= np.sum(chunk_prior, axis=1, keepdims=True)
        candidate[lower:upper] = chunk_candidate
        prior[lower:upper] = chunk_prior
    if not np.all(candidate[:, -1]):
        raise RuntimeError("Zero motion must be a candidate for every Gaussian")
    return candidate, prior


def _role_masks(
    *,
    graph: ForegroundGraph,
    topology: Any,
    alpha_identifiable: np.ndarray,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition FG points into measurement, graph-propagated, and zero roles."""

    contributor_point = np.asarray(
        topology.arrays.contributor_gaussian_index, dtype=np.int64
    )
    offsets = np.asarray(topology.arrays.sample_offsets, dtype=np.int64)
    contributor_count = np.diff(offsets)
    sample_index = np.repeat(np.arange(len(contributor_count), dtype=np.int64), contributor_count)
    sample_view = np.asarray(topology.arrays.sample_view_index, dtype=np.int64)
    row_view = sample_view[sample_index]
    valid_row = np.any(alpha_identifiable[:, row_view], axis=0)
    counts = np.zeros(point_count, dtype=np.int32)
    np.add.at(counts, contributor_point[valid_row], 1)
    supported = counts > 0
    component_supported = np.zeros(len(graph.component_size), dtype=bool)
    component_supported[graph.component_index[supported]] = True
    connected = component_supported[graph.component_index]
    propagated = ~supported & connected
    zero_fallback = ~connected
    if not np.all(supported | propagated | zero_fallback):
        raise RuntimeError("Motion-basis role masks do not cover foreground")
    if np.any((supported & propagated) | (supported & zero_fallback) | (propagated & zero_fallback)):
        raise RuntimeError("Motion-basis role masks overlap")
    return supported, propagated, zero_fallback, counts


def _project_masked_simplex(
    values: np.ndarray,
    candidate_mask: np.ndarray,
    zero_fallback_mask: np.ndarray,
) -> np.ndarray:
    """Project every row onto its local simplex and fix fallback rows to zero."""

    x = np.asarray(values, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    if x.shape != mask.shape:
        raise ValueError("Masked-simplex value and candidate shapes differ")
    masked = np.where(mask, x, -np.inf)
    ordered = np.sort(masked, axis=1)[:, ::-1]
    finite = np.isfinite(ordered)
    cumulative = np.cumsum(np.where(finite, ordered, 0.0), axis=1) - 1.0
    divisor = np.arange(1, x.shape[1] + 1, dtype=np.float64)[None]
    active = finite & (ordered - cumulative / divisor > 0.0)
    rho = np.count_nonzero(active, axis=1)
    if np.any(rho <= 0):
        raise RuntimeError("Masked-simplex projection found an empty candidate set")
    theta = cumulative[np.arange(len(x)), rho - 1] / rho
    projected = np.maximum(x - theta[:, None], 0.0)
    projected[~mask] = 0.0
    if np.any(zero_fallback_mask):
        projected[zero_fallback_mask] = 0.0
        projected[zero_fallback_mask, -1] = 1.0
    return projected


def _build_design_file(
    *,
    path: Path,
    points: np.ndarray,
    topology: Any,
    alphas: np.ndarray,
    basis_translation: np.ndarray,
    basis_rotation: np.ndarray,
    basis_centroid: np.ndarray,
    candidate_mask: np.ndarray,
    config: MotionBasisConfig,
) -> np.memmap:
    """Materialize a reusable contributor operator without a K*G*B*3 field."""

    contributor_point = np.asarray(
        topology.arrays.contributor_gaussian_index, dtype=np.int64
    )
    offsets = np.asarray(topology.arrays.sample_offsets, dtype=np.int64)
    sample_index = np.repeat(
        np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets)
    )
    sample_view = np.asarray(topology.arrays.sample_view_index, dtype=np.int64)
    row_view = sample_view[sample_index]
    jacobian = np.asarray(topology.arrays.contributor_jacobian, dtype=np.float64)
    contribution = np.asarray(topology.arrays.contributor_weight, dtype=np.float64)
    shape = (
        basis_translation.shape[0],
        len(contributor_point),
        basis_translation.shape[1],
        2,
    )
    if path.is_file():
        existing = np.load(path, mmap_mode="r", allow_pickle=False)
        if existing.dtype != np.complex64 or existing.shape != shape:
            raise ValueError("Motion-basis cached pixel design has an invalid shape/dtype")
        for lower in range(0, shape[1], config.sample_chunk_size):
            if not np.isfinite(existing[:, lower : lower + config.sample_chunk_size]).all():
                raise ValueError("Motion-basis cached pixel design is non-finite")
        return existing
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    progress = Progress("motion-basis operator", len(contributor_point), unit="contributors")
    try:
        design = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.complex64, shape=shape
        )
        try:
            for lower in range(0, len(contributor_point), config.sample_chunk_size):
                upper = min(len(contributor_point), lower + config.sample_chunk_size)
                point_index = contributor_point[lower:upper]
                centered = (
                    np.asarray(points[point_index], dtype=np.float64)[:, None, :]
                    - np.asarray(basis_centroid, dtype=np.float64)[None, :, :]
                )
                for mode_lower in range(0, shape[0], config.mode_chunk_size):
                    mode_upper = min(shape[0], mode_lower + config.mode_chunk_size)
                    field = (
                        basis_translation[mode_lower:mode_upper].astype(np.complex128)[
                            :, None, :, :
                        ]
                        + np.cross(
                            basis_rotation[mode_lower:mode_upper].astype(np.complex128)[
                                :, None, :, :
                            ],
                            centered[None, :, :, :],
                            axis=-1,
                        )
                    )
                    projected = np.einsum(
                        "mij,kmbj->kmbi", jacobian[lower:upper], field, optimize=True
                    )
                    projected *= alphas[
                        mode_lower:mode_upper, row_view[lower:upper], None, None
                    ]
                    projected *= contribution[None, lower:upper, None, None]
                    if candidate_mask.ndim == 2:
                        projected *= candidate_mask[point_index][None, :, :, None]
                    else:
                        projected *= candidate_mask[mode_lower:mode_upper, point_index, :, None]
                    design[mode_lower:mode_upper, lower:upper] = projected.astype(
                        np.complex64
                    )
                progress.update(upper, force=upper == len(contributor_point))
            design.flush()
        finally:
            del design
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _sample_loss_weights(
    measurements: np.ndarray,
    sample_views: np.ndarray,
    sample_foreground_alpha: np.ndarray,
    identifiable: np.ndarray,
    config: MotionBasisConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Alpha-weight and equalize mode/view blocks with a low-energy floor."""

    values = np.asarray(measurements)
    mode_count, sample_count, _ = values.shape
    confidence = np.asarray(sample_foreground_alpha, dtype=np.float64)
    if (
        confidence.shape != (sample_count,)
        or not np.isfinite(confidence).all()
        or np.any(confidence <= 0.0)
    ):
        raise ValueError("Motion-basis sample foreground alpha is invalid")
    view_count = identifiable.shape[1]
    rms = np.zeros((mode_count, view_count), dtype=np.float64)
    valid_blocks: list[tuple[int, int, np.ndarray]] = []
    for mode in range(mode_count):
        for view in range(view_count):
            rows = np.flatnonzero(sample_views == view)
            if identifiable[mode, view] and len(rows):
                alpha_sum = float(np.sum(confidence[rows]))
                energy = np.sum(
                    confidence[rows]
                    * np.sum(np.abs(values[mode, rows]) ** 2, axis=1)
                ) / max(alpha_sum, EPSILON)
                rms[mode, view] = math.sqrt(max(float(energy), 0.0))
                valid_blocks.append((mode, view, rows))
    if not valid_blocks:
        raise ValueError("Motion-basis fitting has no identifiable mode/view block")
    positive = rms[rms > 0.0]
    reference = float(np.median(positive)) if len(positive) else 1.0
    floor = max(config.energy_floor_fraction * reference, EPSILON)
    weight = np.zeros((mode_count, sample_count), dtype=np.float64)
    for mode, view, rows in valid_blocks:
        scale = max(rms[mode, view], floor)
        alpha_sum = float(np.sum(confidence[rows]))
        weight[mode, rows] = confidence[rows] / (
            len(valid_blocks) * alpha_sum * scale * scale
        )
    return weight, rms, floor


def _resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda and fail clearly when CUDA was requested."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Motion-basis device='cuda' requires an available CUDA GPU")
    return torch.device(requested)


def _project_masked_simplex_torch(
    values: torch.Tensor,
    candidate_mask: torch.Tensor,
    zero_fallback_mask: torch.Tensor,
) -> torch.Tensor:
    """Project rows to their masked simplex directly on the selected device."""

    negative_infinity = torch.tensor(
        -torch.inf, dtype=values.dtype, device=values.device
    )
    masked = torch.where(candidate_mask, values, negative_infinity)
    ordered = torch.sort(masked, dim=1, descending=True).values
    finite = torch.isfinite(ordered)
    cumulative = torch.cumsum(torch.where(finite, ordered, 0.0), dim=1) - 1.0
    divisor = torch.arange(
        1, values.shape[1] + 1, dtype=values.dtype, device=values.device
    )[None]
    active = finite & (ordered - cumulative / divisor > 0.0)
    rho = torch.count_nonzero(active, dim=1)
    if bool(torch.any(rho <= 0).item()):
        raise RuntimeError("Masked-simplex projection found an empty candidate set")
    rows = torch.arange(len(values), device=values.device)
    theta = cumulative[rows, rho - 1] / rho.to(values.dtype)
    projected = torch.clamp(values - theta[:, None], min=0.0)
    projected = torch.where(candidate_mask, projected, 0.0)
    if bool(torch.any(zero_fallback_mask).item()):
        projected[zero_fallback_mask] = 0.0
        projected[zero_fallback_mask, -1] = 1.0
    return projected


class _TorchObjective:
    """Evaluate the convex pixel, full-FG graph, and distance-prior objective."""

    def __init__(
        self,
        *,
        operator: PixelBasisOperator,
        graph: ForegroundGraph,
        prior: np.ndarray,
        config: MotionBasisConfig,
        device: torch.device,
    ) -> None:
        self.device = device
        self.config = config
        self.design = torch.from_numpy(np.array(operator.design, copy=True)).to(
            device=device, dtype=torch.complex64
        )
        self.measurements = torch.from_numpy(
            np.array(operator.measurements, copy=True)
        ).to(device=device, dtype=torch.complex64)
        self.contributor_sample_index = torch.from_numpy(
            np.asarray(operator.contributor_sample_index, dtype=np.int64)
        ).to(device)
        self.contributor_point_index = torch.from_numpy(
            np.asarray(operator.contributor_point_index, dtype=np.int64)
        ).to(device)
        self.sample_loss_weight = torch.from_numpy(
            np.asarray(operator.sample_loss_weight, dtype=np.float64)
        ).to(device)
        self.edge_index = torch.from_numpy(
            np.asarray(graph.edge_index, dtype=np.int64)
        ).to(device)
        self.edge_weight = torch.from_numpy(
            np.asarray(graph.edge_weight, dtype=np.float64)
        ).to(device)
        self.edge_weight_sum = torch.clamp(torch.sum(self.edge_weight), min=EPSILON)
        self.prior = torch.from_numpy(np.asarray(prior, dtype=np.float64)).to(device)
        self.prior_denominator = max(float(len(prior)), 1.0)
        self.mode_count = operator.mode_count
        self.sample_count = operator.sample_count
        self.mode_chunk_size = config.mode_chunk_size
        self.sample_chunk_size = config.sample_chunk_size

    def value_gradient(
        self,
        weights: torch.Tensor,
        *,
        gradient: bool,
        return_residual: bool = False,
    ) -> tuple[float, dict[str, float], torch.Tensor | None, np.ndarray | None]:
        """Evaluate objective and its exact real gradient for real weights."""

        result_gradient = torch.zeros_like(weights) if gradient else None
        residual_output = (
            np.empty((self.mode_count, self.sample_count), dtype=np.float32)
            if return_residual
            else None
        )
        data_value = torch.zeros((), dtype=torch.float64, device=self.device)
        contributor_count = len(self.contributor_point_index)
        for mode_lower in range(0, self.mode_count, self.mode_chunk_size):
            mode_upper = min(self.mode_count, mode_lower + self.mode_chunk_size)
            prediction = torch.zeros(
                (mode_upper - mode_lower, self.sample_count, 2),
                dtype=torch.complex128,
                device=self.device,
            )
            for lower in range(0, contributor_count, self.sample_chunk_size):
                upper = min(contributor_count, lower + self.sample_chunk_size)
                design = self.design[mode_lower:mode_upper, lower:upper].to(
                    torch.complex128
                )
                point_index = self.contributor_point_index[lower:upper]
                contribution = torch.einsum(
                    "kmbd,mb->kmd",
                    design,
                    weights[point_index].to(torch.complex128),
                )
                prediction.index_add_(
                    1, self.contributor_sample_index[lower:upper], contribution
                )
            target = self.measurements[mode_lower:mode_upper].to(torch.complex128)
            residual = prediction - target
            loss_weight = self.sample_loss_weight[mode_lower:mode_upper]
            data_value += torch.sum(loss_weight[:, :, None] * torch.abs(residual) ** 2)
            if residual_output is not None:
                residual_output[mode_lower:mode_upper] = (
                    torch.sqrt(torch.mean(torch.abs(residual) ** 2, dim=2))
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
            if result_gradient is not None:
                weighted_residual = loss_weight[:, :, None] * residual
                for lower in range(0, contributor_count, self.sample_chunk_size):
                    upper = min(contributor_count, lower + self.sample_chunk_size)
                    design = self.design[mode_lower:mode_upper, lower:upper].to(
                        torch.complex128
                    )
                    rows = self.contributor_sample_index[lower:upper]
                    contribution_gradient = 2.0 * torch.real(
                        torch.einsum(
                            "kmbd,kmd->mb",
                            torch.conj(design),
                            weighted_residual[:, rows],
                        )
                    )
                    result_gradient.index_add_(
                        0,
                        self.contributor_point_index[lower:upper],
                        contribution_gradient,
                    )
        if len(self.edge_index):
            point_i, point_j = self.edge_index[:, 0], self.edge_index[:, 1]
            difference = weights[point_i] - weights[point_j]
            graph_value = (
                self.config.smooth_weight
                * torch.sum(self.edge_weight[:, None] * difference * difference)
                / self.edge_weight_sum
            )
            if result_gradient is not None and self.config.smooth_weight > 0.0:
                edge_gradient = (
                    2.0
                    * self.config.smooth_weight
                    * self.edge_weight[:, None]
                    * difference
                    / self.edge_weight_sum
                )
                result_gradient.index_add_(0, point_i, edge_gradient)
                result_gradient.index_add_(0, point_j, -edge_gradient)
        else:
            graph_value = torch.zeros((), dtype=torch.float64, device=self.device)
        prior_difference = weights - self.prior
        prior_value = (
            self.config.prior_weight
            * torch.sum(prior_difference * prior_difference)
            / self.prior_denominator
        )
        if result_gradient is not None and self.config.prior_weight > 0.0:
            result_gradient += (
                2.0
                * self.config.prior_weight
                * prior_difference
                / self.prior_denominator
            )
        total = data_value + graph_value + prior_value
        if not bool(torch.isfinite(total).item()) or (
            result_gradient is not None
            and not bool(torch.isfinite(result_gradient).all().item())
        ):
            raise FloatingPointError("Motion-basis objective/gradient is non-finite")
        terms = {
            "total": float(total.item()),
            "data": float(data_value.item()),
            "graph": float(graph_value.item()),
            "prior": float(prior_value.item()),
        }
        return terms["total"], terms, result_gradient, residual_output


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write once, then atomically replace with bounded Windows lock retries."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        save_named_arrays(temporary, arrays)
        # Windows file scanners/readers can briefly deny replacement after the
        # NPZ writer has closed. Retry only the atomic rename, never the write
        # or solver step; the old checkpoint remains valid until success.
        retry_delays = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
        for attempt in range(len(retry_delays) + 1):
            try:
                os.replace(temporary, path)
                break
            except OSError as error:
                if (os.name != "nt" or getattr(error, "winerror", None) not in {5, 32, 33}
                        or attempt == len(retry_delays)):
                    raise
                time.sleep(retry_delays[attempt])
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # Preserve the original write/replace error if cleanup also meets
            # a transient lock; never delete or truncate the old checkpoint.
            pass
        raise


def _prepare_work_dir(
    work: Path, source: Mapping[str, Any], *, resume: bool
) -> None:
    """Create or identity-check the resumable motion-basis work directory."""

    manifest_path = work / WORK_MANIFEST_FILENAME
    if work.exists() and not work.is_dir():
        raise FileExistsError(f"Motion-basis work path is not a directory: {work}")
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(
                "Motion-basis work directory already contains a run; pass resume=True"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("solver_run_identity") != source["solver_run_identity"]:
            raise ValueError("Motion-basis work directory belongs to another run")
        return
    if resume:
        raise FileNotFoundError(
            "Motion-basis resume requested but the work manifest does not exist"
        )
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise FileExistsError("Motion-basis work directory is non-empty without manifest")
    payload = {
        "format": "modal_gaussians.motion_basis_work",
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


def _load_checkpoint(
    path: Path,
    *,
    solver_run_identity: str,
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    """Load a complete optimizer state or reject a stale/partial checkpoint."""

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    expected = {
        "weights",
        "previous_weights",
        "iteration",
        "fista_t",
        "lipschitz",
        "stable_count",
        "line_search_steps_total",
        "objective_history",
        "solver_run_identity",
    }
    if set(arrays) != expected:
        raise ValueError("Motion-basis checkpoint fields are invalid")
    if str(arrays["solver_run_identity"]) != solver_run_identity:
        raise ValueError("Motion-basis checkpoint identity differs")
    for name in ("weights", "previous_weights"):
        if arrays[name].shape != expected_shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Motion-basis checkpoint {name} is invalid")
    scalars = {
        "iteration": int(arrays["iteration"]),
        "fista_t": float(arrays["fista_t"]),
        "lipschitz": float(arrays["lipschitz"]),
        "stable_count": int(arrays["stable_count"]),
        "line_search_steps_total": int(arrays["line_search_steps_total"]),
    }
    if (
        scalars["iteration"] < 0
        or scalars["fista_t"] < 1.0
        or scalars["lipschitz"] <= 0.0
        or scalars["line_search_steps_total"] < 0
    ):
        raise ValueError("Motion-basis checkpoint scalar state is invalid")
    history = np.asarray(arrays["objective_history"], dtype=np.float64)
    if history.ndim != 1 or len(history) != scalars["iteration"] + 1 or not np.isfinite(history).all():
        raise ValueError("Motion-basis checkpoint objective history is invalid")
    return {
        **scalars,
        "weights": np.asarray(arrays["weights"], dtype=np.float64),
        "previous_weights": np.asarray(arrays["previous_weights"], dtype=np.float64),
        "objective_history": history.tolist(),
    }


def _save_checkpoint(
    path: Path,
    *,
    solver_run_identity: str,
    weights: np.ndarray,
    previous_weights: np.ndarray,
    iteration: int,
    fista_t: float,
    lipschitz: float,
    stable_count: int,
    line_search_steps_total: int,
    objective_history: Sequence[float],
) -> None:
    """Persist all state needed for an exact next FISTA iteration."""

    _atomic_savez(
        path,
        {
            "weights": np.asarray(weights, dtype=np.float64),
            "previous_weights": np.asarray(previous_weights, dtype=np.float64),
            "iteration": np.asarray(iteration, dtype=np.int32),
            "fista_t": np.asarray(fista_t, dtype=np.float64),
            "lipschitz": np.asarray(lipschitz, dtype=np.float64),
            "stable_count": np.asarray(stable_count, dtype=np.int32),
            "line_search_steps_total": np.asarray(
                line_search_steps_total, dtype=np.int64
            ),
            "objective_history": np.asarray(objective_history, dtype=np.float64),
            "solver_run_identity": np.asarray(solver_run_identity, dtype="<U64"),
        },
    )


def _solve_weights(
    *,
    operator: PixelBasisOperator,
    graph: ForegroundGraph,
    candidate_mask: np.ndarray,
    prior: np.ndarray,
    measurement_supported_mask: np.ndarray,
    zero_fallback_mask: np.ndarray,
    work: Path,
    solver_run_identity: str,
    config: MotionBasisConfig,
    device: torch.device,
    resume: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run resumable monotone projected FISTA on the global convex objective."""

    checkpoint_path = work / CHECKPOINT_FILENAME
    objective = _TorchObjective(
        operator=operator,
        graph=graph,
        prior=prior,
        config=config,
        device=device,
    )
    candidate_tensor = torch.from_numpy(np.asarray(candidate_mask, dtype=bool)).to(
        device
    )
    fallback_tensor = torch.from_numpy(
        np.asarray(zero_fallback_mask, dtype=bool)
    ).to(device)
    del measurement_supported_mask  # Support affects roles, not prior strength.
    if resume and checkpoint_path.is_file():
        state = _load_checkpoint(
            checkpoint_path,
            solver_run_identity=solver_run_identity,
            expected_shape=prior.shape,
        )
        current = torch.from_numpy(state["weights"]).to(device)
        previous = torch.from_numpy(state["previous_weights"]).to(device)
        for name, value in (("weights", current), ("previous_weights", previous)):
            projected_state = _project_masked_simplex_torch(
                value, candidate_tensor, fallback_tensor
            )
            if not bool(
                torch.allclose(value, projected_state, rtol=0.0, atol=1.0e-10)
            ):
                raise ValueError(
                    f"Motion-basis checkpoint {name} violates its masked simplex"
                )
        start_iteration = state["iteration"]
        fista_t = state["fista_t"]
        lipschitz = state["lipschitz"]
        stable_count = state["stable_count"]
        line_search_steps_total = state["line_search_steps_total"]
        history = state["objective_history"]
        report_progress(f"motion-basis fit: resuming after iteration {start_iteration}")
    else:
        initial = _project_masked_simplex(prior, candidate_mask, zero_fallback_mask)
        current = torch.from_numpy(initial).to(device)
        previous = current.clone()
        start_iteration = 0
        fista_t = 1.0
        lipschitz = config.initial_lipschitz
        stable_count = 0
        line_search_steps_total = 0
        initial_value, _, _, _ = objective.value_gradient(current, gradient=False)
        history = [initial_value]
        _save_checkpoint(
            checkpoint_path,
            solver_run_identity=solver_run_identity,
            weights=current.cpu().numpy(),
            previous_weights=previous.cpu().numpy(),
            iteration=0,
            fista_t=fista_t,
            lipschitz=lipschitz,
            stable_count=stable_count,
            line_search_steps_total=line_search_steps_total,
            objective_history=history,
        )
    progress = Progress("motion-basis fit", config.max_iterations, unit="iterations")
    progress.update(start_iteration, force=True)
    converged = stable_count >= config.convergence_patience
    projected_gradient_norm = math.inf
    iteration = start_iteration
    while iteration < config.max_iterations and not converged:
        next_t = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * fista_t * fista_t))
        momentum = (fista_t - 1.0) / next_t
        extrapolated = _project_masked_simplex_torch(
            current + momentum * (current - previous),
            candidate_tensor,
            fallback_tensor,
        )
        extrapolated_value, _, grad, _ = objective.value_gradient(
            extrapolated, gradient=True
        )
        if grad is None:
            raise RuntimeError("Motion-basis optimizer did not return a gradient")

        def line_search(
            origin: torch.Tensor,
            origin_value: float,
            origin_grad: torch.Tensor,
        ) -> tuple[torch.Tensor, float, float, int]:
            trial_lipschitz = lipschitz
            for search_step in range(1, 42):
                candidate = _project_masked_simplex_torch(
                    origin - origin_grad / trial_lipschitz,
                    candidate_tensor,
                    fallback_tensor,
                )
                difference = candidate - origin
                candidate_value, _, _, _ = objective.value_gradient(
                    candidate, gradient=False
                )
                majorizer = (
                    origin_value
                    + float(torch.sum(origin_grad * difference).item())
                    + 0.5
                    * trial_lipschitz
                    * float(torch.sum(difference * difference).item())
                )
                tolerance = 1.0e-12 * max(1.0, abs(origin_value))
                if candidate_value <= majorizer + tolerance:
                    return candidate, candidate_value, trial_lipschitz, search_step
                trial_lipschitz *= config.backtracking_factor
            raise RuntimeError("Motion-basis FISTA backtracking did not find a valid step")

        candidate, candidate_value, lipschitz, searches = line_search(
            extrapolated, extrapolated_value, grad
        )
        line_search_steps_total += searches
        if candidate_value > history[-1] + 1.0e-12 * max(1.0, abs(history[-1])):
            fista_t = 1.0
            current_value, _, current_grad, _ = objective.value_gradient(
                current, gradient=True
            )
            if current_grad is None:
                raise RuntimeError("Motion-basis restart did not return a gradient")
            candidate, candidate_value, lipschitz, searches = line_search(
                current, current_value, current_grad
            )
            line_search_steps_total += searches
            next_t = 1.0
            grad = current_grad
            extrapolated = current
        step_difference = candidate - extrapolated
        variable_count = max(
            int(torch.count_nonzero(candidate_tensor[~fallback_tensor]).item()), 1
        )
        projected_gradient_norm = lipschitz * float(
            torch.linalg.vector_norm(step_difference).item()
        ) / math.sqrt(variable_count)
        relative_change = abs(history[-1] - candidate_value) / max(
            abs(history[-1]), EPSILON
        )
        if (
            relative_change <= config.relative_tolerance
            and projected_gradient_norm <= config.projected_gradient_tolerance
        ):
            stable_count += 1
        else:
            stable_count = 0
        previous, current = current, candidate
        fista_t = next_t
        iteration += 1
        history.append(candidate_value)
        converged = stable_count >= config.convergence_patience
        if iteration % config.checkpoint_every == 0 or converged or iteration == config.max_iterations:
            _save_checkpoint(
                checkpoint_path,
                solver_run_identity=solver_run_identity,
                weights=current.cpu().numpy(),
                previous_weights=previous.cpu().numpy(),
                iteration=iteration,
                fista_t=fista_t,
                lipschitz=lipschitz,
                stable_count=stable_count,
                line_search_steps_total=line_search_steps_total,
                objective_history=history,
            )
        progress.update(
            iteration,
            f"loss={candidate_value:.6g} pg={projected_gradient_norm:.3g}",
            force=converged or iteration == config.max_iterations,
        )
    final_value, final_terms, final_gradient, residual = objective.value_gradient(
        current,
        gradient=True,
        return_residual=True,
    )
    if final_gradient is None or residual is None:
        raise RuntimeError("Motion-basis final diagnostics are incomplete")
    projected = _project_masked_simplex_torch(
        current - final_gradient / max(lipschitz, EPSILON),
        candidate_tensor,
        fallback_tensor,
    )
    projected_gradient_norm = max(lipschitz, EPSILON) * float(
        torch.linalg.vector_norm(projected - current).item()
    ) / math.sqrt(max(int(np.count_nonzero(candidate_mask)), 1))
    return current.cpu().numpy(), {
        "converged": bool(converged),
        "iterations": iteration,
        "stable_iterations": stable_count,
        "initial_objective": float(history[0]),
        "final_objective": final_value,
        "final_terms": final_terms,
        "relative_objective_change": float(
            abs(history[-2] - history[-1]) / max(abs(history[-2]), EPSILON)
            if len(history) > 1
            else 0.0
        ),
        "projected_gradient_norm": projected_gradient_norm,
        "final_lipschitz": float(lipschitz),
        "line_search_steps_total": line_search_steps_total,
        "requested_device": config.device,
        "resolved_device": str(device),
        "objective_history": [float(value) for value in history],
        "sample_residual_rms": residual,
    }


def _completed_phi(
    *,
    points: np.ndarray,
    weights: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    centroid: np.ndarray,
    config: MotionBasisConfig,
) -> np.ndarray:
    """Compose final complex64 displacement without materializing K*G*B*3."""

    mode_count = translation.shape[0]
    output = np.empty((mode_count, len(points), 3), dtype=np.complex64)
    for lower in range(0, len(points), config.sample_chunk_size):
        upper = min(len(points), lower + config.sample_chunk_size)
        centered = (
            np.asarray(points[lower:upper], dtype=np.float64)[:, None, :]
            - np.asarray(centroid, dtype=np.float64)[None, :, :]
        )
        for mode_lower in range(0, mode_count, config.mode_chunk_size):
            mode_upper = min(mode_count, mode_lower + config.mode_chunk_size)
            basis = translation[mode_lower:mode_upper].astype(np.complex128)[
                :, None, :, :
            ] + np.cross(
                rotation[mode_lower:mode_upper].astype(np.complex128)[
                    :, None, :, :
                ],
                centered[None, :, :, :],
                axis=-1,
            )
            output[mode_lower:mode_upper, lower:upper] = np.einsum(
                "kgbd,gb->kgd",
                basis,
                np.asarray(weights[lower:upper], dtype=np.float64),
                optimize=True,
            ).astype(np.complex64)
    if not np.isfinite(output).all():
        raise FloatingPointError("Motion-basis final phi is non-finite")
    return output


def _source_identity_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    """Select path-independent upstream and configuration fields."""

    return {
        "static_scene_identity": source["static_scene_identity"],
        "foreground_identity": source["foreground_identity"],
        "topology_identity": source["topology_identity"],
        "gaussian_measurements_identity": source["gaussian_measurements_identity"],
        "observed_structure_graph_identity": source[
            "observed_structure_graph_identity"
        ],
        "rigid_modes_identity": source["rigid_modes_identity"],
        "modes": source["modes"],
        "views": source["views"],
        "basis_selection": source["basis_selection"],
        "motion_basis": source["motion_basis"],
    }


def _artifact_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select all path-independent scientific v3 artifact fields."""

    return {
        "format": COMPLETED_MODES_FORMAT,
        "version": COMPLETED_MODES_VERSION,
        "completion_method": manifest["completion_method"],
        **_source_identity_payload(manifest),
        "semantics": manifest["semantics"],
        "quality_gate": manifest["quality_gate"],
        "spatial_graph": manifest["spatial_graph"],
        "optimization": manifest["optimization"],
        "diagnostics": manifest["diagnostics"],
        "counts": manifest["counts"],
        "arrays_identity": manifest["arrays_identity"],
    }


def _load_sources(
    *,
    scene_dir: str | Path,
    topology_dir: str | Path,
    measurements_dir: str | Path,
    observed_graph_dir: str | Path,
    rigid_modes_dir: str | Path,
    config: MotionBasisConfig,
) -> tuple[
    dict[str, Any],
    Any,
    Any,
    Any,
    Any,
    Any,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load and cross-check the complete immutable upstream identity chain."""

    scene = load_static_scene(scene_dir, "cpu")
    topology = load_observation_topology(topology_dir)
    measurements = load_gaussian_measurements(measurements_dir)
    observed_graph = load_observed_structure_graph(observed_graph_dir)
    rigid = load_rigid_modes(rigid_modes_dir)
    if scene.manifest is None:
        raise ValueError("Static scene has no manifest")
    scene_identity = scene.manifest["static_scene_identity"]
    foreground_identity = scene.manifest["foreground_identity"]
    topology_identity = topology.manifest["topology_identity"]
    for label, value, expected in (
        ("topology static scene", topology.manifest.get("static_scene_identity"), scene_identity),
        ("rigid static scene", rigid.manifest.get("static_scene_identity"), scene_identity),
        ("topology foreground", topology.manifest.get("foreground_identity"), foreground_identity),
        ("rigid foreground", rigid.manifest.get("foreground_identity"), foreground_identity),
        ("measurement topology", measurements.manifest.get("topology_identity"), topology_identity),
        ("rigid topology", rigid.manifest.get("topology_identity"), topology_identity),
        (
            "observed graph topology",
            observed_graph.manifest.get("topology_identity"),
            topology_identity,
        ),
        (
            "observed graph static scene",
            observed_graph.manifest.get("static_scene_identity"),
            scene_identity,
        ),
        (
            "observed graph foreground",
            observed_graph.manifest.get("foreground_identity"),
            foreground_identity,
        ),
        (
            "rigid observed graph",
            rigid.manifest.get("observed_structure_graph_identity"),
            observed_graph.manifest["observed_structure_graph_identity"],
        ),
        (
            "rigid measurements",
            rigid.manifest.get("gaussian_measurements_identity"),
            measurements.manifest["gaussian_measurements_identity"],
        ),
    ):
        if value != expected:
            raise ValueError(f"Motion-basis {label} identity differs")
    if rigid.manifest.get("modes") != measurements.manifest.get("modes"):
        raise ValueError("Motion-basis rigid and measurement mode order differs")
    rigid_views = rigid.manifest.get("views")
    measurement_views = measurements.manifest.get("views")
    topology_views = topology.manifest.get("views")
    if not (
        isinstance(rigid_views, list)
        and isinstance(measurement_views, list)
        and isinstance(topology_views, list)
        and len(rigid_views) == len(measurement_views) == len(topology_views)
    ):
        raise ValueError("Motion-basis source view counts differ")
    views: list[dict[str, Any]] = []
    for index, (rigid_view, measurement_view, topology_view) in enumerate(
        zip(rigid_views, measurement_views, topology_views)
    ):
        label = topology_view.get("label")
        if any(
            view.get("index") != index or view.get("label") != label
            for view in (rigid_view, measurement_view)
        ):
            raise ValueError("Motion-basis source view order differs")
        for view in (rigid_view, measurement_view):
            if view.get("flow_identity") != topology_view.get("flow_identity"):
                raise ValueError(f"Motion-basis flow identity for {label!r} differs")
        views.append(dict(rigid_view))
    eligible, selected = _select_bases(rigid, config)
    component_indices, translation, rotation, centroid, radius = _basis_arrays(
        rigid, selected
    )
    if config.basis_selection_policy == "trusted_per_mode":
        active = np.asarray(rigid.arrays["component_retained_mask"], dtype=bool)[:, selected]
        translation[:, :-1][~active] = 0.0
        rotation[:, :-1][~active] = 0.0
    points = (
        scene.foreground.raw_tensors(cpu=True)["means"]
        .numpy()
        .astype(np.float32, copy=False)
    )
    if len(points) != int(rigid.manifest["counts"]["foreground_gaussians"]):
        raise ValueError("Motion-basis rigid and static foreground counts differ")
    basis_selection = {
        "eligibility": {"trusted_all_modes": "component_retained_at_every_mode",
                        "all_rigid_components": "all_solved_nontrivial_components_without_trust_filter",
                        "trusted_per_mode": "component_retained_in_current_mode"}[config.basis_selection_policy],
        "selection": "all_eligible_stable_component_index",
        "eligible_component_indices": eligible.astype(int).tolist(),
        "selected_component_indices": selected.astype(int).tolist(),
        "rigid_basis_count": len(selected),
        "zero_basis_index": len(selected),
        "local_rigid_basis_count": config.local_rigid_basis_count,
        "zero_basis_component_index": -1,
        "candidate_distance": "nearest_distance_to_actual_component_member",
        "prior_formula": "exp(-support_distance / distance_temperature)",
        "zero_prior_score": config.zero_prior_score,
    }
    if config.basis_selection_policy == "all_rigid_components":
        retained = np.asarray(rigid.arrays["component_retained_mask"], dtype=bool)
        basis_selection.update(
            policy=config.basis_selection_policy, trust_filter="disabled",
            source_component_count=int(retained.shape[1]),
            source_trusted_all_modes_component_indices=np.flatnonzero(np.all(retained, axis=0)).astype(int).tolist(),
            source_trusted_component_count_per_mode=np.count_nonzero(retained, axis=1).astype(int).tolist(),
        )
    elif config.basis_selection_policy == "trusted_per_mode":
        retained = np.asarray(rigid.arrays["component_retained_mask"], dtype=bool)
        active_counts = np.count_nonzero(retained, axis=1)
        basis_selection.update(
            policy="trusted_per_mode", trust_filter="per_mode_existing_retained_mask",
            selection="union_of_per_mode_trusted_components_stable_component_index",
            count_policy="automatic_trusted_union", source_component_count=int(retained.shape[1]),
            active_component_indices_per_mode=[np.flatnonzero(row).astype(int).tolist() for row in retained],
            active_rigid_basis_counts_per_mode=active_counts.astype(int).tolist(),
            local_rigid_candidate_counts_per_mode=np.minimum(active_counts, config.local_rigid_basis_count).astype(int).tolist(),
            inactive_twist_policy="exact_zero", zero_trusted_mode_policy="error_before_solver",
        )
    source = {
        "static_scene": str(Path(scene_dir).expanduser().resolve()),
        "static_scene_identity": scene_identity,
        "foreground_identity": foreground_identity,
        "topology": str(topology.path),
        "topology_identity": topology_identity,
        "measurements": str(measurements.path),
        "gaussian_measurements_identity": measurements.manifest[
            "gaussian_measurements_identity"
        ],
        "observed_structure_graph": str(observed_graph.path),
        "observed_structure_graph_identity": observed_graph.manifest[
            "observed_structure_graph_identity"
        ],
        "rigid_modes": str(rigid.path),
        "rigid_modes_identity": rigid.manifest["rigid_modes_identity"],
        "modes": [dict(mode) for mode in rigid.manifest["modes"]],
        "views": views,
        "basis_selection": basis_selection,
        "motion_basis": config.to_dict(),
    }
    source["solver_run_identity"] = hashlib.sha256(
        _canonical_json(_source_identity_payload(source))
    ).hexdigest()
    return (
        source,
        scene,
        topology,
        measurements,
        observed_graph,
        rigid,
        points,
        component_indices,
        translation,
        rotation,
        centroid,
        radius,
    )


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    mode_count: int,
    view_count: int,
    point_count: int,
    basis_count: int,
    sample_count: int,
    edge_count: int,
) -> None:
    """Validate the complete v3 array inventory and all role/weight invariants."""

    if set(arrays) != set(ARRAY_DTYPES):
        raise ValueError("Motion-basis v3 array inventory is invalid")
    expected_shapes = {
        "phi": (mode_count, point_count, 3),
        "weights": (point_count, basis_count),
        "spatial_prior_weights": (point_count, basis_count),
        "basis_component_index": (basis_count,),
        "basis_owner_index": (point_count,),
        "basis_translation": (mode_count, basis_count, 3),
        "basis_rotation": (mode_count, basis_count, 3),
        "basis_centroid": (basis_count, 3),
        "basis_radius": (basis_count,),
        "candidate_mask": (point_count, basis_count),
        "measurement_supported_mask": (point_count,),
        "graph_propagated_mask": (point_count,),
        "zero_fallback_mask": (point_count,),
        "measurement_contributor_count": (point_count,),
        "graph_degree": (point_count,),
        "graph_component_index": (point_count,),
        "spatial_edge_index": (edge_count, 2),
        "spatial_edge_distance": (edge_count,),
        "spatial_edge_weight": (edge_count,),
        "weight_entropy": (point_count,),
        "dominant_basis_index": (point_count,),
        "sample_residual_rms": (mode_count, sample_count),
    }
    del view_count  # Views affect the manifest/operator, not persisted array shapes.
    for name, dtype in ARRAY_DTYPES.items():
        value = arrays[name]
        if value.dtype != dtype or value.shape != expected_shapes[name]:
            raise ValueError(
                f"Motion-basis {name} has {value.dtype.name}{value.shape}, "
                f"expected {dtype.name}{expected_shapes[name]}"
            )
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Motion-basis {name} contains NaN or Inf")
    supported = arrays["measurement_supported_mask"]
    propagated = arrays["graph_propagated_mask"]
    fallback = arrays["zero_fallback_mask"]
    if np.any((supported & propagated) | (supported & fallback) | (propagated & fallback)):
        raise ValueError("Motion-basis support-role masks overlap")
    if not np.all(supported | propagated | fallback):
        raise ValueError("Motion-basis support-role masks do not cover foreground")
    weights = arrays["weights"]
    candidate = arrays["candidate_mask"]
    if np.any(weights < 0.0) or not np.allclose(
        np.sum(weights, axis=1), 1.0, rtol=0.0, atol=1.0e-5
    ):
        raise ValueError("Motion-basis weights are outside the simplex")
    if np.any(weights[~candidate] != 0.0):
        raise ValueError("Motion-basis weights use a non-candidate basis")
    if not np.all(candidate[:, -1]):
        raise ValueError("Motion-basis zero basis is not universally available")
    if np.any(fallback):
        expected = np.zeros((int(np.count_nonzero(fallback)), basis_count), dtype=np.float32)
        expected[:, -1] = 1.0
        if not np.array_equal(weights[fallback], expected):
            raise ValueError("Motion-basis fallback points must use exact zero motion")
    if arrays["basis_component_index"][-1] != -1:
        raise ValueError("Motion-basis final slot must be the zero basis")
    if np.any(arrays["basis_owner_index"] < -1) or np.any(
        arrays["basis_owner_index"] >= basis_count - 1
    ):
        raise ValueError("Motion-basis basis owner index is invalid")
    if np.any(arrays["basis_translation"][:, -1] != 0.0) or np.any(
        arrays["basis_rotation"][:, -1] != 0.0
    ):
        raise ValueError("Motion-basis zero twist is not exactly zero")
    if np.any(arrays["measurement_contributor_count"][supported] <= 0) or np.any(
        arrays["measurement_contributor_count"][~supported] != 0
    ):
        raise ValueError("Motion-basis measurement support/counts disagree")
    if np.any(arrays["dominant_basis_index"] < 0) or np.any(
        arrays["dominant_basis_index"] >= basis_count
    ):
        raise ValueError("Motion-basis dominant basis index is invalid")


def load_motion_basis_modes(path: str | Path) -> MotionBasisModesArtifact:
    """Load and strictly validate one completed-mode v3 artifact."""

    root = Path(path).expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    arrays_path = root / COMPLETED_MODES_FILENAME
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"Incomplete motion-basis artifact: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != COMPLETED_MODES_FORMAT or manifest.get("version") != 3:
        raise ValueError("Unsupported motion-basis completed-mode artifact")
    if manifest.get("completion_method") != "shared_motion_basis_blend":
        raise ValueError("Motion-basis completion method is invalid")
    if manifest.get("semantics", {}).get("method") != "joint_pixel_composited_motion_basis_fista":
        raise ValueError("Motion-basis solver semantics are invalid")
    if manifest.get("quality_gate") != {
        "required": True,
        "status": "completion_candidate_unapproved",
    }:
        raise ValueError("Motion-basis artifact must remain an unapproved candidate")
    if manifest.get("arrays_file") != COMPLETED_MODES_FILENAME:
        raise ValueError("Motion-basis array filename is invalid")
    if manifest.get("arrays_file_sha256") != _sha256_file(arrays_path):
        raise ValueError("Motion-basis NPZ SHA-256 differs")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Motion-basis counts are invalid")
    _validate_arrays(
        arrays,
        mode_count=int(counts["modes"]),
        view_count=int(counts["views"]),
        point_count=int(counts["foreground_gaussians"]),
        basis_count=int(counts["bases"]),
        sample_count=int(counts["measurement_samples"]),
        edge_count=int(counts["spatial_edges"]),
    )
    config_payload = manifest.get("motion_basis")
    if not isinstance(config_payload, dict):
        raise ValueError("Motion-basis configuration is invalid")
    settings = _config_from_manifest(config_payload)
    if settings.basis_selection_policy == "trusted_per_mode":
        raise ValueError("Shared completed-mode v3 cannot use trusted_per_mode basis selection")
    (
        expected_source,
        _,
        expected_topology,
        _,
        _,
        expected_rigid,
        points,
        expected_component_indices,
        expected_translation,
        expected_rotation,
        expected_centroid,
        expected_radius,
    ) = _load_sources(
        scene_dir=manifest["static_scene"],
        topology_dir=manifest["topology"],
        measurements_dir=manifest["measurements"],
        observed_graph_dir=manifest["observed_structure_graph"],
        rigid_modes_dir=manifest["rigid_modes"],
        config=settings,
    )
    for name in (
        "static_scene_identity",
        "foreground_identity",
        "topology_identity",
        "gaussian_measurements_identity",
        "observed_structure_graph_identity",
        "rigid_modes_identity",
        "modes",
        "views",
        "basis_selection",
    ):
        if manifest.get(name) != expected_source.get(name):
            raise ValueError(f"Motion-basis source field {name} differs")
    expected_basis_arrays = {
        "basis_component_index": expected_component_indices,
        "basis_translation": expected_translation,
        "basis_rotation": expected_rotation,
        "basis_centroid": expected_centroid,
        "basis_radius": expected_radius,
    }
    for name, expected in expected_basis_arrays.items():
        if not np.array_equal(arrays[name], expected.astype(arrays[name].dtype)):
            raise ValueError(f"Motion-basis {name} differs from rigid source")
    point_component = np.asarray(
        expected_rigid.arrays["point_component_index"], dtype=np.int32
    )
    expected_owner = np.full(len(points), -1, dtype=np.int8)
    for basis, component in enumerate(expected_component_indices[:-1].tolist()):
        expected_owner[point_component == component] = np.int8(basis)
    if not np.array_equal(arrays["basis_owner_index"], expected_owner):
        raise ValueError("Motion-basis owner indices differ from rigid components")
    expected_candidate, expected_prior = _candidate_and_prior_weights(
        points, point_component, expected_component_indices[:-1], settings
    )
    expected_prior[arrays["zero_fallback_mask"]] = 0.0
    expected_prior[arrays["zero_fallback_mask"], -1] = 1.0
    if not np.array_equal(arrays["candidate_mask"], expected_candidate) or not np.allclose(
        arrays["spatial_prior_weights"],
        expected_prior.astype(np.float32),
        rtol=0.0,
        atol=2.0e-7,
    ):
        raise ValueError("Motion-basis candidates/distance prior differ")
    expected_graph = build_foreground_graph(points, settings)
    graph_fields = {
        "spatial_edge_index": expected_graph.edge_index.astype(np.int32),
        "spatial_edge_distance": expected_graph.edge_distance.astype(np.float32),
        "spatial_edge_weight": expected_graph.edge_weight.astype(np.float32),
        "graph_degree": expected_graph.degree,
        "graph_component_index": expected_graph.component_index,
    }
    for name, expected in graph_fields.items():
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"Motion-basis {name} differs from reconstructed graph")
    expected_roles = _role_masks(
        graph=expected_graph,
        topology=expected_topology,
        alpha_identifiable=np.asarray(
            expected_rigid.arrays["alpha_identifiable_mask"], dtype=bool
        ),
        point_count=len(points),
    )
    for name, expected in zip(
        (
            "measurement_supported_mask",
            "graph_propagated_mask",
            "zero_fallback_mask",
            "measurement_contributor_count",
        ),
        expected_roles,
    ):
        if not np.array_equal(arrays[name], expected.astype(arrays[name].dtype)):
            raise ValueError(f"Motion-basis {name} differs from source topology/graph")
    reconstructed_phi = _completed_phi(
        points=points,
        weights=arrays["weights"],
        translation=expected_translation,
        rotation=expected_rotation,
        centroid=expected_centroid,
        config=settings,
    )
    if not np.allclose(
        arrays["phi"], reconstructed_phi, rtol=2.0e-5, atol=2.0e-6
    ):
        raise ValueError("Motion-basis phi differs from weights and rigid bases")
    metadata = {
        name: {"dtype": value.dtype.name, "shape": list(value.shape)}
        for name, value in arrays.items()
    }
    if manifest.get("arrays") != metadata:
        raise ValueError("Motion-basis array metadata differs")
    if manifest.get("arrays_identity") != _arrays_identity(arrays):
        raise ValueError("Motion-basis array identity differs")
    expected_run = hashlib.sha256(
        _canonical_json(_source_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("solver_run_identity") != expected_run:
        raise ValueError("Motion-basis solver run identity differs")
    expected_artifact = hashlib.sha256(
        _canonical_json(_artifact_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("completed_modes_identity") != expected_artifact:
        raise ValueError("Motion-basis artifact identity differs")
    return MotionBasisModesArtifact(root, manifest, arrays)


def _fit_metrics(
    *,
    residual_squared: np.ndarray,
    signal_squared: np.ndarray,
    sample_view: np.ndarray,
    sample_confidence: np.ndarray,
    identifiable: np.ndarray,
    modes: Sequence[int],
    views: Sequence[int],
) -> dict[str, Any]:
    """Aggregate foreground-alpha-weighted complex residual metrics."""

    error = 0.0
    signal = 0.0
    scalar_weight = 0.0
    included_blocks = 0
    for mode in modes:
        for view in views:
            if not identifiable[mode, view]:
                continue
            rows = sample_view == view
            confidence = sample_confidence[rows]
            error += float(np.sum(confidence * residual_squared[mode, rows]))
            signal += float(np.sum(confidence * signal_squared[mode, rows]))
            scalar_weight += 2.0 * float(np.sum(confidence))
            included_blocks += 1
    if included_blocks == 0:
        return {
            "included": False,
            "included_mode_view_blocks": 0,
            "complex_rmse": None,
            "complex_nrmse": None,
            "r2_zero_baseline": None,
        }
    return {
        "included": True,
        "included_mode_view_blocks": included_blocks,
        "complex_rmse": float(math.sqrt(error / max(scalar_weight, EPSILON))),
        "complex_nrmse": float(math.sqrt(error / max(signal, EPSILON))),
        "r2_zero_baseline": float(1.0 - error / max(signal, EPSILON)),
    }


def _seam_diagnostics(
    phi: np.ndarray,
    graph: ForegroundGraph,
    point_component: np.ndarray,
) -> dict[str, Any]:
    """Measure modal discontinuity across old rigid-component graph seams."""

    if len(graph.edge_index):
        source = graph.edge_index[:, 0]
        target = graph.edge_index[:, 1]
        cross = (
            (point_component[source] >= 0)
            & (point_component[target] >= 0)
            & (point_component[source] != point_component[target])
        )
        edges = graph.edge_index[cross]
    else:
        edges = np.empty((0, 2), dtype=np.int64)
    per_mode: list[dict[str, Any]] = []
    all_differences: list[np.ndarray] = []
    for mode in range(phi.shape[0]):
        if len(edges):
            difference = np.linalg.norm(
                phi[mode, edges[:, 0]] - phi[mode, edges[:, 1]], axis=1
            ).astype(np.float64)
            all_differences.append(difference)
            endpoint = 0.5 * (
                np.linalg.norm(phi[mode, edges[:, 0]], axis=1)
                + np.linalg.norm(phi[mode, edges[:, 1]], axis=1)
            ).astype(np.float64)
            relative = difference / np.maximum(endpoint, EPSILON)
            record = {
                "mode_slot": mode,
                "displacement_jump_rms": float(math.sqrt(np.mean(difference**2))),
                "displacement_jump_median": float(np.median(difference)),
                "displacement_jump_p90": float(np.percentile(difference, 90.0)),
                "relative_to_endpoint_magnitude_rms": float(
                    math.sqrt(np.mean(relative**2))
                ),
            }
        else:
            record = {
                "mode_slot": mode,
                "displacement_jump_rms": 0.0,
                "displacement_jump_median": 0.0,
                "displacement_jump_p90": 0.0,
                "relative_to_endpoint_magnitude_rms": 0.0,
            }
        per_mode.append(record)
    if all_differences:
        combined = np.concatenate(all_differences)
        overall = {
            "samples": int(len(combined)),
            "displacement_jump_rms": float(math.sqrt(np.mean(combined**2))),
            "displacement_jump_median": float(np.median(combined)),
            "displacement_jump_p90": float(np.percentile(combined, 90.0)),
        }
    else:
        overall = {
            "samples": 0,
            "displacement_jump_rms": 0.0,
            "displacement_jump_median": 0.0,
            "displacement_jump_p90": 0.0,
        }
    return {
        "cross_component_edge_count": len(edges),
        "overall": overall,
        "per_mode": per_mode,
    }


def _weight_field_diagnostics(
    *,
    weights: np.ndarray,
    entropy: np.ndarray,
    candidate_mask: np.ndarray,
    measurement_supported_mask: np.ndarray,
    point_component: np.ndarray,
    basis_component_index: np.ndarray,
    phi: np.ndarray,
) -> dict[str, Any]:
    """Summarize basis use, zero collapse, and within-component flexibility."""

    values = np.asarray(weights, dtype=np.float64)
    supported = np.asarray(measurement_supported_mask, dtype=bool)
    zero = values[:, -1]
    supported_zero = zero[supported]
    basis_usage: list[dict[str, Any]] = []
    for basis, component in enumerate(basis_component_index.tolist()):
        column = values[:, basis]
        basis_usage.append(
            {
                "basis_index": basis,
                "component_index": int(component),
                "weight_sum": float(np.sum(column)),
                "weight_mean": float(np.mean(column)),
                "weight_median": float(np.median(column)),
                "measurement_supported_weight_mean": (
                    float(np.mean(column[supported])) if np.any(supported) else None
                ),
                "fraction_above_0_01": float(np.mean(column > 0.01)),
                "dominant_gaussians": int(np.count_nonzero(np.argmax(values, axis=1) == basis)),
            }
        )
    own_basis: list[dict[str, Any]] = []
    component_variation: list[dict[str, Any]] = []
    for basis, component in enumerate(basis_component_index[:-1].tolist()):
        members = np.flatnonzero(point_component == component)
        own = values[members, basis]
        own_basis.append(
            {
                "basis_index": basis,
                "component_index": int(component),
                "gaussians": int(len(members)),
                "weight_mean": float(np.mean(own)),
                "weight_median": float(np.median(own)),
                "weight_p10": float(np.percentile(own, 10.0)),
                "weight_p90": float(np.percentile(own, 90.0)),
            }
        )
        member_field = np.asarray(phi[:, members], dtype=np.complex128)
        mean_field = np.mean(member_field, axis=1, keepdims=True)
        deviation = np.linalg.norm(member_field - mean_field, axis=2).reshape(-1)
        magnitude = np.linalg.norm(member_field, axis=2).reshape(-1)
        component_variation.append(
            {
                "basis_index": basis,
                "component_index": int(component),
                "complex_displacement_magnitude_mean": float(np.mean(magnitude)),
                "within_component_deviation_rms": float(
                    math.sqrt(np.mean(np.square(deviation)))
                ),
                "within_component_deviation_median": float(np.median(deviation)),
                "within_component_deviation_p90": float(
                    np.percentile(deviation, 90.0)
                ),
            }
        )
    effective = np.exp(np.asarray(entropy, dtype=np.float64))
    field_magnitude = np.linalg.norm(np.asarray(phi, dtype=np.complex128), axis=2)
    active_count = np.count_nonzero(candidate_mask, axis=1)
    used_nonzero_basis_count = int(
        np.count_nonzero(np.mean(values[:, :-1], axis=0) > 0.01)
    )
    supported_zero_median = (
        float(np.median(supported_zero)) if len(supported_zero) else None
    )
    collapse_warning = bool(
        (supported_zero_median is not None and supported_zero_median > 0.95)
        or used_nonzero_basis_count < 3
    )
    return {
        "basis_usage": basis_usage,
        "measurement_supported_zero_weight": {
            "gaussians": int(np.count_nonzero(supported)),
            "mean": float(np.mean(supported_zero)) if len(supported_zero) else None,
            "median": supported_zero_median,
            "p90": float(np.percentile(supported_zero, 90.0))
            if len(supported_zero)
            else None,
        },
        "effective_basis_count": {
            "mean": float(np.mean(effective)),
            "median": float(np.median(effective)),
            "p90": float(np.percentile(effective, 90.0)),
        },
        "candidate_basis_count": {
            "minimum": int(np.min(active_count)),
            "maximum": int(np.max(active_count)),
        },
        "selected_component_own_basis_weight": own_basis,
        "selected_component_internal_motion": component_variation,
        "field_magnitude": {
            "maximum": float(np.max(field_magnitude)),
            "p99": float(np.percentile(field_magnitude, 99.0)),
            "median": float(np.median(field_magnitude)),
        },
        "nonzero_bases_with_mean_weight_above_0_01": used_nonzero_basis_count,
        "basis_collapse_warning": collapse_warning,
    }


def build_motion_basis_modes_artifact(
    *,
    scene_dir: str | Path,
    topology_dir: str | Path,
    measurements_dir: str | Path,
    observed_graph_dir: str | Path,
    rigid_modes_dir: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
    config: MotionBasisConfig | None = None,
    resume: bool = False,
    command: Sequence[str] = (),
) -> MotionBasisModesArtifact:
    """Fit shared basis weights, resume safely, and atomically publish v3 modes."""

    settings = config or MotionBasisConfig()
    settings.validate()
    if settings.basis_selection_policy == "trusted_per_mode":
        raise ValueError("trusted_per_mode basis selection requires per-frequency weights and completed-mode v6")
    if not isinstance(resume, bool):
        raise TypeError("Motion-basis resume must be a boolean")
    device = _resolve_device(settings.device)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Motion-basis output already exists: {destination}")
    (
        source,
        _,
        topology,
        measurements,
        _,
        rigid,
        points,
        component_indices,
        translation,
        rotation,
        centroid,
        radius,
    ) = _load_sources(
        scene_dir=scene_dir,
        topology_dir=topology_dir,
        measurements_dir=measurements_dir,
        observed_graph_dir=observed_graph_dir,
        rigid_modes_dir=rigid_modes_dir,
        config=settings,
    )
    work = Path(work_dir).expanduser().resolve()
    source["motion_basis"] = {
        **source["motion_basis"],
        "resolved_device": str(device),
    }
    source["solver_run_identity"] = hashlib.sha256(
        _canonical_json(_source_identity_payload(source))
    ).hexdigest()
    _prepare_work_dir(work, source, resume=resume)
    report_progress("motion-basis fit: building full-foreground graph")
    graph = build_foreground_graph(points, settings)
    point_component = np.asarray(
        rigid.arrays["point_component_index"], dtype=np.int32
    )
    basis_owner = np.full(len(points), -1, dtype=np.int8)
    for basis, component in enumerate(component_indices[:-1].tolist()):
        basis_owner[point_component == component] = np.int8(basis)
    candidate, prior = _candidate_and_prior_weights(
        points,
        point_component,
        component_indices[:-1],
        settings,
    )
    identifiable = np.asarray(rigid.arrays["alpha_identifiable_mask"], dtype=bool)
    supported, propagated, zero_fallback, contributor_count = _role_masks(
        graph=graph,
        topology=topology,
        alpha_identifiable=identifiable,
        point_count=len(points),
    )
    prior[zero_fallback] = 0.0
    prior[zero_fallback, -1] = 1.0
    design = _build_design_file(
        path=work / DESIGN_FILENAME,
        points=points,
        topology=topology,
        alphas=np.asarray(rigid.arrays["alphas"], dtype=np.complex128),
        basis_translation=translation,
        basis_rotation=rotation,
        basis_centroid=centroid,
        candidate_mask=candidate,
        config=settings,
    )
    offsets = np.asarray(topology.arrays.sample_offsets, dtype=np.int64)
    contributor_sample = np.repeat(
        np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets)
    )
    contributor_point = np.asarray(
        topology.arrays.contributor_gaussian_index, dtype=np.int64
    )
    sample_view = np.asarray(topology.arrays.sample_view_index, dtype=np.int64)
    sample_loss_weight, block_rms, energy_floor = _sample_loss_weights(
        measurements.measurements,
        sample_view,
        np.asarray(topology.arrays.sample_foreground_alpha, dtype=np.float32),
        identifiable,
        settings,
    )
    operator = PixelBasisOperator(
        design=design,
        measurements=measurements.measurements,
        contributor_sample_index=contributor_sample,
        contributor_point_index=contributor_point,
        sample_loss_weight=sample_loss_weight,
        mode_chunk_size=settings.mode_chunk_size,
    )
    solved_weights, optimization = _solve_weights(
        operator=operator,
        graph=graph,
        candidate_mask=candidate,
        prior=prior,
        measurement_supported_mask=supported,
        zero_fallback_mask=zero_fallback,
        work=work,
        solver_run_identity=source["solver_run_identity"],
        config=settings,
        device=device,
        resume=resume,
    )
    persisted_weights = solved_weights.astype(np.float32)
    persisted_weights[~candidate] = 0.0
    persisted_weights[zero_fallback] = 0.0
    persisted_weights[zero_fallback, -1] = 1.0
    phi = _completed_phi(
        points=points,
        weights=persisted_weights,
        translation=translation,
        rotation=rotation,
        centroid=centroid,
        config=settings,
    )
    entropy = -np.sum(
        persisted_weights
        * np.log(np.maximum(persisted_weights, np.finfo(np.float32).tiny)),
        axis=1,
    ).astype(np.float32)
    dominant = np.argmax(persisted_weights, axis=1).astype(np.int16)
    arrays = {
        "phi": phi,
        "weights": persisted_weights,
        "spatial_prior_weights": prior.astype(np.float32),
        "basis_component_index": component_indices.astype(np.int32),
        "basis_owner_index": basis_owner,
        "basis_translation": translation.astype(np.complex64),
        "basis_rotation": rotation.astype(np.complex64),
        "basis_centroid": centroid.astype(np.float32),
        "basis_radius": radius.astype(np.float32),
        "candidate_mask": candidate.astype(bool),
        "measurement_supported_mask": supported.astype(bool),
        "graph_propagated_mask": propagated.astype(bool),
        "zero_fallback_mask": zero_fallback.astype(bool),
        "measurement_contributor_count": contributor_count.astype(np.int32),
        "graph_degree": graph.degree.astype(np.int32),
        "graph_component_index": graph.component_index.astype(np.int32),
        "spatial_edge_index": graph.edge_index.astype(np.int32),
        "spatial_edge_distance": graph.edge_distance.astype(np.float32),
        "spatial_edge_weight": graph.edge_weight.astype(np.float32),
        "weight_entropy": entropy,
        "dominant_basis_index": dominant,
        "sample_residual_rms": optimization.pop("sample_residual_rms").astype(np.float32),
    }
    mode_count = len(source["modes"])
    view_count = len(source["views"])
    basis_count = len(component_indices)
    sample_count = measurements.measurements.shape[1]
    _validate_arrays(
        arrays,
        mode_count=mode_count,
        view_count=view_count,
        point_count=len(points),
        basis_count=basis_count,
        sample_count=sample_count,
        edge_count=len(graph.edge_index),
    )
    residual = arrays["sample_residual_rms"].astype(np.float64)
    measurement_values = np.asarray(measurements.measurements, dtype=np.complex128)
    residual_squared = 2.0 * np.square(residual)
    signal_squared = np.sum(np.abs(measurement_values) ** 2, axis=2)
    sample_confidence = np.asarray(
        topology.arrays.sample_foreground_alpha, dtype=np.float64
    )
    per_mode_view: list[dict[str, Any]] = []
    for mode in range(mode_count):
        for view in range(view_count):
            per_mode_view.append({
                "mode_slot": mode,
                "view_index": view,
                "view_label": source["views"][view]["label"],
                **_fit_metrics(
                    residual_squared=residual_squared,
                    signal_squared=signal_squared,
                    sample_view=sample_view,
                    sample_confidence=sample_confidence,
                    identifiable=identifiable,
                    modes=[mode],
                    views=[view],
                ),
            })
    overall_fit = _fit_metrics(
        residual_squared=residual_squared,
        signal_squared=signal_squared,
        sample_view=sample_view,
        sample_confidence=sample_confidence,
        identifiable=identifiable,
        modes=list(range(mode_count)),
        views=list(range(view_count)),
    )
    per_view_fit = [
        {
            "view_index": view,
            "view_label": source["views"][view]["label"],
            **_fit_metrics(
                residual_squared=residual_squared,
                signal_squared=signal_squared,
                sample_view=sample_view,
                sample_confidence=sample_confidence,
                identifiable=identifiable,
                modes=list(range(mode_count)),
                views=[view],
            ),
        }
        for view in range(view_count)
    ]
    per_mode_fit = [
        {
            "mode_slot": mode,
            **_fit_metrics(
                residual_squared=residual_squared,
                signal_squared=signal_squared,
                sample_view=sample_view,
                sample_confidence=sample_confidence,
                identifiable=identifiable,
                modes=[mode],
                views=list(range(view_count)),
            ),
        }
        for mode in range(mode_count)
    ]
    seam = _seam_diagnostics(phi, graph, point_component)
    weight_field = _weight_field_diagnostics(
        weights=persisted_weights,
        entropy=entropy,
        candidate_mask=candidate,
        measurement_supported_mask=supported,
        point_component=point_component,
        basis_component_index=component_indices,
        phi=phi,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        arrays_path = temporary / COMPLETED_MODES_FILENAME
        save_named_arrays(arrays_path, arrays)
        counts = {
            "modes": mode_count,
            "views": view_count,
            "foreground_gaussians": len(points),
            "rigid_components": int(rigid.manifest["counts"]["rigid_components"]),
            "eligible_rigid_components": len(
                source["basis_selection"]["eligible_component_indices"]
            ),
            "rigid_bases": settings.rigid_basis_count,
            "bases": basis_count,
            "measurement_samples": sample_count,
            "topology_contributors": len(contributor_point),
            "spatial_edges": len(graph.edge_index),
            "fill_graph_components": len(graph.component_size),
            "measurement_supported_gaussians": int(np.count_nonzero(supported)),
            "graph_propagated_gaussians": int(np.count_nonzero(propagated)),
            "zero_fallback_gaussians": int(np.count_nonzero(zero_fallback)),
        }
        manifest = {
            "format": COMPLETED_MODES_FORMAT,
            "version": COMPLETED_MODES_VERSION,
            "completion_method": "shared_motion_basis_blend",
            "producer": {
                "project_version": __version__,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": list(command),
            },
            **source,
            "work_dir": str(work),
            "mode_selection": {
                "policy": "all_input_modes",
                "source_mode_count": mode_count,
                "output_mode_count": mode_count,
                "source_mode_slots": list(range(mode_count)),
            },
            "semantics": {
                "method": "joint_pixel_composited_motion_basis_fista",
                "field": "complex_3d_displacement_in_normalized_scene_coordinates",
                "playback": "real(phi * exp(i*phase))",
                "weights": "real_nonnegative_simplex_shared_across_modes",
                "pixel_prediction": "alpha_kv_times_sum_contributor_weight_J_phi",
                "pixel_confidence": "sample_foreground_alpha_in_loss_only",
                "zero_basis": "last_basis_exactly_zero",
                "background_gaussians": "excluded",
                "support_roles": [
                    "measurement_supported",
                    "graph_propagated",
                    "zero_fallback",
                ],
            },
            "quality_gate": {
                "required": True,
                "status": "completion_candidate_unapproved",
            },
            "spatial_graph": {
                "policy": "distance_pruned_union_knn_without_rgb_depth_boundaries",
                "neighbors": settings.graph_neighbors,
                "max_distance": settings.graph_max_distance,
                "kernel": "gaussian",
                "kernel_formula": "exp(-(distance / max_distance)^2)",
                "candidate_directed_count": graph.candidate_directed_count,
                "retained_directed_count": graph.retained_directed_count,
            },
            "optimization": {
                **optimization,
                "solver": "monotone_masked_simplex_projected_fista_with_backtracking",
                "mode_view_measurement_rms": block_rms.tolist(),
                "measurement_rms_floor": energy_floor,
            },
            "diagnostics": {
                "overall_fit": overall_fit,
                "per_view_fit": per_view_fit,
                "per_mode_fit": per_mode_fit,
                "mode_view_fit": per_mode_view,
                "old_component_seams": seam,
                "weight_field": weight_field,
                "zero_basis_weight_mean": float(np.mean(persisted_weights[:, -1])),
                "zero_basis_weight_p50": float(np.percentile(persisted_weights[:, -1], 50.0)),
                "zero_basis_weight_p90": float(np.percentile(persisted_weights[:, -1], 90.0)),
                "weight_entropy_mean": float(np.mean(entropy)),
                "weight_entropy_p90": float(np.percentile(entropy, 90.0)),
                "graph_weight_difference_rms": float(
                    math.sqrt(
                        np.average(
                            np.sum(
                                np.square(
                                    persisted_weights[graph.edge_index[:, 0]]
                                    - persisted_weights[graph.edge_index[:, 1]]
                                ),
                                axis=1,
                            ),
                            weights=graph.edge_weight,
                        )
                    )
                    if len(graph.edge_index)
                    else 0.0
                ),
            },
            "counts": counts,
            "arrays_file": COMPLETED_MODES_FILENAME,
            "arrays": {
                name: {"dtype": value.dtype.name, "shape": list(value.shape)}
                for name, value in arrays.items()
            },
            "arrays_identity": _arrays_identity(arrays),
            "arrays_file_sha256": _sha256_file(arrays_path),
        }
        manifest["completed_modes_identity"] = hashlib.sha256(
            _canonical_json(_artifact_identity_payload(manifest))
        ).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_motion_basis_modes(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Motion-basis output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_motion_basis_modes(destination)


__all__ = [
    "MotionBasisConfig",
    "MotionBasisModesArtifact",
    "build_foreground_graph",
    "build_motion_basis_modes_artifact",
    "load_motion_basis_modes",
]
