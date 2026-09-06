"""Independent per-frequency weights over the existing fixed rigid bases.

The contributor operator, candidate bases, graph and FISTA implementation are
shared with the v3 experiment. Only the weight field has a new frequency axis.
Global mode/view loss normalization is computed once: each independent solve
uses K times its slice, so the mean objective is data + mean regularization.
Artifacts use version 4 for a shared basis inventory, or version 6 for the
union of independently trusted per-mode bases with per-mode active candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from modal_gaussians import __version__
from modal_gaussians.motion.rigid import motion_basis as shared
from modal_gaussians.numpy_io import save_named_arrays
from modal_gaussians.progress import report_progress


MotionBasisConfig = shared.MotionBasisConfig
COMPLETED_MODES_FORMAT = shared.COMPLETED_MODES_FORMAT
COMPLETED_MODES_FILENAME = shared.COMPLETED_MODES_FILENAME
COMPLETED_MODES_VERSION = 4
PER_MODE_COMPLETED_MODES_VERSION = 6
COMPLETION_METHOD = "per_frequency_motion_basis_blend"
SOLVER_METHOD = "per_frequency_pixel_composited_motion_basis_fista"
WEIGHT_SHARING = "independent_per_frequency"
OBJECTIVE_REDUCTION = "mean_mode_objectives_with_global_block_normalization"
ARRAY_DTYPES = shared.ARRAY_DTYPES
PER_MODE_ARRAY_DTYPES = {**ARRAY_DTYPES, "basis_active_mask": np.dtype(bool)}

# These arrays gain a leading K dimension relative to the v3 schema.
FREQUENCY_FIELDS = frozenset({
    "weights", "spatial_prior_weights", "measurement_supported_mask",
    "graph_propagated_mask", "zero_fallback_mask", "measurement_contributor_count",
    "weight_entropy", "dominant_basis_index",
})
EXISTING_MODE_FIELDS = frozenset({
    "phi", "basis_translation", "basis_rotation", "sample_residual_rms",
})
ROLE_FIELDS = (
    "measurement_supported_mask", "graph_propagated_mask", "zero_fallback_mask",
    "measurement_contributor_count",
)


@dataclass(frozen=True)
class FrequencyMotionBasisModesArtifact:
    """One source-validated completed-mode v4 or v6 artifact."""

    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _source_identity_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    """Extend the shared source identity with the scientific weight semantics."""

    return {
        **shared._source_identity_payload(source),
        "weight_sharing": source["weight_sharing"],
        "objective_reduction": source["objective_reduction"],
    }


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(shared._canonical_json(payload)).hexdigest()


def _artifact_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Hash every scientific v4 field independently of directory paths."""

    return {
        "format": COMPLETED_MODES_FORMAT,
        "version": manifest["version"],
        "completion_method": manifest["completion_method"],
        **_source_identity_payload(manifest),
        **{name: manifest[name] for name in (
            "semantics", "quality_gate", "spatial_graph", "optimization",
            "diagnostics", "counts", "arrays_identity", "mode_selection",
        )},
    }


def _mode_arrays(arrays: Mapping[str, np.ndarray], mode: int) -> dict[str, np.ndarray]:
    """Expose one frequency using the existing v3 validation shapes."""

    return {
        name: value[mode] if name in FREQUENCY_FIELDS or (name == "candidate_mask" and value.ndim == 3) else (
            value[mode:mode + 1] if name in EXISTING_MODE_FIELDS else value
        )
        for name, value in arrays.items() if name != "basis_active_mask"
    }


def _validate_arrays(
    arrays: Mapping[str, np.ndarray], *, mode_count: int, view_count: int,
    point_count: int, basis_count: int, sample_count: int, edge_count: int,
    version: int = COMPLETED_MODES_VERSION,
) -> None:
    """Check the extra mode axes, then all existing simplex and role invariants."""

    if version not in (COMPLETED_MODES_VERSION, PER_MODE_COMPLETED_MODES_VERSION):
        raise ValueError("Unsupported per-frequency motion-basis array version")
    dtype_map = PER_MODE_ARRAY_DTYPES if version == PER_MODE_COMPLETED_MODES_VERSION else ARRAY_DTYPES
    if mode_count <= 0 or set(arrays) != set(dtype_map):
        raise ValueError(f"Per-frequency motion-basis v{version} array inventory is invalid")
    if version == PER_MODE_COMPLETED_MODES_VERSION:
        active = arrays["basis_active_mask"]
        candidate = arrays["candidate_mask"]
        if (active.dtype != np.bool_ or candidate.dtype != np.bool_ or active.shape != (mode_count, basis_count)
                or candidate.shape != (mode_count, point_count, basis_count)
                or not np.all(active[:, -1]) or not np.all(np.any(active[:, :-1], axis=1))):
            raise ValueError("Per-mode active-basis or candidate mask is invalid")
        if np.any(candidate & ~active[:, None, :]):
            raise ValueError("Per-mode candidates include an inactive basis")
        for name in ("basis_translation", "basis_rotation"):
            if arrays[name].shape != (mode_count, basis_count, 3) or np.any(arrays[name][~active] != 0):
                raise ValueError(f"Per-mode inactive {name} must be exactly zero")
        for name in ("weights", "spatial_prior_weights"):
            if arrays[name].shape != candidate.shape or np.any(arrays[name][~candidate] != 0):
                raise ValueError(f"Per-mode {name} uses inactive or noncandidate bases")
    elif arrays["candidate_mask"].ndim != 2:
        raise ValueError("Per-frequency v4 must retain its shared candidate mask")
    for name in FREQUENCY_FIELDS | EXISTING_MODE_FIELDS:
        if arrays[name].ndim == 0 or arrays[name].shape[0] != mode_count:
            raise ValueError(f"Per-frequency motion-basis {name} mode axis differs")
    for mode in range(mode_count):
        shared._validate_arrays(
            _mode_arrays(arrays, mode), mode_count=1, view_count=view_count,
            point_count=point_count, basis_count=basis_count,
            sample_count=sample_count, edge_count=edge_count,
        )
    expected_entropy = -np.sum(
        arrays["weights"] * np.log(np.maximum(
            arrays["weights"], np.finfo(np.float32).tiny,
        )), axis=2,
    ).astype(np.float32)
    if not np.array_equal(arrays["dominant_basis_index"], np.argmax(arrays["weights"], axis=2)):
        raise ValueError("Per-frequency dominant basis indices differ from weights")
    if not np.allclose(arrays["weight_entropy"], expected_entropy, rtol=0.0, atol=2.0e-7):
        raise ValueError("Per-frequency weight entropy differs from weights")


def _frequency_roles(
    graph: shared.ForegroundGraph, topology: Any, identifiable: np.ndarray,
    point_count: int,
) -> dict[str, np.ndarray]:
    """Keep visibility and unsupported-component fallback separate per mode."""

    per_mode = [shared._role_masks(
        graph=graph, topology=topology,
        alpha_identifiable=identifiable[mode:mode + 1], point_count=point_count,
    ) for mode in range(len(identifiable))]
    return {
        name: np.stack([record[index] for record in per_mode]).astype(ARRAY_DTYPES[name])
        for index, name in enumerate(ROLE_FIELDS)
    }


def _frequency_prior(prior: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Use the same spatial prior except for each mode's forced zero components."""

    result = (np.broadcast_to(prior, (len(fallback), *prior.shape)).copy()
              if prior.ndim == 2 else prior.copy())
    result[fallback] = 0.0
    result[:, :, -1][fallback] = 1.0
    return result


def _completed_phi(
    *, points: np.ndarray, weights: np.ndarray, translation: np.ndarray,
    rotation: np.ndarray, centroid: np.ndarray, config: MotionBasisConfig,
) -> np.ndarray:
    """Compose each frequency with its own Gaussian weight vectors."""

    return np.concatenate([shared._completed_phi(
        points=points, weights=weights[mode], translation=translation[mode:mode + 1],
        rotation=rotation[mode:mode + 1], centroid=centroid, config=config,
    ) for mode in range(len(translation))], axis=0)


def _semantics(version: int = COMPLETED_MODES_VERSION) -> dict[str, Any]:
    result = {
        "method": SOLVER_METHOD,
        "field": "complex_3d_displacement_in_normalized_scene_coordinates",
        "playback": "real(phi * exp(i*phase))",
        "weights": "real_nonnegative_simplex_independent_per_mode",
        "weight_shape": "K_G_B",
        "pixel_prediction": "alpha_kv_times_sum_contributor_weight_J_phi",
        "pixel_confidence": "sample_foreground_alpha_in_loss_only",
        "zero_basis": "last_basis_exactly_zero",
        "background_gaussians": "excluded",
        "support_roles": ["measurement_supported", "graph_propagated", "zero_fallback"],
        "support_role_shape": "K_G",
        "objective_reduction": OBJECTIVE_REDUCTION,
        "sample_loss_weight": "global_weights_slice_times_K_per_independent_solve",
        "regularization": "mean_over_modes_of_same_graph_and_prior_penalties",
    }
    if version == PER_MODE_COMPLETED_MODES_VERSION:
        result.update(
            basis_selection="existing_rigid_retained_mask_independently_per_mode",
            basis_inventory="sorted_union_of_per_mode_trusted_components_plus_zero",
            basis_active_shape="K_B", candidate_shape="K_G_B",
            inactive_basis="exact_zero_twists_candidates_priors_and_weights",
            basis_owner="static_union_membership_forced_into_candidates_only_when_active",
            local_candidates="min_requested_local_count_and_active_rigid_count_plus_zero",
            zero_trusted_mode="error_before_solver",
        )
    elif version != COMPLETED_MODES_VERSION:
        raise ValueError("Unsupported per-frequency semantics version")
    return result


def _mode_selection(mode_count: int) -> dict[str, Any]:
    return {
        "policy": "all_input_modes", "source_mode_count": mode_count,
        "output_mode_count": mode_count, "source_mode_slots": list(range(mode_count)),
    }


def _counts(
    *, source: Mapping[str, Any], rigid: Any, point_count: int, basis_count: int,
    sample_count: int, contributor_count: int, graph: shared.ForegroundGraph,
    roles: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Name union counts explicitly; retain disjoint per-mode role counts."""

    mode_count = len(source["modes"])
    counts = {
        "modes": mode_count, "views": len(source["views"]), "foreground_gaussians": point_count,
        "rigid_components": int(rigid.manifest["counts"]["rigid_components"]),
        "eligible_rigid_components": len(source["basis_selection"]["eligible_component_indices"]),
        "rigid_bases": basis_count - 1, "bases": basis_count,
        "measurement_samples": sample_count, "topology_contributors": contributor_count,
        "spatial_edges": len(graph.edge_index), "fill_graph_components": len(graph.component_size),
        "role_count_unit": "union_over_modes",
        **{name.removesuffix("_mask") + "_gaussians": int(np.count_nonzero(np.any(roles[name], axis=0)))
           for name in ROLE_FIELDS[:3]},
        **{name.removesuffix("_mask") + "_mode_gaussian_pairs": int(np.count_nonzero(roles[name]))
           for name in ROLE_FIELDS[:3]},
        "per_mode": [{"mode_slot": mode,
                      **{name.removesuffix("_mask") + "_gaussians": int(np.count_nonzero(roles[name][mode]))
                         for name in ROLE_FIELDS[:3]}} for mode in range(mode_count)],
    }
    if source["motion_basis"].get("basis_selection_policy") == "trusted_per_mode":
        active_counts = source["basis_selection"]["active_rigid_basis_counts_per_mode"]
        counts["basis_count_unit"] = "trusted_union_inventory"
        counts["active_rigid_bases_per_mode"] = list(active_counts)
        counts["active_bases_per_mode"] = [value + 1 for value in active_counts]
        for mode, record in enumerate(counts["per_mode"]):
            record["active_rigid_bases"] = active_counts[mode]
            record["active_bases"] = active_counts[mode] + 1
            record["local_rigid_candidates"] = min(source["motion_basis"]["local_rigid_basis_count"], active_counts[mode])
    return counts


def _active_bases(rigid: Any, component_indices: np.ndarray) -> np.ndarray:
    """Source trust mask projected onto the stable union bank plus its zero slot."""

    retained = np.asarray(rigid.arrays["component_retained_mask"], dtype=bool)
    return np.concatenate((retained[:, component_indices[:-1]], np.ones((len(retained), 1), dtype=bool)), axis=1)


def load_frequency_motion_basis_modes(path: str | Path) -> FrequencyMotionBasisModesArtifact:
    """Load v4/v6 arrays and reconstruct source-dependent bases, graph and roles."""

    root = Path(path).expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    arrays_path = root / COMPLETED_MODES_FILENAME
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"Incomplete per-frequency motion-basis artifact: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("version")
    if (manifest.get("format") != COMPLETED_MODES_FORMAT
            or type(version) is not int or version not in (COMPLETED_MODES_VERSION, PER_MODE_COMPLETED_MODES_VERSION)
            or manifest.get("completion_method") != COMPLETION_METHOD):
        raise ValueError("Unsupported per-frequency motion-basis completed-mode artifact")
    if (manifest.get("weight_sharing") != WEIGHT_SHARING
            or manifest.get("objective_reduction") != OBJECTIVE_REDUCTION
            or manifest.get("semantics") != _semantics(version)):
        raise ValueError("Per-frequency motion-basis solver semantics differ")
    if manifest.get("quality_gate") != {
        "required": True, "status": "completion_candidate_unapproved",
    }:
        raise ValueError("Per-frequency motion-basis must remain an unapproved candidate")
    if (manifest.get("arrays_file") != COMPLETED_MODES_FILENAME
            or manifest.get("arrays_file_sha256") != shared._sha256_file(arrays_path)):
        raise ValueError("Per-frequency motion-basis array filename or SHA-256 differs")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Per-frequency motion-basis counts are invalid")
    _validate_arrays(
        arrays, mode_count=int(counts["modes"]), view_count=int(counts["views"]),
        point_count=int(counts["foreground_gaussians"]), basis_count=int(counts["bases"]),
        sample_count=int(counts["measurement_samples"]), edge_count=int(counts["spatial_edges"]), version=version,
    )
    payload = manifest.get("motion_basis")
    if not isinstance(payload, dict):
        raise ValueError("Per-frequency motion-basis configuration is invalid")
    settings = shared._config_from_manifest(payload)
    if (version == PER_MODE_COMPLETED_MODES_VERSION) != (settings.basis_selection_policy == "trusted_per_mode"):
        raise ValueError("Per-frequency artifact version and basis selection policy disagree")
    (expected_source, _, topology, measurements, _, rigid, points, component_indices,
     translation, rotation, centroid, radius) = shared._load_sources(
        scene_dir=manifest["static_scene"], topology_dir=manifest["topology"],
        measurements_dir=manifest["measurements"],
        observed_graph_dir=manifest["observed_structure_graph"],
        rigid_modes_dir=manifest["rigid_modes"], config=settings,
    )
    for name in (
        "static_scene_identity", "foreground_identity", "topology_identity",
        "gaussian_measurements_identity", "observed_structure_graph_identity",
        "rigid_modes_identity", "modes", "views", "basis_selection",
    ):
        if manifest.get(name) != expected_source.get(name):
            raise ValueError(f"Per-frequency motion-basis source field {name} differs")
    if (counts["modes"] != len(expected_source["modes"])
            or counts["views"] != len(expected_source["views"])
            or counts["foreground_gaussians"] != len(points)
            or counts["bases"] != len(component_indices)
            or counts["measurement_samples"] != measurements.measurements.shape[1]):
        raise ValueError("Per-frequency motion-basis counts differ from sources")
    for name, expected in {
        "basis_component_index": component_indices, "basis_translation": translation,
        "basis_rotation": rotation, "basis_centroid": centroid, "basis_radius": radius,
    }.items():
        if not np.array_equal(arrays[name], expected.astype(ARRAY_DTYPES[name])):
            raise ValueError(f"Per-frequency motion-basis {name} differs from rigid source")
    active = _active_bases(rigid, component_indices) if version == PER_MODE_COMPLETED_MODES_VERSION else None
    if active is not None and not np.array_equal(arrays["basis_active_mask"], active):
        raise ValueError("Per-mode active basis mask differs from source trust decisions")
    point_component = np.asarray(rigid.arrays["point_component_index"], dtype=np.int32)
    expected_owner = np.full(len(points), -1, dtype=np.int8)
    for basis, component in enumerate(component_indices[:-1].tolist()):
        expected_owner[point_component == component] = np.int8(basis)
    if not np.array_equal(arrays["basis_owner_index"], expected_owner):
        raise ValueError("Per-frequency motion-basis owner indices differ")
    graph = shared.build_foreground_graph(points, settings)
    for name, expected in {
        "spatial_edge_index": graph.edge_index, "spatial_edge_distance": graph.edge_distance,
        "spatial_edge_weight": graph.edge_weight, "graph_degree": graph.degree,
        "graph_component_index": graph.component_index,
    }.items():
        if not np.array_equal(arrays[name], expected.astype(ARRAY_DTYPES[name])):
            raise ValueError(f"Per-frequency motion-basis {name} differs from reconstructed graph")
    roles = _frequency_roles(
        graph, topology, np.asarray(rigid.arrays["alpha_identifiable_mask"], dtype=bool), len(points),
    )
    for name, expected in roles.items():
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"Per-frequency motion-basis {name} differs from source topology/graph")
    if counts != _counts(
        source=expected_source, rigid=rigid, point_count=len(points), basis_count=len(component_indices),
        sample_count=measurements.measurements.shape[1],
        contributor_count=len(topology.arrays.contributor_gaussian_index), graph=graph, roles=roles,
    ):
        raise ValueError("Per-frequency motion-basis counts differ from reconstructed sources/roles")
    if manifest.get("mode_selection") != _mode_selection(len(expected_source["modes"])):
        raise ValueError("Per-frequency motion-basis must retain all source modes in order")
    candidate, prior = shared._candidate_and_prior_weights(
        points, point_component, component_indices[:-1], settings, basis_active_mask=active,
    )
    expected_prior = _frequency_prior(prior, roles["zero_fallback_mask"]).astype(np.float32)
    if (not np.array_equal(arrays["candidate_mask"], candidate)
            or not np.allclose(arrays["spatial_prior_weights"], expected_prior, rtol=0.0, atol=2.0e-7)):
        raise ValueError("Per-frequency motion-basis candidates/distance prior differ")
    reconstructed_phi = _completed_phi(
        points=points, weights=arrays["weights"], translation=translation,
        rotation=rotation, centroid=centroid, config=settings,
    )
    if not np.allclose(arrays["phi"], reconstructed_phi, rtol=2.0e-5, atol=2.0e-6):
        raise ValueError("Per-frequency motion-basis phi differs from weights and rigid bases")
    metadata = {name: {"dtype": value.dtype.name, "shape": list(value.shape)}
                for name, value in arrays.items()}
    if manifest.get("arrays") != metadata or manifest.get("arrays_identity") != shared._arrays_identity(arrays):
        raise ValueError("Per-frequency motion-basis array metadata/identity differs")
    if manifest.get("solver_run_identity") != _identity(_source_identity_payload(manifest)):
        raise ValueError("Per-frequency motion-basis solver run identity differs")
    if manifest.get("completed_modes_identity") != _identity(_artifact_identity_payload(manifest)):
        raise ValueError("Per-frequency motion-basis artifact identity differs")
    return FrequencyMotionBasisModesArtifact(root, manifest, arrays)


def _optimization_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report mean objectives but preserve every independent convergence state."""

    return {
        "converged": all(record["converged"] for record in records),
        "converged_modes": sum(record["converged"] for record in records),
        "mode_count": len(records),
        "iterations": max(record["iterations"] for record in records),
        "iterations_total": sum(record["iterations"] for record in records),
        "initial_objective": float(np.mean([record["initial_objective"] for record in records])),
        "final_objective": float(np.mean([record["final_objective"] for record in records])),
        "final_terms": {name: float(np.mean([record["final_terms"][name] for record in records]))
                        for name in records[0]["final_terms"]},
        "projected_gradient_norm": max(record["projected_gradient_norm"] for record in records),
        "relative_objective_change": max(record["relative_objective_change"] for record in records),
        "line_search_steps_total": sum(record["line_search_steps_total"] for record in records),
        "requested_device": records[0]["requested_device"],
        "resolved_device": records[0]["resolved_device"],
        "objective_reduction": OBJECTIVE_REDUCTION,
        "per_mode": records,
    }


def _weight_diagnostics(
    arrays: Mapping[str, np.ndarray], point_component: np.ndarray,
) -> dict[str, Any]:
    """Keep mode diagnostics and an aggregate over all (mode, Gaussian) pairs."""

    weights = arrays["weights"]
    mode_count, point_count, basis_count = weights.shape
    common = {"basis_component_index": arrays["basis_component_index"]}
    candidates = arrays["candidate_mask"]
    per_mode = [{
        "mode_slot": mode,
        **shared._weight_field_diagnostics(
            weights=weights[mode], entropy=arrays["weight_entropy"][mode],
            candidate_mask=candidates[mode] if candidates.ndim == 3 else candidates,
            measurement_supported_mask=arrays["measurement_supported_mask"][mode],
            point_component=point_component, phi=arrays["phi"][mode:mode + 1], **common,
        ),
    } for mode in range(mode_count)]
    # The helper's field-deviation metric centers over its point dimension. For
    # pooled mode/point weights the correct internal-motion metric is therefore
    # replaced below by an aggregation of the per-mode statistics.
    aggregate = shared._weight_field_diagnostics(
        weights=weights.reshape(-1, basis_count), entropy=arrays["weight_entropy"].reshape(-1),
        candidate_mask=np.broadcast_to(candidates, weights.shape).reshape(-1, basis_count),
        measurement_supported_mask=arrays["measurement_supported_mask"].reshape(-1),
        point_component=np.tile(point_component, mode_count),
        phi=arrays["phi"].reshape(1, mode_count * point_count, 3), **common,
    )
    internal = []
    for basis, component in enumerate(arrays["basis_component_index"][:-1].tolist()):
        members = np.flatnonzero(point_component == component)
        field = arrays["phi"][:, members].astype(np.complex128)
        deviations = np.linalg.norm(field - np.mean(field, axis=1, keepdims=True), axis=2)
        internal.append({
            "basis_index": basis, "component_index": component,
            "complex_displacement_magnitude_mean": float(np.mean(np.linalg.norm(field, axis=2))),
            "within_component_deviation_rms": float(np.sqrt(np.mean(deviations ** 2))),
            "within_component_deviation_median": float(np.median(deviations)),
            "within_component_deviation_p90": float(np.percentile(deviations, 90)),
        })
    aggregate["selected_component_internal_motion"] = internal
    aggregate["aggregation_unit"] = "mode_gaussian_pair"
    aggregate["per_mode"] = per_mode
    if "basis_active_mask" in arrays:
        active_counts = np.count_nonzero(arrays["basis_active_mask"][:, :-1], axis=1)
        minimum_used = np.minimum(3, active_counts)
        for mode, record in enumerate(per_mode):
            zero_median = record["measurement_supported_zero_weight"]["median"]
            record["basis_collapse_warning"] = bool(
                (zero_median is not None and zero_median > 0.95)
                or record["nonzero_bases_with_mean_weight_above_0_01"] < minimum_used[mode]
            )
        aggregate["basis_collapse_warning"] = any(record["basis_collapse_warning"] for record in per_mode)
        aggregate["basis_collapse_warning_reduction"] = "any_mode"
        aggregate["basis_collapse_minimum_used_bases_per_mode"] = minimum_used.astype(int).tolist()
        aggregate["basis_active_counts_per_mode"] = active_counts.astype(int).tolist()
        aggregate["basis_usage_active_mask_per_mode"] = arrays["basis_active_mask"].tolist()
    return aggregate


def build_frequency_motion_basis_modes_artifact(
    *, scene_dir: str | Path, topology_dir: str | Path,
    measurements_dir: str | Path, observed_graph_dir: str | Path,
    rigid_modes_dir: str | Path, work_dir: str | Path, output_dir: str | Path,
    config: MotionBasisConfig | None = None, resume: bool = False,
    command: Sequence[str] = (),
) -> FrequencyMotionBasisModesArtifact:
    """Fit independent weights; publish v6 for independently trusted bases."""

    settings = config or MotionBasisConfig()
    settings.validate()
    version = PER_MODE_COMPLETED_MODES_VERSION if settings.basis_selection_policy == "trusted_per_mode" else COMPLETED_MODES_VERSION
    if not isinstance(resume, bool):
        raise TypeError("Per-frequency motion-basis resume must be a boolean")
    device = shared._resolve_device(settings.device)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Per-frequency motion-basis output already exists: {destination}")
    (source, _, topology, measurements, _, rigid, points, component_indices,
     translation, rotation, centroid, radius) = shared._load_sources(
        scene_dir=scene_dir, topology_dir=topology_dir, measurements_dir=measurements_dir,
        observed_graph_dir=observed_graph_dir, rigid_modes_dir=rigid_modes_dir, config=settings,
    )
    mode_count, view_count = len(source["modes"]), len(source["views"])
    source.update(weight_sharing=WEIGHT_SHARING, objective_reduction=OBJECTIVE_REDUCTION)
    source["motion_basis"] = {**source["motion_basis"], "resolved_device": str(device)}
    source["solver_run_identity"] = _identity(_source_identity_payload(source))
    work = Path(work_dir).expanduser().resolve()
    shared._prepare_work_dir(work, source, resume=resume)
    report_progress("per-frequency motion-basis: building unchanged foreground graph and candidates")
    graph = shared.build_foreground_graph(points, settings)
    point_component = np.asarray(rigid.arrays["point_component_index"], dtype=np.int32)
    basis_owner = np.full(len(points), -1, dtype=np.int8)
    for basis, component in enumerate(component_indices[:-1].tolist()):
        basis_owner[point_component == component] = np.int8(basis)
    active = _active_bases(rigid, component_indices) if version == PER_MODE_COMPLETED_MODES_VERSION else None
    candidate, common_prior = shared._candidate_and_prior_weights(
        points, point_component, component_indices[:-1], settings, basis_active_mask=active,
    )
    identifiable = np.asarray(rigid.arrays["alpha_identifiable_mask"], dtype=bool)
    roles = _frequency_roles(graph, topology, identifiable, len(points))
    prior = _frequency_prior(common_prior, roles["zero_fallback_mask"])
    design = shared._build_design_file(
        path=work / shared.DESIGN_FILENAME, points=points, topology=topology,
        alphas=np.asarray(rigid.arrays["alphas"], dtype=np.complex128),
        basis_translation=translation, basis_rotation=rotation,
        basis_centroid=centroid, candidate_mask=candidate, config=settings,
    )
    offsets = np.asarray(topology.arrays.sample_offsets, dtype=np.int64)
    contributor_sample = np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets))
    contributor_point = np.asarray(topology.arrays.contributor_gaussian_index, dtype=np.int64)
    sample_view = np.asarray(topology.arrays.sample_view_index, dtype=np.int64)
    sample_confidence = np.asarray(topology.arrays.sample_foreground_alpha, dtype=np.float64)
    global_sample_weights, block_rms, energy_floor = shared._sample_loss_weights(
        measurements.measurements, sample_view, sample_confidence, identifiable, settings,
    )
    weights = np.empty((mode_count, len(points), len(component_indices)), dtype=np.float32)
    residual = np.empty(measurements.measurements.shape[:2], dtype=np.float32)
    records: list[dict[str, Any]] = []
    for mode in range(mode_count):
        report_progress(f"per-frequency motion-basis: mode {mode + 1}/{mode_count}")
        mode_work = work / f"mode_{mode:03d}"
        mode_source = {
            **source, "parent_solver_run_identity": source["solver_run_identity"],
            "mode_slot": mode,
            "solver_run_identity": _identity({
                "parent_solver_run_identity": source["solver_run_identity"],
                "mode_slot": mode, "source_mode": source["modes"][mode],
            }),
        }
        # A resumed parent may not have reached this mode yet. An existing mode
        # manifest must match its identity before any checkpoint is reused.
        mode_resume = resume and (mode_work / shared.WORK_MANIFEST_FILENAME).is_file()
        shared._prepare_work_dir(mode_work, mode_source, resume=mode_resume)
        operator = shared.PixelBasisOperator(
            design=design[mode:mode + 1], measurements=measurements.measurements[mode:mode + 1],
            contributor_sample_index=contributor_sample, contributor_point_index=contributor_point,
            sample_loss_weight=global_sample_weights[mode:mode + 1] * mode_count,
            mode_chunk_size=settings.mode_chunk_size,
        )
        mode_candidate = candidate[mode] if active is not None else candidate
        solved, record = shared._solve_weights(
            operator=operator, graph=graph, candidate_mask=mode_candidate, prior=prior[mode],
            measurement_supported_mask=roles["measurement_supported_mask"][mode],
            zero_fallback_mask=roles["zero_fallback_mask"][mode], work=mode_work,
            solver_run_identity=mode_source["solver_run_identity"],
            config=settings, device=device, resume=mode_resume,
        )
        weights[mode] = solved.astype(np.float32)
        weights[mode][~mode_candidate] = 0.0
        fallback = roles["zero_fallback_mask"][mode]
        weights[mode, fallback] = 0.0
        weights[mode, fallback, -1] = 1.0
        residual[mode] = record.pop("sample_residual_rms")[0]
        records.append({"mode_slot": mode, **record})
    phi = _completed_phi(
        points=points, weights=weights, translation=translation, rotation=rotation,
        centroid=centroid, config=settings,
    )
    entropy = -np.sum(weights * np.log(np.maximum(weights, np.finfo(np.float32).tiny)), axis=2)
    arrays = {
        "phi": phi, "weights": weights, "spatial_prior_weights": prior.astype(np.float32),
        "basis_component_index": component_indices.astype(np.int32), "basis_owner_index": basis_owner,
        "basis_translation": translation.astype(np.complex64), "basis_rotation": rotation.astype(np.complex64),
        "basis_centroid": centroid.astype(np.float32), "basis_radius": radius.astype(np.float32),
        "candidate_mask": candidate, **roles,
        "graph_degree": graph.degree.astype(np.int32), "graph_component_index": graph.component_index.astype(np.int32),
        "spatial_edge_index": graph.edge_index.astype(np.int32), "spatial_edge_distance": graph.edge_distance.astype(np.float32),
        "spatial_edge_weight": graph.edge_weight.astype(np.float32), "weight_entropy": entropy.astype(np.float32),
        "dominant_basis_index": np.argmax(weights, axis=2).astype(np.int16), "sample_residual_rms": residual,
    }
    if active is not None:
        arrays["basis_active_mask"] = active
    _validate_arrays(
        arrays, mode_count=mode_count, view_count=view_count, point_count=len(points),
        basis_count=len(component_indices), sample_count=residual.shape[1], edge_count=len(graph.edge_index), version=version,
    )
    fit_args = {
        "residual_squared": 2.0 * residual.astype(np.float64) ** 2,
        "signal_squared": np.sum(np.abs(np.asarray(measurements.measurements, dtype=np.complex128)) ** 2, axis=2),
        "sample_view": sample_view, "sample_confidence": sample_confidence, "identifiable": identifiable,
    }
    all_modes, all_views = list(range(mode_count)), list(range(view_count))
    graph_difference = 0.0
    if len(graph.edge_index):
        difference_squared = np.sum((weights[:, graph.edge_index[:, 0]].astype(np.float64)
                                     - weights[:, graph.edge_index[:, 1]]) ** 2, axis=2)
        graph_difference = float(np.sqrt(np.mean(np.average(difference_squared, axis=1, weights=graph.edge_weight))))
    diagnostics = {
        "overall_fit": shared._fit_metrics(**fit_args, modes=all_modes, views=all_views),
        "per_view_fit": [{"view_index": view, "view_label": source["views"][view]["label"],
                          **shared._fit_metrics(**fit_args, modes=all_modes, views=[view])} for view in all_views],
        "per_mode_fit": [{"mode_slot": mode, **shared._fit_metrics(**fit_args, modes=[mode], views=all_views)} for mode in all_modes],
        "mode_view_fit": [{"mode_slot": mode, "view_index": view, "view_label": source["views"][view]["label"],
                           **shared._fit_metrics(**fit_args, modes=[mode], views=[view])} for mode in all_modes for view in all_views],
        "old_component_seams": shared._seam_diagnostics(phi, graph, point_component),
        "weight_field": _weight_diagnostics(arrays, point_component),
        "zero_basis_weight_mean": float(np.mean(weights[:, :, -1])),
        "zero_basis_weight_p50": float(np.percentile(weights[:, :, -1], 50)),
        "zero_basis_weight_p90": float(np.percentile(weights[:, :, -1], 90)),
        "weight_entropy_mean": float(np.mean(entropy)), "weight_entropy_p90": float(np.percentile(entropy, 90)),
        "graph_weight_difference_rms": graph_difference,
    }
    counts = _counts(
        source=source, rigid=rigid, point_count=len(points), basis_count=len(component_indices),
        sample_count=residual.shape[1], contributor_count=len(contributor_point), graph=graph, roles=roles,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent))
    try:
        arrays_path = temporary / COMPLETED_MODES_FILENAME
        save_named_arrays(arrays_path, arrays)
        manifest = {
            "format": COMPLETED_MODES_FORMAT, "version": version,
            "completion_method": COMPLETION_METHOD,
            "producer": {"project_version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(), "command": list(command)},
            **source, "work_dir": str(work),
            "mode_selection": _mode_selection(mode_count),
            "semantics": _semantics(version),
            "quality_gate": {"required": True, "status": "completion_candidate_unapproved"},
            "spatial_graph": {
                "policy": "distance_pruned_union_knn_without_rgb_depth_boundaries",
                "neighbors": settings.graph_neighbors, "max_distance": settings.graph_max_distance,
                "kernel": "gaussian", "kernel_formula": "exp(-(distance / max_distance)^2)",
                "candidate_directed_count": graph.candidate_directed_count,
                "retained_directed_count": graph.retained_directed_count,
            },
            "optimization": {**_optimization_summary(records),
                             "solver": "independent_monotone_masked_simplex_projected_fista_with_backtracking",
                             "mode_view_measurement_rms": block_rms.tolist(), "measurement_rms_floor": energy_floor},
            "diagnostics": diagnostics, "counts": counts, "arrays_file": COMPLETED_MODES_FILENAME,
            "arrays": {name: {"dtype": value.dtype.name, "shape": list(value.shape)} for name, value in arrays.items()},
            "arrays_identity": shared._arrays_identity(arrays), "arrays_file_sha256": shared._sha256_file(arrays_path),
        }
        manifest["completed_modes_identity"] = _identity(_artifact_identity_payload(manifest))
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        load_frequency_motion_basis_modes(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Per-frequency motion-basis output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_frequency_motion_basis_modes(destination)


__all__ = [
    "MotionBasisConfig", "FrequencyMotionBasisModesArtifact",
    "build_frequency_motion_basis_modes_artifact", "load_frequency_motion_basis_modes",
]
