"""Refine only unobserved per-frequency basis weights with fixed observations.

Version 5 derives from a strictly loaded version 4 artifact; version 7 derives
from version 6 with per-mode candidate and active-basis masks. All measurement
contributors and unsupported components remain bitwise unchanged. The green
variables retain the parent's candidate simplex; regularizers retain the
parent's full graph and foreground denominators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import copy
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from modal_gaussians import __version__
from modal_gaussians.motion.rigid import motion_basis as shared
from modal_gaussians.motion.rigid import motion_basis_frequency as frequency
from modal_gaussians.numpy_io import save_named_arrays
from modal_gaussians.progress import report_progress


COMPLETED_MODES_FORMAT = shared.COMPLETED_MODES_FORMAT
COMPLETED_MODES_FILENAME = shared.COMPLETED_MODES_FILENAME
COMPLETED_MODES_VERSION = 5
PER_MODE_COMPLETED_MODES_VERSION = 7
COMPLETION_METHOD = "fixed_observation_green_basis_refinement"
WEIGHT_SHARING = frequency.WEIGHT_SHARING
OBJECTIVE_REDUCTION = "mean_mode_green_objectives_with_parent_full_graph_and_point_denominators"
MUTABLE_FIELDS = frozenset({"weights", "phi", "weight_entropy", "dominant_basis_index"})
PARENT_MANIFEST_FIELDS = (
    "static_scene", "static_scene_identity", "foreground_identity", "topology",
    "topology_identity", "measurements", "gaussian_measurements_identity",
    "observed_structure_graph", "observed_structure_graph_identity", "rigid_modes",
    "rigid_modes_identity", "modes", "views", "basis_selection", "motion_basis",
    "counts", "mode_selection", "spatial_graph", "quality_gate",
)


@dataclass(frozen=True)
class GreenRefinementConfig:
    blue_green_multiplier: float = 1.0
    green_prior_multiplier: float = 1.0
    max_iterations: int = 3000
    relative_tolerance: float = 1.0e-8
    projected_gradient_tolerance: float = 1.0e-6
    convergence_patience: int = 10
    checkpoint_every: int = 50
    device: str = "auto"

    def validate(self) -> None:
        for name in ("max_iterations", "convergence_patience", "checkpoint_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Green refinement {name} must be a positive integer")
        for name in ("blue_green_multiplier", "green_prior_multiplier",
                     "relative_tolerance", "projected_gradient_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"Green refinement {name} must be finite and nonnegative")
        if self.blue_green_multiplier <= 0:
            raise ValueError("Green refinement blue_green_multiplier must be positive")
        if self.relative_tolerance == self.projected_gradient_tolerance == 0:
            raise ValueError("Green refinement needs at least one convergence tolerance")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Green refinement device must be auto, cpu, or cuda")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GreenRefinedMotionBasisModesArtifact:
    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _semantics(version: int = COMPLETED_MODES_VERSION) -> dict[str, Any]:
    if version not in (COMPLETED_MODES_VERSION, PER_MODE_COMPLETED_MODES_VERSION):
        raise ValueError("Unsupported green-refinement semantics version")
    base = (frequency._semantics(version=6)
            if version == PER_MODE_COMPLETED_MODES_VERSION else frequency._semantics())
    result = {
        **base,
        "method": "fixed_observation_green_masked_simplex_fista",
        "objective_reduction": OBJECTIVE_REDUCTION,
        "regularization": "blue_green_multiplier_times_parent_graph_plus_green_graph_and_reduced_green_prior",
        "optimized_roles": ["graph_propagated"],
        "fixed_roles": ["measurement_supported", "zero_fallback"],
        "initialization": "parent_per_frequency_weights",
        "graph_denominator": "sum_parent_original_full_graph_edge_weights",
        "prior_denominator": "parent_foreground_gaussian_count",
        "sample_loss_weight": "not_reoptimized_fixed_topology_predictions",
        "fixed_observations": "parent_weights_phi_and_sample_residual_rms_bitwise_equal",
    }
    if version == PER_MODE_COMPLETED_MODES_VERSION:
        result.update(
            candidate_shape="K_G_B",
            basis_active_shape="K_B",
            basis_activation="immutable_parent_per_mode_trust_mask",
        )
    return result


def _source_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    version = manifest.get("version", COMPLETED_MODES_VERSION)
    return {
        "version": version, "completion_method": COMPLETION_METHOD,
        "parent_completed_modes_identity": manifest["parent_completed_modes_identity"],
        "green_refinement": manifest["green_refinement"],
        "objective_reduction": OBJECTIVE_REDUCTION, "semantics": _semantics(version),
    }


def _artifact_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": COMPLETED_MODES_FORMAT, **_source_identity_payload(manifest),
        **{name: manifest[name] for name in PARENT_MANIFEST_FIELDS
           if name not in {"static_scene", "topology", "measurements", "observed_structure_graph", "rigid_modes"}},
        **{name: manifest[name] for name in ("weight_sharing", "optimization", "diagnostics", "arrays_identity")},
    }


def _parent_settings(parent: frequency.FrequencyMotionBasisModesArtifact) -> shared.MotionBasisConfig:
    return shared._config_from_manifest(parent.manifest["motion_basis"])


def _points(parent: frequency.FrequencyMotionBasisModesArtifact) -> np.ndarray:
    scene = shared.load_static_scene(parent.manifest["static_scene"], "cpu")
    return scene.foreground.raw_tensors(cpu=True)["means"].numpy().astype(np.float32, copy=False)


class _GreenObjective:
    """Reduced convex quadratic; no data term can depend on these variables."""

    def __init__(
        self, *, initial: np.ndarray, prior: np.ndarray, green: np.ndarray,
        blue: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray,
        smooth_weight: float, prior_weight: float, config: GreenRefinementConfig,
        device: torch.device,
    ) -> None:
        self.device = device
        self.green_index = np.flatnonzero(green)
        self.point_count = len(initial)
        inverse = np.full(self.point_count, -1, dtype=np.int64)
        inverse[self.green_index] = np.arange(len(self.green_index))
        edge = np.asarray(edge_index, dtype=np.int64)
        weight = np.asarray(edge_weight, dtype=np.float64)
        self.edge_denominator = max(float(weight.sum()), shared.EPSILON)
        self.prior_coefficient = prior_weight * config.green_prior_multiplier / max(self.point_count, 1)
        self.prior = torch.as_tensor(prior[self.green_index], dtype=torch.float64, device=device)
        self.groups: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, bool]] = []
        row_bound = np.zeros(len(self.green_index), dtype=np.float64)
        ii, jj = edge[:, 0], edge[:, 1]
        gg = green[ii] & green[jj]
        one_green = green[ii] ^ green[jj]
        oriented_green = np.where(green[ii], ii, jj)
        oriented_fixed = np.where(green[ii], jj, ii)
        bg = one_green & blue[oriented_fixed]
        other = one_green & ~blue[oriented_fixed]
        for name, selected, multiplier, both_green in (
            ("green_green", gg, 1.0, True),
            ("blue_green", bg, config.blue_green_multiplier, False),
            ("green_fixed_other", other, 1.0, False),
        ):
            left = inverse[ii[selected]] if both_green else inverse[oriented_green[selected]]
            right = inverse[jj[selected]] if both_green else oriented_fixed[selected]
            coeff = smooth_weight * multiplier * weight[selected] / self.edge_denominator
            np.add.at(row_bound, left, (4.0 if both_green else 2.0) * coeff)
            if both_green:
                np.add.at(row_bound, right, 4.0 * coeff)
            self.groups.append((
                name, torch.as_tensor(left, dtype=torch.long, device=device),
                torch.as_tensor(right, dtype=torch.long, device=device) if both_green else
                torch.as_tensor(initial[right], dtype=torch.float64, device=device),
                torch.as_tensor(coeff[:, None], dtype=torch.float64, device=device), both_green,
            ))
        self.lipschitz_bound = max(float(np.max(row_bound, initial=0.0)) + 2 * self.prior_coefficient, 1.0e-15)
        fixed = ~(green[ii] | green[jj])
        self.fixed_graph_constant = float(smooth_weight * np.sum(
            weight[fixed, None] * (initial[ii[fixed]].astype(np.float64) - initial[jj[fixed]]) ** 2
        ) / self.edge_denominator)

    def value_gradient(self, weights: torch.Tensor, *, gradient: bool = True):
        grad = torch.zeros_like(weights) if gradient else None
        terms = {}
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        for name, left, right, coeff, both_green in self.groups:
            difference = weights[left] - (weights[right] if both_green else right)
            value = torch.sum(coeff * difference.square())
            total += value
            terms[name] = float(value.item())
            if grad is not None:
                edge_grad = 2 * coeff * difference
                grad.index_add_(0, left, edge_grad)
                if both_green:
                    grad.index_add_(0, right, -edge_grad)
        difference = weights - self.prior
        prior_value = self.prior_coefficient * torch.sum(difference.square())
        total += prior_value
        if grad is not None:
            grad += 2 * self.prior_coefficient * difference
        terms["green_prior"] = float(prior_value.item())
        terms["total"] = float(total.item())
        if not math.isfinite(terms["total"]) or (grad is not None and not bool(torch.isfinite(grad).all())):
            raise FloatingPointError("Green refinement objective or gradient is nonfinite")
        return terms["total"], terms, grad


def _solve_green_weights(
    *, initial: np.ndarray, prior: np.ndarray, candidate_mask: np.ndarray,
    green: np.ndarray, blue: np.ndarray, edge_index: np.ndarray,
    edge_weight: np.ndarray, smooth_weight: float, prior_weight: float,
    config: GreenRefinementConfig, device: torch.device, work: Path,
    solver_run_identity: str, resume: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    objective = _GreenObjective(
        initial=initial, prior=prior, green=green, blue=blue, edge_index=edge_index,
        edge_weight=edge_weight, smooth_weight=smooth_weight, prior_weight=prior_weight,
        config=config, device=device,
    )
    selected = objective.green_index
    mask = torch.as_tensor(candidate_mask[selected], dtype=torch.bool, device=device)
    fallback = torch.zeros(len(selected), dtype=torch.bool, device=device)

    def project(value):
        return shared._project_masked_simplex_torch(value, mask, fallback) if len(selected) else value

    checkpoint = work / shared.CHECKPOINT_FILENAME
    if resume and checkpoint.is_file():
        state = shared._load_checkpoint(checkpoint, solver_run_identity=solver_run_identity,
                                        expected_shape=(len(selected), initial.shape[1]))
        current = torch.as_tensor(state["weights"], device=device)
        previous = torch.as_tensor(state["previous_weights"], device=device)
        for value in (current, previous):
            if not torch.allclose(value, project(value), atol=1.0e-10, rtol=0):
                raise ValueError("Green refinement checkpoint violates the candidate simplex")
        iteration, fista_t, lipschitz = state["iteration"], state["fista_t"], state["lipschitz"]
        stable, searches, history = state["stable_count"], state["line_search_steps_total"], state["objective_history"]
        if (iteration > config.max_iterations or not math.isfinite(lipschitz)
                or not math.isfinite(fista_t) or stable < 0):
            raise ValueError("Green refinement checkpoint optimizer state is invalid")
        actual, _, _ = objective.value_gradient(current, gradient=False)
        if not math.isclose(actual, history[-1], rel_tol=1.0e-10, abs_tol=1.0e-14):
            raise ValueError("Green refinement checkpoint objective differs from weights")
    else:
        current = project(torch.as_tensor(initial[selected], dtype=torch.float64, device=device))
        previous = current.clone()
        iteration, fista_t, stable, searches = 0, 1.0, 0, 0
        lipschitz = objective.lipschitz_bound
        history = [objective.value_gradient(current, gradient=False)[0]]

    def save():
        shared._save_checkpoint(
            checkpoint, solver_run_identity=solver_run_identity,
            weights=current.cpu().numpy(), previous_weights=previous.cpu().numpy(),
            iteration=iteration, fista_t=fista_t, lipschitz=lipschitz,
            stable_count=stable, line_search_steps_total=searches, objective_history=history,
        )

    if not resume or not checkpoint.is_file():
        save()
    variable_count = max(int(mask.count_nonzero()), 1)

    def search(origin, origin_value, grad):
        trial_lipschitz = lipschitz
        for attempts in range(1, 42):
            proposal = project(origin - grad / trial_lipschitz)
            value = objective.value_gradient(proposal, gradient=False)[0]
            change = proposal - origin
            majorizer = origin_value + float(torch.sum(grad * change)) + trial_lipschitz * .5 * float(torch.sum(change.square()))
            tolerance = 1.0e-12 * max(abs(origin_value), 1.0e-12)
            if value <= majorizer + tolerance:
                return proposal, value, trial_lipschitz, attempts
            trial_lipschitz *= 2.0
        raise RuntimeError("Green refinement FISTA backtracking failed")

    converged = stable >= config.convergence_patience or not len(selected)
    while iteration < config.max_iterations and not converged:
        next_t = .5 * (1 + math.sqrt(1 + 4 * fista_t * fista_t))
        origin = project(current + (fista_t - 1) / next_t * (current - previous))
        value, _, grad = objective.value_gradient(origin)
        proposal, value, lipschitz, attempts = search(origin, value, grad)
        searches += attempts
        if value > history[-1] + 1.0e-12 * max(abs(history[-1]), 1.0e-12):
            origin = current
            origin_value, _, grad = objective.value_gradient(origin)
            proposal, value, lipschitz, attempts = search(origin, origin_value, grad)
            searches += attempts
            next_t = 1.0
        relative = abs(history[-1] - value) / max(abs(history[-1]), shared.EPSILON)
        pg = lipschitz * float(torch.linalg.vector_norm(proposal - origin)) / math.sqrt(variable_count)
        if relative <= config.relative_tolerance and pg <= config.projected_gradient_tolerance:
            # Check stationarity at the point that will actually be returned,
            # not only at the accelerated/extrapolated search origin.
            proposal_grad = objective.value_gradient(proposal)[2]
            pg = lipschitz * float(torch.linalg.vector_norm(
                project(proposal - proposal_grad / lipschitz) - proposal
            )) / math.sqrt(variable_count)
            stable = stable + 1 if pg <= config.projected_gradient_tolerance else 0
        else:
            stable = 0
        previous, current = current, proposal
        fista_t, iteration = next_t, iteration + 1
        history.append(value)
        converged = stable >= config.convergence_patience
        if iteration % config.checkpoint_every == 0 or converged or iteration == config.max_iterations:
            save()
        if iteration % 250 == 0 or converged or iteration == config.max_iterations:
            report_progress(f"green refinement: iteration {iteration}, loss={value:.7g}, pg={pg:.3g}")
    final_value, final_terms, grad = objective.value_gradient(current)
    pg = lipschitz * float(torch.linalg.vector_norm(project(current - grad / lipschitz) - current)) / math.sqrt(variable_count)
    output = initial.copy()
    output[selected] = current.cpu().numpy().astype(initial.dtype)
    return output, {
        "converged": bool(converged), "iterations": iteration, "stable_iterations": stable,
        "initial_objective": float(history[0]), "final_objective": final_value, "final_terms": final_terms,
        "projected_gradient_norm": pg,
        "relative_objective_change": abs(history[-1] - history[-2]) / max(abs(history[-2]), shared.EPSILON) if len(history) > 1 else 0.0,
        "line_search_steps_total": searches, "objective_history": history,
        "final_lipschitz": lipschitz, "gershgorin_lipschitz_bound": objective.lipschitz_bound,
        "requested_device": config.device, "resolved_device": str(device),
        "optimized_gaussians": len(selected), "graph_denominator": objective.edge_denominator,
        "prior_denominator": objective.point_count, "fixed_graph_constant_excluded": objective.fixed_graph_constant,
    }


def _validate_against_parent(
    arrays: Mapping[str, np.ndarray], parent: frequency.FrequencyMotionBasisModesArtifact,
) -> None:
    counts = parent.manifest["counts"]
    version_options = {"version": 6} if parent.manifest["version"] == 6 else {}
    frequency._validate_arrays(
        arrays, mode_count=counts["modes"], view_count=counts["views"],
        point_count=counts["foreground_gaussians"], basis_count=counts["bases"],
        sample_count=counts["measurement_samples"], edge_count=counts["spatial_edges"],
        **version_options,
    )
    for name in set(arrays) - MUTABLE_FIELDS:
        if not np.array_equal(arrays[name], parent.arrays[name]):
            raise ValueError(f"Green refinement modified immutable parent array {name}")
    fixed = ~parent.arrays["graph_propagated_mask"]
    for name in MUTABLE_FIELDS:
        if not np.array_equal(arrays[name][fixed], parent.arrays[name][fixed]):
            raise ValueError(f"Green refinement modified fixed parent {name}")


def load_green_refined_motion_basis_modes(path: str | Path) -> GreenRefinedMotionBasisModesArtifact:
    root = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    version = manifest.get("version")
    if (manifest.get("format") != COMPLETED_MODES_FORMAT
            or type(version) is not int or version not in (5, 7)
            or manifest.get("completion_method") != COMPLETION_METHOD
            or manifest.get("weight_sharing") != WEIGHT_SHARING
            or manifest.get("objective_reduction") != OBJECTIVE_REDUCTION
            or manifest.get("semantics") != _semantics(version)):
        raise ValueError("Unsupported green-refined motion-basis artifact or semantics")
    payload = manifest.get("green_refinement", {})
    try:
        config = GreenRefinementConfig(**{name: payload[name] for name in GreenRefinementConfig.__dataclass_fields__})
    except (TypeError, KeyError) as error:
        raise ValueError("Invalid green refinement configuration") from error
    config.validate()
    if payload != {**config.to_dict(), "resolved_device": payload.get("resolved_device")} or payload.get("resolved_device") not in {"cpu", "cuda"}:
        raise ValueError("Green refinement configuration has unexpected fields")
    parent = frequency.load_frequency_motion_basis_modes(manifest["parent_completed_modes"])
    expected_parent_version = 6 if version == PER_MODE_COMPLETED_MODES_VERSION else 4
    if parent.manifest["version"] != expected_parent_version:
        raise ValueError("Green refinement version does not match its parent schema")
    if manifest.get("parent_completed_modes_identity") != parent.manifest["completed_modes_identity"]:
        raise ValueError("Green refinement parent artifact identity differs")
    for name in PARENT_MANIFEST_FIELDS:
        if manifest.get(name) != parent.manifest.get(name):
            raise ValueError(f"Green refinement parent metadata {name} differs")
    arrays_path = root / COMPLETED_MODES_FILENAME
    if manifest.get("arrays_file") != COMPLETED_MODES_FILENAME or manifest.get("arrays_file_sha256") != shared._sha256_file(arrays_path):
        raise ValueError("Green refinement arrays filename or file hash differs")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    _validate_against_parent(arrays, parent)
    diagnostics = manifest.get("diagnostics", {})
    for name in ("overall_fit", "per_view_fit", "per_mode_fit", "mode_view_fit"):
        if diagnostics.get(name) != parent.manifest.get("diagnostics", {}).get(name):
            raise ValueError(f"Green refinement fixed topology diagnostic {name} differs")
    optimization = manifest.get("optimization", {})
    records = optimization.get("per_mode", [])
    if (optimization.get("objective_reduction") != OBJECTIVE_REDUCTION
            or len(records) != len(arrays["weights"])
            or optimization.get("mode_count") != len(records)):
        raise ValueError("Green refinement optimization mode metadata differs")
    denominator = max(float(np.sum(arrays["spatial_edge_weight"].astype(np.float64))), shared.EPSILON)
    for mode, record in enumerate(records):
        if (record.get("mode_slot") != mode
                or record.get("graph_denominator") != denominator
                or record.get("prior_denominator") != arrays["weights"].shape[1]
                or record.get("optimized_gaussians") != int(np.count_nonzero(arrays["graph_propagated_mask"][mode]))
                or record.get("requested_device") != config.device
                or record.get("resolved_device") != payload["resolved_device"]):
            raise ValueError("Green refinement optimization normalization or mode metadata differs")
    phi = frequency._completed_phi(
        points=_points(parent), weights=arrays["weights"], translation=arrays["basis_translation"],
        rotation=arrays["basis_rotation"], centroid=arrays["basis_centroid"], config=_parent_settings(parent),
    )
    if not np.allclose(arrays["phi"], phi, rtol=2.0e-5, atol=2.0e-6):
        raise ValueError("Green refinement phi differs from basis-weight composition")
    metadata = {name: {"dtype": value.dtype.name, "shape": list(value.shape)} for name, value in arrays.items()}
    if manifest.get("arrays") != metadata or manifest.get("arrays_identity") != shared._arrays_identity(arrays):
        raise ValueError("Green refinement array metadata or identity differs")
    if manifest.get("solver_run_identity") != frequency._identity(_source_identity_payload(manifest)):
        raise ValueError("Green refinement solver identity differs")
    if manifest.get("completed_modes_identity") != frequency._identity(_artifact_identity_payload(manifest)):
        raise ValueError("Green refinement completed artifact identity differs")
    return GreenRefinedMotionBasisModesArtifact(root, manifest, arrays)


def _role_diagnostics(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    edges = arrays["spatial_edge_index"]
    edge_weight = arrays["spatial_edge_weight"].astype(np.float64)
    ii, jj = edges[:, 0], edges[:, 1]
    records = []
    for mode, weights in enumerate(arrays["weights"]):
        blue, green = arrays["measurement_supported_mask"][mode], arrays["graph_propagated_mask"][mode]
        difference = np.sum((weights[ii].astype(np.float64) - weights[jj]) ** 2, axis=1)
        record = {"mode_slot": mode}
        for name, mask in (("blue_blue", blue[ii] & blue[jj]),
                           ("blue_green", (blue[ii] & green[jj]) | (green[ii] & blue[jj])),
                           ("green_green", green[ii] & green[jj])):
            record[name] = {"edge_count": int(mask.sum()),
                            "weight_difference_rms": float(np.sqrt(np.average(difference[mask], weights=edge_weight[mask]))) if np.any(mask) else None}
        records.append(record)
    return records


def build_green_refined_motion_basis_artifact(
    *, parent_dir: str | Path, work_dir: str | Path, output_dir: str | Path,
    config: GreenRefinementConfig | None = None, resume: bool = False,
    command: Sequence[str] = (),
) -> GreenRefinedMotionBasisModesArtifact:
    settings = config or GreenRefinementConfig()
    settings.validate()
    if not isinstance(resume, bool):
        raise TypeError("Green refinement resume must be boolean")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Green refinement output already exists: {destination}")
    parent = frequency.load_frequency_motion_basis_modes(parent_dir)
    parent_version = parent.manifest["version"]
    if parent_version not in (4, 6):
        raise ValueError("Green refinement requires per-frequency v4 or v6 parent modes")
    version = (PER_MODE_COMPLETED_MODES_VERSION if parent_version == 6
               else COMPLETED_MODES_VERSION)
    device = shared._resolve_device(settings.device)
    source = {name: copy.deepcopy(parent.manifest[name]) for name in PARENT_MANIFEST_FIELDS}
    source.update(
        parent_completed_modes=str(parent.path), parent_completed_modes_identity=parent.manifest["completed_modes_identity"],
        green_refinement={**settings.to_dict(), "resolved_device": str(device)},
        weight_sharing=WEIGHT_SHARING, objective_reduction=OBJECTIVE_REDUCTION,
    )
    # Keep the legacy v4 -> v5 source/work payload byte-compatible. Version 7
    # explicitly binds the new per-mode candidate schema in its run identity.
    if version == PER_MODE_COMPLETED_MODES_VERSION:
        source["version"] = version
    source["solver_run_identity"] = frequency._identity(_source_identity_payload(source))
    work = Path(work_dir).expanduser().resolve()
    shared._prepare_work_dir(work, source, resume=resume)
    arrays = {name: value.copy() for name, value in parent.arrays.items()}
    parent_config = _parent_settings(parent)
    records = []
    for mode in range(len(arrays["weights"])):
        report_progress(f"green refinement: mode {mode + 1}/{len(arrays['weights'])}, observed weights fixed")
        mode_work = work / f"mode_{mode:03d}"
        mode_source = {**source, "mode_slot": mode,
                       "solver_run_identity": frequency._identity({"parent_solver_run_identity": source["solver_run_identity"], "mode_slot": mode})}
        mode_resume = resume and (mode_work / shared.WORK_MANIFEST_FILENAME).is_file()
        shared._prepare_work_dir(mode_work, mode_source, resume=mode_resume)
        arrays["weights"][mode], record = _solve_green_weights(
            initial=parent.arrays["weights"][mode], prior=arrays["spatial_prior_weights"][mode],
            candidate_mask=(arrays["candidate_mask"][mode] if parent_version == 6
                            else arrays["candidate_mask"]),
            green=arrays["graph_propagated_mask"][mode],
            blue=arrays["measurement_supported_mask"][mode], edge_index=arrays["spatial_edge_index"],
            edge_weight=arrays["spatial_edge_weight"], smooth_weight=parent_config.smooth_weight,
            prior_weight=parent_config.prior_weight, config=settings, device=device,
            work=mode_work, solver_run_identity=mode_source["solver_run_identity"], resume=mode_resume,
        )
        records.append({"mode_slot": mode, **record})
    green = arrays["graph_propagated_mask"]
    reconstructed = frequency._completed_phi(
        points=_points(parent), weights=arrays["weights"], translation=arrays["basis_translation"],
        rotation=arrays["basis_rotation"], centroid=arrays["basis_centroid"], config=parent_config,
    )
    arrays["phi"][green] = reconstructed[green]
    entropy = -np.sum(arrays["weights"] * np.log(np.maximum(arrays["weights"], np.finfo(np.float32).tiny)), axis=2)
    arrays["weight_entropy"][green] = entropy[green]
    arrays["dominant_basis_index"][green] = np.argmax(arrays["weights"], axis=2)[green]
    _validate_against_parent(arrays, parent)
    diagnostics = {name: copy.deepcopy(value) for name, value in parent.manifest["diagnostics"].items()
                   if name in {"overall_fit", "per_view_fit", "per_mode_fit", "mode_view_fit"}}
    diagnostics.update(
        topology_prediction="bitwise_parent_supported_weights_phi_and_sample_residual_rms",
        role_edge_weights=_role_diagnostics(arrays), parent_role_edge_weights=_role_diagnostics(parent.arrays),
        green_weight_change_rms=float(np.sqrt(np.mean((arrays["weights"][green].astype(np.float64) - parent.arrays["weights"][green]) ** 2))) if np.any(green) else 0.0,
    )
    optimization = {**frequency._optimization_summary(records), "objective_reduction": OBJECTIVE_REDUCTION,
                    "solver": "reduced_green_monotone_masked_simplex_fista_with_gershgorin_bound",
                    "constant_observation_and_fixed_regularization_terms": "excluded_from_optimized_objective"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent))
    try:
        arrays_path = temporary / COMPLETED_MODES_FILENAME
        save_named_arrays(arrays_path, arrays)
        manifest = {
            "format": COMPLETED_MODES_FORMAT, "version": version, "completion_method": COMPLETION_METHOD,
            "producer": {"project_version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(), "command": list(command)},
            **source, "work_dir": str(work), "semantics": _semantics(version),
            "optimization": optimization, "diagnostics": diagnostics, "arrays_file": COMPLETED_MODES_FILENAME,
            "arrays": {name: {"dtype": value.dtype.name, "shape": list(value.shape)} for name, value in arrays.items()},
            "arrays_identity": shared._arrays_identity(arrays), "arrays_file_sha256": shared._sha256_file(arrays_path),
        }
        manifest["completed_modes_identity"] = frequency._identity(_artifact_identity_payload(manifest))
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        load_green_refined_motion_basis_modes(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Green refinement output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_green_refined_motion_basis_modes(destination)


__all__ = ["GreenRefinementConfig", "GreenRefinedMotionBasisModesArtifact",
           "build_green_refined_motion_basis_artifact", "load_green_refined_motion_basis_modes"]
