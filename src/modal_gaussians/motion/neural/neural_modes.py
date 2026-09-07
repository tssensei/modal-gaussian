"""Independent neural complex modal fields with frozen full-foreground supervision.

This v8/v10/v11 producer does not train static appearance, temporal coordinates,
or rigid bases. Existing rigid artifacts supply only the fixed complex view
alignment. Production training requires CUDA; the field math is CPU-testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
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
import torch

from modal_gaussians import __version__
from modal_gaussians.numpy_io import save_named_arrays
from modal_gaussians.progress import report_progress
from .pointwise_attachments import MODE_ARRAYS as POINTWISE_MODE_ARRAYS
from .guarded_attachments import MODE_ARRAYS as GUARDED_MODE_ARRAYS
from .component_field import MODE_ARRAYS as COMPONENT_MODE_ARRAYS

COMPLETED_MODES_FORMAT = "modal_gaussians.completed_modes"
COMPLETED_MODES_VERSION = 8
COMPLETION_METHOD = "neural_complex_displacement_field"
ARRAYS_FILENAME = "completed_modes.npz"
MODELS_FILENAME = "networks.pt"
QUALITY_GATE = {"required": True, "status": "completion_candidate_unapproved"}
PREFIX_NORMALIZATION_ARRAYS = {
    "normalization_source_mode_view_rms", "normalization_source_alpha_identifiable_mask",
}
MODE_ARRAYS = {
    "alphas", "alpha_identifiable_mask", "observation_view_mask", "support_class",
    "sample_target", "mode_view_rms", "mode_view_loss_scale", "amplitude_scale",
} | POINTWISE_MODE_ARRAYS | GUARDED_MODE_ARRAYS | COMPONENT_MODE_ARRAYS


@dataclass(frozen=True)
class NeuralModesConfig:
    graph_neighbors: int = 8
    graph_max_distance: float = 0.008
    unknown_max_distance: float = 0.004
    unknown_edge_weight: float = 0.1
    control_radius_fraction: float = 0.03
    max_controls: int = 2048
    hidden_dim: int = 64
    message_layers: int = 3
    local_feature_dim: int = 0
    pixel_sample_stride: int = 2
    alpha_minimum: float = 0.05
    mask_erosion_iterations: int = 1
    energy_floor_fraction: float = 0.05
    huber_delta: float = 1.0
    deformation_weight: float = 1.0
    rotation_weight: float = 0.1
    rotation_length_fraction: float = 0.05
    learning_rate: float = 0.001
    max_iterations: int = 2000
    gradient_clip: float = 1.0
    seed: int = 1729
    convergence_patience: int = 50
    relative_tolerance: float = 1.0e-6
    checkpoint_every: int = 100
    device: str = "auto"
    graph_edge_filter: str = "depth"
    training_fragment_config: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.training_fragment_config is not None:
            from .training_fragments import config_class
            config_class(self.training_fragment_config).from_dict(self.training_fragment_config)
        if self.graph_edge_filter not in ("depth", "none"):
            raise ValueError("Neural graph_edge_filter must be depth or none")
        for name in ("graph_neighbors", "max_controls", "hidden_dim", "message_layers",
                     "pixel_sample_stride", "max_iterations", "convergence_patience",
                     "checkpoint_every"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Neural {name} must be a positive integer")
        for name in ("seed", "mask_erosion_iterations", "local_feature_dim"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Neural {name} must be a non-negative integer")
        for name in ("graph_max_distance", "unknown_max_distance", "unknown_edge_weight",
                     "control_radius_fraction", "alpha_minimum", "energy_floor_fraction",
                     "huber_delta", "rotation_length_fraction", "learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Neural {name} must be finite and positive")
        for name in ("deformation_weight", "rotation_weight", "relative_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Neural {name} must be finite and non-negative")
        if self.unknown_max_distance > self.graph_max_distance:
            raise ValueError("Unknown-edge radius cannot exceed candidate radius")
        if self.alpha_minimum > 1 or self.unknown_edge_weight > 1:
            raise ValueError("Alpha threshold and unknown-edge weight cannot exceed one")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Neural device must be auto, cpu, or cuda")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        if self.graph_edge_filter == "depth":
            result.pop("graph_edge_filter")
        if self.training_fragment_config is None:
            result.pop("training_fragment_config")
        if self.local_feature_dim == 0:
            result.pop("local_feature_dim")
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NeuralModesConfig":
        result = cls(**dict(value))
        result.validate()
        if result.to_dict() != dict(value):
            raise ValueError("Neural configuration is not fully resolved")
        return result


@dataclass(frozen=True)
class NeuralModesArtifact:
    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arrays_identity(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        contiguous = np.ascontiguousarray(value)
        digest.update(_canonical([name, value.dtype.str, list(value.shape)]))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _geometry_config(config: NeuralModesConfig) -> Any:
    from modal_gaussians.motion.neural.geometry_graph import GeometryGraphConfig
    return GeometryGraphConfig(
        max_neighbors=config.graph_neighbors, max_distance=config.graph_max_distance,
        unknown_max_distance=config.unknown_max_distance, unknown_weight=config.unknown_edge_weight,
        alpha_minimum=config.alpha_minimum, control_radius_fraction=config.control_radius_fraction,
        max_controls=config.max_controls,
        edge_filter=config.graph_edge_filter,
    )


def _field_config(config: NeuralModesConfig, mode: int = 0) -> Any:
    from modal_gaussians.motion.neural.neural_field import NeuralFieldConfig
    return NeuralFieldConfig(
        hidden_dim=config.hidden_dim, message_layers=config.message_layers,
        local_feature_dim=config.local_feature_dim,
        learning_rate=config.learning_rate, max_iterations=config.max_iterations,
        gradient_clip=config.gradient_clip, seed=config.seed + mode,
        convergence_patience=config.convergence_patience, relative_tolerance=config.relative_tolerance,
        checkpoint_every=config.checkpoint_every, huber_delta=config.huber_delta,
        deformation_weight=config.deformation_weight, rotation_weight=config.rotation_weight,
        rotation_length_fraction=config.rotation_length_fraction,
    )


def _load_sources(*, scene_dir: str | Path, topology_dir: str | Path,
                  measurements_dir: str | Path, graph_dir: str | Path,
                  alignment_from: str | Path) -> tuple[dict[str, Any], Any, Any, Any, Any, Any, Any]:
    """Validate the existing identity chain; never consume rigid motion or trust."""
    from modal_gaussians.modes import load_complex_2d_modes
    from modal_gaussians.motion.common.sources import load_observed_sources, load_fixed_alignment
    source, scene, topology, measurements, graph = load_observed_sources(
        scene_dir=scene_dir, topology_dir=topology_dir, measurements_dir=measurements_dir,
        graph_dir=graph_dir,
    )
    alignment = load_fixed_alignment(alignment_from)
    for name in ("static_scene_identity", "foreground_identity", "topology_identity",
                 "gaussian_measurements_identity", "observed_structure_graph_identity", "modes", "views"):
        if alignment.manifest.get(name) != source[name]:
            raise ValueError(f"Neural alignment {name} differs from supplied sources")
    dense = load_complex_2d_modes(measurements.manifest["complex_2d_modes"])
    if dense.manifest["complex_2d_modes_identity"] != measurements.manifest["complex_2d_modes_identity"]:
        raise ValueError("Neural dense modal source identity differs")
    if dense.manifest["topology_identity"] != source["topology_identity"] or dense.manifest["modes"] != source["modes"]:
        raise ValueError("Neural dense modal modes/topology differ")
    if len(dense.manifest["views"]) != len(source["views"]):
        raise ValueError("Neural dense modal view count differs")
    for view, expected in zip(dense.manifest["views"], source["views"]):
        for name in ("index", "label", "shape_hw", "flow_identity"):
            if view[name] != expected[name]:
                raise ValueError(f"Neural dense modal view {name} differs")
    source.update(
        static_scene=str(Path(scene_dir).expanduser().resolve()),
        alignment_from=str(alignment.path), alignment_identity=alignment.manifest["rigid_modes_identity"],
        complex_2d_modes=str(dense.path), complex_2d_modes_identity=dense.manifest["complex_2d_modes_identity"],
    )
    return source, scene, topology, measurements, graph, alignment, dense


def _source_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    result = {name: source[name] for name in (
        "static_scene_identity", "foreground_identity", "topology_identity",
        "gaussian_measurements_identity", "observed_structure_graph_identity",
        "alignment_identity", "complex_2d_modes_identity", "modes", "views",
    )}
    result["modes"] = source.get("source_modes", source["modes"])
    return result


def _validated_mode_slots(slots: Sequence[int], count: int) -> list[int]:
    values = list(slots)
    if (not values or any(type(slot) is not int or not 0 <= slot < count for slot in values)
            or values != sorted(set(values))):
        raise ValueError("Selected source mode slots must be nonempty, unique, in range and in source order")
    return values


def _source_mode_slots(manifest: Mapping[str, Any]) -> np.ndarray:
    """Accept complete sources or explicit, hash-bound completed selections."""
    from modal_gaussians.motion.common.mode_mapping import resolve_source_mode_slots
    if ("source_modes" in manifest) != ("mode_selection" in manifest):
        raise ValueError("Neural prefix requires both source_modes and mode_selection")
    modes = manifest["modes"]
    if not isinstance(modes, list) or not modes:
        raise ValueError("Neural modes must be a nonempty list")
    if "mode_selection" not in manifest:
        if any("source_mode_slot" in mode for mode in modes):
            raise ValueError("Neural mapped modes require completed-prefix metadata")
        return np.arange(len(modes), dtype=np.int64)
    source_modes, selection = manifest["source_modes"], manifest["mode_selection"]
    if not isinstance(source_modes, list) or not isinstance(selection, dict):
        raise ValueError("Neural prefix metadata must contain source modes and a selection object")
    slots = resolve_source_mode_slots(modes, source_modes)
    policy = selection.get("policy")
    expected_slots = (list(range(len(modes))) if policy == "completed_prefix" else
                      _validated_mode_slots(selection.get("source_mode_slots", []), len(source_modes)))
    if (policy not in ("completed_prefix", "completed_selection")
            or selection.get("source_mode_slots") != expected_slots
            or any(type(slot) is not int for slot in selection.get("source_mode_slots", []))
            or slots.tolist() != expected_slots
            or any("source_mode_slot" not in mode for mode in modes)):
        raise ValueError("Neural export must select its declared completed source modes in order")

    def is_hash(value: Any) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    if not all(is_hash(selection.get(name)) for name in ("parent_run_identity", "parent_fixed_arrays_identity")):
        raise ValueError("Neural prefix parent identities are invalid")
    checkpoints = selection.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(modes):
        raise ValueError("Neural prefix checkpoint inventory differs")
    for slot, checkpoint in zip(expected_slots, checkpoints):
        if (not isinstance(checkpoint, dict) or type(checkpoint.get("source_mode_slot")) is not int
                or checkpoint["source_mode_slot"] != slot
                or checkpoint.get("filename") != f"mode_{slot:03d}.pt"
                or not is_hash(checkpoint.get("sha256"))):
            raise ValueError("Neural prefix checkpoint order or checksum is invalid")
    return slots


class FrozenModalProjector:
    """Differentiable four-channel rasterization; all geometric tensors are fixed."""

    def __init__(self, scene: Any, camera: Any, jacobian: torch.Tensor,
                 pixels: np.ndarray, foreground_alpha: torch.Tensor) -> None:
        self.scene, self.camera, self.jacobian = scene, camera, jacobian.detach()
        self.x = torch.as_tensor(pixels[:, 0], device=jacobian.device, dtype=torch.long)
        self.y = torch.as_tensor(pixels[:, 1], device=jacobian.device, dtype=torch.long)
        self.alpha = foreground_alpha.detach()

    def sample_features(self, features: torch.Tensor) -> torch.Tensor:
        image, alpha = self.scene.render_features(self.camera, features, composition="foreground")
        if not torch.allclose(alpha[self.y, self.x], self.alpha, rtol=2e-5, atol=2e-6):
            raise RuntimeError("Frozen neural feature rendering changed foreground alpha")
        return image[self.y, self.x] / self.alpha[:, None]

    def __call__(self, phi: torch.Tensor) -> torch.Tensor:
        real = torch.einsum("gij,gj->gi", self.jacobian, phi.real)
        imaginary = torch.einsum("gij,gj->gi", self.jacobian, phi.imag)
        values = self.sample_features(torch.cat((real, imaginary), dim=-1))
        return torch.complex(values[:, :2], values[:, 2:])

    def contribution_mass(self) -> torch.Tensor:
        dummy = torch.zeros((len(self.jacobian), 1), device=self.jacobian.device,
                            dtype=self.jacobian.dtype, requires_grad=True)
        mass, = torch.autograd.grad(self.sample_features(dummy).sum(), dummy)
        if not torch.isfinite(mass).all() or bool((mass < -1e-7).any()):
            raise RuntimeError("Invalid frozen foreground contribution sensitivity")
        return mass[:, 0].detach().clamp_min(0)

    def projection_sensitivity(self) -> torch.Tensor:
        with torch.no_grad():
            return self.sample_features(self.jacobian.square().sum(dim=(1, 2))[:, None])[:, 0]


def support_roles(component_index: np.ndarray, observation_view_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """0 unresolved, 1 direct image support, 2 inferred in a supported component."""
    components = np.asarray(component_index, dtype=np.int64)
    observed = np.asarray(observation_view_mask, dtype=bool)
    if components.ndim != 1 or observed.ndim != 3 or observed.shape[1] != len(components):
        raise ValueError("Neural support domains differ")
    if len(components) == 0 or np.any(components < 0):
        raise ValueError("Neural components must be nonempty and nonnegative")
    roles = np.zeros(observed.shape[:2], dtype=np.int8)
    for mode in range(len(observed)):
        directly_observed = observed[mode].any(axis=1)
        supported_components = np.unique(components[directly_observed])
        roles[mode, np.isin(components, supported_components)] = 2
        roles[mode, directly_observed] = 1
    return roles, roles != 0


def _field_geometry(arrays: Mapping[str, np.ndarray], mode: int, device: Any = "cpu") -> Any:
    from modal_gaussians.motion.neural.neural_field import NeuralFieldGeometry
    supported = np.asarray(arrays["support_class"])[mode] != 0
    training_fill = "t_host_gaussian_index" in arrays
    interpolation_prefix = "t_" if training_fill else "c_"
    control_indices = arrays["c_control_point_index"]
    if training_fill:
        control_indices = arrays["t_host_gaussian_index"][control_indices]
    values = {
        "gaussian_positions": arrays["g_points"], "control_positions": arrays["c_positions"],
        "gaussian_edges": arrays["g_edge_index"], "gaussian_edge_weights": arrays["g_edge_weight"],
        "control_edges": arrays["c_control_edges"], "control_edge_weights": arrays["c_control_edge_weight"],
        "control_edge_lengths": arrays["c_control_edge_length"],
        "interpolation_indptr": arrays[interpolation_prefix + "interpolation_indptr"],
        "interpolation_indices": arrays[interpolation_prefix + "interpolation_indices"],
        "interpolation_weights": arrays[interpolation_prefix + "interpolation_weights"],
        "allow_empty_interpolation": training_fill,
        "gaussian_supported": supported,
        "control_supported": supported[control_indices],
    }
    if "p_source_mask" in arrays:
        source_mask = arrays["p_source_mask"][mode]
        values["interpolation_supported"] = source_mask
        values["control_supported"] = source_mask[control_indices]
        edges = arrays["g_edge_index"]
        keep = source_mask[edges].all(axis=1)
        values["gaussian_edges"], values["gaussian_edge_weights"] = edges[keep], arrays["g_edge_weight"][keep]
        weights = arrays["p_neighbor_weight"][mode]
        rows, slots = np.nonzero(weights > 0)
        values.update(transfer_rows=rows, transfer_sources=arrays["p_neighbor_index"][mode][rows, slots],
                      transfer_weights=weights[rows, slots])
    if "h_source_mask" in arrays:
        learning = arrays["h_source_mask"][mode] | arrays["h_residual_mask"][mode]
        values["interpolation_supported"] = learning
        values["control_supported"] = arrays["h_control_supported"][mode]
        edges = arrays["g_edge_index"]
        keep = learning[edges].all(axis=1)
        values["gaussian_edges"], values["gaussian_edge_weights"] = edges[keep], arrays["g_edge_weight"][keep]
        weights = arrays["h_neighbor_weight"][mode]
        rows, slots = np.nonzero(weights > 0)
        values.update(transfer_rows=rows, transfer_sources=arrays["h_neighbor_index"][mode][rows, slots],
                      transfer_weights=weights[rows, slots], residual_projector=arrays["h_residual_projector"][mode],
                      residual_mask=arrays["h_residual_mask"][mode],
                      residual_prior_weight=float(arrays["h_residual_prior_weight"]))
    if "u_own_field_mask" in arrays:
        own = arrays["u_own_field_mask"][mode]
        values["interpolation_supported"] = own
        values["control_supported"] = own[control_indices]
        # Whole learning components retain every geometric edge, including
        # edges joining directly supervised and structurally inferred members.
        edges = arrays["g_edge_index"]
        keep = own[edges].all(axis=1)
        values["gaussian_edges"], values["gaussian_edge_weights"] = edges[keep], arrays["g_edge_weight"][keep]
        weights = arrays["u_neighbor_weight"][mode]
        rows, slots = np.nonzero(weights > 0)
        values.update(transfer_rows=rows, transfer_sources=arrays["u_neighbor_index"][mode][rows, slots],
                      transfer_weights=weights[rows, slots])
    return NeuralFieldGeometry.from_arrays(values, device=device)


def _artifact_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = {name: manifest[name] for name in (
        "format", "version", "completion_method", "source_identity", "config", "runtime",
        "run_identity", "semantics", "quality_gate", "geometry_graph", "counts",
        "arrays_identity", "networks_sha256", "optimization", "diagnostics",
    )}
    result.update({name: manifest[name] for name in ("source_modes", "mode_selection") if name in manifest})
    return result


SEMANTICS = {
    "field": "sum_a N_ia * (d_a + cross(omega_a, x_i-c_a))",
    "parameters": "independent_network_per_frequency_fixed_geometry_interpolation",
    "alignment": "fixed_source_complex_alpha_reference_view_one",
    "observation": "dense_modal_image_full_static_foreground_feature_render",
    "complex_packing": "u_real,v_real,u_imag,v_imag",
    "units": "source_unnormalized_exact_DFT_not_physical_playback_amplitude",
    "roles": {"0": "unresolved", "1": "directly_image_supervised", "2": "structure_inferred"},
    "support": "selected_pixel_feature_mass_exceeds_max(1e-12,1e-8*view_maximum)",
    "rigid_source_use": "alpha_and_identifiability_only_no_motion_no_trust",
}


def _semantics(config: NeuralModesConfig) -> dict[str, Any]:
    if config.training_fragment_config is None and config.local_feature_dim == 0:
        return SEMANTICS
    result = dict(SEMANTICS)
    if config.local_feature_dim:
        result["control_features"] = {
            "dimension": config.local_feature_dim,
            "parameters": "per_frequency_per_control_in_saved_control_order",
            "input": "normalized_coordinates_and_learned_features",
            "head": "graph_features_coordinates_and_direct_local_features",
            "initialization": "normal_mean_0_std_0.01_frequency_seed; zero_output_head",
        }
    if config.training_fragment_config is None:
        return result
    result.update({
            "parameters": "host_controls_only_fixed_fragment_interpolation_before_loss",
            "fragment_transfer": "N_fragment=sum_anchor beta_anchor*N_anchor; no independent fragment controls",
            "roles": {**SEMANTICS["roles"], "3": "fragment_propagated_during_training"},
            "fragment_support": "host_and_attached_fragments_share_image_support; unattached_zero_unresolved"})
    if config.training_fragment_config.get("strategy") == "surface":
        result.update(fragment_transfer="per_gaussian_surface_anchor_weights_smoothed_on_fragment_graph; N=A*N_host",
                      attachment_support="capped_center_distance_and_projected_covariance_support_gap; unknown_visibility_not_veto",
                      attachment_classification="min_host_nodes_and_provisional_control_count_at_fixed_h",
                      attachment_sources="static_covariance_and_frozen_visibility_depth; no_rgb_no_rigid_motion")
    if config.training_fragment_config.get("strategy") == "pointwise":
        result.update(fragment_transfer="sum_neighbor weight*Phi_neighbor; no_rotation_extrapolation_no_cascades",
                      attachment_classification="component_size_only_single_control_allowed",
                      attachment_support="per_mode_nearest_supported_host_no_distance_gap_or_ambiguity_cutoff",
                      structure_loss="independent_host_edges_only; followers_are_individual_points")
    if config.training_fragment_config.get("strategy") == "guarded":
        result.update(
            parameters="stable_and_qualified_small_component_controls; small_controls_predict_residuals",
            fragment_transfer="sum_stable_neighbor weight*Phi_neighbor; no_rotation_extrapolation_no_cascades",
            attachment_classification="stable_min_size_and_provisional_controls; small_min_size_and_reliable_support",
            attachment_support="visible_surface_with_minimum_feature_mass; nearest_stable_component_no_distance_cutoff",
            residual="Phi=neighbor_prior+observable_projector*raw_field; nullspace_inherits_neighbor",
            residual_prior="mean_squared_residual_over_fixed_amplitude_scale; stable_motion_unpenalized",
            fragment_support="pure_followers_do_not_create_independent_sources; their_images_supervise_shared_donors",
            structure_loss="reliable_learning_edges_only",
            roles={"0": "unresolved", "1": "reliable_direct_or_neighbor_regularized", "3": "stable_neighbor_propagated"})
    if config.training_fragment_config.get("strategy") == "component_field":
        result.update(
            parameters="complete_observed_components_use_own_control_field; donor_eligibility_is_separate",
            fragment_transfer="sum_stable_neighbor weight*Phi_neighbor; no_rotation_extrapolation_no_cascades",
            attachment_classification="own_field_component_size_and_any_effective_component_observation",
            attachment_support="donors_only: min_component_size_controls_reliable_count_and_point_visibility_mass",
            fragment_support="whole_unobserved_components_and_tiny_components_propagate; no_donor_zero_unresolved",
            structure_loss="all_geometry_edges_within_own_field_components",
            observation_refinement="disabled",
            roles={**SEMANTICS["roles"], "3": "stable_neighbor_propagated"})
        if config.training_fragment_config.get("min_learning_controls", 1) > 1:
            result["attachment_classification"] = "own_field_component_size_minimum_controls_and_any_effective_component_observation"
            result["control_eligibility"] = "count_at_fixed_coverage_radius_then_remove_whole_ineligible_components_from_network"
    return result


def _set_support_roles(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if any(name in arrays for name in ("t_motion_component_index", "p_source_mask", "h_source_mask", "u_own_field_mask")):
        from .training_fragments import training_support_roles
        roles, _ = training_support_roles(arrays, arrays["observation_view_mask"])
    else:
        roles, _ = support_roles(arrays["g_component_index"], arrays["observation_view_mask"])
    arrays["support_class"] = roles
    return roles


def _prepare_observation_arrays(scene: Any, source: Mapping[str, Any], dense: Any,
                                config: NeuralModesConfig, device: torch.device,
                                alphas: np.ndarray, identifiable: np.ndarray,
                                frozen_arrays: Mapping[str, np.ndarray] | None = None,
                                flow_loader: Any = None,
                                ) -> tuple[dict[str, np.ndarray], list[FrozenModalProjector], list[Any], list[np.ndarray], list[np.ndarray]]:
    from modal_gaussians.flow.artifact import load_flow_analysis_artifact, flow_artifact_identity
    from modal_gaussians.motion.common.projection import (
        RenderedDesignConfig,
        candidate_pixels,
        projection_jacobian,
    )
    from modal_gaussians.static import cameras_from_scene_manifest

    scene.to(device).eval()
    for parameter in scene.parameters():
        parameter.requires_grad_(False)
    references = {c.label: c for c in cameras_from_scene_manifest(scene.manifest) if c.role == "reference"}
    points = scene.foreground.active()["means"].detach().cpu().numpy()
    pixels_all, confidence_all, views_all, targets_all = [], [], [], []
    projectors, cameras, depths, alpha_images = [], [], [], []
    masses, sensitivities, offsets = [], [], [0]
    sampling = RenderedDesignConfig(pixel_sample_stride=config.pixel_sample_stride,
                                    alpha_minimum=config.alpha_minimum,
                                    mask_erosion_iterations=config.mask_erosion_iterations)
    for view, dense_view, dense_values in zip(source["views"], dense.manifest["views"], dense.view_modes):
        camera = references.get(view["label"])
        if camera is None or camera.to_manifest_record()["camera_identity"] != view["camera_identity"]:
            raise ValueError("Neural reference camera identity differs")
        camera = camera.to(device)
        flow = (flow_loader or load_flow_analysis_artifact)(dense_view["flow_artifact"])
        if flow_artifact_identity(flow) != view["flow_identity"]:
            raise ValueError("Neural sampling flow identity differs")
        if list(flow.arrays.mask_union.shape) != [camera.height, camera.width]:
            raise ValueError("Neural flow mask shape differs from camera")
        with torch.no_grad():
            render = scene.render(camera, composition="foreground", outputs=("alpha", "expected_depth"))
        alpha_image = render["alpha"].cpu().numpy().astype(np.float32)
        depth_image = render["expected_depth"].cpu().numpy().astype(np.float32)
        if frozen_arrays is None:
            pixels, confidence = candidate_pixels(flow.arrays.mask_union, alpha_image, sampling)
        else:
            lo, hi = frozen_arrays["view_sample_offsets"][view["index"]:view["index"] + 2]
            pixels = frozen_arrays["sample_pixels_xy"][lo:hi]
            confidence = frozen_arrays["sample_confidence"][lo:hi]
            if not np.allclose(alpha_image[pixels[:, 1], pixels[:, 0]], confidence, rtol=2e-5, atol=2e-6):
                raise ValueError("Resumed static foreground alpha differs from frozen observation inputs")
        jacobian, _ = projection_jacobian(points, camera.K.cpu().numpy(), camera.world_to_camera.cpu().numpy(), camera.radial_distortion)
        projector = FrozenModalProjector(scene, camera, torch.as_tensor(jacobian, device=device),
                                         pixels, torch.as_tensor(confidence, device=device))
        if frozen_arrays is None:
            masses.append(projector.contribution_mass().cpu().numpy())
            sensitivities.append(projector.projection_sensitivity().cpu().numpy())
        projectors.append(projector)
        cameras.append(camera)
        depths.append(depth_image)
        alpha_images.append(alpha_image)
        pixels_all.append(pixels)
        confidence_all.append(confidence)
        views_all.append(np.full(len(pixels), view["index"], dtype=np.int64))
        targets_all.append(np.asarray(dense_values[:, pixels[:, 1], pixels[:, 0], :], dtype=np.complex64))
        offsets.append(offsets[-1] + len(pixels))
    target = np.concatenate(targets_all, axis=1)
    if not np.isfinite(target).all():
        raise ValueError("Neural sampled modal targets are non-finite")
    if frozen_arrays is not None:
        if not np.array_equal(target, frozen_arrays["sample_target"]):
            raise ValueError("Resumed dense modal targets differ from frozen observations")
        if not np.array_equal(alphas, frozen_arrays["alphas"]) or not np.array_equal(identifiable, frozen_arrays["alpha_identifiable_mask"]):
            raise ValueError("Resumed complex alignment differs from frozen observations")
        if not np.array_equal(points, frozen_arrays["g_points"]):
            raise ValueError("Resumed static points differ from frozen geometry")
        # CUDA feature-gradient atomic reductions are not bitwise deterministic.
        # Their persisted values, including roles and scales, are authoritative.
        return dict(frozen_arrays), projectors, cameras, depths, alpha_images
    confidence = np.concatenate(confidence_all)
    mode_count, view_count = len(target), len(projectors)
    rms = np.zeros((mode_count, view_count), dtype=np.float64)
    for view in range(view_count):
        lower, upper = offsets[view:view + 2]
        weights = confidence[lower:upper].astype(np.float64)
        energy = np.sum(np.abs(target[:, lower:upper].astype(np.complex128)) ** 2, axis=-1)
        rms[:, view] = np.sqrt(np.sum(energy * weights[None], axis=1) / weights.sum())
    positive = rms[identifiable & (rms > 0)]
    floor = max(config.energy_floor_fraction * (float(np.median(positive)) if len(positive) else 1.0), 1e-12)
    masses_value = np.stack(masses, axis=1).astype(np.float32)
    mass_threshold = np.maximum(1e-12, masses_value.max(axis=0) * 1e-8)
    visible = masses_value > mass_threshold[None]
    observation = visible[None] & identifiable[:, None, :]
    L = float(np.linalg.norm(np.ptp(points.astype(np.float64), axis=0)))
    if not math.isfinite(L) or L <= 0:
        raise ValueError("Neural foreground geometry has no positive spatial extent")
    amplitude_scales = np.empty(mode_count, dtype=np.float64)
    sensitivity_means = np.array([np.average(s, weights=c) for s, c in zip(sensitivities, confidence_all)])
    for mode in range(mode_count):
        valid = identifiable[mode]
        if not np.any(valid):
            raise ValueError(f"Neural mode {mode} has no identifiable supervision view")
        denominator = float(np.sum(np.abs(alphas[mode, valid].astype(np.complex128)) ** 2 * sensitivity_means[valid]))
        if denominator <= 0 or not math.isfinite(denominator):
            raise ValueError(f"Neural mode {mode} has zero projection sensitivity")
        numerator = float(np.sum(rms[mode, valid] ** 2))
        amplitude_scales[mode] = max(1e-6 * L, math.sqrt(numerator / denominator))
    arrays = {
        "alphas": np.asarray(alphas, dtype=np.complex64).copy(),
        "alpha_identifiable_mask": np.asarray(identifiable, dtype=bool).copy(),
        "observation_view_mask": observation,
        "sample_pixels_xy": np.concatenate(pixels_all).astype(np.int64),
        "sample_view_index": np.concatenate(views_all),
        "view_sample_offsets": np.asarray(offsets, dtype=np.int64),
        "sample_confidence": confidence.astype(np.float32),
        "sample_target": target,
        "mode_view_rms": rms.astype(np.float64),
        "mode_view_loss_scale": np.maximum(rms, floor).astype(np.float64),
        "measurement_rms_floor": np.asarray(floor, dtype=np.float64),
        "amplitude_scale": amplitude_scales,
        "scene_scale": np.asarray(L, dtype=np.float64),
        "contribution_mass": masses_value,
        "contribution_threshold": mass_threshold.astype(np.float32),
        "sample_projection_sensitivity": np.concatenate(sensitivities).astype(np.float32),
    }
    return arrays, projectors, cameras, depths, alpha_images


def _validate_arrays(arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]) -> None:
    """Check domains, support semantics and sparse geometry before loading weights."""
    from modal_gaussians.motion.neural.geometry_graph import GeometryGraph, ControlGraph
    config = NeuralModesConfig.from_dict(manifest["config"])
    training_fill = config.training_fragment_config is not None
    required = {
        "phi", "sample_prediction", "support_class", "alphas", "alpha_identifiable_mask",
        "observation_view_mask", "sample_pixels_xy", "sample_view_index", "view_sample_offsets",
        "sample_confidence", "sample_target", "mode_view_rms", "mode_view_loss_scale",
        "measurement_rms_floor", "amplitude_scale", "scene_scale", "contribution_mass",
        "contribution_threshold", "sample_projection_sensitivity",
    }
    required.update("g_" + f.name for f in fields(GeometryGraph))
    required.update("c_" + f.name for f in fields(ControlGraph))
    if training_fill:
        from .training_fragments import array_names
        required.update(array_names(config.training_fragment_config))
    selected_slots = None
    if "source_modes" in manifest or "mode_selection" in manifest:
        selected_slots = _source_mode_slots(manifest)
        required.update(PREFIX_NORMALIZATION_ARRAYS)
    if set(arrays) != required:
        raise ValueError("Neural array inventory does not match its schema")
    K, G, V = (int(manifest["counts"][name]) for name in ("modes", "foreground_gaussians", "views"))
    if min(K, G, V) <= 0:
        raise ValueError("Neural artifact counts must be positive")
    if selected_slots is not None and len(selected_slots) != K:
        raise ValueError("Neural prefix mode count differs from selected slots")
    for name, value in arrays.items():
        if value.dtype.kind not in "biufc" or (value.dtype.kind in "fc" and not np.isfinite(value).all()):
            raise ValueError(f"Neural array {name} must be finite numeric data")
    shapes = {"phi": (K, G, 3), "support_class": (K, G), "observation_view_mask": (K, G, V),
              "alphas": (K, V), "alpha_identifiable_mask": (K, V), "amplitude_scale": (K,),
              "mode_view_rms": (K, V), "mode_view_loss_scale": (K, V),
              "g_points": (G, 3), "g_component_index": (G,), "contribution_mass": (G, V),
              "view_sample_offsets": (V + 1,), "contribution_threshold": (V,)}
    for name, shape in shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError(f"Neural {name} shape must be {shape}")
    if arrays["phi"].dtype != np.complex64 or arrays["alphas"].dtype != np.complex64:
        raise ValueError("Neural complex arrays must be complex64")
    for name in ("sample_target", "sample_prediction"):
        if arrays[name].dtype != np.complex64:
            raise ValueError(f"Neural {name} must be complex64")
    for name in ("observation_view_mask", "alpha_identifiable_mask"):
        if arrays[name].dtype != np.bool_:
            raise ValueError(f"Neural {name} must be boolean")
    if arrays["support_class"].dtype != np.int8 or not np.isin(arrays["support_class"], (0, 1, 2, 3) if training_fill else (0, 1, 2)).all():
        raise ValueError("Neural support classes are invalid")
    if "a_view_visible" in arrays and arrays["a_view_visible"].shape != (G, V):
        raise ValueError("Surface attachment view domain differs from observations")
    if not np.all(arrays["alphas"][:, 0] == 1) or not arrays["alpha_identifiable_mask"][:, 0].all():
        raise ValueError("Neural reference alpha gauge must remain one and identifiable")
    if arrays["scene_scale"].shape != () or arrays["measurement_rms_floor"].shape != ():
        raise ValueError("Neural scalar normalization arrays have invalid shapes")
    if np.any(arrays["amplitude_scale"] <= 0) or float(arrays["scene_scale"]) <= 0:
        raise ValueError("Neural normalization scales must be positive")
    if np.any(arrays["mode_view_loss_scale"] <= 0):
        raise ValueError("Neural observation scales must be positive")
    expected_obs = ((arrays["contribution_mass"] > arrays["contribution_threshold"][None])[None]
                    & arrays["alpha_identifiable_mask"][:, None, :])
    if not np.array_equal(arrays["observation_view_mask"], expected_obs):
        raise ValueError("Neural observation support differs from renderer contribution")
    if training_fill:
        from .training_fragments import training_support_roles
        roles, supported = training_support_roles(arrays, expected_obs)
    else:
        roles, supported = support_roles(arrays["g_component_index"], expected_obs)
    if not np.array_equal(roles, arrays["support_class"]):
        raise ValueError("Neural support classes disagree with geometry connectivity")
    if np.any(arrays["phi"][~supported] != 0):
        raise ValueError("Unresolved neural components must have exactly zero motion")
    offsets = arrays["view_sample_offsets"]
    if offsets.dtype != np.int64 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("Neural observation offsets are invalid")
    S = int(offsets[-1])
    for name, shape in {"sample_target": (K, S, 2), "sample_prediction": (K, S, 2),
                        "sample_confidence": (S,), "sample_view_index": (S,),
                        "sample_pixels_xy": (S, 2), "sample_projection_sensitivity": (S,)}.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Neural {name} observation shape differs")
    if np.any(arrays["sample_confidence"] <= 0):
        raise ValueError("Neural sample confidence must be positive")
    if np.any(arrays["sample_confidence"] > 1) or np.any(arrays["contribution_mass"] < 0):
        raise ValueError("Neural alpha confidence or contribution mass is invalid")
    if np.any(arrays["sample_projection_sensitivity"] < 0):
        raise ValueError("Neural projection sensitivity must be nonnegative")
    config = NeuralModesConfig.from_dict(manifest["config"])
    expected_rms = np.zeros((K, V), dtype=np.float64)
    for view in range(V):
        lower, upper = offsets[view:view + 2]
        if np.any(arrays["sample_view_index"][lower:upper] != view):
            raise ValueError("Neural sample view order differs")
        pixels = arrays["sample_pixels_xy"][lower:upper]
        height, width = manifest["views"][view]["shape_hw"]
        if pixels.dtype != np.int64 or np.any(pixels < 0) or np.any(pixels >= [width, height]):
            raise ValueError("Neural sampled pixels are out of bounds")
        confidence = arrays["sample_confidence"][lower:upper].astype(np.float64)
        energy = np.sum(np.abs(arrays["sample_target"][:, lower:upper].astype(np.complex128)) ** 2, axis=-1)
        expected_rms[:, view] = np.sqrt(np.sum(energy * confidence[None], axis=1) / confidence.sum())
        if np.any(arrays["sample_prediction"][~arrays["alpha_identifiable_mask"][:, view], lower:upper] != 0):
            raise ValueError("Excluded neural views must have explicit zero predictions")
    normalization_rms, normalization_identifiable = expected_rms, arrays["alpha_identifiable_mask"]
    if selected_slots is not None:
        normalization_rms = arrays["normalization_source_mode_view_rms"]
        normalization_identifiable = arrays["normalization_source_alpha_identifiable_mask"]
        original_shape = (len(manifest["source_modes"]), V)
        if (normalization_rms.shape != original_shape or normalization_rms.dtype != np.float64
                or normalization_identifiable.shape != original_shape or normalization_identifiable.dtype != np.bool_
                or np.any(normalization_rms < 0) or not normalization_identifiable[:, 0].all()
                or not np.array_equal(normalization_identifiable[selected_slots], arrays["alpha_identifiable_mask"])
                or not np.allclose(normalization_rms[selected_slots], expected_rms, rtol=1e-10, atol=1e-12)):
            raise ValueError("Neural prefix normalization differs from original source mode scope")
    positive = normalization_rms[normalization_identifiable & (normalization_rms > 0)]
    expected_floor = max(config.energy_floor_fraction * (float(np.median(positive)) if len(positive) else 1.0), 1e-12)
    if (not np.allclose(arrays["mode_view_rms"], expected_rms, rtol=1e-10, atol=1e-12)
            or not math.isclose(float(arrays["measurement_rms_floor"]), expected_floor, rel_tol=1e-10)
            or not np.allclose(arrays["mode_view_loss_scale"], np.maximum(expected_rms, expected_floor), rtol=1e-10, atol=1e-12)):
        raise ValueError("Neural observation normalization differs from fixed measurements")
    graph = GeometryGraph.from_dict({name[2:]: value for name, value in arrays.items() if name.startswith("g_")})
    if config.graph_edge_filter == "none":
        from modal_gaussians.motion.neural.geometry_graph import EVIDENCE_SPATIAL_PRIOR, _mutual_knn
        expected_edges, _ = _mutual_knn(graph.points.astype(np.float64), _geometry_config(config))
        expected_weights = 1.0 / np.sqrt(graph.degree[expected_edges[:, 0]].astype(np.float64) * graph.degree[expected_edges[:, 1]])
        if (not np.array_equal(graph.edge_index, expected_edges)
                or not np.array_equal(graph.candidate_edge_index, expected_edges)
                or not np.all(graph.edge_evidence_kind == EVIDENCE_SPATIAL_PRIOR)
                or np.any(graph.edge_view_evidence) or np.any(graph.candidate_view_evidence)
                or np.any(graph.node_visible_view_mask)
                or not np.allclose(graph.edge_weight, expected_weights, rtol=1e-12, atol=1e-14)):
            raise ValueError("Neural KNN-only graph differs from its complete spatial candidate contract")
    elif np.any(graph.edge_evidence_kind == 2):
        raise ValueError("Depth-filtered neural configuration cannot contain unfiltered spatial-prior edges")
    control = ControlGraph.from_dict({name[2:]: value for name, value in arrays.items() if name.startswith("c_")})
    control_graph, control_count = graph, G
    if training_fill:
        from .training_fragments import validate_training_controls, host_subgraph
        validate_training_controls(arrays, graph, geometry_config=_geometry_config(config),
                                   fragment_config=config.training_fragment_config,
                                   scene_scale=float(arrays["scene_scale"]))
        control_graph = host_subgraph(graph, arrays["t_host_gaussian_index"])
        control_count = len(control_graph.points)
    if (len(control.interpolation_indptr) != control_count + 1 or len(control.owner) != control_count
            or np.any(control.control_point_index < 0) or np.any(control.control_point_index >= control_count)):
        raise ValueError("Neural control and Gaussian index domains differ")
    if not np.array_equal(control.positions, control_graph.points[control.control_point_index]):
        raise ValueError("Neural control locations differ from foreground indices")
    if not np.array_equal(control.component_index, control_graph.component_index[control.control_point_index]):
        raise ValueError("Neural control components differ from Gaussian geometry")
    interpolation_rows = np.repeat(np.arange(control_count), np.diff(control.interpolation_indptr))
    if np.any(control_graph.component_index[interpolation_rows] != control.component_index[control.interpolation_indices]):
        raise ValueError("Neural interpolation crosses disconnected geometry")
    if not math.isclose(float(control.scene_scale), float(arrays["scene_scale"]), rel_tol=1e-7):
        raise ValueError("Neural geometry length scales differ")
    view_sensitivity = np.array([
        np.average(arrays["sample_projection_sensitivity"][offsets[v]:offsets[v + 1]],
                   weights=arrays["sample_confidence"][offsets[v]:offsets[v + 1]]) for v in range(V)
    ])
    for mode in range(K):
        valid = arrays["alpha_identifiable_mask"][mode]
        denominator = float(np.sum(np.abs(arrays["alphas"][mode, valid].astype(np.complex128)) ** 2 * view_sensitivity[valid]))
        if denominator <= 0 or not np.any(expected_obs[mode].any(axis=1) & supported[mode]):
            raise ValueError("Neural mode has no effective observation sensitivity")
        expected_scale = max(1e-6 * float(arrays["scene_scale"]), math.sqrt(float(np.sum(expected_rms[mode, valid] ** 2)) / denominator))
        if not math.isclose(float(arrays["amplitude_scale"][mode]), expected_scale, rel_tol=1e-7):
            raise ValueError("Neural amplitude scale differs from fixed projection sensitivity")
        _field_geometry(arrays, mode)


def _check_persisted_sources(manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    source, scene, _, _, _, alignment, dense = _load_sources(
        scene_dir=manifest["static_scene"], topology_dir=manifest["topology"],
        measurements_dir=manifest["measurements"], graph_dir=manifest["observed_structure_graph"],
        alignment_from=manifest["alignment_from"],
    )
    if _source_identity(source) != manifest["source_identity"]:
        raise ValueError("Neural artifact source identities differ")
    slots = _source_mode_slots(manifest)
    if "source_modes" in manifest and manifest["source_modes"] != source["modes"]:
        raise ValueError("Neural prefix original source modes differ")
    if "source_modes" not in manifest and manifest["modes"] != source["modes"]:
        raise ValueError("Neural complete source modes differ")
    for name in ("alphas", "alpha_identifiable_mask"):
        if not np.array_equal(arrays[name], alignment.arrays[name][slots]):
            raise ValueError(f"Neural fixed {name} differs from alignment source")
    if "source_modes" in manifest and not np.array_equal(
            arrays["normalization_source_alpha_identifiable_mask"], alignment.arrays["alpha_identifiable_mask"]):
        raise ValueError("Neural prefix normalization identifiability differs from alignment source")
    points = scene.foreground.active()["means"].detach().cpu().numpy()
    if not np.array_equal(points, arrays["g_points"]):
        raise ValueError("Neural foreground point order differs from static scene")
    if "a_covariance" in arrays:
        from .surface_attachments import scene_covariance
        # Float32 CPU/CUDA activations differ by a few ulps. Off-diagonal
        # covariance entries can nearly cancel, so scale error per matrix,
        # not relative to those near-zero entries.
        expected_covariance = scene_covariance(scene)
        covariance_error = np.linalg.norm(expected_covariance - arrays["a_covariance"], axis=(1, 2))
        covariance_scale = np.linalg.norm(expected_covariance, axis=(1, 2))
        if np.any(covariance_error > 2e-6 * covariance_scale + 1e-15):
            raise ValueError("Surface attachment covariance differs from static scene")
        visible = arrays["contribution_mass"] > arrays["contribution_threshold"][None]
        if np.any(arrays["a_view_visible"] & ~visible):
            raise ValueError("Surface attachment visibility has no observed contribution")
    for view, modes in enumerate(dense.view_modes):
        lo, hi = arrays["view_sample_offsets"][view:view + 2]
        pixels = arrays["sample_pixels_xy"][lo:hi]
        target = np.asarray(modes[:, pixels[:, 1], pixels[:, 0], :])
        if not np.array_equal(target[slots], arrays["sample_target"][:, lo:hi]):
            raise ValueError("Neural target differs from dense modal image source")
        if "source_modes" in manifest:
            confidence = arrays["sample_confidence"][lo:hi].astype(np.float64)
            energy = np.sum(np.abs(target.astype(np.complex128)) ** 2, axis=-1)
            rms = np.sqrt(np.sum(energy * confidence[None], axis=1) / confidence.sum())
            if not np.allclose(rms, arrays["normalization_source_mode_view_rms"][:, view], rtol=1e-10, atol=1e-12):
                raise ValueError("Neural prefix normalization RMS differs from full dense modal source")


def load_neural_completed_modes(path: str | Path) -> NeuralModesArtifact:
    """Load v8/v10/v11/v12/v14 and reproduce baked phi from saved network weights."""
    from modal_gaussians.motion.neural.neural_field import evaluate_model
    root = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != COMPLETED_MODES_FORMAT or type(manifest.get("version")) is not int
            or manifest["version"] not in (8, 10, 11, 12, 14, 16)):
        raise ValueError("Unsupported neural completed-mode artifact")
    config = NeuralModesConfig.from_dict(manifest["config"])
    from .training_fragments import artifact_contract
    training_fill = config.training_fragment_config is not None
    version, method = artifact_contract(config.training_fragment_config)
    if manifest["version"] != version or manifest.get("completion_method") != method:
        raise ValueError("Unsupported neural artifact version/method/configuration combination")
    if manifest.get("semantics") != _semantics(config) or manifest.get("quality_gate") != QUALITY_GATE:
        raise ValueError("Neural completed-mode semantics differ")
    if manifest.get("arrays_file") != ARRAYS_FILENAME or manifest.get("networks_file") != MODELS_FILENAME:
        raise ValueError("Neural artifact file names differ")
    if _sha256(root / ARRAYS_FILENAME) != manifest["arrays_file_sha256"] or _sha256(root / MODELS_FILENAME) != manifest["networks_sha256"]:
        raise ValueError("Neural artifact file checksum differs")
    with np.load(root / ARRAYS_FILENAME, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if manifest.get("arrays") != {name: {"dtype": a.dtype.name, "shape": list(a.shape)} for name, a in arrays.items()}:
        raise ValueError("Neural array inventory differs")
    if _arrays_identity(arrays) != manifest["arrays_identity"]:
        raise ValueError("Neural array identity differs")
    if manifest["source_identity"] != _source_identity(manifest):
        raise ValueError("Neural source metadata differs")
    if _identity(_artifact_identity_payload(manifest)) != manifest.get("completed_modes_identity"):
        raise ValueError("Neural completed-mode identity differs")
    fixed_arrays = {name: value for name, value in arrays.items() if name not in ("phi", "sample_prediction")}
    slots = _source_mode_slots(manifest)
    if len(slots) != manifest["counts"]["modes"]:
        raise ValueError("Neural mode inventory differs from artifact count")
    run_contract = {"format": "modal_gaussians.neural_modes_work", "version": 1,
                    "source_identity": manifest["source_identity"], "config": manifest["config"],
                    "runtime": manifest["runtime"], "geometry_graph": manifest["geometry_graph"],
                    "fixed_arrays_identity": _arrays_identity(fixed_arrays)}
    if "mode_selection" in manifest:
        selection = manifest["mode_selection"]
        parent_contract = {**run_contract, "fixed_arrays_identity": selection["parent_fixed_arrays_identity"]}
        if _identity(parent_contract) != selection["parent_run_identity"]:
            raise ValueError("Neural prefix parent run identity differs from original configuration and sources")
        run_contract["mode_selection"] = selection
    expected_run = _identity(run_contract)
    if expected_run != manifest["run_identity"]:
        raise ValueError("Neural run identity differs from resolved inputs and configuration")
    _validate_arrays(arrays, manifest)
    _check_persisted_sources(manifest, arrays)
    networks = torch.load(root / MODELS_FILENAME, map_location="cpu", weights_only=True)
    if networks.get("run_identity") != manifest["run_identity"] or len(networks.get("model_states", [])) != len(manifest["modes"]):
        raise ValueError("Neural network inventory/run identity differs")
    if "mode_selection" in manifest and networks.get("source_mode_slots") != slots.tolist():
        raise ValueError("Neural network source mode order differs from completed prefix")
    for mode, state in enumerate(networks["model_states"]):
        evaluated = evaluate_model(state, _field_geometry(arrays, mode), length_scale=float(arrays["scene_scale"]),
                                   amplitude_scale=float(arrays["amplitude_scale"][mode]), config=_field_config(config, int(slots[mode])))
        reproduced = evaluated[0].detach().cpu().numpy()
        if not np.allclose(reproduced, arrays["phi"][mode], rtol=3e-5, atol=1e-7 * float(arrays["amplitude_scale"][mode])):
            raise ValueError(f"Neural baked phi differs from network for mode {mode}")
    return NeuralModesArtifact(root, manifest, arrays)


def _mode_diagnostics(arrays: Mapping[str, np.ndarray], mode: int) -> dict[str, Any]:
    offsets = arrays["view_sample_offsets"]
    records = []
    total_error, total_energy = 0.0, 0.0
    for view in range(len(offsets) - 1):
        if not arrays["alpha_identifiable_mask"][mode, view]:
            records.append({"view_index": view, "supervised": False})
            continue
        lo, hi = offsets[view:view + 2]
        target = arrays["sample_target"][mode, lo:hi].astype(np.complex128)
        residual = arrays["sample_prediction"][mode, lo:hi] - target
        w = arrays["sample_confidence"][lo:hi].astype(np.float64)
        error = float(np.sum(w[:, None] * np.abs(residual) ** 2))
        energy = float(np.sum(w[:, None] * np.abs(target) ** 2))
        total_error += error
        total_energy += energy
        records.append({"view_index": view, "supervised": True,
                        "nrmse": math.sqrt(error / max(energy, 1e-24))})
    roles = arrays["support_class"][mode]
    result = {"mode_slot": mode, "full_render_modal_nrmse_fixed_alpha": math.sqrt(total_error / max(total_energy, 1e-24)),
            "per_view": records, "directly_supervised": int(np.sum(roles == 1)),
            "structure_inferred": int(np.sum(roles == 2)), "unresolved": int(np.sum(roles == 0))}
    if "t_host_gaussian_index" in arrays:
        result["fragment_propagated"] = int(np.sum(roles == 3))
    return result


def _prepare_work(work: Path, destination: Path, source: Mapping[str, Any],
                  run_contract: Mapping[str, Any], resume: bool) -> None:
    if type(resume) is not bool:
        raise TypeError("Neural resume must be boolean")
    if work == destination or work.is_relative_to(destination) or destination.is_relative_to(work):
        raise ValueError("Neural work and output directories must be disjoint")
    for name in ("static_scene", "topology", "measurements", "observed_structure_graph", "alignment_from", "complex_2d_modes"):
        parent = Path(source[name]).resolve()
        if work == parent or work.is_relative_to(parent) or destination == parent or destination.is_relative_to(parent):
            raise ValueError("Neural work/output must not modify an input artifact")
    manifest_path = work / "manifest.json"
    if work.exists():
        if not work.is_dir() or work.is_symlink():
            raise ValueError("Neural work path must be a normal directory")
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not resume:
                raise FileExistsError("Neural work exists; use --resume with identical sources/config")
            if previous != dict(run_contract):
                raise ValueError("Neural resume sources, geometry, configuration, or runtime differ")
            return
        if any(work.iterdir()):
            raise FileExistsError("Neural work directory contains unrelated files")
    work.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, run_contract)


def _read_frozen_resume(work: Path, source: Mapping[str, Any], config: NeuralModesConfig,
                        runtime: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Use the original frozen operator inputs, not new nondeterministic GPU reductions."""
    contract = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    if contract.get("format") != "modal_gaussians.neural_modes_work" or contract.get("version") != 1:
        raise ValueError("Unsupported neural work manifest")
    for name, expected in (("source_identity", _source_identity(source)), ("config", config.to_dict()), ("runtime", dict(runtime))):
        if contract.get(name) != expected:
            raise ValueError(f"Neural resume {name} differs")
    if _identity({name: value for name, value in contract.items() if name != "run_identity"}) != contract.get("run_identity"):
        raise ValueError("Neural work run identity differs")
    with np.load(work / "fixed_inputs.npz", allow_pickle=False) as archive:
        fixed = {name: archive[name] for name in archive.files}
    if _arrays_identity(fixed) != contract.get("fixed_arrays_identity"):
        raise ValueError("Neural frozen work inputs were changed")
    K, S, _ = fixed["sample_target"].shape
    G = len(fixed["g_points"])
    # Reuse strict scientific array validation before any saved pixel is indexed.
    validation = {**fixed, "phi": np.zeros((K, G, 3), dtype=np.complex64),
                  "sample_prediction": np.zeros((K, S, 2), dtype=np.complex64)}
    _validate_arrays(validation, {"config": config.to_dict(), "views": source["views"],
                                 "counts": {"modes": K, "foreground_gaussians": G, "views": len(source["views"])}})
    return fixed, contract


def _publish_artifact(destination: Path, source: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
                      model_states: Sequence[Mapping[str, Any]], config: NeuralModesConfig,
                      runtime: Mapping[str, Any], run_identity: str, graph_metadata: Mapping[str, Any],
                      optimization: Sequence[Mapping[str, Any]], command: Sequence[str]) -> NeuralModesArtifact:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Neural output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)).resolve()
    try:
        save_named_arrays(temporary / ARRAYS_FILENAME, arrays)
        networks: dict[str, Any] = {"run_identity": run_identity, "model_states": list(model_states)}
        if "mode_selection" in source:
            networks["source_mode_slots"] = _source_mode_slots(source).tolist()
        torch.save(networks, temporary / MODELS_FILENAME)
        mode_count, point_count, _ = arrays["phi"].shape
        from .training_fragments import artifact_contract
        training_fill = config.training_fragment_config is not None
        version, method = artifact_contract(config.training_fragment_config)
        manifest = {
            "format": COMPLETED_MODES_FORMAT, "version": version,
            "completion_method": method,
            **source, "source_identity": _source_identity(source), "config": config.to_dict(),
            "runtime": dict(runtime), "run_identity": run_identity, "geometry_graph": dict(graph_metadata),
            "producer": {"project_version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(), "command": list(command)},
            "semantics": _semantics(config), "quality_gate": QUALITY_GATE,
            "counts": {"modes": mode_count, "foreground_gaussians": point_count, "views": arrays["alphas"].shape[1],
                       "geometry_edges": len(arrays["g_edge_index"]), "controls": len(arrays["c_positions"]),
                       "measurement_samples": len(arrays["sample_confidence"])},
            "optimization": list(optimization),
            "diagnostics": {"per_mode": [_mode_diagnostics(arrays, mode) for mode in range(mode_count)]},
            "arrays_file": ARRAYS_FILENAME, "networks_file": MODELS_FILENAME,
            "arrays": {name: {"dtype": value.dtype.name, "shape": list(value.shape)} for name, value in arrays.items()},
            "arrays_identity": _arrays_identity(arrays), "arrays_file_sha256": _sha256(temporary / ARRAYS_FILENAME),
            "networks_sha256": _sha256(temporary / MODELS_FILENAME),
        }
        if training_fill:
            from .training_fragments import diagnostics
            manifest["diagnostics"]["training_fragments"] = diagnostics(arrays)
        manifest["completed_modes_identity"] = _identity(_artifact_identity_payload(manifest))
        _atomic_json(temporary / "manifest.json", manifest)
        load_neural_completed_modes(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Neural output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        # Only remove the exact temporary sibling just allocated above.
        if temporary.exists() and temporary.parent == destination.parent.resolve() and temporary.name.startswith(f".{destination.name}."):
            shutil.rmtree(temporary)
        raise
    return load_neural_completed_modes(destination)


def export_neural_prefix_artifact(*, scene_dir: str | Path, topology_dir: str | Path,
                                  measurements_dir: str | Path, graph_dir: str | Path,
                                  alignment_from: str | Path, work_dir: str | Path,
                                  output_dir: str | Path, count: int | None = None,
                                  mode_slots: Sequence[int] | None = None,
                                  flow_loader: Any = None,
                                  command: Sequence[str] = ()) -> NeuralModesArtifact:
    """Bake a completed prefix or explicit selection with original normalization."""
    from modal_gaussians.motion.neural.neural_field import evaluate_model
    if mode_slots is not None and count is not None:
        raise ValueError("Specify either a prefix count or source mode slots")
    if mode_slots is None and (type(count) is not int or count <= 0):
        raise ValueError("Neural prefix count must be a positive integer")
    work = Path(work_dir).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if not work.is_dir() or work.is_symlink():
        raise ValueError("Neural prefix work path must be a normal directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Neural output already exists: {destination}")
    if work == destination or work.is_relative_to(destination) or destination.is_relative_to(work):
        raise ValueError("Neural work and output directories must be disjoint")
    original_contract = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    settings = NeuralModesConfig.from_dict(original_contract["config"])
    source, scene, _, _, _, alignment, dense = _load_sources(
        scene_dir=scene_dir, topology_dir=topology_dir, measurements_dir=measurements_dir,
        graph_dir=graph_dir, alignment_from=alignment_from,
    )
    for name in ("static_scene", "topology", "measurements", "observed_structure_graph", "alignment_from", "complex_2d_modes"):
        parent = Path(source[name]).resolve()
        if (destination == parent or destination.is_relative_to(parent) or parent.is_relative_to(destination)
                or work == parent or work.is_relative_to(parent) or parent.is_relative_to(work)):
            raise ValueError("Neural prefix work/output must be disjoint from input artifacts")
    if mode_slots is None and count > len(source["modes"]):
        raise ValueError("Neural prefix count exceeds the complete source mode count")
    slots = _validated_mode_slots(range(count) if mode_slots is None else mode_slots, len(source["modes"]))
    count = len(slots)
    if not torch.cuda.is_available():
        raise RuntimeError("Full Gaussian neural prefix rendering requires CUDA")
    device = torch.device("cuda")
    runtime = {"device": "cuda", "torch_version": str(torch.__version__), "cuda_version": getattr(torch, "version").cuda,
               "gpu_name": torch.cuda.get_device_name(device)}
    frozen, parent_contract = _read_frozen_resume(work, source, settings, runtime)
    if parent_contract != original_contract or frozen["sample_target"].shape[0] != len(source["modes"]):
        raise ValueError("Neural prefix work contract changed or its complete source count differs")
    checkpoints, payloads = [], []
    for mode in slots:
        checkpoint = work / f"mode_{mode:03d}.pt"
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise ValueError(f"Neural prefix requires completed checkpoint {checkpoint.name}")
        checksum = _sha256(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (payload.get("run_identity") != parent_contract["run_identity"]
                or type(payload.get("mode")) is not int or payload["mode"] != mode
                or payload.get("complete") is not True
                or not isinstance(payload.get("best_model_state"), dict)
                or not isinstance(payload.get("summary"), dict)
                or type(payload["summary"].get("mode_slot")) is not int
                or payload["summary"]["mode_slot"] != mode):
            raise ValueError(f"Neural prefix checkpoint {checkpoint.name} is incomplete or has a different mode/run")
        if _sha256(checkpoint) != checksum:
            raise ValueError(f"Neural prefix checkpoint changed while reading: {checkpoint.name}")
        checkpoints.append({"source_mode_slot": mode, "filename": checkpoint.name, "sha256": checksum})
        payloads.append(payload)
    report_progress(f"neural prefix: baking {count} completed checkpoints with original frozen normalization")
    _, projectors, _, _, _ = _prepare_observation_arrays(
        scene, source, dense, settings, device,
        np.asarray(alignment.arrays["alphas"], dtype=np.complex64),
        np.asarray(alignment.arrays["alpha_identifiable_mask"], dtype=bool), frozen_arrays=frozen,
        flow_loader=flow_loader,
    )
    arrays = {name: value[slots].copy() if name in MODE_ARRAYS else value for name, value in frozen.items()}
    arrays["normalization_source_mode_view_rms"] = frozen["mode_view_rms"].copy()
    arrays["normalization_source_alpha_identifiable_mask"] = frozen["alpha_identifiable_mask"].copy()
    selection = {"policy": "completed_prefix" if mode_slots is None else "completed_selection", "source_mode_slots": slots,
                 "parent_run_identity": parent_contract["run_identity"],
                 "parent_fixed_arrays_identity": parent_contract["fixed_arrays_identity"],
                 "checkpoints": checkpoints}
    subset_source = {**source, "source_modes": source["modes"], "mode_selection": selection,
                     "modes": [{**source["modes"][slot], "mode_slot": local, "source_mode_slot": slot}
                               for local, slot in enumerate(slots)]}
    _source_mode_slots(subset_source)
    run_contract = {"format": "modal_gaussians.neural_modes_work", "version": 1,
                    "source_identity": _source_identity(subset_source), "config": settings.to_dict(),
                    "runtime": runtime, "geometry_graph": parent_contract["geometry_graph"],
                    "fixed_arrays_identity": _arrays_identity(arrays), "mode_selection": selection}
    run_identity = _identity(run_contract)
    arrays["phi"] = np.zeros((count, len(arrays["g_points"]), 3), dtype=np.complex64)
    arrays["sample_prediction"] = np.zeros_like(arrays["sample_target"])
    model_states, optimization = [], []
    for mode, payload in enumerate(payloads):
        state = payload["best_model_state"]
        # Bake on the same CPU backend used by strict loading. CUDA scatter and
        # GEMM roundoff can exceed elementwise tolerances near a mode's zeros.
        field = evaluate_model(state, _field_geometry(arrays, mode, "cpu"),
                               length_scale=float(arrays["scene_scale"]),
                               amplitude_scale=float(arrays["amplitude_scale"][mode]),
                               config=_field_config(settings, slots[mode]))[0].to(device)
        arrays["phi"][mode] = field.detach().cpu().numpy().astype(np.complex64)
        with torch.no_grad():
            for view, projector in enumerate(projectors):
                if arrays["alpha_identifiable_mask"][mode, view]:
                    lo, hi = arrays["view_sample_offsets"][view:view + 2]
                    prediction = complex(arrays["alphas"][mode, view]) * projector(field)
                    arrays["sample_prediction"][mode, lo:hi] = prediction.cpu().numpy()
        if not np.isfinite(arrays["phi"][mode]).all() or not np.isfinite(arrays["sample_prediction"][mode]).all():
            raise FloatingPointError("Neural prefix field or rendered prediction is non-finite")
        model_states.append(state)
        optimization.append({**payload["summary"], "mode_slot": mode, "source_mode_slot": slots[mode]})
    for checkpoint in checkpoints:
        if _sha256(work / checkpoint["filename"]) != checkpoint["sha256"]:
            raise ValueError("Neural prefix checkpoints changed during export")
    return _publish_artifact(destination, subset_source, arrays, model_states, settings, runtime,
                             run_identity, parent_contract["geometry_graph"], optimization, command)


def build_neural_modes_artifact(*, scene_dir: str | Path, topology_dir: str | Path,
                               measurements_dir: str | Path, graph_dir: str | Path,
                               alignment_from: str | Path, work_dir: str | Path,
                               output_dir: str | Path, config: NeuralModesConfig | None = None,
                               resume: bool = False, command: Sequence[str] = (),
                               prepared_inputs: Any = None, timings: Any = None,
                               mode_slots: Sequence[int] | None = None) -> NeuralModesArtifact:
    """Train on explicit invocation only; bake the full field as v8 or v10."""
    from modal_gaussians.motion.neural.geometry_graph import (
        build_geometry_graph_arrays,
        build_control_graph,
        depth_thresholds_from_manifest,
    )
    from modal_gaussians.motion.neural.neural_field import (
        ModalObservation,
        train_single_frequency,
        evaluate_model,
    )
    from .surface_attachments import SurfaceAttachmentConfig
    settings = config or NeuralModesConfig(training_fragment_config=SurfaceAttachmentConfig().to_dict())
    from modal_gaussians.iteration_cache import Timings
    timings = timings or Timings()
    settings.validate()
    destination = Path(output_dir).expanduser().resolve()
    work = Path(work_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Neural output already exists: {destination}")
    if settings.device == "cpu" or not torch.cuda.is_available():
        raise RuntimeError("Full Gaussian neural training requires CUDA; CPU is supported only by synthetic field tests")
    device = torch.device("cuda")
    source, scene, _, _, old_graph, alignment, dense = _load_sources(
        scene_dir=scene_dir, topology_dir=topology_dir, measurements_dir=measurements_dir,
        graph_dir=graph_dir, alignment_from=alignment_from,
    )
    selected_slots = (list(range(len(source["modes"]))) if mode_slots is None else
                      _validated_mode_slots(mode_slots, len(source["modes"])))
    runtime = {"device": "cuda", "torch_version": str(torch.__version__), "cuda_version": getattr(torch, "version").cuda,
               "gpu_name": torch.cuda.get_device_name(device)}
    frozen, previous_contract = None, None
    if resume and (work / "manifest.json").exists():
        frozen, previous_contract = _read_frozen_resume(work, source, settings, runtime)
        _prepare_work(work, destination, source, previous_contract, resume=True)
    report_progress("neural modes: freezing full-foreground observation renderer")
    if prepared_inputs is not None:
        arrays, projectors, cameras, depths, alpha_images = prepared_inputs.training_inputs(
            source, scene, old_graph, settings, device, timings)
        if frozen is not None and _arrays_identity(arrays) != _arrays_identity(frozen):
            raise ValueError("Prepared fixed inputs differ from resumed work")
        frozen = arrays
    else:
        arrays, projectors, cameras, depths, alpha_images = _prepare_observation_arrays(
            scene, source, dense, settings, device,
            np.asarray(alignment.arrays["alphas"], dtype=np.complex64),
            np.asarray(alignment.arrays["alpha_identifiable_mask"], dtype=bool),
            frozen_arrays=frozen,
        )
    endpoint, jump = depth_thresholds_from_manifest(old_graph.manifest, [view["label"] for view in source["views"]])
    geometry_settings = _geometry_config(settings)
    if frozen is None:
        report_progress("neural modes: building appearance-independent foreground geometry/control graphs")
        graph = build_geometry_graph_arrays(
            foreground_means=scene.foreground.active()["means"].detach().cpu().numpy(),
            Ks=np.stack([c.K.cpu().numpy() for c in cameras]),
            radial_coefficients=np.array([c.radial_distortion for c in cameras]),
            world_to_cameras=np.stack([c.world_to_camera.cpu().numpy() for c in cameras]),
            rendered_depths=depths, rendered_alphas=alpha_images,
            endpoint_thresholds=endpoint, depth_jump_thresholds=jump, config=geometry_settings,
        )
        arrays.update({"g_" + name: value for name, value in graph.as_dict().items()})
        if settings.training_fragment_config is not None:
            from .training_fragments import build_training_controls
            attachment_inputs = None
            if settings.training_fragment_config.get("strategy") == "surface":
                from .surface_attachments import observation_inputs
                attachment_inputs = observation_inputs(scene, cameras, depths, alpha_images, endpoint,
                    arrays["contribution_mass"], arrays["contribution_threshold"])
            if settings.training_fragment_config.get("strategy") == "pointwise":
                attachment_inputs = {"p_observation_view_mask": arrays["observation_view_mask"]}
            if settings.training_fragment_config.get("strategy") == "guarded":
                from .guarded_attachments import observation_inputs
                attachment_inputs = observation_inputs(arrays["g_points"], cameras, depths, alpha_images,
                                                        endpoint, settings.alpha_minimum)
                attachment_inputs.update(observation_view_mask=arrays["observation_view_mask"],
                                         contribution_mass=arrays["contribution_mass"])
            if settings.training_fragment_config.get("strategy") == "component_field":
                from .component_field import observation_inputs
                attachment_inputs = observation_inputs(arrays["g_points"], cameras, depths, alpha_images,
                                                        endpoint, settings.alpha_minimum)
                attachment_inputs.update(observation_view_mask=arrays["observation_view_mask"],
                                         contribution_mass=arrays["contribution_mass"])
            arrays.update(build_training_controls(graph, geometry_config=geometry_settings,
                          fragment_config=settings.training_fragment_config, scene_scale=float(arrays["scene_scale"]),
                          attachment_inputs=attachment_inputs))
        else:
            controls = build_control_graph(graph, config=geometry_settings, scene_scale=float(arrays["scene_scale"]))
            arrays.update({"c_" + name: value for name, value in controls.as_dict().items()})
        roles = _set_support_roles(arrays)
    else:
        from modal_gaussians.motion.neural.geometry_graph import GeometryGraph
        graph = GeometryGraph.from_dict({name[2:]: value for name, value in arrays.items() if name.startswith("g_")})
        roles = arrays["support_class"]
    for mode in selected_slots:
        if not np.any(arrays["observation_view_mask"][mode].any(axis=1) & (roles[mode] != 0)):
            raise ValueError(f"Neural mode {mode} has no effective Gaussian observation contribution")
    geometry_metadata = {
        "policy": ("all_foreground_mutual_knn_no_depth_filter" if settings.graph_edge_filter == "none"
                   else "all_foreground_three_state_depth_geometry_no_rgb"),
        "config": geometry_settings.to_dict(),
        "endpoint_thresholds": [float(x) if np.isfinite(x) else None for x in endpoint],
        "depth_jump_thresholds": [float(x) if np.isfinite(x) else None for x in jump],
        "interpolation": "all_graph_distance_supports_within_2h_normalized_wendland_c2",
        "control_edge_length": "original_geometry_graph_shortest_path",
    }
    if settings.training_fragment_config is not None:
        geometry_metadata["interpolation"] = (
            "host_wendland_then_per_point_displacement_transfer"
            if settings.training_fragment_config.get("strategy") in ("pointwise", "guarded", "component_field") else
            "host_wendland_with_fixed_fragment_anchor_composition")
        geometry_metadata["fragment_training"] = settings.training_fragment_config
        from .training_fragments import diagnostics
        report_progress("neural attachment preflight: " + json.dumps(diagnostics(arrays), allow_nan=False))
    run_contract = {"format": "modal_gaussians.neural_modes_work", "version": 1,
                    "source_identity": _source_identity(source), "config": settings.to_dict(), "runtime": runtime,
                    "geometry_graph": geometry_metadata, "fixed_arrays_identity": _arrays_identity(arrays)}
    run_identity = _identity(run_contract)
    run_contract["run_identity"] = run_identity
    _prepare_work(work, destination, source, run_contract, resume)
    if settings.training_fragment_config is not None:
        from .training_fragments import diagnostics
        _atomic_json(work / "attachment_preflight.json", diagnostics(arrays))
    fixed_path = work / "fixed_inputs.npz"
    if fixed_path.exists():
        with np.load(fixed_path, allow_pickle=False) as archive:
            if _arrays_identity({name: archive[name] for name in archive.files}) != run_contract["fixed_arrays_identity"]:
                raise ValueError("Neural fixed work inputs were changed")
    else:
        fixed_temporary = work / "fixed_inputs.tmp.npz"
        save_named_arrays(fixed_temporary, arrays)
        os.replace(fixed_temporary, fixed_path)
    mode_count, sample_count, _ = arrays["sample_target"].shape
    arrays["phi"] = np.zeros((mode_count, len(graph.points), 3), dtype=np.complex64)
    arrays["sample_prediction"] = np.zeros((mode_count, sample_count, 2), dtype=np.complex64)
    model_states, optimization = [], []
    for mode in selected_slots:
        checkpoint = work / f"mode_{mode:03d}.pt"
        payload = None
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload.get("run_identity") != run_identity or payload.get("mode") != mode:
                raise ValueError("Neural checkpoint mode or run identity differs")
        geometry = _field_geometry(arrays, mode, device)
        field_config = _field_config(settings, mode)
        if payload is not None and payload.get("complete") is True:
            state = payload["best_model_state"]
            field = evaluate_model(state, geometry, length_scale=float(arrays["scene_scale"]),
                                   amplitude_scale=float(arrays["amplitude_scale"][mode]), config=field_config)[0]
            summary = payload["summary"]
            report_progress(f"neural mode {mode + 1}/{mode_count}: reusing validated completed checkpoint")
        else:
            observations = []
            for view, projector in enumerate(projectors):
                if not arrays["alpha_identifiable_mask"][mode, view]:
                    continue
                lo, hi = arrays["view_sample_offsets"][view:view + 2]
                observations.append(ModalObservation(
                    target=torch.as_tensor(arrays["sample_target"][mode, lo:hi], device=device),
                    project=projector, alpha=complex(arrays["alphas"][mode, view]),
                    confidence=torch.as_tensor(arrays["sample_confidence"][lo:hi], device=device),
                    normalized_rms=float(arrays["mode_view_loss_scale"][mode, view]),
                    name=str(source["views"][view]["label"]),
                ))

            def save_checkpoint(step: int, trainer_state: dict[str, Any]) -> None:
                _atomic_torch(checkpoint, {"run_identity": run_identity, "mode": mode, "complete": False,
                                           "trainer_state": trainer_state})
                report_progress(f"neural mode {mode + 1}/{mode_count}: step {step}")

            report_progress(f"neural mode {mode + 1}/{mode_count}: fitting {source['modes'][mode]['frequency_hz']:.6g} Hz")
            with timings.stage("network_optimization", mode=mode):
                fitted = train_single_frequency(
                    geometry, observations, length_scale=float(arrays["scene_scale"]),
                    amplitude_scale=float(arrays["amplitude_scale"][mode]), config=field_config,
                    resume_state=None if payload is None else payload["trainer_state"], checkpoint_callback=save_checkpoint,
                )
            timings.records[-1]["iterations"] = fitted.iterations
            field, state = fitted.field, fitted.best_model_state
            summary = {"mode_slot": mode, "iterations": fitted.iterations, "best_step": fitted.best_step,
                       "best_loss": fitted.best_loss, "converged": fitted.converged, "history": fitted.history}
            _atomic_torch(checkpoint, {"run_identity": run_identity, "mode": mode, "complete": True,
                                       "trainer_state": fitted.latest_state, "best_model_state": state, "summary": summary})
        # The saved network, baked field, and rendered predictions must all use
        # the loader's CPU evaluation. Keep optimization and checkpoints on CUDA.
        field = evaluate_model(state, _field_geometry(arrays, mode, "cpu"),
                               length_scale=float(arrays["scene_scale"]),
                               amplitude_scale=float(arrays["amplitude_scale"][mode]),
                               config=field_config)[0].to(device)
        arrays["phi"][mode] = field.detach().cpu().numpy().astype(np.complex64)
        with torch.no_grad():
            for view, projector in enumerate(projectors):
                if not arrays["alpha_identifiable_mask"][mode, view]:
                    continue
                lo, hi = arrays["view_sample_offsets"][view:view + 2]
                prediction = complex(arrays["alphas"][mode, view]) * projector(field)
                arrays["sample_prediction"][mode, lo:hi] = prediction.cpu().numpy()
        if not np.isfinite(arrays["phi"][mode]).all() or not np.isfinite(arrays["sample_prediction"][mode]).all():
            raise FloatingPointError("Neural learned motion or rendered prediction is non-finite")
        model_states.append(state)
        optimization.append(summary)
    if mode_slots is not None:
        # Preserve the full frozen normalization contract and source-index seeds;
        # only completed selected checkpoints are published, never zero placeholders.
        return export_neural_prefix_artifact(scene_dir=scene_dir, topology_dir=topology_dir,
            measurements_dir=measurements_dir, graph_dir=graph_dir, alignment_from=alignment_from,
            work_dir=work, output_dir=destination, mode_slots=selected_slots, command=command,
            flow_loader=None if prepared_inputs is None else prepared_inputs.flow)
    return _publish_artifact(destination, source, arrays, model_states, settings, runtime, run_identity,
                             geometry_metadata, optimization, command)


__all__ = ["NeuralModesConfig", "NeuralModesArtifact", "FrozenModalProjector", "support_roles",
           "build_neural_modes_artifact", "export_neural_prefix_artifact", "load_neural_completed_modes"]
