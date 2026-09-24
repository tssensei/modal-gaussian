"""Train complex component fields from prepared per-frequency observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from modal_gaussians.common.scene_store import resolve_path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from modal_gaussians import __version__
from modal_gaussians.common.numpy_io import save_named_arrays
from modal_gaussians.common.progress import report_progress
from modal_gaussians.motion.component_field import MODE_ARRAYS as COMPONENT_MODE_ARRAYS, ComponentFieldConfig

COMPLETED_MODES_FORMAT = "modal_gaussians.completed_modes"
COMPLETED_MODES_VERSION = 18
COMPLETION_METHOD = "neural_complex_displacement_field"
ARRAYS_FILENAME = "completed_modes.npz"
MODELS_FILENAME = "networks.pt"
QUALITY_GATE = {"required": True, "status": "completion_candidate_unapproved"}
MODE_ARRAYS = {
    "alphas", "alpha_identifiable_mask", "observation_view_mask", "support_class",
    "sample_target", "mode_view_rms", "mode_view_loss_scale", "amplitude_scale",
} | COMPONENT_MODE_ARRAYS


@dataclass(frozen=True)
class NeuralModesConfig:
    graph_neighbors: int = 16
    graph_max_distance: float = 0.08
    control_radius_fraction: float = 0.015
    max_controls: int = 32768
    hidden_dim: int = 256
    message_layers: int = 3
    local_feature_dim: int = 32
    pixel_sample_stride: int = 2
    alpha_minimum: float = 0.05
    mask_erosion_iterations: int = 1
    energy_floor_fraction: float = 0.05
    huber_delta: float = 1.0
    data_loss_normalization: str = "view_rms"
    deformation_weight: float = 0.03
    rotation_weight: float = 0.0
    rotation_length_fraction: float = 0.05
    learning_rate: float = 0.001
    max_iterations: int = 5000
    gradient_clip: float = 1.0
    seed: int = 1729
    convergence_patience: int = 50
    relative_tolerance: float = 1.0e-6
    checkpoint_every: int = 100
    device: str = "auto"
    graph_edge_filter: str = "none"
    modal_projection_backend: str = "dynamic"
    training_fragment_config: dict[str, Any] = field(default_factory=lambda: ComponentFieldConfig().to_dict())

    def validate(self) -> None:
        if self.data_loss_normalization not in ("view_rms", "none"):
            raise ValueError("Neural data_loss_normalization must be view_rms or none")
        if self.modal_projection_backend not in ("dynamic", "cached"):
            raise ValueError("Neural modal_projection_backend must be dynamic or cached")
        ComponentFieldConfig.from_dict(self.training_fragment_config)
        if self.graph_edge_filter != "none":
            raise ValueError("Candidate geometry uses unfiltered KNN")
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
        for name in ("graph_max_distance",
                     "control_radius_fraction", "alpha_minimum", "energy_floor_fraction",
                     "huber_delta", "rotation_length_fraction", "learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Neural {name} must be finite and positive")
        for name in ("deformation_weight", "rotation_weight", "relative_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Neural {name} must be finite and non-negative")
        if self.alpha_minimum > 1:
            raise ValueError("Alpha threshold cannot exceed one")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Neural device must be auto, cpu, or cuda")

    def to_dict(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        result = cls(**dict(value))
        result.validate()
        return result


@dataclass(frozen=True)
class NeuralModesArtifact:
    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    rotation: np.ndarray | None = None
    control_displacement: np.ndarray | None = None


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
    from modal_gaussians.motion.geometry_graph import GeometryGraphConfig
    return GeometryGraphConfig(
        max_neighbors=config.graph_neighbors, max_distance=config.graph_max_distance,
        alpha_minimum=config.alpha_minimum, control_radius_fraction=config.control_radius_fraction,
        max_controls=config.max_controls,
        edge_filter=config.graph_edge_filter,
    )


def _field_config(config: NeuralModesConfig, mode: int = 0) -> Any:
    from modal_gaussians.motion.network import NeuralFieldConfig
    return NeuralFieldConfig(
        hidden_dim=config.hidden_dim, message_layers=config.message_layers,
        local_feature_dim=config.local_feature_dim,
        learning_rate=config.learning_rate, max_iterations=config.max_iterations,
        gradient_clip=config.gradient_clip, seed=config.seed + mode,
        convergence_patience=config.convergence_patience, relative_tolerance=config.relative_tolerance,
        checkpoint_every=config.checkpoint_every, huber_delta=config.huber_delta,
        data_loss_normalization=config.data_loss_normalization,
        deformation_weight=config.deformation_weight, rotation_weight=config.rotation_weight,
        rotation_length_fraction=config.rotation_length_fraction,
    )


def _source_identity(source):
    return {name: source[name] for name in (
        'static_scene_identity', 'foreground_identity', 'topology_identity',
        'endpoint_relative_tolerances', 'observation_region', 'alignment_identity', 'complex_2d_modes_identity', 'modes', 'views',
    )}


def _validated_mode_slots(slots: Sequence[int], count: int) -> list[int]:
    values = list(slots)
    if (not values or any(type(slot) is not int or not 0 <= slot < count for slot in values)
            or values != sorted(set(values))):
        raise ValueError("Selected source mode slots must be nonempty, unique, in range and in source order")
    return values


class FrozenModalProjector:
    """Differentiable four-channel rasterization; all geometric tensors are fixed."""

    def __init__(self, scene: Any, camera: Any, jacobian: torch.Tensor,
                 pixels: np.ndarray, foreground_alpha: torch.Tensor, *, backend: str = "dynamic") -> None:
        if backend not in ("dynamic", "cached"):
            raise ValueError("Modal projection backend must be dynamic or cached")
        self.scene, self.camera, self.jacobian = scene, camera, jacobian.detach()
        self.x = torch.as_tensor(pixels[:, 0], device=jacobian.device, dtype=torch.long)
        self.y = torch.as_tensor(pixels[:, 1], device=jacobian.device, dtype=torch.long)
        self.alpha = foreground_alpha.detach()
        self.cached = None
        if backend == "cached":
            from modal_gaussians.motion.modal_projection import CachedModalRasterizer
            self.cached = CachedModalRasterizer(scene, camera, pixels)

    def sample_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.cached is not None:
            return self.cached(features) / self.alpha[:, None]
        from modal_gaussians.motion.common.projection import render_motion_features

        image, _ = render_motion_features(self.scene, self.camera, features)
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


def _field_geometry(arrays: Mapping[str, np.ndarray], mode: int, device: Any = "cpu", *, validate=False) -> Any:
    from modal_gaussians.motion.network import NeuralFieldGeometry
    supported = np.asarray(arrays["support_class"])[mode] != 0
    interpolation_prefix = "t_"
    control_indices = arrays["c_control_point_index"]
    control_indices = arrays["t_host_gaussian_index"][control_indices]
    values = {
        "gaussian_positions": arrays["g_points"], "control_positions": arrays["c_positions"],
        "gaussian_edges": arrays["g_edge_index"], "gaussian_edge_weights": arrays["g_edge_weight"],
        "control_edges": arrays["c_control_edges"], "control_edge_weights": arrays["c_control_edge_weight"],
        "control_edge_lengths": arrays["c_control_edge_length"],
        "interpolation_indptr": arrays[interpolation_prefix + "interpolation_indptr"],
        "interpolation_indices": arrays[interpolation_prefix + "interpolation_indices"],
        "interpolation_weights": arrays[interpolation_prefix + "interpolation_weights"],
        "allow_empty_interpolation": True,
        "gaussian_supported": supported,
        "control_supported": supported[control_indices],
    }
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
    return NeuralFieldGeometry.from_arrays(values, device=device, validate=validate)


def _artifact_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = {name: manifest[name] for name in (
        "format", "version", "completion_method", "source_identity", "config", "runtime",
        "run_identity", "semantics", "quality_gate", "geometry_graph", "counts",
        "arrays_identity", "networks_sha256", "optimization", "diagnostics",
    )}
    result.update({name: manifest[name] for name in ("source_modes", "mode_selection") if name in manifest})
    return result


def _semantics(config, source):
    return {'field': 'complex_translation_and_angular_displacement',
            'geometry': 'fixed_static_gaussians',
            'coefficients': 'independent_per_recording_not_trained_here',
            'supervision': 'per_view_modal_images_with_complex_gain_and_RMS_normalization',
            'controls': 'component_GNN_with_local_features',
            'transfer': 'displacement_from_reliable_component_donors',
            'observation_region': source['observation_region']}


def _set_support_roles(arrays):
    from modal_gaussians.motion.component_field import support_roles
    roles, _ = support_roles(arrays, arrays['observation_view_mask'])
    arrays['support_class'] = roles
    return roles


def _prepare_observation_arrays(scene: Any, source: Mapping[str, Any], dense: Any,
                                config: NeuralModesConfig, device: torch.device,
                                alphas: np.ndarray, identifiable: np.ndarray,
                                frozen_arrays: Mapping[str, np.ndarray] | None = None,
                                flow_loader: Any = None,
                                observation_renders: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
                                ) -> tuple[dict[str, np.ndarray], list[FrozenModalProjector], list[Any], list[np.ndarray], list[np.ndarray]]:
    from modal_gaussians.preprocessing.reference import load_reference, reference_identity
    from modal_gaussians.motion.common.projection import RenderedDesignConfig, candidate_observation_pixels, projection_jacobian, render_observation_geometry
    from modal_gaussians.geometry.scene import cameras_from_scene_manifest

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
        flow = (flow_loader or load_reference)(dense_view["flow_artifact"])
        if reference_identity(flow) != view["flow_identity"]:
            raise ValueError("Neural sampling flow identity differs")
        if list(flow.arrays.mask_union.shape) != [camera.height, camera.width]:
            raise ValueError("Neural flow mask shape differs from camera")
        with torch.no_grad():
            render = (render_observation_geometry(scene, camera) if observation_renders is None
                      else observation_renders[view["label"]])
        alpha_image = render["alpha"].cpu().numpy().astype(np.float32)
        depth_image = render["expected_depth"].cpu().numpy().astype(np.float32)
        if frozen_arrays is None:
            pixels, confidence = candidate_observation_pixels(scene, flow.arrays.mask_union, alpha_image, sampling)
        else:
            lo, hi = frozen_arrays["view_sample_offsets"][view["index"]:view["index"] + 2]
            pixels = frozen_arrays["sample_pixels_xy"][lo:hi]
            confidence = frozen_arrays["sample_confidence"][lo:hi]
        jacobian, _ = projection_jacobian(points, camera.K.cpu().numpy(), camera.world_to_camera.cpu().numpy(), camera.radial_distortion)
        projector = FrozenModalProjector(scene, camera, torch.as_tensor(jacobian, device=device),
                                         pixels, torch.as_tensor(confidence, device=device),
                                         backend=config.modal_projection_backend)
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


def _validate_arrays(arrays, manifest, *, check_field_geometry=True):
    """Stable entry point; format validation lives in artifacts.py."""
    from modal_gaussians.motion.artifacts import _validate_arrays as implementation
    return implementation(arrays, manifest, check_field_geometry=check_field_geometry)


def load_neural_completed_modes(path, *, validate=False):
    """Stable entry point; artifact loading lives in artifacts.py."""
    from modal_gaussians.motion.artifacts import load_neural_completed_modes as implementation
    return implementation(path, validate=validate)


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
    for name in ("static_scene", "topology", "alignment_from", "complex_2d_modes"):
        parent = resolve_path(source[name])
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
    return fixed, contract


def _publish_artifact(destination: Path, source: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
                      model_states: Sequence[Mapping[str, Any]], config: NeuralModesConfig,
                      runtime: Mapping[str, Any], run_identity: str, graph_metadata: Mapping[str, Any],
                      optimization: Sequence[Mapping[str, Any]], command: Sequence[str],
                      *, rotation: np.ndarray | None = None,
                      control_displacement: np.ndarray | None = None) -> NeuralModesArtifact:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Neural output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)).resolve()
    try:
        arrays = {**arrays, "rotation": np.asarray(rotation, dtype=np.complex64),
                  "control_displacement": np.asarray(control_displacement, dtype=np.complex64)}
        save_named_arrays(temporary / ARRAYS_FILENAME, arrays)
        networks: dict[str, Any] = {"run_identity": run_identity, "model_states": list(model_states)}
        torch.save(networks, temporary / MODELS_FILENAME)
        mode_count, point_count, _ = arrays["phi"].shape
        from modal_gaussians.motion.component_field import VERSION as version, METHOD as method
        manifest = {
            "format": COMPLETED_MODES_FORMAT, "version": version,
            "completion_method": method,
            **source, "source_identity": _source_identity(source), "config": config.to_dict(),
            "runtime": dict(runtime), "run_identity": run_identity, "geometry_graph": dict(graph_metadata),
            "producer": {"project_version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(), "command": list(command)},
            "semantics": _semantics(config, source), "quality_gate": QUALITY_GATE,
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
        from modal_gaussians.motion.component_field import diagnostics
        manifest["diagnostics"]["training_fragments"] = diagnostics(arrays)
        manifest["completed_modes_identity"] = _identity(_artifact_identity_payload(manifest))
        _atomic_json(temporary / "manifest.json", manifest)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Neural output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        # Only remove the exact temporary sibling just allocated above.
        if temporary.exists() and temporary.parent == destination.parent.resolve() and temporary.name.startswith(f".{destination.name}."):
            shutil.rmtree(temporary)
        raise
    return NeuralModesArtifact(destination, manifest, dict(arrays), rotation, control_displacement)


def build_neural_modes_artifact(*, work_dir, output_dir, prepared_inputs, config=None,
                               resume=False, command=(), timings=None, continuation=None):
    """Fit one selected frequency; static geometry and appearance remain frozen."""
    from modal_gaussians.motion.network import ModalObservation, train_single_frequency, evaluate_model
    settings = config or NeuralModesConfig()
    from modal_gaussians.common.cache import Timings
    timings = timings or Timings()
    settings.validate()
    destination = resolve_path(output_dir)
    work = resolve_path(work_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Neural output already exists: {destination}")
    if settings.device == "cpu" or not torch.cuda.is_available():
        raise RuntimeError("Full Gaussian neural training requires CUDA; CPU is supported only by synthetic field tests")
    device = torch.device("cuda")
    from modal_gaussians.geometry.scene import load_static_scene
    source = dict(prepared_inputs.source)
    scene = load_static_scene(source['static_scene'], 'cpu')
    if len(source['modes']) != 1 or prepared_inputs.external_geometry_graph is None:
        raise ValueError('Training needs one selected frequency and its modal-similarity graph')
    selected_slots = [0]
    runtime = {"device": "cuda", "torch_version": str(torch.__version__), "cuda_version": getattr(torch, "version").cuda,
               "gpu_name": torch.cuda.get_device_name(device)}
    frozen, previous_contract = None, None
    if resume and (work / "manifest.json").exists():
        frozen, previous_contract = _read_frozen_resume(work, source, settings, runtime)
        _prepare_work(work, destination, source, previous_contract, resume=True)
    report_progress("neural modes: freezing full-foreground observation renderer")
    if continuation is not None and frozen is None:
        from modal_gaussians.motion.continuation import frozen_inputs
        frozen = frozen_inputs(continuation, _source_identity(source), settings.to_dict(), runtime)
    arrays, projectors, cameras, depths, alpha_images = prepared_inputs.training_inputs(
        source, scene, settings, device, timings,
        **({"frozen_arrays": frozen} if frozen is not None else {}))
    if frozen is not None and arrays is not frozen and _arrays_identity(arrays) != _arrays_identity(frozen):
        raise ValueError("Prepared fixed inputs differ from resumed work")
    frozen = arrays
    endpoint = np.asarray(source["endpoint_relative_tolerances"], dtype=np.float64)
    geometry_settings = _geometry_config(settings)
    from modal_gaussians.motion.geometry_graph import GeometryGraph
    graph = GeometryGraph.from_dict({name[2:]: value for name, value in arrays.items() if name.startswith("g_")})
    roles = arrays["support_class"]
    for mode in selected_slots:
        if not np.any(arrays["observation_view_mask"][mode].any(axis=1) & (roles[mode] != 0)):
            raise ValueError(f"Neural mode {mode} has no effective Gaussian observation contribution")
    geometry_metadata = {
        "policy": "per_frequency_modal_similarity_graph",
        "config": geometry_settings.to_dict(),
        "endpoint_thresholds": [float(x) if np.isfinite(x) else None for x in endpoint],
        "interpolation": "all_graph_distance_supports_within_2h_normalized_wendland_c2",
        "control_edge_length": "original_geometry_graph_shortest_path",
    }
    if graph.edge_propagation_length is not None:
        geometry_metadata["control_sampling_distance"] = "original_geometry_graph_shortest_path"
        geometry_metadata["interpolation_attenuation"] = "geometric_distance_over_propagation_distance_then_row_normalize"
    geometry_metadata['external_graph'] = prepared_inputs.external_geometry_contract
    geometry_metadata['fragment_training'] = settings.training_fragment_config
    from modal_gaussians.motion.component_field import diagnostics
    report_progress('component field preflight: ' + json.dumps(diagnostics(arrays), allow_nan=False))
    run_contract = {"format": "modal_gaussians.neural_modes_work", "version": 1,
                    "source_identity": _source_identity(source), "config": settings.to_dict(), "runtime": runtime,
                    "geometry_graph": geometry_metadata, "fixed_arrays_identity": _arrays_identity(arrays)}
    parent_work = None
    if continuation is not None:
        from modal_gaussians.motion.continuation import bind_work
        parent_work = bind_work(continuation, run_contract)
        run_contract["continuation"] = continuation
    run_identity = _identity(run_contract)
    run_contract["run_identity"] = run_identity
    _prepare_work(work, destination, source, run_contract, resume)
    from modal_gaussians.motion.component_field import diagnostics
    _atomic_json(work / "attachment_preflight.json", diagnostics(arrays))
    fixed_path = work / "fixed_inputs.npz"
    if not fixed_path.exists():
        fixed_temporary = work / "fixed_inputs.tmp.npz"
        save_named_arrays(fixed_temporary, arrays)
        os.replace(fixed_temporary, fixed_path)
    mode_count, sample_count, _ = arrays["sample_target"].shape
    arrays["phi"] = np.zeros((mode_count, len(graph.points), 3), dtype=np.complex64)
    arrays["sample_prediction"] = np.zeros((mode_count, sample_count, 2), dtype=np.complex64)
    model_states, optimization = [], []
    rotations, control_displacements = [], []
    for mode in selected_slots:
        checkpoint = work / f"mode_{mode:03d}.pt"
        payload = None
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload.get("run_identity") != run_identity or payload.get("mode") != mode:
                raise ValueError("Neural checkpoint mode or run identity differs")
        elif parent_work is not None:
            from modal_gaussians.motion.continuation import checkpoint as parent_checkpoint
            payload = parent_checkpoint(parent_work, mode, continuation["run_identity"], run_identity)
            if payload is not None:
                _atomic_torch(checkpoint, payload)
        field_config = _field_config(settings, mode)
        if payload is not None and payload.get("complete") is True:
            state = payload["best_model_state"]
            summary = payload["summary"]
            report_progress(f"neural mode {mode + 1}/{mode_count}: reusing completed checkpoint")
        else:
            geometry = _field_geometry(arrays, mode, device)
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
            start_step = 0 if payload is None else int(payload["trainer_state"]["step"])
            with timings.stage("network_optimization", mode=mode):
                fitted = train_single_frequency(
                    geometry, observations, length_scale=float(arrays["scene_scale"]),
                    amplitude_scale=float(arrays["amplitude_scale"][mode]), config=field_config,
                    resume_state=None if payload is None else payload["trainer_state"], checkpoint_callback=save_checkpoint,
                )
            timings.records[-1]["iterations"] = fitted.iterations
            updates = fitted.iterations - start_step
            timings.records[-1].update(start_step=start_step, updates_performed=updates,
                mean_seconds_per_update=timings.records[-1]["seconds"] / updates if updates else None)
            state = fitted.best_model_state
            summary = {"mode_slot": mode, "iterations": fitted.iterations, "best_step": fitted.best_step,
                       "best_loss": fitted.best_loss, "converged": fitted.converged, "history": fitted.history}
            _atomic_torch(checkpoint, {"run_identity": run_identity, "mode": mode, "complete": True,
                                       "trainer_state": fitted.latest_state, "best_model_state": state, "summary": summary})
        # The saved network, baked field, and rendered predictions must all use
        # the loader's CPU evaluation. Keep optimization and checkpoints on CUDA.
        evaluated = evaluate_model(state, _field_geometry(arrays, mode, "cpu"),
                               length_scale=float(arrays["scene_scale"]),
                               amplitude_scale=float(arrays["amplitude_scale"][mode]),
                               config=field_config)
        field = evaluated[0].to(device)
        rotations.append(evaluated[1].detach().cpu().numpy())
        control_displacements.append(evaluated[2].detach().cpu().numpy())
        arrays["phi"][mode] = field.detach().cpu().numpy().astype(np.complex64)
        with torch.no_grad():
            for view, projector in enumerate(projectors):
                if not arrays["alpha_identifiable_mask"][mode, view]:
                    continue
                lo, hi = arrays["view_sample_offsets"][view:view + 2]
                prediction = complex(arrays["alphas"][mode, view]) * projector(field)
                arrays["sample_prediction"][mode, lo:hi] = prediction.cpu().numpy()
        model_states.append(state)
        optimization.append(summary)
    return _publish_artifact(destination, source, arrays, model_states, settings, runtime, run_identity,
                             geometry_metadata, optimization, command,
                             rotation=np.stack(rotations),
                             control_displacement=np.stack(control_displacements))
