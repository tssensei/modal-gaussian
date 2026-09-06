"""Independent-frequency complex graph fields and a resumable Torch optimizer.

All geometry, interpolation, observation gains and confidence weights are fixed.
The amplitude scale is numerical conditioning of raw DFT coefficients; it is
not a physical oscillation amplitude.  This module performs no artifact I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class NeuralFieldConfig:
    hidden_dim: int = 64
    message_layers: int = 3
    learning_rate: float = 0.001
    max_iterations: int = 2000
    gradient_clip: float = 1.0
    seed: int = 1729
    convergence_patience: int = 50
    relative_tolerance: float = 1.0e-6
    checkpoint_every: int = 100
    huber_delta: float = 1.0
    deformation_weight: float = 1.0
    rotation_weight: float = 0.1
    rotation_length_fraction: float = 0.05

    @property
    def edge_weight(self) -> float:
        return self.deformation_weight

    def validate(self) -> None:
        for name in ("hidden_dim", "message_layers", "max_iterations",
                     "convergence_patience", "checkpoint_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Neural field {name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Neural field seed must be a non-negative integer")
        for name in ("learning_rate", "gradient_clip", "huber_delta",
                     "rotation_length_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Neural field {name} must be finite and positive")
        for name in ("relative_tolerance", "deformation_weight", "rotation_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Neural field {name} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "NeuralFieldConfig":
        data = dict(values)
        if "edge_weight" in data:
            if "deformation_weight" in data:
                raise ValueError("Specify deformation_weight or edge_weight, not both")
            data["deformation_weight"] = data.pop("edge_weight")
        result = cls(**data)
        result.validate()
        return result


@dataclass(frozen=True)
class NeuralFieldGeometry:
    gaussian_positions: Tensor
    control_positions: Tensor
    interpolation_indptr: Tensor
    interpolation_indices: Tensor
    interpolation_weights: Tensor
    interpolation_rows: Tensor
    gaussian_edges: Tensor
    gaussian_edge_weights: Tensor
    control_edges: Tensor
    control_edge_weights: Tensor
    control_edge_lengths: Tensor
    gaussian_supported: Tensor
    control_supported: Tensor

    @classmethod
    def from_arrays(
        cls, arrays: Mapping[str, Any], *, device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "NeuralFieldGeometry":
        """Import fixed geometry; interpolation is exclusively CSR, never padded."""
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Geometry dtype must be float32 or float64")

        def tensor(name: str, kind: torch.dtype = dtype) -> Tensor:
            return torch.as_tensor(arrays[name], device=device, dtype=kind).detach()

        points, controls = tensor("gaussian_positions"), tensor("control_positions")
        for name, value in (("gaussian_positions", points), ("control_positions", controls)):
            if value.ndim != 2 or value.shape[1] != 3 or len(value) == 0:
                raise ValueError(f"{name} must have non-empty shape [N,3]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        count, control_count = len(points), len(controls)
        indptr = tensor("interpolation_indptr", torch.long)
        indices = tensor("interpolation_indices", torch.long)
        weights = tensor("interpolation_weights")
        if (indptr.shape != (count + 1,) or indices.ndim != 1
                or weights.shape != indices.shape or int(indptr[0]) != 0
                or int(indptr[-1]) != len(indices)
                or bool((indptr[1:] < indptr[:-1]).any())):
            raise ValueError("Invalid interpolation CSR structure")
        if (bool((indices < 0).any()) or bool((indices >= control_count).any())
                or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())):
            raise ValueError("Interpolation needs valid controls and finite non-negative weights")
        rows = torch.repeat_interleave(
            torch.arange(count, device=device), indptr[1:] - indptr[:-1],
        )
        sums = weights.new_zeros(count).index_add_(0, rows, weights)
        if not torch.allclose(sums, torch.ones_like(sums), atol=2e-6, rtol=2e-6):
            raise ValueError("Interpolation weights must sum to one in every Gaussian row")

        def edge_arrays(prefix: str, positions: Tensor) -> tuple[Tensor, Tensor]:
            edge = tensor(prefix + "_edges", torch.long)
            if edge.ndim != 2:
                raise ValueError(f"{prefix}_edges must have shape [E,2]")
            if edge.shape[1] != 2 and edge.shape[0] == 2:
                edge = edge.T.contiguous()
            if edge.shape[1] != 2 or bool((edge < 0).any()) or bool((edge >= len(positions)).any()):
                raise ValueError(f"Invalid {prefix}_edges")
            name = prefix + "_edge_weights"
            edge_weights = tensor(name) if name in arrays else points.new_ones(len(edge))
            if (edge_weights.shape != (len(edge),)
                    or not bool(torch.isfinite(edge_weights).all())
                    or bool((edge_weights < 0).any())):
                raise ValueError(f"Invalid {name}")
            if len(edge) and bool((edge[:, 0] == edge[:, 1]).any()):
                raise ValueError(f"{prefix}_edges contain self edges")
            # Distinct controls can coincide in space while their material path
            # has positive length, for example on folds or duplicate centers.
            uses_path_lengths = prefix == "control" and "control_edge_lengths" in arrays
            if not uses_path_lengths and len(edge) and bool((torch.linalg.vector_norm(
                positions[edge[:, 1]] - positions[edge[:, 0]], dim=-1,
            ) <= 0).any()):
                raise ValueError(f"{prefix}_edges contain zero-length edges")
            return edge, edge_weights

        gaussian_edges, gaussian_weights = edge_arrays("gaussian", points)
        control_edges, control_weights = edge_arrays("control", controls)
        control_lengths = (tensor("control_edge_lengths") if "control_edge_lengths" in arrays else
                           torch.linalg.vector_norm(
                               controls[control_edges[:, 1]] - controls[control_edges[:, 0]], dim=-1,
                           ))
        if (control_lengths.shape != (len(control_edges),)
                or not bool(torch.isfinite(control_lengths).all())
                or bool((control_lengths <= 0).any())):
            raise ValueError("control_edge_lengths must be finite positive graph-path lengths")

        def supported(name: str, size: int) -> Tensor:
            value = (tensor(name, torch.bool) if name in arrays
                     else torch.ones(size, dtype=torch.bool, device=device))
            if value.shape != (size,):
                raise ValueError(f"{name} must have shape [{size}]")
            return value

        return cls(points, controls, indptr, indices, weights, rows,
                   gaussian_edges, gaussian_weights, control_edges, control_weights, control_lengths,
                   supported("gaussian_supported", count),
                   supported("control_supported", control_count))


def _positive_scale(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def weighted_neighbor_mean(features: Tensor, edges: Tensor, weights: Tensor) -> Tensor:
    """Undirected fixed-weight mean; isolated nodes receive a zero message."""
    if not len(edges):
        return torch.zeros_like(features)
    source = torch.cat((edges[:, 0], edges[:, 1]))
    destination = torch.cat((edges[:, 1], edges[:, 0]))
    directed_weights = torch.cat((weights, weights))
    numerator = torch.zeros_like(features).index_add_(
        0, destination, features[source] * directed_weights[:, None],
    )
    denominator = features.new_zeros(len(features)).index_add_(0, destination, directed_weights)
    return numerator / denominator.clamp_min(torch.finfo(features.dtype).tiny)[:, None]


class PerFrequencyModalGNN(nn.Module):
    """One frequency's residual GNN; no parameters are shared across frequencies."""

    def __init__(self, config: NeuralFieldConfig | None = None) -> None:
        super().__init__()
        self.config = config or NeuralFieldConfig()
        self.config.validate()
        width = self.config.hidden_dim
        self.encoder = nn.Sequential(nn.Linear(3, width), nn.SiLU())
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(self.config.message_layers)
        ])
        self.head = nn.Linear(width + 3, 12)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, geometry: NeuralFieldGeometry, length_scale: float) -> tuple[Tensor, Tensor]:
        length = _positive_scale(length_scale, "length_scale")
        coordinates = (geometry.control_positions - geometry.control_positions.mean(0)) / length
        hidden = self.encoder(coordinates)
        for layer in self.layers:
            message = weighted_neighbor_mean(hidden, geometry.control_edges, geometry.control_edge_weights)
            hidden = hidden + layer(torch.cat((hidden, message), dim=-1))
        values = self.head(torch.cat((hidden, coordinates), dim=-1))
        values = values * geometry.control_supported[:, None]
        displacement = torch.complex(values[:, :3], values[:, 3:6])
        rotation = torch.complex(values[:, 6:9], values[:, 9:12])
        return displacement, rotation


@dataclass(frozen=True)
class ComposedField:
    field: Tensor
    rotation: Tensor
    control_displacement: Tensor
    control_rotation: Tensor


def compose_field(geometry: NeuralFieldGeometry, displacement: Tensor, rotation: Tensor) -> ComposedField:
    """Interpolate original-unit complex control fields with local rotations."""
    expected = geometry.control_positions.shape
    if displacement.shape != expected or rotation.shape != expected:
        raise ValueError("Control displacement and rotation must have shape [C,3]")
    if not displacement.is_complex() or not rotation.is_complex():
        raise ValueError("Control fields must be complex tensors")
    displacement = displacement * geometry.control_supported[:, None]
    rotation = rotation * geometry.control_supported[:, None]
    rows, indices, weights = (geometry.interpolation_rows, geometry.interpolation_indices,
                              geometry.interpolation_weights)
    offset = (geometry.gaussian_positions[rows] - geometry.control_positions[indices]).to(displacement.dtype)
    contributions = displacement[indices] + torch.linalg.cross(rotation[indices], offset, dim=-1)
    field = displacement.new_zeros(geometry.gaussian_positions.shape).index_add_(
        0, rows, weights[:, None] * contributions,
    )
    blended_rotation = rotation.new_zeros(geometry.gaussian_positions.shape).index_add_(
        0, rows, weights[:, None] * rotation[indices],
    )
    return ComposedField(field * geometry.gaussian_supported[:, None],
                         blended_rotation * geometry.gaussian_supported[:, None],
                         displacement, rotation)


def model_field(model: PerFrequencyModalGNN, geometry: NeuralFieldGeometry, *,
                length_scale: float, amplitude_scale: float) -> ComposedField:
    amplitude = _positive_scale(amplitude_scale, "amplitude_scale")
    length = _positive_scale(length_scale, "length_scale")
    displacement, rotation = model(geometry, length)
    return compose_field(geometry, amplitude * displacement, (amplitude / length) * rotation)


def structural_losses(geometry: NeuralFieldGeometry, field: ComposedField, *,
                      length_scale: float, amplitude_scale: float,
                      rotation_length_fraction: float = 0.05) -> tuple[Tensor, Tensor]:
    """Dimensionless final-Gaussian strain and control-rotation variation."""
    length = _positive_scale(length_scale, "length_scale")
    amplitude = _positive_scale(amplitude_scale, "amplitude_scale")
    bending_length = _positive_scale(rotation_length_fraction, "rotation_length_fraction")
    normalized_field = field.field / amplitude
    normalized_rotation = field.rotation * (length / amplitude)
    zero = normalized_field.real.sum() * 0.0

    def valid_edges(edges: Tensor, weights: Tensor, support: Tensor) -> tuple[Tensor, Tensor]:
        keep = support[edges[:, 0]] & support[edges[:, 1]] & (weights > 0)
        return edges[keep], weights[keep]

    edges, weights = valid_edges(geometry.gaussian_edges, geometry.gaussian_edge_weights,
                                 geometry.gaussian_supported)
    edge_loss = zero
    if len(edges):
        left, right = edges[:, 0], edges[:, 1]
        edge = (geometry.gaussian_positions[right] - geometry.gaussian_positions[left]) / length
        distance = torch.linalg.vector_norm(edge, dim=-1)
        direction = (edge / distance[:, None]).to(normalized_field.dtype)
        mean_rotation = (normalized_rotation[left] + normalized_rotation[right]) * 0.5
        residual = ((normalized_field[right] - normalized_field[left]) / distance[:, None]
                    - torch.linalg.cross(mean_rotation, direction, dim=-1))
        edge_loss = (weights * residual.abs().square().sum(-1)).sum() / weights.sum()

    control_keep = (geometry.control_supported[geometry.control_edges[:, 0]]
                    & geometry.control_supported[geometry.control_edges[:, 1]]
                    & (geometry.control_edge_weights > 0))
    edges, weights = geometry.control_edges[control_keep], geometry.control_edge_weights[control_keep]
    rotation_loss = zero
    if len(edges):
        left, right = edges[:, 0], edges[:, 1]
        # The control graph follows material paths; close folds can have long paths.
        distance = geometry.control_edge_lengths[control_keep] / length
        controls = field.control_rotation * (length / amplitude)
        residual = (controls[right] - controls[left]) * (bending_length / distance[:, None])
        rotation_loss = (weights * residual.abs().square().sum(-1)).sum() / weights.sum()
    return edge_loss, rotation_loss


def radial_huber(residual: Tensor, delta: float = 1.0) -> Tensor:
    """Huber of each pixel's joint complex-vector radius, with finite zero gradient."""
    threshold = _positive_scale(delta, "huber_delta")
    squared = residual.abs().square().sum(dim=-1)
    radius = torch.sqrt(squared.clamp_min(torch.finfo(squared.dtype).tiny))
    return torch.where(squared <= threshold * threshold, squared * 0.5,
                       threshold * (radius - threshold * 0.5))


def modal_image_rms(target: Tensor, confidence: Tensor | None = None) -> Tensor:
    """Confidence-weighted RMS of the full complex 2D pixel vector."""
    if target.ndim != 2 or target.shape[-1] != 2:
        raise ValueError("Modal target must have shape [P,2]")
    weights = target.real.new_ones(len(target)) if confidence is None else confidence
    if weights.shape != (len(target),) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("Confidence must be a finite non-negative pixel vector")
    if not bool(torch.isfinite(target).all()) or float(weights.sum()) <= 0:
        raise ValueError("Modal target needs finite values and positive total confidence")
    return torch.sqrt((weights * target.abs().square().sum(-1)).sum() / weights.sum())


def fixed_rms_scales(values: Tensor, floor_fraction: float = 0.05,
                     absolute_floor: float = 1.0e-12) -> Tensor:
    """Freeze per-block RMS with a global fraction-of-positive-median floor."""
    fraction = _positive_scale(floor_fraction, "floor_fraction")
    minimum = _positive_scale(absolute_floor, "absolute_floor")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("RMS values must be finite non-negative real floating values")
    positive = values[values > 0]
    floor = values.new_tensor(minimum)
    if positive.numel():
        floor = torch.maximum(floor, fraction * torch.quantile(positive, 0.5))
    return values.clamp_min(floor).detach()


def observation_amplitude_scale(observed_energy: float, projection_energy: float, *,
                                length_scale: float, floor_fraction: float = 1.0e-6) -> float:
    """S=sqrt(sum(w|M|²)/sum(w|alpha|² sensitivity²)); no physical calibration."""
    length = _positive_scale(length_scale, "length_scale")
    floor = _positive_scale(floor_fraction, "floor_fraction") * length
    numerator, denominator = float(observed_energy), float(projection_energy)
    if not math.isfinite(numerator) or numerator < 0 or not math.isfinite(denominator) or denominator < 0:
        raise ValueError("Amplitude-scale energies must be finite and non-negative")
    if denominator == 0:
        if numerator > 0:
            raise ValueError("Nonzero observations have zero projection sensitivity")
        return floor
    return max(floor, math.sqrt(numerator / denominator))


@dataclass(frozen=True)
class ModalObservation:
    target: Tensor
    project: Callable[[Tensor], Tensor]
    alpha: complex = 1.0 + 0.0j
    confidence: Tensor | None = None
    normalized_rms: float | Tensor | None = None
    name: str = ""


@dataclass(frozen=True)
class TrainingResult:
    field: Tensor
    rotation: Tensor
    control_displacement: Tensor
    control_rotation: Tensor
    best_model_state: dict[str, Any]
    latest_state: dict[str, Any]
    history: list[dict[str, Any]]
    iterations: int
    best_step: int
    best_loss: float
    converged: bool

    @property
    def model_state(self) -> dict[str, Any]:
        return self.best_model_state


def _cpu_snapshot(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _validate_finite_payload(value: Any, context: str) -> None:
    """Reject corrupted model/optimizer payloads before evaluation or resume."""
    if isinstance(value, Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{context} contains a non-finite tensor")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_payload(item, f"{context}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_finite_payload(item, f"{context}[{index}]")
    elif isinstance(value, (float, complex)):
        number = complex(value)
        if not math.isfinite(number.real) or not math.isfinite(number.imag):
            raise ValueError(f"{context} contains a non-finite number")


@torch.no_grad()
def evaluate_model(model_state: Mapping[str, Any], geometry: NeuralFieldGeometry,
                   length_scale: float, amplitude_scale: float,
                   config: NeuralFieldConfig | Mapping[str, Any] | None = None
                   ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reconstruct exported original-unit fields from the saved best network."""
    settings = (NeuralFieldConfig.from_dict(config) if isinstance(config, Mapping)
                else config or NeuralFieldConfig())
    _validate_finite_payload(model_state, "Neural model state")
    # Initializing a throwaway evaluation model must not change training RNG.
    with torch.random.fork_rng(devices=[]):
        model = PerFrequencyModalGNN(settings).to(geometry.gaussian_positions)
    model.load_state_dict(model_state)
    model.eval()
    result = model_field(model, geometry, length_scale=length_scale, amplitude_scale=amplitude_scale)
    _validate_finite_payload((result.field, result.rotation, result.control_displacement,
                              result.control_rotation), "Evaluated neural field")
    return result.field, result.rotation, result.control_displacement, result.control_rotation


def train_single_frequency(
    geometry: NeuralFieldGeometry | Mapping[str, Any], observations: Sequence[ModalObservation], *,
    length_scale: float, amplitude_scale: float,
    config: NeuralFieldConfig | Mapping[str, Any] | None = None,
    resume_state: Mapping[str, Any] | None = None,
    checkpoint_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> TrainingResult:
    """Train all views each step, then structure once; save latest independently of best.

    ``max_iterations`` is the total update count, including resumed updates.
    Checkpoints describe an evaluated model after exactly ``step`` updates and
    carry optimizer/RNG/patience state.  The caller owns input identity checks
    and all serialization.  Callback payloads are detached CPU snapshots.
    """
    if not isinstance(geometry, NeuralFieldGeometry):
        geometry = NeuralFieldGeometry.from_arrays(geometry)
    settings = (NeuralFieldConfig.from_dict(config) if isinstance(config, Mapping)
                else config or NeuralFieldConfig())
    settings.validate()
    length = _positive_scale(length_scale, "length_scale")
    amplitude = _positive_scale(amplitude_scale, "amplitude_scale")
    if not observations:
        raise ValueError("Single-frequency training requires effective observations")
    points = geometry.gaussian_positions
    complex_dtype = torch.complex64 if points.dtype == torch.float32 else torch.complex128
    prepared: list[tuple[ModalObservation, Tensor, Tensor, complex]] = []
    measured_rms: list[Tensor] = []
    for observation in observations:
        target = torch.as_tensor(observation.target, device=points.device, dtype=complex_dtype).detach()
        confidence = (points.new_ones(len(target)) if observation.confidence is None else
                      torch.as_tensor(observation.confidence, device=points.device, dtype=points.dtype).detach())
        rms = modal_image_rms(target, confidence)
        alpha = complex(observation.alpha)
        if not math.isfinite(alpha.real) or not math.isfinite(alpha.imag) or abs(alpha) == 0:
            raise ValueError("Effective observation alpha must be finite and nonzero")
        prepared.append((observation, target, confidence / confidence.sum(), alpha))
        measured_rms.append(rms)
    automatic_rms = fixed_rms_scales(torch.stack(measured_rms))
    scales = [(_positive_scale(float(observation.normalized_rms), "normalized_rms")
               if observation.normalized_rms is not None else float(automatic_rms[index]))
              for index, observation in enumerate(observations)]

    torch.manual_seed(settings.seed)
    model = PerFrequencyModalGNN(settings).to(points)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    step, best_step, stale_steps = 0, 0, 0
    best_loss = math.inf
    best_state: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    latest_state: dict[str, Any] | None = None
    already_recorded = False
    if resume_state is not None:
        _validate_finite_payload(resume_state, "Neural checkpoint")
        if resume_state.get("version") != 1:
            raise ValueError("Unsupported neural field checkpoint version")
        previous_config = dict(resume_state["config"])
        current_config = settings.to_dict()
        for mutable in ("max_iterations", "checkpoint_every"):
            previous_config.pop(mutable, None)
            current_config.pop(mutable, None)
        if previous_config != current_config:
            raise ValueError("Resume changes the neural model or optimizer configuration")
        if float(resume_state["length_scale"]) != length or float(resume_state["amplitude_scale"]) != amplitude:
            raise ValueError("Resume changes fixed neural field scales")
        expected_shape = [len(points), len(geometry.control_positions)]
        if list(resume_state["geometry_shape"]) != expected_shape:
            raise ValueError("Resume changes neural geometry dimensions")
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        step = int(resume_state["step"])
        if step > settings.max_iterations:
            raise ValueError("max_iterations precedes the resumed step")
        best_step, best_loss = int(resume_state["best_step"]), float(resume_state["best_loss"])
        stale_steps = int(resume_state["stale_steps"])
        best_state = _cpu_snapshot(resume_state["best_model_state"])
        history = copy.deepcopy(resume_state["history"])
        torch.random.set_rng_state(resume_state["rng_state"]["cpu"].cpu())
        cuda_rng = resume_state["rng_state"].get("cuda", [])
        if cuda_rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)
        already_recorded = True

    def objective(backward: bool) -> dict[str, float]:
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            field = model_field(model, geometry, length_scale=length, amplitude_scale=amplitude)
            data_value = 0.0
            # Backpropagate one view at a time; do not retain all rasterizer graphs.
            for (observation, target, confidence, alpha), scale in zip(prepared, scales):
                prediction = observation.project(field.field)
                if prediction.shape != target.shape:
                    raise ValueError(f"Projection shape differs from target for {observation.name!r}")
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError("Non-finite neural modal projection")
                residual = (alpha * prediction - target) / scale
                term = (confidence * radial_huber(residual, settings.huber_delta)).sum() / len(prepared)
                data_value += float(term.detach())
                if backward:
                    term.backward(retain_graph=True)
                del prediction, residual, term
            edge, rotation = structural_losses(
                geometry, field, length_scale=length, amplitude_scale=amplitude,
                rotation_length_fraction=settings.rotation_length_fraction,
            )
            regularizer = settings.deformation_weight * edge + settings.rotation_weight * rotation
            if backward:
                regularizer.backward()
            edge_value, rotation_value = float(edge.detach()), float(rotation.detach())
            total = data_value + settings.deformation_weight * edge_value + settings.rotation_weight * rotation_value
            if not math.isfinite(total):
                raise FloatingPointError("Non-finite neural field objective")
            return {"loss": total, "data_loss": data_value, "edge_loss": edge_value,
                    "rotation_loss": rotation_value}

    def snapshot(losses: Mapping[str, float], status: str) -> dict[str, Any]:
        return _cpu_snapshot({
            "version": 1, "step": step, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "best_model_state": best_state,
            "best_step": best_step, "best_loss": best_loss, "stale_steps": stale_steps,
            "history": history, "last_losses": dict(losses), "status": status,
            "rng_state": {"cpu": torch.random.get_rng_state(),
                          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
            "config": settings.to_dict(), "length_scale": length, "amplitude_scale": amplitude,
            "geometry_shape": [len(points), len(geometry.control_positions)],
        })

    while True:
        can_update = step < settings.max_iterations
        losses = objective(backward=can_update)
        if not already_recorded:
            improvement = best_loss - losses["loss"]
            significant = (not math.isfinite(best_loss)
                           or improvement > settings.relative_tolerance * max(abs(best_loss), 1.0e-12))
            if losses["loss"] < best_loss:
                best_loss, best_step = losses["loss"], step
                best_state = _cpu_snapshot(model.state_dict())
            stale_steps = 0 if significant else stale_steps + 1
            history.append({"step": step, **losses})
        already_recorded = False
        converged = stale_steps >= settings.convergence_patience
        finished = not can_update or converged
        if finished or (step > 0 and step % settings.checkpoint_every == 0):
            latest_state = snapshot(losses, "converged" if converged else "complete" if finished else "running")
            if checkpoint_callback is not None:
                checkpoint_callback(step, latest_state)
        if finished:
            break
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("Non-finite neural field gradient")
        optimizer.step()
        step += 1

    # The terminating iteration always snapshots its evaluated latest state.
    assert latest_state is not None
    # Never overwrite the actual latest optimizer/model pair with the best model.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        field = model_field(model, geometry, length_scale=length, amplitude_scale=amplitude)
    return TrainingResult(field.field, field.rotation, field.control_displacement,
                          field.control_rotation, best_state, latest_state, history,
                          step, best_step, best_loss, converged)


__all__ = ["NeuralFieldConfig", "NeuralFieldGeometry", "PerFrequencyModalGNN", "ComposedField",
           "ModalObservation", "TrainingResult", "weighted_neighbor_mean", "compose_field",
           "model_field", "structural_losses", "radial_huber", "modal_image_rms",
           "fixed_rms_scales", "observation_amplitude_scale", "evaluate_model",
           "train_single_frequency"]
