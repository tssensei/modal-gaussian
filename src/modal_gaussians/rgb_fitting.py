"""Fit per-frame complex coordinates against RGB, keeping the spatial modes fixed."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class RGBFitConfig:
    scales: tuple[float, ...] = (0.25, 0.5, 1.0)
    epochs_per_scale: int = 10
    learning_rate: float = 0.01
    final_learning_rate: float = 0.001
    anchor_weight: float = 1e-4
    final_anchor_weight: float = 1e-6
    offset_steps: int = 30
    batch_size: int = 4
    seed: int = 1729

    def validate(self) -> None:
        if not isinstance(self.scales, (tuple, list)) or not self.scales or any(
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or not 0.0 < scale <= 1.0
            for scale in self.scales
        ):
            raise ValueError("RGB-fit scales must be finite and in (0, 1]")
        if any(a >= b for a, b in zip(self.scales, self.scales[1:])):
            raise ValueError("RGB-fit scales must be strictly increasing")
        for name in ("learning_rate", "final_learning_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"RGB-fit {name} must be finite and positive")
        for name in ("anchor_weight", "final_anchor_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"RGB-fit {name} must be finite and non-negative")
        for name in ("epochs_per_scale", "offset_steps", "batch_size", "seed"):
            value = getattr(self, name)
            minimum = 0 if name in ("offset_steps", "seed") else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"RGB-fit {name} must be an integer >= {minimum}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rgb_loss(prediction: Tensor, target: Tensor) -> Tensor:
    from pytorch_msssim import ssim

    if (
        prediction.ndim != 3
        or prediction.shape[-1] != 3
        or min(prediction.shape[:2]) < 2
        or prediction.shape != target.shape
    ):
        raise ValueError("RGB-fit render and target must have matching [H>=2, W>=2, 3] shapes")
    if not torch.all(torch.isfinite(target) & (target >= 0.0) & (target <= 1.0)):
        raise ValueError("RGB-fit target must contain finite RGB values in [0, 1]")
    window = min(11, min(prediction.shape[:2]))
    window -= 1 - window % 2
    structural = ssim(
        prediction.permute(2, 0, 1)[None],
        target.permute(2, 0, 1)[None],
        data_range=1.0,
        size_average=True,
        win_size=window,
    )
    return 0.8 * (prediction - target).abs().mean() + 0.2 * (1.0 - structural)


def solve_rgb_coordinates_view(
    initial: np.ndarray,
    pair_scales: np.ndarray,
    reference_index: int,
    render: Callable[[Tensor, float], Tensor],
    target: Callable[[int, float], Tensor],
    config: RGBFitConfig = RGBFitConfig(),
    device: str | torch.device = "cuda",
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize one independently captured sequence with frozen rendering inputs.

    ``render(q, scale)`` receives the original-unit complex [K] coefficients;
    it must return differentiable float RGB [H, W, 3]. ``target(frame, scale)``
    supplies the corresponding [0, 1] image, including the moving silhouette
    and its surrounding background. Both callbacks must use the same resize.

    Internally p = pair_scales * q equalizes the mode amplitudes. A shared
    offset first places reference-relative flow coordinates in the static
    scene's frame. This result anchors the subsequent unconstrained per-frame
    fit; no zero-mean, temporal, or oscillator constraint is applied.
    """
    config.validate()
    initial = np.asarray(initial)
    pair_scales = np.asarray(pair_scales)
    if (
        initial.ndim != 2
        or min(initial.shape) < 1
        or not np.iscomplexobj(initial)
        or not np.isfinite(initial).all()
    ):
        raise ValueError("RGB-fit initial coordinates must be a finite complex [T, K] array")
    if (
        pair_scales.shape != (initial.shape[1],)
        or np.iscomplexobj(pair_scales)
        or not np.isfinite(pair_scales).all()
        or np.any(pair_scales <= 0.0)
    ):
        raise ValueError("RGB-fit pair_scales must be finite positive real [K] values")
    if (
        isinstance(reference_index, bool)
        or not isinstance(reference_index, (int, np.integer))
        or not 0 <= reference_index < len(initial)
    ):
        raise ValueError("RGB-fit reference_index is outside the input sequence")

    scales = torch.tensor(pair_scales, dtype=torch.float32, device=device)
    relative = torch.tensor(initial, dtype=torch.complex64, device=device)
    relative = relative - relative[reference_index].clone()
    normalized = torch.view_as_real(relative * scales).clone()
    if not torch.isfinite(normalized).all() or not torch.all(torch.isfinite(scales) & (scales > 0)):
        raise ValueError("RGB-fit coordinates and scales must be representable in float32")
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, Any]] = []

    def record(row: dict[str, Any]) -> None:
        history.append(row)
        if on_progress is not None:
            on_progress(dict(row))

    def complex_coordinates(value: Tensor) -> Tensor:
        return torch.complex(value[..., 0], value[..., 1]) / scales

    def photometric(value: Tensor, frame: int, scale: float) -> Tensor:
        prediction = render(complex_coordinates(value), scale)
        observed = target(frame, scale).detach().to(device=prediction.device, dtype=prediction.dtype)
        loss = _rgb_loss(prediction, observed)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite RGB-fit loss for frame {frame} at scale {scale}")
        return loss

    # Only this offset is shared; independently captured videos call this solver separately.
    offset = torch.nn.Parameter(torch.zeros_like(normalized[0]))
    offset_optimizer = torch.optim.Adam([offset], lr=config.learning_rate)
    for step in range(config.offset_steps):
        frames = rng.choice(len(initial), min(config.batch_size, len(initial)), replace=False)
        offset_optimizer.zero_grad(set_to_none=True)
        rgb_sum = 0.0
        for index in frames:
            loss = photometric(normalized[index] + offset, int(index), config.scales[0])
            (loss / len(frames)).backward(inputs=[offset])
            rgb_sum += float(loss.detach())
        if offset.grad is None or not torch.isfinite(offset.grad).all():
            raise ValueError("Non-finite or disconnected RGB-fit offset gradient")
        offset_optimizer.step()
        record({"phase": "offset", "step": step + 1, "scale": config.scales[0],
                "rgb_loss": rgb_sum / len(frames)})

    anchor = (normalized + offset.detach()).detach()
    # Separate leaves let Adam skip unsampled frames completely (including stale momentum).
    coordinates = [torch.nn.Parameter(row.clone()) for row in anchor]
    optimizer = torch.optim.Adam(coordinates, lr=config.learning_rate)
    total_epochs = len(config.scales) * config.epochs_per_scale
    epoch_index = 0
    for scale in config.scales:
        for epoch in range(config.epochs_per_scale):
            fraction = epoch_index / max(1, total_epochs - 1)
            learning_rate = config.learning_rate + fraction * (config.final_learning_rate - config.learning_rate)
            anchor_weight = config.anchor_weight + fraction * (config.final_anchor_weight - config.anchor_weight)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            order = rng.permutation(len(initial))
            rgb_sum = anchor_sum = 0.0
            for start in range(0, len(order), config.batch_size):
                frames = order[start:start + config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                for index in frames:
                    parameter = coordinates[index]
                    rgb = photometric(parameter, int(index), scale)
                    penalty = (parameter - anchor[index]).square().sum(-1).mean()
                    loss = rgb + anchor_weight * penalty
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite RGB-fit objective for frame {index}")
                    (loss / len(frames)).backward(inputs=[parameter])
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise ValueError(f"Non-finite or disconnected RGB-fit gradient for frame {index}")
                    rgb_sum += float(rgb.detach())
                    anchor_sum += float(penalty.detach())
                optimizer.step()
            epoch_index += 1
            record({"phase": "coordinates", "epoch": epoch_index, "scale_epoch": epoch + 1,
                    "scale": scale, "learning_rate": learning_rate, "anchor_weight": anchor_weight,
                    "rgb_loss": rgb_sum / len(initial), "anchor_loss": anchor_sum / len(initial)})

    result = complex_coordinates(torch.stack([row.detach() for row in coordinates])).cpu().numpy()
    shared_offset = complex_coordinates(offset.detach()).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("RGB-fit produced non-finite coordinates")
    return result, {
        "config": config.to_dict(),
        "frame_count": len(initial),
        "mode_count": initial.shape[1],
        "reference_index": int(reference_index),
        "shared_offset_real": shared_offset.real.tolist(),
        "shared_offset_imag": shared_offset.imag.tolist(),
        "history": history,
        "loss_sampling": "training_steps_before_update; no evaluation renders",
    }
