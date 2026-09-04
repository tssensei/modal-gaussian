"""Training, density control, bundle export, and offline QA for static 3DGS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any, Literal, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from modal_gaussians.static import (
    GAUSSIAN_FIELDS,
    Camera,
    ForegroundBackgroundScene,
    GaussianSet,
    StaticDataset,
    cameras_from_scene_manifest,
    initialize_static_scene,
    load_static_dataset,
    load_static_scene,
    tensor_dictionary_identity,
)
from modal_gaussians.progress import Progress, report_progress


@dataclass(frozen=True)
class StaticTrainConfig:
    """Hold the intentionally small accepted static-training configuration."""

    epochs: int = 100
    batch_size: int = 8
    num_foreground: int = 40_000
    num_background: int = 80_000
    seed: int = 42
    mask_loss_weight: float = 1.0
    mask_erosion_kernel_size: int = 7
    mask_trim_quantile: float = 0.98
    checkpoint_every_steps: int = 200
    max_lr_steps: int = 5_000
    density_warmup_steps: int = 200
    density_control_every: int = 100
    density_stop_step: int = 4_000
    foreground_densify_stop_step: int = 4_000
    background_densify_stop_step: int = 1_000
    max_background_gaussians: int = 160_000
    opacity_reset_every: int = 3_000
    densify_gradient_threshold: float = 2e-4
    densify_scale_threshold: float = 0.01
    densify_screen_threshold: float = 0.05
    cull_opacity_threshold: float = 0.1
    cull_scale_threshold: float = 0.5
    cull_screen_threshold: float = 0.15

    def validate(self) -> None:
        """Reject nonsensical values before creating any work directory."""

        positive_integers = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "num_foreground": self.num_foreground,
            "num_background": self.num_background,
            "mask_erosion_kernel_size": self.mask_erosion_kernel_size,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "max_lr_steps": self.max_lr_steps,
            "density_control_every": self.density_control_every,
            "density_stop_step": self.density_stop_step,
            "foreground_densify_stop_step": self.foreground_densify_stop_step,
            "background_densify_stop_step": self.background_densify_stop_step,
            "max_background_gaussians": self.max_background_gaussians,
            "opacity_reset_every": self.opacity_reset_every,
        }
        invalid = [name for name, value in positive_integers.items() if value <= 0]
        if invalid:
            raise ValueError(f"Static training values must be positive: {invalid}")
        if self.density_warmup_steps < 0:
            raise ValueError("density_warmup_steps must be non-negative")
        if self.mask_erosion_kernel_size % 2 == 0:
            raise ValueError("mask_erosion_kernel_size must be odd")
        if self.max_background_gaussians < self.num_background:
            raise ValueError(
                "max_background_gaussians must not be smaller than num_background"
            )
        if not (
            self.density_warmup_steps
            < self.background_densify_stop_step
            <= self.foreground_densify_stop_step
            <= self.density_stop_step
        ):
            raise ValueError(
                "Density schedule must satisfy warmup < background stop <= "
                "foreground stop <= control stop"
            )
        finite_nonnegative = {
            "mask_loss_weight": self.mask_loss_weight,
            "densify_gradient_threshold": self.densify_gradient_threshold,
            "densify_scale_threshold": self.densify_scale_threshold,
            "densify_screen_threshold": self.densify_screen_threshold,
            "cull_opacity_threshold": self.cull_opacity_threshold,
            "cull_scale_threshold": self.cull_scale_threshold,
            "cull_screen_threshold": self.cull_screen_threshold,
        }
        invalid_float = [
            name
            for name, value in finite_nonnegative.items()
            if not math.isfinite(value) or value < 0.0
        ]
        if invalid_float:
            raise ValueError(f"Static training values must be finite/non-negative: {invalid_float}")
        if not math.isfinite(self.mask_trim_quantile) or not (
            0.0 < self.mask_trim_quantile <= 1.0
        ):
            raise ValueError("mask_trim_quantile must be finite and in (0, 1]")

    def resolved(self) -> dict[str, Any]:
        """Serialize configuration together with fixed loss and LR conventions."""

        return {
            **asdict(self),
            "representation": "vanilla_3dgs_direct_rgb",
            "training_camera_roles": ["sweep", "reference"],
            "camera_optimization": False,
            "distortion_applied": False,
            "loss": {
                "rgb_l1_weight": 0.8,
                "rgb_dssim_weight": 0.2,
                "mask_weight": self.mask_loss_weight,
                "mask_type": "trimmed_l1_foreground_membership",
                "mask_trim_quantile": self.mask_trim_quantile,
                "mask_erosion_kernel_size": self.mask_erosion_kernel_size,
                "alpha_coverage_weight": 0.0,
                "depth_weights": [0.0, 0.0, 0.0],
            },
            "learning_rates": _learning_rates(),
        }


def _learning_rates() -> dict[str, dict[str, float]]:
    """Return the accepted field-wise FG/BG Adam learning rates."""

    return {
        "foreground": {
            "means": 1.6e-4,
            "opacity_logits": 1e-2,
            "log_scales": 5e-3,
            "quaternions": 1e-3,
            "color_logits": 1e-2,
        },
        "background": {
            "means": 1.6e-4,
            "opacity_logits": 5e-2,
            "log_scales": 5e-3,
            "quaternions": 1e-3,
            "color_logits": 1e-2,
        },
    }


def _trimmed_l1_loss(prediction: Tensor, target: Tensor, quantile: float) -> Tensor:
    """Average per-pixel L1 errors after discarding the largest tail."""

    errors = torch.abs(prediction - target)
    if errors.numel() == 0 or quantile >= 1.0:
        return errors.mean()
    threshold = torch.quantile(errors.detach(), quantile)
    retained = errors[errors <= threshold]
    return retained.mean() if retained.numel() > 0 else errors.mean()


def _sha256_file(path: Path) -> str:
    """Compute one file identity without loading the full artifact into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    """Hash one JSON-compatible value using canonical serialization."""

    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Write a torch checkpoint atomically inside its final directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_write(payload: Any, path: Path) -> None:
    """Write one JSON record atomically without leaving a partial report."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _set_random_seed(seed: int) -> None:
    """Seed every RNG used by initialization, view shuffling, and optimization."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scene_from_tensor_dictionary(tensors: Mapping[str, Tensor]) -> ForegroundBackgroundScene:
    """Recreate trainable FG/BG parameters from a class-free resume dictionary."""

    parts: dict[str, GaussianSet] = {}
    for part in ("foreground", "background"):
        raw = {field: tensors[f"{part}.{field}"].float() for field in GAUSSIAN_FIELDS}
        parts[part] = GaussianSet(**raw)
    return ForegroundBackgroundScene(parts["foreground"], parts["background"])


def _quaternion_to_rotation_matrix(quaternions: Tensor) -> Tensor:
    """Convert normalized wxyz quaternions to differentiable 3x3 rotations."""

    values = F.normalize(quaternions, dim=-1)
    w, x, y, z = values.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


class StaticTrainer:
    """Optimize a static scene and maintain separate FG/BG density-control state."""

    def __init__(
        self,
        scene: ForegroundBackgroundScene,
        dataset: StaticDataset,
        config: StaticTrainConfig,
        device: torch.device,
        work_dir: Path,
        classification_summary: Mapping[str, int],
    ) -> None:
        """Create optimizers, schedulers, and resumable running statistics."""

        self.scene = scene.to(device)
        self.dataset = dataset
        self.config = config
        self.device = device
        self.work_dir = work_dir
        self.classification_summary = dict(classification_summary)
        self.config_identity = _sha256_json(config.resolved())
        self.optimizers, self.schedulers = self._configure_optimizers()
        self.density_stats = self._new_density_stats()
        self.density_events: list[dict[str, Any]] = []
        self.epoch_summaries: list[dict[str, Any]] = []
        self.epoch = 0
        self.global_step = 0
        self.current_batches: list[list[int]] | None = None
        self.next_batch_index = 0
        self.epoch_accumulator = self._empty_epoch_accumulator()
        self.elapsed_seconds_before_resume = 0.0
        self.started_at = time.time()

    def elapsed_seconds(self) -> float:
        """Return active training-process time accumulated across resume boundaries."""

        return self.elapsed_seconds_before_resume + (time.time() - self.started_at)

    def _configure_optimizers(
        self,
    ) -> tuple[dict[str, torch.optim.Adam], dict[str, torch.optim.lr_scheduler.LambdaLR]]:
        """Create one Adam and one explicit scheduler for every Gaussian field."""

        optimizers: dict[str, torch.optim.Adam] = {}
        schedulers: dict[str, torch.optim.lr_scheduler.LambdaLR] = {}
        rates = _learning_rates()
        for part_name in ("foreground", "background"):
            part = getattr(self.scene, part_name)
            for field, parameter in part.params.items():
                name = f"{part_name}.{field}"
                learning_rate = rates[part_name][field]
                optimizer = torch.optim.Adam(
                    [{"params": [parameter], "lr": learning_rate, "name": name}]
                )
                if field == "log_scales":
                    decay = lambda step, maximum=self.config.max_lr_steps: 0.1 ** min(
                        float(step) / float(maximum), 1.0
                    )
                else:
                    decay = lambda _step: 1.0
                optimizers[name] = optimizer
                schedulers[name] = torch.optim.lr_scheduler.LambdaLR(optimizer, decay)
        return optimizers, schedulers

    def _new_density_stats(self) -> dict[str, Tensor]:
        """Allocate zeroed combined-order density statistics on the training device."""

        return {
            "screen_gradient_sum": torch.zeros(self.scene.count, device=self.device),
            "visibility_count": torch.zeros(
                self.scene.count, dtype=torch.int64, device=self.device
            ),
            "maximum_screen_radius": torch.zeros(
                self.scene.count, device=self.device
            ),
        }

    @staticmethod
    def _empty_epoch_accumulator() -> dict[str, float]:
        """Create resumable scalar sums for one epoch summary."""

        return {
            "batch_count": 0.0,
            "loss_sum": 0.0,
            "rgb_loss_sum": 0.0,
            "mask_loss_sum": 0.0,
            "psnr_sum": 0.0,
            "ssim_sum": 0.0,
        }

    def _make_epoch_batches(self) -> list[list[int]]:
        """Group views by resolution, batch within groups, then shuffle batch order."""

        groups: dict[tuple[int, int], list[int]] = {}
        for index, camera in enumerate(self.dataset.cameras):
            groups.setdefault((camera.width, camera.height), []).append(index)
        batches: list[list[int]] = []
        for indices in groups.values():
            random.shuffle(indices)
            batches.extend(
                indices[start : start + self.config.batch_size]
                for start in range(0, len(indices), self.config.batch_size)
            )
        random.shuffle(batches)
        return batches

    def _load_batch(
        self, indices: Sequence[int]
    ) -> tuple[list[Camera], Tensor, Tensor, Tensor]:
        """Load RGB plus eroded FG targets and valid non-boundary mask pixels."""

        cameras = [self.dataset.cameras[index] for index in indices]
        images = torch.stack(
            [self.dataset.load_rgb(camera) for camera in cameras], dim=0
        ).to(self.device)
        kernel_size = self.config.mask_erosion_kernel_size
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        foreground_masks: list[Tensor] = []
        valid_masks: list[Tensor] = []
        for camera in cameras:
            foreground = self.dataset.load_binary_mask(camera)
            foreground_eroded = cv2.erode(
                foreground.astype(np.uint8), kernel, iterations=1
            ).astype(bool)
            background_eroded = cv2.erode(
                (~foreground).astype(np.uint8), kernel, iterations=1
            ).astype(bool)
            foreground_masks.append(torch.from_numpy(foreground_eroded))
            valid_masks.append(torch.from_numpy(foreground_eroded | background_eroded))
        foreground_targets = torch.stack(foreground_masks, dim=0).to(
            self.device, dtype=images.dtype
        )
        valid = torch.stack(valid_masks, dim=0).to(
            self.device, dtype=images.dtype
        )
        return cameras, images, foreground_targets, valid

    def _compute_loss(
        self,
        cameras: Sequence[Camera],
        targets: Tensor,
        foreground_targets: Tensor,
        valid_masks: Tensor,
    ) -> tuple[Tensor, dict[str, float], Mapping[str, Tensor]]:
        """Compute RGB and semantic foreground-mask supervision."""

        rendered, info = self.scene.render_batch(
            cameras,
            composition="all",
            retain_screen_grad=True,
            return_foreground_mask=True,
        )
        predictions = rendered["rgb"]
        valid_rgb = valid_masks[..., None]
        masked_predictions = predictions * valid_rgb + (1.0 - valid_rgb)
        masked_targets = targets * valid_rgb + (1.0 - valid_rgb)
        try:
            from pytorch_msssim import ssim
        except ImportError as error:
            raise RuntimeError("Static RGB training requires pytorch-msssim") from error
        l1 = F.l1_loss(masked_predictions, masked_targets)
        structural = ssim(
            masked_predictions.permute(0, 3, 1, 2),
            masked_targets.permute(0, 3, 1, 2),
            data_range=1.0,
            size_average=True,
        )
        rgb_loss = 0.8 * l1 + 0.2 * (1.0 - structural)
        mask_loss = _trimmed_l1_loss(
            rendered["foreground_mask"],
            foreground_targets,
            self.config.mask_trim_quantile,
        )
        loss = rgb_loss + self.config.mask_loss_weight * mask_loss
        mse = F.mse_loss(masked_predictions.detach(), masked_targets)
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        stats = {
            "loss": float(loss.detach().item()),
            "rgb_loss": float(rgb_loss.detach().item()),
            "mask_loss": float(mask_loss.detach().item()),
            "psnr": float(psnr.item()),
            "ssim": float(structural.detach().item()),
        }
        return loss, stats, info

    @torch.no_grad()
    def _accumulate_density_stats(
        self,
        info: Mapping[str, Tensor],
        cameras: Sequence[Camera],
    ) -> None:
        """Accumulate normalized means2d gradients and maximum visible radii."""

        means2d = info.get("means2d")
        radii = info.get("radii")
        if means2d is None or radii is None or means2d.grad is None:
            raise RuntimeError("gsplat density control requires means2d gradients and radii")
        if radii.ndim == means2d.ndim and radii.shape[-1] == 2:
            visible = (radii > 0).all(dim=-1)
            radii = radii.amax(dim=-1)
        else:
            visible = radii > 0
        if means2d.ndim != 3 or radii.ndim != 2:
            raise ValueError(
                f"Unexpected gsplat density shapes: means2d={means2d.shape}, radii={radii.shape}"
            )
        gradients = means2d.grad.detach().clone()
        gradients[..., 0] *= cameras[0].width / 2.0 * len(cameras)
        gradients[..., 1] *= cameras[0].height / 2.0 * len(cameras)
        for camera_index in range(len(cameras)):
            indices = torch.nonzero(visible[camera_index], as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            self.density_stats["screen_gradient_sum"].index_add_(
                0, indices, gradients[camera_index, indices].norm(dim=-1)
            )
            self.density_stats["visibility_count"].index_add_(
                0,
                indices,
                torch.ones_like(indices, dtype=torch.int64),
            )
            normalized_radii = radii[camera_index, indices] / max(
                cameras[camera_index].width, cameras[camera_index].height
            )
            previous = self.density_stats["maximum_screen_radius"].index_select(
                0, indices
            )
            self.density_stats["maximum_screen_radius"].index_put_(
                (indices,), torch.maximum(previous, normalized_radii)
            )

    def _replace_optimizer_parameter(
        self,
        name: str,
        old_parameter: Tensor,
        new_parameter: Tensor,
        state_transform: Any,
    ) -> None:
        """Attach a resized parameter and consistently transform its Adam state."""

        optimizer = self.optimizers[name]
        old_state = optimizer.state.pop(old_parameter, {})
        optimizer.param_groups[0]["params"] = [new_parameter]
        if old_state:
            transformed: dict[str, Any] = {}
            for key, value in old_state.items():
                transformed[key] = state_transform(value)
            optimizer.state[new_parameter] = transformed

    @torch.no_grad()
    def _densify_part(self, part_name: str, split: Tensor, duplicate: Tensor) -> None:
        """Split/duplicate one semantic set without crossing its index domain."""

        part: GaussianSet = getattr(self.scene, part_name)
        old_count = part.count
        added_count = int(duplicate.sum().item()) + 2 * int(split.sum().item())
        split_means = part.params["means"].detach()[split]
        if split_means.numel() > 0:
            split_scales = torch.exp(part.params["log_scales"].detach()[split])
            split_rotations = _quaternion_to_rotation_matrix(
                part.params["quaternions"].detach()[split]
            )
            local_offsets = torch.randn(
                (len(split_means), 2, 3),
                device=self.device,
                dtype=split_means.dtype,
            ) * split_scales[:, None, :]
            world_offsets = torch.einsum(
                "nij,nkj->nki", split_rotations, local_offsets
            )
            child_means = (split_means[:, None, :] + world_offsets).reshape(-1, 3)
        else:
            child_means = split_means.new_empty((0, 3))
        for field in GAUSSIAN_FIELDS:
            old_parameter = part.params[field]
            value = old_parameter.detach()
            split_values = value[split].repeat_interleave(2, dim=0)
            if field == "means":
                split_values = child_means
            elif field == "log_scales":
                split_values = split_values - math.log(1.6)
            resized = torch.cat(
                [value[~split], value[duplicate], split_values], dim=0
            )
            new_parameter = part.replace_parameter(field, resized)

            def transform_state(
                state_value: Any,
                *,
                count: int = old_count,
                keep: Tensor = ~split,
                additions: int = added_count,
            ) -> Any:
                """Remove split parents and append zeroed state for new children."""

                if not isinstance(state_value, Tensor) or state_value.ndim == 0:
                    return state_value
                if state_value.shape[0] != count:
                    return state_value
                zeros = state_value.new_zeros((additions,) + state_value.shape[1:])
                return torch.cat([state_value[keep], zeros], dim=0)

            self._replace_optimizer_parameter(
                f"{part_name}.{field}",
                old_parameter,
                new_parameter,
                transform_state,
            )

    @staticmethod
    def _limit_densify_candidates(
        split: Tensor,
        duplicate: Tensor,
        scores: Tensor,
        maximum_additions: int,
    ) -> tuple[Tensor, Tensor, int]:
        """Keep the strongest candidates while respecting a Gaussian-count cap."""

        selected = split | duplicate
        requested = int(selected.sum().item())
        if requested <= maximum_additions:
            return split, duplicate, 0
        keep = torch.zeros_like(selected)
        if maximum_additions > 0:
            candidate_indices = torch.nonzero(selected, as_tuple=False).flatten()
            order = torch.argsort(
                scores[candidate_indices], descending=True, stable=True
            )
            keep[candidate_indices[order[:maximum_additions]]] = True
        return split & keep, duplicate & keep, requested - maximum_additions

    @torch.no_grad()
    def _densify(self, step: int) -> dict[str, Any]:
        """Densify FG/BG under independent stop times and the BG count cap."""

        visibility = self.density_stats["visibility_count"].clamp_min(1)
        gradients = self.density_stats["screen_gradient_sum"] / visibility
        high_gradient = gradients > self.config.densify_gradient_threshold
        scales = torch.cat(
            [
                self.scene.foreground.active()["scales"],
                self.scene.background.active()["scales"],
            ],
            dim=0,
        )
        large = scales.amax(dim=-1) > self.config.densify_scale_threshold
        large_screen = (
            self.density_stats["maximum_screen_radius"]
            > self.config.densify_screen_threshold
        )
        split = high_gradient & (large | large_screen)
        duplicate = high_gradient & ~split
        foreground_count = self.scene.foreground.count
        foreground_split = split[:foreground_count]
        foreground_duplicate = duplicate[:foreground_count]
        background_split = split[foreground_count:]
        background_duplicate = duplicate[foreground_count:]
        requested_foreground_split = int(foreground_split.sum().item())
        requested_foreground_duplicate = int(foreground_duplicate.sum().item())
        requested_background_split = int(background_split.sum().item())
        requested_background_duplicate = int(background_duplicate.sum().item())
        foreground_enabled = step < self.config.foreground_densify_stop_step
        background_enabled = step < self.config.background_densify_stop_step
        if not foreground_enabled:
            foreground_split = torch.zeros_like(foreground_split)
            foreground_duplicate = torch.zeros_like(foreground_duplicate)
        background_skipped_by_cap = 0
        if not background_enabled:
            background_split = torch.zeros_like(background_split)
            background_duplicate = torch.zeros_like(background_duplicate)
        else:
            background_capacity = max(
                self.config.max_background_gaussians
                - self.scene.background.count,
                0,
            )
            background_split, background_duplicate, background_skipped_by_cap = (
                self._limit_densify_candidates(
                    background_split,
                    background_duplicate,
                    gradients[foreground_count:],
                    background_capacity,
                )
            )
        decisions = {
            "foreground_densify_enabled": foreground_enabled,
            "background_densify_enabled": background_enabled,
            "foreground_requested_split": requested_foreground_split,
            "foreground_requested_duplicate": requested_foreground_duplicate,
            "background_requested_split": requested_background_split,
            "background_requested_duplicate": requested_background_duplicate,
            "foreground_split": int(foreground_split.sum().item()),
            "foreground_duplicate": int(foreground_duplicate.sum().item()),
            "background_split": int(background_split.sum().item()),
            "background_duplicate": int(background_duplicate.sum().item()),
            "background_skipped_by_cap": background_skipped_by_cap,
            "background_capacity_after_densify": max(
                self.config.max_background_gaussians
                - self.scene.background.count
                - int(background_split.sum().item())
                - int(background_duplicate.sum().item()),
                0,
            ),
        }
        self._densify_part(
            "foreground", foreground_split, foreground_duplicate
        )
        self._densify_part(
            "background", background_split, background_duplicate
        )
        remapped_stats: dict[str, Tensor] = {}
        for name, values in self.density_stats.items():
            foreground_values = values[:foreground_count]
            background_values = values[foreground_count:]
            remapped_stats[name] = torch.cat(
                [
                    foreground_values[~foreground_split],
                    foreground_values[foreground_duplicate],
                    foreground_values[foreground_split].repeat_interleave(2),
                    background_values[~background_split],
                    background_values[background_duplicate],
                    background_values[background_split].repeat_interleave(2),
                ]
            )
        self.density_stats = remapped_stats
        return decisions

    @torch.no_grad()
    def _cull_part(self, part_name: str, cull: Tensor) -> None:
        """Remove culled entries and the matching leading Adam-state rows."""

        part: GaussianSet = getattr(self.scene, part_name)
        if int((~cull).sum().item()) < 2:
            raise RuntimeError(f"Density control would leave fewer than two {part_name} Gaussians")
        old_count = part.count
        for field in GAUSSIAN_FIELDS:
            old_parameter = part.params[field]
            resized = old_parameter.detach()[~cull]
            new_parameter = part.replace_parameter(field, resized)

            def transform_state(
                state_value: Any,
                *,
                count: int = old_count,
                keep: Tensor = ~cull,
            ) -> Any:
                """Keep only Adam rows belonging to surviving Gaussians."""

                if not isinstance(state_value, Tensor) or state_value.ndim == 0:
                    return state_value
                if state_value.shape[0] != count:
                    return state_value
                return state_value[keep]

            self._replace_optimizer_parameter(
                f"{part_name}.{field}",
                old_parameter,
                new_parameter,
                transform_state,
            )

    @torch.no_grad()
    def _cull(self) -> dict[str, int]:
        """Cull low-opacity and late-stage oversized Gaussians in each set."""

        opacity = torch.cat(
            [
                self.scene.foreground.active()["opacities"],
                self.scene.background.active()["opacities"],
            ]
        )
        cull = opacity < self.config.cull_opacity_threshold
        scales = torch.cat(
            [
                self.scene.foreground.active()["scales"],
                self.scene.background.active()["scales"],
            ],
            dim=0,
        )
        cull |= scales.amax(dim=-1) > self.config.cull_scale_threshold
        cull |= (
            self.density_stats["maximum_screen_radius"]
            > self.config.cull_screen_threshold
        )
        foreground_count = self.scene.foreground.count
        decisions = {
            "foreground_culled": int(cull[:foreground_count].sum().item()),
            "background_culled": int(cull[foreground_count:].sum().item()),
        }
        foreground_cull = cull[:foreground_count]
        background_cull = cull[foreground_count:]
        self._cull_part("foreground", foreground_cull)
        self._cull_part("background", background_cull)
        self.density_stats = {
            name: torch.cat(
                [
                    values[:foreground_count][~foreground_cull],
                    values[foreground_count:][~background_cull],
                ]
            )
            for name, values in self.density_stats.items()
        }
        return decisions

    @torch.no_grad()
    def _reset_opacity(self) -> dict[str, int]:
        """Reset both sets to opacity 0.08 and clear their Adam state."""

        reset_value = float(torch.logit(torch.tensor(0.08)).item())
        for part_name in ("foreground", "background"):
            part: GaussianSet = getattr(self.scene, part_name)
            part.params["opacity_logits"].fill_(reset_value)
            optimizer = self.optimizers[f"{part_name}.opacity_logits"]
            state = optimizer.state.get(part.params["opacity_logits"], {})
            for value in state.values():
                if isinstance(value, Tensor):
                    value.zero_()
        return {
            "foreground_reset": self.scene.foreground.count,
            "background_reset": self.scene.background.count,
        }

    @torch.no_grad()
    def _run_density_control(self) -> None:
        """Run scheduled densify/cull/reset actions and then reset running stats."""

        step = self.global_step
        cfg = self.config
        should_control = (
            step > cfg.density_warmup_steps
            and step % cfg.density_control_every == 0
            and step < cfg.density_stop_step
        )
        if not should_control:
            return
        event: dict[str, Any] = {
            "step": step,
            "before_foreground": self.scene.foreground.count,
            "before_background": self.scene.background.count,
        }
        event.update(self._densify(step))
        event.update(self._cull())
        if step % cfg.opacity_reset_every == 0:
            event.update(self._reset_opacity())
        event["after_foreground"] = self.scene.foreground.count
        event["after_background"] = self.scene.background.count
        event["background_capacity_remaining"] = max(
            cfg.max_background_gaussians - self.scene.background.count, 0
        )
        self.density_events.append(event)
        self.density_stats = self._new_density_stats()
        report_progress(f"static density control: {event}")

    def _update_epoch_accumulator(self, stats: Mapping[str, float]) -> None:
        """Add one batch result to the resumable epoch scalar sums."""

        self.epoch_accumulator["batch_count"] += 1.0
        self.epoch_accumulator["loss_sum"] += stats["loss"]
        self.epoch_accumulator["rgb_loss_sum"] += stats["rgb_loss"]
        self.epoch_accumulator["mask_loss_sum"] += stats["mask_loss"]
        self.epoch_accumulator["psnr_sum"] += stats["psnr"]
        self.epoch_accumulator["ssim_sum"] += stats["ssim"]

    def _finish_epoch(self) -> dict[str, Any]:
        """Finalize mean metrics and reset state for the next epoch."""

        count = self.epoch_accumulator["batch_count"]
        if count <= 0:
            raise RuntimeError("Cannot finish an epoch without batches")
        summary = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "loss": self.epoch_accumulator["loss_sum"] / count,
            "rgb_loss": self.epoch_accumulator["rgb_loss_sum"] / count,
            "mask_loss": self.epoch_accumulator["mask_loss_sum"] / count,
            "psnr": self.epoch_accumulator["psnr_sum"] / count,
            "ssim": self.epoch_accumulator["ssim_sum"] / count,
            "foreground_gaussians": self.scene.foreground.count,
            "background_gaussians": self.scene.background.count,
        }
        self.epoch_summaries.append(summary)
        self.epoch += 1
        self.current_batches = None
        self.next_batch_index = 0
        self.epoch_accumulator = self._empty_epoch_accumulator()
        return summary

    def _rng_state(self) -> dict[str, Any]:
        """Capture every RNG needed to continue view order and optimizer behavior."""

        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: Mapping[str, Any]) -> None:
        """Restore RNGs after all checkpoint objects have been reconstructed."""

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def save_resume(self) -> Path:
        """Atomically save model, optimizer, density, progress, and RNG state."""

        path = self.work_dir / "resume.pt"
        payload = {
            "format": "modal_gaussians.static_training_resume",
            "version": 1,
            "dataset_identity": self.dataset.dataset_identity,
            "config_identity": self.config_identity,
            "resolved_config": self.config.resolved(),
            "scene_tensors": self.scene.tensor_dictionary(),
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
            },
            "schedulers": {
                name: scheduler.state_dict() for name, scheduler in self.schedulers.items()
            },
            "density_stats": {
                name: value.detach().cpu() for name, value in self.density_stats.items()
            },
            "density_events": self.density_events,
            "classification_summary": self.classification_summary,
            "epoch_summaries": self.epoch_summaries,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "current_batches": self.current_batches,
            "next_batch_index": self.next_batch_index,
            "epoch_accumulator": self.epoch_accumulator,
            "elapsed_seconds": self.elapsed_seconds(),
            "rng_state": self._rng_state(),
        }
        _atomic_torch_save(payload, path)
        return path

    def load_resume_state(self, payload: Mapping[str, Any]) -> None:
        """Restore optimizer, scheduler, density, progress, metrics, and RNG state."""

        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(payload["schedulers"][name])
        self.density_stats = {
            name: value.to(self.device) for name, value in payload["density_stats"].items()
        }
        if any(value.shape != (self.scene.count,) for value in self.density_stats.values()):
            raise ValueError("Resume density statistics do not match Gaussian count")
        self.density_events = list(payload["density_events"])
        self.epoch_summaries = list(payload["epoch_summaries"])
        self.epoch = int(payload["epoch"])
        self.global_step = int(payload["global_step"])
        batches = payload["current_batches"]
        self.current_batches = (
            [[int(index) for index in batch] for batch in batches]
            if batches is not None
            else None
        )
        self.next_batch_index = int(payload["next_batch_index"])
        self.epoch_accumulator = dict(payload["epoch_accumulator"])
        self.elapsed_seconds_before_resume = float(payload["elapsed_seconds"])
        self.started_at = time.time()
        self._restore_rng_state(payload["rng_state"])

    def train(self) -> dict[str, Any]:
        """Run or resume all configured epochs and leave an end-state checkpoint."""

        self.scene.train()
        group_counts = Counter((camera.width, camera.height) for camera in self.dataset.cameras)
        batches_per_epoch = sum(
            math.ceil(count / self.config.batch_size) for count in group_counts.values()
        )
        progress = Progress(
            "static train", self.config.epochs * batches_per_epoch,
            unit="steps", initial=self.global_step,
        )
        while self.epoch < self.config.epochs:
            if self.current_batches is None:
                self.current_batches = self._make_epoch_batches()
                self.next_batch_index = 0
            while self.next_batch_index < len(self.current_batches):
                batch = self.current_batches[self.next_batch_index]
                cameras, targets, foreground_targets, valid_masks = self._load_batch(
                    batch
                )
                loss, stats, info = self._compute_loss(
                    cameras, targets, foreground_targets, valid_masks
                )
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(f"Non-finite static loss at step {self.global_step}")
                loss.backward()
                self._accumulate_density_stats(info, cameras)
                for optimizer in self.optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in self.schedulers.values():
                    scheduler.step()
                self.global_step += 1
                self.next_batch_index += 1
                self._update_epoch_accumulator(stats)
                self._run_density_control()
                if self.global_step % self.config.checkpoint_every_steps == 0:
                    self.save_resume()
                progress.update(
                    self.global_step,
                    f"epoch={self.epoch + 1}/{self.config.epochs} "
                    f"batch={self.next_batch_index}/{len(self.current_batches)} "
                    f"loss={stats['loss']:.6f} mask={stats['mask_loss']:.6f} "
                    f"psnr={stats['psnr']:.3f} "
                    f"ssim={stats['ssim']:.4f} "
                    f"fg={self.scene.foreground.count} bg={self.scene.background.count}",
                    force=self.next_batch_index == len(self.current_batches),
                )
            summary = self._finish_epoch()
            self.save_resume()
            report_progress(
                "static epoch "
                f"{summary['epoch'] + 1}/{self.config.epochs}: "
                f"loss={summary['loss']:.6f}, mask={summary['mask_loss']:.6f}, "
                f"psnr={summary['psnr']:.3f}, "
                f"fg={summary['foreground_gaussians']}, "
                f"bg={summary['background_gaussians']}"
            )
        self.scene.eval()
        self.save_resume()
        final = self.epoch_summaries[-1]
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "final_loss": final["loss"],
            "final_rgb_loss": final["rgb_loss"],
            "final_mask_loss": final["mask_loss"],
            "final_psnr": final["psnr"],
            "final_ssim": final["ssim"],
        }


def _runtime_information(device: torch.device) -> dict[str, Any]:
    """Record the exact tensor/rasterizer environment used by this training run."""

    try:
        import gsplat

        gsplat_version = getattr(gsplat, "__version__", "unknown")
    except ImportError:
        gsplat_version = "unavailable"
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    return {
        "python_torch": torch.__version__,
        "torch_cuda": torch_cuda_version,
        "gsplat": gsplat_version,
        "device": str(device),
        "gpu": gpu_name,
    }


def export_static_bundle(
    trainer: StaticTrainer,
    output_dir: str | Path,
    training_result: Mapping[str, Any],
) -> Path:
    """Atomically publish the pure-tensor scene, manifest, and training summary."""

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        tensors = trainer.scene.tensor_dictionary()
        tensors_path = temporary_root / "tensors.pt"
        torch.save(tensors, tensors_path)
        tensors_hash = _sha256_file(tensors_path)
        foreground_identity = tensor_dictionary_identity(tensors, "foreground.")
        background_identity = tensor_dictionary_identity(tensors, "background.")
        camera_records = [
            camera.to_manifest_record() for camera in trainer.dataset.cameras
        ]
        reference_identities = {
            str(record["label"]): record["camera_identity"]
            for record in camera_records
            if record["role"] == "reference"
        }
        identity_payload = {
            "dataset_identity": trainer.dataset.dataset_identity,
            "foreground_identity": foreground_identity,
            "background_identity": background_identity,
            "normalization": trainer.dataset.normalization.to_dict(),
            "representation": "vanilla_3dgs_direct_rgb",
        }
        static_scene_identity = _sha256_json(identity_payload)
        manifest = {
            "format": "modal_gaussians.static_scene",
            "version": 1,
            "static_scene_identity": static_scene_identity,
            "foreground_identity": foreground_identity,
            "background_identity": background_identity,
            "reference_camera_identities": reference_identities,
            "tensors_sha256": tensors_hash,
            "representation": {
                "gaussian": "3dgs",
                "rasterizer": "gsplat",
                "rasterize_mode": "classic",
                "color": "direct_rgb_sigmoid",
                "sh_degree": None,
                "quaternion_convention": "wxyz_normalized",
                "scale_activation": "exp",
                "opacity_activation": "sigmoid",
                "camera_projection": "pinhole_K_only",
                "colmap_distortion_recorded_but_applied": False,
                "foreground_local_index_domain": [0, trainer.scene.foreground.count],
                "background_local_index_domain": [0, trainer.scene.background.count],
                "combined_foreground_index_domain": [0, trainer.scene.foreground.count],
                "combined_background_index_domain": [
                    trainer.scene.foreground.count,
                    trainer.scene.count,
                ],
            },
            "counts": {
                "foreground": trainer.scene.foreground.count,
                "background": trainer.scene.background.count,
            },
            "scene_normalization": trainer.dataset.normalization.to_dict(),
            "dataset": {
                "input_root": str(trainer.dataset.root),
                "dataset_identity": trainer.dataset.dataset_identity,
                "file_identities": dict(trainer.dataset.file_identities),
                "training_roles": ["sweep", "reference"],
            },
            "cameras": camera_records,
            "training_config": trainer.config.resolved(),
        }
        (temporary_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        summary = {
            "format": "modal_gaussians.static_training_summary",
            "version": 1,
            "static_scene_identity": static_scene_identity,
            "dataset_identity": trainer.dataset.dataset_identity,
            "result": dict(training_result),
            "classification": trainer.classification_summary,
            "initial_counts": {
                "foreground": trainer.classification_summary[
                    "initial_foreground_gaussians"
                ],
                "background": trainer.classification_summary[
                    "initial_background_gaussians"
                ],
            },
            "final_counts": {
                "foreground": trainer.scene.foreground.count,
                "background": trainer.scene.background.count,
            },
            "epoch_summaries": trainer.epoch_summaries,
            "density_events": trainer.density_events,
            "runtime": _runtime_information(trainer.device),
            "elapsed_seconds": trainer.elapsed_seconds(),
            "end_reason": "configured_epochs_completed",
        }
        (temporary_root / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_root, output_dir)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)
    return output_dir


def _load_resume_payload(path: Path) -> Mapping[str, Any]:
    """Load and minimally validate one internal trusted training resume file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "modal_gaussians.static_training_resume"
        or payload.get("version") != 1
    ):
        raise ValueError(f"Unsupported static resume checkpoint: {path}")
    return payload


def run_static_training(
    *,
    input_dir: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
    config: StaticTrainConfig | None = None,
    resume: bool = False,
    device: str | torch.device | None = None,
) -> Path:
    """Validate inputs, train/resume the static scene, export it, and render QA."""

    config = StaticTrainConfig() if config is None else config
    config.validate()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    work_dir = Path(work_dir).expanduser().resolve()
    report_progress("static: loading and validating joint COLMAP inputs")
    dataset = load_static_dataset(input_dir)
    if device is None:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)
    if selected_device.type != "cuda":
        raise RuntimeError("Static 3DGS training requires a CUDA device")
    _set_random_seed(config.seed)
    resume_path = work_dir / "resume.pt"
    if resume:
        report_progress(f"static: restoring {resume_path}")
        payload = _load_resume_payload(resume_path)
        config_identity = _sha256_json(config.resolved())
        if payload["dataset_identity"] != dataset.dataset_identity:
            raise ValueError("Resume dataset identity does not match current joint COLMAP input")
        if payload["config_identity"] != config_identity:
            raise ValueError("Resume config identity does not match requested static config")
        scene = _scene_from_tensor_dictionary(payload["scene_tensors"])
        trainer = StaticTrainer(
            scene,
            dataset,
            config,
            selected_device,
            work_dir,
            payload["classification_summary"],
        )
        trainer.load_resume_state(payload)
    else:
        if work_dir.exists() and any(work_dir.iterdir()):
            raise FileExistsError(
                f"Static work directory is not empty; use --resume or a new path: {work_dir}"
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        report_progress("static: classifying points and initializing foreground/background")
        scene, classification = initialize_static_scene(
            dataset,
            num_foreground=config.num_foreground,
            num_background=config.num_background,
            seed=config.seed,
        )
        trainer = StaticTrainer(
            scene,
            dataset,
            config,
            selected_device,
            work_dir,
            classification,
        )
        trainer.save_resume()
    result = trainer.train()
    report_progress("static: exporting trained bundle")
    bundle = export_static_bundle(trainer, output_dir, result)
    report_progress("static: rendering offline QA")
    qa_dir = render_static_bundle(
        scene_dir=bundle,
        output_dir=work_dir / "qa",
        role="all",
        device=selected_device,
    )
    qa_metrics = json.loads((qa_dir / "metrics.json").read_text(encoding="utf-8"))
    summary_path = bundle / "training_summary.json"
    training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    training_summary["qa"] = qa_metrics
    _atomic_json_write(training_summary, summary_path)
    return bundle


def _write_rgb(path: Path, image: np.ndarray) -> None:
    """Write one float RGB render as a standard uint8 PNG."""

    rgb = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image: {path}")


def _depth_visualization(depth: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Map valid expected depth to a robust inferno visualization."""

    valid = (alpha > 0.01) & np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.quantile(depth[valid], [0.02, 0.98])
        if high <= low:
            high = low + 1e-6
        values = np.clip((depth - low) / (high - low), 0.0, 1.0)
        normalized[valid] = np.rint(values[valid] * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    colored[~valid] = 0
    return colored


def _qa_metrics(prediction: Tensor, target: Tensor) -> tuple[float, float]:
    """Compute full-frame PSNR and SSIM for one registered training camera."""

    mse = F.mse_loss(prediction, target).clamp_min(1e-12)
    psnr = float((-10.0 * torch.log10(mse)).item())
    try:
        from pytorch_msssim import ssim
    except ImportError as error:
        raise RuntimeError("Static QA requires pytorch-msssim") from error
    value = ssim(
        prediction.permute(2, 0, 1)[None],
        target.permute(2, 0, 1)[None],
        data_range=1.0,
        size_average=True,
    )
    return psnr, float(value.item())


@torch.inference_mode()
def render_static_bundle(
    *,
    scene_dir: str | Path,
    output_dir: str | Path,
    role: Literal["all", "sweep", "reference"] = "all",
    device: str | torch.device | None = None,
) -> Path:
    """Render all requested stored cameras into an atomic offline QA directory."""

    if role not in ("all", "sweep", "reference"):
        raise ValueError(f"Unsupported QA camera role: {role}")
    scene_dir = Path(scene_dir).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    selected_device = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else device or "cpu"
    )
    if selected_device.type != "cuda":
        raise RuntimeError("Static 3DGS rendering requires a CUDA device")
    scene = load_static_scene(scene_dir, selected_device)
    if scene.manifest is None:
        raise ValueError("Loaded static scene has no manifest")
    cameras = cameras_from_scene_manifest(scene.manifest)
    cameras = tuple(
        camera for camera in cameras if role == "all" or camera.role == role
    )
    if not cameras:
        raise ValueError(f"Static scene contains no cameras for role={role}")
    dataset_root = Path(scene.manifest["dataset"]["input_root"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    metrics: list[dict[str, Any]] = []
    progress = Progress("static QA", len(cameras), unit="views")
    try:
        for camera in cameras:
            image_path = dataset_root / camera.image_relative_path
            if not image_path.is_file() or _sha256_file(image_path) != camera.image_sha256:
                raise ValueError(f"QA source image is missing or changed: {image_path}")
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(image_path)
            target_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            target = torch.from_numpy(target_np).to(selected_device)
            full = scene.render(camera, composition="all")
            foreground = scene.render(camera, composition="foreground")
            background = scene.render(camera, composition="background", outputs=("rgb",))
            psnr, ssim_value = _qa_metrics(full["rgb"], target)
            safe_name = camera.name.replace("/", "__").replace("\\", "__")
            _write_rgb(temporary_root / f"{safe_name}__full_rgb.png", full["rgb"].cpu().numpy())
            _write_rgb(
                temporary_root / f"{safe_name}__foreground_rgb.png",
                foreground["rgb"].cpu().numpy(),
            )
            _write_rgb(
                temporary_root / f"{safe_name}__background_rgb.png",
                background["rgb"].cpu().numpy(),
            )
            alpha = foreground["alpha"].cpu().numpy()
            depth = foreground["expected_depth"].cpu().numpy()
            alpha_u8 = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
            if not cv2.imwrite(
                str(temporary_root / f"{safe_name}__foreground_alpha.png"), alpha_u8
            ):
                raise OSError("Could not write foreground alpha")
            np.save(temporary_root / f"{safe_name}__foreground_depth.npy", depth)
            if not cv2.imwrite(
                str(temporary_root / f"{safe_name}__foreground_depth.png"),
                _depth_visualization(depth, alpha),
            ):
                raise OSError("Could not write foreground depth visualization")
            rendered_u8 = np.clip(
                np.rint(full["rgb"].cpu().numpy() * 255.0), 0, 255
            ).astype(np.uint8)
            comparison = np.concatenate(
                [
                    np.clip(np.rint(target_np * 255.0), 0, 255).astype(np.uint8),
                    rendered_u8,
                ],
                axis=1,
            )
            if not cv2.imwrite(
                str(temporary_root / f"{safe_name}__gt_render.png"),
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
            ):
                raise OSError("Could not write GT/render comparison")
            metrics.append(
                {
                    "camera": camera.name,
                    "role": camera.role,
                    "label": camera.label,
                    "psnr": psnr,
                    "ssim": ssim_value,
                }
            )
            progress.update(
                len(metrics),
                f"camera={camera.name} psnr={psnr:.3f} ssim={ssim_value:.4f}",
                force=True,
            )
        role_metrics: dict[str, dict[str, float | int]] = {}
        for camera_role in ("sweep", "reference"):
            role_entries = [
                entry for entry in metrics if entry["role"] == camera_role
            ]
            if role_entries:
                role_metrics[camera_role] = {
                    "camera_count": len(role_entries),
                    "mean_psnr": float(
                        np.mean([entry["psnr"] for entry in role_entries])
                    ),
                    "mean_ssim": float(
                        np.mean([entry["ssim"] for entry in role_entries])
                    ),
                }
        payload = {
            "format": "modal_gaussians.static_qa",
            "version": 1,
            "static_scene_identity": scene.manifest["static_scene_identity"],
            "role": role,
            "camera_count": len(metrics),
            "mean_psnr": float(np.mean([entry["psnr"] for entry in metrics])),
            "mean_ssim": float(np.mean([entry["ssim"] for entry in metrics])),
            "roles": role_metrics,
            "cameras": metrics,
        }
        (temporary_root / "metrics.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_root, output_dir)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)
    return output_dir


__all__ = [
    "StaticTrainConfig",
    "export_static_bundle",
    "render_static_bundle",
    "run_static_training",
]
