"""Shared foreground refinement with independent recording/frame coefficients."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from modal_gaussians.common.cache import atomic_json, exclusive_work, identity, module_revision
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.geometry import density
from modal_gaussians.geometry.scene import GAUSSIAN_FIELDS, cameras_from_scene_manifest
from modal_gaussians.motion import reference_field
from modal_gaussians.motion.training import _atomic_torch
from .fitting import RGBFitConfig, _rgb_loss
from .rendering import apply_angular_rotation, deform_baked, resize_rgb, scaled_camera
from .rgb import load_rgb_frame
from .refinement_artifacts import load_prepared, publish_refinement
from .sequences import frame_stride, sequence_weights


@dataclass(frozen=True)
class RefinementConfig:
    image_scale: float = 1.0
    rounds: int = 2
    coefficient_epochs_per_round: int = 1
    fixed_geometry_fps: float = 5.0
    sweep_geometry_fps: float = 10.0
    coefficient_lr: float = 0.001
    coefficient_final_lr: float = 0.0001
    anchor_weight: float = 1e-4
    position_lr: float = 4e-5
    quaternion_lr: float = 2.5e-4
    scale_lr: float = 1.25e-3
    color_lr: float = 2.5e-3
    opacity_lr: float = 2.5e-3
    gaussian_final_factor: float = 0.1
    shape_radius_fraction: float = 0.25
    density_enabled: bool = True
    density_warmup: int = 34
    density_interval: int = 17
    maximum_count_factor: float = 2.0
    densify_gradient: float = 2e-4
    split_world_scale: float = 0.01
    split_screen_scale: float = 0.05
    cull_opacity: float = 0.005
    cull_world_scale: float = 0.5
    cull_screen_scale: float = 0.15
    minimum_visibility: int = 5
    newborn_protection: int = 17
    checkpoint_interval: int = 200
    query_block_size: int = 4096
    seed: int = 1729

    def validate(self):
        RGBFitConfig(scales=(self.image_scale,), epochs_per_scale=self.coefficient_epochs_per_round,
            learning_rate=self.coefficient_lr, final_learning_rate=self.coefficient_final_lr,
            anchor_weight=self.anchor_weight, seed=self.seed).validate()
        if self.image_scale != 1.0:
            raise ValueError('Scene refinement uses full resolution only')
        for name in ("rounds", "coefficient_epochs_per_round", "density_warmup", "density_interval", "minimum_visibility", "newborn_protection",
                     "checkpoint_interval", "query_block_size"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ("density_warmup", "newborn_protection") else 1):
                raise ValueError(f"Invalid refinement {name}")
        for name in ("fixed_geometry_fps", "sweep_geometry_fps", "position_lr", "quaternion_lr", "scale_lr", "color_lr", "opacity_lr",
                     "gaussian_final_factor", "shape_radius_fraction", "maximum_count_factor", "densify_gradient",
                     "split_world_scale", "split_screen_scale", "cull_opacity", "cull_world_scale", "cull_screen_scale"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid refinement {name}")
        if self.maximum_count_factor < 1 or self.gaussian_final_factor > 1 or type(self.density_enabled) is not bool:
            raise ValueError("Invalid refinement density cap or learning rate decay")


class RefinementTrainer:
    """Alternate sparse joint geometry updates and exhaustive frozen-geometry q passes."""
    def __init__(self, scene, field, initial, views, config=RefinementConfig(), device="cuda"):
        config.validate()
        self.scene, self.field, self.views, self.config = scene.to(device), field, views, config
        scene.background.requires_grad_(False)
        scene.foreground.requires_grad_(True)
        self.device = torch.device(device)
        if any(v.get('kind') == 'sweep' and v['fps_hz'] != 30 for v in views):
            raise ValueError('Refinement requires a 30 FPS sweep preparation; prepare new inputs')
        self.weights = sequence_weights(views)
        self.scales = torch.as_tensor(initial["scales"], dtype=torch.float32, device=device)
        q = torch.as_tensor(initial["coordinates"], dtype=torch.complex64, device=device)
        if (self.scales.shape != (len(views), field.mode_count) or q.shape !=
                (sum(v["frame_count"] for v in views), field.mode_count)
                or not torch.isfinite(q).all() or not torch.isfinite(self.scales).all() or not (self.scales > 0).all()):
            raise ValueError("Invalid initial refinement coefficients/scales")
        self.q, self.q0 = [], []
        for v, scale in zip(views, self.scales):
            p = q[v["frame_offset"]:v["frame_offset"]+v["frame_count"]] * scale
            self.q0.extend(torch.view_as_real(p).unbind(0))
            self.q.extend(torch.nn.Parameter(row.clone()) for row in torch.view_as_real(p).unbind(0))
        self.qopt = [torch.optim.Adam([row], lr=config.coefficient_lr) for row in self.q]
        self.rates = dict(means=config.position_lr, quaternions=config.quaternion_lr,
            log_scales=config.scale_lr, color_logits=config.color_lr, opacity_logits=config.opacity_lr)
        self.optimizers = {k: torch.optim.Adam([scene.foreground.params[k]], lr=v) for k,v in self.rates.items()}
        n = scene.foreground.count
        protected = np.zeros(n, bool); protected[field.a["controls"]] = True
        self.mapping = dict(uid=np.arange(n, dtype=np.int64), root_id=np.arange(n, dtype=np.int64),
                            protected=protected, birth_step=np.zeros(n, np.int64))
        self.next_uid, self.initial_count, self.step = n, n, 0
        self.stats = {k: torch.zeros(n, device=device) for k in ("gradient", "visibility", "radius")}
        self.rng = np.random.default_rng(config.seed)
        self.sparse_frames = [[self.select_geometry_frames(v) for v in views] for _ in range(config.rounds)]
        self.geometry_steps_per_round = max(map(len, self.sparse_frames[0]))
        self.coefficient_steps_per_round = len(self.q) * config.coefficient_epochs_per_round
        self.round_steps = self.geometry_steps_per_round + self.coefficient_steps_per_round
        self.total_steps = config.rounds * self.round_steps
        self.geometry_step = self.coefficient_step = 0
        self.baked_fields = None
        self.sampler_phase = None
        self.permutations, self.cursors, self.coefficient_order = [], [], np.empty(0, np.int64)
        self.enter_phase()

    def select_geometry_frames(self, view):
        fps = self.config.sweep_geometry_fps if view.get('kind') == 'sweep' else self.config.fixed_geometry_fps
        stride = frame_stride(view['fps_hz'], fps)
        count = view['frame_count']
        selected = np.asarray([self.rng.integers(lo, min(lo+stride, count))
                               for lo in range(0, count, stride)], np.int64)
        selected[0] = 0
        # A one-bin sequence has one sample; retain its first frame.
        if len(selected) > 1:
            selected[-1] = count-1
        return selected

    @property
    def phase(self):
        if self.step == self.total_steps:
            return 'complete'
        return 'geometry' if self.step % self.round_steps < self.geometry_steps_per_round else 'coefficient'

    def enter_phase(self):
        key = (self.step // self.round_steps, self.phase)
        if key == self.sampler_phase:
            return
        self.sampler_phase = key
        self.scene.foreground.requires_grad_(self.phase == 'geometry')
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        if self.phase == 'geometry':
            self.baked_fields = None
            self.permutations = [self.rng.permutation(p) for p in self.sparse_frames[key[0]]]
            self.cursors = [0] * len(self.views)
            self.coefficient_order = np.empty(0, np.int64)
        elif self.phase == 'coefficient':
            # Every row appears once per pass; row-local Adam states persist across phases.
            self.coefficient_order = np.concatenate([self.rng.permutation(len(self.q))
                for _ in range(self.config.coefficient_epochs_per_round)])

    def frozen_fields(self):
        if self.baked_fields is None:
            self.baked_fields = self.field.bake(self.scene.foreground.params['means'], self.mapping['root_id'])
        return self.baked_fields

    def sample(self):
        sampled = []
        for v, view in enumerate(self.views):
            if self.cursors[v] == len(self.permutations[v]):
                self.permutations[v] = self.rng.permutation(self.sparse_frames[self.step // self.round_steps][v])
                self.cursors[v] = 0
            frame = int(self.permutations[v][self.cursors[v]])
            self.cursors[v] += 1
            sampled.append((v, frame, view["frame_offset"] + frame))
        return sampled

    @torch.no_grad()
    def accumulate(self, info, camera, weight):
        n = self.scene.foreground.count
        if info["means2d"].grad is None:
            raise RuntimeError("Dynamic renderer did not retain density gradients")
        gradient = info["means2d"].grad[0, :n].detach().clone()
        gradient *= gradient.new_tensor([camera.width / 2, camera.height / 2]) / weight
        radius = info["radii"][0, :n]
        visible = (radius > 0).all(-1) if radius.ndim == 2 else radius > 0
        radius = radius.amax(-1) if radius.ndim == 2 else radius
        self.stats["gradient"][visible] += gradient[visible].norm(dim=-1)
        self.stats["visibility"][visible] += 1
        self.stats["radius"][visible] = torch.maximum(self.stats["radius"][visible],
                                                   radius[visible] / max(camera.width, camera.height))

    @torch.no_grad()
    def density_control(self):
        c, part = self.config, self.scene.foreground
        score = self.stats["gradient"] / self.stats["visibility"].clamp_min(1)
        eligible = torch.as_tensor(~self.mapping["protected"] &
            (self.geometry_step-self.mapping["birth_step"] >= c.newborn_protection), device=self.device)
        eligible &= self.stats["visibility"] >= c.minimum_visibility
        size = part.params["log_scales"].exp().amax(-1)
        cull = eligible & ((part.params["opacity_logits"].sigmoid() < c.cull_opacity)
                          | (size > c.cull_world_scale) | (self.stats["radius"] > c.cull_screen_scale))
        if int((~cull).sum()) < 2:
            raise RuntimeError("Refinement culling would leave fewer than two foreground points")
        candidates = eligible & ~cull & (score > c.densify_gradient)
        available = max(0, int(self.initial_count*c.maximum_count_factor) - part.count + int(cull.sum()))
        ids = candidates.nonzero().flatten()
        order = torch.argsort(score[ids], descending=True, stable=True)
        candidates[:] = False; candidates[ids[order[:available]]] = True
        split = candidates & ((size > c.split_world_scale) | (self.stats["radius"] > c.split_screen_scale))
        duplicate = candidates & ~split
        children = density.split_positions(part, split)
        cancelled = 0
        if len(children):
            parents = split.nonzero().flatten().cpu().numpy()
            roots = np.repeat(self.mapping["root_id"][parents], 2)
            valid = (self.field.valid(children, roots)
                     & self.field.shape_valid(children, roots, c.shape_radius_fraction)).reshape(-1,2).all(-1)
            cancelled = int((~valid).sum())
            split[torch.as_tensor(parents, device=self.device)[~valid]] = False
            children = children.reshape(-1,2,3)[valid].reshape(-1,3)
        added, removed = int(duplicate.sum()) + 2*int(split.sum()), int(cull.sum()) + int(split.sum())
        if bool((split | duplicate).any()):
            rows, newborn = density.densify_parameters(part, self.optimizers, split, duplicate, children)
            row_np, born_np = rows.cpu().numpy(), newborn.cpu().numpy()
            self.mapping = {k: v[row_np].copy() for k,v in self.mapping.items()}
            self.mapping["uid"][born_np] = np.arange(self.next_uid, self.next_uid+int(newborn.sum()))
            self.next_uid += int(newborn.sum())
            self.mapping["birth_step"][born_np] = self.geometry_step
            cull = cull[rows]; cull[newborn] = False
        if bool(cull.any()):
            rows = (~cull).nonzero().flatten()
            density.remap_parameters(part, self.optimizers, rows)
            self.mapping = {k:v[rows.cpu().numpy()].copy() for k,v in self.mapping.items()}
        self.stats = {k: torch.zeros(part.count, device=self.device) for k in self.stats}
        self.baked_fields = None
        return dict(added=added, removed=removed, cancelled_splits=cancelled)

    def update(self, render, target):
        if self.step >= self.total_steps:
            raise RuntimeError('Refinement has completed')
        self.enter_phase()
        round_index, phase = self.sampler_phase
        phase_step = self.step % self.round_steps
        if phase == 'coefficient':
            phase_step -= self.geometry_steps_per_round
            row = self.coefficient_update(render, target, phase_step)
        else:
            row = self.geometry_update(render, target)
        return dict(row, step=self.step, round=round_index, phase=phase, phase_step=phase_step+1,
                    scale=self.config.image_scale, geometry_step=self.geometry_step,
                    coefficient_step=self.coefficient_step,
                    phase_boundary=(self.step // self.round_steps, self.phase) != self.sampler_phase)

    def coefficient_rate(self):
        c = self.config
        return c.coefficient_lr + self.step / max(1, self.total_steps-1) * (c.coefficient_final_lr-c.coefficient_lr)

    def coefficient_update(self, render, target, phase_step):
        c, part = self.config, self.scene.foreground
        started = time.perf_counter()
        phi, omega = self.frozen_fields()
        bake_seconds = time.perf_counter()-started
        index = int(self.coefficient_order[phase_step])
        v = next(v for v, view in enumerate(self.views) if view['frame_offset'] <= index < view['frame_offset']+view['frame_count'])
        frame = index-self.views[v]['frame_offset']
        optimizer = self.qopt[index]
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]['lr'] = self.coefficient_rate()
        started = time.perf_counter()
        q = torch.view_as_complex(self.q[index]) / self.scales[v]
        means, rotations = deform_baked(part.params['means'], part.params['quaternions'], q, phi, omega)
        result, _ = render(v, frame, c.image_scale, means, rotations)
        rgb_loss = _rgb_loss(result['rgb'], target(v, frame, c.image_scale))
        anchor = (self.q[index]-self.q0[index]).square().sum(-1).mean()
        loss = self.weights[v] * (rgb_loss+c.anchor_weight*anchor)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite coefficient phase loss')
        render_seconds = time.perf_counter()-started
        started = time.perf_counter()
        loss.backward(inputs=[self.q[index]])
        if self.q[index].grad is None or not bool(torch.isfinite(self.q[index].grad).all()):
            raise FloatingPointError('Missing or non-finite coefficient phase gradient')
        backward_seconds = time.perf_counter()-started
        optimizer.step()
        if not bool(torch.isfinite(self.q[index]).all()):
            raise FloatingPointError('Non-finite coefficient phase parameter')
        self.step += 1
        self.coefficient_step += 1
        return dict(views=[dict(label=self.views[v]['label'], frame=frame,
                    rgb_loss=float(rgb_loss.detach()), coefficient_offset=float(anchor.detach()))],
                    gaussian_count=part.count, support_backtracks=0, shape_backtracks=0, position_backtracks=0,
                    control_position_error=0.,
                    added=0, removed=0, cancelled_splits=0, density_seconds=0.,
                    bake_seconds=bake_seconds, query_seconds=0., query_backward_seconds=0.,
                    render_seconds=render_seconds, backward_seconds=backward_seconds)

    def geometry_update(self, render, target):
        c = self.config
        scale = c.image_scale
        fraction = self.geometry_step / max(1, c.rounds*self.geometry_steps_per_round-1)
        for name, optimizer in self.optimizers.items():
            optimizer.param_groups[0]["lr"] = self.rates[name] * c.gaussian_final_factor**fraction
            optimizer.zero_grad(set_to_none=True)
        part = self.scene.foreground
        previous = part.params["means"].detach().clone()
        sampled, losses = self.sample(), []
        timing = dict(render_seconds=0., backward_seconds=0., density_seconds=0.)
        for v, frame, index in sampled:
            self.qopt[index].zero_grad(set_to_none=True)
            self.qopt[index].param_groups[0]["lr"] = self.coefficient_rate()
        started = time.perf_counter()
        q = torch.stack([torch.view_as_complex(self.q[index]) / self.scales[v] for v, _, index in sampled])
        means, angular = self.field.deform(part.params["means"], self.mapping["root_id"], q)
        timing["query_seconds"] = time.perf_counter()-started
        # Render graphs end at these leaves; traverse the shared query graph only once.
        render_means = means.detach().requires_grad_(True)
        render_angular = angular.detach().requires_grad_(True)
        for s, (v, frame, index) in enumerate(sampled):
            started = time.perf_counter()
            rotations = apply_angular_rotation(part.params["quaternions"], render_angular[s])
            result, camera = render(v, frame, scale, render_means[s], rotations)
            rgb_loss = _rgb_loss(result["rgb"], target(v, frame, scale))
            anchor = (self.q[index]-self.q0[index]).square().sum(-1).mean()
            loss = (rgb_loss + c.anchor_weight * anchor) * self.weights[v]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite refinement loss")
            timing["render_seconds"] += time.perf_counter()-started
            started = time.perf_counter()
            loss.backward()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timing["backward_seconds"] += time.perf_counter()-started
            if c.density_enabled and self.step // self.round_steps == 0:
                self.accumulate(result["info"], camera, self.weights[v])
            losses.append(dict(label=self.views[v]["label"], frame=frame,
                               rgb_loss=float(rgb_loss.detach()), coefficient_offset=float(anchor.detach())))
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in (render_means, render_angular)):
            raise FloatingPointError("Missing or non-finite deformation gradient")
        started = time.perf_counter()
        torch.autograd.backward((means, angular), (render_means.grad, render_angular.grad))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timing["query_backward_seconds"] = time.perf_counter()-started
        del means, angular, render_means, render_angular, q, result, rotations, loss, rgb_loss, anchor
        parameters = list(part.parameters()) + [self.q[index] for _,_,index in sampled]
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError("Missing or non-finite refinement gradient")
        protected = torch.as_tensor(self.mapping["protected"], device=self.device)
        part.params["means"].grad[protected] = 0
        with torch.no_grad():
            for value in self.optimizers["means"].state.get(part.params["means"], {}).values():
                if isinstance(value, torch.Tensor) and value.shape == part.params["means"].shape:
                    value[protected] = 0
        for optimizer in self.optimizers.values():
            optimizer.step()
        for _,_,index in sampled:
            self.qopt[index].step()
        with torch.no_grad():
            part.params["means"][protected] = previous[protected]
        if any(not bool(torch.isfinite(p).all()) for p in parameters):
            raise FloatingPointError("Non-finite refinement parameter")
        backtracks = self.field.constrain(part.params["means"], previous, self.mapping["root_id"],
            self.optimizers["means"], shape_radius_fraction=c.shape_radius_fraction)
        self.step += 1
        self.geometry_step += 1
        self.baked_fields = None
        decisions = dict(added=0, removed=0, cancelled_splits=0)
        if (c.density_enabled and self.geometry_step <= self.geometry_steps_per_round
                and self.geometry_step > c.density_warmup
                and self.geometry_step % c.density_interval == 0):
            started = time.perf_counter()
            decisions = self.density_control()
            timing["density_seconds"] = time.perf_counter()-started
        fixed = self.mapping["protected"]
        expected = torch.as_tensor(self.field.a["points"][self.mapping["root_id"][fixed]], device=self.device)
        drift = float((part.params["means"][torch.as_tensor(fixed, device=self.device)] - expected).abs().max())
        if drift != 0:
            raise RuntimeError("Protected canonical position changed")
        return dict(views=losses, gaussian_count=part.count, **backtracks,
                    bake_seconds=0.,
                    control_position_error=drift, **decisions, **timing)

    def coordinates(self):
        rows = torch.stack([torch.view_as_complex(row.detach()) for row in self.q])
        for v, scale in zip(self.views, self.scales):
            lo = v["frame_offset"]; rows[lo:lo+v["frame_count"]] /= scale
        return rows.cpu().numpy().astype(np.complex64)

    def state_dict(self):
        return dict(step=self.step, geometry_step=self.geometry_step, coefficient_step=self.coefficient_step,
            sampler_phase=self.sampler_phase,
            sparse_frames=[[torch.from_numpy(p) for p in r] for r in self.sparse_frames],
            coefficient_order=torch.from_numpy(self.coefficient_order),
            scene={name:p.detach().cpu() for name,p in self.scene.state_dict().items()},
            mapping={k:torch.from_numpy(v) for k,v in self.mapping.items()}, next_uid=self.next_uid,
            q=[p.detach().cpu() for p in self.q], qopt=[o.state_dict() for o in self.qopt],
            optimizers={k:o.state_dict() for k,o in self.optimizers.items()}, stats=self.stats,
            permutations=[torch.from_numpy(p) for p in self.permutations], cursors=self.cursors,
            rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            numpy_rng={"name":np.random.get_state()[0], "keys":torch.from_numpy(np.random.get_state()[1].astype(np.int64)),
                       "position":np.random.get_state()[2], "gauss":np.random.get_state()[3], "cached":np.random.get_state()[4]})

    def load_state_dict(self, state):
        if not {'geometry_step', 'coefficient_step', 'sampler_phase', 'sparse_frames', 'coefficient_order'} <= state.keys():
            raise ValueError('Checkpoint predates alternating refinement; use a new work directory')
        n = len(state["mapping"]["uid"])
        mapping = state["mapping"]
        if (set(mapping) != set(self.mapping) or any(v.shape != (n,) for v in mapping.values())
                or n < 2 or len(torch.unique(mapping["uid"])) != n
                or int(mapping["root_id"].min()) < 0 or int(mapping["root_id"].max()) >= len(self.field.a["points"])
                or not 0 <= state["step"] <= self.total_steps or state["next_uid"] <= int(mapping["uid"].max())
                or len(state["q"]) != len(self.q) or len(state["qopt"]) != len(self.qopt)
                or len(state["permutations"]) != len(self.views) or len(state["cursors"]) != len(self.views)
                or any(v.shape != (n,) for v in state["stats"].values())):
            raise ValueError("Invalid refinement checkpoint domains")
        geometry_step = (state['step'] // self.round_steps * self.geometry_steps_per_round
                         + min(state['step'] % self.round_steps, self.geometry_steps_per_round))
        if geometry_step != state['geometry_step'] or state['coefficient_step'] != state['step']-geometry_step:
            raise ValueError('Checkpoint phase counters differ')
        def phase_key(step):
            return (step // self.round_steps, 'geometry' if step % self.round_steps < self.geometry_steps_per_round else 'coefficient')
        key = tuple(state['sampler_phase'])
        if key not in (phase_key(state['step']), phase_key(max(0, state['step']-1))) or key[0] >= self.config.rounds:
            raise ValueError('Checkpoint sampler phase differs')
        if (len(state['sparse_frames']) != len(self.sparse_frames)
                or any(len(saved) != len(expected) or any(not np.array_equal(a.cpu().numpy(), b)
                    for a,b in zip(saved, expected)) for saved,expected in zip(state['sparse_frames'], self.sparse_frames))):
            raise ValueError('Checkpoint sparse frame selection differs')
        for i, (permutation, cursor) in enumerate(zip(state["permutations"], state["cursors"])):
            expected = torch.from_numpy(self.sparse_frames[key[0]][i])
            if not torch.equal(permutation.sort().values, expected) or not 0 <= cursor <= len(expected):
                raise ValueError("Invalid refinement checkpoint sampler")
        order = state['coefficient_order']
        if key[1] == 'geometry':
            valid_order = order.numel() == 0
        else:
            valid_order = (order.shape == (self.coefficient_steps_per_round,) and
                all(torch.equal(p.sort().values, torch.arange(len(self.q))) for p in order.split(len(self.q))))
        if not valid_order or any(p.shape != q.shape or not torch.isfinite(p).all() for p,q in zip(state['q'],self.q)):
            raise ValueError('Invalid coefficient phase checkpoint')
        protected_roots = mapping["root_id"][mapping["protected"]].cpu().numpy()
        if not np.array_equal(np.sort(protected_roots), np.sort(self.field.a["controls"])):
            raise ValueError("Checkpoint lost a protected control")
        expected = torch.from_numpy(self.field.a["points"][protected_roots])
        if not torch.equal(state["scene"]["foreground.params.means"][mapping["protected"]].cpu(), expected):
            raise ValueError("Checkpoint moved a protected control")
        if not bool(self.field.shape_valid(state['scene']['foreground.params.means'].to(self.device),
                mapping['root_id'].cpu().numpy(), self.config.shape_radius_fraction).all()):
            raise ValueError('Checkpoint positions violate the fixed graph shape constraint')
        for part in ("foreground", "background"):
            for field in GAUSSIAN_FIELDS:
                getattr(self.scene, part).replace_parameter(field, state["scene"][f"{part}.params.{field}"].to(self.device))
        self.scene.background.requires_grad_(False)
        self.scene.foreground.requires_grad_(key[1] == 'geometry')
        for name, optimizer in self.optimizers.items():
            optimizer.param_groups[0]["params"] = [self.scene.foreground.params[name]]
            optimizer.load_state_dict(state["optimizers"][name])
        with torch.no_grad():
            for p, value, optimizer, saved in zip(self.q, state["q"], self.qopt, state["qopt"]):
                p.copy_(value); optimizer.load_state_dict(saved)
        self.mapping = {k:v.cpu().numpy().copy() for k,v in state["mapping"].items()}
        self.step, self.next_uid = state["step"], state["next_uid"]
        self.geometry_step, self.coefficient_step = state['geometry_step'], state['coefficient_step']
        self.sampler_phase = key
        self.coefficient_order = order.cpu().numpy().copy()
        self.baked_fields = None
        self.stats = {k:v.to(self.device) for k,v in state["stats"].items()}
        self.permutations = [v.cpu().numpy().copy() for v in state["permutations"]]
        self.cursors = state["cursors"]
        self.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        r = state["numpy_rng"]
        np.random.set_state((r["name"], r["keys"].cpu().numpy().astype(np.uint32), r["position"], r["gauss"], r["cached"]))


def refine_scene(*, prepared_dir, work_dir, output_dir, config=RefinementConfig(), resume=False, device="cuda"):
    config.validate()
    root, manifest, scene, arrays, initial = load_prepared(prepared_dir)
    work, destination = resolve_path(work_dir), resolve_path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    immutable = [resolve_path(v['directory']) for v in manifest['images']]
    immutable.extend(resolve_path(v['path']) for v in manifest['rgb_sources'])
    immutable.extend(resolve_path(v['path']) for v in manifest['mode_sources'] if v.get('path'))
    if manifest.get('mode_bank'): immutable.append(resolve_path(manifest['mode_bank']))
    if any(p.is_relative_to(source) for p in (work, destination) for source in immutable):
        raise ValueError('Refinement work/output must be outside immutable source artifacts and images')
    for a, b in ((work, root), (destination, root), (destination, work), (work, destination),
                 (work, resolve_path(manifest["static_scene"])), (destination, resolve_path(manifest["static_scene"]))):
        if a.is_relative_to(b):
            raise ValueError("Refinement work/output must be separate from immutable inputs and each other")
    from . import fitting, rendering, refinement_artifacts, rgb, sequences
    from modal_gaussians.geometry import scene as scene_module
    from modal_gaussians.common import camera_rendering
    contract = dict(preparation_identity=manifest["preparation_identity"], settings=asdict(config),
        supervision_weights=sequence_weights(manifest["views"]),
        implementation=module_revision(sys.modules[__name__], reference_field, density, fitting,
                                       rendering, refinement_artifacts, rgb, sequences, scene_module, camera_rendering), device=str(device))
    run_identity = identity(contract)
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "checkpoint.pt"
    with exclusive_work(work / ".lock"):
        if not resume and ((work / "run.json").exists() or checkpoint.exists()):
            raise FileExistsError("Refinement work exists; use --resume or a new work directory")
        if resume and not checkpoint.is_file():
            raise FileNotFoundError("No complete refinement checkpoint to resume")
        torch.manual_seed(config.seed); np.random.seed(config.seed)
        trainer = RefinementTrainer(scene, reference_field.ReferenceField(arrays, config.query_block_size),
                                    initial, manifest["views"], config, device)
        if resume:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["run_identity"] != run_identity:
                raise ValueError("Resume contract/config/implementation differs")
            trainer.load_state_dict(saved["trainer"])
            log_path = work / "training.jsonl"
            if log_path.exists():
                # A killed writer may leave an incomplete last line after the checkpoint.
                rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines(keepends=True)
                        if line.endswith("\n")]
                log_path.write_text("".join(json.dumps(row)+"\n" for row in rows if row["step"] <= trainer.step), encoding="utf-8")
        cameras = {c.name:c.to(device) for c in cameras_from_scene_manifest(scene.manifest)}
        @lru_cache(maxsize=None)
        def camera_at(name, scale):
            return scaled_camera(cameras[name], scale)
        image_records = {v["label"]:v for v in manifest["images"]}
        def render(v, frame, scale, means, rotations):
            name = manifest['views'][v]['frames'][frame]['camera_name']
            camera = camera_at(name, scale)
            density_grad = config.density_enabled and trainer.phase == 'geometry' and trainer.step < trainer.round_steps
            return scene.render_deformed(camera, means, foreground_quaternions=rotations, density_grad=density_grad), camera

        @lru_cache(maxsize=8)
        def target(v, frame, scale):
            view = manifest["views"][v]
            record = image_records[view["label"]]
            entry = record["files"][frame]
            if Path(entry["name"]).name != entry["name"] or entry["name"] != f"{view['frame_names'][frame]}.png":
                raise ValueError("RGB image order differs from prepared recording")
            path = resolve_path(record["directory"], strict=True) / entry["name"]
            image, _ = load_rgb_frame(path, view["shape_hw"], entry["sha256"])
            return resize_rgb(image, scale).to(device)

        def save():
            _atomic_torch(checkpoint, dict(run_identity=run_identity, contract=contract, trainer=trainer.state_dict()))
        atomic_json(work / "run.json", dict(**contract, run_identity=run_identity, status="running"))
        if not resume:
            save()
        try:
            with (work / "training.jsonl").open("a", encoding="utf-8") as log:
                while trainer.step < trainer.total_steps:
                    row = trainer.update(render, target)
                    log.write(json.dumps(row, allow_nan=False)+"\n"); log.flush()
                    if trainer.step % config.checkpoint_interval == 0 or row['phase_boundary']:
                        save()
                    if trainer.step % 100 == 0:
                        print(f"refinement {trainer.step}/{trainer.total_steps}: {row['gaussian_count']} Gaussians", flush=True)
            publish_refinement(destination, (root, manifest), scene, trainer.field, trainer.mapping,
                trainer.coordinates(), run_identity=run_identity, settings=asdict(config), baked_fields=trainer.frozen_fields())
        except BaseException as error:
            atomic_json(work / "run.json", dict(**contract, run_identity=run_identity, status="failed",
                                                error=str(error), last_complete_checkpoint=str(checkpoint)))
            raise
        atomic_json(work / "run.json", dict(**contract, run_identity=run_identity, status="complete", output=str(destination)))
    return destination
