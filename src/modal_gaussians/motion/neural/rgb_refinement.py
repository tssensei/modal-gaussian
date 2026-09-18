"""RGB refinement of shared frequency-labelled modes and per-video responses.

This is an independent experiment: it does not rewrite neural artifacts or fit
mass/stiffness/damping. Time is measured from each video's first frame, with no
forced zero displacement at its reference frame.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

from modal_gaussians.iteration_cache import atomic_json, exclusive_work, identity, module_revision
from modal_gaussians.progress import Progress, report_progress

FORMAT = "modal_gaussians.rgb_mode_refinement"


@dataclass(frozen=True)
class RGBRefinementConfig:
    envelope_bandwidth_hz: float = 0.05
    coefficient_warmup_steps: int = 200
    alternating_rounds: int = 4
    coefficient_steps: int = 50
    shape_steps: int = 50
    coefficient_learning_rate: float = 0.01
    shape_learning_rate: float = 0.001
    shape_prior_weight: float = 0.01
    max_width: int = 0
    background_weight: float = 0.05
    seed: int = 1729
    device: str = "auto"
    early_stopping_patience: int = 0
    early_stopping_min_rounds: int = 5
    early_stopping_relative_delta: float = 1e-4
    early_stopping_frames_per_view: int = 16

    def validate(self):
        for name in ("coefficient_warmup_steps", "alternating_rounds", "coefficient_steps", "shape_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_width) is not int or self.max_width < 0:
            raise ValueError("max_width must be nonnegative; zero preserves original resolution")
        for name in ("coefficient_learning_rate", "shape_learning_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("envelope_bandwidth_hz", "shape_prior_weight", "background_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.background_weight > 1 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("Invalid background weight or seed")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if type(self.early_stopping_patience) is not int or self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be nonnegative; zero disables monitoring")
        for name in ("early_stopping_min_rounds", "early_stopping_frames_per_view"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.early_stopping_relative_delta, bool)
                or not math.isfinite(self.early_stopping_relative_delta)
                or not 0 <= self.early_stopping_relative_delta < 1):
            raise ValueError("early_stopping_relative_delta must be in [0,1)")


def envelope_basis(times: torch.Tensor, duration: float, bandwidth_hz: float):
    """Real DCT envelope frequencies n/(2T), including DC, bounded by bandwidth."""
    count = 1 + math.floor(2 * duration * bandwidth_hz)
    order = torch.arange(count, dtype=times.dtype, device=times.device)
    return torch.cos(math.pi * times[..., None] * order / duration)


class SharedRGBModes(nn.Module):
    def __init__(self, phi0, frequencies_hz, durations_seconds, *, bandwidth_hz=0.05,
                 initial_envelopes=None, coefficient_scales=None, rotation=None):
        super().__init__()
        initial = torch.as_tensor(phi0, dtype=torch.complex64).detach().clone()
        frequencies = torch.as_tensor(frequencies_hz, dtype=torch.float32, device=initial.device)
        durations = [float(value) for value in durations_seconds]
        if (initial.ndim != 3 or initial.shape[-1] != 3 or min(initial.shape) < 1
                or frequencies.shape != (len(initial),) or not torch.isfinite(initial).all()
                or not torch.isfinite(frequencies).all() or (frequencies <= 0).any()):
            raise ValueError("Expected finite complex modes [K,G,3] and positive frequencies [K]")
        if not durations or any(not math.isfinite(t) or t <= 0 for t in durations):
            raise ValueError("Each video must have a positive duration")
        bandwidth = float(bandwidth_hz)
        if not math.isfinite(bandwidth) or bandwidth < 0 or bandwidth >= float(frequencies.min()):
            raise ValueError("Envelope bandwidth must be nonnegative and below every carrier frequency")
        if len(frequencies) > 1 and (torch.diff(frequencies.sort().values) <= 2 * bandwidth).any():
            raise ValueError("Frequency bands must be distinct and non-overlapping")
        rms = initial.abs().square().mean(dim=(1, 2)).sqrt()
        if (rms <= 0).any():
            raise ValueError("Cannot refine a zero-energy mode")
        self.register_buffer("phi0", initial)
        self.register_buffer("frequencies_hz", frequencies)
        self.register_buffer("mode_rms", rms)
        self.durations_seconds = durations
        self.bandwidth_hz = bandwidth
        self.delta = nn.Parameter(torch.zeros((*initial.shape, 2), device=initial.device))
        scale = torch.ones((len(durations), len(initial)), device=initial.device) if coefficient_scales is None else torch.as_tensor(coefficient_scales, dtype=torch.float32, device=initial.device)
        if scale.shape != (len(durations), len(initial)) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Coefficient scales must be positive [V,K]")
        self.register_buffer("coefficient_scales", scale.clone())
        seeds = torch.zeros_like(scale, dtype=torch.complex64) if initial_envelopes is None else torch.as_tensor(initial_envelopes, dtype=torch.complex64, device=initial.device)
        if seeds.shape != scale.shape or not torch.isfinite(seeds).all():
            raise ValueError("Initial envelopes must be finite [V,K]")
        self.envelopes = nn.ParameterList()
        for view, duration in enumerate(durations):
            values = torch.zeros((len(initial), 1 + math.floor(2 * duration * bandwidth), 2), device=initial.device)
            values[:, 0] = torch.view_as_real(seeds[view] / scale[view])
            self.envelopes.append(nn.Parameter(values))
        angular = None if rotation is None else torch.as_tensor(rotation, dtype=torch.complex64, device=initial.device).detach().clone()
        if angular is not None and (angular.shape != initial.shape or not torch.isfinite(angular).all()):
            raise ValueError("Rotation modes must be finite [K,G,3]")
        self.register_buffer("rotation", angular)

    def delta_phi(self):
        raw = torch.view_as_complex(self.delta) * self.mode_rms[:, None, None]
        # Fix the complex gain/phase gauge: the residual cannot rescale phi0.
        projection = (self.phi0.conj() * raw).sum(dim=(1, 2)) / self.phi0.abs().square().sum(dim=(1, 2))
        return raw - self.phi0 * projection[:, None, None]

    def phi(self):
        return self.phi0 + self.delta_phi()

    def coefficients(self, view: int, times):
        t = torch.as_tensor(times, dtype=self.frequencies_hz.dtype, device=self.phi0.device)
        basis = envelope_basis(t, self.durations_seconds[view], self.bandwidth_hz)
        weights = torch.view_as_complex(self.envelopes[view]) * self.coefficient_scales[view, :, None]
        envelope = torch.einsum("...l,kl->...k", basis.to(weights.dtype), weights)
        carrier = torch.exp(2j * math.pi * t[..., None] * self.frequencies_hz)
        return envelope * carrier

    def displacement(self, view, times):
        # Real(c*phi) = Re(c)Re(phi) - Im(c)Im(phi).
        return torch.einsum("...k,kgd->...gd", self.coefficients(view, times), self.phi()).real

    def shape_penalty(self):
        return (self.delta_phi() / self.mode_rms[:, None, None]).abs().square().mean()


def optimization_phases(config):
    phases = [("coefficients", config.coefficient_warmup_steps)]
    for _ in range(config.alternating_rounds):
        phases.extend((("shape", config.shape_steps), ("coefficients", config.coefficient_steps)))
    return phases


def _rgb_frame_loss(model, scene, view, view_index, frame, base_means, base_quaternions):
    """The identical RGB objective for gradient steps and requested early stopping."""
    from modal_gaussians.motion.common.rotations import rotate_gaussian_quaternions

    target, weight = view.read_frame(frame)
    coefficients = model.coefficients(view_index, float(view.times[frame]))
    means = base_means + torch.einsum("k,kgd->gd", coefficients, model.phi()).real
    quaternions = None
    if model.rotation is not None:
        angles = torch.einsum("k,kgd->gd", coefficients, model.rotation).real
        quaternions = rotate_gaussian_quaternions(base_quaternions, angles)
    prediction = scene.render_deformed(view.camera, means, foreground_quaternions=quaternions)["rgb"]
    if prediction.shape != target.shape or weight.shape != target.shape[:2] or not bool(weight.sum() > 0):
        raise ValueError("RGB target/weight dimensions or support differ from the render")
    return ((prediction - target).abs() * weight[..., None]).sum() / (3 * weight.sum())


@torch.no_grad()
def _monitor_rgb_objective(model, scene, views, frames, config, base_means, base_quaternions):
    # These frames stay eligible for training; there is no held-out validation set.
    per_view = []
    for view_index, indices in enumerate(frames):
        losses = [_rgb_frame_loss(model, scene, views[view_index], view_index, frame,
                                  base_means, base_quaternions).item() for frame in indices]
        per_view.append(float(np.mean(losses)))
    rgb_loss = float(np.mean(per_view))
    prior = model.shape_penalty().item()
    loss = rgb_loss + config.shape_prior_weight * prior
    if not math.isfinite(loss):
        raise FloatingPointError("Non-finite early-stopping training objective")
    return dict(loss=loss, rgb_loss=rgb_loss, shape_prior=prior,
                relative_shape_rms=math.sqrt(prior), per_view_rgb=per_view)


def optimize_rgb_modes(model, scene, views, config, *, checkpoint=None, on_phase_complete=None):
    """One RGB frame per step; only one parameter block is trainable at a time."""
    config.validate()
    scene.eval()
    for parameter in scene.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    foreground = scene.foreground.active()
    base_means = foreground["means"].detach()
    base_quaternions = foreground["quaternions"].detach()
    if base_means.shape != model.phi0.shape[1:] or len(views) != len(model.envelopes):
        raise ValueError("Scene or videos do not match the modal domain")
    optimizers = {
        "coefficients": torch.optim.Adam(model.envelopes.parameters(), lr=config.coefficient_learning_rate),
        "shape": torch.optim.Adam([model.delta], lr=config.shape_learning_rate),
    }
    rng = np.random.default_rng(config.seed)
    history, start_phase = [], 0
    stopping = dict(frames=[], records=[], best_loss=None, best_model=None, best_round=None,
                    best_step=None, anchor_loss=None, stale_rounds=0, stop_reason=None,
                    completed_steps=0, shape_steps=0, coefficient_steps=0,
                    training_seconds=0., monitor_seconds=0.)
    if config.early_stopping_patience:
        monitor_rng = np.random.default_rng(config.seed + 1)
        for view in views:
            count = min(len(view.frame_names), config.early_stopping_frames_per_view)
            edges = np.linspace(0, len(view.frame_names), count + 1, dtype=np.int64)
            stopping["frames"].append([int(monitor_rng.integers(lo, hi))
                                       for lo, hi in zip(edges[:-1], edges[1:])])
    if checkpoint is not None:
        start_phase = checkpoint["next_phase"]
        if type(start_phase) is not int or not 0 <= start_phase <= len(optimization_phases(config)):
            raise ValueError("Checkpoint next_phase is outside the optimization schedule")
        model.load_state_dict(checkpoint["model"])
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][key])
        rng.bit_generator.state = checkpoint["rng_state"]
        history, start_phase = list(checkpoint["history"]), checkpoint["next_phase"]
        stopping = checkpoint["stopping"]
    for phase_index, (phase, steps) in enumerate(optimization_phases(config)):
        if stopping["stop_reason"] is not None:
            break
        if phase_index < start_phase:
            continue
        model.delta.requires_grad_(phase == "shape")
        for parameter in model.envelopes:
            parameter.requires_grad_(phase == "coefficients")
        parameters = [p for p in model.parameters() if p.requires_grad]
        progress = Progress(f"RGB refinement {phase_index + 1}: {phase}", steps, unit="steps")
        started = time.perf_counter()
        for step in range(steps):
            view_index = int(rng.integers(len(views)))
            view = views[view_index]
            frame = int(rng.integers(len(view.frame_names)))
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            rgb_loss = _rgb_frame_loss(model, scene, view, view_index, frame, base_means, base_quaternions)
            prior = model.shape_penalty()
            loss = rgb_loss + config.shape_prior_weight * prior
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite RGB refinement loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizers[phase].step()
            if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
                history.append(dict(phase_index=phase_index, phase=phase, step=step + 1,
                                    view=view_index, frame=frame, loss=float(loss.detach()),
                                    rgb_loss=float(rgb_loss.detach()), shape_prior=float(prior.detach())))
            progress.update(step + 1)
        stopping["training_seconds"] += time.perf_counter() - started
        stopping["completed_steps"] += steps
        stopping["shape_steps" if phase == "shape" else "coefficient_steps"] += steps
        if config.early_stopping_patience and phase == "coefficients":
            started = time.perf_counter()
            record = _monitor_rgb_objective(model, scene, views, stopping["frames"], config,
                                            base_means, base_quaternions)
            stopping["monitor_seconds"] += time.perf_counter() - started
            round_index = phase_index // 2
            record.update(round=round_index, completed_steps=stopping["completed_steps"])
            stopping["records"].append(record)
            score = record["loss"]
            if stopping["best_loss"] is None or score < stopping["best_loss"]:
                stopping.update(best_loss=score, best_round=round_index, best_step=stopping["completed_steps"],
                                best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            anchor = stopping["anchor_loss"]
            if anchor is None or anchor - score > config.early_stopping_relative_delta * max(abs(anchor), 1e-12):
                stopping.update(anchor_loss=score, stale_rounds=0)
            else:
                stopping["stale_rounds"] += 1
            if round_index >= config.early_stopping_min_rounds and stopping["stale_rounds"] >= config.early_stopping_patience:
                stopping["stop_reason"] = "early_stopping_plateau"
            report_progress(f"RGB training monitor: round={round_index} loss={score:.8g} "
                f"rgb={record['rgb_loss']:.8g} shape_rms={record['relative_shape_rms']:.3%} "
                f"best={stopping['best_loss']:.8g} stale={stopping['stale_rounds']}/{config.early_stopping_patience}")
        if phase_index == len(optimization_phases(config)) - 1 and stopping["stop_reason"] is None:
            stopping["stop_reason"] = "step_limit_reached" if config.early_stopping_patience else "fixed_schedule_completed"
        if on_phase_complete is not None:
            on_phase_complete(dict(model=model.state_dict(), optimizers={k: v.state_dict() for k, v in optimizers.items()},
                                   rng_state=rng.bit_generator.state, next_phase=phase_index + 1, history=history,
                                   stopping=stopping))
    # Keep checkpoint model/Adam/RNG at the last completed block. Best is restored
    # only for publication, never paired with the latest optimizer on resume.
    if stopping["best_model"] is not None:
        model.load_state_dict(stopping["best_model"])
    model.optimization_summary = {k: v for k, v in stopping.items() if k != "best_model"}
    model.optimization_summary["early_stopping_enabled"] = bool(config.early_stopping_patience)
    model.optimization_summary["published_relative_shape_rms"] = model.shape_penalty().detach().sqrt().item()
    report_progress(f"RGB refinement stopped: {stopping['stop_reason']}; "
                    f"steps={stopping['completed_steps']} shape_steps={stopping['shape_steps']} "
                    f"selected_round={stopping['best_round']}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return history


def _mode_directory(path):
    root = Path(path).expanduser().resolve(strict=True)
    outputs = root / "outputs.json"
    if outputs.is_file():
        root = Path(json.loads(outputs.read_text(encoding="utf-8"))["completed_modes"]).resolve(strict=True)
    return root


def _load_initial_modes(paths, prepared_manifest, scene):
    from modal_gaussians.motion.common.completed_modes import load_completed_modes

    expected = prepared_manifest["source"]
    rows, parents, seen = [], [], set()
    for path in paths:
        artifact = load_completed_modes(_mode_directory(path))
        m = artifact.manifest
        if m.get("source_identity") != prepared_manifest["source_identity"]:
            raise ValueError("Initial modes and prepared RGB inputs have different source domains")
        for key in ("static_scene_identity", "foreground_identity"):
            if m[key] != expected[key] or m[key] != scene.manifest[key]:
                raise ValueError(f"Initial modes differ in {key}")
        if [(v["label"], v["camera_identity"]) for v in m["views"]] != [(v["label"], v["camera_identity"]) for v in expected["views"]]:
            raise ValueError("Initial modes and prepared videos have different view domains")
        if m.get("version") != 16:
            raise ValueError("RGB refinement currently requires v16 component-field initial modes")
        parents.append(dict(path=str(artifact.path), identity=m["completed_modes_identity"]))
        for local, mode in enumerate(m["modes"]):
            slot = int(mode.get("source_mode_slot", mode["mode_slot"]))
            if slot in seen or not 0 <= slot < len(expected["modes"]) or not math.isclose(mode["frequency_hz"], expected["modes"][slot]["frequency_hz"], rel_tol=0, abs_tol=1e-9):
                raise ValueError("Duplicate or mismatched source mode slot")
            seen.add(slot)
            rows.append((slot, mode, artifact.arrays["phi"][local], artifact.rotation[local],
                         artifact.arrays["alphas"][local], artifact.arrays["alpha_identifiable_mask"][local]))
    if not rows:
        raise ValueError("At least one initial mode is required")
    rows.sort(key=lambda row: row[0])
    phi = np.stack([row[2] for row in rows])
    if phi.shape[1:] != tuple(scene.foreground.active()["means"].shape):
        raise ValueError("Initial mode foreground count differs from scene")
    return rows, parents


def refine_rgb_modes(*, prepared_dir, initial_modes, output_dir, config_path=None, resume=False):
    """Create an independent RGB artifact; never write into an input artifact."""
    from modal_gaussians.static import load_static_scene
    from modal_gaussians.motion.neural.neural_modes import _atomic_torch
    from modal_gaussians.motion.common import rotations
    from . import rgb_video
    from .rgb_video import build_video_views

    values = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
    if not isinstance(values, dict) or set(values) - {field.name for field in fields(RGBRefinementConfig)}:
        raise ValueError("RGB configuration must be an object of RGBRefinementConfig fields")
    config = RGBRefinementConfig(**values)
    config.validate()
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))
    root = Path(output_dir).expanduser().resolve()
    prepared = Path(prepared_dir).expanduser().resolve(strict=True)
    if root == prepared or root.is_relative_to(prepared):
        raise ValueError("RGB output must not overwrite prepared inputs")
    if root.exists() and not resume:
        raise FileExistsError(root)
    if resume and not (root / "run.json").is_file():
        raise ValueError("No RGB refinement run to resume")
    m = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if m.get("format") != "modal_gaussians.neural_prepared" or m.get("version") != 1:
        raise ValueError("Unsupported prepared metadata")
    scene = load_static_scene(m["source"]["static_scene"], str(device))
    rows, parents = _load_initial_modes(initial_modes, m, scene)
    inputs = [prepared, Path(m["source"]["static_scene"]).resolve(),
              *(Path(parent["path"]) for parent in parents),
              *(Path(path).expanduser().resolve() for path in initial_modes)]
    if any(root == path or root.is_relative_to(path) for path in inputs):
        raise ValueError("RGB output must be outside its input artifacts")
    views = build_video_views(m, scene, max_width=config.max_width, device=device, background_weight=config.background_weight)
    frequencies = [row[1]["frequency_hz"] for row in rows]
    if any(max(frequencies) + config.envelope_bandwidth_hz >= view.fps_hz / 2 for view in views):
        raise ValueError("Carrier plus envelope band must be below each video's Nyquist frequency")
    # Approximate inverse of the original unnormalized, mean-centred Hann DFT.
    # RGB warmup calibrates deviations from the single-tone approximation.
    scales = np.array([2.0 / np.hanning(len(v.frame_names)).sum() for v in views], np.float32)
    scales = np.repeat(scales[:, None], len(rows), axis=1)
    alpha = np.stack([row[4] for row in rows], axis=1)
    accepted = np.stack([row[5] for row in rows], axis=1)
    model = SharedRGBModes(np.stack([r[2] for r in rows]), frequencies,
                           [v.duration_seconds for v in views], bandwidth_hz=config.envelope_bandwidth_hz,
                           coefficient_scales=scales, initial_envelopes=np.where(accepted, alpha, 0) * scales,
                           rotation=np.stack([r[3] for r in rows])).to(device)
    contract = dict(format=FORMAT, version=1, prepared=str(prepared), prepared_identity=m["prepared_identity"],
                    static_scene=m["source"]["static_scene"], static_scene_identity=m["source"]["static_scene_identity"],
                    parents=parents, modes=[dict(source_mode_slot=r[0], frequency_hz=r[1]["frequency_hz"]) for r in rows],
                    views=[v.metadata for v in views], config=asdict(config), device=str(device),
                    code_revision=module_revision(sys.modules[__name__], rgb_video, rotations))
    run_identity = identity(contract)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_work(root / "run.lock"):
        if (root / "manifest.json").exists():
            raise FileExistsError("RGB refinement is already complete")
        if resume:
            if json.loads((root / "run.json").read_text(encoding="utf-8")) != contract:
                raise ValueError("RGB inputs/settings changed; use a new output directory")
        else:
            atomic_json(root / "run.json", contract)
        checkpoint = torch.load(root / "checkpoint.pt", map_location=device, weights_only=True) if resume and (root / "checkpoint.pt").exists() else None
        if checkpoint is not None and checkpoint.get("run_identity") != run_identity:
            raise ValueError("RGB checkpoint belongs to a different run")

        def save_checkpoint(state):
            _atomic_torch(root / "checkpoint.pt", dict(run_identity=run_identity, **state))

        atomic_json(root / "status.json", dict(status="running", validation_enabled=False, viewer_started=False))
        try:
            history = optimize_rgb_modes(model, scene, views, config, checkpoint=checkpoint, on_phase_complete=save_checkpoint)
            with torch.no_grad():
                payload = dict(phi0=model.phi0.cpu(), phi=model.phi().cpu(), delta_phi=model.delta_phi().cpu(),
                               rotation=model.rotation.cpu(), frequencies_hz=model.frequencies_hz.cpu(),
                               envelopes=[(torch.view_as_complex(p) * model.coefficient_scales[i, :, None]).cpu() for i, p in enumerate(model.envelopes)],
                               times=[torch.as_tensor(v.times).cpu() for v in views],
                               coefficients=[model.coefficients(i, v.times).cpu() for i, v in enumerate(views)])
            _atomic_torch(root / "motion.pt", payload)
            atomic_json(root / "history.json", history)
            atomic_json(root / "optimization.json", model.optimization_summary)
            atomic_json(root / "manifest.json", dict(**contract, run_identity=run_identity, motion_file="motion.pt",
                        optimization_file="optimization.json",
                        semantics=dict(spatial="shared_phi0_plus_complex_orthogonal_residual", temporal="per_view_DCT_envelope_times_positive_carrier",
                                       time_origin="video_first_frame", reference_subtraction=False, static_scene="frozen",
                                       rotations="fixed_initial_field_driven_by_same_coefficients", supervision="source_RGB_weighted_L1",
                                       gauge="Hermitian_inner_product_phi0_delta_equals_zero", validation_enabled=False,
                                       selection="best_fixed_training_subset_objective" if config.early_stopping_patience else "last_step")))
            atomic_json(root / "status.json", dict(status="rgb_modes_ready", viewer_started=False, visualization_checked=False,
                        stop_reason=model.optimization_summary["stop_reason"],
                        completed_steps=model.optimization_summary["completed_steps"],
                        selected_round=model.optimization_summary["best_round"]))
        except BaseException:
            atomic_json(root / "status.json", dict(status="failed", resume="last_completed_parameter_block", validation_enabled=False))
            raise
    return root


def load_rgb_refinement(path):
    """Load the independent result, without replaying optimization or validation."""
    root = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("version") != 1 or manifest.get("motion_file") != "motion.pt":
        raise ValueError("Unsupported RGB refinement artifact")
    return manifest, torch.load(root / "motion.pt", map_location="cpu", weights_only=True)
