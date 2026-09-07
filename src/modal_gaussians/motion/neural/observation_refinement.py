"""Optional full-render correction: legacy v13 followers or guarded v15 residuals."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import math
import os
import shutil
import tempfile
import numpy as np
import torch

from modal_gaussians.iteration_cache import module_revision
from modal_gaussians.numpy_io import save_named_arrays
from modal_gaussians.progress import report_progress
from . import neural_modes as nm
from .neural_field import ModalObservation, radial_huber

VERSION = 13
METHOD = "neural_pointwise_observation_refinement"
SEMANTICS = {
    "parent": "immutable_v12_pointwise_displacement_fill",
    "variables": "observable_complex_corrections_of_propagated_points_only",
    "loss": "full_foreground_complex_radial_huber_plus_normalized_correction_prior",
    "nullspace": "retain_parent_via_fixed_visible_jacobian_rowspace_projection",
    "optimizer": "projected_gradient_armijo_backtracking",
    "hosts": "bitwise_unchanged_no_network_training",
}
GUARDED_METHOD = "neural_guarded_observation_refinement"
GUARDED_SEMANTICS = {**SEMANTICS,
    "parent": "immutable_v14_guarded_neighbor_residual_field",
    "variables": "reliable_small_component_residual_points_only; pure_followers_and_sources_fixed",
    "loss": "full_foreground_complex_radial_huber_plus_total_stable_neighbor_residual_prior",
    "nullspace": "retain_parent_via_unchanged_training_observable_projector",
    "prior_weight": "max_requested_and_parent_residual_prior_weight"}


def artifact_contract(parent_version):
    if parent_version == 12: return VERSION, METHOD, SEMANTICS
    if parent_version == 14: return 15, GUARDED_METHOD, GUARDED_SEMANTICS
    raise ValueError("Refinement requires v12 or v14 neural modes")
DELTA_NAMES = {"r_delta_phi", "r_refinement_mask", "r_observable_projector",
               "r_visible_view_mask", "sample_prediction"}


@dataclass(frozen=True)
class ObservationRefinementConfig:
    max_iterations: int = 20
    prior_weight: float = 0.1
    initial_step: float = 1.0
    max_backtracking: int = 16
    relative_tolerance: float = 1e-6
    observable_rtol: float = 1e-3

    def validate(self):
        for name in ("max_iterations", "max_backtracking"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Refinement {name} must be a positive integer")
        for name in ("prior_weight", "initial_step", "relative_tolerance", "observable_rtol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Refinement {name} must be finite and positive")
        if self.observable_rtol >= 1:
            raise ValueError("Refinement observable_rtol must be below one")

    def to_dict(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        result = cls(**value); result.validate()
        return result


def observable_projectors(jacobians, visible, candidate, rtol):
    """Project corrections onto well-observed directions, retaining weak/null axes."""
    J = np.asarray(jacobians, dtype=np.float64)  # [V,G,2,3]
    if (J.ndim != 4 or J.shape[2:] != (2, 3) or visible.shape != (J.shape[1], J.shape[0])
            or candidate.shape != (J.shape[1],) or not np.isfinite(J).all()):
        raise ValueError("Refinement Jacobian/support domains differ")
    norm = np.linalg.norm(J, axis=(2, 3), keepdims=True)
    normalized = J / np.maximum(norm, 1e-30)
    gram = np.einsum('vgai,vgaj,gv->gij', normalized, normalized, visible)
    eigenvalues, vectors = np.linalg.eigh(gram)
    active = (eigenvalues > np.maximum(eigenvalues[:, -1:] * rtol**2, 1e-12)) & candidate[:, None]
    return ((vectors * active[:, None, :]) @ vectors.transpose(0, 2, 1)).astype(np.float32)


def refine_mode(base, observations, projector, *, amplitude_scale, huber_delta, config, prior_offset=None):
    """Joint full-pixel optimization; never duplicate a pixel target per Gaussian."""
    config.validate()
    if not observations or not math.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError("Refinement requires observations and a positive fixed amplitude scale")
    if not base.is_complex() or base.ndim != 2 or base.shape[1] != 3 or not bool(torch.isfinite(base).all()):
        raise ValueError("Refinement parent must be finite complex [G,3]")
    base = base.detach()
    P = torch.as_tensor(projector, dtype=base.real.dtype, device=base.device)
    if P.shape != (len(base), 3, 3) or not bool(torch.isfinite(P).all()):
        raise ValueError("Refinement projector shape/values differ")
    rows = torch.nonzero(P.abs().sum((1, 2)) > 0).flatten()
    if not len(rows):
        return torch.zeros_like(base), {"iterations": 0, "status": "no_visible_propagated_points", "history": []}
    local_P = P[rows].to(base.dtype)
    offset = torch.zeros_like(base[rows])
    if prior_offset is not None:
        prior_offset = torch.as_tensor(prior_offset, device=base.device, dtype=base.dtype).detach()
        if prior_offset.shape != base.shape or not bool(torch.isfinite(prior_offset).all()):
            raise ValueError("Invalid fixed neighbor residual offset")
        offset = prior_offset[rows] / amplitude_scale
    prepared = []
    for observation in observations:
        target = torch.as_tensor(observation.target, device=base.device, dtype=base.dtype).detach()
        weight = torch.as_tensor(observation.confidence, device=base.device, dtype=base.real.dtype).detach()
        scale = float(observation.normalized_rms)
        alpha = complex(observation.alpha)
        if (target.ndim != 2 or target.shape[1] != 2 or weight.shape != target.shape[:1]
                or not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(weight).all())
                or bool((weight <= 0).any()) or not math.isfinite(scale) or scale <= 0
                or not math.isfinite(alpha.real) or not math.isfinite(alpha.imag) or abs(alpha) == 0):
            raise ValueError("Invalid frozen refinement supervision")
        prepared.append((observation.project, target, weight / weight.sum(), scale, alpha))

    def project(z):
        return torch.einsum('gij,gj->gi', local_P, z)

    def objective(value, backward):
        z = value.detach().requires_grad_(backward)
        with torch.set_grad_enabled(backward):
            delta = project(z)
            field = base.index_add(0, rows, amplitude_scale * delta)
            data = 0.0
            for render, target, weight, scale, alpha in prepared:
                prediction = render(field)
                if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError("Invalid full-render refinement prediction")
                term = (weight * radial_huber((alpha * prediction - target) / scale, huber_delta)).sum() / len(prepared)
                data += float(term.detach())
                if backward: term.backward(retain_graph=True)
                del prediction, term
            prior = config.prior_weight * (delta + offset).abs().square().sum(-1).mean()
            if backward: prior.backward()
            total = data + float(prior.detach())
        if not math.isfinite(total) or (backward and not bool(torch.isfinite(z.grad).all())):
            raise FloatingPointError("Nonfinite refinement loss/gradient")
        return total, (z.grad.detach() if backward else None), {"loss": total, "data_loss": data, "prior_loss": total-data}

    z = base.new_zeros((len(rows), 3))
    loss, gradient, record = objective(z, True)
    history = [{"step": 0, **record}]
    status = "iteration_limit"
    for step in range(1, config.max_iterations + 1):
        direction = -len(rows) * project(gradient)
        slope = float((gradient.conj() * direction).real.sum())
        if slope >= -1e-20:
            status = "stationary"; break
        step_size = config.initial_step
        for _ in range(config.max_backtracking):
            candidate = project(z + step_size * direction)
            next_loss, _, next_record = objective(candidate, False)
            if next_loss <= loss + 1e-4 * step_size * slope: break
            step_size *= 0.5
        else:
            status = "line_search_stopped"; break
        improvement = loss - next_loss
        z = candidate.detach()
        history.append({"step": step, "step_size": step_size, **next_record})
        if improvement <= config.relative_tolerance * max(abs(loss), 1e-12):
            status = "converged"; break
        loss, gradient, _ = objective(z, True)
    result = torch.zeros_like(base).index_add(0, rows, amplitude_scale * project(z))
    return result.detach(), {"iterations": len(history)-1, "status": status, "history": history}


def _sources(parent, prepared):
    """Bind the frozen snapshot to the parent; no flow or spectrum reads."""
    if parent.manifest["version"] not in (12, 14) or nm._source_identity(parent.manifest) != prepared.manifest["source_identity"]:
        raise ValueError("Observation refinement needs matching v12/v14 modes and preparation")
    config = nm.NeuralModesConfig.from_dict(parent.manifest["config"])
    prepared.validate_sources(parent.manifest, config)
    slots = nm._source_mode_slots(parent.manifest)
    for name in ("g_points", "sample_pixels_xy", "sample_confidence", "view_sample_offsets",
                 "contribution_mass", "contribution_threshold", "sample_target", "alphas", "alpha_identifiable_mask",
                 "amplitude_scale", "mode_view_loss_scale"):
        expected = prepared.arrays["o_" + name]
        if name in nm.MODE_ARRAYS: expected = expected[slots]
        if not np.array_equal(parent.arrays[name], expected):
            raise ValueError(f"Refinement frozen source differs: {name}")
    from modal_gaussians.static import cameras_from_scene_manifest
    from modal_gaussians.camera_geometry import project_camera
    from .geometry_graph import depth_thresholds_from_manifest
    scene_manifest = json.loads((Path(parent.manifest["static_scene"])/"manifest.json").read_text())
    reference = {c.label: c for c in cameras_from_scene_manifest(scene_manifest) if c.role == "reference"}
    cameras = [reference[v["label"]] for v in parent.manifest["views"]]
    graph_manifest = json.loads((Path(parent.manifest["observed_structure_graph"])/"manifest.json").read_text())
    tolerance, _ = depth_thresholds_from_manifest(graph_manifest, [v["label"] for v in parent.manifest["views"]])
    points = parent.arrays["g_points"].astype(np.float64)
    if parent.manifest["version"] == 14:
        from .guarded_attachments import observation_inputs
        for v, camera in enumerate(cameras):
            if camera.to_manifest_record()["camera_identity"] != parent.manifest["views"][v]["camera_identity"]:
                raise ValueError("Refinement camera differs")
        expected = observation_inputs(points, cameras,
            [prepared.arrays[f"v{v}_depth"] for v in range(len(cameras))],
            [prepared.arrays[f"v{v}_alpha"] for v in range(len(cameras))], tolerance, config.alpha_minimum)
        J = np.stack([prepared.arrays[f"v{v}_jacobian"] for v in range(len(cameras))])
        if (not np.array_equal(expected["h_surface_visible"], parent.arrays["h_surface_visible"])
                or not np.array_equal(expected["h_view_jacobian"], parent.arrays["h_view_jacobian"])
                or not np.array_equal(expected["h_view_jacobian"], J)):
            raise ValueError("Guarded visibility/Jacobian differs from the frozen preparation")
        return cameras, J, expected["h_surface_visible"]
    visible = np.zeros((len(points), len(cameras)), bool)
    for v, camera in enumerate(cameras):
        if camera.to_manifest_record()["camera_identity"] != parent.manifest["views"][v]["camera_identity"]:
            raise ValueError("Refinement camera differs")
        transform = camera.world_to_camera.cpu().numpy()
        xyz = points @ transform[:3,:3].T + transform[:3,3]
        uv = project_camera(xyz, camera.K.cpu().numpy(), camera.radial_distortion)
        depth, alpha = prepared.arrays[f"v{v}_depth"], prepared.arrays[f"v{v}_alpha"]
        valid = np.isfinite(uv).all(1) & (xyz[:,2] > 0)
        xy = np.zeros((len(points),2), np.int64); xy[valid] = np.rint(uv[valid]).astype(np.int64)
        valid &= (xy >= 0).all(1) & (xy[:,0] < depth.shape[1]) & (xy[:,1] < depth.shape[0])
        rows = np.flatnonzero(valid)
        z = depth[xy[rows,1],xy[rows,0]]
        visible[rows,v] = (np.isfinite(z) & (z > 0) & (tolerance[v] > 0)
            & (np.abs(xyz[rows,2]-z) <= tolerance[v])
            & (alpha[xy[rows,1],xy[rows,0]] >= .05))
    return cameras, np.stack([prepared.arrays[f"v{v}_jacobian"] for v in range(len(cameras))]), visible


def _masks(parent, jacobians, visible, config):
    if parent.manifest["version"] == 14:
        if not np.array_equal(visible, parent.arrays["h_surface_visible"]):
            raise ValueError("Guarded refinement cannot change training visibility")
        return (parent.arrays["h_reliable_view_mask"].copy(), parent.arrays["h_residual_mask"].copy(),
                parent.arrays["h_residual_projector"].copy())
    views = parent.arrays["observation_view_mask"] & visible[None]
    mask = (parent.arrays["support_class"] == 3) & views.any(2)
    projectors = np.stack([observable_projectors(jacobians, views[k], mask[k], config.observable_rtol)
                          for k in range(len(mask))])
    return views, mask, projectors


def _effective_settings(parent, settings):
    if parent.manifest["version"] == 14:
        return replace(settings, prior_weight=max(settings.prior_weight, float(parent.arrays["h_residual_prior_weight"])))
    return settings


def _prior_offset(parent, mode):
    if parent.manifest["version"] != 14: return None
    arrays = parent.arrays
    weights, neighbors = arrays["h_neighbor_weight"][mode], arrays["h_neighbor_index"][mode]
    field = arrays["phi"][mode]
    prior = np.sum(weights[..., None] * field[np.maximum(neighbors, 0)], axis=1)
    return (field - prior) * arrays["h_residual_mask"][mode, :, None]


def identity_payload(manifest):
    return {k: v for k, v in manifest.items() if k not in ("completed_modes_identity", "producer")}


def load_refined_modes(path):
    from .prepared import load_prepared
    root = Path(path).resolve(strict=True)
    manifest = json.loads((root/"manifest.json").read_text())
    expected = artifact_contract(14 if manifest.get("version") == 15 else 12)
    if (manifest.get("format") != nm.COMPLETED_MODES_FORMAT or type(manifest.get("version")) is not int
            or manifest["version"] != expected[0] or manifest.get("completion_method") != expected[1]
            or manifest.get("semantics") != expected[2]):
        raise ValueError("Unsupported observation-refined modes")
    if nm._identity(identity_payload(manifest)) != manifest.get("completed_modes_identity"):
        raise ValueError("Refinement identity differs")
    parent_path = Path(manifest["parent_completed_modes"]).resolve(strict=True)
    if parent_path == root or parent_path.is_relative_to(root) or root.is_relative_to(parent_path):
        raise ValueError("Refinement and parent paths overlap")
    parent = nm.load_neural_completed_modes(parent_path)
    if artifact_contract(parent.manifest["version"]) != expected:
        raise ValueError("Refinement parent version differs")
    prepared = load_prepared(manifest["prepared"])
    if (parent.manifest["completed_modes_identity"] != manifest["parent_completed_modes_identity"]
            or prepared.manifest["prepared_identity"] != manifest["prepared_identity"]):
        raise ValueError("Refinement parent/preparation identity differs")
    overridden = {"version", "completion_method", "semantics", "completed_modes_identity", "diagnostics",
                  "arrays", "arrays_identity", "arrays_file_sha256", "producer", "networks_file", "networks_sha256"}
    for name, value in parent.manifest.items():
        if name not in overridden and manifest.get(name) != value:
            raise ValueError(f"Refinement inherited metadata differs: {name}")
    if manifest.get("arrays_file") != nm.ARRAYS_FILENAME or nm._sha256(root/nm.ARRAYS_FILENAME) != manifest["arrays_file_sha256"]:
        raise ValueError("Refinement array checksum differs")
    with np.load(root/nm.ARRAYS_FILENAME, allow_pickle=False) as archive:
        delta = {k: archive[k] for k in archive.files}
    if set(delta) != DELTA_NAMES or nm._arrays_identity(delta) != manifest["arrays_identity"]:
        raise ValueError("Refinement array inventory/identity differs")
    if manifest["arrays"] != {k: {"dtype": v.dtype.name, "shape": list(v.shape)} for k,v in delta.items()}:
        raise ValueError("Refinement array schema differs")
    config = ObservationRefinementConfig.from_dict(manifest["refinement_config"])
    if manifest["version"] == 15 and manifest.get("effective_refinement_config") != _effective_settings(parent, config).to_dict():
        raise ValueError("Guarded refinement effective prior differs")
    if config.to_dict() != manifest["refinement_config"]:
        raise ValueError("Refinement configuration must be fully resolved")
    contract = manifest.get("refinement_contract", {})
    if (set(contract) != {"parent", "prepared", "config", "code"}
            or contract["parent"] != manifest["parent_completed_modes_identity"]
            or contract["prepared"] != manifest["prepared_identity"]
            or contract["config"] != config.to_dict()
            or not isinstance(contract["code"], str) or len(contract["code"]) != 64):
        raise ValueError("Refinement contract differs from its sources/configuration")
    _, jacobians, visibility = _sources(parent, prepared)
    views, mask, projectors = _masks(parent, jacobians, visibility, config)
    for name, expected in (("r_visible_view_mask", views), ("r_refinement_mask", mask), ("r_observable_projector", projectors)):
        if delta[name].dtype != expected.dtype or not np.array_equal(delta[name], expected):
            raise ValueError(f"Refinement support/projector differs: {name}")
    change = delta["r_delta_phi"]
    if change.dtype != np.complex64 or change.shape != parent.arrays["phi"].shape or not np.isfinite(change).all():
        raise ValueError("Invalid refinement displacement")
    if np.any(change[~mask] != 0):
        raise ValueError("Refinement changed hosts or unobserved points")
    projected = np.einsum('kgij,kgj->kgi', projectors, change)
    scale = parent.arrays["amplitude_scale"][:,None,None]
    if not np.all(np.abs(projected-change) <= 3e-5*np.abs(change) + 1e-7*scale):
        raise ValueError("Refinement changed an unobservable direction")
    prediction = delta["sample_prediction"]
    if (prediction.dtype != np.complex64 or prediction.shape != parent.arrays["sample_prediction"].shape
            or not np.isfinite(prediction).all()):
        raise ValueError("Invalid refined sampled predictions")
    for k,v in np.argwhere(~parent.arrays["alpha_identifiable_mask"]):
        lo,hi=parent.arrays["view_sample_offsets"][v:v+2]
        if np.any(prediction[k,lo:hi] != 0): raise ValueError("Refinement excluded view must be zero")
    arrays = {**parent.arrays, **delta, "phi": parent.arrays["phi"] + change}
    if not np.isfinite(arrays["phi"]).all(): raise ValueError("Nonfinite refined field")
    return nm.NeuralModesArtifact(root, manifest, arrays)


def build_refined_modes(*, parent, prepared, output_dir, config=None, work_dir=None):
    """No NN training, geometry preparation, DFT or coordinates; reusable per-mode checkpoints."""
    from modal_gaussians.static import load_static_scene
    settings = config or ObservationRefinementConfig(); settings.validate()
    effective_settings = _effective_settings(parent, settings)
    version, method, semantics = artifact_contract(parent.manifest["version"])
    destination = Path(output_dir).resolve()
    if destination.exists(): raise FileExistsError(destination)
    if destination.is_relative_to(parent.path) or parent.path.is_relative_to(destination):
        raise ValueError("Refinement output and parent must be disjoint")
    cameras, jacobians, visible = _sources(parent, prepared)
    views, mask, P = _masks(parent, jacobians, visible, settings)
    if not torch.cuda.is_available(): raise RuntimeError("Full-render observation refinement requires CUDA")
    scene = load_static_scene(parent.manifest["static_scene"], "cuda"); scene.eval()
    for parameter in scene.parameters(): parameter.requires_grad_(False)
    projectors = []
    a = parent.arrays
    for v,camera in enumerate(cameras):
        lo,hi = a["view_sample_offsets"][v:v+2]
        projectors.append(nm.FrozenModalProjector(scene,camera.to("cuda"),torch.as_tensor(jacobians[v],device="cuda"),
            a["sample_pixels_xy"][lo:hi],torch.as_tensor(a["sample_confidence"][lo:hi],device="cuda")))
    import sys
    contract = {"parent": parent.manifest["completed_modes_identity"], "prepared": prepared.manifest["prepared_identity"],
                "config": settings.to_dict(), "code": module_revision(sys.modules[__name__])}
    work = Path(work_dir).resolve() if work_dir else destination.with_name(destination.name+"_work")
    for protected in (parent.path.resolve(), prepared.path.resolve(), destination):
        if work == protected or work.is_relative_to(protected) or protected.is_relative_to(work):
            raise ValueError("Refinement work and input/output artifacts must be disjoint")
    work.mkdir(parents=True, exist_ok=True)
    if (work/"contract.json").exists():
        if json.loads((work/"contract.json").read_text()) != contract:
            raise ValueError("Refinement resume configuration/sources/code changed")
    else: nm._atomic_json(work/"contract.json",contract)
    changes, reports = [], []
    prediction = np.zeros_like(a["sample_prediction"])
    for k in range(len(a["phi"])):
        checkpoint = work/f"mode_{k:03d}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint,map_location="cpu",weights_only=True)
            if saved["contract"] != contract or saved["mode"] != k:
                raise ValueError("Refinement checkpoint differs")
            change = saved["delta"].numpy(); report = saved["report"]
            if (change.dtype != np.complex64 or change.shape != a["phi"][k].shape
                    or not np.isfinite(change).all() or np.any(change[~mask[k]] != 0)):
                raise ValueError("Invalid refinement checkpoint")
        else:
            observations=[]
            for v,render in enumerate(projectors):
                if not a["alpha_identifiable_mask"][k,v]: continue
                lo,hi=a["view_sample_offsets"][v:v+2]
                observations.append(ModalObservation(a["sample_target"][k,lo:hi],render,
                    alpha=complex(a["alphas"][k,v]),confidence=a["sample_confidence"][lo:hi],
                    normalized_rms=float(a["mode_view_loss_scale"][k,v])))
            report_progress(f"observation refinement: mode {k+1}/{len(a['phi'])}, eligible correction points={int(mask[k].sum())}")
            delta,report=refine_mode(torch.as_tensor(a["phi"][k],device="cuda"),observations,P[k],
                amplitude_scale=float(a["amplitude_scale"][k]),huber_delta=parent.manifest["config"]["huber_delta"],
                config=effective_settings,prior_offset=_prior_offset(parent,k))
            change=delta.cpu().numpy().astype(np.complex64)
            nm._atomic_torch(checkpoint,{"contract":contract,"mode":k,"delta":torch.from_numpy(change),"report":report})
        changes.append(change); reports.append(report)
        with torch.no_grad():
            field=torch.as_tensor(a["phi"][k]+change,device="cuda")
            for v,render in enumerate(projectors):
                if a["alpha_identifiable_mask"][k,v]:
                    lo,hi=a["view_sample_offsets"][v:v+2]
                    prediction[k,lo:hi]=(complex(a["alphas"][k,v])*render(field)).cpu().numpy()
    delta={"r_delta_phi":np.stack(changes),"r_refinement_mask":mask,"r_observable_projector":P,
           "r_visible_view_mask":views,"sample_prediction":prediction}
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=f".{destination.name}.",dir=destination.parent)).resolve()
    try:
        save_named_arrays(temporary/nm.ARRAYS_FILENAME,delta)
        manifest={k:v for k,v in parent.manifest.items() if k not in ("networks_file","networks_sha256")}
        manifest.update(version=version,completion_method=method,semantics=semantics,
            parent_completed_modes=str(parent.path),parent_completed_modes_identity=parent.manifest["completed_modes_identity"],
            prepared=str(prepared.path),prepared_identity=prepared.manifest["prepared_identity"],refinement_config=settings.to_dict(),
            refinement_contract=contract,diagnostics={"per_mode":reports,"refined_points":mask.sum(1).tolist()},
            arrays={k:{"dtype":v.dtype.name,"shape":list(v.shape)} for k,v in delta.items()},
            arrays_identity=nm._arrays_identity(delta),arrays_file_sha256=nm._sha256(temporary/nm.ARRAYS_FILENAME))
        if version == 15:
            manifest["effective_refinement_config"] = effective_settings.to_dict()
        manifest["completed_modes_identity"]=nm._identity(identity_payload(manifest))
        nm._atomic_json(temporary/"manifest.json",manifest)
        validated=load_refined_modes(temporary)
        os.rename(temporary,destination)
    except BaseException:
        if temporary.exists() and temporary.parent == destination.parent and temporary.name.startswith(f".{destination.name}."):
            shutil.rmtree(temporary)
        raise
    return nm.NeuralModesArtifact(destination,validated.manifest,validated.arrays)
