"""Postprocess v8 neural fields by attaching compact fragments to fixed hosts.

The derived v9 artifact binds its immutable v8 parent and replays the entire
geometric assignment and complex displacement transfer in its strict loader.
No optimization, new observation, or modification of the geometry graph occurs.
"""
from __future__ import annotations

from collections import deque
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import torch

from modal_gaussians.motion.neural import neural_modes as nm
from modal_gaussians.numpy_io import save_named_arrays

VERSION = 9
METHOD = "neural_fragment_motion_propagation"
STATUS = {0: "not_fragment", 1: "attached", 2: "no_eligible_host",
          3: "ambiguous_host_patch", 4: "insufficient_local_anchors",
          5: "fragment_exceeds_attachment_distance", 6: "insufficient_supported_anchors"}
SEMANTICS = {
    "parent": "immutable_v8_neural_field",
    "assignment": "one_local_geodesic_patch_per_complete_fragment_no_cascades",
    "transfer": "sum_j beta_Cj * (parent_phi_j + cross(parent_R_j, x_i-x_j))",
    "weights": "shared_within_fragment_normalized_wendland_centroid_distance",
    "roles": {"0": "unresolved", "1": "directly_image_supervised",
              "2": "structure_inferred", "3": "fragment_propagated"},
    "observation_view_mask": "original_image_contribution_not_postprocess_role",
    "geometry_graph": "unchanged_parent_graph_attachments_recorded_separately",
    "units": nm.SEMANTICS["units"],
}
MUTABLE = {"phi", "support_class", "sample_prediction"}


@dataclass(frozen=True)
class FragmentPropagationConfig:
    max_fragment_nodes: int = 16
    max_fragment_extent: float = 0.016
    attachment_distance: float = 0.008
    patch_radius: float = 0.008
    core_degree: int = 3
    min_anchors: int = 3
    host_size_ratio: float = 4.0
    ambiguity_ratio: float = 1.25

    def validate(self) -> None:
        for name in ("max_fragment_nodes", "core_degree", "min_anchors"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Fragment {name} must be a positive integer")
        for name in ("max_fragment_extent", "attachment_distance", "patch_radius",
                     "host_size_ratio", "ambiguity_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Fragment {name} must be finite and positive")
        if self.host_size_ratio <= 1 or self.ambiguity_ratio < 1:
            raise ValueError("Fragment host ratio must exceed one and ambiguity ratio must be >=1")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FragmentPropagationConfig:
        result = cls(**value)
        if result.to_dict() != dict(value):
            raise ValueError("Fragment config must contain all resolved settings")
        return result


def core_mask(adjacency: csr_matrix, minimum_degree: int) -> np.ndarray:
    """Iteratively peel low-degree vertices; degree alone is not a k-core."""
    degree = np.diff(adjacency.indptr).copy()
    keep = np.ones(len(degree), dtype=bool)
    queue = deque(np.flatnonzero(degree < minimum_degree).tolist())
    while queue:
        node = queue.popleft()
        if not keep[node]:
            continue
        keep[node] = False
        for neighbor in adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]:
            if keep[neighbor]:
                degree[neighbor] -= 1
                if degree[neighbor] == minimum_degree - 1:
                    queue.append(int(neighbor))
    return keep


def build_attachments(arrays: Mapping[str, np.ndarray], config: FragmentPropagationConfig) -> dict[str, np.ndarray]:
    """Geometry-only deterministic assignment, shared across frequency slots."""
    config.validate()
    points = np.asarray(arrays["g_points"], dtype=np.float64)
    edges = np.asarray(arrays["g_edge_index"], dtype=np.int64)
    component = np.asarray(arrays["g_component_index"], dtype=np.int64)
    sizes = np.asarray(arrays["g_component_size"], dtype=np.int64)
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    adjacency = csr_matrix((np.tile(lengths, 2),
                            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                           shape=(len(points), len(points)))
    core = core_mask(adjacency, config.core_degree)
    members = [np.flatnonzero(component == c) for c in range(len(sizes))]
    extent = np.array([np.linalg.norm(np.ptp(points[m], axis=0)) for m in members])
    fragment = (sizes <= config.max_fragment_nodes) & (extent <= config.max_fragment_extent)
    hosts = np.flatnonzero(core & ~fragment[component])
    tree = cKDTree(points[hosts]) if len(hosts) else None
    status = np.zeros(len(sizes), dtype=np.int8)
    owner = np.full(len(sizes), -1, dtype=np.int64)
    offsets, indices, weights = [0], [], []
    for c, m in enumerate(members):
        anchors, beta = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        if fragment[c]:
            status[c] = 2
            center = points[m].mean(axis=0)
            nearby = (hosts[tree.query_ball_point(center, config.attachment_distance)]
                      if tree is not None else np.empty(0, dtype=np.int64))
            nearby = nearby[sizes[component[nearby]] >= config.host_size_ratio * sizes[c]]
            if len(nearby):
                distance = np.linalg.norm(points[nearby] - center, axis=1)
                order = np.lexsort((nearby, distance))
                nearby, distance = nearby[order], distance[order]
                seed = int(nearby[0])
                geodesic = dijkstra(adjacency, directed=False, indices=seed,
                                    limit=2 * config.patch_radius)
                # Equally close but geodesically separate structures are not mixed.
                alternatives = nearby[distance <= config.ambiguity_ratio * max(distance[0], 1e-12)]
                if np.any(geodesic[alternatives] > 2 * config.patch_radius):
                    status[c] = 3
                else:
                    patch = np.flatnonzero(core & (component == component[seed])
                                           & (geodesic <= config.patch_radius))
                    radial = np.linalg.norm(points[patch] - center, axis=1) / config.attachment_distance
                    patch, radial = patch[radial < 1], radial[radial < 1]
                    status[c] = 4
                    if len(patch) >= config.min_anchors:
                        coverage = cKDTree(points[patch]).query(points[m])[0]
                        status[c] = 5
                        if np.max(coverage) <= config.attachment_distance:
                            beta = (1 - radial) ** 4 * (4 * radial + 1)
                            beta /= beta.sum()
                            anchors = patch
                            status[c] = 1
                            owner[c] = component[seed]
        indices.extend(anchors.tolist())
        weights.extend(beta.tolist())
        offsets.append(len(indices))
    return {
        "f_candidate_component_mask": fragment, "f_component_extent": extent,
        "f_core_mask": core, "f_component_status": status, "f_host_component": owner,
        "f_anchor_indptr": np.array(offsets, dtype=np.int64),
        "f_anchor_indices": np.array(indices, dtype=np.int64),
        "f_anchor_weights": np.array(weights, dtype=np.float64),
    }


def transfer_motion(parent: Mapping[str, np.ndarray], rotation: np.ndarray,
                    attachment: Mapping[str, np.ndarray], config: FragmentPropagationConfig) -> dict[str, np.ndarray]:
    """Copy hosts bitwise; every accepted fragment shares one local motion law."""
    points = np.asarray(parent["g_points"], dtype=np.float64)
    phi = np.asarray(parent["phi"])
    if rotation.shape != phi.shape or rotation.dtype != np.complex64 or not np.isfinite(rotation).all():
        raise ValueError("Fragment parent rotation must be finite complex64 [K,G,3]")
    result, roles = phi.copy(), parent["support_class"].copy()
    status = np.repeat(attachment["f_component_status"][None], len(phi), axis=0)
    mode_weights = np.zeros((len(phi), len(attachment["f_anchor_indices"])), dtype=np.float64)
    propagated = np.zeros(phi.shape[:2], dtype=bool)
    for c in np.flatnonzero(attachment["f_component_status"] == 1):
        members = np.flatnonzero(parent["g_component_index"] == c)
        lo, hi = attachment["f_anchor_indptr"][c:c + 2]
        anchors = attachment["f_anchor_indices"][lo:hi]
        for k in range(len(phi)):
            usable = parent["support_class"][k, anchors] != 0
            if np.sum(usable) < config.min_anchors:
                status[k, c] = 6
                continue
            beta = attachment["f_anchor_weights"][lo:hi] * usable
            beta /= beta.sum()
            live = anchors[usable]
            if np.max(cKDTree(points[live]).query(points[members])[0]) > config.attachment_distance:
                status[k, c] = 5
                continue
            # Always read the frozen parent, never already-propagated fragments.
            omega = rotation[k, anchors].astype(np.complex128)
            offset = points[members, None] - points[anchors][None]
            value = phi[k, anchors][None] + np.cross(omega[None], offset)
            result[k, members] = np.sum(beta[None, :, None] * value, axis=1).astype(np.complex64)
            roles[k, members] = 3
            mode_weights[k, lo:hi] = beta
            propagated[k, members] = True
    if not np.isfinite(result).all():
        raise FloatingPointError("Fragment propagated field is nonfinite")
    return {"phi": result, "support_class": roles, "f_propagated_mask": propagated,
            "f_mode_status": status, "f_mode_anchor_weights": mode_weights,
            "f_parent_rotation": rotation.copy()}


def parent_rotation(parent: nm.NeuralModesArtifact) -> np.ndarray:
    from modal_gaussians.motion.neural.neural_field import evaluate_model
    config = nm.NeuralModesConfig.from_dict(parent.manifest["config"])
    models = torch.load(parent.path / nm.MODELS_FILENAME, map_location="cpu", weights_only=True)
    rotations = []
    with torch.no_grad():
        for k, state in enumerate(models["model_states"]):
            field = evaluate_model(state, nm._field_geometry(parent.arrays, k),
                                   length_scale=float(parent.arrays["scene_scale"]),
                                   amplitude_scale=float(parent.arrays["amplitude_scale"][k]),
                                   config=nm._field_config(config, int(nm._source_mode_slots(parent.manifest)[k])))
            rotations.append(field[1].cpu().numpy().astype(np.complex64))
    return np.stack(rotations)


def render_predictions(parent: nm.NeuralModesArtifact, phi: np.ndarray) -> np.ndarray:
    """Rerender the original fixed sampling operator; do not reuse stale residuals."""
    from modal_gaussians.static import load_static_scene, cameras_from_scene_manifest
    from modal_gaussians.motion.common.projection import projection_jacobian
    if not torch.cuda.is_available():
        raise RuntimeError("Fragment full-foreground rendering requires CUDA")
    scene = load_static_scene(parent.manifest["static_scene"], "cuda")
    scene.eval()
    cameras = {c.label: c for c in cameras_from_scene_manifest(scene.manifest) if c.role == "reference"}
    arrays = parent.arrays
    result = np.zeros_like(arrays["sample_prediction"])
    with torch.no_grad():
        for v, view in enumerate(parent.manifest["views"]):
            camera = cameras[view["label"]].to("cuda")
            if camera.to_manifest_record()["camera_identity"] != view["camera_identity"]:
                raise ValueError("Fragment rendering camera identity differs")
            lo, hi = arrays["view_sample_offsets"][v:v + 2]
            pixels, confidence = arrays["sample_pixels_xy"][lo:hi], arrays["sample_confidence"][lo:hi]
            alpha = scene.render(camera, composition="foreground", outputs=("alpha",))["alpha"].cpu().numpy()
            if not np.allclose(alpha[pixels[:, 1], pixels[:, 0]], confidence, rtol=2e-5, atol=2e-6):
                raise ValueError("Fragment fixed foreground alpha differs")
            jacobian, _ = projection_jacobian(arrays["g_points"], camera.K.cpu().numpy(),
                                               camera.world_to_camera.cpu().numpy(), camera.radial_distortion)
            projector = nm.FrozenModalProjector(scene, camera, torch.as_tensor(jacobian, device="cuda"),
                                                pixels, torch.as_tensor(confidence, device="cuda"))
            for k in range(len(phi)):
                if arrays["alpha_identifiable_mask"][k, v]:
                    predicted = complex(arrays["alphas"][k, v]) * projector(torch.as_tensor(phi[k], device="cuda"))
                    result[k, lo:hi] = predicted.cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Fragment rendered predictions are nonfinite")
    return result


def diagnostics(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    modes = []
    for k in range(len(arrays["phi"])):
        record = nm._mode_diagnostics(arrays, k)
        record["fragment_propagated"] = int(arrays["f_propagated_mask"][k].sum())
        record["components_by_status"] = {name: int(np.sum(arrays["f_mode_status"][k] == code))
                                          for code, name in STATUS.items()}
        modes.append(record)
    return {"candidate_components": int(arrays["f_candidate_component_mask"].sum()),
            "candidate_gaussians": int(arrays["g_component_size"][arrays["f_candidate_component_mask"]].sum()),
            "per_mode": modes}


def identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    # Parent identity binds inherited sources, configuration, networks and graph.
    return {name: manifest[name] for name in (
        "format", "version", "completion_method", "parent_completed_modes_identity",
        "fragment_propagation", "semantics", "arrays_identity", "diagnostics", "quality_gate",
    )}


def load_fragment_modes(path: str | Path) -> nm.NeuralModesArtifact:
    root = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != nm.COMPLETED_MODES_FORMAT or type(manifest.get("version")) is not int
            or manifest["version"] != VERSION or manifest.get("completion_method") != METHOD):
        raise ValueError("Unsupported fragment completed modes")
    config = FragmentPropagationConfig.from_dict(manifest["fragment_propagation"])
    if manifest.get("semantics") != SEMANTICS or manifest.get("quality_gate") != nm.QUALITY_GATE:
        raise ValueError("Fragment semantics differ")
    parent_path = Path(manifest["parent_completed_modes"]).resolve(strict=True)
    if parent_path == root or parent_path.is_relative_to(root) or root.is_relative_to(parent_path):
        raise ValueError("Fragment output and parent paths overlap")
    parent = nm.load_neural_completed_modes(parent_path)
    if parent.manifest["completed_modes_identity"] != manifest["parent_completed_modes_identity"]:
        raise ValueError("Fragment parent identity differs")
    overrides = {"version", "completion_method", "semantics", "completed_modes_identity", "diagnostics",
                 "arrays", "arrays_identity", "arrays_file_sha256", "producer"}
    for name, value in parent.manifest.items():
        if name not in overrides and manifest.get(name) != value:
            raise ValueError(f"Fragment inherited manifest field differs: {name}")
    if manifest.get("arrays_file") != nm.ARRAYS_FILENAME:
        raise ValueError("Fragment arrays filename differs")
    if nm._sha256(root / nm.MODELS_FILENAME) != parent.manifest["networks_sha256"]:
        raise ValueError("Fragment copied parent network checksum differs")
    if nm._sha256(root / nm.ARRAYS_FILENAME) != manifest["arrays_file_sha256"]:
        raise ValueError("Fragment arrays checksum differs")
    with np.load(root / nm.ARRAYS_FILENAME, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if nm._arrays_identity(arrays) != manifest["arrays_identity"]:
        raise ValueError("Fragment arrays identity differs")
    if manifest["arrays"] != {name: {"dtype": a.dtype.name, "shape": list(a.shape)} for name, a in arrays.items()}:
        raise ValueError("Fragment array inventory differs")
    if nm._identity(identity_payload(manifest)) != manifest["completed_modes_identity"]:
        raise ValueError("Fragment completed-mode identity differs")
    attachment = build_attachments(parent.arrays, config)
    transferred = transfer_motion(parent.arrays, parent_rotation(parent), attachment, config)
    expected = {**parent.arrays, **attachment, **transferred}
    if set(arrays) != set(expected):
        raise ValueError("Fragment arrays keys differ")
    for name, value in expected.items():
        if name != "sample_prediction" and (arrays[name].dtype != value.dtype or not np.array_equal(arrays[name], value)):
            raise ValueError(f"Fragment replay differs: {name}")
    prediction = arrays["sample_prediction"]
    if prediction.dtype != np.complex64 or prediction.shape != parent.arrays["sample_prediction"].shape or not np.isfinite(prediction).all():
        raise ValueError("Fragment sampled predictions must be finite complex64 [K,P,2]")
    for k, v in np.argwhere(~arrays["alpha_identifiable_mask"]):
        lo, hi = arrays["view_sample_offsets"][v:v + 2]
        if np.any(prediction[k, lo:hi] != 0):
            raise ValueError("Fragment unidentifiable view prediction must remain zero")
    if manifest["diagnostics"] != diagnostics(arrays):
        raise ValueError("Fragment diagnostics differ")
    return nm.NeuralModesArtifact(root, manifest, arrays)


def build_fragment_modes(*, parent_dir: str | Path, output_dir: str | Path,
                         config: FragmentPropagationConfig | None = None,
                         command: Sequence[str] = ()) -> nm.NeuralModesArtifact:
    config = config or FragmentPropagationConfig()
    config.validate()
    parent = nm.load_neural_completed_modes(parent_dir)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Fragment output already exists: {destination}")
    if destination.is_relative_to(parent.path) or parent.path.is_relative_to(destination):
        raise ValueError("Fragment output and parent must be disjoint")
    attachment = build_attachments(parent.arrays, config)
    transferred = transfer_motion(parent.arrays, parent_rotation(parent), attachment, config)
    arrays = {**parent.arrays, **attachment, **transferred}
    arrays["sample_prediction"] = render_predictions(parent, arrays["phi"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)).resolve()
    try:
        save_named_arrays(temporary / nm.ARRAYS_FILENAME, arrays)
        # Keep the original network file for provenance; parent + transfer, not
        # this network alone, reconstructs v9 phi. It is never trained again.
        shutil.copy2(parent.path / nm.MODELS_FILENAME, temporary / nm.MODELS_FILENAME)
        manifest = {**copy.deepcopy(parent.manifest), "version": VERSION, "completion_method": METHOD,
                    "parent_completed_modes": str(parent.path),
                    "parent_completed_modes_identity": parent.manifest["completed_modes_identity"],
                    "fragment_propagation": config.to_dict(), "semantics": SEMANTICS,
                    "producer": {"created_utc": datetime.now(timezone.utc).isoformat(), "command": list(command)},
                    "diagnostics": diagnostics(arrays),
                    "arrays": {name: {"dtype": a.dtype.name, "shape": list(a.shape)} for name, a in arrays.items()},
                    "arrays_identity": nm._arrays_identity(arrays),
                    "arrays_file_sha256": nm._sha256(temporary / nm.ARRAYS_FILENAME)}
        manifest["completed_modes_identity"] = nm._identity(identity_payload(manifest))
        nm._atomic_json(temporary / "manifest.json", manifest)
        load_fragment_modes(temporary)
        if destination.exists():
            raise FileExistsError(f"Fragment output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists() and temporary.parent == destination.parent.resolve() and temporary.name.startswith(f".{destination.name}."):
            shutil.rmtree(temporary)
        raise
    return load_fragment_modes(destination)
