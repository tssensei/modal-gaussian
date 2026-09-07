"""v14: stable donors, observable small-component residuals, and pure followers."""
from dataclasses import asdict, dataclass, replace
import math
import numpy as np
from scipy.sparse import coo_matrix

from .geometry_graph import build_control_graph
from .pointwise_attachments import assign_points

VERSION = 14
METHOD = "neural_field_with_guarded_neighbor_residuals"
INPUT_NAMES = {"h_surface_visible", "h_view_jacobian"}
MODE_ARRAYS = {"h_source_mask", "h_residual_mask", "h_reliable_view_mask", "h_residual_projector",
               "h_neighbor_index", "h_neighbor_weight", "h_motion_component_index", "h_control_supported"}
ARRAY_NAMES = INPUT_NAMES | MODE_ARRAYS | {"h_provisional_control_count", "h_stable_component_mask",
    "h_residual_prior_weight", "f_candidate_component_mask", "t_host_gaussian_index",
    "t_interpolation_indptr", "t_interpolation_indices", "t_interpolation_weights"}


@dataclass(frozen=True)
class GuardedAttachmentConfig:
    strategy: str = "guarded"
    min_residual_nodes: int = 10
    min_source_nodes: int = 101
    min_source_controls: int = 2
    neighbors: int = 4
    min_observation_mass: float = 0.05
    min_observed_points: int = 3
    min_residual_observed_fraction: float = 0.2
    observable_rtol: float = 0.1
    residual_prior_weight: float = 1.0

    def validate(self):
        if self.strategy != "guarded": raise ValueError("Unknown guarded attachment strategy")
        for name in ("min_residual_nodes", "min_source_nodes", "min_source_controls", "neighbors", "min_observed_points"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Guarded {name} must be a positive integer")
        if self.min_residual_nodes >= self.min_source_nodes:
            raise ValueError("Residual size range must precede the source size range")
        for name in ("min_observation_mass", "min_residual_observed_fraction", "observable_rtol", "residual_prior_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Guarded {name} must be finite and positive")
        if self.min_residual_observed_fraction > 1 or self.observable_rtol >= 1:
            raise ValueError("Invalid guarded observation fraction or rank tolerance")

    def to_dict(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        result = cls(**value)
        if result.to_dict() != dict(value): raise ValueError("Guarded configuration must be fully resolved")
        return result


def observation_inputs(points, cameras, depths, alphas, tolerances, alpha_minimum):
    """Freeze center visibility and the same Jacobian used by full-image training."""
    from modal_gaussians.camera_geometry import project_camera
    from modal_gaussians.motion.common.projection import projection_jacobian
    points = np.asarray(points, np.float64)
    visible = np.zeros((len(points), len(cameras)), bool)
    jacobians = []
    for v, camera in enumerate(cameras):
        K, transform = camera.K.cpu().numpy(), camera.world_to_camera.cpu().numpy()
        xyz = points @ transform[:3, :3].T + transform[:3, 3]
        J, projectable = projection_jacobian(points, K, transform, camera.radial_distortion)
        jacobians.append(J)
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = project_camera(xyz, K, camera.radial_distortion)
        valid = projectable & np.isfinite(uv).all(1) & (xyz[:, 2] > 0)
        xy = np.zeros((len(points), 2), np.int64)
        xy[valid] = np.rint(uv[valid]).astype(np.int64)
        valid &= (xy >= 0).all(1) & (xy[:, 0] < depths[v].shape[1]) & (xy[:, 1] < depths[v].shape[0])
        rows = np.flatnonzero(valid)
        depth = depths[v][xy[rows, 1], xy[rows, 0]]
        visible[rows, v] = ((tolerances[v] > 0) & np.isfinite(depth) & (depth > 0)
            & (np.abs(xyz[rows, 2]-depth) <= tolerances[v])
            & (alphas[v][xy[rows, 1], xy[rows, 0]] >= alpha_minimum))
    return {"h_surface_visible": visible, "h_view_jacobian": np.stack(jacobians).astype(np.float32)}


def residual_projectors(jacobians, reliable, mass, allowed, rtol):
    """Retain only sufficiently strong visible directions; null/weak axes stay at the prior."""
    J = np.asarray(jacobians, np.float64)
    normalized = J / np.maximum(np.linalg.norm(J, axis=(2, 3), keepdims=True), 1e-30)
    weight = np.where(reliable, mass, 0).astype(np.float64)
    weight /= np.maximum(weight.max(1, keepdims=True), 1e-30)
    gram = np.einsum('vgai,vgaj,gv->gij', normalized, normalized, weight)
    eigenvalues, vectors = np.linalg.eigh(gram)
    active = (eigenvalues > np.maximum(eigenvalues[:, -1:] * rtol**2, 1e-12)) & allowed[:, None]
    return ((vectors * active[:, None, :]) @ vectors.transpose(0, 2, 1)).astype(np.float32)


def build_guarded_controls(graph, *, geometry_config, fragment_config, scene_scale, attachment_inputs):
    from .training_fragments import host_subgraph
    settings = GuardedAttachmentConfig.from_dict(fragment_config)
    inputs = attachment_inputs
    observed, mass = inputs["observation_view_mask"], inputs["contribution_mass"]
    visible, J = inputs["h_surface_visible"], inputs["h_view_jacobian"]
    G, C = len(graph.points), len(graph.component_size)
    if (observed.dtype != np.bool_ or observed.ndim != 3 or observed.shape[1] != G
            or visible.dtype != np.bool_ or visible.shape != observed.shape[1:]
            or mass.shape != visible.shape or J.shape != (visible.shape[1], G, 2, 3)
            or J.dtype != np.float32 or not np.isfinite(J).all()
            or not np.isfinite(mass).all() or np.any(mass < 0)):
        raise ValueError("Guarded frozen observation domains/values differ")
    component, sizes = graph.component_index, graph.component_size
    reliable = observed & visible[None] & (mass >= settings.min_observation_mass)[None]
    direct = reliable.any(2)
    observed_count = np.stack([np.bincount(component[row], minlength=C) for row in direct])
    stable_size = sizes >= settings.min_source_nodes
    residual_size = (sizes >= settings.min_residual_nodes) & ~stable_size
    stable_observed = stable_size[None] & (observed_count >= settings.min_observed_points)
    residual_observed = (residual_size[None] & (observed_count >= settings.min_observed_points)
        & (observed_count >= settings.min_residual_observed_fraction * sizes[None]))
    # Geometry must remain identical when a saved run selects fewer frequencies.
    # Build the static control domain from all available views; gate learning per mode below.
    static_direct = (visible & (mass >= settings.min_observation_mass)).any(1)
    static_count = np.bincount(component[static_direct], minlength=C)
    static_enough = static_count >= settings.min_observed_points
    static_residual = residual_size & static_enough & (static_count >= settings.min_residual_observed_fraction*sizes)
    candidate_components = (stable_size & static_enough) | static_residual
    provisional_hosts = np.flatnonzero(candidate_components[component]).astype(np.int64)
    if not len(provisional_hosts): raise ValueError("Guarded fill has no sufficiently observed learning component")
    provisional = build_control_graph(host_subgraph(graph, provisional_hosts),
        config=replace(geometry_config, max_controls=max(geometry_config.max_controls, len(provisional_hosts))),
        scene_scale=scene_scale)
    control_counts = np.bincount(component[provisional_hosts[provisional.control_point_index]], minlength=C).astype(np.int64)
    stable_components = stable_size & (control_counts >= settings.min_source_controls)
    stable_observed &= stable_components[None]
    source = stable_observed[:, component] & direct
    available = source.any(1)
    residual = residual_observed[:, component] & direct & available[:, None]
    learning_components = stable_components | static_residual
    hosts = np.flatnonzero(learning_components[component]).astype(np.int64)
    if not len(hosts) or not available.any():
        raise ValueError("Guarded fill has no stable observed source meeting size/control requirements")
    controls = build_control_graph(host_subgraph(graph, hosts), config=geometry_config, scene_scale=scene_scale)
    rows = np.repeat(hosts, np.diff(controls.interpolation_indptr))
    N = coo_matrix((controls.interpolation_weights, (rows, controls.interpolation_indices)),
        shape=(G, len(controls.positions))).tocsr()
    N.sort_indices()
    indices, weights, owners, projectors, control_support = [], [], [], [], []
    for k in range(len(observed)):
        ids, beta, owner = assign_points(graph.points.astype(np.float64), component, source[k],
            np.flatnonzero(~source[k]), settings.neighbors)
        owner[source[k]] = component[source[k]]
        P = residual_projectors(J, reliable[k], mass, residual[k], settings.observable_rtol)
        residual[k] &= np.any(P != 0, axis=(1, 2))
        active_controls = np.zeros(len(controls.positions), bool)
        active_controls[N[source[k] | residual[k]].indices] = True
        indices.append(ids); weights.append(beta); owners.append(owner); projectors.append(P); control_support.append(active_controls)
    return {**{"c_"+k:v for k,v in controls.as_dict().items()},
        "h_surface_visible": visible.copy(), "h_view_jacobian": J.copy(),
        "h_provisional_control_count": control_counts, "h_stable_component_mask": stable_components,
        "h_residual_prior_weight": np.array(settings.residual_prior_weight, np.float64),
        "h_source_mask": source, "h_residual_mask": residual, "h_reliable_view_mask": reliable,
        "h_residual_projector": np.stack(projectors), "h_control_supported": np.stack(control_support),
        "h_neighbor_index": np.stack(indices), "h_neighbor_weight": np.stack(weights),
        "h_motion_component_index": np.stack(owners),
        "f_candidate_component_mask": ~learning_components, "t_host_gaussian_index": hosts,
        "t_interpolation_indptr": N.indptr.astype(np.int64), "t_interpolation_indices": N.indices.astype(np.int64),
        "t_interpolation_weights": N.data.astype(np.float64)}


def support_roles(arrays, observed):
    reliable = arrays["h_reliable_view_mask"]
    if reliable.shape != observed.shape or np.any(reliable & ~observed):
        raise ValueError("Guarded reliable evidence lacks original observation support")
    roles = np.zeros(observed.shape[:2], np.int8)
    roles[arrays["h_motion_component_index"] >= 0] = 3
    roles[arrays["h_source_mask"] | arrays["h_residual_mask"]] = 1
    return roles, roles != 0


def diagnostics(arrays):
    return {"stable_components": int(arrays["h_stable_component_mask"].sum()),
        "controls": len(arrays["c_positions"]),
        "per_mode_donor_gaussians": arrays["h_source_mask"].sum(1).tolist(),
        "per_mode_residual_gaussians": arrays["h_residual_mask"].sum(1).tolist(),
        "per_mode_pure_propagation_gaussians": (arrays["support_class"] == 3).sum(1).tolist(),
        "per_mode_unresolved_gaussians": (arrays["support_class"] == 0).sum(1).tolist()}
