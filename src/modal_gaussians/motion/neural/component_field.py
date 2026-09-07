"""v16: whole-component control fields, with separate reliable propagation donors."""
from dataclasses import asdict, dataclass, replace
import math

import numpy as np
from scipy.sparse import coo_matrix

from .geometry_graph import build_control_graph
from .pointwise_attachments import assign_points

VERSION = 16
METHOD = "neural_component_field_with_stable_donors"
INPUT_NAMES = {"u_surface_visible"}
MODE_ARRAYS = {"u_own_field_mask", "u_donor_mask", "u_component_observed_mask",
               "u_donor_component_mask", "u_neighbor_index", "u_neighbor_weight",
               "u_motion_component_index"}
ARRAY_NAMES = INPUT_NAMES | MODE_ARRAYS | {"u_control_count", "f_candidate_component_mask",
    "t_host_gaussian_index", "t_interpolation_indptr", "t_interpolation_indices", "t_interpolation_weights"}


@dataclass(frozen=True)
class ComponentFieldConfig:
    strategy: str = "component_field"
    min_learning_nodes: int = 10
    min_learning_controls: int = 2
    min_source_nodes: int = 101
    min_source_controls: int = 2
    neighbors: int = 4
    min_observation_mass: float = 0.05
    min_observed_points: int = 3

    def validate(self):
        if self.strategy != "component_field":
            raise ValueError("Unknown component field strategy")
        for name in ("min_learning_nodes", "min_learning_controls", "min_source_nodes", "min_source_controls", "neighbors", "min_observed_points"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Component field {name} must be a positive integer")
        if self.min_learning_nodes > self.min_source_nodes:
            raise ValueError("Learning size threshold must not exceed donor size threshold")
        value = self.min_observation_mass
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("Donor minimum observation mass must be finite and positive")

    def to_dict(self):
        self.validate()
        value = asdict(self)
        # Original v16 artifacts allowed single-control learning components.
        # Preserve their exact configuration and scientific identity on load.
        if self.min_learning_controls == 1:
            value.pop("min_learning_controls")
        return value

    @classmethod
    def from_dict(cls, value):
        result = cls(**{"min_learning_controls": 1, **value})
        if result.to_dict() != dict(value):
            raise ValueError("Component field configuration must be fully resolved")
        return result


def observation_inputs(points, cameras, depths, alphas, tolerances, alpha_minimum):
    # Use exactly the previous donor visibility test, without storing or applying
    # its observation-direction projector to the learned displacement field.
    from .guarded_attachments import observation_inputs as guarded_inputs
    data = guarded_inputs(points, cameras, depths, alphas, tolerances, alpha_minimum)
    return {"u_surface_visible": data["h_surface_visible"]}


def build_component_controls(graph, *, geometry_config, fragment_config, scene_scale, attachment_inputs):
    from .training_fragments import host_subgraph
    settings = ComponentFieldConfig.from_dict(fragment_config)
    observed = attachment_inputs["observation_view_mask"]
    mass = attachment_inputs["contribution_mass"]
    visible = attachment_inputs["u_surface_visible"]
    G, C = len(graph.points), len(graph.component_size)
    if (observed.dtype != np.bool_ or observed.ndim != 3 or observed.shape[1] != G
            or observed.shape[0] == 0 or observed.shape[2] == 0
            or visible.dtype != np.bool_ or visible.shape != observed.shape[1:]
            or mass.shape != visible.shape or not np.isfinite(mass).all() or np.any(mass < 0)):
        raise ValueError("Component field frozen observation domains/values differ")
    component, sizes = graph.component_index, graph.component_size
    eligible = sizes >= settings.min_learning_nodes
    # Static control ordering is independent of frequency selection and donor
    # thresholds. Entire unsupported components are deactivated per mode below.
    hosts = np.flatnonzero(eligible[component]).astype(np.int64)
    if not len(hosts):
        raise ValueError("Component field has no component meeting min_learning_nodes")
    host_graph = host_subgraph(graph, hosts)
    sampling_config = (replace(geometry_config, max_controls=max(geometry_config.max_controls, len(hosts)))
                       if settings.min_learning_controls > 1 else geometry_config)
    controls = build_control_graph(host_graph, config=sampling_config, scene_scale=scene_scale)
    # Count controls at the configured coverage radius before removing whole
    # components. Rejected controls must not be inputs to the final network.
    control_count = np.bincount(component[hosts[controls.control_point_index]], minlength=C).astype(np.int64)
    if settings.min_learning_controls > 1:
        eligible &= control_count >= settings.min_learning_controls
        hosts = np.flatnonzero(eligible[component]).astype(np.int64)
        if not len(hosts):
            raise ValueError("Component field has no component meeting min_learning_nodes/min_learning_controls")
        controls = build_control_graph(host_subgraph(graph, hosts), config=geometry_config, scene_scale=scene_scale)
    rows = np.repeat(hosts, np.diff(controls.interpolation_indptr))
    N = coo_matrix((controls.interpolation_weights, (rows, controls.interpolation_indices)),
                   shape=(G, len(controls.positions))).tocsr()
    N.sort_indices()
    # Effective observation uses the original renderer contribution threshold and
    # per-mode alpha identifiability. The stronger donor gates never remove a
    # member from its own component field, even inside an occluded region.
    component_observed = np.stack([
        np.bincount(component[direct], minlength=C) > 0 for direct in observed.any(2)])
    own_field = (eligible[None] & component_observed)[:, component]
    reliable = (observed & visible[None] & (mass >= settings.min_observation_mass)[None]).any(2)
    reliable_count = np.stack([np.bincount(component[row], minlength=C) for row in reliable])
    donor_component = (eligible & (sizes >= settings.min_source_nodes)
        & (control_count >= settings.min_source_controls))[None] & (reliable_count >= settings.min_observed_points)
    donors = own_field & donor_component[:, component] & reliable
    indices, weights, owners = [], [], []
    for k in range(len(observed)):
        ids, beta, owner = assign_points(graph.points.astype(np.float64), component, donors[k],
                                        np.flatnonzero(~own_field[k]), settings.neighbors)
        owner[own_field[k]] = component[own_field[k]]
        indices.append(ids); weights.append(beta); owners.append(owner)
    # No donor is not a reason to disable an observed component's own field.
    # Recipients without any eligible donor remain explicitly unresolved/zero.
    return {**{"c_" + k: v for k, v in controls.as_dict().items()},
        "u_surface_visible": visible.copy(), "u_control_count": control_count,
        "u_component_observed_mask": component_observed, "u_donor_component_mask": donor_component,
        "u_own_field_mask": own_field, "u_donor_mask": donors,
        "u_neighbor_index": np.stack(indices), "u_neighbor_weight": np.stack(weights),
        "u_motion_component_index": np.stack(owners),
        "f_candidate_component_mask": ~eligible, "t_host_gaussian_index": hosts,
        "t_interpolation_indptr": N.indptr.astype(np.int64),
        "t_interpolation_indices": N.indices.astype(np.int64),
        "t_interpolation_weights": N.data.astype(np.float64)}


def support_roles(arrays, observed):
    own = arrays["u_own_field_mask"]
    roles = np.zeros(observed.shape[:2], np.int8)
    roles[own] = 2
    roles[own & observed.any(2)] = 1
    roles[~own & (arrays["u_motion_component_index"] >= 0)] = 3
    return roles, roles != 0


def diagnostics(arrays):
    own = arrays["u_own_field_mask"]
    edges = arrays["g_edge_index"]
    return {"controls": len(arrays["c_positions"]),
        "eligible_components": int((~arrays["f_candidate_component_mask"]).sum()),
        "per_mode_own_field_components": (arrays["u_component_observed_mask"]
            & ~arrays["f_candidate_component_mask"][None]).sum(1).tolist(),
        "per_mode_own_field_gaussians": own.sum(1).tolist(),
        "per_mode_donor_gaussians": arrays["u_donor_mask"].sum(1).tolist(),
        "per_mode_donor_components": arrays["u_donor_component_mask"].sum(1).tolist(),
        "per_mode_structure_edges": [int(row[edges].all(1).sum()) for row in own],
        "per_mode_structure_inferred_gaussians": (arrays["support_class"] == 2).sum(1).tolist(),
        "per_mode_propagated_gaussians": (arrays["support_class"] == 3).sum(1).tolist(),
        "per_mode_unresolved_gaussians": (arrays["support_class"] == 0).sum(1).tolist()}
