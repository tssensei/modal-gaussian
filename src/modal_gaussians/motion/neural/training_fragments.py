"""Fixed fragment attachment compiled into differentiable control interpolation.

Only host components receive controls. For an attached fragment, N_i = sum_j
beta_j N_j exactly implements sum_j beta_j (Phi_j + R_j cross (x_i-x_j)).
The full Gaussian renderer and its fixed alpha remain unchanged.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from .geometry_graph import GeometryGraph, ControlGraph, build_control_graph, _components
from .fragment_propagation import FragmentPropagationConfig, build_attachments

VERSION = 10
METHOD = "neural_field_with_training_fragment_fill"
ATTACHMENT_NAMES = {
    "f_candidate_component_mask", "f_component_extent", "f_core_mask",
    "f_component_status", "f_host_component", "f_anchor_indptr",
    "f_anchor_indices", "f_anchor_weights",
}
ARRAY_NAMES = ATTACHMENT_NAMES | {
    "t_host_gaussian_index", "t_motion_component_index", "t_interpolation_indptr",
    "t_interpolation_indices", "t_interpolation_weights",
}


def host_subgraph(graph: GeometryGraph, host_indices: np.ndarray) -> GeometryGraph:
    """An induced union of complete host components, with local index domains."""
    inverse = np.full(len(graph.points), -1, dtype=np.int64)
    inverse[host_indices] = np.arange(len(host_indices))
    keep = np.all(inverse[graph.edge_index] >= 0, axis=1)
    candidate_keep = np.all(inverse[graph.candidate_edge_index] >= 0, axis=1)
    edges = inverse[graph.edge_index[keep]]
    degree, component, sizes = _components(len(host_indices), edges)
    return replace(graph, points=graph.points[host_indices],
                   node_gaussian_index=np.arange(len(host_indices), dtype=np.int64),
                   edge_index=edges, edge_length=graph.edge_length[keep],
                   edge_weight=graph.edge_weight[keep], edge_evidence_kind=graph.edge_evidence_kind[keep],
                   edge_view_evidence=graph.edge_view_evidence[keep],
                   node_visible_view_mask=graph.node_visible_view_mask[host_indices],
                   degree=degree, component_index=component, component_size=sizes,
                   candidate_edge_index=inverse[graph.candidate_edge_index[candidate_keep]],
                   candidate_view_evidence=graph.candidate_view_evidence[candidate_keep])


def config_class(value):
    if not isinstance(value, dict):
        raise ValueError("Training fragment configuration must be an object")
    if value.get("strategy") == "component_field":
        from .component_field import ComponentFieldConfig
        return ComponentFieldConfig
    if value.get("strategy") == "guarded":
        from .guarded_attachments import GuardedAttachmentConfig
        return GuardedAttachmentConfig
    if value.get("strategy") == "surface":
        from .surface_attachments import SurfaceAttachmentConfig
        return SurfaceAttachmentConfig
    if value.get("strategy") == "pointwise":
        from .pointwise_attachments import PointwiseAttachmentConfig
        return PointwiseAttachmentConfig
    if "strategy" in value:
        raise ValueError("Unknown training fragment strategy")
    return FragmentPropagationConfig


def artifact_contract(value):
    if value is None:
        return 8, "neural_complex_displacement_field"
    if value.get("strategy") == "component_field":
        from .component_field import VERSION as version, METHOD as method
        return version, method
    if value.get("strategy") == "guarded":
        from .guarded_attachments import VERSION as version, METHOD as method
        return version, method
    if value.get("strategy") == "surface":
        from .surface_attachments import VERSION as version, METHOD as method
        return version, method
    if value.get("strategy") == "pointwise":
        from .pointwise_attachments import VERSION as version, METHOD as method
        return version, method
    return VERSION, METHOD


def array_names(value):
    if value.get("strategy") == "component_field":
        from .component_field import ARRAY_NAMES as names
        return names
    if value.get("strategy") == "guarded":
        from .guarded_attachments import ARRAY_NAMES as names
        return names
    if value.get("strategy") == "pointwise":
        from .pointwise_attachments import ARRAY_NAMES as names
        return names
    if value.get("strategy") == "surface":
        from .surface_attachments import ARRAY_NAMES as names
        return names
    return ARRAY_NAMES


def build_training_controls(graph, *, geometry_config, fragment_config, scene_scale, attachment_inputs=None):
    if fragment_config.get("strategy") == "component_field":
        from .component_field import build_component_controls
        return build_component_controls(graph, geometry_config=geometry_config, fragment_config=fragment_config,
                                        scene_scale=scene_scale, attachment_inputs=attachment_inputs)
    if fragment_config.get("strategy") == "guarded":
        from .guarded_attachments import build_guarded_controls
        return build_guarded_controls(graph, geometry_config=geometry_config, fragment_config=fragment_config,
                                      scene_scale=scene_scale, attachment_inputs=attachment_inputs)
    if fragment_config.get("strategy") == "pointwise":
        from .pointwise_attachments import build_pointwise_controls
        return build_pointwise_controls(graph, geometry_config=geometry_config, fragment_config=fragment_config,
                                        scene_scale=scene_scale, attachment_inputs=attachment_inputs)
    if fragment_config.get("strategy") == "surface":
        from .surface_attachments import build_surface_controls
        return build_surface_controls(graph, geometry_config=geometry_config, fragment_config=fragment_config,
                                      scene_scale=scene_scale, attachment_inputs=attachment_inputs)
    settings = FragmentPropagationConfig.from_dict(fragment_config)
    attachment = build_attachments({"g_" + k: v for k, v in graph.as_dict().items()}, settings)
    fragment = attachment["f_candidate_component_mask"][graph.component_index]
    hosts = np.flatnonzero(~fragment).astype(np.int64)
    if not len(hosts):
        raise ValueError("Fragment training has no host component; all components are small fragments")
    host_graph = host_subgraph(graph, hosts)
    controls = build_control_graph(host_graph, config=geometry_config, scene_scale=scene_scale)
    host_N = csr_matrix((controls.interpolation_weights, controls.interpolation_indices,
                         controls.interpolation_indptr), shape=(len(hosts), len(controls.positions)))
    inverse = np.full(len(graph.points), -1, dtype=np.int64)
    inverse[hosts] = np.arange(len(hosts))
    # Component-to-host anchor map. The assignment uses static geometry only.
    offsets = attachment["f_anchor_indptr"]
    anchor_indices = attachment["f_anchor_indices"]
    anchors = csr_matrix((attachment["f_anchor_weights"], inverse[anchor_indices], offsets),
                         shape=(len(graph.component_size), len(hosts)))
    if len(anchor_indices) and np.any(inverse[anchor_indices] < 0):
        raise ValueError("Fragment attachment must reference host Gaussians only")
    fragment_N = anchors @ host_N
    final_N = fragment_N[graph.component_index].tocoo()
    local_N = host_N.tocoo()
    effective = coo_matrix((np.r_[final_N.data, local_N.data],
                            (np.r_[final_N.row, hosts[local_N.row]],
                             np.r_[final_N.col, local_N.col])),
                           shape=(len(graph.points), len(controls.positions))).tocsr()
    effective.sum_duplicates()
    effective.eliminate_zeros()
    effective.sort_indices()
    motion_component = graph.component_index.copy()
    motion_component[fragment] = attachment["f_host_component"][graph.component_index[fragment]]
    return {**{"c_" + k: v for k, v in controls.as_dict().items()}, **attachment,
            "t_host_gaussian_index": hosts,
            "t_motion_component_index": motion_component,
            "t_interpolation_indptr": effective.indptr.astype(np.int64),
            "t_interpolation_indices": effective.indices.astype(np.int64),
            "t_interpolation_weights": effective.data.astype(np.float64)}


def training_support_roles(arrays, observed):
    """A host and its attached fragments share support, including fragment images."""
    if "u_own_field_mask" in arrays:
        from .component_field import support_roles
        return support_roles(arrays, observed)
    if "h_source_mask" in arrays:
        from .guarded_attachments import support_roles
        return support_roles(arrays, observed)
    if "p_source_mask" in arrays:
        from .pointwise_attachments import support_roles
        return support_roles(arrays, observed)
    groups = arrays["t_motion_component_index"]
    valid = groups >= 0
    fragment = arrays["f_candidate_component_mask"][arrays["g_component_index"]]
    roles = np.zeros(observed.shape[:2], dtype=np.int8)
    for mode in range(len(observed)):
        direct = observed[mode].any(axis=1) & valid
        supported = valid & np.isin(groups, np.unique(groups[direct]))
        roles[mode, supported] = 2
        roles[mode, direct & supported] = 1
        roles[mode, fragment & supported] = 3
    return roles, roles != 0


def validate_training_controls(arrays, graph, *, geometry_config, fragment_config, scene_scale):
    """Replay the fixed assignment/interpolation, including unresolved empty rows."""
    from .surface_attachments import INPUT_NAMES
    inputs = {k: arrays[k] for k in INPUT_NAMES} if fragment_config.get("strategy") == "surface" else None
    if fragment_config.get("strategy") == "pointwise":
        inputs = {"p_observation_view_mask": arrays["observation_view_mask"]}
    if fragment_config.get("strategy") == "guarded":
        from .guarded_attachments import INPUT_NAMES as guarded_inputs
        inputs = {k: arrays[k] for k in guarded_inputs | {"observation_view_mask", "contribution_mass"}}
    if fragment_config.get("strategy") == "component_field":
        from .component_field import INPUT_NAMES as component_inputs
        inputs = {k: arrays[k] for k in component_inputs | {"observation_view_mask", "contribution_mass"}}
    expected = build_training_controls(graph, geometry_config=geometry_config,
                                      fragment_config=fragment_config, scene_scale=scene_scale, attachment_inputs=inputs)
    for name, reference in expected.items():
        value = arrays[name]
        if value.dtype != reference.dtype or not np.array_equal(value, reference):
            raise ValueError(f"Training fragment geometry/attachment differs: {name}")
    ControlGraph.from_dict({k[2:]: v for k, v in expected.items() if k.startswith("c_")})


def diagnostics(arrays):
    if "u_own_field_mask" in arrays:
        from .component_field import diagnostics as component_diagnostics
        return component_diagnostics(arrays)
    if "h_source_mask" in arrays:
        from .guarded_attachments import diagnostics as guarded_diagnostics
        return guarded_diagnostics(arrays)
    if "p_source_mask" in arrays:
        from .pointwise_attachments import diagnostics as pointwise_diagnostics
        return pointwise_diagnostics(arrays)
    if "a_point_status" in arrays:
        from .surface_attachments import diagnostics as surface_diagnostics
        return surface_diagnostics(arrays)
    candidate = arrays["f_candidate_component_mask"]
    attached = arrays["f_component_status"] == 1
    return {"host_gaussians": len(arrays["t_host_gaussian_index"]),
            "host_controls": len(arrays["c_positions"]),
            "fragment_components": int(candidate.sum()),
            "attached_fragment_components": int(attached.sum()),
            "unattached_fragment_components": int((candidate & ~attached).sum()),
            "per_mode_propagated_gaussians": (arrays["support_class"] == 3).sum(axis=1).tolist()}
