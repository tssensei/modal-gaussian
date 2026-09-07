"""Fixed fragment attachment compiled into differentiable control interpolation.

Only host components receive controls. For an attached fragment, N_i = sum_j
beta_j N_j exactly implements sum_j beta_j (Phi_j + R_j cross (x_i-x_j)).
The full Gaussian renderer and its fixed alpha remain unchanged.
"""
from __future__ import annotations

from modal_gaussians.motion.common.graph_ops import host_subgraph
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from modal_gaussians.motion.neural.geometry_graph import build_control_graph
from modal_gaussians.motion.neural import strategies
from modal_gaussians.motion.neural.strategies import config_class, artifact_contract, array_names, validate_training_controls
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


def build_training_controls(graph, *, geometry_config, fragment_config, scene_scale, attachment_inputs=None):
    if fragment_config.get("strategy") is not None:
        return strategies.build_training_controls(graph, geometry_config=geometry_config,
            fragment_config=fragment_config, scene_scale=scene_scale, attachment_inputs=attachment_inputs)
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
    if any(key in arrays for key in ("u_own_field_mask", "h_source_mask", "p_source_mask")):
        return strategies.training_support_roles(arrays, observed)
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


def diagnostics(arrays):
    if any(key in arrays for key in ("u_own_field_mask", "h_source_mask", "p_source_mask", "a_point_status")):
        return strategies.diagnostics(arrays)
    candidate = arrays["f_candidate_component_mask"]
    attached = arrays["f_component_status"] == 1
    return {"host_gaussians": len(arrays["t_host_gaussian_index"]),
            "host_controls": len(arrays["c_positions"]),
            "fragment_components": int(candidate.sum()),
            "attached_fragment_components": int(attached.sum()),
            "unattached_fragment_components": int((candidate & ~attached).sum()),
            "per_mode_propagated_gaussians": (arrays["support_class"] == 3).sum(axis=1).tolist()}
