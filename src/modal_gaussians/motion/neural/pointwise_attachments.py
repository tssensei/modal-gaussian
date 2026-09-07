"""v12: size-qualified hosts and per-point displacement transfer without cutoffs."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from .geometry_graph import build_control_graph

VERSION = 12
METHOD = "neural_field_with_pointwise_displacement_fill"
MODE_ARRAYS = {"p_observation_view_mask", "p_source_mask", "p_neighbor_index",
               "p_neighbor_weight", "p_motion_component_index"}
ARRAY_NAMES = MODE_ARRAYS | {"f_candidate_component_mask", "t_host_gaussian_index",
    "t_interpolation_indptr", "t_interpolation_indices", "t_interpolation_weights"}


@dataclass(frozen=True)
class PointwiseAttachmentConfig:
    strategy: str = "pointwise"
    min_host_nodes: int = 6
    neighbors: int = 4

    def validate(self):
        if self.strategy != "pointwise":
            raise ValueError("Unknown pointwise attachment strategy")
        for name in ("min_host_nodes", "neighbors"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Pointwise {name} must be a positive integer")

    def to_dict(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        result = cls(**value)
        if result.to_dict() != dict(value):
            raise ValueError("Pointwise configuration must be fully resolved")
        return result


def nearest(points, source_ids, queries, count):
    """Distance then original Gaussian index, including exact boundary ties."""
    tree = cKDTree(points[source_ids])
    k = min(count, len(source_ids))
    probe = min(k + 1, len(source_ids))
    distances, indices = tree.query(queries, k=probe)
    distances = np.asarray(distances).reshape(len(queries), probe)
    indices = np.asarray(indices).reshape(len(queries), probe)
    result = np.empty((len(queries), k), np.int64)
    ordered_distances = np.empty((len(queries), k), np.float64)
    for row in range(len(queries)):
        local = indices[row]
        if probe > k and distances[row, k] == distances[row, k - 1]:
            local = np.asarray(tree.query_ball_point(queries[row], np.nextafter(distances[row, k - 1], np.inf)))
        ids = source_ids[local]
        distance = np.linalg.norm(points[ids] - queries[row], axis=1)
        order = np.lexsort((ids, distance))[:k]
        result[row], ordered_distances[row] = ids[order], distance[order]
    return result, ordered_distances


def assign_points(points, component, sources, targets, neighbors):
    """Choose one nearest source component per point; never cascade followers."""
    ids = np.full((len(points), neighbors), -1, np.int64)
    weights = np.zeros((len(points), neighbors), np.float64)
    owner = np.full(len(points), -1, np.int64)
    source_ids = np.flatnonzero(sources)
    if not len(source_ids) or not len(targets):
        return ids, weights, owner
    seed, _ = nearest(points, source_ids, points[targets], 1)
    owner[targets] = component[seed[:, 0]]
    for group in np.unique(owner[targets]):
        rows = targets[owner[targets] == group]
        anchors = np.flatnonzero(sources & (component == group))
        chosen, distance = nearest(points, anchors, points[rows], neighbors)
        # Scaling inverse-square weights by the nearest distance avoids overflow.
        zero = distance == 0
        raw = np.square(distance[:, :1] / np.maximum(distance, np.finfo(float).tiny))
        raw[zero.any(1)] = zero[zero.any(1)]
        raw /= raw.sum(1, keepdims=True)
        ids[rows, :len(chosen[0])] = chosen
        weights[rows, :len(chosen[0])] = raw
    return ids, weights, owner


def build_pointwise_controls(graph, *, geometry_config, fragment_config, scene_scale, attachment_inputs):
    from .training_fragments import host_subgraph
    config = PointwiseAttachmentConfig.from_dict(fragment_config)
    observed = np.asarray(attachment_inputs["p_observation_view_mask"])
    points, component = graph.points.astype(np.float64), graph.component_index
    if observed.dtype != np.bool_ or observed.ndim != 3 or observed.shape[1] != len(points):
        raise ValueError("Pointwise support must be boolean [K,G,V]")
    eligible_components = graph.component_size >= config.min_host_nodes
    eligible = eligible_components[component]
    hosts = np.flatnonzero(eligible).astype(np.int64)
    if not len(hosts):
        raise ValueError("Pointwise fill has no component meeting min_host_nodes")
    controls = build_control_graph(host_subgraph(graph, hosts), config=geometry_config, scene_scale=scene_scale)
    # Base interpolation only evaluates eligible host points, not followers.
    rows = np.repeat(hosts, np.diff(controls.interpolation_indptr))
    N = coo_matrix((controls.interpolation_weights, (rows, controls.interpolation_indices)),
                  shape=(len(points), len(controls.positions))).tocsr()
    N.sort_indices()
    _, _, initial_owner = assign_points(points, component, eligible, np.flatnonzero(~eligible), 1)
    initial_owner[eligible] = component[eligible]
    source_masks, indices, weights, owners = [], [], [], []
    for direct in observed.any(axis=2):
        # Attached observations can supervise their initial host, as in v11.
        supported_groups = np.unique(initial_owner[direct])
        sources = eligible & np.isin(component, supported_groups[supported_groups >= 0])
        neighbors, beta, owner = assign_points(points, component, sources,
                                               np.flatnonzero(~sources), config.neighbors)
        owner[sources] = component[sources]
        source_masks.append(sources); indices.append(neighbors); weights.append(beta); owners.append(owner)
    return {**{"c_" + k: v for k, v in controls.as_dict().items()},
        "f_candidate_component_mask": ~eligible_components,
        "t_host_gaussian_index": hosts,
        "t_interpolation_indptr": N.indptr.astype(np.int64),
        "t_interpolation_indices": N.indices.astype(np.int64),
        "t_interpolation_weights": N.data.astype(np.float64),
        "p_observation_view_mask": observed.copy(), "p_source_mask": np.stack(source_masks),
        "p_neighbor_index": np.stack(indices), "p_neighbor_weight": np.stack(weights),
        "p_motion_component_index": np.stack(owners)}


def support_roles(arrays, observed):
    if not np.array_equal(observed, arrays["p_observation_view_mask"]):
        raise ValueError("Pointwise attachment observation support differs")
    sources = arrays["p_source_mask"]
    roles = np.zeros(observed.shape[:2], np.int8)
    roles[sources] = 2
    roles[sources & observed.any(axis=2)] = 1
    roles[~sources & (arrays["p_motion_component_index"] >= 0)] = 3
    return roles, roles != 0


def diagnostics(arrays):
    sizes = arrays["g_component_size"]
    return {"eligible_components": int((~arrays["f_candidate_component_mask"]).sum()),
        "eligible_gaussians": len(arrays["t_host_gaussian_index"]),
        "controls": len(arrays["c_positions"]),
        "small_fragment_gaussians": int(sizes[arrays["f_candidate_component_mask"]].sum()),
        "per_mode_independent_gaussians": arrays["p_source_mask"].sum(1).tolist(),
        "per_mode_propagated_gaussians": (arrays["support_class"] == 3).sum(1).tolist(),
        "per_mode_unresolved_gaussians": (arrays["support_class"] == 0).sum(1).tolist()}
