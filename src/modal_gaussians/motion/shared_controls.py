"""Shared control geometry; frequency-specific weights never resample controls."""
from dataclasses import fields, replace
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from modal_gaussians.common.cache import cached, identity, module_revision
from modal_gaussians.common.progress import Progress
from modal_gaussians.motion import component_field, control_propagation, geometry_graph, training as nm
from modal_gaussians.motion.geometry_graph import ControlGraph
from modal_gaussians.motion.common.graph_ops import host_subgraph


STATIC_NAMES = {"u_control_count", "f_candidate_component_mask", "t_host_gaussian_index"}
WEIGHT_NAMES = {"c_control_edge_weight", "c_interpolation_weights"}
LAYOUT_NAMES = STATIC_NAMES | {"c_" + f.name for f in fields(ControlGraph)} - WEIGHT_NAMES


def geometry_contract(graph, geometry_config, fragment_config, scene_scale):
    """Exclude modal evidence, weights, donor gates and identifiable-view masks."""
    return {"implementation": "component_control_geometry_v1",
        "geometry": nm._arrays_identity({name: getattr(graph, name) for name in
            ("points", "node_gaussian_index", "edge_index", "edge_length", "component_index", "component_size")}),
        "scene_scale": float(scene_scale), "control_radius_fraction": geometry_config.control_radius_fraction,
        "max_controls": geometry_config.max_controls,
        "min_learning_nodes": fragment_config["min_learning_nodes"],
        "min_learning_controls": fragment_config["min_learning_controls"],
        "code": module_revision(geometry_graph, component_field, sys.modules[__name__])}


def _support_by_control(layout):
    rows = np.repeat(np.arange(len(layout["c_owner"])), np.diff(layout["c_interpolation_indptr"]))
    columns = layout["c_interpolation_indices"]
    # Values point back into the original CSR; explicit zero is an index, not an absent entry.
    return coo_matrix((np.arange(len(rows)), (rows, columns)),
        shape=(len(layout["c_owner"]), len(layout["c_control_point_index"])) ).tocsc()


def _adjacency(graph, lengths):
    a, b = graph.edge_index.T
    return coo_matrix((np.concatenate((lengths, lengths)),
        (np.concatenate((a, b)), np.concatenate((b, a)))),
        shape=(len(graph.points), len(graph.points))).tocsr()


def _geometric_support_distances(graph, layout):
    """Compute material shortest-path distances for the fixed control supports."""
    support = _support_by_control(layout)
    adjacency = _adjacency(graph, graph.edge_length)
    result = np.empty(len(support.data), np.float64)
    radius = 2 * float(layout["c_coverage_radius"])
    progress = Progress("shared control geometry distances", len(layout["c_control_point_index"]), unit="controls")
    for slot, node in enumerate(layout["c_control_point_index"]):
        lo, hi = support.indptr[slot:slot + 2]
        distances = dijkstra(adjacency, directed=False, indices=int(node), limit=radius)[support.indices[lo:hi]]
        if not np.isfinite(distances).all() or np.any(distances >= radius):
            raise ValueError("Saved control supports do not match the geometric radius/topology")
        result[support.data[lo:hi]] = distances
        progress.update(slot + 1)
    return result


def shared_geometry(graph, *, geometry_config, fragment_config, scene_scale, cache_dir, timer):
    contract = geometry_contract(graph, geometry_config, fragment_config, scene_scale)

    def build():
        source = component_field.build_component_geometry(replace(graph, edge_propagation_length=None),
            geometry_config=geometry_config, fragment_config=fragment_config, scene_scale=scene_scale)
        layout = {name: source[name] for name in LAYOUT_NAMES}
        host = host_subgraph(graph, layout["t_host_gaussian_index"])
        return {**layout, "support_distance": _geometric_support_distances(host, layout)}

    root = Path(cache_dir) / "control_geometry"
    return cached(root, contract, build, timer, "shared_control_geometry"), contract


def reweight_geometry(graph, shared, *, backend="cupy", workspace=None, propagated=None):
    """Exact existing attenuation on fixed supports; bounded searches retain shortest paths."""
    host = host_subgraph(graph, shared["t_host_gaussian_index"])
    support = _support_by_control(shared)
    distance = shared["support_distance"]
    propagation = host.propagation_lengths()
    attenuation = np.ones_like(distance)
    if not np.array_equal(propagation, host.edge_length):
        adjacency = _adjacency(host, propagation)
        maximum_stretch = float(np.max(propagation / host.edge_length))
        attenuation = (control_propagation.attenuation(adjacency, shared["c_control_point_index"],
            support, distance, maximum_stretch, backend=backend, workspace=workspace) if propagated is None
            else np.divide(distance, propagated, out=np.ones_like(distance), where=propagated > 0))
    ratio = distance / (2 * float(shared["c_coverage_radius"]))
    weights = (1 - ratio) ** 4 * (4 * ratio + 1) * attenuation
    interpolation = csr_matrix((weights, shared["c_interpolation_indices"], shared["c_interpolation_indptr"]),
        shape=(len(host.points), len(shared["c_control_point_index"])))
    totals = np.asarray(interpolation.sum(axis=1)).ravel()
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("Shared controls do not cover all hosts")
    weights /= np.repeat(totals, np.diff(shared["c_interpolation_indptr"]))

    owners = np.sort(shared["c_owner"][host.edge_index], axis=1)
    crossing = owners[:, 0] != owners[:, 1]
    edges = shared["c_control_edges"]
    count = len(shared["c_control_point_index"])
    keys = edges[:, 0] * count + edges[:, 1]
    crossing_keys = owners[crossing, 0] * count + owners[crossing, 1]
    slots = np.searchsorted(keys, crossing_keys)
    if np.any(slots >= len(keys)) or not np.array_equal(keys[slots], crossing_keys):
        raise ValueError("Saved control adjacency differs from the current topology")
    if not np.isfinite(host.edge_weight).all() or np.any(host.edge_weight < 0):
        raise ValueError("Control edge weights must be finite and nonnegative")
    control_weights = np.bincount(slots, weights=host.edge_weight[crossing], minlength=len(edges)).astype(np.float64)
    return {**{name: shared[name] for name in LAYOUT_NAMES},
        "c_interpolation_weights": weights, "c_control_edge_weight": control_weights}


def weight_contract(graph, geometry, backend="cupy"):
    return {"implementation": "component_control_weights_v1", "geometry": identity(geometry),
        "propagation_code": module_revision(control_propagation),
        "propagation": control_propagation.backend_identity(backend),
        "weights": nm._arrays_identity({"edge_weight": graph.edge_weight,
                                        "propagation_length": graph.propagation_lengths()})}


def weighted_geometry(graph, *, geometry_config, fragment_config, scene_scale, cache_dir, timer,
                      backend="cupy", workspace=None):
    shared, contract = shared_geometry(graph, geometry_config=geometry_config, fragment_config=fragment_config,
        scene_scale=scene_scale, cache_dir=cache_dir, timer=timer)
    return cached(Path(cache_dir) / "control_weights", weight_contract(graph, contract, backend),
        lambda: reweight_geometry(graph, shared, backend=backend, workspace=workspace), timer, "frequency_control_weights")
