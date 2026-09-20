"""Shared control geometry; frequency-specific weights never resample controls."""
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from modal_gaussians.iteration_cache import Timings, cached, identity, load_entry, module_revision
from modal_gaussians.progress import Progress
from . import component_field, control_propagation, geometry_graph, neural_modes as nm
from .geometry_graph import ControlGraph, GeometryGraph
from ..common.graph_ops import host_subgraph


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
        "min_learning_controls": fragment_config.get("min_learning_controls", 1),
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
    """Old artifacts saved supports, but not their material shortest-path distances."""
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


def shared_geometry(graph, *, geometry_config, fragment_config, scene_scale, cache_dir, timer, seed=None):
    contract = geometry_contract(graph, geometry_config, fragment_config, scene_scale)

    def build():
        source = (component_field.build_component_geometry(replace(graph, edge_propagation_length=None),
            geometry_config=geometry_config, fragment_config=fragment_config, scene_scale=scene_scale)
            if seed is None else seed())
        layout = {name: source[name] for name in LAYOUT_NAMES}
        if "support_distance" in source:
            if source["support_distance"].shape != layout["c_interpolation_indices"].shape:
                raise ValueError("Imported support distances differ from the saved support layout")
            return {**layout, "support_distance": source["support_distance"]}
        host = host_subgraph(graph, layout["t_host_gaussian_index"])
        return {**layout, "support_distance": _geometric_support_distances(host, layout)}

    root = Path(cache_dir) / "control_geometry"
    return cached(root, contract, build, timer, "shared_control_geometry"), contract


def reweight_geometry(graph, shared):
    """Exact existing attenuation on fixed supports; bounded searches retain shortest paths."""
    host = host_subgraph(graph, shared["t_host_gaussian_index"])
    support = _support_by_control(shared)
    distance = shared["support_distance"]
    propagation = host.propagation_lengths()
    attenuation = np.ones_like(distance)
    if not np.array_equal(propagation, host.edge_length):
        adjacency = _adjacency(host, propagation)
        maximum_stretch = float(np.max(propagation / host.edge_length))
        attenuation = control_propagation.attenuation(adjacency, shared["c_control_point_index"],
                                                     support, distance, maximum_stretch)
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


def weighted_geometry(graph, *, geometry_config, fragment_config, scene_scale, cache_dir, timer):
    shared, contract = shared_geometry(graph, geometry_config=geometry_config, fragment_config=fragment_config,
        scene_scale=scene_scale, cache_dir=cache_dir, timer=timer)
    weight_contract = {"implementation": "component_control_weights_v1", "geometry": identity(contract),
        "propagation_code": module_revision(control_propagation),
        "weights": nm._arrays_identity({"edge_weight": graph.edge_weight,
                                        "propagation_length": graph.propagation_lengths()})}
    return cached(Path(cache_dir) / "control_weights", weight_contract,
        lambda: reweight_geometry(graph, shared), timer, "frequency_control_weights")


def prepare_shared_controls(*, prepared_dir, geometry_graph_dir, controls_from, config_path=None):
    """Import a v16 layout or an existing geometry cache, without network replay."""
    from .prepared import load_prepared
    from .iteration import resolve_config
    from .baseline import baseline_overrides
    prepared = load_prepared(prepared_dir)
    overrides = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else baseline_overrides()
    config = resolve_config(prepared.manifest["defaults"], overrides)
    neural = nm.NeuralModesConfig.from_dict(config["neural"])
    if config["fragment"].get("strategy") != "component_field" or neural.graph_edge_filter != "none":
        raise ValueError("Shared preparation requires the unfiltered component-field baseline")
    root = Path(geometry_graph_dir).expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    contract = manifest.get("contract", {})
    if (contract.get("implementation") != "neural_geometry_cache_v1"
            or contract.get("foreground") != prepared.source["foreground_identity"]
            or contract.get("config", {}).get("graph_edge_filter") != "none"
            or contract["config"].get("graph_neighbors") != neural.graph_neighbors
            or contract["config"].get("graph_max_distance") != neural.graph_max_distance
            or root.name != identity(contract)):
        raise ValueError("Expected the matching unfiltered KNN cache; no KNN rebuild is performed")
    graph = GeometryGraph.from_dict(load_entry(root.parent, contract))
    if not np.array_equal(graph.points, prepared.arrays["o_g_points"]):
        raise ValueError("Candidate graph and prepared Gaussian order differ")
    scale = float(prepared.arrays["o_scene_scale"])
    source = Path(controls_from).expanduser().resolve(strict=True)

    def seed():
        saved = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if saved.get("contract", {}).get("implementation") == "component_control_geometry_v1":
            old = saved["contract"]
            expected = geometry_contract(graph, nm._geometry_config(neural), config["fragment"], scale)
            if ({k: v for k, v in old.items() if k != "code"}
                    != {k: v for k, v in expected.items() if k != "code"} or source.name != identity(old)):
                raise ValueError("Imported geometry cache has different geometry/control settings")
            return load_entry(source.parent, old)
        if (saved.get("format") != "modal_gaussians.completed_modes" or saved.get("version") != 16
                or saved.get("arrays_file") != "completed_modes.npz"
                or saved.get("foreground_identity") != prepared.source["foreground_identity"]
                or saved.get("static_scene_identity") != prepared.source["static_scene_identity"]):
            raise ValueError("Control layout source must be a v16 result of the same static scene")
        old = saved["config"]
        if (old.get("training_fragment_config", {}).get("strategy") != "component_field"
                or any(old.get(k) != getattr(neural, k) for k in ("control_radius_fraction", "max_controls"))
                or any(old["training_fragment_config"].get(k, 1) != config["fragment"].get(k, 1)
                       for k in ("min_learning_nodes", "min_learning_controls"))):
            raise ValueError("Saved layout control/host settings differ")
        with np.load(source / saved["arrays_file"], allow_pickle=False) as archive:
            if ("g_edge_propagation_length" in archive.files
                    and saved["geometry_graph"].get("control_sampling_distance")
                    != "original_geometry_graph_shortest_path"):
                raise ValueError("Cannot reuse a layout sampled using frequency-dependent propagation distances")
            for name in ("points", "node_gaussian_index", "edge_index", "edge_length", "component_index", "component_size"):
                if not np.array_equal(archive["g_" + name], getattr(graph, name)):
                    raise ValueError(f"Saved layout has different geometry: {name}")
            layout = {name: archive[name] for name in LAYOUT_NAMES}
        if float(layout["c_scene_scale"]) != scale:
            raise ValueError("Saved layout scene scale differs")
        return layout

    timer = Timings()
    shared, key = shared_geometry(graph, geometry_config=nm._geometry_config(neural),
        fragment_config=config["fragment"], scene_scale=scale, cache_dir=prepared.cache_dir, timer=timer, seed=seed)
    destination = prepared.cache_dir / "control_geometry" / identity(key)
    timer.save(destination / "timings.json")
    return destination, len(shared["c_control_point_index"])
