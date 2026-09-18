"""Induced geometry subgraphs, independent of motion strategy."""
from dataclasses import replace
import numpy as np
from ..neural.geometry_graph import GeometryGraph, _components

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
                   edge_propagation_length=(None if graph.edge_propagation_length is None
                                            else graph.edge_propagation_length[keep]),
                   edge_weight=graph.edge_weight[keep], edge_evidence_kind=graph.edge_evidence_kind[keep],
                   edge_view_evidence=graph.edge_view_evidence[keep],
                   node_visible_view_mask=graph.node_visible_view_mask[host_indices],
                   degree=degree, component_index=component, component_size=sizes,
                   candidate_edge_index=inverse[graph.candidate_edge_index[candidate_keep]],
                   candidate_view_evidence=graph.candidate_view_evidence[candidate_keep])

