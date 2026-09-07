"""Deterministic, non-cascading pointwise transfer from reliable source points."""
import numpy as np
from scipy.spatial import cKDTree

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

