"""Differentiable local queries on an immutable, frequency-weighted material graph."""
import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from torch.utils.checkpoint import checkpoint
from modal_gaussians.common.progress import Progress, report_progress


def expand_rows(ptr, rows):
    """CSR entries and their local row numbers, including repeated/empty rows."""
    rows = np.asarray(rows, np.int64)
    counts = ptr[rows + 1] - ptr[rows]
    local = np.repeat(np.arange(len(rows)), counts)
    starts = np.repeat(ptr[rows] - np.r_[0, np.cumsum(counts)[:-1]], counts)
    return starts + np.arange(int(counts.sum())), local


def _device_rows(ptr, rows):
    counts = ptr[rows + 1] - ptr[rows]
    local = torch.repeat_interleave(torch.arange(len(rows), device=rows.device), counts)
    starts = ptr[rows] - counts.cumsum(0) + counts
    return starts[local] + torch.arange(len(local), device=rows.device), local


def _segment_sum(values, counts):
    # CSR groups have a fixed order; avoid nondeterministic CUDA index_add atomics.
    if values.is_complex():
        return torch.view_as_complex(torch.segment_reduce(torch.view_as_real(values), "sum",
                                                          lengths=counts, unsafe=True))
    return torch.segment_reduce(values, "sum", lengths=counts, unsafe=True)


def prepare_paths(points, edges, lengths, propagation, controls, radius, *, backend="cupy"):
    """Cache sparse portal/control pairs; scipy is an explicit synthetic-test backend."""
    points, edges = np.asarray(points), np.asarray(edges, np.int64)
    lengths, propagation = np.asarray(lengths, np.float64), np.asarray(propagation, np.float64)
    controls = np.asarray(controls, np.int64)
    n, c = len(points), len(controls)
    if (points.shape != (n, 3) or not n or not c or not np.isfinite(points).all()
            or edges.shape != (len(lengths), 2) or propagation.ndim != 2
            or propagation.shape[1] != len(edges) or not len(propagation)
            or np.any(edges < 0) or np.any(edges >= n) or np.any(lengths <= 0)
            or not np.isfinite(lengths).all() or not np.isfinite(propagation).all()
            or np.any(propagation < lengths) or np.any(controls < 0) or np.any(controls >= n)
            or len(np.unique(controls)) != c or not np.isfinite(radius) or radius <= 0):
        raise ValueError("Invalid immutable reference graph")
    if backend not in ("cupy", "scipy"):
        raise ValueError("Unknown path backend")
    a, b = edges.T
    def adjacency(values):
        return coo_matrix((np.r_[values, values], (np.r_[a,b], np.r_[b,a])), shape=(n,n)).tocsr()
    graph = adjacency(lengths)
    if graph.nnz != 2 * len(edges):
        raise ValueError("Reference graph contains duplicate/self edges")
    portal = adjacency(np.arange(len(edges), dtype=np.int64) + 1)
    portal.setdiag(-1)
    portal.sort_indices()
    connectivity = portal.copy()
    connectivity.data = np.ones(len(connectivity.data), np.int64)
    row, col = [], []
    progress = Progress('Reference: geometric support', c, unit='controls')
    for k, node in enumerate(controls):
        distance = dijkstra(graph, directed=False, indices=int(node), limit=radius)
        ids = np.flatnonzero(distance < radius)
        row.append(ids); col.append(np.full(len(ids), k, np.int64))
        progress.update(k + 1)
    support = coo_matrix((np.ones(sum(map(len,row)), np.int64),
                          (np.concatenate(row), np.concatenate(col))), shape=(n,c)).tocsr()
    candidate = (connectivity @ support).tocsr()
    candidate.sort_indices()
    candidate.data[:] = 1
    # Every portal must have a distance to every candidate, even outside radius.
    needed = (connectivity @ candidate).tocsr()
    needed.sort_indices()
    needed.data = np.arange(needed.nnz, dtype=np.int64)
    by_control = needed.tocsc()
    geometric = np.empty(needed.nnz, np.float64)
    report_progress(f'Reference: {candidate.nnz} root/control candidates, {needed.nnz} node/control path pairs')
    bound = radius + 2 * (float(lengths.max()) if len(lengths) else 0.)
    progress = Progress('Reference: portal/control geometric paths', c, unit='controls')
    for k, node in enumerate(controls):
        lo, hi = by_control.indptr[k:k+2]
        distance = dijkstra(graph, directed=False, indices=int(node), limit=np.nextafter(bound, np.inf))
        geometric[by_control.data[lo:hi]] = distance[by_control.indices[lo:hi]]
        progress.update(k + 1)
    if not np.isfinite(geometric).all():
        raise ValueError("A reference query cannot reach its candidate controls")
    weighted = []
    workspace = None
    try:
        if backend == "cupy":
            from modal_gaussians.motion.control_propagation_gpu import Workspace
            workspace = Workspace()
        progress = Progress('Reference: frequency propagation paths', len(propagation), unit='modes')
        for mode_index, costs in enumerate(propagation):
            if workspace is not None:
                weighted.append(workspace.search(adjacency(costs), controls, by_control,
                    geometric, max(1., float(np.max(costs / lengths))) if len(lengths) else 1.))
            else:
                values = np.empty_like(geometric)
                for k, node in enumerate(controls):
                    lo, hi = by_control.indptr[k:k+2]
                    distance = dijkstra(adjacency(costs), directed=False, indices=int(node))
                    values[by_control.data[lo:hi]] = distance[by_control.indices[lo:hi]]
                weighted.append(values)
            progress.update(mode_index + 1, force=True)
    finally:
        if workspace is not None:
            workspace.close()
    roots = np.repeat(np.arange(n), np.diff(candidate.indptr))
    query_portal, query_candidate = expand_rows(portal.indptr, roots)
    query_node = portal.indices[query_portal]
    path_keys = np.repeat(np.arange(n), np.diff(needed.indptr)) * c + needed.indices
    keys = query_node * c + candidate.indices[query_candidate]
    path_index = np.searchsorted(path_keys, keys)
    if not np.array_equal(path_keys[path_index], keys):
        raise ValueError("Incomplete reference path cache")
    return dict(points=points.astype(np.float32), edges=edges, lengths=lengths,
        propagation=propagation, controls=controls, radius=np.asarray(radius),
        candidate_ptr=candidate.indptr.astype(np.int64), candidate_control=candidate.indices.astype(np.int64),
        query_ptr=np.r_[0, np.cumsum(np.diff(portal.indptr)[roots])].astype(np.int64),
        query_node=query_node.astype(np.int64), query_edge=(portal.data[query_portal]-1).astype(np.int64),
        query_path=path_index.astype(np.int64), geometric=geometric, weighted=np.stack(weighted))


class ReferenceField:
    """Cache frozen sparse tables on the query device; recompute bounded blocks."""
    def __init__(self, arrays, block_size=4096):
        self.a = arrays
        self.block_size = block_size
        if type(block_size) is not int or block_size < 1:
            raise ValueError("Reference query block size must be positive")
        self.mode_count = len(arrays["displacement"])
        self._device = None
        self._tables = {}

    def _on_device(self, device):
        if self._device != device:
            names = ("points", "lengths", "propagation", "controls", "candidate_control",
                     "candidate_ptr", "query_ptr", "query_node", "query_edge", "query_path", "geometric", "weighted",
                     "displacement", "angular", "control_valid")
            self._tables = {name: torch.as_tensor(self.a[name], device=device)
                            for name in names if name in self.a}
            self._device = device
        return self._tables

    def _own(self, x, roots, k, *, support_only=False, stencil=False):
        a = self.a
        t = self._on_device(x.device)
        roots = torch.as_tensor(roots, device=x.device)
        candidates, rows = _device_rows(t["candidate_ptr"], roots)
        queries, qrows = _device_rows(t["query_ptr"], candidates)
        if np.ndim(k):
            mode = torch.as_tensor(k, device=x.device)
            control_mode, path_mode = mode[rows], mode[rows[qrows]]
        else:
            control_mode = path_mode = k
        nodes = t["query_node"][queries]
        # Match the original float64 path/weight normalization before field composition.
        offsets = x[rows[qrows]].double() - t["points"][nodes].double()
        distance = torch.linalg.vector_norm(offsets, dim=-1)
        edges = t["query_edge"][queries]
        stretch = torch.ones(len(edges), device=x.device, dtype=torch.float64)
        valid_edges = edges >= 0
        if len(a["lengths"]):
            safe_edges = edges.clamp_min(0)
            stretch = torch.where(valid_edges, t["propagation"][path_mode, safe_edges] / t["lengths"][safe_edges], stretch)
        path = t["query_path"][queries]
        d0 = distance.new_full((len(candidates),), float("inf")).scatter_reduce(
            0, qrows, distance + t["geometric"][path], reduce="amin", include_self=True)
        dk = distance.new_full((len(candidates),), float("inf")).scatter_reduce(
            0, qrows, distance * stretch + t["weighted"][path_mode,path],
            reduce="amin", include_self=True)
        ratio = d0 / float(a["radius"])
        attenuation = torch.where(dk > 0, d0 / dk.clamp_min(torch.finfo(dk.dtype).tiny), torch.ones_like(dk))
        raw = (1-ratio).clamp_min(0).pow(4) * (1+4*ratio) * attenuation
        controls = t["candidate_control"][candidates]
        if "control_valid" in a:
            raw = raw * t["control_valid"][control_mode, controls].to(x.dtype)
        counts = t["candidate_ptr"][roots+1] - t["candidate_ptr"][roots]
        total = _segment_sum(raw, counts)
        good = total > 0
        if support_only:
            return good
        weight = (raw / total[rows].clamp_min(torch.finfo(raw.dtype).tiny)).to(x.dtype)
        angular = t["angular"][control_mode,controls]
        displacement = t["displacement"][control_mode,controls]
        kind = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        angular, displacement = angular.to(kind), displacement.to(kind)
        lever = (x[rows] - t["points"][t["controls"][controls]].to(x.dtype)).to(kind)
        if stencil:
            return counts, controls, weight, lever.real, good
        phi = _segment_sum(weight[:,None] * (displacement + torch.linalg.cross(angular, lever)), counts)
        omega = _segment_sum(weight[:,None] * angular, counts)
        return phi, omega, good

    def _mode(self, x, roots, k, *, support_only=False):
        a = self.a
        own = a["own"][k,roots]
        donor = a["donor_ids"][k,roots]
        beta = a["donor_weights"][k,roots]
        target, slots = np.nonzero((beta > 0) & ~own[:,None])
        own_rows = np.flatnonzero(own)
        query_roots = np.r_[roots[own_rows], donor[target,slots]]
        output_rows = np.r_[own_rows, target]
        weights = np.r_[np.ones(len(own_rows)), beta[target,slots]]
        row = torch.as_tensor(output_rows, device=x.device, dtype=torch.long)
        origin = torch.as_tensor(a["points"][roots[output_rows]], device=x.device, dtype=x.dtype)
        donor_origin = torch.as_tensor(a["points"][query_roots], device=x.device, dtype=x.dtype)
        # Subtract the root first: canonical recipients must query the exact donor position.
        result = self._own((x[row] - origin) + donor_origin, query_roots,
                          k[output_rows] if np.ndim(k) else k, support_only=support_only)
        if support_only:
            bad = torch.zeros(len(x), device=x.device, dtype=torch.long).index_add(0, row, (~result).long())
            return bad == 0
        phi, omega, good = result
        weight = torch.as_tensor(weights, device=x.device, dtype=x.dtype)[:,None]
        order = torch.as_tensor(np.argsort(output_rows, kind="stable"), device=x.device)
        counts = torch.as_tensor(np.bincount(output_rows, minlength=len(x)), device=x.device)
        result = _segment_sum((weight * phi)[order], counts)
        rotation = _segment_sum((weight * omega)[order], counts)
        bad = torch.zeros(len(x), device=x.device, dtype=torch.long).index_add(0, row, (~good).long())
        return result, rotation, bad == 0

    def mode(self, x, roots, k, *, recompute=False):
        roots = np.asarray(roots, np.int64)
        if x.shape != (len(roots),3) or np.any(roots < 0) or np.any(roots >= len(self.a["points"])):
            raise ValueError("Reference query roots/positions differ")
        parts = []
        for start in range(0, len(x), self.block_size):
            ids = roots[start:start+self.block_size].copy()
            def query(value, ids=ids):
                return self._mode(value, ids, k)
            block = x[start:start+self.block_size]
            parts.append(checkpoint(query, block, use_reentrant=False) if recompute and block.requires_grad
                         else query(block))
        return tuple(torch.cat([part[i] for part in parts]) for i in range(3))
