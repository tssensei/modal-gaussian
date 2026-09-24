"""Double-precision alpha geometry, residuals and bounded TRF on CUDA.

CuPy is lazy: importing artifact readers never initializes a CUDA context.
Only geometry cache publication and final small diagnostics leave the device.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import hashlib
import time

import numpy as np

from modal_gaussians.motion.observations import alpha as sync
from modal_gaussians.common.cache import identity, load_entry, put_entry


@dataclass
class Constraints:
    information: object
    column_energy: object
    rows: object
    blocks: object
    equation_count: object
    gram: object
    rhs: object
    normalization: object
    jacobian: object
    observation: object
    views: object

    def __len__(self):
        return len(self.information)


class Workspace:
    def __init__(self, *, max_blocks=None, reserve_bytes=2 * 1024**3):
        started = time.perf_counter()
        import cupy as cp
        self.cp = cp
        if max_blocks is not None and (type(max_blocks) is not int or max_blocks < 1):
            raise ValueError("Alpha block limit must be positive")
        if reserve_bytes < 0:
            raise ValueError("Alpha memory reservation must be nonnegative")
        self.max_blocks, self.reserve_bytes = max_blocks, reserve_bytes
        self.key, self.base, self.base_device = None, None, None
        self.groups, self.device_groups = {}, {}
        self.cache_dir = None
        self.stats = {}
        self.revision = sync.backend_identity("cupy")
        cp.cuda.get_current_stream().synchronize()
        self.initialization_seconds = time.perf_counter() - started

    @contextmanager
    def timed(self, name):
        self.cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.cp.cuda.get_current_stream().synchronize()
            self.stats[name] = self.stats.get(name, 0.) + time.perf_counter() - started
            self.stats["peak_pool_bytes"] = max(self.stats.get("peak_pool_bytes", 0),
                                                self.cp.get_default_memory_pool().total_bytes())
            free, total = self.cp.cuda.runtime.memGetInfo()
            self.stats["peak_device_used_bytes"] = max(self.stats.get("peak_device_used_bytes", 0), total-free)

    def close(self):
        self.cp.cuda.get_current_stream().synchronize()
        self.key = self.base = self.base_device = None
        self.groups.clear()
        self.device_groups.clear()
        self.cp.get_default_memory_pool().free_all_blocks()

    def _budget(self, required=0):
        # Reclaim only unused blocks; live alpha arrays are never evicted mid-solve.
        free, _ = self.cp.cuda.runtime.memGetInfo()
        available = free + self.cp.get_default_memory_pool().free_bytes() - self.reserve_bytes - required
        if available <= 0:
            raise MemoryError("Alpha GPU workspace cannot preserve its reserved memory")
        return available

    def _chunks(self, count, bytes_per_block, operation, *, required=0):
        if count == 0:
            return
        size = min(count, self._budget(required) // max(1, bytes_per_block))
        if size < 1:
            raise MemoryError("One alpha block exceeds the reserved-memory budget")
        if self.max_blocks is not None:
            size = min(size, self.max_blocks)
        start = 0
        while start < count:
            end = min(count, start + size)
            try:
                value = operation(start, end)
            except self.cp.cuda.memory.OutOfMemoryError:
                if size == 1:
                    raise MemoryError("One alpha block does not fit on the GPU") from None
                size = max(1, size // 2)
                self.cp.get_default_memory_pool().free_all_blocks()
                self.stats["allocation_retries"] = self.stats.get("allocation_retries", 0) + 1
                continue
            yield start, end, value
            start = end

    def _cache(self, contract, build):
        root = None if self.cache_dir is None else Path(self.cache_dir) / "alpha_geometry"
        arrays = None if root is None else load_entry(root, contract)
        self.stats["geometry_cache_hits"] = self.stats.get("geometry_cache_hits", 0) + int(arrays is not None)
        if arrays is None:
            arrays = build()
            if root is not None:
                with self.timed("geometry_publish_seconds"):
                    arrays = put_entry(root, contract, arrays)
        return arrays

    def prepare(self, points, topology, measurements, labels, *, topology_identity, cache_dir):
        if topology_identity is None:
            raise ValueError("Cached alpha preparation requires the original topology identity")
        contract = {"kind": "alpha_observation_geometry_v1", "topology": topology_identity,
                    "views": list(labels), "point_count": len(points), "weights": "contribution_per_point_view_count",
                    "precision": "float64_complex128", "implementation": self.revision}
        key = identity(contract)
        if key != self.key:
            self.close()
            self.key, self.cache_dir = key, cache_dir
            with self.timed("geometry_load_build_seconds"):
                def build():
                    prepared = sync.prepare_observations(points=points, topology=topology,
                        sample_measurements=measurements, view_labels=labels)
                    counts = np.array([len(r) for r in prepared.rows_by_point], np.int64)
                    return dict(sample_index=np.repeat(np.arange(len(topology.sample_view_index)),
                                                        np.diff(topology.sample_offsets)),
                        point_index=prepared.obs_point_index, view_index=prepared.obs_view_index,
                        weights=prepared.obs_weights, jacobian=prepared.obs_jacobian,
                        sorted_rows=np.concatenate(prepared.rows_by_point) if len(points) else np.empty(0, np.int64),
                        offsets=np.r_[0, np.cumsum(counts)], raw_counts=sync._raw_shared_counts(prepared))
                self.base = self._cache(contract, build)
        else:
            self.stats["resident_geometry_hits"] = self.stats.get("resident_geometry_hits", 0) + 1
        a = self.base
        rows = tuple(a["sorted_rows"][lo:hi] for lo, hi in zip(a["offsets"][:-1], a["offsets"][1:]))
        return sync.PreparedObservations(points, a["point_index"], a["view_index"],
            measurements[a["sample_index"]], a["jacobian"], a["weights"], rows, tuple(labels), key)

    def _bind(self, prepared):
        if prepared.geometry_key is not None and prepared.geometry_key == self.key:
            return
        # Direct/synthetic callers have no artifact identity. Bind to actual geometry.
        digest = hashlib.sha256()
        for a in (prepared.obs_point_index, prepared.obs_view_index, prepared.obs_jacobian, prepared.obs_weights):
            a = np.ascontiguousarray(a)
            digest.update(str((a.shape, a.dtype)).encode())
            digest.update(a.tobytes())
        key = identity({"arrays": digest.hexdigest(), "views": prepared.view_labels, "points": len(prepared.points)})
        if key != self.key:
            self.close()
            self.key, self.cache_dir = key, None
            self.base = {"raw_counts": sync._raw_shared_counts(prepared)}

    def _geometry(self, prepared, allowed):
        subset = tuple(np.flatnonzero(allowed).tolist())
        if subset in self.groups:
            return self.groups[subset]
        cp = self.cp
        contract = {"kind": "alpha_nullspace_v1", "geometry": self.key, "views": list(subset),
                    "rank_rtol": 1e-8, "zero_atol": sync.EPSILON, "implementation": self.revision}
        def build():
            groups, result = {}, {}
            for point, point_rows in enumerate(prepared.rows_by_point):
                rows = point_rows[allowed[prepared.obs_view_index[point_rows]] & (prepared.obs_weights[point_rows] > 0)]
                if len(np.unique(prepared.obs_view_index[rows])) >= 2:
                    groups.setdefault(len(rows), []).append((point, rows))
            number = 0
            for width, records in sorted(groups.items()):
                rows = np.stack([r for _, r in records])
                points = np.array([p for p, _ in records])
                def decompose(lo, hi):
                    r = rows[lo:hi]
                    weights = cp.asarray(prepared.obs_weights[r])
                    weighted = cp.sqrt(weights)[..., None, None] * cp.asarray(prepared.obs_jacobian[r], cp.float64)
                    with __import__('cupyx').errstate(linalg='raise'):
                        u, s, _ = cp.linalg.svd(weighted.reshape(-1, 2*width, 3), full_matrices=True)
                    ranks = cp.sum(s > 1e-8*s[:, :1], axis=1)
                    ranks[s[:, 0] <= sync.EPSILON] = -1
                    view = cp.asarray(prepared.obs_view_index[r])
                    gram = cp.einsum('brki,brkj->brij', weighted, weighted)
                    by_view = cp.zeros((hi-lo, prepared.num_views, 3, 3), cp.float64)
                    for v in subset:
                        by_view[:, v] = cp.sum(gram * (view == v)[..., None, None], axis=1)
                    # Cache only the nullspace columns, packed by rank below.
                    return u, ranks.get(), weighted, by_view
                for lo, hi, (u, ranks, weighted, gram) in self._chunks(len(rows), 8*(12*width*width+100*width), decompose):
                    for rank in np.unique(ranks):
                        if rank < 0:
                            continue
                        mask = np.flatnonzero(ranks == rank)
                        tag = f"g{number}_"
                        result.update({tag+"rows": rows[lo:hi][mask], tag+"points": points[lo:hi][mask],
                                       tag+"null": u[mask, :, int(rank):].get(),
                                       tag+"weighted": weighted[mask].get(), tag+"gram": gram[mask].get()})
                        number += 1
                    del u, weighted, gram
            result["group_count"] = np.array(number, np.int64)
            return result
        with self.timed("geometry_load_build_seconds"):
            self.groups[subset] = self._cache(contract, build)
        return self.groups[subset]

    def constraints(self, prepared, allowed=None):
        cp = self.cp
        self._bind(prepared)
        allowed = np.ones(prepared.num_views, bool) if allowed is None else allowed
        subset = tuple(np.flatnonzero(allowed).tolist())
        arrays = self._geometry(prepared, allowed)
        with self.timed("upload_seconds"):
            if self.base_device is None:
                self._budget(prepared.obs_jacobian.size*8)
                self.base_device = cp.asarray(prepared.obs_jacobian, cp.float64)
            if subset not in self.device_groups:
                self._budget(sum(a.nbytes for a in arrays.values()))
                self.device_groups[subset] = {k: cp.asarray(v) for k, v in arrays.items() if k != "group_count"}
            geo = self.device_groups[subset]
            y = cp.asarray(prepared.obs_y, cp.complex128)
            view = cp.asarray(prepared.obs_view_index)
            sqrt_w = cp.sqrt(cp.asarray(prepared.obs_weights))
        with self.timed("constraint_seconds"):
            hs, energies, pts, grams, rows_list = [], [], [], [], []
            for i in range(int(arrays["group_count"])):
                tag = f"g{i}_"
                all_rows, null = geo[tag+"rows"], geo[tag+"null"]
                width = all_rows.shape[1]
                def make(lo, hi):
                    rows = all_rows[lo:hi]
                    obs = sqrt_w[rows][..., None] * y[rows]
                    b = cp.zeros((hi-lo, width, 2, prepared.num_views), cp.complex128)
                    for v in subset:
                        b[..., v] = obs * (view[rows] == v)[..., None]
                    c = null[lo:hi].swapaxes(1, 2) @ b.reshape(hi-lo, 2*width, prepared.num_views)
                    h = c.conj().swapaxes(1, 2) @ c
                    energy = cp.sum(cp.abs(obs)**2, axis=(1, 2))
                    keep = cp.sqrt(cp.sum(cp.abs(c)**2, axis=(1, 2))) > sync.EPSILON
                    return h[keep], energy[keep], geo[tag+"points"][lo:hi][keep], geo[tag+"gram"][lo:hi][keep], rows[keep]
                for _, _, values in self._chunks(len(all_rows), 16*(8*width*prepared.num_views+40), make):
                    h, energy, points, gram, rows = values
                    hs.append(h); energies.append(energy); pts.append(points); grams.append(gram)
                    rows_list.append(rows)
            if not hs or sum(len(h) for h in hs) == 0:
                return Constraints(cp.empty((0, prepared.num_views, prepared.num_views)), None,
                                   None, None, None, None, None, None, None, None, None)
            point = cp.concatenate(pts)
            order = cp.argsort(point)
            inverse = cp.empty_like(order); inverse[order] = cp.arange(len(order))
            h, energy = cp.concatenate(hs)[order], cp.concatenate(energies)[order]
            count = cp.concatenate([cp.full(len(r), r.shape[1], cp.int64) for r in rows_list])
            rows = cp.concatenate([r.ravel() for r in rows_list])
            blocks = inverse[cp.repeat(cp.arange(len(count)), count)]
            row_order = cp.argsort(blocks * len(prepared.obs_y) + rows)
            rows, blocks = rows[row_order], blocks[row_order]
            weighted = self.base_device[rows] * sqrt_w[rows, None, None]
            obs = y[rows] * sqrt_w[rows, None]
            row_view = view[rows]
            rhs = cp.zeros((len(h), prepared.num_views, 3), cp.complex128)
            # Rows are sorted by block; reduceat avoids complex atomic accumulation.
            starts = cp.r_[cp.array([0], cp.int64), cp.cumsum(count[order])[:-1]]
            row_rhs = cp.einsum('rki,rk->ri', weighted, obs)
            for v in subset:
                rhs[:, v] = cp.add.reduceat(row_rhs * (row_view == v)[:, None], starts, axis=0)
            return Constraints(h / cp.maximum(energy, sync.EPSILON**2)[:, None, None],
                cp.real(cp.diagonal(h, axis1=1, axis2=2)), rows, blocks, 2*count[order],
                cp.concatenate(grams)[order], rhs, cp.maximum(cp.sqrt(energy), sync.EPSILON),
                weighted, obs, row_view)

    def raw_counts(self, prepared):
        self._bind(prepared)
        return self.base["raw_counts"]

    def edge_statistics(self, constraints):
        cp = self.cp
        h = constraints.information
        v = h.shape[1]
        scale = cp.maximum(cp.real(cp.trace(h, axis1=1, axis2=2)), sync.EPSILON)
        present = cp.abs(h) > 1e-12 * scale[:, None, None]
        count = cp.sum(present, axis=0).astype(cp.int32)
        information = cp.sum(cp.where(present, cp.abs(h), 0), axis=0)
        cp.fill_diagonal(count, 0); cp.fill_diagonal(information, 0)
        return count.get(), information.get()

    def constraint_counts(self, constraints):
        return self.cp.sum(constraints.column_energy > sync.EPSILON**2, axis=0).astype(self.cp.int32).get()

    def residuals(self, c, parameters, candidate_views):
        """Batch parameters and Gaussian blocks without host residual transfers."""
        cp = self.cp
        nparam = len(candidate_views)-1
        alpha = cp.ones((len(parameters), c.gram.shape[1]), cp.complex128)
        unknown = candidate_views[candidate_views != 0]
        alpha[:, unknown] = 1 / cp.exp(-parameters[:, nparam:] - 1j*parameters[:, :nparam])
        # Reserve live residual/Jacobian and thin-SVD buffers before temporary batches.
        required = 8 * len(c.rows) * 4 * (6*(2*nparam)+8)
        self._budget(required)
        residual = cp.empty((len(parameters), len(c.rows), 2), cp.complex128)
        norms = cp.empty((len(parameters), len(c)), cp.float64)
        row_starts = cp.r_[cp.array([0], cp.int64), cp.cumsum(c.equation_count//2)]
        def solve(lo, hi):
            gram = cp.einsum('pv,bvij->pbij', cp.abs(alpha)**2, c.gram[lo:hi]).astype(cp.complex128)
            rhs = cp.einsum('pv,bvi->pbi', alpha.conj(), c.rhs[lo:hi])
            with __import__('cupyx').errstate(linalg='raise'):
                eig, vectors = cp.linalg.eigh(gram)
            cutoff = (cp.finfo(cp.float64).eps*cp.maximum(c.equation_count[lo:hi], 3))**2 * cp.maximum(eig[..., -1], 0)
            inverse = cp.where(eig > cutoff[..., None], 1 / cp.where(eig > cutoff[..., None], eig, 1), 0)
            projected = cp.einsum('pbji,pbj->pbi', vectors.conj(), rhs)
            phi = cp.einsum('pbij,pbj->pbi', vectors, inverse*projected)
            first, last = int(row_starts[lo]), int(row_starts[hi])
            pred = alpha[:, c.views[first:last], None] * cp.einsum('rki,pri->prk',
                c.jacobian[first:last], phi[:, c.blocks[first:last]-lo])
            r = (pred-c.observation[None, first:last]) / c.normalization[c.blocks[first:last]][None, :, None]
            sq = cp.sum(cp.abs(r)**2, axis=-1)
            norm = cp.sqrt(cp.maximum(cp.add.reduceat(sq, row_starts[lo:hi]-first, axis=1), 0))
            return first, last, r, norm
        max_rows = int(cp.max(c.equation_count))//2
        for lo, hi, (first, last, r, norm) in self._chunks(len(c), len(parameters)*(2000+max_rows*160), solve, required=required):
            residual[:, first:last] = r
            norms[:, lo:hi] = norm
        return residual, norms

    def refine(self, prepared, c, candidate_views, config):
        cp = self.cp
        from modal_gaussians.motion.observations._alpha_trf import least_squares
        views = cp.asarray(candidate_views)
        unknown = views[views != 0]
        info = c.information[:, views][:, :, views]
        with self.timed("initialization_seconds"):
            total = cp.sum(info, axis=0)
            beta = cp.linalg.lstsq(total[1:, 1:], -total[1:, 0], rcond=None)[0]
            x = cp.concatenate((-cp.angle(beta), cp.clip(-cp.log(cp.maximum(cp.abs(beta), sync.EPSILON)),
                                                       np.log(config.gain_minimum), np.log(config.gain_maximum))))
            n = len(unknown)
            lb = cp.r_[cp.full(n, -cp.inf), cp.full(n, np.log(config.gain_minimum))]
            ub = cp.r_[cp.full(n, cp.inf), cp.full(n, np.log(config.gain_maximum))]
            _, initial_norm = self.residuals(c, x[None], views)
            huber = cp.maximum(cp.median(initial_norm), 1e-6)
        def fun_many(parameters):
            with self.timed("residual_jacobian_seconds"):
                packed = cp.empty((len(parameters), 4*len(c.rows)), cp.float64)
                def evaluate(lo, hi):
                    residual, norms = self.residuals(c, parameters[lo:hi], views)
                    ratio = huber/cp.maximum(norms, sync.EPSILON)
                    factor = cp.where(norms > huber, cp.sqrt(cp.maximum(2*ratio-ratio**2, 0)), 1.)
                    residual *= factor[:, c.blocks, None]
                    return cp.concatenate((residual.real.reshape(hi-lo, -1),
                                           residual.imag.reshape(hi-lo, -1)), axis=1)
                for lo, hi, values in self._chunks(len(parameters), len(c.rows)*160+len(c)*2000, evaluate):
                    packed[lo:hi] = values
                self.stats["residual_parameter_evaluations"] = self.stats.get("residual_parameter_evaluations", 0)+len(parameters)
                return packed
        with self.timed("trf_seconds"):
            result = least_squares(fun_many, x, (lb, ub))
        with self.timed("diagnostics_seconds"):
            alpha = cp.ones(len(views), cp.complex128)
            alpha[1:] = 1/cp.exp(-result.x[n:] - 1j*result.x[:n])
            _, norm = self.residuals(c, result.x[None], views)
            norm = norm[0]
            robust = cp.where(norm > huber, huber/cp.maximum(norm, sync.EPSILON), 1.)
            constraint_information = cp.einsum('b,bij->ij', robust, info)
            parameter_information = result.jac.T @ result.jac
            singular = cp.sqrt(cp.maximum(cp.linalg.eigvalsh(.5*(parameter_information+parameter_information.T)), 0))[::-1]
            rank_ratio = float(singular[-1]/cp.maximum(singular[0], sync.EPSILON))
            information_ratio = float(singular[-1]/np.sqrt(max(len(c), 1)))
            condition = float(singular[0]/cp.maximum(singular[-1], sync.EPSILON))
            active = cp.r_[cp.array([False]), result.active_mask[n:] != 0]
            phase_std, gain_std = cp.full(len(views), cp.inf), cp.full(len(views), cp.inf)
            phase_std[0] = gain_std[0] = 0
            if bool(singular[-1] > sync.EPSILON):
                variance = cp.sum(result.fun**2)/max(len(result.fun)-len(result.x), 1)
                std = cp.sqrt(cp.maximum(cp.diag(cp.linalg.pinv(parameter_information))*variance, 0))
                phase_std[1:], gain_std[1:] = std[:n], std[n:]
                gain_std[active] = cp.inf
            self.stats.update(nfev=result.nfev, njev=result.njev, optimizer_status=result.status)
        with self.timed("download_seconds"):
            return sync.AlphaCandidateSolve(alpha.get(), float(cp.sqrt(cp.mean(norm**2))), singular.get(),
                rank_ratio, information_ratio, condition, phase_std.get(), gain_std.get(),
                constraint_information.get(), parameter_information.get(), result.success, result.status,
                result.message, active.get(), "profiled_exact_block_huber_gauss_newton")
