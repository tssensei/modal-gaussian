"""Exact double-precision frontier relaxation, with reusable CuPy device storage.

CuPy is imported only when a Workspace is constructed. No saved array uses the
internal uint64 frontier numbering. Kernel boundaries separate frontier waves.
"""
import time

import numpy as np

from modal_gaussians.common.progress import report_progress


SOURCE = r'''
extern "C" __global__ void seed(double* d, unsigned long long* q,
    const long long* source, unsigned long long n, unsigned long long c) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c) { unsigned long long p = i*n + source[i]; d[p] = 0.; q[i] = p; }
}
extern "C" __global__ void clear_marks(const unsigned long long* q,
    unsigned int* marks, unsigned long long count) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) marks[q[i]] = 0;
}
extern "C" __global__ void relax(double* d, const unsigned long long* q,
    unsigned long long* next, unsigned int* marks, unsigned long long* count,
    const long long* ptr, const long long* index, const double* cost,
    const double* upper, unsigned long long n, unsigned long long offset,
    unsigned long long items, unsigned long long capacity, unsigned int* error) {
    unsigned long long warp = ((unsigned long long)blockIdx.x*blockDim.x + threadIdx.x)/32;
    if (warp >= items) return;
    unsigned long long p = q[offset+warp], c = p/n, u = p%n;
    // All reads that can overlap distance updates are atomic too.
    double du = __longlong_as_double(atomicCAS((unsigned long long*)(d+p), 0ULL, 0ULL));
    for (long long e = ptr[u] + threadIdx.x%32; e < ptr[u+1]; e += 32) {
        double value = du + cost[e];
        if (!isfinite(value)) { atomicOr(error, 1U); continue; }
        if (value > upper[c]) continue;
        unsigned long long v = c*n + index[e];
        unsigned long long bits = __double_as_longlong(value);
        unsigned long long old = atomicMin((unsigned long long*)(d+v), bits);
        if (bits < old && atomicExch(marks+v, 1U) == 0U) {
            unsigned long long slot = atomicAdd(count, 1ULL);
            if (slot < capacity) next[slot] = v;
            else atomicOr(error, 2U);
        }
    }
}
extern "C" __global__ void gather(const double* d, const long long* control,
    const long long* target, double* out, unsigned long long n, unsigned long long count) {
    unsigned long long i = (unsigned long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < count) out[i] = d[(unsigned long long)control[i]*n + target[i]];
}
'''


class Workspace:
    """One frequency at a time; topology, kernels and large buffers survive calls."""
    def __init__(self, *, max_controls=None, reserve_bytes=2*1024**3):
        import cupy as cp
        self.cp = cp
        if max_controls is not None and (type(max_controls) is not int or max_controls < 1):
            raise ValueError("Diagnostic control batch size must be positive")
        if reserve_bytes < 0:
            raise ValueError("Reserved GPU memory cannot be negative")
        self.max_controls = max_controls  # Explicit diagnostic override for batch equivalence checks.
        self.reserve_bytes = reserve_bytes
        start = time.perf_counter()
        self.module = cp.RawModule(code=SOURCE, options=("--std=c++11",), backend="nvrtc")
        self.kernels = {name: self.module.get_function(name) for name in ("seed", "clear_marks", "relax", "gather")}
        cp.cuda.get_current_stream().synchronize()
        self.compile_seconds = time.perf_counter() - start
        self.topology = None
        self.buffers = None
        self.stats = {}

    def _sync(self):
        self.cp.cuda.get_current_stream().synchronize()

    def close(self):
        self._sync()
        self.buffers, self.topology = None, None
        self.ptr = self.index = None
        self.cp.get_default_memory_pool().free_all_blocks()

    def _allocate(self, nodes, controls):
        cp = self.cp
        if (self.buffers is not None and self.nodes == nodes
                and (self.capacity >= controls or self.requested_controls == controls)):
            return
        self.buffers = None
        cp.get_default_memory_pool().free_all_blocks()
        free, _ = cp.cuda.runtime.memGetInfo()
        # D + two uint64 frontiers + uint32 dedup marks, exactly 28 bytes per pair.
        batch = min(controls, max(0, (free-self.reserve_bytes)//(28*nodes)))
        if self.max_controls is not None:
            batch = min(batch, self.max_controls)
        retried = False
        while batch > 0:
            try:
                size = int(batch*nodes)
                arrays = []
                for dtype in (cp.float64, cp.uint64, cp.uint64, cp.uint32):
                    arrays.append(cp.empty(size, dtype))
                self.buffers = tuple(arrays)
                self.nodes, self.capacity = nodes, int(batch)
                self.requested_controls = controls
                self.allocation_reason = "all controls fit" if batch == controls else "free-memory budget or explicit diagnostic batch limit"
                if retried:
                    self.allocation_reason = "allocation failure; reduced control batch"
                report_progress(f"CuPy propagation: {batch}/{controls} controls per batch, {size*28/1024**3:.3f} GiB workspace; {self.allocation_reason}")
                return
            except cp.cuda.memory.OutOfMemoryError:
                arrays.clear()
                cp.get_default_memory_pool().free_all_blocks()
                batch //= 2
                retried = True
        raise MemoryError("No float64 propagation control batch fits after the reserved GPU memory")

    def search(self, adjacency, controls, support, distance, maximum_stretch):
        cp = self.cp
        started = time.perf_counter()
        n, c = adjacency.shape[0], len(controls)
        if (adjacency.shape != (n, n) or n == 0 or c == 0 or support.shape != (n, c)
                or adjacency.format != "csr" or support.format != "csc"
                or not np.issubdtype(controls.dtype, np.integer)
                or np.any(adjacency.indices < 0) or np.any(adjacency.indices >= n)
                or np.any(support.indices < 0) or np.any(support.indices >= n)
                or adjacency.data.dtype != np.float64 or distance.dtype != np.float64
                or not np.isfinite(adjacency.data).all() or np.any(adjacency.data < 0)
                or not np.isfinite(distance).all() or np.any(distance < 0)
                or not np.isfinite(maximum_stretch) or maximum_stretch < 1
                or np.any(controls < 0) or np.any(controls >= n)
                or np.any(np.diff(support.indptr) <= 0)
                or len(support.data) != len(distance)
                or not np.array_equal(np.sort(support.data), np.arange(len(distance)))):
            raise ValueError("Invalid float64 propagation graph, controls or saved support")
        if (adjacency != adjacency.T).nnz:
            raise ValueError("GPU propagation requires symmetric CSR adjacency")
        bounds = np.array([max(float(distance[support.data[support.indptr[i]:support.indptr[i+1]]].max()),
            np.finfo(np.float64).eps) for i in range(c)]) * maximum_stretch * (1+1e-12)
        if not np.isfinite(bounds).all():
            raise ValueError("Propagation search bound overflow")
        upload = time.perf_counter()
        if (self.topology is None or not np.array_equal(self.topology[0], adjacency.indptr)
                or not np.array_equal(self.topology[1], adjacency.indices)):
            self.topology = adjacency.indptr.copy(), adjacency.indices.copy()
            self.ptr = cp.asarray(adjacency.indptr, dtype=cp.int64)
            self.index = cp.asarray(adjacency.indices, dtype=cp.int64)
        cost = cp.asarray(adjacency.data)
        source = cp.asarray(controls, dtype=cp.int64)
        upper = cp.asarray(bounds)
        counter, error = cp.zeros(1, cp.uint64), cp.zeros(1, cp.uint32)
        self._sync()
        upload_seconds = time.perf_counter()-upload
        allocated = time.perf_counter()
        self._allocate(n, c)
        allocation_seconds = time.perf_counter()-allocated
        result = np.empty_like(distance)
        stats = dict(upload_seconds=upload_seconds, allocation_seconds=allocation_seconds,
            initialization_seconds=0., search_seconds=0., download_seconds=0., batches=0,
            waves=[], active_items=0, workspace_bytes=self.capacity*n*28,
            controls_per_batch=self.capacity, allocation_reason=self.allocation_reason)
        d, q0, q1, marks = self.buffers
        for first in range(0, c, self.capacity):
            count_controls = min(self.capacity, c-first)
            size = count_controls*n
            start = time.perf_counter()
            d[:size].fill(cp.inf)
            marks[:size].fill(0)
            error.fill(0)
            self.kernels["seed"](((count_controls+255)//256,), (256,),
                (d, q0, source[first:first+count_controls], np.uint64(n), np.uint64(count_controls)))
            self._sync()
            stats["initialization_seconds"] += time.perf_counter()-start
            start = time.perf_counter()
            current, following, active, wave = q0, q1, count_controls, 0
            last_report = start
            while active:
                if wave >= n:
                    raise RuntimeError("GPU propagation exceeded the host-node wave bound")
                for offset in range(0, active, 2**20):
                    items = min(active-offset, 2**20)
                    self.kernels["clear_marks"](((items+255)//256,), (256,),
                        (current[offset:offset+items], marks, np.uint64(items)))
                counter.fill(0)
                for offset in range(0, active, 2**20):
                    items = min(active-offset, 2**20)
                    self.kernels["relax"](((items+7)//8,), (256,),
                        (d, current, following, marks, counter, self.ptr, self.index, cost,
                         upper[first:first+count_controls], np.uint64(n), np.uint64(offset),
                         np.uint64(items), np.uint64(size), error))
                stats["active_items"] += active
                active = int(counter.get()[0])
                if active > size or int(error.get()[0]):
                    raise RuntimeError("GPU propagation distance overflow or frontier corruption")
                current, following = following, current
                wave += 1
                if time.perf_counter()-last_report >= 5:
                    report_progress(f"CuPy propagation: controls {first+1}-{first+count_controls}/{c}, wave {wave}, next frontier {active}, processed {stats['active_items']}")
                    last_report = time.perf_counter()
            stats["search_seconds"] += time.perf_counter()-start
            stats["waves"].append(wave)
            stats["batches"] += 1
            start = time.perf_counter()
            lo, hi = support.indptr[first], support.indptr[first+count_controls]
            targets = cp.asarray(support.indices[lo:hi], dtype=cp.int64)
            slots = cp.asarray(np.repeat(np.arange(count_controls), np.diff(support.indptr[first:first+count_controls+1])), dtype=cp.int64)
            out = cp.empty(hi-lo, cp.float64)
            self.kernels["gather"](((hi-lo+255)//256,), (256,), (d, slots, targets, out, np.uint64(n), np.uint64(hi-lo)))
            values = out.get()
            if not np.isfinite(values).all():
                raise ValueError("Soft propagation cannot reach a saved geometric support")
            result[support.data[lo:hi]] = values
            stats["download_seconds"] += time.perf_counter()-start
        stats["total_seconds"] = time.perf_counter()-started
        self.stats = stats
        return result
