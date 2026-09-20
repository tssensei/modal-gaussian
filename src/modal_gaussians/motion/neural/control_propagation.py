"""Independent, exact control-source searches; workers hold one read-only graph."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

import numpy as np
from scipy.sparse.csgraph import dijkstra

from modal_gaussians.progress import Progress, report_progress


_worker_inputs = None


def _initialize(inputs):
    global _worker_inputs
    _worker_inputs = inputs
    adjacency, controls, support, distance, _ = inputs
    for array in (controls, distance, adjacency.data, adjacency.indices, adjacency.indptr,
                  support.data, support.indices, support.indptr):
        array.flags.writeable = False


def _solve(inputs, slot):
    adjacency, controls, support, distance, maximum_stretch = inputs
    lo, hi = support.indptr[slot:slot + 2]
    entries, targets = support.data[lo:hi], support.indices[lo:hi]
    geometric = distance[entries]
    limit = max(float(geometric.max()), np.finfo(np.float64).eps)
    upper = limit * maximum_stretch * (1 + 1e-12)
    # Keep the original adaptive search, including routes outside material support.
    while True:
        propagated = dijkstra(adjacency, directed=False, indices=int(controls[slot]), limit=limit)[targets]
        if np.isfinite(propagated).all():
            break
        if limit >= upper:
            raise ValueError("Soft propagation cannot reach a saved geometric support")
        limit = min(limit * 2, upper)
    return slot, np.divide(geometric, propagated, out=np.ones_like(geometric), where=propagated > 0)


def _solve_block(slots):
    return [_solve(_worker_inputs, slot) for slot in slots]


def attenuation(adjacency, controls, support, distance, maximum_stretch, *, workers=None):
    if workers is None:
        workers = int(os.environ.get("MODAL_GAUSSIANS_PROPAGATION_WORKERS", "1"))
    if type(workers) is not int or workers < 1:
        raise ValueError("Propagation workers must be a positive integer")
    workers = min(workers, max(1, len(controls)))
    inputs = adjacency, controls, support, distance, maximum_stretch
    result = np.ones_like(distance)
    progress = Progress("frequency control attenuation", len(controls), unit="controls")
    report_progress(f"soft propagation: {workers} CPU workers, {len(controls)} controls")

    def collect(blocks):
        completed = 0
        for block in blocks:
            for slot, values in block:
                lo, hi = support.indptr[slot:slot + 2]
                result[support.data[lo:hi]] = values
            completed += len(block)
            progress.update(completed)

    if workers == 1:
        collect(([_solve(inputs, slot)] for slot in range(len(controls))))
    else:
        # ponytail: one CSR copy per process; shared memory only if this limits concurrency.
        blocks = (range(start, min(start + 16, len(controls))) for start in range(0, len(controls), 16))
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize, initargs=(inputs,)) as pool:
            collect(pool.map(_solve_block, blocks))
    return result
