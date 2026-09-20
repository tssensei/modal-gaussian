"""GPU soft propagation; CPU is an explicit legacy compatibility route."""
import numpy as np


def backend_identity(backend):
    if backend == "cpu":
        return {"backend": "cpu"}
    if backend != "cupy":
        raise ValueError(f"Unknown propagation backend: {backend}")
    from . import control_propagation_gpu
    from modal_gaussians.iteration_cache import module_revision
    return {"backend": "cupy", "kernel_revision": module_revision(control_propagation_gpu)}


def support_distances(adjacency, controls, support, distance, maximum_stretch, *, workers=None,
                      backend="cupy", workspace=None):
    backend_identity(backend)
    if backend == "cupy":
        from .control_propagation_gpu import Workspace
        if workspace is not None:
            return workspace.search(adjacency, controls, support, distance, maximum_stretch)
        owned = Workspace()
        try:
            return owned.search(adjacency, controls, support, distance, maximum_stretch)
        finally:
            owned.close()
    from modal_gaussians.motion.legacy.neural.control_propagation import support_distances as legacy_distances
    return legacy_distances(adjacency, controls, support, distance, maximum_stretch, workers=workers)


def attenuation(adjacency, controls, support, distance, maximum_stretch, *, workers=None,
                backend="cupy", workspace=None):
    propagated = support_distances(adjacency, controls, support, distance, maximum_stretch,
        workers=workers, backend=backend, workspace=workspace)
    return np.divide(distance, propagated, out=np.ones_like(distance), where=propagated > 0)
