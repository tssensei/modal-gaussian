"""GPU soft propagation on each frequency graph."""
import numpy as np


def backend_identity(backend):
    if backend != "cupy":
        raise ValueError(f"Unknown propagation backend: {backend}")
    from modal_gaussians.motion import control_propagation_gpu
    from modal_gaussians.common.cache import module_revision
    return {"backend": "cupy", "kernel_revision": module_revision(control_propagation_gpu)}


def support_distances(adjacency, controls, support, distance, maximum_stretch, *, backend="cupy", workspace=None):
    backend_identity(backend)
    from modal_gaussians.motion.control_propagation_gpu import Workspace
    if workspace is not None:
        return workspace.search(adjacency, controls, support, distance, maximum_stretch)
    owned = Workspace()
    try:
        return owned.search(adjacency, controls, support, distance, maximum_stretch)
    finally:
        owned.close()


def attenuation(adjacency, controls, support, distance, maximum_stretch, *, backend="cupy", workspace=None):
    propagated = support_distances(adjacency, controls, support, distance, maximum_stretch,
        backend=backend, workspace=workspace)
    return np.divide(distance, propagated, out=np.ones_like(distance), where=propagated > 0)
