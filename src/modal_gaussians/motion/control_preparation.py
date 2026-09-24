"""Populate the existing soft-weight cache before reserving a GPU training slot."""
import json
import os
from pathlib import Path
from modal_gaussians.common.scene_store import resolve_path, logical_path

import numpy as np

from modal_gaussians.common.cache import Timings, atomic_json, exclusive_work, identity
from modal_gaussians.motion import training as nm
from modal_gaussians.motion.geometry_graph import GeometryGraph
from modal_gaussians.motion.iteration import resolve_config, training_revision
from modal_gaussians.motion.prepared import load_prepared
from modal_gaussians.motion.shared_controls import weighted_geometry
from modal_gaussians.motion.control_propagation import backend_identity


FORMAT = "modal_gaussians.control_weights_ready"


def _read(path):
    return json.loads(resolve_path(path).read_text(encoding="utf-8-sig"))


def ready(path, *, prepared_dir, graph_dir, frequency_hz, revision, config_identity, backend="cupy"):
    """Inspect publication metadata only; never read back numerical cache payloads."""
    path = resolve_path(path)
    if not path.is_file():
        return False
    record = _read(path)
    expected = {"format": FORMAT, "version": 1, "status": "complete",
        "prepared": str(logical_path(prepared_dir)), "graph": str(logical_path(graph_dir)),
        "frequency_hz": frequency_hz, "revision": revision, "config_identity": config_identity}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Control preparation differs: {path}")
    if record.get("propagation") != backend_identity(backend):
        raise ValueError(f"Control preparation backend/kernel differs: {path}")
    if (record["prepared_identity"] != _read(Path(prepared_dir) / "manifest.json")["prepared_identity"]
            or record["graph_identity"] != identity(_read(Path(graph_dir) / "manifest.json"))):
        raise ValueError(f"Control preparation inputs changed: {path}")
    caches = record.get("caches", {})
    if set(caches) != {"shared_control_geometry", "frequency_control_weights"}:
        raise ValueError(f"Control preparation is missing cache locations: {path}")
    if "propagation" in record:
        weight_path = resolve_path(caches["frequency_control_weights"])
        if (weight_path / "manifest.json").is_file():
            weight_contract = _read(weight_path / "manifest.json")["contract"]
            if (weight_contract.get("propagation") != record["propagation"]
                    or weight_path.name != identity(weight_contract)):
                raise ValueError(f"Control weight cache identity differs: {path}")
    return all((resolve_path(cache) / "manifest.json").is_file() and (resolve_path(cache) / "arrays.npz").is_file()
               for cache in caches.values())


def prepare_control_weights(*, prepared_dir, geometry_graph_dir, config_path, frequency_hz, output_path,
                            backend="cupy", workspace=None):
    propagation = backend_identity(backend)
    output = resolve_path(output_path)
    timer = Timings()
    with exclusive_work(output.with_suffix(".lock")):
        prepared = load_prepared(prepared_dir)
        prepared.attach_geometry_graph(geometry_graph_dir)
        config = resolve_config(prepared.manifest["defaults"], _read(config_path))
        neural = nm.NeuralModesConfig.from_dict(config["neural"])
        external = prepared.external_geometry_contract
        if config["fragment"].get("strategy") != "component_field":
            raise ValueError("Weight preparation requires the component-field baseline")
        if (len(prepared.source["modes"]) != 1
                or not np.isclose(prepared.source["modes"][0]["frequency_hz"], frequency_hz, rtol=0, atol=1e-9)
                or not np.isclose(external["frequency_hz"], frequency_hz, rtol=0, atol=1e-9)):
            raise ValueError("Prepared observations and modal graph must match the requested frequency")
        if (neural.graph_edge_filter != "none" or neural.graph_neighbors != external["config"]["graph_neighbors"]
                or not np.isclose(neural.graph_max_distance, external["config"]["graph_max_distance"], rtol=0, atol=1e-12)):
            raise ValueError("Control preparation requires matching candidate K/radius and no extra graph filter")
        revision, config_id = training_revision(config["fragment"]), identity(config)
        graph_dir = resolve_path(geometry_graph_dir)
        if ready(output, prepared_dir=prepared.path, graph_dir=graph_dir,
                 frequency_hz=frequency_hz, revision=revision, config_identity=config_id, backend=backend):
            return output
        weighted_geometry(GeometryGraph.from_dict(prepared.external_geometry_graph),
            geometry_config=nm._geometry_config(neural), fragment_config=config['fragment'],
            scene_scale=float(prepared.arrays['o_scene_scale']), cache_dir=prepared.cache_dir, timer=timer,
            backend=backend, workspace=workspace)
        record = {"format": FORMAT, "version": 1, "status": "complete",
            "prepared": str(logical_path(prepared.path)), "prepared_identity": prepared.manifest["prepared_identity"],
            "graph": str(logical_path(graph_dir)), "graph_identity": external["manifest_identity"],
            "frequency_hz": frequency_hz, "revision": revision, "config_identity": config_id,
            "caches": {stage["stage"]: stage["cache_path"] for stage in timer.records
                       if stage["stage"] in ("shared_control_geometry", "frequency_control_weights")},
            "timings": timer.records,
            "propagation": propagation}
        atomic_json(output, record)
    return output


def run_gpu_worker(root, parent_pid):
    """One private batch subprocess, a single atomic request at a time."""
    import time
    import traceback
    from modal_gaussians.motion.control_propagation_gpu import Workspace
    root = Path(root)
    workspace = None
    operation = None
    # A killed scheduler must not leave an idle CUDA context holding 20+ GiB.
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, 0, parent_pid)  # SYNCHRONIZE
        if not handle:
            raise OSError("Cannot monitor the owning batch process")

    def parent_alive():
        if os.name == "nt":
            return kernel.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT
        try:
            os.kill(parent_pid, 0)
            return True
        except ProcessLookupError:
            return False

    last = None
    try:
        while parent_alive():
            request_file = root / "propagation_request.json"
            if not request_file.exists():
                time.sleep(.2)
                continue
            request = _read(request_file)
            if request["id"] == last:
                time.sleep(.2)
                continue
            last = request["id"]
            if request.get("stop"):
                break
            requested = request.get("operation", "weights")
            atomic_json(root / "propagation_status.json", {"id": last, "operation": requested,
                "status": "running", "pid": os.getpid()})
            try:
                if requested not in {"prepare", "weights"}:
                    raise ValueError("Unknown GPU preparation operation")
                if operation == "weights" and requested == "prepare":
                    raise ValueError("Alpha preparation cannot restart after the weights barrier")
                if requested != operation:
                    if workspace is not None:
                        workspace.close()
                        workspace = None
                    if requested == "prepare":
                        from modal_gaussians.motion.observations.alpha_gpu import Workspace as AlphaWorkspace
                        workspace = AlphaWorkspace()
                    else:
                        workspace = Workspace()
                    operation = requested
                workspace.stats = {}
                if operation == "prepare":
                    from modal_gaussians.motion.selected_modal import prepare_selected_modal
                    prepare_selected_modal(**request["arguments"], alpha_backend="cupy", alpha_workspace=workspace)
                else:
                    prepare_control_weights(**request["arguments"], backend="cupy", workspace=workspace)
            except BaseException:
                atomic_json(root / "propagation_status.json", {"id": last, "operation": requested,
                    "status": "failed", "error": traceback.format_exc()})
                raise
            atomic_json(root / "propagation_status.json", {"id": last, "operation": operation,
                "status": "complete", "propagation_workers": None, "gpu": workspace.stats})
    finally:
        if workspace is not None:
            workspace.close()
        if os.name == "nt":
            kernel.CloseHandle(handle)


if __name__ == "__main__":
    import sys
    run_gpu_worker(sys.argv[1], int(sys.argv[2]))
