"""Populate the existing soft-weight cache before reserving a GPU training slot."""
import json
import os
from pathlib import Path
from modal_gaussians.scene_store import resolve_path, logical_path

import numpy as np

from modal_gaussians.iteration_cache import Timings, atomic_json, exclusive_work, identity
from . import neural_modes as nm
from .geometry_graph import GeometryGraph
from .iteration import resolve_config, training_revision
from .prepared import load_prepared
from .shared_controls import weighted_geometry


FORMAT = "modal_gaussians.control_weights_ready"


def _read(path):
    return json.loads(resolve_path(path).read_text(encoding="utf-8-sig"))


def propagation_count(control, default):
    workers = control.get("propagation_workers", default)
    if type(workers) is not int or workers < 1:
        raise ValueError("propagation_workers must be a positive integer")
    return workers


def ready(path, *, prepared_dir, graph_dir, frequency_hz, revision, config_identity):
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
    if (record["prepared_identity"] != _read(Path(prepared_dir) / "manifest.json")["prepared_identity"]
            or record["graph_identity"] != identity(_read(Path(graph_dir) / "manifest.json"))):
        raise ValueError(f"Control preparation inputs changed: {path}")
    caches = record.get("caches", {})
    if set(caches) != {"shared_control_geometry", "frequency_control_weights"}:
        raise ValueError(f"Control preparation is missing cache locations: {path}")
    return all((resolve_path(cache) / "manifest.json").is_file() and (resolve_path(cache) / "arrays.npz").is_file()
               for cache in caches.values())


def prepare_control_weights(*, prepared_dir, geometry_graph_dir, config_path, frequency_hz, output_path):
    output = resolve_path(output_path)
    timer = Timings()
    with exclusive_work(output.with_suffix(".lock")):
        prepared = load_prepared(prepared_dir)
        prepared.attach_geometry_graph(geometry_graph_dir)
        config = resolve_config(prepared.manifest["defaults"], _read(config_path))
        neural = nm.NeuralModesConfig.from_dict(config["neural"])
        external = prepared.external_geometry_contract
        if config["fragment"].get("strategy") != "component_field":
            raise ValueError("CPU weight preparation requires the component-field baseline")
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
                 frequency_hz=frequency_hz, revision=revision, config_identity=config_id):
            return output
        # Each batch child reads the live setting once, before creating its pool.
        # Existing pools finish unchanged; standalone calls retain their environment setting.
        batch_root = output.parent.parent
        control_file = batch_root / "batch_workers.json"
        control = (_read(control_file) if control_file.is_file() and (batch_root / "batch_contract.json").is_file() else {})
        key = "MODAL_GAUSSIANS_PROPAGATION_WORKERS"
        previous = os.environ.get(key)
        workers = propagation_count(control, int(previous or "1"))
        try:
            os.environ[key] = str(workers)
            weighted_geometry(GeometryGraph.from_dict(prepared.external_geometry_graph),
                geometry_config=nm._geometry_config(neural), fragment_config=config["fragment"],
                scene_scale=float(prepared.arrays["o_scene_scale"]), cache_dir=prepared.cache_dir, timer=timer)
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        record = {"format": FORMAT, "version": 1, "status": "complete",
            "prepared": str(logical_path(prepared.path)), "prepared_identity": prepared.manifest["prepared_identity"],
            "graph": str(logical_path(graph_dir)), "graph_identity": external["manifest_identity"],
            "frequency_hz": frequency_hz, "revision": revision, "config_identity": config_id,
            "caches": {stage["stage"]: stage["cache_path"] for stage in timer.records
                       if stage["stage"] in ("shared_control_geometry", "frequency_control_weights")},
            "timings": timer.records, "propagation_workers": workers}
        atomic_json(output, record)
    return output
