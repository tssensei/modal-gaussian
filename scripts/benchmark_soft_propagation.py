"""Explicit three-frequency Bush CPU/CuPy comparison; never runs training/upstream stages."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

import numpy as np

from modal_gaussians.iteration_cache import atomic_json, put_entry, identity
from modal_gaussians.motion.neural import control_propagation as propagation, shared_controls as shared
from modal_gaussians.motion.neural.geometry_graph import GeometryGraph
from modal_gaussians.motion.common.graph_ops import host_subgraph


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def publish_results(root):
    """Import measured results into the normal cache without another shortest-path run."""
    from modal_gaussians.motion.neural.control_preparation import prepare_control_weights
    root = Path(root).resolve()
    report = read(root / "report.json")
    if report["status"] != "complete":
        raise ValueError("Only a completed comparison can be published")
    library = Path("scene_library").resolve()
    catalog = read(library / "bush/catalog.json")
    assets = catalog["assets"]
    geometry_dir, _ = shared.prepare_shared_controls(prepared_dir=library/assets["prepared"],
        geometry_graph_dir=library/assets["candidate_graph"], controls_from=library/assets["control_geometry"],
        config_path=Path("configs/neural_component_field.json"))
    geometry = read(geometry_dir/"manifest.json")["contract"]
    report["imported_control_geometry"] = str(geometry_dir)
    for row in report["frequencies"]:
        record = next(m for m in catalog["modes"] if m["frequency_hz"] == row["frequency_hz"])
        graph = GeometryGraph.from_dict(arrays(library/record["graph"]/"graph.npz"))
        cache = Path(row["weight_cache"])
        measured = read(cache/"manifest.json")["contract"]
        current = shared.weight_contract(graph, geometry, "cupy")
        if any(measured[k] != current[k] for k in current if k != "geometry"):
            raise ValueError("Measured GPU implementation/weights changed; do not relabel the result")
        weights = arrays(cache/"arrays.npz")
        start = time.perf_counter()
        put_entry(library/assets["cache"]/"control_weights", current, weights)
        row["gpu_canonical_publish_seconds"] = time.perf_counter()-start
        row["canonical_gpu_weights"] = str(library/assets["cache"]/"control_weights"/identity(current))
        weights.update(arrays(root/f"cpu_weights_{row['frequency_hz']:g}.npz"))
        start = time.perf_counter()
        put_entry(library/assets["cache"]/"control_weights", shared.weight_contract(graph,geometry,"cpu"), weights)
        row["cpu_cache_publish_seconds"] = time.perf_counter()-start
        # Same entry training will consume. This publishes metadata, never launches a GNN.
        marker = root/f"ready_{row['frequency_hz']:g}.json"
        prepare_control_weights(prepared_dir=library/record["prepared"], geometry_graph_dir=library/record["graph"],
            config_path=Path("configs/neural_component_field.json"), frequency_hz=row["frequency_hz"],
            output_path=marker, backend="cupy")
        row["ready_marker"] = str(marker)
        row["cpu_end_to_end_seconds"] = row["load_seconds"]+row["cpu_soft_seconds"]+row["cpu_cache_publish_seconds"]
        row["gpu_end_to_end_seconds"] = row["load_seconds"]+row["gpu_soft_median"]+row["gpu_canonical_publish_seconds"]
        row["end_to_end_speedup"] = row["cpu_end_to_end_seconds"]/row["gpu_end_to_end_seconds"]
    atomic_json(root/"report.json",report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    library = Path("scene_library").resolve()
    catalog = read(library / "bush/catalog.json")
    geometry_dir = library / catalog["assets"]["control_geometry"]
    start = time.perf_counter()
    layout = arrays(geometry_dir / "arrays.npz")
    geometry_contract = read(geometry_dir / "manifest.json")["contract"]
    support = shared._support_by_control(layout)
    load_geometry_seconds = time.perf_counter()-start
    report = {"status": "running", "cpu_workers": 4, "rtol": 1e-9, "atol": 1e-11,
        "shared_geometry": str(geometry_dir), "shared_load_seconds": load_geometry_seconds,
        "controls": len(layout["c_control_point_index"]), "support_entries": len(support.data), "frequencies": []}
    stop = threading.Event()

    def monitor():
        with (root / "gpu_usage.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time", "memory_used_mib", "memory_total_mib", "utilization_percent"])
            while not stop.is_set():
                sample = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                for row in csv.reader(sample.stdout.splitlines()): writer.writerow([time.time(), *row])
                stream.flush()
                stop.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    jobs = []
    try:
        # CPU references run first; no GPU search or other benchmark job overlaps them.
        for frequency in (.25, 5., 10.25):
            record = next(m for m in catalog["modes"] if m["frequency_hz"] == frequency)
            start = time.perf_counter()
            graph = GeometryGraph.from_dict(arrays(library / record["graph"] / "graph.npz"))
            host = host_subgraph(graph, layout["t_host_gaussian_index"])
            adjacency = shared._adjacency(host, host.propagation_lengths())
            stretch = float(np.max(host.propagation_lengths()/host.edge_length))
            inputs = adjacency, layout["c_control_point_index"], support, layout["support_distance"], stretch
            row = {"frequency_hz": frequency, "graph": record["graph"], "load_seconds": time.perf_counter()-start}
            print(f"CPU {frequency:g} Hz", flush=True)
            start = time.perf_counter()
            distances = propagation.support_distances(*inputs, workers=4, backend="cpu")
            row["cpu_search_seconds"] = time.perf_counter()-start
            start = time.perf_counter()
            weights = shared.reweight_geometry(graph, layout, propagated=distances)
            row["cpu_weight_seconds"] = time.perf_counter()-start
            row["cpu_soft_seconds"] = row["cpu_search_seconds"]+row["cpu_weight_seconds"]
            # Only requested support distances, never a dense control x node matrix.
            np.save(root / f"cpu_support_{frequency:g}.npy", distances)
            np.savez(root / f"cpu_weights_{frequency:g}.npz", **{k: weights[k] for k in shared.WEIGHT_NAMES})
            jobs.append((graph, inputs, row, distances, weights))
            report["frequencies"].append(row)
            atomic_json(root / "report.json", report)
        from modal_gaussians.motion.neural.control_propagation_gpu import Workspace
        start = time.perf_counter()
        workspace = Workspace()
        report["gpu_cold_seconds"] = time.perf_counter()-start
        report["compile_seconds"] = workspace.compile_seconds
        report["cupy_version"] = workspace.cp.__version__
        report["cuda_runtime"] = workspace.cp.cuda.runtime.runtimeGetVersion()
        report["cuda_driver"] = workspace.cp.cuda.runtime.driverGetVersion()
        report["gpu"] = str(workspace.cp.cuda.runtime.getDeviceProperties(0)["name"])
        print("GPU full-size warmup", flush=True)
        workspace.search(*jobs[0][1])
        report["warmup"] = dict(workspace.stats)
        for graph, inputs, row, expected, expected_weights in jobs:
            row["gpu_runs"] = []
            for repeat in range(3):
                print(f"GPU {row['frequency_hz']:g} Hz repeat {repeat+1}/3", flush=True)
                start = time.perf_counter()
                distances = workspace.search(*inputs)
                search_wall = time.perf_counter()-start
                np.testing.assert_allclose(distances, expected, rtol=1e-9, atol=1e-11)
                start = time.perf_counter()
                weights = shared.reweight_geometry(graph, layout, propagated=distances)
                weight_seconds = time.perf_counter()-start
                for name in shared.WEIGHT_NAMES:
                    np.testing.assert_allclose(weights[name], expected_weights[name], rtol=1e-9, atol=1e-11)
                row["gpu_runs"].append({**workspace.stats, "wall_seconds": search_wall,
                    "cpu_weight_seconds": weight_seconds, "soft_seconds": search_wall+weight_seconds,
                    "max_distance_error": float(np.max(np.abs(distances-expected))),
                    "max_weight_error": float(np.max(np.abs(weights["c_interpolation_weights"]-expected_weights["c_interpolation_weights"])))})
                atomic_json(root / "report.json", report)
            # Benchmark publication is isolated under this experiment; no baseline changes.
            contract = shared.weight_contract(graph, geometry_contract, "cupy")
            start = time.perf_counter()
            put_entry(root / "control_weights", contract, weights)
            row["cache_publish_seconds"] = time.perf_counter()-start
            row["weight_cache"] = str(root / "control_weights" / identity(contract))
            row["gpu_search_median"] = statistics.median(r["search_seconds"] for r in row["gpu_runs"])
            row["gpu_soft_median"] = statistics.median(r["soft_seconds"] for r in row["gpu_runs"])
            row["search_speedup"] = row["cpu_search_seconds"]/row["gpu_search_median"]
            row["soft_speedup"] = row["cpu_soft_seconds"]/row["gpu_soft_median"]
            atomic_json(root / "report.json", report)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        stop.set()
        thread.join(timeout=10)
        atomic_json(root / "report.json", report)
    publish_results(root)


if __name__ == "__main__":
    main()
