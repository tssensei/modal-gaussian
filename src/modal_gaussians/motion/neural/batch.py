"""Separate CPU preparation and GPU training queues around single-frequency CLI stages."""
import csv
import json
import os
from pathlib import Path
from modal_gaussians.scene_store import resolve_path, logical_path
import subprocess
import sys
import time

from modal_gaussians.iteration_cache import atomic_json, exclusive_work, identity
from modal_gaussians.progress import report_progress
from .iteration import resolve_config, training_revision
from .control_preparation import ready as weights_ready, propagation_count


def _read(path):
    return json.loads(resolve_path(path).read_text(encoding="utf-8-sig"))


def _jobs(export_dir, prepared, graph, config, output, experiment_name):
    manifest = _read(export_dir / "manifest.json")
    if (manifest.get("format") != "modal_gaussians.spectrum_export"
            or manifest.get("version") != 1 or manifest.get("status") != "complete"):
        raise ValueError("Batch input must be a completed cached-spectrum export")
    bins = manifest["selection"]["bins"]
    frequencies = manifest["selection"]["frequencies_hz"]
    if not bins or len(set(bins)) != len(bins) or len(bins) != len(frequencies):
        raise ValueError("Batch requires distinct selected bins and matching frequencies")
    jobs = []
    for index, frequency in zip(bins, frequencies):
        if type(index) is not int or index <= 0 or not 0 < frequency < float("inf"):
            raise ValueError("Invalid positive frequency/bin")
        name = f"bin_{index:04d}"
        folder = output / name
        views, labels = [], set()
        for record in manifest["modes"]:
            if record["bin_index"] != index:
                continue
            path = (export_dir / record["path"]).resolve(strict=True)
            if (not path.is_relative_to(export_dir) or record["view"] in labels
                    or record["frequency_hz"] != frequency):
                raise ValueError("Export view path, frequency or labels differ")
            labels.add(record["view"])
            views.extend(["--view", record["view"], str(path)])
        if not views:
            raise ValueError(f"No exported views for {name}")
        target, weights, experiment = folder / "prepared", folder / "graph", folder / experiment_name
        stages = [
            ("prepare", target / "manifest.json", ["motion", "prepare-selected-modal", "--prepared", str(prepared),
                *views, "--frequency-hz", str(frequency), "--output", str(target)]),
            ("graph", weights / "manifest.json", ["graph", "build-modal-similarity", "--prepared", str(target),
                "--geometry-graph", str(graph), *views, "--frequency", str(frequency),
                "--soft-weights", "--minimum-edge-factor", "0.05", "--output", str(weights)]),
            ("weights", folder / "control_weights_ready.json", ["motion", "prepare-control-weights",
                "--prepared", str(target), "--geometry-graph", str(weights), "--config", str(config),
                "--frequency-hz", str(frequency), "--output", str(folder / "control_weights_ready.json")]),
            ("train", experiment / "status.json", ["motion", "iterate-neural", "--prepared", str(target),
                "--geometry-graph", str(weights), "--config", str(config), "--frequency-hz", str(frequency),
                "--output", str(experiment), "--stage", "modes"]),
        ]
        jobs.append({"bin": index, "frequency_hz": frequency, "folder": folder, "stages": stages,
                     "prepared_dir": target, "graph_dir": weights,
                     "next": 0, "state": "pending", "pid": None, "stage": None})
    return manifest, jobs


def _published(job, stage, parent):
    name, marker, _ = stage
    if name == "weights":
        return weights_ready(marker, prepared_dir=job["prepared_dir"], graph_dir=job["graph_dir"],
            frequency_hz=job["frequency_hz"], revision=job["revision"], config_identity=job["config_identity"])
    if not marker.is_file():
        return False
    record = _read(marker)
    if name == "prepare":
        source = record["source"]
        if (record.get("format") != "modal_gaussians.neural_prepared"
                or [m["frequency_hz"] for m in source["modes"]] != [job["frequency_hz"]]
                or source["selected_modal_supervision"]["parent_prepared_identity"] != parent["prepared_identity"]):
            raise ValueError(f"Existing preparation differs: {marker}")
    elif name == "graph":
        if (record.get("format") != "modal_gaussians.modal_similarity_graph"
                or record["frequency_hz"] != job["frequency_hz"]
                or resolve_path(record["source"]["prepared"]) != job["prepared_dir"]):
            raise ValueError(f"Existing graph differs: {marker}")
    elif record.get("status") != "modes_ready":
        return False
    return True


def _gpu_sample(stream, active):
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode == 0:
            for row in csv.reader(result.stdout.splitlines()):
                csv.writer(stream).writerow([time.time(), len(active), *[v.strip() for v in row]])
            stream.flush()
    except (OSError, subprocess.TimeoutExpired):
        pass  # Monitoring availability must not turn a successful training run into a failure.


def _completed_from(source, contract, jobs):
    """Explicitly carry completed modes across attempts; never rewrite old identities."""
    previous = _read(source / "batch_contract.json")
    keys = ("format", "version", "modal_images", "export_identity", "prepared", "prepared_identity",
            "geometry_graph", "geometry_identity", "config")
    if any(previous.get(key) != contract.get(key) for key in keys):
        raise ValueError("Resume source has different selection, geometry or scientific configuration")
    snapshot = _read(source / "batch_state.json")
    records = {record["bin"]: record for record in snapshot["jobs"]}
    for job in jobs:
        record = records.get(job["bin"])
        if record is None or record["state"] != "complete":
            continue
        result = resolve_path(record.get("result_dir") or source / job["folder"].name / previous["experiment_name"])
        if record["frequency_hz"] != job["frequency_hz"] or _read(result / "status.json").get("status") != "modes_ready":
            raise ValueError("Resume source is not a matching completed frequency")
        job.update(state="complete", stage="modes_ready", next=len(job["stages"]), result_dir=str(result))


def _continued_from(source, contract, jobs):
    """Reuse published CPU inputs but revisit every mode under the increased cap."""
    from .continuation import increased_limit, model_contract
    previous = _read(source / "batch_contract.json")
    keys = ("format", "version", "modal_images", "export_identity", "prepared", "prepared_identity",
            "geometry_graph", "geometry_identity")
    if (any(previous.get(key) != contract.get(key) for key in keys)
            or any(previous["config"][key] != contract["config"][key] for key in ("fragment", "design"))):
        raise ValueError("Continuation changes batch inputs or scientific configuration")
    increased_limit(previous["config"]["neural"], contract["config"]["neural"])
    records = {record["bin"]: record for record in _read(source / "batch_state.json")["jobs"]}
    for job in jobs:
        record = records.get(job["bin"])
        if record is None or record["frequency_hz"] != job["frequency_hz"]:
            raise ValueError("Continuation source frequencies differ")
        result = resolve_path(record["result_dir"])
        old_iteration = result / "iteration.json"
        inputs = {"prepared_dir": source / job["folder"].name / "prepared",
                  "graph_dir": source / job["folder"].name / "graph"}
        if old_iteration.is_file():
            iteration = _read(old_iteration)
            inputs = {"prepared_dir": resolve_path(iteration["prepared"]),
                      "graph_dir": resolve_path(iteration["external_geometry_graph"]["path"])}
            job["stages"][-1][2].extend(["--continue-from", str(result)])
            prepared_manifest = _read(inputs["prepared_dir"] / "manifest.json")
            work = resolve_path(Path(prepared_manifest["cache_dir"]) / "neural_work" / identity(model_contract(iteration)))
            if (work / "fixed_inputs.npz").is_file() and (work / "manifest.json").is_file():
                job["stages"] = [stage for stage in job["stages"] if stage[0] != "weights"]
        for key, path in inputs.items():
            if not (path / "manifest.json").is_file():
                continue
            old = job[key]
            job[key] = path
            for index, (name, marker, arguments) in enumerate(job["stages"]):
                if (key == "prepared_dir" and name == "prepare") or (key == "graph_dir" and name == "graph"):
                    marker = path / "manifest.json"
                arguments[:] = [str(path) if value == str(old) else value for value in arguments]
                job["stages"][index] = name, marker, arguments


def run_batch(*, modal_images, prepared_dir, geometry_graph_dir, config_path, output_dir,
              cpu_workers=3, gpu_workers=2, threads_per_worker=2, propagation_workers=4,
              experiment_name="experiment_shared_001", resume_from=None, continue_from=None):
    if not (1 <= cpu_workers <= 8 and 1 <= gpu_workers <= 8) or threads_per_worker < 1 or propagation_workers < 1:
        raise ValueError("Use 1–8 workers per queue and positive CPU thread/propagation counts")
    if not experiment_name or Path(experiment_name).name != experiment_name or experiment_name in (".", ".."):
        raise ValueError("Experiment name must be a single directory name")
    exported, prepared, graph, config = [resolve_path(p, strict=True) for p in
                                        (modal_images, prepared_dir, geometry_graph_dir, config_path)]
    root = resolve_path(output_dir)
    parent = _read(prepared / "manifest.json")
    resolved = resolve_config(parent["defaults"], _read(config))
    if resolved["fragment"].get("strategy") != "component_field":
        raise ValueError("This batch runner uses the component-field baseline")
    export, jobs = _jobs(exported, prepared, graph, config, root, experiment_name)
    contract = {"format": "modal_gaussians.neural_batch", "version": 1,
        "modal_images": str(logical_path(exported)), "export_identity": identity(export),
        "prepared": str(logical_path(prepared)), "prepared_identity": parent["prepared_identity"],
        "geometry_graph": str(logical_path(graph)), "geometry_identity": identity(_read(graph / "manifest.json")),
        "config": resolved, "revision": training_revision(resolved["fragment"]),
        "experiment_name": experiment_name, "threads_per_worker": threads_per_worker}
    for job in jobs:
        job.update(revision=contract["revision"], config_identity=identity(resolved))
    if continue_from is not None:
        if resume_from is not None:
            raise ValueError("Use only one of --continue-from or --resume-from")
        source = resolve_path(continue_from, strict=True)
        if source == root:
            raise ValueError("Continuation requires a new batch directory")
        contract["continue_from"] = str(source)
        with exclusive_work(source / "batch.lock"):
            _continued_from(source, contract, jobs)
    if resume_from is not None:
        source = resolve_path(resume_from, strict=True)
        if source == root:
            raise ValueError("Resume into a new batch directory; old results remain immutable")
        contract["resume_from"] = str(source)
        # No live source batch may launch another copy while results are inherited.
        with exclusive_work(source / "batch.lock"):
            _completed_from(source, contract, jobs)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_work(root / "batch.lock"):
        contract_file = root / "batch_contract.json"
        if contract_file.exists() and _read(contract_file) != contract:
            raise ValueError("Batch inputs/configuration/code changed; use a new batch directory")
        if not contract_file.exists():
            if any(job["stages"][-1][1].parent.exists() for job in jobs):
                raise FileExistsError("Use a fresh experiment name for the initial batch")
            atomic_json(contract_file, contract)
        limit_file = root / "batch_workers.json"
        if not limit_file.exists():
            atomic_json(limit_file, {"cpu_workers": cpu_workers, "gpu_workers": gpu_workers,
                                     "propagation_workers": propagation_workers})
        env = dict(os.environ, PYTHONUNBUFFERED="1",
                   MODAL_GAUSSIANS_PROPAGATION_WORKERS=str(propagation_workers))
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(threads_per_worker)
        cpu_env = dict(env, CUDA_VISIBLE_DEVICES="")
        active, failed = {}, False

        def queue(stage):
            return "gpu" if stage == "train" else "cpu"

        def counts():
            return {key: sum(queue(job["stage"]) == key for _, _, job in active.values()) for key in limits}

        def save(status):
            running = counts()
            snapshot = {"status": status, "pid": os.getpid(),
                "updated": time.time(), "completed": sum(j["state"] == "complete" for j in jobs),
                "total": len(jobs), "workers": sum(limits.values()),
                "cpu_workers": limits["cpu"], "gpu_workers": limits["gpu"],
                "active_cpu": running["cpu"], "active_gpu": running["gpu"],
                "ready_for_gpu": sum(j["state"] == "pending" and j["stage"] == "train" for j in jobs),
                "propagation_workers": propagation_workers,
                "jobs": [{**{k: job[k] for k in ("bin", "frequency_hz", "state", "stage", "pid")},
                          "queue": queue(job["stage"]) if job["state"] != "complete" else None,
                          "result_dir": job.get("result_dir", str(job["stages"][-1][1].parent))} for job in jobs]}
            try:
                atomic_json(root / "batch_state.json", snapshot)
            except OSError as error:
                if (status != "running" or os.name != "nt"
                        or getattr(error, "winerror", None) not in {5, 32, 33}):
                    raise
                # Monitoring is not a training artifact; retry its refresh next tick.
                report_progress(f"batch WARNING status refresh deferred (Windows file lock); computation continues: {error}")

        limits = {"cpu": cpu_workers, "gpu": gpu_workers}
        with (root / "gpu_usage.csv").open("a", newline="", encoding="utf-8") as resource_log:
            if resource_log.tell() == 0:
                csv.writer(resource_log).writerow(["unix_time", "active_processes", "gpu_timestamp",
                    "memory_used_mib", "memory_total_mib", "gpu_util_percent", "memory_util_percent", "power_w"])
            last_sample = 0.
            try:
                while True:
                    control = _read(limit_file)
                    requested_propagation = propagation_count(control, propagation_workers)
                    if requested_propagation != propagation_workers:
                        report_progress(f"batch propagation workers {propagation_workers} -> {requested_propagation} (new pools only)")
                        propagation_workers = requested_propagation
                        env["MODAL_GAUSSIANS_PROPAGATION_WORKERS"] = str(propagation_workers)
                        cpu_env["MODAL_GAUSSIANS_PROPAGATION_WORKERS"] = str(propagation_workers)
                    requested = {key: control.get(key + "_workers") for key in limits}
                    if any(type(value) is not int or not 0 <= value <= 8 for value in requested.values()):
                        raise ValueError("batch_workers.json requires cpu_workers and gpu_workers integers from 0 to 8")
                    if requested != limits:
                        report_progress(f"batch queue limits {limits} -> {requested} (running tasks are preserved)")
                    limits = requested
                    for pid, (process, log, job) in list(active.items()):
                        code = process.poll()
                        if code is None:
                            continue
                        log.close()
                        del active[pid]
                        job["pid"] = None
                        if code != 0 or not _published(job, job["stages"][job["next"]], parent):
                            job["state"], failed = "failed", True
                            report_progress(f"batch FAILED {job['frequency_hz']:g} Hz {job['stage']} exit={code}; no new launches")
                        else:
                            job["next"] += 1
                            job["state"] = "pending"
                    for job in jobs:
                        if job["state"] != "pending":
                            continue
                        # Completed pre-split experiments need no new CPU preparation.
                        if _published(job, job["stages"][-1], parent):
                            job["next"] = len(job["stages"])
                        while job["next"] < len(job["stages"]) and _published(job, job["stages"][job["next"]], parent):
                            job["next"] += 1
                        if job["next"] == len(job["stages"]):
                            job["state"], job["stage"] = "complete", "modes_ready"
                            report_progress(f"batch complete {sum(j['state'] == 'complete' for j in jobs)}/{len(jobs)} | {job['frequency_hz']:g} Hz")
                            continue
                        stage, _, arguments = job["stages"][job["next"]]
                        job["stage"] = stage
                        resource = queue(stage)
                        if failed or counts()[resource] >= limits[resource]:
                            continue
                        job["folder"].mkdir(parents=True, exist_ok=True)
                        log = (job["folder"] / f"parallel_{stage}.log").open("a", encoding="utf-8")
                        try:
                            process = subprocess.Popen([sys.executable, "-m", "modal_gaussians.cli", *arguments],
                                stdout=log, stderr=subprocess.STDOUT, env=cpu_env if resource == "cpu" else env,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                        except BaseException:
                            log.close()
                            raise
                        job.update(state="running", stage=stage, pid=process.pid)
                        active[process.pid] = process, log, job
                        report_progress(f"batch start {job['frequency_hz']:g} Hz | {stage} | {resource.upper()} | pid={process.pid}")
                    if time.monotonic() - last_sample >= 5:
                        _gpu_sample(resource_log, active)
                        last_sample = time.monotonic()
                    done = all(job["state"] == "complete" for job in jobs)
                    save("complete" if done else "failed" if failed else "running")
                    if done or (failed and not active):
                        break
                    time.sleep(1)
            except BaseException:
                for process, _, _ in active.values():
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW)
                    else:
                        process.terminate()
                for process, log, job in active.values():
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    log.close()
                    job.update(state="interrupted", pid=None)
                active.clear()
                save("interrupted")
                raise
        if failed:
            raise RuntimeError(f"Batch failed; inspect {root / 'batch_state.json'} and per-stage logs")
    return root
