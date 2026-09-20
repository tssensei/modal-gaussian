"""Explicit iteration-cap extension into a new experiment; parents stay immutable."""
import copy
import json
from pathlib import Path
from modal_gaussians.scene_store import resolve_path, logical_path

import torch
import numpy as np

from modal_gaussians.iteration_cache import identity
from modal_gaussians.progress import report_progress


def increased_limit(previous, current):
    old, new = dict(previous), dict(current)
    before, after = old.pop("max_iterations"), new.pop("max_iterations")
    if old != new or after <= before:
        raise ValueError("Continuation may only increase max_iterations; all other training settings must match")


def model_contract(iteration):
    result = {"prepared": iteration["prepared_identity"], "config": iteration["config"]["neural"],
              "code": iteration["neural_revision"]}
    result.update({k: iteration[k] for k in ("external_geometry_graph", "source_mode_slots", "continuation") if k in iteration})
    return result


def source_work(experiment, current):
    experiment = resolve_path(experiment, strict=True)
    previous = json.loads((experiment / "iteration.json").read_text(encoding="utf-8"))
    old, new = copy.deepcopy(previous), copy.deepcopy(current)
    increased_limit(old["config"]["neural"], new["config"]["neural"])
    for record in (old, new):
        record["config"]["neural"].pop("max_iterations")
        for key in ("neural_revision", "continuation"):
            record.pop(key, None)
    if old != new:
        raise ValueError("Continuation changes prepared inputs, graph, modes or other configuration")
    prepared = json.loads((resolve_path(previous["prepared"]) / "manifest.json").read_text(encoding="utf-8"))
    work = resolve_path(Path(prepared["cache_dir"]) / "neural_work" / identity(model_contract(previous)))
    manifest = work / "manifest.json"
    return {"experiment": str(logical_path(experiment)), "work": str(logical_path(work)),
            "run_identity": json.loads(manifest.read_text(encoding="utf-8"))["run_identity"] if manifest.exists() else None}


def bind_work(origin, current):
    """Compare the saved numerical-input identity, not a replay of old arrays."""
    if origin["run_identity"] is None:
        report_progress("continuation: source work unavailable; training from initialization")
        return None
    previous = json.loads((resolve_path(origin["work"]) / "manifest.json").read_text(encoding="utf-8"))
    if previous["run_identity"] != origin["run_identity"]:
        raise ValueError("Continuation source work changed")
    increased_limit(previous["config"], current["config"])
    for key in ("format", "version", "source_identity", "runtime", "geometry_graph", "fixed_arrays_identity"):
        if previous[key] != current[key]:
            report_progress(f"continuation: incompatible {key}; training from initialization")
            return None
    return resolve_path(origin["work"])


def frozen_inputs(origin, source_identity, config, runtime):
    work = resolve_path(origin["work"])
    if origin["run_identity"] is None or not (work / "fixed_inputs.npz").is_file():
        return None
    previous = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    if previous["run_identity"] != origin["run_identity"]:
        raise ValueError("Continuation source work changed")
    increased_limit(previous["config"], config)
    if previous["source_identity"] != source_identity or previous["runtime"] != runtime:
        return None
    with np.load(work / "fixed_inputs.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    report_progress("continuation: reusing saved fixed geometry, observations, controls and interpolation")
    return arrays


def checkpoint(work, mode, parent_identity, run_identity):
    path = work / f"mode_{mode:03d}.pt"
    if not path.exists():
        report_progress(f"continuation: no checkpoint for mode {mode}; training from initialization")
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("run_identity") != parent_identity or payload.get("mode") != mode:
        raise ValueError("Continuation checkpoint belongs to different inputs or mode")
    payload = dict(payload, run_identity=run_identity,
                   complete=bool(payload.get("complete") and payload.get("summary", {}).get("converged")))
    report_progress(f"continuation: mode {mode} resumes step {payload['trainer_state']['step']}"
                    + (" (already early-stopped)" if payload["complete"] else ""))
    return payload
