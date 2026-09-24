"""Resumable single-frequency 3D-mode experiments."""
from __future__ import annotations
import copy
from dataclasses import fields
import json
from pathlib import Path
from modal_gaussians.common.scene_store import resolve_path, logical_path
import time

import numpy as np
from modal_gaussians.common.cache import Timings, atomic_json, exclusive_work, identity, module_revision
from modal_gaussians.common.progress import report_progress
from modal_gaussians.motion.prepared import load_prepared
from modal_gaussians.motion import training as nm, network as neural_field, geometry_graph
from modal_gaussians.motion.common import projection
from modal_gaussians.geometry import scene as static
from modal_gaussians.common import camera_geometry


def resolve_config(defaults, overrides):
    from modal_gaussians.motion.component_field import ComponentFieldConfig
    resolved = copy.deepcopy(defaults)
    types = {'neural': nm.NeuralModesConfig, 'fragment': ComponentFieldConfig}
    if not isinstance(overrides, dict) or set(overrides) - set(types):
        raise ValueError('Configuration sections must be neural or fragment')
    for section, values in overrides.items():
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(types[section])}:
            raise ValueError(f'Unknown parameters in {section}')
        resolved[section].update(values)
    resolved['neural']['training_fragment_config'] = copy.deepcopy(resolved['fragment'])
    for section, cls in types.items():
        settings = cls(**resolved[section])
        settings.validate()
        resolved[section] = settings.to_dict()
    return resolved


def resolve_mode_slots(modes, frequencies_hz):
    if frequencies_hz is None:
        return None
    slots = []
    for frequency in frequencies_hz:
        matches = [i for i, mode in enumerate(modes)
                   if np.isfinite(frequency) and np.isclose(mode["frequency_hz"], frequency, rtol=0, atol=1e-9)]
        if len(matches) != 1 or matches[0] in slots:
            raise ValueError(f"Requested frequency {frequency} must occur exactly once in the prepared modes and request")
        slots.append(matches[0])
    return nm._validated_mode_slots(sorted(slots), len(modes))


def training_revision(strategy_config=None):
    from modal_gaussians.motion import artifacts, continuation, control_propagation, prepared, shared_controls, modal_projection, component_field
    from modal_gaussians.common import camera_rendering
    from modal_gaussians.motion.common import graph_ops, point_transfer, visibility
    return module_revision(nm, neural_field, geometry_graph, projection, static,
        camera_geometry, camera_rendering, modal_projection, artifacts, continuation, prepared, shared_controls,
        control_propagation, component_field, graph_ops, point_transfer, visibility)


def iterate_neural(*, prepared_dir, config_path, output_dir, stage="modes", frequencies_hz=None,
                   geometry_graph_dir=None, continue_from=None,
                   propagation_backend="cupy"):
    from modal_gaussians.motion.control_propagation import backend_identity
    propagation = backend_identity(propagation_backend)
    if stage != 'modes':
        raise ValueError('Iteration only produces modes; fitting and export are separate commands')
    root = resolve_path(output_dir)
    timer = Timings()
    with timer.stage("prepared_load"):
        prepared = load_prepared(prepared_dir)
        prepared.propagation_backend = propagation_backend
        if geometry_graph_dir is not None:
            prepared.attach_geometry_graph(geometry_graph_dir)
        overrides = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
        config = resolve_config(prepared.manifest["defaults"], overrides)
        settings = nm.NeuralModesConfig.from_dict(config["neural"])
        prepared.validate_sources(prepared.source, settings)
        mode_slots = resolve_mode_slots(prepared.source["modes"], frequencies_hz)
        if getattr(prepared, "external_geometry_contract", None) is not None:
            external = prepared.external_geometry_contract
            slots = range(len(prepared.source["modes"])) if mode_slots is None else mode_slots
            if any(not np.isclose(prepared.source["modes"][slot]["frequency_hz"], external["frequency_hz"],
                                  rtol=0, atol=1e-9) for slot in slots):
                raise ValueError("External modal graph may only train its selected frequency")
            if (settings.graph_neighbors != external["config"]["graph_neighbors"]
                    or not np.isclose(settings.graph_max_distance, external["config"]["graph_max_distance"], rtol=0, atol=1e-12)
                    or settings.graph_edge_filter != "none"):
                raise ValueError("External graph requires matching candidate K/radius and graph_edge_filter=none")
    if root == prepared.path or root.is_relative_to(prepared.path):
        raise ValueError("Experiment must not overwrite prepared inputs")
    strategy_config = settings.training_fragment_config
    neural_revision = training_revision(strategy_config)
    contract = {"version": 2, "prepared": str(logical_path(prepared.path)),
                "prepared_identity": prepared.manifest["prepared_identity"],
                "config": config, "neural_revision": neural_revision}
    contract["propagation"] = propagation
    if getattr(prepared, "external_geometry_contract", None) is not None:
        contract["external_geometry_graph"] = prepared.external_geometry_contract
    if mode_slots is not None:
        contract["source_mode_slots"] = mode_slots
    continuation = None
    if continue_from is not None:
        from modal_gaussians.motion.continuation import source_work
        if resolve_path(continue_from) == root:
            raise ValueError("Continuation requires a new experiment directory")
        continuation = source_work(continue_from, contract)
        contract["continuation"] = continuation
    if root.exists() and not (root / "iteration.json").is_file():
        raise FileExistsError(f"Existing directory is not a neural iteration: {root}")
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_work(root / "iteration.lock"):
        if (root / "iteration.json").exists():
            if json.loads((root / "iteration.json").read_text()) != contract:
                raise ValueError("Experiment inputs/configuration/code changed; use a new output directory")
        else:
            atomic_json(root / "iteration.json", contract)
            atomic_json(root / "overrides.json", overrides)
        try:
            result_root = _run_stages(root, prepared, config, neural_revision, stage, timer, mode_slots, continuation)
        except BaseException:
            atomic_json(root / "status.json", {"status": "failed", "stage_requested": stage})
            raise
        finally:
            timer.save(root / f"timings_{stage}.json")
    return result_root or root


def _build_modes_with_publication_retry(**kwargs):
    """A final directory lock can retry publication using the completed checkpoints."""
    destination = resolve_path(kwargs["output_dir"])
    delays = (0.0, 0.25, 1.0)
    for attempt, delay in enumerate(delays):
        if delay:
            time.sleep(delay)
        try:
            return nm.build_neural_modes_artifact(**kwargs)
        except OSError as error:
            target = getattr(error, "filename2", None)
            if (getattr(error, "winerror", None) not in {5, 32, 33}
                    or target is None or resolve_path(target) != destination
                    or not (Path(kwargs["work_dir"]) / "manifest.json").is_file()
                    or attempt == len(delays) - 1):
                raise
            kwargs["resume"] = True
            report_progress(f"neural publication: Windows file lock; retry {attempt + 1}/2 using saved checkpoints")


def _run_stages(root, prepared, config, neural_revision, stage, timer, mode_slots=None, continuation=None):
    source = prepared.source
    model_contract = {"prepared": prepared.manifest["prepared_identity"], "config": config["neural"], "code": neural_revision}
    from modal_gaussians.motion.control_propagation import backend_identity
    model_contract["propagation"] = backend_identity(prepared.propagation_backend)
    if getattr(prepared, "external_geometry_contract", None) is not None:
        model_contract["external_geometry_graph"] = prepared.external_geometry_contract
    if mode_slots is not None:
        model_contract["source_mode_slots"] = mode_slots
    if continuation is not None:
        model_contract["continuation"] = continuation
    model_key = identity(model_contract)
    raw_path = resolve_path(prepared.cache_dir / "trained_modes" / model_key)
    with timer.stage("training_stage"):
        with exclusive_work(prepared.cache_dir / "locks" / (model_key + ".lock")):
            hit = raw_path.exists()
            if hit:
                raw = nm.load_neural_completed_modes(raw_path)
                if raw.manifest["config"] != config["neural"] or raw.manifest["source_identity"] != prepared.manifest["source_identity"]:
                    raise ValueError("Cached neural modes differ from experiment")
            else:
                work = resolve_path(prepared.cache_dir / "neural_work" / model_key)
                raw = _build_modes_with_publication_retry(
                    work_dir=work, output_dir=raw_path, config=nm.NeuralModesConfig.from_dict(config["neural"]),
                    resume=(work / "manifest.json").exists(), prepared_inputs=prepared, timings=timer,
                    continuation=continuation,
                    command=["motion", "iterate-neural", str(root)])
    timer.records[-1]["cache_hit"] = hit
    completed_path = raw_path
    outputs = {"prepared": str(prepared.path), "raw_modes": str(raw_path),
               "completed_modes": str(completed_path)}
    atomic_json(root / 'outputs.json', outputs)
    atomic_json(root / 'status.json', {'status': 'modes_ready', 'viewer_started': False,
        'stage_requested': stage, 'outputs': outputs})
    return root
