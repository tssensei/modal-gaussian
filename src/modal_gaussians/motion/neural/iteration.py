"""Resumable 3D-mode experiments with optional preview and coordinate stages."""
from __future__ import annotations
import copy
from dataclasses import fields
import json
from pathlib import Path
from typing import Any

import numpy as np
from modal_gaussians.iteration_cache import Timings, atomic_json, exclusive_work, identity, module_revision
from modal_gaussians.motion.neural.prepared import load_prepared
from modal_gaussians.motion.neural import neural_modes as nm, neural_field, geometry_graph, fragment_propagation as fp
from modal_gaussians.motion.neural import training_fragments, surface_attachments, pointwise_attachments, guarded_attachments
from modal_gaussians.motion.neural import component_field
from modal_gaussians.motion.common import projection
from modal_gaussians import static, camera_geometry
from modal_gaussians.rendered_design import RenderedDesignConfig, RenderedDesignViewInput, build_rendered_modal_design_artifact, load_rendered_modal_design
from modal_gaussians.direct_coordinates import DirectCoordinateViewInput, build_direct_modal_coordinates_artifact, load_direct_modal_coordinates
from modal_gaussians.motion.neural.preview import build_preview, load_preview


def resolve_config(defaults, overrides):
    resolved = copy.deepcopy(defaults)
    if not isinstance(overrides, dict) or set(overrides) - set(resolved):
        raise ValueError("Iteration configuration sections must be neural, fragment or design")
    fragment_changes = overrides.get("fragment", {})
    if not isinstance(fragment_changes, dict):
        raise ValueError("Fragment overrides must be an object")
    if "strategy" in fragment_changes and fragment_changes["strategy"] != resolved["fragment"].get("strategy"):
        if fragment_changes["strategy"] not in ("surface", "pointwise", "guarded", "component_field"):
            raise ValueError("Unknown fragment strategy")
        previous = resolved["fragment"]
        if fragment_changes["strategy"] == "component_field":
            resolved["fragment"] = component_field.ComponentFieldConfig().to_dict()
        elif fragment_changes["strategy"] == "guarded":
            resolved["fragment"] = guarded_attachments.GuardedAttachmentConfig().to_dict()
        elif fragment_changes["strategy"] == "pointwise":
            resolved["fragment"] = pointwise_attachments.PointwiseAttachmentConfig().to_dict()
        else:
            resolved["fragment"] = (surface_attachments.SurfaceAttachmentConfig(legacy_config=previous)
                if "strategy" not in previous else surface_attachments.SurfaceAttachmentConfig()).to_dict()
    types = {"neural": nm.NeuralModesConfig, "fragment": training_fragments.config_class(resolved["fragment"]), "design": RenderedDesignConfig}
    for section, values in overrides.items():
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(types[section])}:
            raise ValueError(f"Unknown iteration parameters in {section}")
        resolved[section].update(values)
    # Fragment thresholds have one user-facing home. Explicit null retains the
    # legacy train-then-propagate path for reproducible comparisons.
    fragment_override = overrides.get("neural", {}).get("training_fragment_config", "default")
    if fragment_override is None:
        if resolved["fragment"].get("strategy") in ("surface", "pointwise", "guarded", "component_field"):
            raise ValueError("Surface/pointwise/guarded/component_field attachments require in-training fill")
        resolved["neural"]["training_fragment_config"] = None
    else:
        if fragment_override != "default" and fragment_override != resolved["fragment"]:
            raise ValueError("Set fragment thresholds in the fragment section; neural values disagree")
        resolved["neural"]["training_fragment_config"] = copy.deepcopy(resolved["fragment"])
    for section, cls in types.items():
        allowed = {f.name for f in fields(cls)}
        settings = cls(**{k: v for k, v in resolved[section].items() if k in allowed})
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


def iterate_neural(*, prepared_dir, config_path, output_dir, stage="modes", frequencies_hz=None,
                   refine_observations=False, refinement_config_path=None):
    if stage not in ("modes", "preview", "full"):
        raise ValueError("Iteration stage must be modes, preview or full")
    refinement = None
    if refinement_config_path is not None and not refine_observations:
        raise ValueError("--refinement-config requires --refine-observations")
    if refine_observations:
        from .observation_refinement import ObservationRefinementConfig
        values = json.loads(Path(refinement_config_path).read_text()) if refinement_config_path else {}
        refinement = ObservationRefinementConfig.from_dict(values)
    root = Path(output_dir).expanduser().resolve()
    timer = Timings()
    with timer.stage("prepared_load"):
        prepared = load_prepared(prepared_dir)
        overrides = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
        config = resolve_config(prepared.manifest["defaults"], overrides)
        settings = nm.NeuralModesConfig.from_dict(config["neural"])
        if refinement is not None and (settings.training_fragment_config or {}).get("strategy") == "component_field":
            raise ValueError("Observation refinement is disabled for component_field; omit --refine-observations")
        if refinement is not None and (settings.training_fragment_config or {}).get("strategy") not in ("pointwise", "guarded"):
            raise ValueError("Observation refinement requires strategy=pointwise or guarded")
        prepared.validate_sources(prepared.source, settings)
        mode_slots = resolve_mode_slots(prepared.source["modes"], frequencies_hz)
    if root == prepared.path or root.is_relative_to(prepared.path):
        raise ValueError("Experiment must not overwrite prepared inputs")
    neural_revision = module_revision(nm, neural_field, geometry_graph, projection, static, camera_geometry, training_fragments, fp, surface_attachments)
    if settings.training_fragment_config and settings.training_fragment_config.get("strategy") == "pointwise":
        neural_revision = identity({"base": neural_revision, "pointwise": module_revision(pointwise_attachments)})
    if settings.training_fragment_config and settings.training_fragment_config.get("strategy") == "guarded":
        neural_revision = identity({"base": neural_revision, "guarded": module_revision(guarded_attachments, pointwise_attachments)})
    if settings.training_fragment_config and settings.training_fragment_config.get("strategy") == "component_field":
        neural_revision = identity({"base": neural_revision, "component_field":
            module_revision(component_field, guarded_attachments, pointwise_attachments)})
    contract = {"version": 1, "prepared": str(prepared.path), "prepared_identity": prepared.manifest["prepared_identity"],
                "config": config, "neural_revision": neural_revision, "fragment_revision": module_revision(fp)}
    if mode_slots is not None:
        contract["source_mode_slots"] = mode_slots
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
            result_root = _run_stages(root, prepared, config, neural_revision, stage, timer, mode_slots, refinement)
        except BaseException:
            atomic_json(root / ("refinement_status.json" if refinement else "status.json"), {"status": "failed", "stage_requested": stage})
            raise
        finally:
            timer.save(root / f"timings_{stage}{'_refined' if refinement else ''}.json")
    return result_root or root


def _run_stages(root, prepared, config, neural_revision, stage, timer, mode_slots=None, refinement=None):
    from modal_gaussians.result import materialize_modal_result, load_modal_result
    source = prepared.source
    model_contract = {"prepared": prepared.manifest["prepared_identity"], "config": config["neural"], "code": neural_revision}
    if mode_slots is not None:
        model_contract["source_mode_slots"] = mode_slots
    model_key = identity(model_contract)
    raw_path = prepared.cache_dir / "trained_modes" / model_key
    with timer.stage("training_stage"):
        with exclusive_work(prepared.cache_dir / "locks" / (model_key + ".lock")):
            hit = raw_path.exists()
            if hit:
                raw = nm.load_neural_completed_modes(raw_path)
                if raw.manifest["config"] != config["neural"] or raw.manifest["source_identity"] != prepared.manifest["source_identity"]:
                    raise ValueError("Cached neural modes differ from experiment")
                expected_slots = list(range(len(source["modes"]))) if mode_slots is None else mode_slots
                if nm._source_mode_slots(raw.manifest).tolist() != expected_slots:
                    raise ValueError("Cached neural frequency selection differs from experiment")
            else:
                work = prepared.cache_dir / "neural_work" / model_key
                raw = nm.build_neural_modes_artifact(
                    scene_dir=source["static_scene"], topology_dir=source["topology"], measurements_dir=source["measurements"],
                    graph_dir=source["observed_structure_graph"], alignment_from=source["alignment_from"],
                    work_dir=work, output_dir=raw_path, config=nm.NeuralModesConfig.from_dict(config["neural"]),
                    resume=(work / "manifest.json").exists(), prepared_inputs=prepared, timings=timer,
                    mode_slots=mode_slots,
                    command=["motion", "iterate-neural", str(root)])
    timer.records[-1]["cache_hit"] = hit
    if config["neural"].get("training_fragment_config") is not None:
        completed, completed_path = raw, raw.path
        if raw.manifest["version"] != training_fragments.artifact_contract(config["neural"]["training_fragment_config"])[0]:
            raise ValueError("Training fill version differs from its attachment strategy")
    else:
        completed_path = root / "neural_completed_modes"
        with timer.stage("fragment_propagation"):
            if completed_path.exists():
                completed = fp.load_fragment_modes(completed_path)
            else:
                completed = fp.build_fragment_modes(parent_dir=raw.path, output_dir=completed_path,
                                                    config=fp.FragmentPropagationConfig.from_dict(config["fragment"]))
            if (completed.manifest["parent_completed_modes_identity"] != raw.manifest["completed_modes_identity"]
                    or completed.manifest["fragment_propagation"] != config["fragment"]):
                raise ValueError("Existing propagation is not this experiment")
    propagation_path = completed_path
    if refinement is not None:
        from . import observation_refinement as ref
        ref_key = identity({"parent": completed.manifest["completed_modes_identity"],
                            "prepared": prepared.manifest["prepared_identity"],
                            "config": refinement.to_dict(), "code": module_revision(ref)})
        # The same training cache serves both variants; never overwrite the base preview.
        root = root / "refinements" / ref_key
        root.mkdir(parents=True, exist_ok=True)
        completed_path = root / "completed_modes"
        with timer.stage("observation_refinement"):
            if completed_path.exists():
                refined = ref.load_refined_modes(completed_path)
            else:
                refined = ref.build_refined_modes(parent=completed, prepared=prepared, output_dir=completed_path,
                                                  config=refinement, work_dir=root / "work")
            if (refined.manifest["parent_completed_modes_identity"] != completed.manifest["completed_modes_identity"]
                    or refined.manifest["refinement_config"] != refinement.to_dict()):
                raise ValueError("Refinement differs from the selected parent/configuration")
            completed = refined
    outputs = {"prepared": str(prepared.path), "raw_modes": str(raw_path),
               "completed_modes": str(completed_path)}
    if refinement is not None:
        outputs["propagation_modes"] = str(propagation_path)
    if stage == "modes":
        atomic_json(root / "outputs.json", outputs)
        atomic_json(root / "diagnostics.json", completed.manifest["diagnostics"])
        atomic_json(root / "status.json", {"status": "modes_ready", "viewer_started": False,
                                          "stage_requested": stage, "outputs": outputs})
        return root
    design_path = root / "rendered_design"
    with timer.stage("rendered_design"):
        if design_path.exists():
            design = load_rendered_modal_design(design_path)
        else:
            design = build_rendered_modal_design_artifact(scene_dir=source["static_scene"], completed_modes_dir=completed_path,
                views=[RenderedDesignViewInput(v["label"], Path(f["path"])) for v, f in zip(source["views"], prepared.manifest["flows"])],
                output_dir=design_path, flow_loader=prepared.flow,
                config=RenderedDesignConfig(**{f.name: config["design"][f.name] for f in fields(RenderedDesignConfig)}))
        if design.manifest["completed_modes_identity"] != completed.manifest["completed_modes_identity"] or design.manifest["settings"] != config["design"]:
            raise ValueError("Existing rendered design differs from experiment")
    preview_path = root / "preview"
    with timer.stage("preview_publication"):
        if not preview_path.exists():
            preview = build_preview(prepared_dir=prepared.path, scene_dir=source["static_scene"], completed_modes_dir=completed_path,
                                    rendered_design_dir=design_path, output_dir=preview_path)
        else:
            preview = load_preview(preview_path)
        if preview.manifest["completed_modes_identity"] != completed.manifest["completed_modes_identity"]:
            raise ValueError("Existing preview differs from experiment")
    # Publication validates artifact bindings. Visual/manual-playback evaluation
    # is performed by the user after launch, not as an extra iteration stage.
    outputs.update(rendered_design=str(design_path), preview=str(preview_path))
    if stage == "full":
        coordinates_path = root / "direct_coordinates"
        with timer.stage("coordinates"):
            if coordinates_path.exists():
                coordinates = load_direct_modal_coordinates(coordinates_path)
            else:
                coordinates = build_direct_modal_coordinates_artifact(rendered_design_dir=design_path,
                    views=[DirectCoordinateViewInput(v["label"], Path(f["path"])) for v, f in zip(source["views"], prepared.manifest["flows"])],
                    output_dir=coordinates_path)
            if coordinates.manifest["rendered_design_identity"] != design.manifest["rendered_design_identity"]:
                raise ValueError("Coordinates differ from this design")
        with timer.stage("full_result"):
            result_path = root / "modal_result"
            if not result_path.exists():
                materialize_modal_result(scene_dir=source["static_scene"], completed_modes_dir=completed_path,
                    coordinates_dir=coordinates_path, output_dir=result_path)
            result = load_modal_result(result_path)
            if result.manifest["completed_modes_identity"] != completed.manifest["completed_modes_identity"]:
                raise ValueError("Full result modes differ")
            outputs["result"] = str(result_path)
        with timer.stage("full_spectrum_cache"):
            from modal_gaussians.vis.viewer import ModalViewerData
            data = ModalViewerData(preview_path, preview=True)
            for label in data.spectrum.available_view_ids:
                data.spectrum.load_full_spectrum(label)
            del data
    atomic_json(root / "outputs.json", outputs)
    atomic_json(root / "diagnostics.json", completed.manifest["diagnostics"])
    atomic_json(root / "status.json", {"status": "preview_ready" if stage == "preview" else "viser_ready",
                                      "viewer_started": False, "visualization_checked": False,
                                      "stage_requested": stage, "outputs": outputs})
    return root
