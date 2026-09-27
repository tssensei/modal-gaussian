---
name: modal-gaussians-pipeline
description: Run or recover the Modal Gaussians pipeline from matching inputs through 3D modes, joint mode/coefficient refinement, Viser-ready results, original/before/after reconstruction videos and metrics. Respect explicitly shorter endpoints; reviewing or editing the skill does not launch experiments.
---

# Modal Gaussians pipeline

For a request to run the pipeline with no narrower endpoint, finish spatial modes,
fixed-scene joint motion/coefficient refinement, reconstruction exports and metrics.
Do not stop at the mode batch index. An explicit stage-only, modes-only, evaluation-only
or other limited request takes precedence. Editing/reviewing code or this skill alone
never starts a real-scene run.

1. Read [README](../../README.md),
   [storage](../../SCENE_STORAGE.md),
   [baseline](../../BASELINE.md),
   [reconstruction](../../COEFFICIENT_FITTING.md),
   [rebuild boundaries](../../REBUILD.md),
   and the requested scene's catalog/index. Resolve stored paths with
   `modal_gaussians.common.scene_store.resolve_path()`.
2. Establish scene, exact bins/frequency order, resolution, recordings/frame range
   and numerical recipe from the request and established experiment context. Reuse
   only compatible inputs. Do not silently make 20 modes, 300 frames, 6000 updates
   or an experiment's rotation weight universal defaults. Read repository configs
   when no recipe is established. Ask only for genuinely missing choices.
3. Use [commands](references/commands.md) for raw registration, optional DA3 depth,
   static SH geometry, union-of-boxes subject selection, stabilization, SEA-RAFT,
   shared FFT, alpha alignment and component-field training. Preserve original
   modal observation views separately from reconstruction supervision. A sweep
   supplies static cameras/geometry; do not automatically add it to RGB refinement.
   Preserve an established view1-only run. Separate recordings have independent q.
4. Continue using [reconstruction and delivery](references/reconstruction.md):
   prepare fixed interpolation, select validated flow initialization or zero-q
   warmup, alternate joint mode/q and exhaustive q updates, then publish the final
   selected state with the unchanged static scene. Retain a clearly identified
   unrefined-mode baseline on exactly the same sequence bindings.
5. Complete all default deliverables below. These evaluation/export stages are
   included in a full-pipeline run request; do not ask again merely to cross a stage
   boundary. Individual stage APIs remain independent; orchestrate their calls.
6. Diagnose and recover in-scope failures with [recovery](references/validation-recovery.md).
   Keep logs/checkpoints and immutable parents. Never bypass identity checks,
   silently reduce inputs/training, or repeat an unchanged deterministic failure.

## Default completion contract

- **Viser ready:** validated materialized baseline and final results, including
  displacement AND angular fields, correct per-recording coefficients, cameras,
  frame clocks and source identities. Supply concrete absolute-path viewer commands
  and a writable viewer work directory. Ready does not mean the server is running;
  launch it only if requested. No viewer initialization in training or loaders.
- **Videos, per supervised sequence:** `original.mp4`, `unrefined.mp4`, `refined.mp4`,
  plus a synchronized `comparison_3way.mp4` in that order. Original means the actual
  registered/stabilized PNG sequence used for fitting; label it accordingly. Never
  compare raw unstabilized input against registered reconstructions without saying so.
  Preserve frame count, time grid and native pixel size; document codec padding.
- **Metrics:** baseline and final native-resolution valid-support PSNR, SSIM and
  RMSE, per-frame CSV, per-sequence/aggregate JSON, and refined-minus-baseline deltas.
  Include LPIPS when its dependency/weights are available; otherwise state it was
  not computed. An explicitly requested missing metric remains incomplete.
- **Handoff:** paths and identities, exact bins/views/frame ranges, baseline meaning,
  initialization/config, actual versus published steps, measured stage timings,
  metric changes and failures/limitations. No fabricated timing or quality claims.
  Report partial completion if any required output failed; do not call it done.

The spatial model is the component field with separate reliable donors, GPU alpha
and soft propagation, per-view RMS normalization, deformation 0.03 and spatial
control-rotation regularization 0. Motion-refinement relative rotation is a separate
setting (repository default 0.001). Joint refinement freezes all Gaussian attributes,
counts, cameras, control geometry and donor roles; it never optimizes Gaussian shape
or adds adjacent-flow loss, frequency locking or new modes implicitly.

Publish new products in new scene experiment paths and preserve raw data. A pipeline
run includes necessary fitting, recovery, exports and evaluation, but not commit,
push, uploads, package downgrades, raw deletion or broad cache cleanup. Scientific
acceptance and a launched/visually reviewed Viser session are separate from successful
execution and validated artifacts.

`static refine-scene` is an explicitly requested optional stage, never appended by
default. It changes the Gaussian scene/density while freezing motion and sweep q;
see the scene-refinement commands in [commands](references/commands.md).
