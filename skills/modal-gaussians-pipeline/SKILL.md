---
name: modal-gaussians-pipeline
description: Run or recover the current SEA-RAFT/shared-FFT/component-field pipeline through saved 3D modes. Do not fit temporal coefficients or start Viser unless separately requested.
---

# Modal Gaussians pipeline

Use for executing or recovering this repository's spatial-mode pipeline. For
read-only design/code review, read the project docs instead of launching stages.

1. Read [README](../../README.md), [storage](../../SCENE_STORAGE.md),
   [baseline](../../BASELINE.md), [rebuild boundaries](../../REBUILD.md), and the
   selected scene's catalog/index. Resolve saved paths through
   `modal_gaussians.common.scene_store.resolve_path()`.
2. Establish the requested scene, exact frequencies/bins and output endpoint.
   Inspect existing matching inputs before computing. Separately recorded views
   are asynchronous; never match frames as simultaneous observations.
3. Use [commands](references/commands.md) for missing stages. Keep source pixels,
   static identities, Gaussian order, reference selections and numerical settings
   fixed. Publish in a new scene experiment. Never rewrite immutable metadata to
   make an incompatible cache appear usable.
4. Stop after requested mode publication. Do not automatically run real-scene
   validation, coordinate fitting, exports, Viser, benchmark comparisons or metric
   reports. A requested GUI command can be handed to the user without launching it.
5. For an authorized run, diagnose errors, preserve logs/checkpoints, fix in-scope
   implementation problems, and resume compatible work. Follow
   [recovery](references/validation-recovery.md). Do not repeat an unchanged failure
   or silently reduce frequency count, resolution, training or view count.

The sole spatial model is the component field with separate reliable donors.
Use GPU alpha and soft propagation, per-view RMS normalization, deformation 0.03
and control-rotation regularization 0. A selected modal graph is frequency specific;
only matching static/control geometry is shared. Current models save displacement
and angular fields directly. No old rigid, attachment, CPU solver, preview or
model-format compatibility path remains.

A code cleanup request does not authorize real experiments. A run request includes
necessary computation and recovery, but does not authorize commit, push, upload,
package downgrades, raw-input deletion or broad cache cleanup. Distinguish successful
execution from visual inspection and scientific approval when reporting results.
