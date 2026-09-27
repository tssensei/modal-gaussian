# Reconstruction and final delivery

Read COEFFICIENT_FITTING.md before executing these stages. Uppercase tokens are
scene-resolved inputs, not literal paths. Use separate new prepared/work/published
and export directories under the scene experiment. Examples use one fixed view;
repeat `--view` for the established selected recordings, preserving bank order.

## Mode bank, initialization and refinement

```sh
modal-gaussians coordinates prepare --scene SCENE_NAME --index BATCH/index.json --status complete --expected-modes MODE_COUNT --output EXP/coordinates_preparation
modal-gaussians coordinates prepare-refinement --scene STATIC --modes EXP/coordinates_preparation/mode_bank --view VIEW --output EXP/prepared
modal-gaussians coordinates refine-motion --prepared EXP/prepared --config RUN_CONFIG --work-dir EXP/work --output EXP/refined
```

`coordinates prepare` collects the completed mode bank and direct-coordinate
inputs; it does not run RGB fitting. Current mode banks need no legacy importer.
Only explicitly retained v16 sources use `tools/import_refinement_reference.py`.
Keep original observation views intact even when only one recording is supervised.
Add `--sweep-metadata` to preparation only when sweep supervision is in scope;
it is not implied by using sweep images for static geometry. Preserve actual
30 FPS sweep subsets, source indices, timestamps, PNG hashes and calibrated cameras.

Resolve RUN_CONFIG from the established run; otherwise use
`configs/motion_refinement.json`. This is fixed-scene motion refinement, not
Gaussian optimization: all Gaussian attributes/counts, camera geometry, controls,
propagation weights and donor roles are frozen. Keep world-frame SH and angular
motion during rendering. The current default query block is 32768 with fixed
stencil caching, four modes/group and checkpoint recomputation. The spatial GNN's
rotation regularizer 0 is unrelated to refinement's `rotation_weight=0.001`.

Default initialization without a selected flow source is zero q with the configured
frozen-field RGB warmup, followed by alternating sparse joint updates and exhaustive
coefficient-only passes. Preserve q row-local Adam, original projection scales and
post-warmup anchor. Do not replace this with frequency locking or adjacent-flow loss.

For the established flow-initialized route, validate and supply the exact source:

```sh
modal-gaussians coordinates refine-motion --prepared EXP/prepared --flow-initialization FLOW_DIAGNOSTIC --config RUN_CONFIG --work-dir EXP/work --output EXP/refined
```

Choose ONE refine command, not both. `load_flow_initialization` imports
`solution/reference_only.npy`, already including shared reference offset, and its
scales; it skips zero-q warmup and restricts supervision to the exact verified
fixed-view prefix. Never add the offset twice, recenter q, or reuse it after changing
modes/scene/frames. Do not silently shrink a requested full recording to an available
prefix. If the established run requires a flow source and it is unavailable, recover
that source or report the blocker instead of changing initialization. A new adjacent
flow comparison experiment is not a mandatory prerequisite for ordinary refinement.

Run the configured schedule or configured early stopping to completion. With early
stopping, publish the driver's best checked state; distinguish actual and published
steps. Do not hardcode the historical 6000-step snapshot as a universal budget or
call an interrupted checkpoint the final result.

## Define the unrefined baseline precisely

Unless a specific baseline was requested, use the **pre-joint state**: original
spatial displacement/angular fields plus the q anchor immediately before joint
updates. For zero-start runs this is the end-of-warmup q; for flow initialization
it is the validated reference-only solution including its shared offset. A zero-q
static video is not a reconstruction baseline. Final q combined with old modes is
also not the pre-joint baseline.

There is no baseline-snapshot CLI flag. A small experiment-local orchestration
script can publish this state with existing APIs, without a second training run:

1. Read and verify `work/run.json`, its checkpoint/run contract and the checkpoint
   hash. Load the same preparation using `load_prepared`; reapply
   `load_flow_initialization` when present to recover the actual views and scales.
   Construct a separate `RefinementTrainer` with the same config and validate the
   saved trainer state using `load_state_dict`. Do not mutate the running trainer,
   checkpoint or source manifests. Use an isolated process for snapshot publication.
2. Require non-null saved `q0` of shape `[total_frames,K,2]`. It is the unchanged
   anchor, even in a later checkpoint. In the separate publication object, zero
   `controls.delta`, copy q0 rows into q under `torch.no_grad()`, and invalidate
   `baked_fields`. Call `trainer.coordinates()` to undo each sequence's original
   scales: raw q = `view_as_complex(q0) / scales[sequence]`. Do not divide twice.
3. Verify original control displacement/angular fields and baked fields against
   the prepared original-field contract, including donor lever arms and unresolved
   zeros. This is a derived export, not an optimizer resume or additional update.
4. Use `publish_refinement` with the selected prepared metadata, original scene,
   `trainer.field`, `trainer.coordinates()`, `trainer.controls()` and
   `trainer.frozen_fields()` in a NEW `EXP/unrefined` directory. Compute a distinct
   derived baseline identity with the existing identity helper, binding source
   run/checkpoint hash and the `pre_joint_anchor` derivation. Preserve initialization
   provenance; record the baseline derivation in a copied metadata object's
   `training_selection` before publication. Do not inherit final-best step/loss
   claims for this baseline. The v20 wrapper does not mean the field was optimized;
   label this result as original fields plus pre-joint coefficients in the handoff.
5. Reload and validate it through the current result loader. Require identical
   scene, frequency order, views, frames, timestamps, camera and validity bindings
   for the baseline/final comparison. Never edit an old artifact to force a match.

If the user explicitly requests an independently converged fixed-mode baseline,
use `coordinates fit-rgb` on the matching original bank/direct inputs instead,
then materialize it normally. Record its separate optimizer/budget; do not present
it as the pre-joint q anchor or an equal-budget ablation without evidence.

## Materialize, evaluate and export

```sh
modal-gaussians result materialize --scene STATIC --modes EXP/unrefined/mode_bank --coordinates EXP/unrefined/coordinates --output EXP/result_unrefined
modal-gaussians result materialize --scene STATIC --modes EXP/refined/mode_bank --coordinates EXP/refined/coordinates --output EXP/result_refined
modal-gaussians result evaluate --result EXP/result_unrefined --output EXP/evaluation_unrefined --lpips
modal-gaussians result evaluate --result EXP/result_refined --baseline EXP/evaluation_unrefined --output EXP/evaluation_refined --lpips
modal-gaussians result export-video --result EXP/result_unrefined --view VIEW --output EXP/exports/VIEW/unrefined_pair
modal-gaussians result export-video --result EXP/result_refined --view VIEW --output EXP/exports/VIEW/refined_pair
```

Evaluate both using the SAME protocol and LPIPS setting. Omit `--lpips` from both
only when its dependency/weights are unavailable, and report that omission. Default
evaluation covers every supervised frame. PSNR/SSIM/RMSE and optional LPIPS come from
native float renders versus the source PNGs, preserving stabilization-valid support;
never measure the exported MP4s. Deliver `metrics.json`, `per_frame.csv`, manifests
and the final evaluation's `per_frame_delta.csv`. Summarize per-sequence and aggregate
scores, metric directions, and degradations as well as improvements. Original versus
itself is not a useful third metric row. These are fitted-recording metrics.

The stock exporter outputs `comparison.mp4` with target ABOVE reconstruction.
It does not have a three-video or three-way flag. For each sequence, read both
export manifests, verify matched frames/FPS, and derive a new delivery directory
with the three independent videos and one labeled horizontal comparison. Use the
existing FFmpeg resolution (`preprocessing.frames.media_executable`) and ordinary
crop/hstack filters, not a second renderer. These are display encodes; metrics have
already been computed from uncompressed sources.

Let W,H be `video.panel_shape_hw` in width/height order (stored as [H,W]). Crop the
source right padding before adding encoder-required even right/bottom padding.
Use new paths, `-n`, no FPS conversion or frame interpolation:

```sh
ffmpeg -nostdin -n -i UNREFINED_PAIR/comparison.mp4 -an -vf "crop=W:H:0:0:exact=1,pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart DELIVERY/original.mp4
ffmpeg -nostdin -n -i UNREFINED_PAIR/comparison.mp4 -an -vf "crop=W:H:0:H:exact=1,pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart DELIVERY/unrefined.mp4
ffmpeg -nostdin -n -i REFINED_PAIR/comparison.mp4 -an -vf "crop=W:H:0:H:exact=1,pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart DELIVERY/refined.mp4
ffmpeg -nostdin -n -i DELIVERY/original.mp4 -i DELIVERY/unrefined.mp4 -i DELIVERY/refined.mp4 -filter_complex "[0:v]drawtext=text='Original (fitting input)':x=8:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[a];[1:v]drawtext=text='Unrefined modes':x=8:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[b];[2:v]drawtext=text='Refined modes':x=8:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[c];[a][b][c]hstack=inputs=3[v]" -map "[v]" -an -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart DELIVERY/comparison_3way.mp4
```

Replace W/H with integers, not guessed fixed 540p values. The `exact=1` crop option
preserves odd dimensions instead of rounding to the chroma grid. Resolve a local font for drawtext if
needed. Preserve complete labels without resizing the scene pixels; a caption band
is acceptable. Original here is a display encode of the actual fitting input, which
may be stabilized, not the untouched raw camera movie. Record that source meaning,
identities, time grid, panel order, native dimensions, padding, encoder settings and
output hashes in a delivery manifest; retain logs. Assemble in a temporary sibling
and atomically publish the completed delivery directory using existing helpers.
Verify all files with FFprobe and inspect first/middle/last frames.

## Viser readiness and handoff

Reload `EXP/result_unrefined` and `EXP/result_refined` using
`results.artifact.load_modal_result` with its normal source validation. Video export
also exercises actual full SH/displacement/angular rendering. Supply absolute-path
commands with separate writable viewer work directories, for example:

```sh
modal-gaussians viewer --input EXP/result_refined --work-dir EXP/viewer_refined --host 127.0.0.1 --port 8080
modal-gaussians viewer --input EXP/result_unrefined --work-dir EXP/viewer_unrefined --host 127.0.0.1 --port 8081
```

Commands are handoff instructions, not automatic service launches. The default
viewer includes Spectrum: validate its inherited observation/FFT dependencies.
For deliberately 3D-only playback, provide `--no-spectrum` and label that scope;
do not hide missing Spectrum inputs behind an unlabeled fallback. Refined modal
observations remain inherited, not newly measured supervision. Return concrete
result/video/evaluation paths, source/config identities, baseline definition,
actual/published steps, measured timings and missing outputs. Viser-ready does not
claim a running server, visual acceptance or scientific approval.
