# Current pipeline commands

These are recipes, not a script to run automatically. Replace uppercase paths,
labels, timing and frequency choices with inspected scene inputs. All outputs
belong under `scene_library/<scene>/experiments/<new_name>/` or a matching scene
cache. The [skill](../SKILL.md) and [baseline](../../../BASELINE.md) define scope.
Run `modal-gaussians GROUP COMMAND --help` for additional options.

## Pixels, cameras and references

`prepare gui --root-dir PREPARATION` opens the optional frame/mask tool. Reuse
existing extracted pixels and masks when available. First register the raw sweep
and raw fixed-view reference images, then build static geometry:

```sh
modal-gaussians colmap prepare --frames SWEEP_IMAGES --frame-masks SWEEP_MASKS --reference view1 REF_IMAGE1 REF_MASK1 --reference view2 REF_IMAGE2 REF_MASK2 --sample-stride STRIDE --output COLMAP_INPUT
modal-gaussians static prepare-depth --input COLMAP_INPUT --model DA3_LOCAL_MODEL --python DA3_PYTHON --output DEPTH
modal-gaussians static train --input COLMAP_INPUT --depth DEPTH --depth-weight 0.01 --depth-until-step 3000 --iterations 3000 --batch-size 4 --work-dir STATIC_WORK --output STATIC
modal-gaussians prepare reference --images IMAGES --masks MASKS --fps FPS --reference-frame FRAME --scene STATIC --view view1 --output REFERENCE
```

The static pass is coarse geometry: 3,000 updates at batch 4, progressively activated SH up
to degree 3, and optional posed DA3 depth L2. `--iterations` replaces `--epochs`; the
author-aligned numerical defaults are in BASELINE. Use new static/work paths:
old direct-RGB bundles/checkpoints cannot be resumed into SH training.
Static training uses RGB L1 + 0.2 DSSIM plus depth L2 when supplied, without mask loss or a
`--mask-weight` option. Masks remain required for initial point classification.
Use a new work directory for earlier static checkpoints (current resume v4).

For an explicitly requested wall-clock experiment, keep the intended maximum
iteration schedule and add `--time-budget-seconds SECONDS`, optionally
`--time-milestones-seconds T1 T2`. This requires fresh work (no `--resume`), saves
at complete update/density boundaries and reports actual time/overshoot in
`WORK/training_timing.json`. It does not automatically evaluate or export video.
See [the paired benchmark protocol](../../../docs/adaptive-sfm-phase2-20260926.md).

For a deliberately RGB-only run omit `--depth`, `--depth-weight` and the depth
preparation stage. Missing/incompatible requested depth is an error, never a fallback.
DA3 requires a separate CUDA Python environment because upstream pins NumPy < 2
while this project needs NumPy >= 2. Install the official
[DA3 package](https://github.com/ByteDance-Seed/Depth-Anything-3#installation) there
and obtain the `depth-anything/DA3-LARGE-1.1` pretrained snapshot in a shared tool path.
`DA3_LOCAL_MODEL` must contain `config.json` and safetensors; `DA3_PYTHON` is that
environment's executable. No installer/downloader is invoked by these commands.
Use a multi-view model; do not substitute the mono/metric-only model.
Preparation defaults: `--process-res 1008 --chunk-size 16 --confidence-percentile 10`.
Each chunk adds up to three spatially spread registered cameras for pose-scale
conditioning; all original targets are retained once. It does not use motion modes,
flow, coefficients or a trained static scene, and never starts static training.
Preparation and real RGB-D training must only run within the user's authorized scope.

Stabilization is the default: background PnP poses plus static-depth reprojection.
The raw reference PNG hash/resolution must match the registered camera. Only when
the user explicitly confirms tripod capture, replace `--scene STATIC --view view1`
with `--tripod`. Never use tripod as a workaround for missing geometry. The moving
sweep stays on its own cameras. Old stabilized PNGs/MP4 previews cannot be relabeled
as new stabilized references. Optional `--config` supplies stabilization settings.

Use the current SIMPLE_RADIAL projection. Before downstream motion preparation,
ask the user to save a 3D subject box on this static scene, then apply it:

```sh
modal-gaussians viewer --scene STATIC --select-subject --work-dir SELECTION_WORK
modal-gaussians static apply-selection --scene STATIC --selection SAVED_NPZ --output SUBJECT_SCENE
```

Use `Add box`, `Active box`, and `Remove active box` to edit multiple boxes; only
their union is selected. `Save selection` writes all boxes in selection v3.
The applied union is the authoritative motion subject. Use SUBJECT_SCENE for the
subsequent stages below, not the automatic XMem-derived partition. Masks still
serve bootstrap and background stabilization. Wait for the saved manual selection
if missing; do not substitute an automatic partition. Never reuse Gaussian
indices or scene-bound descendants after changing the partition.

## Motion reference, flow and FFT

The motion reference can differ from the geometry reference:

```sh
modal-gaussians flow select-reference --scene STATIC --view-label view1 --reference REFERENCE1 --output SELECTION1
modal-gaussians flow compute --images IMAGES1 --reference REFERENCE1 --reference-selection SELECTION1 --output FLOW1
modal-gaussians spectrum build --view view1 FLOW1 --view view2 FLOW2 --scene STATIC --nfft NFFT --output FFT
modal-gaussians spectrum select --input FFT --count COUNT --max-frequency MAX_HZ --output BIN_SELECTION
modal-gaussians spectrum export --input FFT --selection BIN_SELECTION/selection.json --output MODAL_IMAGES
```

Repeat selection/flow for every view. All views must share FPS and NFFT, and NFFT
must cover every video. Preserve the actual bin list for exact reproduction.
`--max-frequency` is optional; selection otherwise spans positive bins to Nyquist.
A requested `spectrum viewer --input FFT --work-dir VIEWER_WORK` can save manually
selected bins. Export copies cached complex slices; it never recomputes DFT.

## Reusable geometry and frequency training

Freeze observations/KNN geometry directly from the static scene and references.
Per-view relative depth tolerances are explicit calibration values; do not invent
them from another scene. Read the resulting `manifest.json` for `geometry_graph`.

```sh
modal-gaussians motion prepare-neural --scene STATIC --view view1 REFERENCE1 TOLERANCE1 --view view2 REFERENCE2 TOLERANCE2 --config configs/neural_component_field.json --cache-dir SCENE_CACHE --output PREPARED
modal-gaussians motion batch-neural --modal-images MODAL_IMAGES --prepared PREPARED --geometry-graph GEOMETRY_GRAPH --config configs/neural_component_field.json --output BATCH --cpu-workers 3 --gpu-workers 2 --threads-per-worker 2
```

The batch prepares each frequency, constructs its soft graph, computes GPU control
weights, then trains the GNN. `--stage weights` stops before training. A completed
mode batch writes `index.json`, usable directly by `viewer --input BATCH/index.json`
and coefficient preparation. The viewer still requires an explicit work directory.
No viewer or temporal fitting starts automatically.
Add `--no-spectrum` to `viewer` for 3D playback without loading the optional FFT
panel. This does not relax source validation when Spectrum is enabled.

The same stages can be called individually for one exported bin:

```sh
modal-gaussians motion prepare-selected-modal --prepared PREPARED --view view1 BIN/view1 --view view2 BIN/view2 --frequency-hz HZ --output FREQUENCY_PREPARED
modal-gaussians graph build-modal-similarity --prepared FREQUENCY_PREPARED --geometry-graph GEOMETRY_GRAPH --view view1 BIN/view1 --view view2 BIN/view2 --frequency HZ --output SOFT_GRAPH
modal-gaussians motion prepare-control-weights --prepared FREQUENCY_PREPARED --geometry-graph SOFT_GRAPH --config configs/neural_component_field.json --frequency-hz HZ --output WEIGHTS_READY.json
modal-gaussians motion iterate-neural --prepared FREQUENCY_PREPARED --geometry-graph SOFT_GRAPH --config configs/neural_component_field.json --frequency-hz HZ --output EXPERIMENT
```

Mode output is bound by `EXPERIMENT/status.json`. Shared control geometry is created
or reused automatically. Never use a soft graph from another frequency. To proceed
to video reconstruction, follow [COEFFICIENT_FITTING.md](../../../COEFFICIENT_FITTING.md)
for the full-pipeline endpoint. See [recovery](validation-recovery.md) for resume semantics.

## Optional fixed-scene motion refinement (separate authorization)

For an explicitly requested coefficient-only sweep fit with frozen refined modes:

```sh
modal-gaussians coordinates fit-sweep --scene STATIC --modes REFINED/mode_bank --scale-source REFINED/coordinates --metadata SWEEP_METADATA --config RGB_CONFIG --output EXP/sweep_coordinates --fps 30
modal-gaussians result materialize --scene STATIC --modes REFINED/mode_bank --coordinates EXP/sweep_coordinates --output EXP/result
```

The scale source must be a matching single fixed-view RGB/refined fit. Sweep q
starts independently; no motion refinement runs. Use zero initial/final anchor
weights in RGB_CONFIG for RGB-only fitting.

```sh
modal-gaussians coordinates prepare-refinement --scene STATIC --modes BANK20 --view view1 --sweep-metadata SWEEP_METADATA --reference MOTION_REFERENCE --output EXP/prepared
modal-gaussians coordinates refine-motion --prepared EXP/prepared --config configs/motion_refinement.json --work-dir EXP/work --output EXP/refined
modal-gaussians result materialize --scene STATIC --modes EXP/refined/mode_bank --coordinates EXP/refined/coordinates --output EXP/result
```

Preparation does not train. The trainer freezes the complete Gaussian scene, starts
q at zero with ten in-run warmup passes, then refines control fields/q. Omit sweep
metadata for fixed views only. It consumes no pre-fitted RGB coordinates and writes
no new scene. Current sources can omit `--reference`; v16 needs the explicit
validated importer. A full-pipeline request includes fitting/refinement, evaluation and export; Viser
is prepared but not launched unless requested. See COEFFICIENT_FITTING.md for all contracts.

For an explicitly requested flow-initialized run, add `--flow-initialization
DIAGNOSTIC_ROOT` to `refine-motion`. It uses only the reference-only diagnostic's
checked fixed-view prefix and skips warmup, preserving the included shared offset
and pixel-pair normalization. Use a fresh work/output directory; resume binds the
same diagnostic source. This does not change the default zero-start recipe.

## Optional frozen-motion sweep scene refinement

Only append this stage when explicitly requested; the default endpoint remains
joint motion/coefficient refinement plus Viser-ready reconstruction delivery.

```sh
modal-gaussians static refine-scene --scene STATIC --modes REFINED/mode_bank --coordinates SWEEP_COORDINATES --motion fitted --config configs/scene_refinement.json --work-dir EXP/work --output EXP/published
modal-gaussians result materialize --scene EXP/published/scene --modes EXP/published/mode_bank --coordinates EXP/published/coordinates --output EXP/result
modal-gaussians result evaluate --result EXP/result --output EXP/evaluation
modal-gaussians result export-video --result EXP/result --view sweep --output EXP/video
modal-gaussians viewer --input EXP/result --work-dir EXP/viewer
```

Use `--motion zero` with separate work/output for the q=0 ablation. Add `--resume`
only for a matching scene-refinement checkpoint, never the old static work. Train
only on coordinate-bound sweep frames; original modal observation views are inherited.
The viewer command is a handoff, not an automatic service launch. Check actual CLI
help for export arguments. Keep the original view1 q for cross-view validation,
without refitting or using that loss to select the scene checkpoint.

The explicitly requested A/B run can be composed with
`python tools/sweep_scene_experiment.py --recipe EXP/recipe.json --stage STAGE`,
where STAGE is `smoke`, `train`, `deliver` or `report`. The recipe records absolute
scene, modes, sweep coordinates, view1 coordinates and experiment paths, plus the
scene-refinement config object. Smoke runs each arm to step 25; train resumes those
same checkpoints to the configured budget. Deliver binds results, measures both
initial baselines and both outputs, transfers the original view1 q for its check,
and exports the four independent videos and four-column comparison. Report writes
paired metric CSVs, Gaussian-count CSVs, timing/density summaries and RUN_REPORT.md.
Each stage must succeed before the next one is called. No service starts.

If the user authorizes early stopping during an existing run, record an explicit
`EXP/early_stopping_policy.json` with `patience`, `minimum_step` and
`relative_improvement` (for example 3 evaluations, step 2000, and 0.002).
Use only full-sweep evaluations. Once the policy is met, stop the owned trainer
after its complete evaluation checkpoint, then call `--stage finish-early`.
This stage requires an unlocked work directory and the original unchanged
checkpoint contract; it publishes the saved best state. It does not stop a live
process or fabricate resumed optimizer state. Continue with `deliver` and `report`.
Record unequal A/B budgets and any discarded updates after the retained checkpoint.
For a user-requested immediate stop, set policy `mode` to
`user_requested_checkpoint_stop` and record the reason. The stage then performs
only a final full-frame evaluation of the retained checkpoint and best-state
publication, without further Gaussian updates. The original process's allocator
peak is unavailable after an external stop and is reported as unrecorded.
