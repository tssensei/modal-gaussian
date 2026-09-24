# Cleanup and required rebuilds — 2026-09-23

## Remove static mask supervision — 2026-09-24

Static optimization now uses only `L1_RGB + 0.2 * (1 - SSIM)`. The trimmed mask
loss, erosion/quantile/weight configuration, `--mask-weight`, per-step target mask
loading/caching, membership rendering and mask-loss logs have been deleted. Masks
are still required for initial point classification and other stages; valid-pixel
support in stabilization, modal learning and coefficient fitting is unchanged.

This changes the method, not just performance. Static resume is v3 and training
summary is v2; earlier checkpoints cannot resume. Use new static work/output
directories and bind any newly trained scene's downstream products to its new
identity. Existing published scene tensors remain valid v5/v6/v7 sources; no old
manifest is rewritten, no catalog redirected and no experiment automatically run.
Batch 4, 3,000 updates, SH/density/LR settings and pixel/grid caching are retained.

## Static execution optimization — 2026-09-24 (before mask-loss removal)

After commit `5f53bdb`, static training adds a bounded CPU pixel/mask cache and
requests RGB/membership without unused expected depth. The shared radial renderer
reuses device sampling grids including batch-dependent padding. Batch 4, 3,000
updates, SH schedule, RGB/mask losses, sampling, Adam and density rules are unchanged.

- No artifact format changes: existing compatible SH scenes, references, modes and
  coordinates remain readable; this optimization alone does not mandate rebuilding
  published artifacts. Existing raw frames, masks and COLMAP inputs remain reusable.
- Static checkpoint identity now includes the radial-render implementation in
  addition to training/scene/density. Use new work directories for prior checkpoints.
  Refinement also hashes the shared scene/rendering modules and cannot resume an
  older implementation. Normal implementation-keyed cache misses must not be bypassed.
- When deliberately retraining a new static scene, publish into a new directory;
  downstream products must bind that new scene, as usual. No catalog or real
  experiment was changed or run for this optimization.

Synthetic benchmark: RTX 5090, batch 4, 960x540, 12,288 Gaussians, SH degree 2,
four synthetic PNGs, three warm-up and eight measured updates, synchronized stage
timings. Median update time was 61.63 ms before versus 23.50 ms after; loading
24.40 -> 4.90 ms, rendering 19.81 -> 4.53 ms, peak allocated GPU memory
803.63 -> 690.62 MiB. This is a warm-cache synthetic measurement, not a Bush
speedup estimate; checkpoints/density topology changes are outside that timing.
Loss, all Gaussian gradients, screen-gradient statistics, first Adam update and
optimizer state matched at `rtol=2e-5, atol=1e-7`. Pixel and cached-grid values
match exactly. Local reproducible checks/results are in ignored
`tests/benchmark_static_training.py`, `tests/static_perf_before.json`,
`tests/static_perf_after.json`, and `tests/test_static_performance.py`.
These saved before/after timings include the former mask loss. They do not measure
the subsequent RGB-only objective above; rerunning the local benchmark now uses
the current RGB-only trainer and must write a new result filename.

## SH coarse bootstrap — 2026-09-24

This supersedes the direct-RGB scene versions mentioned in historical sections
below. Static training now uses 3,000 updates at batch 4 by default (`--iterations`,
not `--epochs`), author-aligned coarse learning rates/density control and world-frame
SH up to degree 3. Depth loss is still disabled. See BASELINE for the complete recipe.
The batch-size change changes the resolved training contract; batch-1 checkpoints
cannot resume under the new default. No training was launched by this adjustment.

| Stage / artifact | Required action |
| --- | --- |
| Raw recordings, native frames/masks, COLMAP | Reusable if resolution, masks and camera contract are unchanged. The planned 540p run still needs matching 540p inputs/cameras. |
| Static scene and resume | New static v5 and resume v2; old v2/v3/v4 scenes and v1 checkpoints are not loaded or converted. Use new work/output directories. |
| Subject partition | Recreate for new Gaussian order, producing v6; old selections bind the old scene. |
| Stabilization / render-matched references / flow / FFT | Rebuild: static depth, rendered reference and scene identities change. Raw frame timing is unchanged. |
| Geometry/KNN, modal preparation/weights, modes, mode bank | Rebuild against new Gaussian identities/order and observations. |
| Coefficients, refinement reference/preparation/checkpoint | Refit/reprepare in new directories; SH and source identities change. No cross-implementation resume. |
| Refinement publication | Scene v7 with SH, modes v19 and current refined coordinates; identities regenerated. |
| Results, evaluation, video and viewer projection caches | Recreate from new ancestors when explicitly requested; preserve frozen historical baselines. |

No published source, catalog entry or experiment was modified by this code change.
The fixed-mode fitter freezes SH; shared refinement optimizes foreground SH only.
Synthetic checks cover view-dependent colors and position gradients, static versus
deformed rendering, SH row/moment preservation and exact-update recovery. A real
540p run and quality/timing comparison remain unperformed.

## Next experiment decision — 2026-09-24 (not started)

The user accepted the visual quality of the Bush 540p input previews. The next
experiment should rerun the full pipeline from the beginning with 960x540 inputs
for sweep and fixed-view recordings, recording each stage's elapsed time and the
total runtime. Keep the current 30 FPS reconstruction convention. The separately
accepted stabilization change below also applies to that future run.

Preview sources/results: `scene_library/bush/experiments/input_resolution_preview_20260924_001/`.
These MP4s are visual previews, not replacement training inputs: prepare fresh
540p inputs from original sources without an extra lossy preview-video decode.
Create new resolution-bound artifacts/caches and preserve all existing inputs,
experiments and frozen baselines. Record reused inputs and excluded work when
reporting timing. This is a pending plan, not an executed or validated 540p run.
Do not start preparation, training or downstream stages until explicitly requested.

## Stabilization replacement — 2026-09-24 (implemented; real rerun pending)

Use the [new stabilization baseline](BASELINE.md#video-stabilization-baseline--accepted-2026-09-24)
for future fixed-view recordings: fixed-map background PnP poses plus static-depth
reprojection, with sparse background depth completion. The user accepted the Bush
view1 preview in `pose_stabilized_view1_540p_20260924_001`; its inputs and the
`background_pose_view1_540p_20260924_001` pose experiment remain preserved.
The old homography implementation and `--stabilize` flag are removed. Default
preparation requires `--scene STATIC --view LABEL`; only explicit tripod recordings
use `--tripod`. No catalog has been redirected.

- Required ordering: raw frames/masks -> sweep COLMAP including raw fixed-view
  reference images -> initial static 3DGS -> per-recording poses and target depth
  -> stabilized lossless frames/masks/validity and fixed target camera -> motion
  references, flow, FFT and downstream motion/reconstruction stages. Keep the
  moving sweep on its own calibrated per-frame cameras; do not stabilize it into
  a fixed-view modal observation. Report reused geometry explicitly in timing.
- New references bind poses, intrinsics/distortion, depth and its completion,
  source frame hashes, target camera, masks and validity into the new cache
  identity. Do not train on black missing pixels as if they were observations;
  define valid support consistently for flow, RGB losses and evaluation.
- Changed stabilized pixels/camera bindings require new sequence references,
  flow, FFT/modal images, modes/projections, coefficients and result/evaluation
  products. Reuse static/control geometry only where its own contracts match.
  Preserve old results as historical baselines; never relabel their inputs.
- New contracts: reference/stabilized sequence v2, SEA-RAFT v3, shared spectrum
  and selected-frequency export v2, neural preparation v3, fixed RGB coordinates v2, refinement
  preparation/coordinates v3, evaluation v2. Rebuild into new outputs; no manifest
  relabeling or old-format compatibility reader is provided. Geometry preparation
  now stores valid masks/sample validity, so rebuild it against new references.
- Changed training code identity invalidates old checkpoints; use new work
  directories. Synthetic checks do not constitute real-scene validation. The full
  540p rerun remains stopped until explicitly requested.

## Viewer-only update

Viewer-only update (2026-09-24): `viewer --no-spectrum` explicitly skips optional
FFT/Spectrum loading for 3D playback. Default strict validation is unchanged; no
legacy reader is added. No scientific artifact, projection identity or training
implementation changes, and no downstream rebuild is required for this option.

This cleanup deliberately breaks old module imports, commands and artifact schemas.
It preserves the current component-field numerical recipe and fixed-mode RGB fitting.
No scene-library artifact, raw video, model, cache or catalog was changed. No real
experiment was run to validate the cleaned pipeline. Synthetic checks are development checks.

## Joint-refinement addition

- Current static v2/v3 scenes, v18 single-frequency models, v17 mode banks and
  fixed-mode direct/RGB coordinates remain input formats. The earlier cleanup's
  Bush/Corn rebuild requirements below still apply; no catalog was redirected.
- New refinement requires models with one common reference graph/control layout
  and RGB fits for selected bank recordings (`--view`; default all). Preparation
  lists missing videos, replays each saved network once and checks field equivalence.
  It never automatically fits missing coordinates or retrains upstream modes.
- Density changes create scene v4, completed modes v19 and refined coordinates
  v3 (including valid image support). Existing linear designs cannot be relabeled for the new Gaussian order.
  Refined playback/export use the final baked fields, without rebuilding ridge designs.
- `modal_result` is now v3. Rematerialize existing supported scene/mode/coordinate
  combinations into new directories; no result-v1/v2 reader is retained. Existing MP4s remain historical
  files; new exports are explicit and require new directories.
- Refinement does not change camera/input/reference/FFT contracts and therefore
  does not itself require new flow or FFT. Original scene bindings remain intact
  as parent provenance. Viewer projection caches bind the new scene/mode identity.
- Shared density/rendering code revisions can invalidate exact-code work/cache
  contracts, even where fixed-mode mathematics is unchanged. No real rebuild or
  refinement has been run as part of this implementation.

## What was removed

- Farneback/legacy flow analysis, selected-frequency repeated DFT and greedy selection.
- Rigid mode completion; surface, pointwise, guarded and refinement training branches.
- CPU alpha fitting and CPU soft-propagation alternatives.
- Physics/oscillator coordinate fitting and automatic post-training coordinate stages.
- Old model readers, network replay just to recover runtime rotation, preview artifacts
  and old viewer aliases. Models now save displacement, angular fields and control displacement.
- Old migration/benchmark scripts, historical recipes and tests dedicated to removed branches.

Modules moved into `preprocessing`, `geometry`, `flow`, `spectrum`, `motion`,
`coordinates`, `results`, `vis` and `common`; no import shims were added.
Git history contains the removed code. Restoring it would require reverting or
cherry-picking the relevant changes with their dependencies.

## Artifact boundary changes

| Artifact | Current contract | Treatment of existing data |
| --- | --- | --- |
| Raw RGB/masks; existing stabilized PNGs | pixel files unchanged | Reuse; do not re-extract/re-stabilize solely for cleanup. |
| Sequence reference | `sequence_reference` v1 | New lightweight metadata/mask union; replaces `flow_analysis` reference artifacts. |
| Static scene | v2/v3 with SIMPLE_RADIAL projection | Retain supported scenes; pinhole-only static bundles require retraining. |
| Motion reference selection | v1 bound to new reference identity | Select again for the current static scene/grid. |
| SEA-RAFT flow | v3 with selection and valid-support binding | Recompute into new directories. |
| Shared FFT and selected modal images | current transform + new flow/reference identities | Rebuild after flow; export exact bins. |
| Geometry/prepared observations | `neural_prepared` v3; KNN cache v2 | Rebuild directly from static scene and references. No old observed graph is required. |
| Per-frequency soft graph | `modal_similarity_graph` v2 | Rebuild for the matching prepared frequency. |
| Controls/alpha/work checkpoints | current input/code contracts | Rebuild; pre-cleanup checkpoints cannot resume into this implementation. |
| Single-frequency completed mode | v18, baked angular/control fields | Retrain; readers reject the old schemas. |
| Mode bank, design, direct/RGB coordinates, bound result/video | existing formats with new source identities | Rebuild downstream of the new modes. |

The stabilization replacement supersedes reuse of old stabilized PNGs as new
references: prepare from raw registered inputs using the current method. Preserve
historical stabilized pixels and their baselines without relabeling them.

## Bush and Corn

Read-only manifest inspection confirmed the following catalog targets. Some newer
experiment subdirectories denied reads; their complete contents/quality were not audited.

| Scene | Static target | Required work |
| --- | --- | --- |
| Bush | `bush/geometry/static`: v3, full-scene visible-mask partition, distortion applied | Static scene can be reused. New references -> selected motion references -> flow -> FFT -> geometry/observations -> 20 modes -> coefficient fit/result/export. |
| Corn | `corn/geometry/static`: v3 manual selection, distortion **not** applied | Retrain a distortion-aware static scene, redo its subject selection, then rebuild downstream references/flow/FFT/geometry/modes. Existing COLMAP calibration/raw inputs can be reused if their input contract still matches. |

Existing Bush targets include `uniform20_0to5_newref_20260920/batch` and
`coefficient20_0to5_newref_20260920/`. Corn's target is
`uniform10_0to2p5_newref_20260921/batch`. These remain on disk as historical results;
the cleaned code must not treat their existence as reusable current output.
Choose fresh experiment names and update catalog pointers only after the new runs finish.

Preserve the intended FFT grids unless changing the experiment: Bush 30 FPS,
NFFT=1250, delta-f=0.024 Hz; Corn 20 FPS, NFFT=1600, delta-f=0.0125 Hz.
The historical Bush selection included 0.744 Hz. Inspect the saved bin list if
exact reproduction is required; a new uniform selection need not reproduce that list.

`motion prepare-neural --view LABEL REFERENCE RELATIVE_DEPTH_TOLERANCE` now makes
visibility calibration explicit. Readable ancestor graph manifests contained:

| Scene | view1 | view2 | view3 |
| --- | --- | --- | --- |
| Bush | 0.01824313479256816 | 0.022121385373454542 | 0.014344444496370852 |
| Corn | 0.010864540798950474 | 0.00919846558852587 | — |

Sources: `bush/geometry/ancestors/bush_neural_dense_controls_001/observed_graph/manifest.json`
and `corn/geometry/ancestors/corn_static_20fps_001/observed_graph/manifest.json`,
`thresholds.views[].endpoint_gap.threshold`. They are recorded calibration values,
not new measurements. Reassess them when geometry/projection changes, especially Corn.
The runtime does not read these old graphs.

The old Bush 40-mode experiment was previously deleted. Its favorable user review
is not an acceptance result for the 20-mode data or this cleanup.

## Historical reconstruction baseline verified on 2026-09-23

The previously unreadable Bush `coefficient20_0to5_newref_20260920` inputs were
checked explicitly for the requested baseline. Its static scene v3, baked mode
bank v17, rendered design v2, direct coordinates v2 and view1 RGB coordinates v1
pass the current result loader. The scene tensors, bank files, coordinate array
and all 1170 actual input PNGs passed checksum validation during evaluation.

They were linked into a fresh result v2 at
`bush/experiments/baseline_pre_refinement_20260923_001/result`, with full-frame
PSNR/SSIM/RMSE/LPIPS in the sibling `evaluation/`. See [baseline values](BASELINE.md#frozen-historical-bush-baseline-2026-09-23).
Original artifacts and catalog pointers were not changed. Rematerializing this
baked bank did not reload its old GNNs or require flow/FFT recomputation.

This is a narrow replay/evaluation reuse boundary. Source single-frequency models
remain v16; general model loading still rejects them. The explicit
`tools/import_refinement_reference.py` now permits a verified fixed-reference
import for this experiment, without upstream retraining. `prepare-refinement`
requires that imported `--reference` for v16 and can select only `--view view1`.
No upstream rebuild, new RGB fit or refinement ran for the historical measurement.

## Alternating refinement and 30 FPS sweep (2026-09-24)

The sole `refine-scene` schedule is now two rounds of sparse geometry/coefficient
updates followed by exhaustive frozen-geometry coefficient passes, at scale 1.0.
This changes sampling, budgets and learning-rate clocks. It is a method change,
not merely a faster execution of the old all-frame joint optimizer. Old
`scales`/`epochs_per_scale` refinement settings and old trainer checkpoints are
rejected; the fixed-mode RGB fitter retains its own multiscale settings.

| Input/output | Action for the new experiment |
| --- | --- |
| Original scene, 20-mode bank, imported motion reference | Reuse after normal identity/checksum checks; path interpolation is unchanged. |
| Original view1 RGB initialization | Reuse all 1170 rows at 30 FPS, preserving offsets/scales. |
| Existing sweep RGB `[724,20]` at 60 FPS | `downsample-sweep --fps 30` into a new path: rows 0,2,...,722; no optimization. For new fits, `fit-sweep --fps 30` selects before fitting. |
| Old preparation `[1894,20]` | Regenerate using the 30 FPS sweep: `[1532,20]`. Keep original three-view modal sources. |
| Historical frozen view1/60 FPS sweep baselines | Preserve. Materialize/evaluate the same view1 plus 30 FPS sweep into a new baseline; do not compare different frame sets. |
| Old work/checkpoints | Preserve historical logs/source snapshots; never resume across changed implementation/config. Start new work. |
| Final scene/modes/coordinates/results | New identities; formats remain scene v4, modes v19, refined coordinates v2, result v3. |
| Flow, FFT, original motion references and cameras/PNGs | No recomputation or relabeling required solely for this change. |

The verified historical experiment `refinement20_view1_sweep_20260924_001/` contains
`motion_reference/`, `sweep_coordinates/`, `baseline_result/`, `baseline_evaluation/`
and `prepared/`. Its saved verification reports document reference equivalence,
source-file preservation, camera bindings and baseline metrics. The historical
training attempts were paused (one update, then 23 updates in `work_002/`); no final
refined scene was published. Keep their logs and frozen baselines. New 30 FPS
selection/preparation/baseline/training have not been run by this implementation.

The training implementation identity includes refinement, reference query,
rendering, sequence bindings, publication and shared density kernels. Checkpoints
store separate geometry/total clocks, round/phase, selections, all q/Gaussian Adam
states, topology, density statistics and RNG. Transient GPU basis tensors are
rebuilt on resume. Schema readers are not extended to accept old trainer states.

## Fixed KNN shape bound (2026-09-24)

Joint refinement now accepts canonical positions only within their permanent
root's original incident-segment neighborhood, as well as the existing motion
support. Radius is `shape_radius_fraction` (default 0.25) times the original
median incident edge length; isolated roots cannot move. Child roots/radii and
per-frequency propagation/donor data remain inherited and immutable. No graph
rebuild or extra loss is introduced. This is a method change, not a numerically
equivalent optimization of the unconstrained geometry updates.

Original scenes, mode banks, imported references, 30 FPS initial coefficients and
matching prepared v2 inputs can be reused after normal validation. This constraint
does not itself require new flow, FFT, motion-reference import, preparation or
baseline evaluation. The separate 60-to-30 FPS migration requirements above still
apply. Preserve old baselines and work; start a **new work/output directory**.
The new setting and changed query/trainer/publication code alter the run identity,
so checkpoints from before this constraint cannot resume. Output formats remain
scene v4, modes v19, refined coordinates v2 and result v3.

No real training, fitting, evaluation, export or viewer is started by this change.
The performance measurements below predate the shape bound and are not timings
of the constrained implementation.

Validation: all **203** local synthetic tests passed. New checks cover finite
segments, isolated nodes, frequency independence, CPU/CUDA and repeated roots,
joint support/shape rollback and Adam clearing, rejected/accepted splits, cloned
field amplitudes, inherited roots through culling, and invalid checkpoint/output
rejection. Existing checks cover split/clone recovery, Gaussian freeze during
coefficient passes and the complete publication/render/evaluation/export chain.
Tests remain Git-ignored (`tests/graph_shape_full_suite.log`); these checks do not
establish real-scene shape quality.

## Uncommitted-change audit for the alternating-pipeline implementation

Audit scope is the worktree delta from HEAD, including untracked production files.
No unrelated production change was found to restore. The following groups are the
necessary dependency closure of the new pipeline; this table does not claim they
were all introduced in this implementation turn.

| Files retained | Why required |
| --- | --- |
| `coordinates/reference.py`, `motion/reference_field.py`, `tools/import_refinement_reference.py` | Validate/import the existing 20-mode reference; fixed local interpolation, shared query and frozen-basis baking. |
| `coordinates/refinement.py`, `coordinates/refinement_artifacts.py`, `configs/scene_refinement.json` | Alternating training, complete recovery and atomic derived-artifact publication. |
| `coordinates/sequences.py`, `coordinates/sweep.py` | Actual FPS selection and authoritative per-frame camera/clock bindings. |
| `coordinates/fitting.py`, `rendering.py`, `rgb.py` | Shared fixed-RGB numerical path, moving-camera fitting, GPU-resident basis and immutable PNG loading. |
| `coordinates/preparation.py`, `motion/common/completed_modes.py` | Bank checksum/source validation and baked v19 loading. |
| `geometry/density.py`, `geometry/training.py`, `geometry/scene.py` | One density/Adam-row implementation, dynamic screen gradients and derived static scenes. |
| `motion/common/projection.py` | Preserve visible-subject observation policy for derived scenes. |
| `results/artifact.py`, `results/video.py`, `results/evaluation.py`, `pyproject.toml` | Bind independent sequences, correct FPS/cameras, frozen-baseline metrics and optional LPIPS. |
| `vis/inputs.py`, `projections.py`, `spectrum.py`, `viewer.py` | Consume new geometry/baked fields; keep modal observation views distinct from sweep playback. |
| `cli.py` | Explicit independent commands; no implicit downstream execution. |
| `AGENTS.md`, `README.md`, `BASELINE.md`, `SCENE_STORAGE.md`, `COEFFICIENT_FITTING.md`, `REBUILD.md` | Ownership, single supported recipe, contracts and rebuild instructions. |

Removed/replaced: all-frame joint scheduling; refinement scale loops and obsolete
config fields; old checkpoint interpretation; repeated final field queries; root
documentation recommending obsolete training budgets. No compatibility schedule,
switch or old-training loader remains. The authorized v16 importer is the only
legacy boundary and remains necessary for the current 20-mode experiment.

Local Git-ignored snapshots `_refinement_before_shared_query.py`,
`_reference_field_before_shared_query.py`, `_reference_field_before_performance.py`
and obsolete diagnostics `profile_refinement_query.py`, `diagnose_refinement_cost.py`,
`diagnose_reference_precision.py`, `benchmark_refinement_update.py` and
`benchmark_shared_refinement_query.*` are removed. Independent mathematical oracles
replace snapshot-based tests; `benchmark_alternating_refinement.py` measures the
current two phases. Historical experiment snapshots are preserved. Tests stay
Git-ignored; no real scene or catalog is modified by cleanup.

Verification: all 197 local synthetic regression tests passed, including exact
Gaussian/Adam freeze, phase-boundary recovery, frame subsetting, fixed RGB fitting,
static density operations and publication/evaluation/video consumers. CLI help,
Python compilation and `git diff --check` passed. A warmed CUDA benchmark with
16,384 live points, three reference nodes, 20 modes and 256x256 targets measured
median geometry/coefficient steps of 0.253/0.0072 seconds, peak allocated memory
55.7/76.8 MiB, and a separate coefficient-phase bake of 0.177 seconds. The geometry
step renders two frames; the coefficient step renders one. Cached coefficient
steps performed no graph query/backward. See local
`tests/benchmark_alternating_refinement.json`; these are not Bush runtime or quality
measurements. Real 30 FPS artifacts and training remain pending.
