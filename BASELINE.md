# Current numerical recipe

## Accepted coefficient visual baseline — 2026-09-25

The user selected **reference-only flow coordinates + one shared reference-frame
RGB pose offset** (bottom-left panel) as the new visual baseline. In the user's
review, both flow-based reconstructions looked better than the earlier zero-start
per-frame RGB fit; adding adjacent-flow constraints gave no obvious visual benefit.
Use the simpler reference-only method as the comparison target for subsequent
coefficient experiments. This acceptance covers **Bush view1 frames 0–299 only**:
960x540, 30 FPS, original fixed 20-mode bank and unchanged Gaussian scene.

- Experiment: `scene_library/bush/experiments/flow_coordinates540_view1_10s_20260925_001`.
- Selected coordinates: `solution/reference_only.npy`, complex64 [300,20], already
  containing the shared offset. **Do not add `reference_offset.npy` again.**
- Coordinate SHA-256: `ab8128c7874a5cd01016b6314d1ee90bfcfb1aa9d516c1e2504d49a7f3d5dcba`.
- Method: reference frame index 853 (`00854`), projected-pixel RMS normalization,
  ridge 1e-4, no adjacent-flow term, no temporal centering or per-frame RGB refinement.
  Shared offset uses 200 full-resolution RGB Adam updates, LR .01 -> .001,
  keeping the best loss including zero initialization.
- Visual evidence: `comparison/comparison.mp4`, bottom-left panel. Preserve all
  other panels, coefficients and diagnostics as experimental comparisons.
- Valid-image frame-mean metrics in `evaluation_001/`: PSNR **22.310178 dB**,
  SSIM **0.753528**, RMSE **0.077343**, LPIPS-Alex **0.167549**. All 300 native PNGs
  use the existing evaluation kernels with float predictions clamped to [0,1],
  valid SSIM windows and spatial masked LPIPS. Earlier video diagnostics used
  unclipped RGB, explaining their slightly different PSNR/RMSE values.
  Visual preference is not a claim of superior framewise metrics: the previous RGB
  fit measured 22.635570 dB / 0.768559 SSIM / 0.166089 LPIPS. Full-video,
  other-view and refinement acceptance remain untested.

This records a visual baseline; it does not replace the production `fit-rgb` or
`refine-motion` implementation. The diagnostic still computes both alternatives;
no isolated reference-only runtime was measured. Original artifacts remain frozen.

The explicit `refine-motion --flow-initialization` option can now initialize a new
motion-refinement experiment from this solution, selecting only its 300 frames,
retaining the included offset/scales and skipping zero-q warmup. It remains a
candidate refinement, not an automatically accepted replacement for this baseline.

## Motion subject selection — 2026-09-24

Use an explicitly saved union of one or more oriented 3D boxes on the current static scene, applied with
`static apply-selection`, as the authoritative motion foreground for subsequent
graph/control construction and modal learning. XMem masks do not define this final
subject. They remain inputs for bootstrap partitioning and background-only camera
stabilization. Never reuse selection indices across static scenes.
The user-saved Bush 540p union was tested at 0.744 Hz with deformation weight
0.03 unchanged in `pipeline540_box0744_20260924_001` (completed 2026-09-25).
It has 322,162 foreground Gaussians and 4,085 controls; training stopped at step
1,009 under the existing convergence rule. Publication/checksum loading passed;
visual acceptance and reconstruction quality remain unverified. See the experiment's
`RUN_REPORT.md`. Future scenes still require the user's saved subject selection.

Configuration: [`configs/neural_component_field.json`](configs/neural_component_field.json).
Pipeline/entry points: [README](README.md). Cleanup compatibility: [REBUILD](REBUILD.md).

## Static coarse geometry — SH recipe, 2026-09-24

The first static pass supplies coarse geometry for stabilization and mode learning.
`static train` now budgets **3,000 Adam updates**, batch size 4, through
`--iterations` (replaces `--epochs`). This follows the author's generic coarse
budget, not the much longer supplied experiment launch. No real 540p run has yet
validated this recipe or its speed/quality. Batch 4 is the user's iteration-speed
choice, differing from the author's batch 1. The update budget and all step-based
schedules stay unchanged; full batches give 12,000 image exposures (resolution
groups may end with smaller batches).

| Setting | Current default |
| --- | --- |
| Appearance | World-frame SH, degree 0 initially, +1 every 1,000 updates, maximum 3 |
| Initialization | COLMAP RGB converted to SH DC; higher coefficients zero; opacity 0.1 |
| Initial scale / rotation | RMS distance to up to three nearest neighbors within each retained partition (squared floor `1e-7`); identity quaternion |
| Position LR | `1.6e-4 * camera_extent`, exponential to `1.6e-6 * camera_extent` over 20,000 updates |
| SH DC / higher-order LR | 0.0025 / 0.000125 |
| Opacity / scale / rotation LR | 0.05 / 0.005 / 0.001, constant |
| Adam epsilon | `1e-15` |
| RGB objective | Full-frame `L1 + 0.2 * (1 - SSIM)` |
| Depth objective | With `--depth`: camera-Z L2, weight 0.01, first 3,000 updates; otherwise disabled |
| Density | After step 500, every 100 updates, before step 9,000 or training end |
| Densify gradient / split scale | `2e-4` / `0.01 * camera_extent` |
| Cull opacity | Below 0.005 |
| Large-point cull | After step 6,000 only: scale above `0.1 * camera_extent` or radius above 20 pixels |
| Opacity reset | Cap at 0.01 at step 500 (white background), then every 6,000 while density is active |

Camera extent is 1.1 times the maximum distance of normalized camera centers from
their mean. The 3,000-update run stops before the position LR reaches its final
value, and degree 3 is activated only on its last update. Longer budgets train
the highest SH band further; use `--iterations 30000` when explicitly requested.
Training uses all registered sweep/reference cameras at their input resolution.

Retained project-specific choices: foreground/background partitions, initial
40k/80k point caps, 160k background cap and shuffled camera
passes. We do not introduce the author's learned Gaussian retain-mask or depth
pretraining. His supplied launch uses an 80k coarse loop but its optimizer gate
stops at the global 30k budget; our budget counts actual updates and does not copy
that gate. Numerical alignment is therefore not an exact reproduction.

All photometric renderers evaluate SH from the current (possibly deformed)
Gaussian position and camera center. The basis stays in world coordinates;
Gaussian covariance rotation does not rotate SH. Fixed-mode coefficient fitting
and motion refinement freeze every SH coefficient and the active degree. Coefficient/refinement RGB loss keeps its existing 0.8/0.2 weights.

Static training uses full-frame `L1 + 0.2 * (1 - SSIM)` and optional depth L2. Mask loss, its
erosion/quantile settings and `--mask-weight` have been removed. Original masks
still define the initial foreground/background partition and serve other stages;
the optimization loop does not read them or render foreground-membership targets.

Static execution caches decoded RGB bytes in a per-run 2 GiB
CPU LRU; larger inputs remain loadable without caching. RGB normalization stays
on CPU to preserve the original float32 pixel values. The shared radial renderer
caches up to 16 device sampling grids, keyed by intrinsics, distortion, dimensions,
batch padding, device and dtype. Static training requests expected depth only
while depth supervision is active; normal scene/depth render calls retain depth. Caching is an execution
optimization; removal of mask supervision is a separate loss change.
Caches are transient and are rebuilt lazily after restoring a checkpoint.

### COLMAP-conditioned DA3 depth supervision

`static prepare-depth` uses the exact registered sweep/reference PNGs and raw
COLMAP world-to-camera poses. A local multi-view DA3 model runs in an isolated
Python environment (upstream requires NumPy < 2). Inference receives undistorted
images and pinhole K, including the half-pixel conversion from the renderer to
DA3's integer-center convention. Returned K handles DA3 resizing/cropping; depth
is mapped back to the original distorted training grid. Source pixels/cameras
are never replaced by predicted cameras. Camera-Z depth aligned by DA3 to raw
input translation scale is divided by the existing scene-normalization scale once.
It is COLMAP-scale supervision, not a claim of metric ground truth.

Defaults: processing long side 1008, 16 target images per group plus at most
three spatially spread context cameras; every target is published once. Each
group is pose-conditioned and scale-aligned. This bounds memory and is not one
global all-images attention pass. The lowest 10% confidence is excluded per image,
as are invalid/nonpositive depth and undistortion/resize padding. These inference
and confidence settings are recorded experimental starting points, not tuned results.

With `static train --depth DEPTH`, the default objective is
`L1_RGB + 0.2 * (1 - SSIM) + 0.01 * mean_images(mean_valid_pixels((ED - DA3_Z)^2))`.
`ED` is alpha-normalized expected depth from all foreground/background Gaussians.
Targets and valid support are fixed; predicted opacity does not gate the loss.
`--depth-weight` and `--depth-until-step` configure weight and number of supervised
updates (defaults 0.01 and 3000). Later updates request RGB only. No per-image
scale/shift is fitted during training, and no inverse-depth or mask loss is added.

The author code in `summer2023/x3d/experimental/forestGaussians/train.py` uses
coarse-stage depth L2 before `depth_iterations` (and a separate optional disparity
L2). We retain that objective family, replacing its input with posed DA3 depth and
filtering unreliable predictions. The weight 0.01 is our starting setting, not
the author's validated setting; his generic `lambda_depth` defaults to zero.
Moving foliage and group-dependent predictions remain pseudo-depth limitations.
Bush input preparation has now completed with DA3-LARGE-1.1: 362 sweep frames at
30 FPS plus three raw reference images, all 960x540, under
`scene_library/bush/experiments/static540_da3_20260924_001/`. Existing COLMAP
poses were reused with resized intrinsics/observations; this was not a new SfM
run. All depth/input contracts passed preflight. The first RGB-D static run then
completed 3,000 updates at batch 4: 280.4 seconds training, 299.1 seconds including
input loading/initialization/publication. It published `static_scene/` with 375,410
foreground and 159,915 background Gaussians. Reload/checksum/source-identity checks
passed; no formal reconstruction evaluation, viewer or downstream stage was run.
See the experiment's `training_verification.json`. Completion does not establish
depth accuracy or visual reconstruction quality.

References: [DA3 API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md),
[depth backprojection](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/utils/export/glb.py).

## Video stabilization baseline — accepted 2026-09-24

The user accepted **fixed-map background camera poses + reference-depth
reprojection** as the baseline for future fixed-view video stabilization.
This replaces the earlier smoothed 2D homography method as the chosen method;
it is now the sole production stabilization implementation. Fixed-view recordings
default to it; only explicit `--tripod` skips stabilization. Supply `--scene STATIC
--view LABEL` with the same-resolution raw reference PNG registered by COLMAP.
The moving sweep remains a moving-camera input.

1. Register the raw reference image in the sweep COLMAP map. Exclude the moving
   subject with the existing foreground masks; use static background 2D–3D points.
2. Track directly from the reference to each frame with forward/backward LK
   checks. Estimate each pose against the same fixed map with EPnP RANSAC, then
   refine inlier reprojection error with LM. Keep map and intrinsics fixed;
   do not accumulate pairwise transforms or apply temporal smoothing.
3. Render static 3DGS expected depth at the fixed target pose. For low-opacity
   depth holes, interpolate background sparse-point inverse depth; outside its
   convex hull use nearest sparse depth. This fills depth, not RGB pixels.
4. Inverse-warp original RGB into the target camera using this depth, the full
   per-frame rotation/translation and radial distortion. Preserve timing and
   field of view. Leave out-of-image regions black; no border replication,
   RGB inpainting or crop/zoom is part of the accepted trial.

Accepted Bush view1 trial: **960x540, 30 FPS, 1170 frames**, target frame `00559`.
Pose source: `scene_library/bush/experiments/background_pose_view1_540p_20260924_001/`.
Stabilized preview, scripts, settings and checksums:
`scene_library/bush/experiments/pose_stabilized_view1_540p_20260924_001/`.
The pose fit used 1172 background points with 235 held out. Background tracking
on 39 decoded output frames gave a time median of per-frame median displacement
of **0.168 px** within the common valid region. This and the user's visual
acceptance establish the preview baseline, not foreground motion accuracy or
validation of other recordings. Static depth can still misrepresent moving
boundaries and changing occlusions.

The trial reused the existing static scene; it was not a cold 540p pipeline run.
Production targets the exact registered static camera, rather than the trial's
separately PnP-refined reference pose, keeping downstream projections consistent.
Pixel thresholds scale with image height from the accepted 540p settings; override
the settings dataclass through `--config` when needed. Production publishes PNGs,
nearest-neighbor warped masks, per-frame/common validity, poses and depth. Flow
retains time-complete valid trajectories; invalid modal samples do not supervise
gains, soft graphs or modal loss. RGB L1 uses common valid pixels, SSIM wholly
valid windows. Masked LPIPS uses equally zero-masked RGB and spatial valid-support
weighting; boundary feature context remains. This evaluation protocol is distinct
from the historical full-frame baseline below.
MP4 previews are not training inputs. The next full run must publish lossless
frames, consistently warped masks, valid-pixel masks and matching target-camera
identities before rebuilding downstream stages; see [REBUILD](REBUILD.md).
Historical reconstruction baselines and source artifacts remain unchanged.

## Spatial modes

- Static geometry, appearance and cameras remain fixed during mode learning.
- Fixed views are independent recordings. Cross-view complex gains align the
  spatial modal observations; they do not synchronize video frames.
- Use explicit render-matched motion-reference selections on the fixed camera grid.
  Geometry reference and motion reference may be different frames.
- SEA-RAFT computes full-frame reference-to-frame flow in input pixels.
- Shared FFT uses temporal-mean detrending, a symmetric Hann window and a common
  zero-padded RFFT grid. Export exact cached bins; do not recompute selected DFTs.
- Candidate geometry is mutual KNN: K=16, maximum normalized-world distance 0.08.
  Each frequency has its own modal-similarity weights, with minimum factor 0.05.
  Shared control geometry can be reused; one frequency's soft weights cannot.
- GPU alpha fitting and GPU soft propagation use CuPy. There is no CPU fallback.
  The resident preparation worker exits before GNN training starts.
  Alpha's bounded TRF solver QR-reduces the augmented Jacobian and residual
  together before its exact SVD. Float64, two-point differences, block Huber
  loss, gain bounds, stopping tolerances and view-identifiability rules are unchanged.
- Control radius fraction 0.015; maximum 32,768 controls. GNN hidden dimension 256,
  local feature dimension 32, three message layers, maximum 5,000 updates.
- Per-view RMS-normalized modal-image loss, deformation weight 0.03,
  control-rotation regularization weight 0. Angular displacement is still learned
  and saved; zero regularization weight does not disable rotation.
- The default modal projector is dynamic. `modal_projection_backend="cached"`
  is an optional execution optimization for fixed geometry, with the same math.

Component fields require >=10 Gaussians and >=2 controls, plus effective
observations at the selected frequency. A whole eligible component keeps its own
field, including weak or occluded members. Propagation donors additionally require
>=101 component points, >=2 controls and >=3 reliable observed points. Recipients
without eligible donors remain zero/unresolved. See [component fields](docs/component-field.md).

Automatic mask partitions use foreground/mask sampling. Applied manual selections
use full-scene visibility and subject contribution, without inherited mask erosion.
The prepared artifact records the sampling policy and per-view relative depth
visibility tolerance explicitly. Tolerances are calibration inputs, not a hidden
application of an old graph's thresholds.

## Temporal reconstruction

`x(t) = x_static + Re(sum_k(q_k(t) * phi_k))`.
Angular fields use the same coefficients and an exponential-map rotation.
RGB fitting freezes static Gaussians, appearance, cameras and both spatial fields.
Only complex coefficients change. Each video has its own coefficients.

There is currently no temporal smoothing, oscillator model or frequency locking.
A frequency label identifies the learned spatial mode, not a constraint on fitted
coefficient time series. Training losses do not substitute for reconstruction metrics.

Use explicit `result evaluate` to freeze a pre-refinement reconstruction baseline:
native-resolution, full-frame PNG vs float render PSNR/SSIM/RMSE, with `--lpips`
for LPIPS-Alex. Compare identical frames/cameras and evaluation protocol. These
metrics describe fitted recordings, not novel-view or geometric accuracy.

### Frozen historical Bush baseline (2026-09-23)

Experiment: [`baseline_pre_refinement_20260923_001`](scene_library/bush/experiments/baseline_pre_refinement_20260923_001/BASELINE.md).
The existing 20-mode bank and view1 RGB fit were rematerialized as result v2 and
evaluated on all 1170 stabilized 1920x1080 PNGs. No fitting or refinement ran.

| Full-frame, arithmetic frame mean | Value |
| --- | ---: |
| PSNR (dB, higher better) | 20.184717 |
| SSIM (higher better) | 0.598855 |
| LPIPS-Alex v0.1 (lower better) | 0.365822 |
| RMSE in [0,1] RGB (lower better) | 0.098730 |

[`evaluation/metrics.json`](scene_library/bush/experiments/baseline_pre_refinement_20260923_001/evaluation/metrics.json)
also records pooled RMSE 0.099578 and pooled PSNR 20.036769 dB. Only view1 is
covered; view2/view3 have no corresponding RGB fit in this baseline. Source models
are v16. The explicit fixed-reference import passed network/full-field equivalence
with zero error on all 20 modes on 2026-09-24; see the Bush experiment
`refinement20_view1_sweep_20260924_001/verification.json`. General loaders remain current.

## Optional fixed-scene motion refinement

`configs/motion_refinement.json` replaces geometry/density refinement. All static
Gaussian attributes and counts remain fixed; learn valid control displacement/
angular corrections and free per-frame complex q. Cameras, graph, control layout,
frequency weights and donor rules remain fixed. Control corrections use the original
field RMS, complex-orthogonal gauge and anchor; q keeps the original **2D projection
pair scales**, with sweep inheriting the first selected fixed view's scale.

Full resolution only: ten zero-start full-frame RGB warmup passes (q LR 0.01 to
0.001), then two rounds of sparse joint motion/q updates (fixed 5 FPS, sweep 10 FPS)
and one exhaustive coefficient pass. Post-warmup q LR is 0.001 to 0.0001 on the
post-warmup update clock; control LR decays exponentially 0.001 to 0.0001 on the
joint clock. Preserve per-frame Adam state across all phases. Warmup-end q is the
anchor. RGB weights 0.8/0.2; q anchor 1e-4, field anchor 1e-2, dynamic rigidity and
relative rotation each 1e-3. Frame-zero skips temporal regularizers. The previous
frame's q is detached. No modal-image/alpha loss, density, depth, carrier or damping.

Joint field execution caches fixed stencil indices/segment lengths and defaults
to `query_block_size=32768`; frequency groups remain four modes. The two-stage
donor sum and checkpoint recomputation are retained. This is an execution change,
not a new loss or sampling recipe; smaller blocks remain configurable for memory.

Group weighting remains equal fixed/sweep, with fixed views averaged. Exhaustive
updates correct for unequal sequence lengths. See [the reconstruction contract](COEFFICIENT_FITTING.md#fixed-scene-motion-refinement) for exact regularizers,
gauge, equations and commands. Seed 1729; checkpoints every 200 updates and phase
boundaries. For view1 1170 + sweep 362 frames: 15,320 warmup updates plus two rounds
of 195 joint + 1532 coefficient updates = 18,774 updates, 19,164 RGB renders.
These defaults and counts do not establish runtime, convergence or real quality.

## Resume boundaries

| Situation | Action |
| --- | --- |
| Interrupted batch, identical code/input/config | Rerun the same command/output. |
| Increase only the training iteration cap | New output with `--continue-from`; compatible optimizer/RNG state is retained. |
| New batch attempt inheriting complete modes | New output with `--resume-from`; only matching completed results are inherited. |
| Interrupted coefficient preparation | Same command with `--resume`; exact published stages are reused. |
| Interrupted RGB optimization | New output directory; no optimizer checkpoint exists. |
| Interrupted alternating refinement, identical contract | Same work directory with `--resume`; reconstruct transient GPU basis if needed. |
| Changed identity, schema, loss, reference or code contract | Rebuild affected dependencies into new outputs. |

These mechanisms apply to the current schema. They do not upgrade pre-cleanup artifacts.
