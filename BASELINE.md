# Current numerical recipe

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
| Depth objective | Disabled; no predicted depth supervision yet |
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
freezes SH; joint refinement optimizes DC at `color_lr`, higher bands at
`color_lr / 20`. Coefficient/refinement RGB loss keeps its existing 0.8/0.2 weights.

Static training uses only full-frame `L1 + 0.2 * (1 - SSIM)`. Mask loss, its
erosion/quantile settings and `--mask-weight` have been removed. Original masks
still define the initial foreground/background partition and serve other stages;
the optimization loop does not read them or render foreground-membership targets.

Static execution caches decoded RGB bytes in a per-run 2 GiB
CPU LRU; larger inputs remain loadable without caching. RGB normalization stays
on CPU to preserve the original float32 pixel values. The shared radial renderer
caches up to 16 device sampling grids, keyed by intrinsics, distortion, dimensions,
batch padding, device and dtype. Static training requests RGB without expected
depth; normal scene/depth render calls retain depth. Caching is an execution
optimization; removal of mask supervision is a separate loss change.
Caches are transient and are rebuilt lazily after restoring a checkpoint.

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

## Optional joint scene refinement

`configs/scene_refinement.json` defines two alternating rounds, all at image scale
1.0. Initialize selected bank recordings (`--view`; default all) and sweep with
compatible RGB coefficients; retain pose offsets and amplitude scales. Sweep must
be a genuine 30 FPS subset. Existing 60 FPS coefficients can be selected without
re-fitting. This method is separate from fixed-mode RGB fitting.

Freeze the original graph, frequency-specific paths/donors, control positions and
fields, cameras, normalization and background. Foreground appearance, orientation
and scale are trainable during geometry phases; only non-control positions move.
Controls never split/clone/cull. No depth, temporal smoothing, frequency locking,
GNN training or global position anchor is added.

Constrain each canonical center to the union of tubes around its permanent root's
original incident **line segments**. The fixed tube radius is
`shape_radius_fraction * median(original incident edge lengths)`, default **0.25**.
This is an experimental starting value, independent of frequency weights and
motion radius `2h`. Isolated nodes have zero radius and cannot move. Children
inherit the root and radius; movement never rebuilds KNN or expands the region.
This bounds center drift; it does not constrain Gaussian scale/opacity or guarantee
unchanged rendered silhouettes.

Each round has two phases:

1. **Geometry:** one sample per 6-frame fixed-view bin (30 to 5 FPS) and 3-frame
   sweep bin (30 to 10 FPS). Keep first/last frames in their endpoint bins; seeded
   random selection in other bins. A one-bin sequence retains its first frame.
   Shuffle sequences independently; shorter sequences cycle. Each shared update
   renders one frame per sequence with one shared differentiable field query.
2. **Coefficient:** freeze every Gaussian parameter, bake displacement/angular modes
   once on GPU, then visit every reconstruction frame once in random order. Only
   that frame's q and its existing Adam state update. The next round rebakes after
   geometry changes. Publish the final cache without another graph query.

Fixed-view losses average within their group; fixed and sweep groups have equal
weight. The loss per frame is `0.8*L1 + 0.2*(1-SSIM) + 1e-4*mean(|s*(q-q0)|²)`.
The original q0 and scales s remain fixed across phases and rounds. Coefficient
LR decays linearly 0.001 to 0.0001 over **all optimization updates**. Gaussian LRs
are means 4e-5, quaternion 2.5e-4, log-scale 1.25e-3, color/opacity 2.5e-3;
exponential decay to 0.1 over **geometry updates only**.

Density changes run only in round 1's geometry phase: warmup 34 geometry updates,
interval 17, newborn protection 17. Require five visible records; cap foreground
at twice its initial count; no opacity reset. Thresholds remain gradient 2e-4,
split world/screen scale 0.01/0.05, cull opacity 0.005 and world/screen scale
0.5/0.15. Undo the actual loss weight in density statistics. Cancel a split if
either child violates shape or motion support. Backtrack invalid position updates
up to eight midpoints toward the previous accepted position, then restore it if
necessary. Clear position Adam moments on every initially rejected row.

For Bush's fixed 20-mode experiment:

| Budget | Per round | Two rounds |
| --- | ---: | ---: |
| view1 / sweep reconstruction frames | 1170 / 362 | same frames |
| Geometry candidate frames | 195 / 121 | reselect per round |
| Shared geometry updates | 195 | 390 |
| Coefficient-only updates | 1532 | 3064 |
| Total optimizer updates | 1727 | 3454 |
| RGB renders | 1922 | 3844 |

Density events are geometry steps 51, 68, ..., 187 (nine events). Gaussian LR uses
`geometry_step/389`; coefficient LR uses `step/3453` before each update. Checkpoint
every 200 total updates and every phase boundary. These are operation counts,
not runtime estimates or evidence of reconstruction improvement.

The verified reference, historical sweep `[724,20]` fit and 60 FPS common baseline
remain immutable in `refinement20_view1_sweep_20260924_001/`. Historical training
attempts were paused; no refined scene was published. Logs/measurements remain in
that experiment. The new recipe requires a 30 FPS sweep subset, new prepared
inputs, matching 30 FPS baseline and new work directory; see [REBUILD](REBUILD.md).
This implementation does not run those stages.

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
