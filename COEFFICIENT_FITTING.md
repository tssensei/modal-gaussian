# Coefficient fitting and optional scene refinement

Spatial mode learning and temporal fitting are separate stages. Input modes are
immutable complex displacement and angular fields. The static scene, appearance,
cameras, Gaussian order and both fields remain frozen throughout RGB fitting.
Videos were recorded separately; fit independent coefficients for each video.

```text
completed batch index
  -> fixed mode bank + rendered linear projection design
  -> SEA-RAFT ridge initialization
  -> shared RGB pose offset + per-frame complex coefficients
  -> result binding -> explicit video export
```

`coordinates.npy` is `complex64 [total_selected_frames, mode_count]`.
Displacement is `Re(sum_k(q_k(t) * phi_k))`. Saved angular fields drive the
exponential-map rotation of static Gaussian orientations.

## Commands

Replace uppercase paths with current, contract-compatible outputs. All new paths
must be inside the scene's new experiment. Catalogs may still point to artifacts
listed in [REBUILD.md](REBUILD.md).

```sh
modal-gaussians coordinates prepare --scene bush --index BATCH/index.json --status complete --expected-modes 20 --output FIT
modal-gaussians coordinates fit-rgb --scene STATIC --modes FIT/mode_bank --input FIT/direct_coordinates --view view1 --config configs/rgb_coordinates.json --output FIT/rgb_coordinates_view1
modal-gaussians result materialize --scene STATIC --modes FIT/mode_bank --coordinates FIT/rgb_coordinates_view1 --output FIT/result_view1
modal-gaussians result evaluate --result FIT/result_view1 --output FIT/evaluation_view1 --lpips
modal-gaussians result export-video --result FIT/result_view1 --view view1 --output FIT/exports/view1_001
```

`prepare` produces `preparation.json`, `mode_bank/`, `rendered_design/` and
`direct_coordinates/`. Use `--resume` only for an identical preparation contract.
It never starts RGB fitting. `--status` and `--expected-modes` are explicit, so an
old experiment count/status cannot be selected accidentally.

`fit-rgb --view LABEL` reads and fits only that video while reusing shared mode
inputs. Omit `--view` to fit all recordings independently. Initialization subtracts
the reference coefficient, estimates a shared RGB pose offset, then optimizes
per-frame coefficients over multiple resolutions. Loss is
`0.8 * L1 + 0.2 * (1 - SSIM)` plus a weak, decaying initialization anchor.
There is no optimizer-resume checkpoint for RGB fitting.

Targets are the actual inference-grid PNGs, which may be stabilized. Export keeps
input FPS and each panel's resolution: target above, reconstruction below. Output
contains `comparison.mp4`, `manifest.json`, and `encode.log`.

## Reconstruction baseline

Materialize the existing scene/modes/RGB coefficients into a new result before
refinement, then run `result evaluate`. Default coverage is every fitted frame
of every result view; repeat `--view LABEL` to select a subset. This does not fit
missing recordings or modify any input. Ordinary RGB and refined RGB results
use the same evaluator and saved coefficient offsets.

The fixed protocol is full-frame RGB at native PNG resolution, using the recorded
camera and distortion. Predictions are float32 clamped to [0,1], without video
compression, quantization, resizing, exposure alignment or reference subtraction.
PSNR uses an MSE floor of 1e-12 (120 dB ceiling); SSIM uses the existing static-QA
11-pixel Gaussian window (sigma 1.5, data range 1); RMSE is in [0,1] RGB units.
Optional `--lpips` uses pretrained LPIPS-Alex v0.1 at native resolution. Install
`pip install -e ".[evaluation]"`; its first use downloads the official AlexNet
weights. Set `TORCH_HOME` to `scene_library/_shared/tools/torch` to keep the cache
in the scene library.

Each new evaluation publishes `per_frame.csv`, `metrics.json` and `manifest.json`.
Summaries include arithmetic per-frame means, pooled pixel-weighted RMSE/PSNR,
per-view results and an equal-view mean. Compare matching views, frame hashes,
resolution and protocol identity. The manifest records result identity, input
PNGs, implementation revision, package versions and LPIPS weight hash. Original
artifacts remain immutable and source tensors/fields/frames are checksum-verified.

Add `--baseline BASELINE_EVALUATION` when evaluating the refined result to save
`per_frame_delta.csv` and per-sequence mean deltas (refined minus baseline).
The evaluator checks protocol/revision, timestamps, camera identities and PNG
hashes before comparison. Re-evaluate historical baselines into new directories
with the current evaluator; retain their original reports. Check individual sweep
frame deltas as well as means so a few degraded viewing directions remain visible.

These are fitted-recording reconstruction metrics, not held-out-time or novel-view
quality and not a geometry accuracy score. Full-frame metrics include background;
inspect subject detail and sweep-view degradation separately. A view1-only result
does not establish a three-view baseline.

## Limits

- No temporal smoothing, oscillator constraint or hard frequency locking.
- Labeled high-frequency spatial modes may help fit lower-frequency movement.
- GPU fitting currently uses per-frame renders with gradient accumulation.
  Image loading/caching and synchronization remain potential optimization work.
- Training losses and compressed comparison videos are not formal reconstruction
  metrics. RMSE/PSNR/SSIM/LPIPS reports require a separate requested evaluation.
- Fitting does not start exports, metrics or Viser automatically.

## Joint scene refinement

Use fixed-mode RGB fits as initialization for an explicit second method.
`prepare-refinement` freezes reference inputs; `refine-scene` alternates sparse
geometry/coefficient updates and exhaustive coefficient-only passes. There are
two rounds, entirely at image scale 1.0. The replaced all-frame joint schedule and
multiscale refinement configuration are no longer supported.

In each geometry phase, select one frame per temporal bin: six source frames for
30 FPS fixed views (5 FPS geometry density), three for sweep (10 FPS). Preserve
first/last frames in endpoint bins; choose other bins with the seeded RNG. A single
bin retains its first frame. Reselect each round, shuffle independently, and cycle
shorter sequences. A shared update consumes one frame per sequence. Average fixed
views within their group; fixed and sweep groups have equal weight (view1+sweep:
0.5/0.5). Videos remain asynchronous and retain their own coefficients/cameras.

In each coefficient phase, freeze every Gaussian attribute and query the final
geometry's modes once into detached GPU tensors. Visit every reconstruction frame
once, including sparse geometry frames. Each step renders one frame and updates
only its q and persistent Adam state; it performs no reference-field query,
Gaussian update, support check or density operation. Geometry changes invalidate
the basis; checkpoint load reconstructs it once. Final publication reuses it.

The commands below are independent stages. Preparation requires RGB
fits for selected bank views (`--view LABEL`, repeatable; default all bank views), rejects duplicate/missing recordings and mismatched
scenes/mode order, and preserves each fitted pose offset. No reference subtraction
or mean removal occurs again. Preparation re-evaluates saved GNNs without
gradients to extract fixed control displacement/angular fields, checks them against
the bank, computes sparse path tables and releases its CuPy workspace.

Each live Gaussian carries a permanent UID and original reference root. Its local
portals are that root and its original one-hop neighbors. Current positions attach
to these fixed portals; geometric distances and frequency-weighted shortest paths
determine Wendland weights with radius `2h` and attenuation `d0/dk` (zero/zero = 1).
All portal-to-candidate distances are retained, including lower-cost detours.
Control fields, graph nodes and weights stay fixed; distances, normalized weights
and angular lever arms differentiate with respect to current position.

Propagation is inherited, never re-estimated after moving or adding a render point:

- Original edge `i-j` keeps geometric length `ell_ij` and each mode's propagation
  cost `L_ij[k]`; attenuation is `a_ij[k] = ell_ij / L_ij[k]`.
- A point rooted at `i` connects only to `i` and its original neighbors. Its
  connector to `i` has attenuation 1; to neighbor `j` it inherits `a_ij[k]`.
  Connector costs are `|x-p_j|` geometrically and `|x-p_j|/a_ij[k]` per mode.
  Minimize connector-plus-cached-path cost independently for geometric and modal
  distance, then apply the existing normalized Wendland rule.
- These are query connectors, not new reference edges or routes for other points.
  GNN degree weights are not renormalized: refinement never runs the GNN.
  Each mode retains its control-valid mask and root's own/donor/unresolved roles.
- Clone/split children inherit roots, donor indices/weights and these same rules;
  they get new UIDs. Culling removes only live rows. Neither a parent's modal
  amplitude nor per-frame coefficients are divided between its children.

Canonical position updates also obey a frequency-independent **shape bound**.
For root `i`, let `S_i` be the union of the original segments `[p_i,p_j]` over
its original neighbors. Require
`distance(x,S_i) <= epsilon_i`, where
`epsilon_i = shape_radius_fraction * median_j(ell_ij)`, default fraction **0.25**.
Closest points are clamped to segment endpoints, not infinite lines. Isolated
roots have `S_i={p_i}` and zero tolerance. Original edges/lengths define the bound
once; root assignment and tolerance never follow a moving parent or its children.
There is no global nearest-neighbor reassignment or frequency-dependent shape.
This allows local redistribution along graph edges while bounding off-graph center
drift. Gaussian scale, orientation and opacity remain trainable; this is not a
guarantee of unchanged silhouettes or preserved surface coverage after culling.

The query engine keeps fixed sparse tables on the computation device, batches up
to four frequencies within each Gaussian block, and recomputes activations for
backpropagation. Path distances and weight normalization retain float64 precision.
Ordered segmented sums preserve the original control/donor accumulation order,
avoiding CUDA atomic-sum rounding drift. Support checks skip field composition;
bisection rechecks only rejected points. These changes preserve the model and loss.

Each shared update queries every Gaussian/frequency block once for all sampled
sequences, applying their independent coefficients to the same block fields.
Rendering/backpropagation remains sequential. Detached dynamic-position/angular
leaves collect the render gradients, then one backward traverses the shared query
graph (including its checkpoint recomputation). This preserves the direct position,
interpolation, coefficient and anchor gradients; optimizers step only afterward.
The shared graph lasts one update and is rebuilt after movement or density changes.

Own-field displacement is `sum(w * (d + omega × (x-control_position)))`; angular
displacement is `sum(w * omega)`. Recipients query their frozen donors at
`x + donor_position - root_position`, preserving original copied-field semantics.
Unresolved frequency components remain zero. Both the shape bound and all previously
valid motion supports must hold after an update. On failure, try up to eight
midpoints toward the previous accepted position, then restore that position if
needed; clear position Adam moments for every initially rejected point. Failure
of either child cancels its parent's entire split. This is a hard acceptance check,
not an added RGB loss. Checkpoints and final publication also validate the bound.

Control Gaussian canonical positions remain exact through gradient/momentum masks
and restoration. Their orientation, scale, color and opacity may change, and they
still move dynamically with q. Other foreground Gaussians may move/clone/split/cull;
background and cameras remain fixed. Shared density operations remap all Gaussian
parameters and Adam rows; newborns receive zero moments and inherited roots.

Each sequence loss is `0.8*L1 + 0.2*(1-SSIM) + 1e-4*mean_k(|s*(q-q0)|²)`, using
the actual full-frame PNGs and frozen direct-fit scales `s`. Defaults and density
thresholds are in [BASELINE.md](BASELINE.md) and `configs/scene_refinement.json`.
The original q0 and scales remain fixed for both rounds; no offset reinitialization
or anchor reset occurs. Gaussian LR/density/newborn age use geometry-update counts;
coefficient LR uses all updates. Density is permitted only in round 1's geometry
phase (warmup 34, interval 17, newborn protection 17); round 2 keeps point count
fixed. No time smoothing,
oscillator constraint, depth loss or GNN refinement is included.

Every 200 total updates and at each phase boundary, save a complete atomic
checkpoint: Gaussian/q parameters and Adam state, density/identity arrays, total
and per-phase counters, per-round selected rows, samplers and all RNG state.
`training.jsonl` records per-view RGB/coefficient loss, count/density decisions,
`support_backtracks`, `shape_backtracks` and their union `position_backtracks`
(initially rejected point counts, not bisection iterations), control-position error,
round/phase/update counters and timings. `query_seconds` and
`query_backward_seconds` separate shared deformation from `render_seconds` and
`backward_seconds` (render/loss forward and backward). These runtime wall-clock
fields are diagnostic; synchronized CUDA timing belongs in local benchmarks.
Non-finite values
stop the run before final publication. `bake_seconds` records coefficient-cache
preparation separately. Final cached fields are saved as complex64
`[K,G_new,3]`; viewer/export use ordinary modal superposition without graph queries.
Spectrum shows inherited modal observations with new-geometry projections; its
reference-graph overlay retains original node coordinates.
`viewer --no-spectrum` skips the optional Spectrum panel and its FFT dependencies;
saved-mode and coefficient playback still use the same scene and motion arrays.

Synthetic checks cover graph equivalence/gradients, density identities, GPU
optimization/resume, publication/reload and short-video export. They do not
establish real reconstruction quality. Supported improvement is local to the fixed
control/reference support; wholly missing unsupported structure needs a later method.

### Fixed 20-mode view1 + dynamic sweep

The existing Bush bank retains all three modal observation views. Only view1 and
sweep supervise this experiment. Sweep does not acquire flow/FFT/modal-image roles.
The 724 registered 60 FPS sweep frames are decimated to rows 0,2,...,722: 362
frames at 30 FPS for fitting, refinement, evaluation and export. View1 retains
1170 frames at 30 FPS. Each selected frame keeps its original PNG, camera, source
index and timestamp; the sequences are never paired in time. `sequences.py` owns
the shared selector and rejects irregular grids, upsampling and noninteger ratios.

For v16 component-field sources, first use the isolated importer. It verifies
manifests, network/array hashes, Gaussian/control order and frequency bindings,
then extracts frozen controls and prepares frequency-specific path tables.
Both network replay and complete reference queries must reproduce the bank's
displacement/angular fields within 1e-5 after per-frequency amplitude normalization.
Failure stops publication; no automatic retraining or approximate replacement.
Current v18 sources share this builder inside preparation when `--reference` is
omitted. General completed-model loaders still reject v16.

```sh
python tools/import_refinement_reference.py --scene STATIC --modes BANK20 --output EXP/motion_reference
modal-gaussians coordinates fit-sweep --scene STATIC --modes BANK20 --scale-source RGB_VIEW1 --metadata SWEEP_EXTRACTION_METADATA --fps 30 --config configs/rgb_coordinates.json --output EXP/sweep_coordinates
# If a valid 60 FPS fit already exists, use this INSTEAD of fit-sweep:
modal-gaussians coordinates downsample-sweep --input SWEEP60 --fps 30 --output EXP/sweep_coordinates
modal-gaussians result materialize --scene STATIC --modes BANK20 --coordinates RGB_VIEW1 --coordinates EXP/sweep_coordinates --output EXP/baseline_result
modal-gaussians result evaluate --result EXP/baseline_result --lpips --output EXP/baseline_evaluation
modal-gaussians coordinates prepare-refinement --scene STATIC --modes BANK20 --coordinates RGB_VIEW1 --view view1 --sweep-coordinates EXP/sweep_coordinates --reference EXP/motion_reference --output EXP/prepared
modal-gaussians coordinates refine-scene --prepared EXP/prepared --config configs/scene_refinement.json --work-dir EXP/work --output EXP/refined
modal-gaussians result materialize --scene EXP/refined/scene --modes EXP/refined/mode_bank --coordinates EXP/refined/coordinates --output EXP/result
modal-gaussians result evaluate --result EXP/result --lpips --baseline EXP/baseline_evaluation --output EXP/evaluation
modal-gaussians result export-video --result EXP/result --view sweep --output EXP/exports/sweep
```

`fit-sweep` shares the fixed RGB optimizer and a single resident scene/mode basis.
It initializes q to zero, fits a shared pose offset, then independent frame q;
it does not subtract a reference or temporal mean. `--scale-source` must be a
compatible single-view RGB fit: only its direct-fit `mode_pair_scales` are reused,
never its q. Defaults are 0.25/0.5/1 scales, ten epochs each, LR 0.01 to 0.001,
anchor 1e-4 to 1e-6, 30 offset steps, accumulation of four frames, seed 1729.
There is no sweep optimizer resume; failed fits publish nothing.

`downsample-sweep` publishes a new immutable sweep v1 artifact with selected q,
all selected bindings, parent identity and row indices. It runs no fitting and
records zero newly optimized frames. Preserve the original 60 FPS fit and baseline.
Rebuild preparation and materialize/evaluate a matching 30 FPS baseline in new
paths; comparing to a 724-frame sweep mean would mix different frame sets.

Each round now has 195 shared geometry updates (195 view1 / 121 sweep candidate
frames) followed by 1532 coefficient updates. Totals: 390 geometry updates, 3064
coefficient updates, 3454 optimizer steps and 3844 renders. Checkpoints bind all
frame/camera identities, normalization, initialization, weights and samplers. Old
60 FPS preparations and old trainer checkpoints cannot enter the new schedule;
use new preparation/work directories. The verified fixed reference is reusable.
These counts do not predict real runtime; the new recipe has not run on real data.

Result v3 accepts repeatable disjoint `--coordinates` inputs. Evaluation reports
view1 and sweep separately. Export preserves each sequence's FPS; viewer offers
sweep coefficient playback with free viewing or **Follow recorded frame camera**.
Spectrum retains only original view1/view2/view3 modal images. No command above
automatically starts the next stage. Preserve the frozen historical view1 baseline
and remeasure its rematerialized result separately before real comparisons.
