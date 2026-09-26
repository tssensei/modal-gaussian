# Coefficient fitting and optional motion refinement

Spatial mode learning and temporal fitting are separate stages. Input modes are
immutable complex displacement and angular fields. The static scene, appearance,
cameras, Gaussian order and both fields remain frozen throughout RGB fitting.
Videos were recorded separately; fit independent coefficients for each video.

Appearance now uses world-frame SH (static scene v5/v6, refined v7). Fixed RGB and
sweep fitting freeze all SH parameters while evaluating colors at each frame's
camera and deformed positions. Joint refinement updates foreground SH DC at
`color_lr` and higher bands at `color_lr / 20`; background SH stays frozen. Baked
fields, result rendering, evaluation and playback use the same SH-aware renderer.
This does not change the existing 0.8 L1 / 0.2 DSSIM coefficient/refinement loss.

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

## Reference + adjacent flow diagnostic

`tools/compare_flow_coordinates.py` is an independent frozen-scene experiment,
not a replacement for `fit-rgb` or motion-refinement warmup. Run its explicit
`--stage design`, `pairs`, `solve`, `export`, then `plot` with the same
`--prepared PREPARED --flow REFERENCE_FLOW --rgb-preview CHECKPOINT_PREVIEW
--output NEW_EXPERIMENT --view view1 --frames 300` arguments. The plotting
interpreter needs Matplotlib; numerical/rendering stages use the CUDA environment.
The comparison preview must bind the exact same scene, modes, frames and cameras.

The experiment uses existing reference-to-frame flow and new adjacent forward /
backward SEA-RAFT flow. Adjacent flow is sampled at `x + F_ref_to_previous(x)`,
with valid bilinear footprints and forward/backward consistency. A float64 sparse
block-tridiagonal ridge solve compares adjacent weights 0 and 1 (ridge `1e-4`),
normalizing each frame/pair by its valid real-component count. Missing pair support
adds no constraint; missing reference support is an error. This constrains observed
motion increments, not a zero-velocity or fixed-frequency prior.

Both solutions share one RGB-fitted reference-frame offset (200 full-resolution
Adam updates, LR `.01 -> .001`, best loss including zero initialization). Coordinates
are `offset + relative_q`, without temporal mean subtraction or per-frame RGB
refinement. Static canonical projection remains a small-deformation approximation;
flow design does not model covariance-rotation appearance changes. Video diagnostics
use the full displacement/angular fields and SH. The four panels are input, previous
RGB fit, reference-only solve, and reference-plus-adjacent solve. Uncompressed RGB
errors use stabilization-valid pixels; these are used-frame diagnostics, not an
equal-budget or novel-view benchmark. No FFT, catalog or production artifact changes.

## Valid pixels after stabilization

Fixed-view stabilization is now the default; only an explicitly confirmed tripod
recording skips it. Prepare COLMAP/static geometry first, then run `prepare reference
--scene STATIC --view LABEL`. The target is the registered static camera, so the
coefficient renderer uses that same camera. Do not substitute preview MP4s for PNGs.

Fixed RGB artifacts bind common valid support. Fit and joint refinement exclude
black missing pixels from L1 and require entirely valid SSIM windows; sweep remains
full-frame. Evaluation v2 uses the same pixel support for PSNR/RMSE and records its
identity. LPIPS uses zero-masked inputs with spatial valid-support weighting;
boundary feature context remains. Re-evaluate comparable baselines with the same
inputs/support/protocol; historical full-frame numbers are not directly comparable.

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

The fixed protocol is valid-support RGB at native PNG resolution, using the recorded
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

## Fixed-scene motion refinement

`coordinates refine-motion` is a separate method from fixed-mode `fit-rgb` and
`fit-sweep`. It freezes **all Gaussian positions, rotations, scales, world-frame
SH, opacity and counts**, including the background. Cameras and normalization also
remain fixed. It learns shared complex control-point displacement/angular
corrections and independent free complex coefficients for every video frame.
Frequency labels/order stay fixed; q has no carrier lock, oscillator or damping model.
There is no Gaussian optimizer, densify/cull, position rollback or new scene output.

Preparation accepts selected bank recordings (`--view`, repeatable; default all)
directly, with optional registered sweep metadata. No pre-fitted coefficient input
is required. Original modal observation views remain intact for Spectrum; sweep
has no FFT/modal-image role. Preparation runs no training.

The reference graph, frequency-specific propagation weights, controls and their
canonical positions remain fixed. Preparation collapses canonical path queries into
a sparse control operator. Own fields obey
`phi = sum(W*(d + omega cross lever))` and `Omega = sum(W*omega)`.
Recipients then mix those donor fields with the original donor weights. Retain
both sum stages and each donor's canonical lever: coalescing the weights changes
float32 accumulation and can exceed the original-field tolerance.
Unresolved rows stay zero. Original displacement/angular fields must be reproduced
within `1e-5` after per-frequency amplitude normalization. Training uses blocks of
4096 Gaussians and at most four modes per group, with checkpoint recomputation;
it never executes the GNN or shortest-path query.

Write `B0 = [d0, radius*omega0]`. A real/imag correction parameter is normalized by
the original valid-control RMS of B0 and projected onto B0's complex orthogonal
complement per mode (`<B0, delta B> = 0`). This fixes the overall complex scale/phase
ambiguity with q. Invalid controls remain unchanged. The correction anchor is a
soft penalty, not a hard per-control motion limit. All-zero/non-finite modes fail.
Coefficient normalization is the **original rendered-design pixel-pair RMS**, not
3D displacement RMS. Preparation reuses projection kernels and the direct solver's
scale formula without solving ridge coordinates. Every fixed view gets its own
frozen scale; sweep uses the first selected fixed view in bank order (recorded).

### Schedule and objective

All stages use resolution 1.0 and independent video clocks:

1. Initialize every q to zero. Freeze original control fields and run **10 full
   shuffled passes** over all reconstruction frames, RGB loss only. Update just
   the sampled q row and its Adam state. No shared offset or reference subtraction.
   Snapshot normalized q after warmup as the fixed anchor for subsequent stages.
2. Each joint round selects fixed-view frames at 5 FPS and sweep frames at 10 FPS.
   Preserve first/last frames in endpoint bins, select other bins with seeded RNG,
   shuffle independently and cycle shorter sequences. One shared update samples
   one frame per sequence and updates control corrections and sampled q together.
   There is no inner solve of q to convergence. Compose shared field blocks once,
   render frames sequentially, then backpropagate accumulated output gradients once.
3. Freeze fields, bake one detached GPU displacement/angular basis, and visit
   **every reconstruction frame once** to update only q. Preserve each row's Adam
   state across all phases. Run two joint/coefficient rounds. Invalidate the baked
   basis after joint updates and checkpoint load.

RGB loss remains `0.8*L1 + 0.2*(1-SSIM)` on valid PNG support. After warmup add:
`lambda_q * mean(|s*(q-q_warmup)|^2)`, local dynamic rigidity, neighbor relative
rotation, and (once per joint update) normalized control-field correction penalty.
No original modal-image/alpha loss or depth loss is added.

Dynamic regularizers compare the current frame with its immediately preceding
frame **in the same reconstruction sequence**, even if that predecessor is not a
sparse joint sample. Previous q is detached; control fields at both times remain
differentiable in joint phases. Frame zero skips these terms. For each original
geometric edge, rotate the current edge back by its center's relative rotation,
compare with the previous edge, and normalize squared error by original squared
edge length. Relative rotations are compared as rotation matrices (quaternion
sign invariant), squared Frobenius distance divided by three. Evaluate both edge
directions; average neighbors per node, then average supported nonisolated nodes.
Use original geometry edges whose endpoints have motion support, never one mode's
soft weights. Previous-frame regularization adds deformation work but no RGB render.

Average fixed views within their group and weight fixed/sweep groups equally.
An exhaustive single-frame update uses `group_sequence_weight * total_frames /
sequence_frames`, so uniform frame sampling preserves this same objective.

| Default | Value |
| --- | --- |
| Warmup passes; q LR | 10; linear 0.01 to 0.001 |
| Joint rounds; exhaustive passes per round | 2; 1 |
| Fixed/sweep joint sampling | 5 / 10 FPS |
| Post-warmup q LR | linear 0.001 to 0.0001 over post-warmup updates |
| Normalized control LR | exponential 0.001 to 0.0001 over joint updates |
| q anchor / field anchor | 1e-4 / 1e-2 |
| Rigidity / relative rotation | 1e-3 / 1e-3 |
| Checkpoint / seed | every 200 updates and phase boundaries / 1729 |

For 1170 view1 and 362 sweep frames: warmup 15,320 updates, then two rounds of
195 joint + 1532 coefficient-only updates. Total **18,774 updates and 19,164 RGB
renders**. These are operation counts, not runtime estimates or convergence claims.
Defaults live in `configs/motion_refinement.json` and are experimental starting values.

### Commands, recovery and outputs

Optional early stopping uses `early_stop_patience` (0 disables it),
`early_stop_joint_interval` (default 25 joint updates), and
`early_stop_min_relative_improvement` (default 0.001 = 0.1%). It measures mean
RGB loss on **all supervised frames** with the original sequence-group weights,
after initialization/warmup, at interior joint intervals, and after each complete
coefficient phase. The joint/coeff boundary waits for the coefficient phase instead
of counting two adjacent checks. This is training-set convergence, not validation.
Small gains accumulate relative to the last significant improvement; every true
minimum is independently retained. On plateau or budget completion, publish the
best checked state, which can precede the last update. `work/checkpoint.pt` keeps
the latest state and plateau history; `work/best/` keeps atomic best checkpoints.
Resume restores both counters and best-file hashes. Final coordinates record
`training_selection` with actual/published steps and the selection loss.

To start from the reference-only flow diagnostic, pass `--flow-initialization
DIAGNOSTIC_ROOT` to `coordinates refine-motion`. This selects only the diagnostic's
exact fixed-view prefix (Bush frames 0–299), even if preparation also contains
sweep or later frames. The loader checks scene/mode/preparation identities, PNGs,
cameras, timestamps, validity, reference binding and consumed solution checksums.
It uses `solution/reference_only.npy` **including the shared offset**; no second
offset, reference subtraction or temporal centering is applied. The diagnostic's
pixel-pair scales are frozen, q is converted into normalized parameters and copied
as the anchor. Warmup is skipped regardless of `warmup_passes`; fresh Adam states
start directly in joint refinement. Without the option, zero-start behavior stays.

Selection and initialization hashes enter the run contract and final coordinate
provenance. Prepared and diagnostic files remain immutable. Resume requires the
same argument/source/config/implementation; importing q starts a new experiment,
not a continuation of an old warmup optimizer. One round consists of sparse joint
updates followed by an exhaustive q pass.

```sh
# Only v16 sources need the explicit importer; current sources can build during prepare.
python tools/import_refinement_reference.py --scene STATIC --modes BANK20 --output EXP/motion_reference
modal-gaussians coordinates prepare-refinement --scene STATIC --modes BANK20 --view view1 --sweep-metadata SWEEP_METADATA --reference EXP/motion_reference --output EXP/prepared
modal-gaussians coordinates refine-motion --prepared EXP/prepared --config configs/motion_refinement.json --work-dir EXP/work --output EXP/refined
modal-gaussians result materialize --scene STATIC --modes EXP/refined/mode_bank --coordinates EXP/refined/coordinates --output EXP/result
modal-gaussians result evaluate --result EXP/result --lpips --output EXP/evaluation
modal-gaussians result export-video --result EXP/result --view sweep --output EXP/exports/sweep
```

For calibrated resized/subsampled sweep inputs, pass the original extraction
metadata. The loader validates the saved derivation and parent camera hashes,
uses the scene's resized PNGs/cameras, and preserves original source indices and
timestamps. An already selected 30 FPS sweep is not subsampled a second time.

Omit sweep metadata for fixed-view-only refinement. The registered 60 FPS sweep is
actually decimated to 30 FPS, with PNGs, source indices, timestamps and cameras kept
together. Do not merely relabel FPS. Fixed views retain their original frame grids.
Standalone `fit-sweep` and `downsample-sweep` remain fixed-mode baseline tools;
they are not prerequisites for motion refinement.

Prepared v5 contains reference, fixed operator and normalization inputs. Work owns
atomic checkpoint/log/run state. Final output has only manifest, `mode_bank/`
(completed modes v20) and `coordinates/` (refined RGB v5); it references the unchanged
static scene. Save original/final controls, operator, final complex64 `[K,G,3]`
displacement/angular arrays and original observation provenance. Result v3 directly
binds these with the original scene, without a ridge design. Viewer/Spectrum use
the new baked fields; observation roles remain inherited sources, not new modal
supervision. No stage automatically starts a later stage.

`--resume` requires identical prepared/config/code/device contracts. Checkpoints
include corrections, q, every Adam state, post-warmup anchor, phase counters,
per-round sparse selections, exhaustive permutations, cursors and NumPy/PyTorch/
CUDA RNG. Transient GPU basis/render graphs are rebuilt. Non-finite loss/gradient
stops without final publication. Logs separate query/render/regularization/backward
times and record per-view losses, correction/q magnitudes and displacement/angular
RMS. Synchronized CUDA timing belongs in the local benchmark, not daily logging.

The old `refine-scene`, density/shape configuration and fitted-coordinate preparation
arguments are removed. Old preparations/checkpoints cannot resume this method;
use new prepared/work/output directories. Preserve historical artifacts/baselines.
Matching static scenes, original banks and verified references remain reusable.
This method can change the span of the existing control-supported fields but adds
neither modes nor support for unresolved structure. Real quality requires a separate
authorized comparison with fixed-mode free-q fitting on the same inputs and budget.
