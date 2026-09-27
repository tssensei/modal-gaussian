# Adaptive SfM comparison: Phase 4

Evaluate the four Phase 3 budget states independently of training, using the
44 common held-out PNGs. Preserve Phase 1 inputs and all Phase 3 checkpoints.
No coefficient fitting, new training, catalog redirection or static-scene import
is part of this phase.

## Explicit stages

In the Modal environment, render either static v4 checkpoint:

```text
python tools/evaluate_adaptive_sfm.py render-static --input EXP/data --checkpoint CHECKPOINT --scene-manifest EXP/phase3_clean/static/scene/manifest.json --output NEW_RENDER_DIRECTORY
```

Use `work/time_milestones/milestone_000.pt` for 300 seconds and `work/resume.pt`
for 900 seconds. Load the original 318-frame dataset and verify its identity to
recover normalization; do not recompute normalization from the held-out cameras.
Restore the checkpoint's active SH degree from its update count. Held-out cameras
have no exported training masks; full-frame RGB evaluation does not require them.

In the separate Adaptive environment:

```text
python tools/render_adaptive_sfm.py --input EXP/data --adaptive-repo ADAPTIVE_REPO --model EXP/phase3_clean/adaptive/model --checkpoint time_300 --frozen-times 0 0.5 1 --output NEW_RENDER_DIRECTORY
```

Use `final` for 900 seconds. Set `TI_OFFLINE_CACHE_FILE_PATH` to an experiment
cache directory. The worker uses the native model loader, cameras, point-cloud
query and renderer. It requires the already published SfM PLY so the external
loader cannot generate files inside the immutable dataset.

Each renderer writes unquantized float32 RGB arrays, PNG previews and a manifest
with frame/camera/source/checkpoint identities. Output directories must be new.
Both renderers retain their training background (Static white, Adaptive black),
world-frame SH and native rasterizer. Their numerical rendering implementations
are not forcibly replaced with a common renderer.

Compute all metrics in the Modal environment with the existing evaluator kernel:

```text
python tools/evaluate_adaptive_sfm.py metrics --input EXP/data --render static_300=RENDERS/static_300 --render static_900=RENDERS/static_900 --render adaptive_300=RENDERS/adaptive_300 --render adaptive_900=RENDERS/adaptive_900 --torch-home scene_library/_shared/tools/torch --output NEW_METRICS_DIRECTORY
```

`--torch-home` requires local pretrained AlexNet weights; this command does not
implicitly download weights. Omitting it explicitly omits LPIPS. Outputs include
per-frame CSV and arithmetic frame means plus pooled PSNR/RMSE. Predictions are
float32, clamped to [0,1], at native resolution; no PNG/video quantization, resizing,
exposure fitting, crops or color linearization are used for metrics. SSIM uses the
existing 11-pixel Gaussian window with sigma 1.5; LPIPS is Alex v0.1, spatial=True.

The separately callable `visuals` subcommand takes the same four `--render`
bindings and exports review videos and first/middle/last contact sheets. The 44
held-out frames are discontinuous in source time, so these are labeled 2 FPS
review slideshows, not 30 FPS reconstructed recordings. Detail crops use one fixed
rectangle chosen on input pixels before inspecting reconstructions.

## Interpretation

- Matched-time Adaptive reconstruction uses `index/(frame_count-1)`, not the
  loader's `index/frame_count`. Static has no temporal conditioning.
- A preselected frozen `t=.5` is the primary static-snapshot diagnostic; `t=0/1`
  check sensitivity. Do not select the best frozen time using held-out scores.
  The native model rounds `.5 * 361` to frame 180.
- Frozen-pose errors against original sweep frames also contain pose differences.
  They do not measure deblurring. No sharp, motion-matched ground truth is available.
- Matched-time 4D scores alone cannot establish that an extracted static scene is
  better. Compare the frozen renderings and disclose this limitation explicitly.
- Preserve native initialization/batch/loss differences and actual budget times
  from Phase 3, including its recorded checkpoint I/O costs. These experiments do
  not establish published-dataset superiority over the broader literature.

The scene-owned `phase4/` directory holds run commands, logs, render artifacts,
metrics, visual review and the final report. Successful loading/metrics and visual
inspection are recorded separately. An export/import into the mainline static
scene representation remains Phase 5.

## Completed experiment: 2026-09-26

Experiment: `scene_library/bush/experiments/adaptive_sfm_540p_20260926_001/phase4`.
Only `phase3_clean/` checkpoints participate; the interrupted earlier `phase3/`
run is excluded. All numbers below are arithmetic means over the same 44 held-out
frames. These static models were trained from SfM for this benchmark; they are not
the previously published coarse scene or the later A/B scene-refinement runs.

### Matched-time reconstruction

| Method | Training budget | PSNR dB (higher) | SSIM (higher) | RMSE (lower) | LPIPS (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Modal static | 300 s | 26.5088 | 0.865689 | 0.047540 | 0.101256 |
| Adaptive, frame time | 300 s | 25.6318 | 0.847641 | 0.052605 | 0.133837 |
| Modal static | 900 s | 23.3000 | 0.750069 | 0.068868 | 0.246335 |
| Adaptive, frame time | 900 s | 28.3963 | 0.896378 | 0.038190 | 0.072587 |

At 300 seconds, static has better aggregate results on all four metrics; it wins
PSNR on 40/44 frames and LPIPS on 44/44. At 900 seconds, Adaptive wins every metric
on every frame; mean PSNR is 5.0963 dB higher than static at the same budget.

Static deteriorates from 300 to 900 seconds: PSNR falls 3.2088 dB and all four
metrics worsen on all 44 frames. Adaptive improves 2.7645 dB over its own 300-second
state. Adaptive 900 seconds also exceeds Static 300 seconds by 1.8875 dB, but that
comparison has unequal compute budgets. These are paired descriptive results
from one scene and one seed, not a multi-dataset statistical claim.

### Frozen-time diagnostic

| Adaptive budget | Frozen time | PSNR dB | SSIM | RMSE | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 300 s | 0 | 23.0174 | 0.754140 | 0.072053 | 0.174787 |
| 300 s | .5 (primary) | 24.2309 | 0.812943 | 0.062286 | 0.146177 |
| 300 s | 1 | 24.1804 | 0.814007 | 0.062295 | 0.147130 |
| 900 s | 0 | 21.5011 | 0.683296 | 0.085724 | 0.177225 |
| 900 s | .5 (primary) | 22.4001 | 0.755426 | 0.076935 | 0.134960 |
| 900 s | 1 | 22.4798 | 0.758859 | 0.075737 | 0.131578 |

Freezing the 900-second model at the preselected middle time retains useful fine
texture, but its cross-time PSNR is 22.4001 rather than the matched-time 28.3963.
Against Static 900 it has better SSIM/LPIPS but worse PSNR/RMSE. Against Static 300
all aggregate metrics are worse, although the budgets differ. Frozen predictions
are compared with moving original poses, so these scores do not establish the
quality of deblurring or rule out a useful static snapshot.

### Visual review and conclusion

Inspected held-out frames 28, 190 and 351, full images and the same input-selected
detail crop. Adaptive 900 matched-time preserves more leaf edges and flower detail
than Static 300 in these inspected crops. The frozen .5 snapshots also retain
detail, with visible differences in flower/leaf positions. This is visual evidence
on those views, not a geometry-accuracy measurement or proof of view-independent
surface consistency.

Static 900 shows substantial smoothing in both foliage and stationary background
surfaces. Thus its entire deficit cannot be attributed to moving foliage or to
Adaptive removing motion blur. Phase 3 already recorded a loss rebound near the
static opacity-reset/culling schedule; that is a diagnostic lead, not an identified
cause. No corrective retraining or recipe changes were made during evaluation.

The result supports Adaptive for this sweep's matched-time dynamic reconstruction
at 900 seconds. It does not yet justify replacing the mainline static base with an
extracted snapshot. Keep the stronger Static 300 state visible in any follow-up;
diagnose the static training regression before treating Static 900 as a healthy
long-budget baseline. Static-snapshot import and controlled downstream evaluation
remain a separate Phase 5 decision.

### Delivery, timing and verification

- `evaluation/metrics.json`: mean and pooled results, metric definitions and LPIPS
  weight identities; `evaluation/per_frame.csv`: 440 rows across 10 variants.
- `review/heldout_300s.mp4` and `heldout_900s.mp4`: input/static/Adaptive panels.
  `review/frozen_300s.mp4` and `frozen_900s.mp4`: static and three frozen times.
  All are 44-frame, 2 FPS review slideshows, verified with ffprobe.
- `review/`: 24 full/detail PNG contact sheets. `analysis/`: three additional
  detail sheets comparing input, Static 300, Adaptive 900 matched and frozen .5.
  These additional sheets explicitly compare unequal budgets.
- `analysis/paired_comparisons.json`: framewise wins and mean/median deltas.
  `analysis/verification.json`: post-evaluation input, model and artifact checks.
- Each successful `*_command.json` records exact argv, return code and wall time.
  Commands took 11.70/11.43 s for Static 300/900 rendering, 17.57/20.37 s for
  Adaptive 300/900 rendering (each includes all four time variants), 14.95 s for
  common metrics and 17.61 s for review exports. These are end-to-end command
  times, not renderer-only FPS. Training times remain those in Phase 3.
- All 1010 prepared input hashes and Phase 3 training-source hashes match; all
  four checkpoint hashes match the rendering records. All 880 float/PNG output
  files, metric files and review exports pass their saved hash checks. Frozen
  frame/time membership was independently checked (effective frames 0/180/361).
  Static normalization residual is at most 1.407e-6 in normalized camera units.
- Five focused local tests passed (camera normalization, frame/time/hash handoff,
  floating-point metrics, native mapping and CLI loading), plus `git diff --check`.
  A separate read-only review found no blocking camera/time/render/metric error.
- The first static render attempt failed before publication because held-out rows
  intentionally have no training-mask hash. The adapter now uses empty unavailable
  mask metadata; its synthetic test covers this case. The failed attempt log is
  retained, and all four subsequent rendering commands completed successfully.

Fairness limits remain the native initial point counts, batch sizes, backgrounds,
regularizers, renderers and checkpoint cadence described above and in Phase 3.
The common SfM includes all sweep frames and three fixed references, while RGB
optimization uses only the 318 training frames. No sharp motion-matched ground
truth, independent-scene generalization test or static geometry ground truth was
available. No training, catalog redirect, static-scene import or viewer launch was
performed in Phase 4. Scene-owned outputs are Git-ignored; this document and the
two tools are the versionable record of the procedure and numerical findings.
