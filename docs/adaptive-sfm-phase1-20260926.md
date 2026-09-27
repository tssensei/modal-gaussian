# Adaptive SfM comparison: Phase 1

Scope: prepare the runtime and common inputs, then validate actual loaders and
synthetic GPU forward/backward. No Gaussian training, reconstruction evaluation,
video export, snapshot import or viewer is launched by these tools.

## Inputs and comparison boundary

- External checkout: `../adaptive-spatio-temporal-gaussians`, upstream commit
  `f6020830e2ccb5825d0501bc2fd54c9139ee317e`.
- Source: `scene_library/bush/experiments/static540_da3_20260924_001/colmap_540p`.
- New experiment: `scene_library/bush/experiments/adaptive_sfm_540p_20260926_001`.
- 362 sweep frames, 960x540, actual 30 FPS; source indices 0,2,...,722 and
  timestamps 0 through 12.0333333333 seconds.
- Shared split: `llffhold=8`, `test_set_segment_length=4`, final three frames
  forced into training. 318 train / 44 test; test indices 28–31,60–63,...,348–351.
- Only sweep images participate in future RGB training. The existing SfM includes
  all sweep images and three fixed reference images; this is a shared calibrated
  SfM protocol, not a reconstruction using training images alone.
- Original coarse training also used fixed references and DA3 depth. It remains
  an engineering reference, not the matched RGB-only held-out baseline.

## Data preparation

Run in the existing Modal environment. Source paths are resolved through the
scene-store resolver. The output must not already exist; source inputs and catalog
entries are not changed. A failed preparation does not publish a partial dataset.

```powershell
$taskPython = 'C:\Users\zitengsong\anaconda3\envs\modal-gaussian\python.exe'
$taskRepo = 'C:\Users\zitengsong\Documents\school\research\modal-gaussian'
$taskExp = Join-Path $taskRepo 'scene_library\bush\experiments\adaptive_sfm_540p_20260926_001'
Set-Location $taskRepo

& $taskPython tools/prepare_adaptive_sfm.py `
  --input scene_library/bush/experiments/static540_da3_20260924_001/colmap_540p `
  --output "$taskExp\data"
```

The adapter resamples RGB and initial-partition masks to one centered pinhole
camera. Every output pixel must have valid source support. It transforms SfM 2D
observations and updates both directions of filtered tracks while preserving all
3D point coordinates/colors and raw camera poses. Numeric filenames represent the
uniform time grid; metadata separately records physical seconds, loader time
`i/362` and normalized model time `i/361`.

Adaptive uses flat images and SIMPLE_PINHOLE cameras. Modal uses the identical
training PNGs and SIMPLE_RADIAL cameras with zero radial coefficient. Its existing
foreground/background initialization still uses the resampled masks; this does
not introduce mask supervision. Points without retained observations remain in
both initialization clouds and are explicitly counted. Neither output is a new
SfM optimization.

The camera convention is COLMAP/gsplat pixel centers `(column+.5,row+.5)` with
principal point `(width/2,height/2)`. OpenCV remapping subtracts .5 from source
coordinates; Adaptive CUDA screen coordinates are correspondingly .5 smaller.
This difference is checked explicitly rather than changing the physical camera.

## Independent loader verification

After the external environment is available:

```powershell
& $taskPython tools/verify_adaptive_sfm.py `
  --input "$taskExp\data" `
  --source scene_library/bush/experiments/static540_da3_20260924_001/colmap_540p `
  --adaptive-repo 'C:\Users\zitengsong\Documents\school\research\adaptive-spatio-temporal-gaussians' `
  --adaptive-python 'C:\Users\zitengsong\anaconda3\envs\adaptive-gs\python.exe' `
  --output "$taskExp\loader_verification"
```

The verifier invokes each project's real loader in its own environment and checks
train/test membership, physical clock, decoded RGB, source camera poses, SfM
XYZ/RGB and projection equivalence. Adaptive reads the prepublished PLY instead
of generating a new file in the immutable input directory. Validation reports are
written separately from the data manifest. Use a fresh output directory when
repeating verification.

## Environment and recorded results

Environment installation/compatibility changes and measured Phase 1 checks are
recorded in this experiment's `environment/` and `PHASE1_REPORT.md`. The main Modal
environment remains separate. Python 3.12 is required by the external source;
RTX 5090 requires a compatible Blackwell PyTorch/CUDA stack. Successful installation
alone is not a GPU validation: require actual rasterizer forward/backward and
Taichi checks before marking Phase 1 complete.

## Later phases

Phase 2 adds trustworthy wall-clock budget/save control. Phase 3 runs matched
training budgets. Phase 4 evaluates the shared held-out set and inspects frozen
time renders. Phase 5 exports/imports a static snapshot only if those results
justify it. None of these later phases is implied by preparing Phase 1.
