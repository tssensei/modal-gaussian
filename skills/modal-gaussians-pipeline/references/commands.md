# Pipeline commands

These commands follow `src/modal_gaussians/cli.py`. Check current `--help` before a run; do not guess new flags. Run them separately and apply the stage gates in [validation-recovery.md](validation-recovery.md). This document is not a script to execute wholesale.

## Bind inputs once

Copy the skill's JSON template to a run-owned `run-spec.json`, resolve its required nulls, and add/remove view records to match the dataset. The two template views are illustrative, not a mandatory view count. For the intended multi-view rigid mainline, validate that sufficient independent camera observations actually exist.

The examples below are PowerShell. All input/output paths must be absolute on the execution host. `reference_frame` is a PNG stem without an extension. Validate the directories using the repository sequence loader before assuming the derived `.png` paths exist.

```powershell
$SpecPath = 'C:\absolute\run-spec.json'
$Spec = Get-Content -LiteralPath $SpecPath -Raw | ConvertFrom-Json
$MgPython = [string]$Spec.python
$RunRoot = [string]$Spec.run_root
Set-Location -LiteralPath $Spec.repo

$ViewArgs = @()
$ReferenceArgs = @()
foreach ($View in $Spec.views) {
    $ViewArgs += @('--view', [string]$View.label, "$RunRoot/flow/$($View.label)")
    $RefImage = Join-Path $View.images ($View.reference_frame + '.png')
    $RefMask = Join-Path $View.masks ($View.reference_frame + '.png')
    $ReferenceArgs += @('--reference', [string]$View.label, $RefImage, $RefMask)
}
```

Use native argument-array splatting. Do not concatenate input paths into executable shell strings or use `Invoke-Expression`. This avoids broken quoting and accidental shell interpretation of filenames. If a later attempt changes an artifact path, rebuild all argument arrays that refer to it.

## 0. Environment and input preflight

```powershell
& $MgPython -c "import sys, modal_gaussians; print(sys.executable); print(modal_gaussians.__file__)"
& $MgPython -m modal_gaussians.cli --help
& $MgPython -m pip check
& $Spec.colmap_executable -h
nvidia-smi
nvcc --version
```

Run `nvcc` from the resolved 12.8 toolkit, not an unrelated executable earlier on PATH. Check `torch.version.cuda`, driver compatibility, and an actual rasterization/backward as described in the validation reference. `nvidia-smi`'s advertised CUDA version is not the installed toolkit version.

If the environment is missing and setup is authorized, use `conda env create -f environment.yml` from the repository. If only the editable package registration is missing, use the selected interpreter's `-m pip install -e .` after reviewing dependencies. Inspect an existing environment before changing it; do not downgrade packages until the pipeline happens to run. Retain the CUDA 12.8 requirement.

Resolve COLMAP's actual binary separately. On the existing cluster its environment is named `colmap`; the project runs in `modal-gaussian`. `--colmap-command` must be an executable path/name, not the string `conda run -n colmap colmap`.

## 1. Flow and full-spectrum rFFT, one fixed view at a time

Select the next unresolved view by its agreed index; repeat after validating each result. Do not rerun a view whose artifact already validates against the current run contract.

```powershell
$View = $Spec.views[0]
$StabilizeArgs = @()
if ($View.stabilize) { $StabilizeArgs = @('--stabilize') }
& $MgPython -m modal_gaussians.cli flow analyze --images $View.images --masks $View.masks --fps $View.fps --reference-frame $View.reference_frame --smoothing $View.smoothing --sigma-b-px $View.sigma_b_px --sigma-c-px $View.sigma_c_px @StabilizeArgs --output "$RunRoot/flow/$($View.label)"
```

Output: per-view `manifest.json`, `flow.npy` `[T,H,W,2] float32`, `mask_union.npy`, and `spectrum.npy` `[floor(T/2)+1,H,W,2] complex64`, plus optional stabilization data. `none` and `weighted-gaussian` are the smoothing choices. Keep the full rFFT for the future/current spectrum panel; dense selected modes below do not replace it.

## 2. Joint COLMAP

```powershell
& $MgPython -m modal_gaussians.cli colmap prepare --frames $Spec.sweep.images --frame-masks $Spec.sweep.masks @ReferenceArgs --sample-stride $Spec.sweep.sample_stride --colmap-command $Spec.colmap_executable --output "$RunRoot/joint_colmap"
```

Output: copied RGB/semantic masks, `sparse/0/{cameras.bin,images.bin,points3D.bin}`, `cameras.json`, `point_cloud.ply`, `colmap.log`. Every sampled sweep frame and every fixed-view reference must register in one model. Reference labels become `references/LABEL.png`. COLMAP uses full-image features, not foreground feature masks.

## 3. Static 3DGS training

```powershell
& $MgPython -m modal_gaussians.cli static train --input "$RunRoot/joint_colmap" --work-dir "$RunRoot/work/static" --output "$RunRoot/static_scene" --epochs $Spec.static.epochs --batch-size $Spec.static.batch_size --num-fg $Spec.static.num_fg --num-bg $Spec.static.num_bg --seed $Spec.static.seed
```

Output: `static_scene/{manifest.json,tensors.pt,training_summary.json}`. Work state: `work/static/resume.pt`. Training also renders `work/static/qa` automatically. Defaults: 100 epochs, batch 8, initialization caps FG 40,000 / BG 100,000, seed 42. Loss is RGB-only `0.8*L1 + 0.2*(1-SSIM)`, with no scale regularizer.

To resume interrupted training, append `--resume` to this exact command only after checking unchanged input/config and that the final bundle does not already exist. If export succeeded but QA failed, validate that bundle and rerender QA; do not retrain merely because the original command failed after export.

## 4. Inspect static QA; render only if needed

Review the existing `work/static/qa/metrics.json` and images first. For missing/failed QA or a fresh comparison, use a new directory:

```powershell
& $MgPython -m modal_gaussians.cli static render --scene "$RunRoot/static_scene" --output "$RunRoot/work/static_qa_retry01" --role all
```

Output: all/FG/BG RGB, FG alpha and expected depth, GT/render pairs, and per-view PSNR/SSIM. This is a recovery/optional command, not an obligatory duplicate render after successful training.

## 5. Pixel-to-Gaussian observation topology

```powershell
& $MgPython -m modal_gaussians.cli topology build --scene "$RunRoot/static_scene" @ViewArgs --pixel-stride 4 --candidate-count 4 --preselect-count 32 --alpha-min 0.05 --min-contribution 1e-12 --mask-erode-iters 1 --output "$RunRoot/topology"
```

Output: `manifest.json`, `topology.npz`. Records FG-only pixel/contributor relationships and binds exact static/foreground/reference/flow identities. All `--view` repetitions from here onward use the same ordered labels and flow artifacts.

## 6. Greedy frequency prefix

```powershell
& $MgPython -m modal_gaussians.cli frequency select --topology "$RunRoot/topology" @ViewArgs --min-hz $Spec.frequency.min_hz --max-hz $Spec.frequency.max_hz --step-hz $Spec.frequency.step_hz --count $Spec.frequency.count --output "$RunRoot/frequency_selection"
```

Output: `manifest.json`, `selection.npz`, including candidate grid, greedy-ordered selected indices/frequencies, and prefix fit diagnostics. Selection uses topology samples and equal-view scoring. Do not sort the selected prefix by Hz. Range/step/K are required user choices, not hardcoded defaults.

## 7. Dense complex 2D modal fields

```powershell
& $MgPython -m modal_gaussians.cli frequency export-modes --selection "$RunRoot/frequency_selection" @ViewArgs --output "$RunRoot/complex_2d_modes"
```

Output: manifest and per-view `view_NNN.npy` `[K,H,W,2] complex64`, in greedy mode order. Full-resolution exact DFT at selected frequencies, not interpolation of rFFT bins. No amplitude clamp or mask clearing.

## 8. Measurement bank

```powershell
& $MgPython -m modal_gaussians.cli measurements build --topology "$RunRoot/topology" --modes "$RunRoot/complex_2d_modes" --output "$RunRoot/measurements"
```

Output: `manifest.json`, `measurements.npy` `[K,P,2] complex64`. This samples the dense fields at topology pixels; it is not one independent displacement per Gaussian before the solver.

## 9. Observed structure graph

```powershell
& $MgPython -m modal_gaussians.cli graph build --scene "$RunRoot/static_scene" --topology "$RunRoot/topology" --max-neighbors 8 --max-distance 0.008 --color-mad-multiplier 3 --depth-mad-multiplier 3 --depth-samples 5 --min-shared-views 1 --min-component-nodes 4 --min-component-edges 3 --output "$RunRoot/observed_graph"
```

Output: `manifest.json`, `graph.npz`, an unapproved candidate rigid-component graph. Distances are in normalized scene coordinates; do not reinterpret 0.008 as a raw COLMAP/metre distance.

## 10. Complex alpha offsets and rigid motion

```powershell
& $MgPython -m modal_gaussians.cli rigid solve --scene "$RunRoot/static_scene" --topology "$RunRoot/topology" --measurements "$RunRoot/measurements" --graph "$RunRoot/observed_graph" --work-dir "$RunRoot/work/rigid" --output "$RunRoot/rigid_modes"
```

Output: `manifest.json`, `rigid_modes.npz`. Work directory supports identity-checked per-mode recovery by repeating the same command; there is no `--resume` flag here. Preserve bounded complex view synchronization and trust/observability checks.

## 11. Motion fill

```powershell
& $MgPython -m modal_gaussians.cli motion fill --scene "$RunRoot/static_scene" --topology "$RunRoot/topology" --measurements "$RunRoot/measurements" --graph "$RunRoot/observed_graph" --rigid "$RunRoot/rigid_modes" --work-dir "$RunRoot/work/motion_fill" --neighbors 8 --max-distance 0.008 --max-anchor-hops 8 --observable-ratio 0.01 --ray-direction-fraction 0.8 --max-finite-drift 2.0 --output "$RunRoot/completed_modes"
```

Output: `manifest.json`, `completed_modes.npz`, including full-FG `phi` `[K,G_fg,3] complex64`, fill graph and support classes. Same-command per-mode recovery; no `--resume` flag. Unresolved motion remains zero and must be reported separately from solved motion.

## 12. Rendered modal design

```powershell
& $MgPython -m modal_gaussians.cli coordinates render-design --scene "$RunRoot/static_scene" --modes "$RunRoot/completed_modes" @ViewArgs --pixel-stride 2 --alpha-min 0.05 --mask-erode-iters 1 --modes-per-batch 8 --output "$RunRoot/rendered_design"
```

Output: `manifest.json`, `design.npy` `[P,2,2K] float32`, `samples.npz`. Columns are built through actual foreground feature rasterization, not a substitute topology-weight approximation. Preserve the real/imaginary sign convention.

## 13. Direct modal coordinates

```powershell
& $MgPython -m modal_gaussians.cli coordinates solve-direct --design "$RunRoot/rendered_design" @ViewArgs --ridge-relative 0.0001 --frame-chunk-size 64 --output "$RunRoot/direct_coordinates"
```

Output: manifest, `coordinates.npy` `[sum(T_view),K] complex64`, `diagnostics.npz`. Fit each view's time series independently and retain its frame offsets/FPS. Store mean-zero coordinates; flow reconstruction is reference-relative.

## 14. Physics coordinate post-fit

```powershell
& $MgPython -m modal_gaussians.cli coordinates physics-fit --input "$RunRoot/direct_coordinates" --damping-ratio 0.05 --forcing-weight 0.1 --forcing-difference-weight 0 --assigned-band-half-width-hz 0.1 --frame-chunk-size 64 --output "$RunRoot/physics_coordinates"
```

Output: manifest, `coordinates.npy`, `diagnostics.npz`. Compare direct and physics flow fits and oscillator residuals. Keep both artifacts. A regularized result need not improve raw flow R2; report the measured tradeoff without changing weights automatically.

## 15. Result materialization

```powershell
& $MgPython -m modal_gaussians.cli result materialize --scene "$RunRoot/static_scene" --modes "$RunRoot/completed_modes" --coordinates "$RunRoot/physics_coordinates" --output "$RunRoot/modal_result"
```

Output: linked `manifest.json`, not another copy of all tensors. It validates identities and references existing artifacts by absolute paths. Do not move/delete ancestors after materialization. For a user-requested direct-coordinate comparison, materialize another result pointing to `direct_coordinates`; do not overwrite the physics result.

## 16. Headless readiness check and handoff

Run the final headless data-readiness check in [validation-recovery.md](validation-recovery.md). It creates no Viser server and requires no browser. Stop execution after it passes.

Give the user this fully resolved command **without executing it**:

```powershell
& $MgPython -m modal_gaussians.cli viewer --result "$RunRoot/modal_result" --work-dir "$RunRoot/work/viewer" --host $Spec.viewer.host --port $Spec.viewer.port --viewer-res $Spec.viewer.resolution
```

Use `127.0.0.1` by default, not the CLI's broad `0.0.0.0` default. Give the expected URL `http://127.0.0.1:PORT` but label it as not yet running. Do not start a background process or try to open the page. This workflow validates data readiness, not browser behavior.

## Linux / cluster translation

The Python CLI arguments and dependency order are identical. Activate the verified project environment inside the actual GPU allocation, set `MG_PYTHON` to its absolute interpreter and `RUN` to the run root. Construct ordered Bash arrays, for example:

```bash
VIEW_ARGS=(--view view1 "$RUN/flow/view1" --view view2 "$RUN/flow/view2")
REF_ARGS=(--reference view1 /absolute/view1/images/REF.png /absolute/view1/masks/REF.png --reference view2 /absolute/view2/images/REF.png /absolute/view2/masks/REF.png)
"$MG_PYTHON" -m modal_gaussians.cli topology build --scene "$RUN/static_scene" "${VIEW_ARGS[@]}" --pixel-stride 4 --candidate-count 4 --preselect-count 32 --alpha-min 0.05 --min-contribution 1e-12 --mask-erode-iters 1 --output "$RUN/topology"
```

Replace the example labels/ref paths with the resolved spec. For each command above, translate `& $MgPython` to `"$MG_PYTHON"`, `@ViewArgs` to `"${VIEW_ARGS[@]}"`, `@ReferenceArgs` to `"${REF_ARGS[@]}"`, and spec properties to their resolved argument values. Record the expanded argv before running. Do not use unquoted array expansion or `eval`.

For a future cluster viewing session, keep Viser on loopback and document only the user's known SSH forwarding route. If the viewer will run on the same SSH host, the shape is `ssh -N -L LOCAL_PORT:127.0.0.1:REMOTE_PORT USER@HOST`. A compute node behind a login host requires the actual permitted jump/forward arrangement; a tunnel to the login node's loopback will not reach a different node's loopback. Do not establish the tunnel during this workflow, invent hostnames, or claim that the viewer/tunnel was tested.
