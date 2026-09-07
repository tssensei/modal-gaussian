# Pipeline commands

These commands follow `src/modal_gaussians/cli.py`. Reuse a known working invocation; consult `--help` only for uncertain or changed flags. Run them separately and apply the stage gates in [validation-recovery.md](validation-recovery.md). This document is not a script to execute wholesale.

If a program fails while executing an authorized experiment, follow the [autonomous repair and recovery rules](../SKILL.md#autonomous-repair-and-recovery): preserve evidence, diagnose and repair, verify the fix, then retry/resume and continue. The recovery commands below are part of the run authorization; do not wait for new instructions after each error. Preserve the run contract and validate prerequisites before consuming their outputs.

## Bind inputs once

Copy the skill's JSON template to a run-owned `run-spec.json`, resolve its required nulls, and add/remove view records to match the dataset. The two template views are illustrative, not a mandatory view count. For fixed complex view alignment, validate the actual independent camera observations. The default motion representation is the accepted neural field plus fragment propagation.

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

## Live text progress

For each compute command below, insert `--log-file "$RunRoot/logs/STAGE_ATTEMPT.log"` immediately after `-m modal_gaussians.cli`, before `flow`, `static`, or another subcommand. Choose the actual stage/attempt name, record it in `run-status.md`, and keep logs outside artifact output directories. Logs append rather than overwrite; use a new attempt name for each retry. Continue retaining stdout/stderr for output from third-party libraries.

Training reports step/epoch/batch, loss, PSNR, SSIM and FG/BG counts; other instrumented loops report frames, columns, selected frequencies, views or modes. Reports flush immediately, normally at most once per five seconds, plus the first completed unit and phase/epoch/mode boundaries. ETA uses this process's completed work (not pre-resume steps) and is an estimate. A phase's `100%` is not artifact validation or whole-pipeline completion.

After the log exists, the user can follow it in another PowerShell terminal:

```powershell
Get-Content -LiteralPath "$RunRoot/logs/static_train_attempt01.log" -Tail 20 -Wait
```

Ctrl+C ends only this log follower. No preview images, server, or Viser launch is needed for numeric monitoring.

## 0. Environment and input preflight (new environment or actual failure only)

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

Output: version-7 per-view `manifest.json`, `flow.zarr/` `[T,H,W,2] float32`, `mask_union.npy`, and `spectrum.zarr/` `[floor(T/2)+1,H,W,2] complex64`, plus optional stabilization data. The Zarr v3 directories use lossless Zstd compression and sharding; every pixel and rFFT bin is retained. `none` and `weighted-gaussian` are the unchanged smoothing choices. Keep the full rFFT for the spectrum panel; dense selected modes below do not replace it.

Use the repository artifact loader, not hardcoded `.npy` paths. It still accepts version-6 NPY artifacts without changing their identities. New runs require the pinned `zarr==3.1.6`; CLI arguments and run-spec fields are unchanged. Flow generation streams frames, and transforms/consumers read bounded tiles or selected pixels. Do not call `np.asarray` on a complete Zarr array. Copy entire `.zarr` directories, including all shards and metadata. Compression ratio depends on the data; estimate disk headroom conservatively, not from a previous sparse-mask experiment. Existing outputs are not automatically converted, and a new flow identity cannot replace an old result's ancestor without rebuilding its descendants.

## 2. Joint COLMAP

```powershell
& $MgPython -m modal_gaussians.cli colmap prepare --frames $Spec.sweep.images --frame-masks $Spec.sweep.masks @ReferenceArgs --sample-stride $Spec.sweep.sample_stride --colmap-command $Spec.colmap_executable --output "$RunRoot/joint_colmap"
```

Output: copied RGB/semantic masks, `sparse/0/{cameras.bin,images.bin,points3D.bin}`, `cameras.json`, `point_cloud.ply`, `colmap.log`. Every sampled sweep frame and every fixed-view reference must register in one model. Reference labels become `references/LABEL.png`. COLMAP uses full-image features, not foreground feature masks.

## 3. Static 3DGS training

```powershell
& $MgPython -m modal_gaussians.cli static train --input "$RunRoot/joint_colmap" --work-dir "$RunRoot/work/static" --output "$RunRoot/static_scene" --epochs $Spec.static.epochs --batch-size $Spec.static.batch_size --num-fg $Spec.static.num_fg --num-bg $Spec.static.num_bg --mask-weight $Spec.static.mask_weight --fg-densify-stop-step $Spec.static.fg_densify_stop_step --bg-densify-stop-step $Spec.static.bg_densify_stop_step --max-bg-gaussians $Spec.static.max_bg_gaussians --seed $Spec.static.seed
```

Output: `static_scene/{manifest.json,tensors.pt,training_summary.json}`. Work state: `work/static/resume.pt`. Training also renders `work/static/qa` automatically. Defaults: 100 epochs, batch 8, initialization caps FG 40,000 / BG 80,000, seed 42. The objective is `0.8*L1 + 0.2*(1-SSIM) + 1.0*trimmed_L1(FG-mask)`, with a 7×7 erosion kernel and no scale regularizer. BG densification stops at step 1,000 and is capped at 160,000 Gaussians; FG densification stops at 4,000. Depth supervision remains disabled until its input artifact is specified and implemented.

Resume interrupted training by appending `--resume` to this exact command only after checking unchanged input/config and that the final bundle does not already exist. If export succeeded but QA failed, diagnose and fix QA, validate the existing bundle, and rerender QA rather than retraining merely because the original command failed after export.

## 4. Static QA is user-evaluated

Do not inspect, rerender or produce additional QA images during ordinary runs.
Use successful static export and its built-in checks, then continue. If the user
asks for visual diagnosis or a rendering operation actually fails, address that
specific request/failure without retraining valid static weights unnecessarily.

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

## 11. Prepare once, then produce the requested 3D modes

Stop after final modes by default. Ordinary run authorization does not authorize fitting per-frame modal coordinates or evaluating a reconstructed video. Resolve the accepted neural/fragment configuration instead of inheriting the obsolete rigid-fill recipe below.

If a validated neural baseline result exists, import its fixed inputs and resolved parameters once:

```powershell
& $MgPython -m modal_gaussians.cli motion prepare-neural --from-result $Spec.motion.baseline_result --cache-dir $Spec.motion.cache_dir --output "$RunRoot/prepared"
```

For new data with the upstream artifacts from stages 1–10, prepare explicitly. `--config` here is a flat JSON object of resolved neural settings:

```powershell
& $MgPython -m modal_gaussians.cli motion prepare-neural --scene "$RunRoot/static_scene" --topology "$RunRoot/topology" --measurements "$RunRoot/measurements" --graph "$RunRoot/observed_graph" --alignment-from "$RunRoot/rigid_modes" --config $Spec.motion.prepare_config --cache-dir $Spec.motion.cache_dir --output "$RunRoot/prepared"
```

Use the chosen path as `$Prepared`. Reuse an existing prepared snapshot without rebuilding matching inputs. The optional iteration JSON uses `neural`, `fragment`, and `design` sections; omitted values inherit the snapshot. Pass `--config` only when a real override file is provided.

```powershell
$Prepared = "$RunRoot/prepared"
$IterationArgs = @()
if ($Spec.motion.iteration_config) { $IterationArgs = @('--config', [string]$Spec.motion.iteration_config) }
& $MgPython -m modal_gaussians.cli motion iterate-neural --prepared $Prepared @IterationArgs --output "$RunRoot/experiment" --stage modes
```

`modes` is the CLI default. It trains/reuses the raw neural field, applies the accepted fragment propagation, validates the final `neural_completed_modes` and records `modes_ready`. This is the successful end of the ordinary run. It does not build a preview or fit coordinates. Preserve shared cache/prepared dependencies.

## 12. Optional manual preview, only when requested

Reuse exactly the same experiment and settings:

```powershell
& $MgPython -m modal_gaussians.cli motion iterate-neural --prepared $Prepared @IterationArgs --output "$RunRoot/experiment" --stage preview
```

This adds rendered-design, an independent preview artifact with built-in source binding checks; it does not initialize Viewer data or inspect visual results. Rendering the modal image is not fitting modal coordinates. Full flow/spectrum loading and entire-spectrum statistics are not required for initialization. Do not append `--stage full` to validate this preview.

Hand off the following fully resolved command **without executing it**:

```powershell
& $MgPython -m modal_gaussians.cli viewer --preview "$RunRoot/experiment/preview" --work-dir "$RunRoot/work/viewer" --host $Spec.viewer.host --port $Spec.viewer.port --viewer-res $Spec.viewer.resolution
```

Use `127.0.0.1` by default. Give the expected local URL, labelled not running. Controls use manual gain/phase/oscillation; no stored video trajectory or coordinate-dependent flow score is implied.

## Historical compatibility, not part of the default run

Existing `motion fill`, `coordinates solve-direct`, `coordinates physics-fit`, `result materialize`, `iterate-neural --stage full` and `viewer --result` APIs remain available for old results or a later explicit request for those operations. Do not run them as automatic follow-ups to mode generation, preview checks, or generic “complete pipeline” instructions. Inspect current CLI help and the old artifact's own contract when such a request actually occurs; preserve the existing result and its scientific conventions.

## Linux / cluster translation

The Python CLI arguments and dependency order are identical. Activate the verified project environment inside the actual GPU allocation, set `MG_PYTHON` to its absolute interpreter and `RUN` to the run root. Construct ordered Bash arrays, for example:

```bash
VIEW_ARGS=(--view view1 "$RUN/flow/view1" --view view2 "$RUN/flow/view2")
REF_ARGS=(--reference view1 /absolute/view1/images/REF.png /absolute/view1/masks/REF.png --reference view2 /absolute/view2/images/REF.png /absolute/view2/masks/REF.png)
"$MG_PYTHON" -m modal_gaussians.cli topology build --scene "$RUN/static_scene" "${VIEW_ARGS[@]}" --pixel-stride 4 --candidate-count 4 --preselect-count 32 --alpha-min 0.05 --min-contribution 1e-12 --mask-erode-iters 1 --output "$RUN/topology"
```

Replace the example labels/ref paths with the resolved spec. For each command above, translate `& $MgPython` to `"$MG_PYTHON"`, `@ViewArgs` to `"${VIEW_ARGS[@]}"`, `@ReferenceArgs` to `"${REF_ARGS[@]}"`, and spec properties to their resolved argument values. Record the expanded argv before running. Do not use unquoted array expansion or `eval`.

For a future cluster viewing session, keep Viser on loopback and document only the user's known SSH forwarding route. If the viewer will run on the same SSH host, the shape is `ssh -N -L LOCAL_PORT:127.0.0.1:REMOTE_PORT USER@HOST`. A compute node behind a login host requires the actual permitted jump/forward arrangement; a tunnel to the login node's loopback will not reach a different node's loopback. Do not establish the tunnel during this workflow, invent hostnames, or claim that the viewer/tunnel was tested.
