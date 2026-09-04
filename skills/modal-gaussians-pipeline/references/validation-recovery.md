# Validation and recovery

Read before executing the pipeline. Contents: input/resource preflight; CUDA probe; stage gates; headless final readiness; recovery and handoff.

## Input and resource preflight

Use `modal_gaussians.data.sequence.validate_image_mask_sequence` for each fixed view. It checks all PNG pairs without retaining the full decoded sequence. Its keyword arguments are `image_dir`, `mask_dir`, `fps_hz`, and `reference_frame_name`. Verify:

- RGB is uint8, three-channel PNG; masks are uint8 single-channel binary PNGs with the same dimensions and matching stems. Foreground is positive. Directories may not contain unrelated non-PNG files.
- Lexicographic frame order is temporal order; zero padding is important. At least three frames exist. Do not guess capture FPS from filenames. If frames were decimated, use their actual sampling FPS.
- Reference stem resolves to an actual image/mask pair in that view. Hash/check these exact files against the copies used by joint COLMAP. No resizing, crop, changing intrinsics, or alternate reference image may be hidden between the flow and static/topology branches.
- Labels are unique and accepted by `colmap.py`; preserve their order in every command. The intended multi-view solve needs independent observations, not duplicated views under different names.
- Sweep sampling is explicit. Current COLMAP camera grouping assumes one shared sweep intrinsics group and another shared reference group. Different cameras/resolutions/crops within one group require a user-approved input or implementation correction, not silent grouping.
- Call `FrequencySelectionConfig(minimum_hz, maximum_hz, step_hz, count).frequency_grid()` to validate an inclusive aligned grid and K. Every candidate must satisfy every view's Nyquist limit. Record per-view duration and warn about weakly observable low frequencies or an excessively rich K-prefix; do not silently change the request.
- Stabilization is reference-anchored. Its optional warped frames/masks and validity region belong to the flow artifact. Keep the original reference RGB available for the GUI.

Estimate sizes from the actual T/H/W before allocation. Per view, flow alone uses `8*T*H*W` bytes; rFFT uses `16*(floor(T/2)+1)*H*W`; selected dense modes use `16*K*H*W`. These are lower bounds: decoded grayscale/masks, FFT temporaries, stabilization buffers, hashing, and solver matrices add memory/disk pressure. Initial flow processing retains full sequences; a later mmap output does not make this stage streaming. Design storage is `16*P*K` bytes and measurements `16*P*K` bytes, with potentially different P. Include static checkpoint/Adam/densification and temporary publication headroom. Check available RAM, GPU memory, scratch quota and allocation duration. Do not discover infeasibility by launching all views in parallel.

## CUDA/environment probe

First record interpreter, package import path, Python/PyTorch/CUDA/gsplat/Viser versions, GPU name, `nvcc --version`, `pip check`, and resolved COLMAP binary. The target is Python 3.11, PyTorch 2.7.1+cu128, toolkit/runtime 12.8, gsplat 1.5.3, Viser 1.1.0 from this repository's environment definition. CUDA runtime and toolkit must both match 12.8; the GPU driver's advertised maximum CUDA version may be newer.

Run this inline in the selected environment (or in an agent-created temporary file, then remove only that file). This is a two-Gaussian forward/backward probe, not a training result. It uses the project's loader so the existing Windows build compatibility path is exercised. On Linux, ensure a compatible compiler and CUDA toolkit exist inside the allocated environment. Retain build-failure output; do not bypass gsplat with a fake renderer.

```python
import importlib.metadata
import torch
import torch.version as torch_version
from modal_gaussians.static import _load_gsplat_rasterization

assert torch_version.cuda == "12.8", torch_version.cuda
assert torch.cuda.is_available(), "CUDA device unavailable"
assert importlib.metadata.version("gsplat") == "1.5.3"
assert importlib.metadata.version("viser") == "1.1.0"
device = torch.device("cuda")
means = torch.tensor([[-0.1, 0.0, 2.0], [0.1, 0.0, 2.5]], device=device, requires_grad=True)
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device, requires_grad=True)
scales = torch.full((2, 3), 0.08, device=device, requires_grad=True)
opacities = torch.full((2,), 0.7, device=device, requires_grad=True)
colors = torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.4, 0.9]], device=device, requires_grad=True)
K = torch.tensor([[[40.0, 0.0, 16.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]]], device=device)
render = _load_gsplat_rasterization()
rgb, alpha, _ = render(
    means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
    viewmats=torch.eye(4, device=device)[None], Ks=K, width=32, height=32,
    packed=False, render_mode="RGB", rasterize_mode="classic", camera_model="pinhole",
)
assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
assert alpha.max() > 0
(rgb.square().mean() + alpha.mean()).backward()
for value in (means, quats, scales, opacities, colors):
    assert value.grad is not None and torch.isfinite(value.grad).all()
torch.cuda.synchronize()
print("CUDA forward/backward passed", torch.__version__, torch_version.cuda,
      torch.cuda.get_device_name(device))
```

Importing gsplat alone is not this probe. A program failure here during an actual experiment's preflight triggers the [immediate stop-and-report rule](../SKILL.md#repair-and-stopping-rules), just like a real-data stage. This does not prohibit iterative development checks during a separate authorized code-change task. Once diagnosis is authorized, inspect interpreter/PATH, toolkit, compiler, extension build and GPU compatibility before spending time on the real pipeline. Never relax the CUDA 12.8 constraint to obtain a passing import.

## Stage gates

Every loader below is under `modal_gaussians`. Use it rather than manually deciding that the expected filenames are enough. Most verify hashes/array schemas and many cross-links, but also compare upstream identities/settings to the current run contract. For flow, call `flow_artifact_identity()` explicitly and record/compare it. Chunk finite/statistical checks for dense data instead of converting all mmap arrays into RAM.

| Stage | Loader | Evidence to record before continuing |
| --- | --- | --- |
| Flow | `flow.artifact.load_flow_analysis_artifact` and `flow_artifact_identity` | T/H/W/FPS/ref identity; nonempty valid mask; finite flow/spectrum; exactly zero reference flow; displacement percentiles, mask overlay and several representative frames. Inspect stabilization if used. |
| COLMAP | `static.load_static_dataset` | Every requested sampled sweep/ref registered; nonempty finite sparse cloud; reference labels; rigid poses and finite K; no foreground mask passed to feature extraction; copied RGB/masks unchanged. Check a few point reprojections and camera layout. |
| Static | `static.load_static_scene(path, "cpu")` | Pure-tensor `weights_only=True` bundle; FG/BG identities/counts; training step/epoch/termination summary; sweep and refs used; finite loss and reasonable alpha/depth; every ref and representative sweep GT/render pairs. All-white/empty foreground or obviously wrong cameras are blockers, not a metric warning to ignore. |
| Topology | `topology.load_observation_topology` | Per-view samples and contributor coverage; indices strictly within the frozen foreground domain; matching K/ref pixel geometry; finite weights and valid offsets. |
| Frequency | `frequency.load_frequency_selection` | Exactly requested K unique candidates in greedy order; per-prefix macro/worst-view R2, rank, conditioning, energy and sample counts. Weak fit is a reported limitation; invalid/no observable motion is not successful reconstruction. |
| Dense modes | `modes.load_complex_2d_modes` | Per-view complex64 `[K,H,W,2]`, unchanged order, signs/window/time convention. Compare selected topology pixels to direct exact-DFT values for a few modes, including an off-bin candidate when present. |
| Measurements | `measurements.load_gaussian_measurements` | `[K,P,2]`, exact agreement with dense fields at sampled topology pixels and correct sample/view order. |
| Structure graph | `structure_graph.load_observed_structure_graph` | Candidate/kept edges, component sizes, isolated nodes and accepted components; filtering reasons. Empty or unusable structure needs diagnosis, not automatic threshold relaxation. |
| Rigid | `rigid.load_rigid_modes` | Per-mode identifiable views/complex alphas, overlap, solver residuals/rank/conditioning, trusted components and trusted Gaussian counts. Zero trusted support cannot be disguised as solved motion. |
| Motion fill | `motion_fill.load_completed_modes` | Finite complex phi with frozen FG indexing; per-mode counts of trusted rigid, promoted rigid, pointwise fill and unresolved. Verify trusted seeds unchanged. Unresolved zeros are explicitly unsolved. |
| Rendered design | `rendered_design.load_rendered_modal_design` | Sample counts per view, `[P,2,2K]`, real/imaginary sign and finite/nonzero projected energy; built-in render-linearity checks passed. All-zero columns will also break the spectrum data preparation. |
| Direct coordinates | `direct_coordinates.load_direct_modal_coordinates` | Per-view frame offsets/FPS and mean-zero q; fit residual/R2 and conditioning; reconstruction uses `q(t)-q(ref)`; finite values across all frames. |
| Physics post-fit | `physics_coordinates.load_physics_modal_coordinates` | Same K/view/FG identities; direct-vs-post-fit flow residuals/R2, coordinate deviation, oscillator/forcing diagnostics. Report regularization tradeoffs without expecting every fit metric to improve. |
| Materialized result | `result.load_modal_result` | Static/mode/design/direct/physics cross-identities; exact coordinate kind; every linked source accessible; manifest result identity. Then run the headless readiness check below. |

Do not invent dataset-independent PSNR/R2/trust-coverage acceptance thresholds. Respect user-supplied criteria. Report measured quality and limitations and continue with usable provisional candidates; if a physical/observability failure prevents a meaningful result, ask for the necessary scientific decision. Structural corruption, mismatched identities and invalid geometry always block downstream consumption.

## Final readiness without running Viser

A linked result manifest alone is insufficient for this repository's full viewer. The spectrum panel additionally loads measurement/topology/dense-mode/flow artifacts, the full rFFT, and **original reference RGBs from the image directories recorded in each flow manifest**. Keep those files available at their bound paths.

After all stage gates pass, run this headless check with the materialized result path as `sys.argv[1]`. It uses the actual viewer's data-preparation class but never constructs `ModalViserViewer`, `ViserServer`, or a browser. It validates all spectrum inputs, camera data, GPU tensor loading, finite deformations at representative frames, and one real static-scene rasterization per reference view. It does not validate UI behavior. The rasterization check is required because gsplat's Windows JIT backend may otherwise be initialized for the first time inside a Viewer worker.

```python
import json
import sys
import torch
from modal_gaussians.vis.viewer import ModalViewerData
from modal_gaussians.static import cameras_from_scene_manifest

data = ModalViewerData(sys.argv[1], device="cuda")
assert len(data.cameras) == len(data.result.manifest["views"])
reference_cameras = {
    camera.label: camera
    for camera in cameras_from_scene_manifest(data.scene.manifest)
    if camera.role == "reference"
}
with torch.no_grad():
    for index, view in enumerate(data.result.manifest["views"]):
        frames = {0, int(view["reference_frame_index"]),
                  int(view["frame_count"]) // 2, int(view["frame_count"]) - 1}
        for frame in sorted(frames):
            means = data.deformed_means(data.coordinate(index, frame))
            assert means.shape == (data.scene.foreground.count, 3)
            assert torch.isfinite(means).all(), (view["label"], frame)
        camera = reference_cameras[view["label"]]
        rendered = data.scene.render_deformed(
            camera,
            data.deformed_means(
                data.coordinate(index, int(view["reference_frame_index"]))
            ),
        )
        assert rendered["rgb"].shape == (camera.height, camera.width, 3)
        assert all(torch.isfinite(value).all() for value in rendered.values())
torch.cuda.synchronize()
print(json.dumps({
    "status": "viser_ready",
    "viewer_started": False,
    "modal_result_identity": data.result.manifest["modal_result_identity"],
    "views": [view["label"] for view in data.result.manifest["views"]],
    "mode_count": len(data.frequencies_hz),
    "foreground_count": data.scene.foreground.count,
}, indent=2))
```

Capture the readiness output in the run log. A temporary Python file is acceptable when the shell cannot conveniently pass this snippet and its argument; create it with the approved file editor, and remove only that disposable file afterwards. Do not launch the CLI `viewer` command as a test. If there is no GPU on the intended viewing host, state that requirement; a CPU-only artifact load is not proof of runtime readiness on a different host.

The two Python blocks are probes/templates, not persistent test-suite additions. Check project source before execution if these internal APIs change; do not make the implementation conform to stale probe assumptions. A program failure during an actual experiment's readiness check requires a report and pause. Ordinary code editing, inspection-command corrections, and temporary development tests may be fixed and rerun within scope without this gate.

## Resume, output ownership, and code repairs

The [experiment-error stop-and-report gate](../SKILL.md#repair-and-stopping-rules) takes precedence over every recovery action below during actual pipeline execution. After an experiment program error, these are recovery options to explain to the user, not permission to execute them. Wait for explicit instructions about the reported failure; neither an existing checkpoint nor an apparently trivial fix permits an automatic retry. This is not a restriction on iterative fixes during ordinary user-requested development.

| Situation | Correct action |
| --- | --- |
| Program error in an actual experiment's compute stage or validation command | Immediately stop pipeline progression, preserve and report the error/logs/completed progress, record `failed` with next action `awaiting_user_instruction`, and end the turn. Do not diagnose further, patch, retry, or continue until explicitly instructed. |
| Successful output already exists | Reload, verify requested inputs/config/code match, and skip. Never append or overwrite. |
| Static training interrupted before export | Reuse its `work/static/resume.pt` with the identical command plus `--resume`. Reject input/config mismatch. This includes epoch/batch/seed settings. |
| Static export exists but automatic QA failed | Validate bundle, run `static render` into a fresh QA directory, and retain the error in the run record. Do not retrain or overwrite the bundle just to fix QA. |
| Rigid/motion-fill interrupted | Repeat the identical command with the same work directory; validated `mode_NNN.npz` entries resume automatically. Do not add an unsupported `--resume` flag. |
| Other compute stage interrupted | There is no general resume flag. Inspect output/temporary state and live processes. Recompute the incomplete stage into a new unused attempt path if necessary; reuse its valid ancestors. |
| COLMAP fails | Preserve the external CLI progress log and captured stdout/stderr: COLMAP output streams there as it arrives, while its temporary workspace/internal log is still removed in `finally` on failure. The exception includes only a log tail. Report before a real-data retry; do not assume its failed workspace survives. |
| Output directory exists but is invalid | Do not delete user data or overwrite it. Record the failure and choose a fresh target; repair the actual cause and update downstream arguments. |
| Artifact hash/identity differs | Find the changed producer/input/config; rerun the earliest affected stage and descendants. Never edit identities to claim equivalence. |
| Out of memory / disk / walltime | A traceback or crash still triggers the stop-and-report gate. Otherwise measure the actual bottleneck within scope and use authorized additional resources; obtain explicit authorization before an implementation fix. Changes to K, images, resolution, epochs, graph thresholds or scientific settings require direction. Even performance-only CLI changes must be checked for identity/resume effects. |
| Source code changed after an artifact was produced | Decide whether its behavior affects that artifact. Rebuild the affected suffix under new attempt paths, even if old loader hashes still pass. Save code/config provenance; do not silently mix implementations. |
| Inputs/results moved to another host/path | Validate the full chain there. Absolute upstream paths are embedded throughout, not only in the final result. Rematerializing one manifest cannot repair all broken nested links. Do not hand-edit scientific identities. |

For a demonstrated implementation defect encountered in an actual experiment, report it first and wait. After the user authorizes repair, make a minimal patch and iteratively run/fix focused development regressions using synthetic/temporary data. Retry the real stage only if the user's instructions also authorize retry or continuation; another program failure in that resumed experiment requires another stop and report. Record what failed and why the fix preserves the mainline. Preserve the user's edits. Passing a temporary test does not mark its real-data stage complete. Do not create or retain a permanent `tests/` directory for these checks.

## Run record and final handoff

Keep one concise `run-status.md` with a stage table (`pending`, `running`, `validated`, `failed`, `blocked`), attempts/paths/identities, metric summaries, logs, code/config changes, process details, and next action. For a code-error pause, record `awaiting_user_instruction` as the next action; do not present a retry as already authorized. Write `viser_ready / viewer_not_started` only after the final check above. Include the exact launch command, expected local URL (not active), required interpreter/GPU, and the fact that linked ancestors/original reference RGBs must remain available.

Do not claim that Viser was visually tested or that the scientific result was approved. If blocked, retain completed outputs and state the smallest missing input/decision plus a precise resume command. Merely writing a completion note or producing a synthetic scene never satisfies a real-data run request.
