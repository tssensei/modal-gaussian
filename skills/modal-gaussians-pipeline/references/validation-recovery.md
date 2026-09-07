# Validation and recovery

Reference for actual errors, changed inputs/environments, or explicit diagnostic requests. Routine runs rely on built-in stage validation; do not repeat these checklists, inspect visuals, or create extra verification scripts. Contents: input/resource preflight; CUDA probe; stage gates; 3D-mode completion and optional headless preview; recovery and handoff.

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

Importing gsplat alone is not this probe. If this preflight fails, follow the [autonomous recovery rules](../SKILL.md#autonomous-repair-and-recovery): inspect interpreter/PATH, toolkit, compiler, extension build, and GPU compatibility; repair the cause and rerun the probe before expensive stages. Never relax the CUDA 12.8 constraint or replace the real renderer to obtain a passing check.

## Stage gates

Every loader below is under `modal_gaussians`. Use it rather than manually deciding that the expected filenames are enough. Most verify hashes/array schemas and many cross-links, but also compare upstream identities/settings to the current run contract. For flow, call `flow_artifact_identity()` explicitly and record/compare it. Chunk finite/statistical checks for dense data instead of converting all mmap arrays into RAM.

| Stage | Loader | Evidence to record before continuing |
| --- | --- | --- |
| Flow | `flow.artifact.load_flow_analysis_artifact` and `flow_artifact_identity` | T/H/W/FPS/ref identity; nonempty valid mask; finite flow/spectrum; exactly zero reference flow; existing flow diagnostics; visual assessment is left to the user. |
| COLMAP | `static.load_static_dataset` | Every requested sampled sweep/ref registered; nonempty finite sparse cloud; reference labels; rigid poses and finite K; no foreground mask passed to feature extraction; copied RGB/masks unchanged. Check a few point reprojections and camera layout. |
| Static | `static.load_static_scene(path, "cpu")` | Pure-tensor `weights_only=True` bundle; FG/BG identities/counts; training step/epoch/termination summary; sweep and refs used; finite loss and reasonable alpha/depth; saved numeric training diagnostics; do not inspect or rerender QA images during ordinary runs. |
| Topology | `topology.load_observation_topology` | Per-view samples and contributor coverage; indices strictly within the frozen foreground domain; matching K/ref pixel geometry; finite weights and valid offsets. |
| Frequency | `frequency.load_frequency_selection` | Exactly requested K unique candidates in greedy order; per-prefix macro/worst-view R2, rank, conditioning, energy and sample counts. Weak fit is a reported limitation; invalid/no observable motion is not successful reconstruction. |
| Dense modes | `modes.load_complex_2d_modes` | Per-view complex64 `[K,H,W,2]`, unchanged order, signs/window/time convention. Compare selected topology pixels to direct exact-DFT values for a few modes, including an off-bin candidate when present. |
| Measurements | `measurements.load_gaussian_measurements` | `[K,P,2]`, exact agreement with dense fields at sampled topology pixels and correct sample/view order. |
| Structure graph | `motion.rigid.structure_graph.load_observed_structure_graph` | Candidate/kept edges, component sizes, isolated nodes and accepted components; filtering reasons. Empty or unusable structure needs diagnosis, not automatic threshold relaxation. |
| Rigid | `motion.rigid.rigid.load_rigid_modes` | Per-mode identifiable views/complex alphas, overlap, solver residuals/rank/conditioning, trusted components and trusted Gaussian counts. Zero trusted support cannot be disguised as solved motion. |
| Neural preparation | `motion.neural.prepared.load_prepared` | Snapshot identity, fixed alpha/targets/normalization, source-frequency mapping, accepted baseline settings; matching independent caches. |
| Neural modes | `motion.neural.neural_modes.load_neural_completed_modes` | Exactly requested frequency slots, finite complex64 `[K,G,3]`, foreground order, fixed alpha, network reconstruction, geometry/controls/interpolation, direct/structural/unresolved roles. |
| Fragment propagation / final modes | `motion.neural.fragment_propagation.load_fragment_modes` or unified `motion.common.completed_modes.load_completed_modes` | Accepted attachment rules, immutable parent, unchanged hosts, explicit unresolved points, requested mode mapping. Record `modes_ready` and STOP for a default run. |
| Optional rendered-design | `rendered_design.load_rendered_modal_design` | Only for a requested preview: source identities, sample counts, `[P,2,2K]`, signs and render-linearity checks. No time-coordinate solve. |
| Optional preview | `motion.neural.preview.load_preview` | Independent scene/modes/design/prepared bindings; manual oscillation, no coordinates; publication checks only; no Viewer initialization or visual inspection. |

Do not invent dataset-independent PSNR/R2/trust-coverage acceptance thresholds. Respect user-supplied criteria. Report measured quality and limitations and continue with usable provisional candidates; if a physical/observability failure prevents a meaningful result, ask for the necessary scientific decision. Structural corruption, mismatched identities and invalid geometry always block downstream consumption.

## Final mode completion

Rely on the successful command and its built-in completed-modes validation; only when diagnosing an actual inconsistency, compare its ordered modes to the requested frequency/source slots. Check the scene/foreground identity, finite complex `phi`, geometry/control diagnostics, fixed alpha, support classes and fragment invariants using the loaders above. Report `modes_ready`; this is complete without time-dependent coordinates, rendered-design, full spectra or a Viewer.

Do not run coordinate fitting as a “remaining validation” step and do not require coordinate-dependent optical-flow R². Report the training modal-image residual and structure/support diagnostics already available from the mode artifact. Existing baseline flow-fit scores are historical references, not mandatory new metrics.

## Preview handoff

`--stage preview` produces the rendered-design and independent preview artifact,
with the pipeline's built-in source checks. Stop there and give the launch command.
Do not initialize `ModalViewerData`, cycle phases/views, inspect images, or run
headless visualization checks. Record `visualization_checked: false`; the user
launches Viser and evaluates quality. No coordinates, extra PNGs or full spectra.

## Resume, output ownership, and code repairs

The [autonomous recovery rules](../SKILL.md#autonomous-repair-and-recovery) apply throughout the authorized run. Use the actions below directly when needed to diagnose, repair, verify, and resume; a failure or repeated error does not itself require renewed permission. Preserve the scientific contract, output ownership, and actual execution permissions.

| Situation | Correct action |
| --- | --- |
| Program error in an actual experiment's compute stage or validation command | Preserve the failed attempt and logs, report progress without ending the turn, mark the stage `repairing`, diagnose and patch the cause, run focused checks, and retry/resume. Proceed downstream after validation succeeds. |
| Successful output already exists | Reload, verify requested inputs/config/code match, and skip. Never append or overwrite. |
| Static training interrupted before export | Reuse its `work/static/resume.pt` with the identical command plus `--resume`. Reject input/config mismatch. This includes epoch/batch/seed settings. |
| Static export exists but automatic QA failed | Validate bundle, run `static render` into a fresh QA directory, and retain the error in the run record. Do not retrain or overwrite the bundle just to fix QA. |
| Historical rigid/motion-fill interrupted (only when selected) | Repeat the identical command with the same work directory; validated `mode_NNN.npz` entries resume automatically. Do not add an unsupported `--resume` flag. |
| Neural iteration interrupted | Repeat the same prepared/config/output and selected `--stage modes` or requested `--stage preview`; valid stages and checkpoints are reused. Do not change to `full` during recovery. |
| Other compute stage interrupted | There is no general resume flag. Inspect output/temporary state and live processes. Recompute the incomplete stage into a new unused attempt path if necessary; reuse its valid ancestors. |
| COLMAP fails | Preserve the external CLI progress log and captured stdout/stderr: its temporary workspace/internal log is removed in `finally` on failure, and the exception contains only a log tail. Diagnose the command/input/environment or implementation defect, repair within the run contract, and rerun; do not assume the failed workspace survives. |
| Output directory exists but is invalid | Do not delete user data or overwrite it. Record the failure and choose a fresh target; repair the actual cause and update downstream arguments. |
| Artifact hash/identity differs | Find the changed producer/input/config; rerun the earliest affected stage and descendants. Never edit identities to claim equivalence. |
| Out of memory / disk / walltime | Measure the actual bottleneck, repair avoidable allocations or use streaming/chunking while preserving the objective, normalization, and data coverage, and resume with available authorized resources. Check identity/resume effects of implementation changes. Request input only if necessary resources are unavailable or the agreed scientific settings must change; do not silently reduce K, images, resolution, epochs, or graph thresholds. |
| Source code changed after an artifact was produced | Decide whether its behavior affects that artifact. Rebuild the affected suffix under new attempt paths, even if old loader hashes still pass. Save code/config provenance; do not silently mix implementations. |
| Inputs/results moved to another host/path | Validate the full chain there. Absolute upstream paths are embedded throughout, not only in the final result. Rematerializing one manifest cannot repair all broken nested links. Do not hand-edit scientific identities. |

For a demonstrated implementation defect, make a minimal patch, run/fix focused development regressions using synthetic/temporary data, then retry the real stage and continue automatically. If it fails again, diagnose the new evidence and repeat the recovery process. Record why the fix preserves the mainline and which artifacts need rebuilding. Preserve user edits. Passing a temporary test does not mark the real-data stage complete; it must pass its own validation. A new permanent `tests/` directory is not required for these checks.

## Run record and final handoff

Keep one concise `run-status.md` with a stage table (`pending`, `running`, `repairing`, `validated`, `failed`, `blocked`), attempts/paths/identities, metric summaries, logs, code/config changes, process details, and next action. Retain failed attempts in the history while recording the active diagnostic/repair/retry action; use `blocked` only when progress actually requires unavailable input, access, resources, or a user decision. Write `modes_ready` after strict final-mode validation. Only for a requested preview, write `preview_ready / viewer_not_started` after successful preview publication, with visualization_checked: false and include the exact `viewer --preview` launch command. Retain linked source/cache/prepared dependencies; no coordinate-backed result is required.

Do not claim that Viser was visually tested or that the scientific result was approved. If blocked, retain completed outputs and state the smallest missing input/decision plus a precise resume command. Merely writing a completion note or producing a synthetic scene never satisfies a real-data run request.
