---
name: modal-gaussians-pipeline
description: Run, validate, debug, and resume this repository's standalone modal-Gaussian pipeline from prepared image/mask sequences to validated 3D modes at the requested frequencies, without fitting per-frame modal coordinates. Optional manual previews do not start Viser. Use for end-to-end pipeline execution or recovery, not generic 3DGS advice or pipeline execution when the user only requests a plan.
---

# Modal Gaussians Pipeline

Run the required real-data stages until the **final 3D modes at the user's requested frequencies** are saved and strictly validated. The default endpoint is `modes_ready`, including the accepted neural-field and small-fragment propagation recipe. Stop after this endpoint.

**Do not automatically fit modal coordinates to video optical flow.** Do not append `coordinates solve-direct`, `coordinates physics-fit`, `iterate-neural --stage full`, video replay, coordinate-dependent flow R², or a coordinate-backed result merely to call the pipeline complete. An old run-spec, a previous benchmark, or the phrase “run the entire pipeline” does not opt into these stages. A later explicit user request for coordinate/video reconstruction can change that scope.

The existing flow/DFT inputs, selected 2D modal images, fixed complex view alpha and modal-image supervision remain part of learning the 3D modes. This stopping rule does not remove that supervision or change the established frequency-selection algorithm.

When the user requests viewing the modes, append only `--stage preview`: rendered-design and an independent preview artifact. Hand off the launch command without initializing Viewer data or inspecting visualization results. Use manual oscillation, never invented or fitted video coordinates. Do not compute full spectra automatically. **Do not start Viser, open a browser, or configure a tunnel unless separately requested.**

## Execution preference: minimal extra work

The user evaluates scientific and visual quality. Run the specified workflow and stop when its requested outputs are produced.

- Rely on the pipeline's built-in validation. Do not add repeat strict loads, independent artifact replays, repeated hashes, synthetic training probes, or broad test suites to ordinary experiment runs. Recheck only after a relevant change, an actual error, or an explicit user request.
- Do not inspect images, render comparison PNGs, initialize `ModalViewerData`, cycle views/modes/phases, or perform headless visual checks on the user's behalf. Preview publication means data prepared, not a visually tested or user-approved result.
- Prefer existing CLI commands and reusable tools. Use inline commands for small diagnostics; do not create a new `tmp` launcher, verifier, reporter, or recovery script for each task. Create a temporary file only when a concrete operation requires one, and reuse it where possible. Normal pipeline outputs, atomic writes, caches and checkpoints remain necessary.
- Reuse saved configurations, manifests, logs and timings. Keep at most one concise additional run note when existing records are insufficient; do not routinely duplicate source trees, per-file hash inventories, git diffs, validation JSONs and comparison reports.
- On an actual failure, preserve its log/checkpoint, fix the cause, run only the smallest relevant check if needed, and resume. These preferences do not authorize bypassing a failed built-in integrity check or changing scientific settings.

This preference overrides the reference documents' diagnostic checklists for routine execution: they are troubleshooting aids, not extra stages to run every time.

## Start from the user's actual request

- For a request to write, explain, or plan the workflow, produce the requested guide; do not start training.
- For an authorized run-through, proceed across successful stages without asking for routine confirmation at each stage. Give concise progress updates. Follow a narrower user-selected stopping point when given.
- During an authorized run, diagnose failures, make necessary in-scope repairs, verify them, and continue automatically, including failures in preflight and validation. Give progress updates without treating each error as a new approval gate. Preserve the agreed scientific model, inputs, output ownership, and resource/permission boundaries.
- Work during the active task using the available process/session wait tools. This skill does not itself schedule future turns or guarantee unattended execution after the task ends. Only create a schedule if the user requests one.

## 1. Resolve the run contract

Locate the repository containing `src/modal_gaussians/cli.py`, read its applicable instructions, and inspect its dirty worktree without reverting user changes. Read [commands.md](references/commands.md) before assembling commands. When executing a run, also read [validation-recovery.md](references/validation-recovery.md) before the first expensive stage.

Use [run-spec.example.json](assets/run-spec.example.json) as a template for one run-owned `run-spec.json`. It is a planning record, not a new CLI config format. Fill in:

- Repository, exact Python interpreter, exact COLMAP executable, and absolute run-root paths on the execution host.
- Sweep PNG and mask directories; an explicit sweep sampling stride.
- An ordered list of fixed views: unique label, image/mask directories, FPS, reference **stem**, stabilization and smoothing choices. Derive each reference RGB/mask path from this same view, not another export.
- Frequency range, grid step, and requested greedy prefix length K. Never infer these from an old bush experiment.
- Explicit endpoint (`modes` by default; `preview` only when requested), `fit_modal_coordinates: false`, resolved training settings and optional viewer host/port; resource/time constraints and any scientific overrides the user has actually authorized.

Ask one consolidated question for missing information that cannot be discovered safely. Do not guess FPS, view synchronization, reference frames, masks, frequency settings, or cluster allocation/account. Defaults shown in the command reference are the current mainline, not evidence that they fit every dataset.

Keep run data under a dedicated absolute root, preferably outside the source tree or under its ignored `outputs/`. Do not create an artifact's output directory in advance. The pipeline rejects existing targets. The agent may create the run root and `logs/`.

## 2. Preflight before expensive work

Use the known CLI and environment; consult help or source when the invocation is uncertain or changed. Reuse prior successful environment/input checks for unchanged inputs. Run targeted input/CUDA checks only for a new environment, relevant change or actual failure. Check resource headroom when the requested workload changes substantially. Existing configuration and logs are sufficient provenance; do not export a dirty diff or per-file hashes by default.

Use the project's `modal-gaussian` conda environment with PyTorch CUDA **12.8**, gsplat **1.5.3**, and Viser **1.1.0**. COLMAP may remain in its separate `colmap` environment. The CLI's `--colmap-command` accepts a binary path/name, **not** `conda run ...` as a compound command.

On a cluster, use an authorized GPU allocation and its actual paths; do not train on a login node or invent scheduler parameters. Keep the environment consistent across compute stages. Installing dependencies or accessing remote machines still follows the current task's approval boundaries.

## 3. Execute the dependency chain

Use the numbered commands in the reference:

```text
per-view flow + diagnostic rFFT
    + sampled sweep / fixed-view references -> joint COLMAP
    -> static 3DGS / required static QA / accepted foreground partition
    -> topology -> requested frequency selection -> dense exact-DFT modes
    -> measurements -> observed graph -> fixed complex alpha alignment
    -> neural preparation (or reuse the immutable prepared snapshot)
    -> independent neural 3D modes for the requested frequencies
    -> accepted small-fragment propagation
    -> strict completed-modes validation; modes_ready; STOP

Only when a preview is requested:
    existing 3D modes -> rendered-design -> manual preview -> preview_ready; STOP
```

Use `motion prepare-neural` once and `motion iterate-neural --stage modes` for repeated experiments. Reuse identity-matched observation, fine-graph and control/interpolation caches. Recompute support roles for a changed geometry graph. The current alignment input comes from a rigid artifact, but the neural field consumes only its fixed alpha and identifiability data; it does not use rigid motion, rigid trust filtering, legacy motion fill or green refinement. Keep the accepted neural/fragment parameters unless the user changes them.


Run one stage at a time initially. Per-view flow and COLMAP are independent, but do not parallelize large jobs without checking resource headroom. For every stage:

1. Validate prerequisite artifacts and their ordered identities using the repository loaders; verify reuse matches the current input, settings, and output-affecting code.
2. Record the fully expanded argument vector, target paths, start time, and log location. Launch through a managed process and retain its session/PID/job ID.
3. Monitor actual progress without duplicating a live process. Capture exit code and failure output; long computation without stdout alone is not a failure.
4. Use the command's exit status, built-in validation and existing output manifest/status. Do not load the artifact again solely to repeat checks or inspect visual evidence.
5. Continue when the stage succeeds. Scientific and visual evaluation belongs to the user. A manifest saying `unapproved` is not itself a blocker; never approve it automatically.
6. On failure, follow the autonomous recovery rules below: preserve evidence, diagnose and fix the cause, verify the repair, and retry or resume. Continue later stages only after the failed prerequisite passes.

Reuse existing pipeline status/configuration/log files. Add one concise `run-status.md` only when needed to explain multiple stages or recovery; avoid a separate tracking framework or duplicate reports.

Enable the CLI's global `--log-file` before the stage subcommand, using one path under `logs/` per stage/attempt, never inside an artifact target. Share that path and the live-tail command from the command reference. Text progress includes completed counts, training metrics, elapsed time and estimated remaining time; counts are local to each named phase, not a whole-pipeline percentage. Continue capturing process stdout/stderr as well, since third-party output is not all routed through the progress logger.

## Scientific invariants

- Ready binary masks are inputs; do not add segmentation or masking services. RGB/mask/reference pixel geometry and FPS must remain consistent. Stabilization and smoothing are explicit choices.
- COLMAP sees sampled sweep RGB and one reference per fixed view, with full-image features. Semantic masks classify static FG/BG only. Keep raw and normalized camera conventions intact. Current grouping shares intrinsics within sweep and within references; flag incompatible input cameras instead of silently accepting them.
- Static training uses fixed cameras, separate FG/BG parameter domains, joint depth-ordered rasterization, direct RGB, RGB L1 + SSIM, and the accepted eroded semantic-mask loss. BG densification stops earlier than FG and obeys its configured hard count cap. Depth supervision is still disabled until the aligned-depth artifact is specified; do not invent depth inputs or restore scale/dynamic/track state or Shape-of-Motion dependencies.
- Freeze final foreground indexing. Preserve static-scene, foreground, reference-camera, flow, and downstream identities. Never repair a mismatch by rewriting hashes or weakening validation.
- Preserve view order and greedy mode slots everywhere. Frequency-sorted GUI labels must map back to immutable greedy slots, not reorder arrays.
- Preserve dense rFFT artifacts as upstream data; full-spectrum GUI statistics are optional cached work, not a mode-completion requirement. Selected dense modes are negative-exponent exact DFT with temporal-mean detrend, symmetric Hann, complex64, `(u,v)` order, and no mask/clamp/amplitude normalization.
- Preserve fixed complex alpha alignment, neural geometry/interpolation, structure losses, support roles and unresolved zeros. Apply the accepted fragment propagation without modifying host motion. Do not reintroduce rigid trust or legacy fill into the neural pipeline. When explicitly running a historical rigid pipeline, preserve that pipeline's own trust/fill contract.
- A rendered-design or manual preview does not require fitting time-dependent coordinates. Manual display uses `means + Re(sum(q*phi))` with user-controlled sinusoidal gain/phase. Do not report a fitted-video flow R² when no coordinates were fitted.
- Preserve existing complete results and their historical coordinates. They are optional compatibility inputs, not a reason to repeat coordinate fitting in future runs.

## Autonomous repair and recovery

**Experiment error -> preserve evidence -> diagnose -> repair -> verify -> resume.** An authorized pipeline run includes fixing implementation and execution problems needed to finish that run. Exceptions, assertions, API/type/shape/import errors, native crashes, and failures in preflight, QA, validation, or readiness checks do not require another user confirmation. Apply the same recovery loop to new failures after a retry; an initially unknown cause calls for investigation, not an automatic end to the turn.

1. Preserve the failed attempt's command, traceback, logs, checkpoints, outputs, and live-process details. Give a concise progress update while continuing work; do not end the turn merely to announce an error. Record the failed attempt and the stage's active `repairing` status in `run-status.md`, with a concrete diagnostic or repair next action.
2. Inspect the actual failure and make a small, evidence-backed code, invocation, or environment repair within the existing run contract and execution permissions. Preserve user edits, scientific settings, and accepted baselines. Use focused development checks or a small reproduction when they help verify the fix.
3. Revalidate affected prerequisites and retry the failed check or compute stage. Reuse compatible checkpoints and validated unaffected ancestors. If outputs or checkpoints are incompatible, use a fresh attempt path and rebuild the affected suffix. Continue downstream automatically once its prerequisites pass; never skip a failed check or weaken validation to claim success.
4. Record each repair, changed code/config, retry command, validation outcome, and artifact identity. Do not repeatedly rerun an unchanged deterministic failure; use new evidence to revise the diagnosis. For transient failures, use bounded retries, then investigate if they persist.

Use [validation-recovery.md](references/validation-recovery.md) for stage-specific resume support. Retain useful logs and QA; remove only agent-created disposable files after checking their exact paths. This workflow does not require a new permanent test suite.

Ask for user input only when progress depends on something that cannot be resolved with the available data, tools, resources, and authorization, such as an inaccessible required input, missing access, or a necessary change to the scientific objective. First complete independent work that can still proceed, then state the concrete blocker, preserved progress, attempted fixes, and smallest required decision. Ordinary program errors and uncertainty about their cause are not such blockers. Resource limits are not permission to silently reduce K, resolution, training, or view count. Run authorization does not itself authorize commit, push, upload, package downgrade, input deletion, or broad cache cleanup.

Changing output-affecting code invalidates the affected artifact and its descendants even if old hashes still validate. Keep validated unaffected ancestors. Use new attempt paths, or a fresh run root for a broad upstream change; update every downstream link. Never mix experiments under one apparent result.

## Finish with evidence

For the default run, completion means the mode-generation command succeeds with its built-in artifact checks and records `modes_ready`. Do not add an independent strict load/replay after successful publication. No rendered-design, modal coordinates, full spectrum, materialized video result or Viser initialization is needed to satisfy this endpoint.

Hand off the output path, requested frequencies and the launch command when applicable. Mention actual failures or limitations and a few useful metrics already in the logs; do not compute a new comparison/evaluation report unless requested.

For an explicitly requested preview, stop after preview publication and its built-in source checks. Record `preview_ready / viewer_not_started / visualization_checked: false`; give the exact `viewer --preview` launch command. Do not run headless/manual-deformation/image checks. The user will launch and evaluate the result.

Distinguish **execution verified**, **viewer visually tested**, and **scientific result approved**. Never silently replace the accepted baseline. Continue repairing in-scope failures until the chosen endpoint passes, rather than expanding the run to coordinate fitting or another endpoint.
