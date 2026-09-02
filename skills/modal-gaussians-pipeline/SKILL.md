---
name: modal-gaussians-pipeline
description: Run, validate, debug, and resume this repository's standalone modal-Gaussian pipeline from prepared image/mask sequences to a complete Viser-ready result, without starting Viser. Use for end-to-end pipeline execution or recovery, not generic 3DGS advice or pipeline execution when the user only requests a plan.
---

# Modal Gaussians Pipeline

Execute the real dataset stage by stage until the materialized modal result and every dependency needed by Viser are available and validated. **Stop there: do not start Viser, open a browser, or configure a tunnel.** Provide the user with the exact launch command. Treat this codebase as unverified on real data; successful imports and synthetic tests are not substitutes for completing the real pipeline.

Keep the implementation simple and preserve the accepted scientific mainline. This is an agent-operated workflow, not a batch script that blindly launches all stages.

## Start from the user's actual request

- For a request to write, explain, or plan the workflow, produce the requested guide; do not start training.
- For an authorized run-through, proceed across successful stages without asking for routine confirmation at each stage. Give concise progress updates and explain any repair. Follow a narrower user-selected stopping point when given.
- Repair implementation defects within the requested repository when the user authorizes running and fixing the pipeline. Do not interpret persistence as permission to change the scientific model, spend money, overwrite data, or publish code.
- Work during the active task using the available process/session wait tools. This skill does not itself schedule future turns or guarantee unattended execution after the task ends. Only create a schedule if the user requests one.

## 1. Resolve the run contract

Locate the repository containing `src/modal_gaussians/cli.py`, read its applicable instructions, and inspect its dirty worktree without reverting user changes. Read [commands.md](references/commands.md) before assembling commands. When executing a run, also read [validation-recovery.md](references/validation-recovery.md) before the first expensive stage.

Use [run-spec.example.json](assets/run-spec.example.json) as a template for one run-owned `run-spec.json`. It is a planning record, not a new CLI config format. Fill in:

- Repository, exact Python interpreter, exact COLMAP executable, and absolute run-root paths on the execution host.
- Sweep PNG and mask directories; an explicit sweep sampling stride.
- An ordered list of fixed views: unique label, image/mask directories, FPS, reference **stem**, stabilization and smoothing choices. Derive each reference RGB/mask path from this same view, not another export.
- Frequency range, grid step, and requested greedy prefix length K. Never infer these from an old bush experiment.
- Resolved training settings and viewer host/port; resource/time constraints and any scientific overrides the user has actually authorized.

Ask one consolidated question for missing information that cannot be discovered safely. Do not guess FPS, view synchronization, reference frames, masks, frequency settings, or cluster allocation/account. Defaults shown in the command reference are the current mainline, not evidence that they fit every dataset.

Keep run data under a dedicated absolute root, preferably outside the source tree or under its ignored `outputs/`. Do not create an artifact's output directory in advance. The pipeline rejects existing targets. The agent may create the run root and `logs/`.

## 2. Preflight before expensive work

Verify current CLI help against the command reference; source is authoritative if flags have changed. Record the code revision and current dirty diff, resolved settings, and environment. Run input and CUDA checks in the validation reference. Estimate RAM, VRAM, disk, and runtime before loading dense flows.

Use the project's `modal-gaussian` conda environment with PyTorch CUDA **12.8**, gsplat **1.5.3**, and Viser **1.0.30**. COLMAP may remain in its separate `colmap` environment. The CLI's `--colmap-command` accepts a binary path/name, **not** `conda run ...` as a compound command.

On a cluster, use an authorized GPU allocation and its actual paths; do not train on a login node or invent scheduler parameters. Keep the environment consistent across compute stages. Installing dependencies or accessing remote machines still follows the current task's approval boundaries.

## 3. Execute the dependency chain

Use the numbered commands in the reference:

```text
per-view flow + diagnostic rFFT
    + sampled sweep / one reference per fixed view -> joint COLMAP
    -> static 3DGS + static QA
    -> observation topology
    -> greedy K-frequency selection
    -> dense complex exact-DFT modes
    -> measurement bank
    -> observed structure graph
    -> complex alpha synchronization + rigid solve
    -> motion fill
    -> rendered modal design
    -> direct coordinates
    -> physics coordinate post-fit
    -> result materialization
    -> headless Viser-data readiness check (no server)
    -> hand off the Viser launch command; stop
```

Run one stage at a time initially. Per-view flow and COLMAP are independent, but do not parallelize large jobs without checking resource headroom. For every stage:

1. Validate prerequisite artifacts and their ordered identities using the repository loaders; verify reuse matches the current input, settings, and output-affecting code.
2. Record the fully expanded argument vector, target paths, start time, and log location. Launch through a managed process and retain its session/PID/job ID.
3. Monitor actual progress without duplicating a live process. Capture exit code and failure output; long computation without stdout alone is not a failure.
4. Load the result strictly, inspect the stage-specific metrics and representative visual evidence, then record its identity and status.
5. Continue if structurally valid and scientifically usable as a provisional candidate. A manifest saying `unapproved` is not itself a blocker. Never convert that status to approved automatically.
6. On failure, apply the recovery rules below, fix the earliest demonstrated cause, and rerun only the affected dependency suffix.

Maintain one concise `run-status.md` under the run root, updated after each stage and before ending a turn. Include stage/attempt, command and log, artifact path/identity, code/config used, checks and metrics, pending warnings, live process details, and the exact next action. Reuse this record on continuation; independently verify it against files and live processes. Do not build a separate tracking framework.

## Scientific invariants

- Ready binary masks are inputs; do not add segmentation or masking services. RGB/mask/reference pixel geometry and FPS must remain consistent. Stabilization and smoothing are explicit choices.
- COLMAP sees sampled sweep RGB and one reference per fixed view, with full-image features. Semantic masks classify static FG/BG only. Keep raw and normalized camera conventions intact. Current grouping shares intrinsics within sweep and within references; flag incompatible input cameras instead of silently accepting them.
- Static training uses fixed cameras, separate FG/BG parameter domains, joint depth-ordered rasterization, direct RGB, and **RGB L1 + SSIM only**. Do not restore scale regularization, dynamic state, mask/depth/track loss, or Shape-of-Motion dependencies.
- Freeze final foreground indexing. Preserve static-scene, foreground, reference-camera, flow, and downstream identities. Never repair a mismatch by rewriting hashes or weakening validation.
- Preserve view order and greedy mode slots everywhere. Frequency-sorted GUI labels must map back to immutable greedy slots, not reorder arrays.
- Retain dense rFFT for the spectrum GUI. Selected dense modes are negative-exponent exact DFT with temporal-mean detrend, symmetric Hann, complex64, `(u,v)` order, and no mask/clamp/amplitude normalization.
- Preserve bounded complex alpha synchronization, rigid trust gates, fill support classes, and unresolved zeros. Do not invent observations, seeds, motion, or looser thresholds to make a stage pass.
- Rendered design, direct coordinates, and physics post-fit are separate stages. Use reference-relative `q(t)-q(ref)` for flow reconstruction; final deformation uses the result contract `means + Re(sum(q*phi))`. Do not conflate those conventions.
- Physics post-fit changes coordinates, not frequencies or modes. Keep both direct and post-fit results and report the tradeoff. Do not silently substitute one for the other.

## Repair and stopping rules

Use [validation-recovery.md](references/validation-recovery.md) for exact resume support and diagnosis. Prefer small, evidence-backed fixes with inline or temporary regression checks. Remove only agent-created disposable test files after checking their exact paths; keep useful run logs and QA. Do not add a permanent `tests/` directory for this workflow.

Do not repeatedly rerun an unchanged deterministic failure. Exhaust safe, materially different checks within scope. Pause with the failure evidence, preserved progress, and smallest required decision when a fix requires missing data, access, a different scientific choice, an unavailable resource, or expanded authorization. Resource limits are not permission to silently reduce K, resolution, training, or view count. No automatic commit, push, upload, package downgrade, input deletion, or broad cache cleanup.

Changing output-affecting code invalidates the affected artifact and its descendants even if old hashes still validate. Keep validated unaffected ancestors. Use new attempt paths, or a fresh run root for a broad upstream change; update every downstream link. Never mix experiments under one apparent result.

## Finish with evidence

Completion means every real-data compute stage is validated, the materialized result strictly loads, and the headless readiness check in the reference can load the scene, modes, coordinates, cameras, spectra, measurements, topology, and original reference RGBs. The CUDA preflight and static QA must also have passed. No Viser server or browser test is required or authorized by this workflow.

Report `viser_ready / viewer_not_started` when these gates pass. If the computation is complete but required viewer data or its execution environment is unavailable, name that limitation rather than claiming readiness. Launching Viser later is a separate user action/request.

Hand off the result path/identity, exact Viser launch command and expected local URL (explicitly not running), QA paths, stage metrics and limitations, any code fixes, and resume instructions. Distinguish **pipeline execution verified** from **viewer visually tested** and **scientific result approved**; do not claim either of the latter.
