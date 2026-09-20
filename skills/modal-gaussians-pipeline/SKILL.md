---
name: modal-gaussians-pipeline
description: Run, debug, and resume this repository's SEA-RAFT/shared-FFT modal-Gaussian pipeline from existing reference metadata and prepared geometry to saved 3D modes at requested frequencies, without fitting per-frame coordinates. Optional manual previews do not start Viser. Use for pipeline execution or recovery, not generic 3DGS advice or plan-only requests.
---

# Modal Gaussians Pipeline

## Scene storage: look up reusable data first

Read `SCENE_STORAGE.md` and `scene_library/<scene>/catalog.json` before starting
new Bush/Corn work. Use `modal-gaussians storage list --scene bush|corn` to find
SEA-RAFT, shared FFT, prepared observations, KNN candidates, shared controls,
accepted results and saved checkpoints. `storage path --scene bush --asset
mode:0.25` locates an imported completed mode. Keep new experiments under
`scene_library/<scene>/experiments/<new_name>`; prepared training routes new
caches and model/checkpoint outputs to that scene automatically. Prefer the
existing exact cache contracts and explicitly matching results over rebuilding.
Do not edit old identity-bearing manifests or hard-linked immutable arrays.
Historical `outputs/...` strings are provenance aliases resolved by
`modal_gaussians.scene_store.resolve_path`; use it for direct file I/O from
saved metadata. Raw inputs under `data/` remain required. The old-output
classification is `scene_library/old_outputs_classification.csv`; do not delete
unreviewed files or resume the stopped Bush batch without a user request.

Run the required real-data stages until the **final 3D modes at the user's requested frequencies** are saved. Do not run validation stages for ordinary experiments. The default endpoint is `modes_ready`, including the accepted component field and its donor/follower propagation. Stop after this endpoint.

**Do not automatically fit modal coordinates to video optical flow.** Do not append `coordinates solve-direct`, `coordinates physics-fit`, `iterate-neural --stage full`, video replay, coordinate-dependent flow R², or a coordinate-backed result merely to call the pipeline complete. An old run-spec, a previous benchmark, or the phrase “run the entire pipeline” does not opt into these stages. A later explicit user request for coordinate/video reconstruction can change that scope.

Use SEA-RAFT flow, a shared-grid FFT cache, and direct cached-bin export for new frequency work. Selected complex U/V images, fixed complex view alpha and modal-image supervision remain part of learning the 3D modes. Existing selected-frequency inputs remain readable; do not recompute them just to migrate formats. Farneback and historical repeated-DFT selection require the explicit `legacy` CLI group and a historical-work request. When requested, `spectrum select --method uniform|greedy` saves frequencies from the shared cache. Cached greedy uses existing topology pixels and SEA-RAFT targets with the original equal-view grouped reconstruction objective; its required fitting statistics are selection work, not experiment validation. Do not substitute largest-peak selection or recompute DFT.

When the user requests viewing the modes, append only `--stage preview`: rendered-design and an independent preview artifact. Hand off the launch command without initializing Viewer data or inspecting visualization results. Use manual oscillation, never invented or fitted video coordinates. Do not compute full spectra automatically. **Do not start Viser, open a browser, or configure a tunnel unless separately requested.**

## 禁止 validate / No experiment validation

**禁止 validate。普通新实验、缓存复用、训练发布、preview 和 Viewer 加载均不得运行自动或额外 validation。** This is the user's explicit execution policy and overrides every validation/QA checklist in this skill's references.

- Use the default non-validating experiment/load paths. Do not enable `validate=True` or run validation commands unless the user explicitly requests validation.
- Do not rebuild KNN graphs, controls, interpolation or donor propagation for comparison; replay networks to compare saved phi; revisit upstream sources; rehash/read back outputs for verification; run packing-test renders, finite-array audit passes, independent probes or quality-gate stages.
- Compute required data once. A first control-graph build is necessary computation; replaying it as a check is forbidden. Recovering runtime ellipsoid rotation from saved weights is necessary computation, not permission to compare/revalidate the baked displacement.
- Keep safe deserialization, basic input format/shape/index/configuration guards, ordinary I/O errors, training-divergence errors, atomic writes, checkpoints, output ownership and no-overwrite protections. Cache keys and hashes written as provenance may remain; do not add a second pass to verify them.
- Do not retrain or launch real-scene checks to prove an optimization. A minimal development test for a requested code change is separate from experiment validation; do not attach it to ordinary runs.
- Report completion from command exit status and existing outputs/status. Do not describe an unchecked output as validated, visually tested or scientifically approved.

## Execution preference: minimal extra work

The user evaluates scientific and visual quality. Run the specified workflow and stop when its requested outputs are produced.

- Run no experiment validation, including formerly built-in strict checks. Use existing configuration, logs and outputs; only an explicit user request enables diagnostic validation.
- Do not inspect images, render comparison PNGs, initialize `ModalViewerData`, cycle views/modes/phases, or perform headless visual checks on the user's behalf. Preview publication means data prepared, not a visually tested or user-approved result.
- Prefer existing CLI commands and reusable tools. Use inline commands for small diagnostics; do not create a new `tmp` launcher, verifier, reporter, or recovery script for each task. Create a temporary file only when a concrete operation requires one, and reuse it where possible. Normal pipeline outputs, atomic writes, caches and checkpoints remain necessary.
- Reuse saved configurations, manifests, logs and timings. Keep at most one concise additional run note when existing records are insufficient; do not routinely duplicate source trees, per-file hash inventories, git diffs, validation JSONs and comparison reports.
- On an actual failure, preserve its log/checkpoint, fix the cause, run only the smallest relevant check if needed, and resume. Preserve basic input/I/O and no-overwrite errors; do not change scientific settings or introduce validation as a recovery stage.

This preference overrides the reference documents' diagnostic checklists for routine execution: they are troubleshooting aids, not extra stages to run every time.

## Start from the user's actual request

- For a request to write, explain, or plan the workflow, produce the requested guide; do not start training.
- For an authorized run-through, proceed across successful stages without asking for routine confirmation at each stage. Give concise progress updates. Follow a narrower user-selected stopping point when given.
- During an authorized run, diagnose failures, make necessary in-scope repairs and continue automatically, including failures in preflight and validation. Give progress updates without treating each error as a new approval gate. Preserve the agreed scientific model, inputs, output ownership, and resource/permission boundaries.
- Work during the active task using the available process/session wait tools. This skill does not itself schedule future turns or guarantee unattended execution after the task ends. Only create a schedule if the user requests one.

## Accepted baseline (2026-09-18)

Read the repository's `BASELINE.md` for accepted results and immutable paths before a new experiment. The current baseline is **Corn 0.225 Hz** and **Bush 0.744 Hz** with soft weights, fixed control placement, Gaussian rigidity **0.03** and control-rotation loss **0**. Future experiments start from this recipe unless the user specifies a comparison; acceptance does not imply evaluation on other scenes/frequencies.

- Use **SEA-RAFT** reference-to-frame flow and cached complex U/V images for both graph evidence and training targets. `flow compute` reuses existing timing/reference/stabilization metadata without reading Farneback arrays. `spectrum build` computes one shared grid across views; `spectrum export` copies chosen slices with no FFT/DFT fallback. Do not silently fall back to Farneback targets or snap an off-grid requested frequency.
- With an existing prepared geometry snapshot, use `motion prepare-selected-modal` to sample the new fields and recompute fixed complex view alpha and target normalization. Old flow metadata serves reference geometry/timing only. The accepted three-view snapshot is `outputs/bush_neural_modal_similarity_0744_001/prepared`.
- Start with unfiltered mutual-KNN candidates: **K=16**, maximum radius **0.08** in scene units. Use `graph build-modal-similarity --soft-weights`: similarity threshold **0.20**, conflict threshold **0.30**, amplitude floor fraction **0.02**, maximum projected endpoint distance **32 pixels**. Keep all candidate edges: reliable support without conflict retains original weight; all other edges get factor **0.05**. Do not precede this with depth-discontinuity pruning; depth/alpha still supplies visibility evidence.
- Pass the saved matching graph through `motion iterate-neural --geometry-graph` with neural `graph_edge_filter=none`; use exact paths in `BASELINE.md`. Controls, owners and support use original geometric distances. Soft propagation attenuates existing interpolation weights and must not add controls. Numeric K/radius defaults alone do not select the modal graph. Do not apply a saved frequency-specific graph to another scene/frequency.
- Preserve Corn's manual subject selection and visible-subject supervision instead of returning to old fine masks. The failed weak-edge-zero ablations are removed; do not restore them as part of pipeline migration.
- Keep width **256**, local features **32**, **3** message layers, control radius **0.015L**, maximum 32,768 controls, deformation/rotation weights **0.03/0**, and the existing component-field donor rules. Use `configs/neural_component_field.json` for new runs; exact historical reproduction uses that run's frozen config. Keep learning rate 0.001, maximum 5,000 total steps per frequency (updated by the user on 2026-09-19), patience 50, relative tolerance 1e-6, seed 1729. Use explicit `--continue-from` into a new experiment/batch to extend compatible checkpoints without resetting optimizer, RNG or patience. Already-converged modes remain early-stopped; missing/incompatible checkpoints restart with an explicit log. Local field rotations and Viewer ellipsoid rotations remain enabled. Donor propagation remains separate.
- Custom partial configs inherit omitted settings from the prepared snapshot; explicitly include the current loss weights when reusing an older preparation.
- Preserve the accepted outputs and ancestors. Approval is recorded in `BASELINE.md`, not by modifying hashed artifact manifests. Do not retrain to register a baseline. Continue the no-validation/no-coordinate-fitting policy and manual preview handoff above.

The current command reference covers this migration using existing reference metadata and prepared geometry. A fresh scene without those ancestors still needs a separately resolved bootstrap; do not silently execute the archived Farneback chain. Prepared component-field training now reuses `control_geometry` independently of modal weights/observations, and stores frequency-dependent attenuation in `control_weights`. Before a multi-frequency batch, use `motion prepare-shared-controls` to import compatible existing v16 controls and fill their missing geometric support distances once. Reuse the existing unfiltered KNN cache. Each frequency still needs its own matching soft weights and identifiable-view/donor decisions; do not share those merely because topology matches.

## 1. Resolve the run contract

Locate the repository containing `src/modal_gaussians/cli.py`, read its applicable instructions, and inspect its dirty worktree without reverting user changes. Read [commands.md](references/commands.md) before assembling commands. Read [validation-recovery.md](references/validation-recovery.md) only for a relevant failure; its validation checklists do not override the no-validation policy.

Use one concise run note only when existing configuration/logs do not capture the requested changes. An older run-spec is historical context, not a command to repeat its Farneback/greedy stages. Resolve:

- Repository, exact Python interpreter, exact COLMAP executable, and absolute run-root paths on the execution host.
- Existing static scene, manual subject selection when applicable, parent prepared snapshot and candidate geometry graph; sweep/mask inputs only if static bootstrap is explicitly part of the request.
- An ordered list of fixed views: unique label, image directory, reusable timing/reference/stabilization artifact, sample rate and existing SEA-RAFT source. Preserve their reference pixel geometry; no fine mask clips the flow or spectrum.
- Shared Nfft and sample rate, spectrum cache, selected bins and exact frequencies. The grid is `fps/Nfft`, with Nfft at least every sequence length; each view uses its original-length mean/Hann window before padding. Never infer these from another dataset. GUI selection is manual with no automatic snapping.
- Explicit endpoint (`modes` by default; `preview` only when requested), `fit_modal_coordinates: false`, resolved training settings and optional viewer host/port; resource/time constraints and any scientific overrides the user has actually authorized.

Ask one consolidated question for missing information that cannot be discovered safely. Do not guess FPS, view synchronization, reference frames, masks, frequency settings, or cluster allocation/account. Defaults shown in the command reference are the current mainline, not evidence that they fit every dataset.

Keep run data under a dedicated absolute root, preferably outside the source tree or under its ignored `outputs/`. Do not create an artifact's output directory in advance. The pipeline rejects existing targets. The agent may create the run root and `logs/`.

## 2. Preflight before expensive work

Use the known CLI and environment; consult help or source when the invocation is uncertain or changed. Reuse prior successful environment/input checks for unchanged inputs. Run targeted input/CUDA checks only for a new environment, relevant change or actual failure. Check resource headroom when the requested workload changes substantially. Existing configuration and logs are sufficient provenance; do not export a dirty diff or per-file hashes by default.

Use the project's `modal-gaussian` conda environment with PyTorch CUDA **12.8**, gsplat **1.5.3**, and Viser **1.1.0**. COLMAP may remain in its separate `colmap` environment. The CLI's `--colmap-command` accepts a binary path/name, **not** `conda run ...` as a compound command.

On a cluster, use an authorized GPU allocation and its actual paths; do not train on a login node or invent scheduler parameters. Keep the environment consistent across compute stages. Installing dependencies or accessing remote machines still follows the current task's approval boundaries.

## 3. Execute the dependency chain

Use the current commands in the reference. Reuse existing complete upstream data:

```text
existing timing/reference/stabilization metadata -> SEA-RAFT flow (or reuse it)
    -> one shared-grid FFT cache -> manual frequency selection
    -> cached-bin U/V export (no repeated DFT)
    + existing prepared geometry / accepted subject partition
    -> selected-modal preparation with fresh complex view alignment
    -> matching soft graph from unfiltered KNN candidates
    -> neural 3D modes with the existing donor/follower propagation
    -> save completed modes; modes_ready; STOP (no validation)

Only when a preview is requested:
    existing 3D modes -> rendered-design -> manual preview -> preview_ready; STOP
```

Use `motion prepare-selected-modal` with an existing prepared parent; new frequencies may be absent from the parent and receive fresh alignment/normalization. Reuse identity-matched geometry and use `motion iterate-neural --stage modes` for training. Recompute support roles for a changed geometry graph. Historical `motion prepare-neural` and rigid alignment artifacts remain compatibility paths, not reasons to repeat legacy upstream stages. The neural field consumes fixed alpha and identifiability data; it does not use rigid motion, rigid trust filtering, legacy motion fill or green refinement.


Run one stage at a time initially. Per-view flow and COLMAP are independent, but do not parallelize large jobs without checking resource headroom. For every stage:

1. Load required prerequisites through the default non-validating paths. Use existing cache contracts to select the current inputs/settings/code; do not rerun source-chain or content verification.
2. Record the fully expanded argument vector, target paths, start time, and log location. Launch through a managed process and retain its session/PID/job ID.
3. Monitor actual progress without duplicating a live process. Capture exit code and failure output; long computation without stdout alone is not a failure.
4. Use the command's exit status and existing output manifest/status. Do not load artifacts again to validate or inspect visual evidence.
5. Continue when the stage succeeds. Scientific and visual evaluation belongs to the user. A manifest saying `unapproved` is not itself a blocker; never approve it automatically.
6. On failure, follow the autonomous recovery rules below: preserve evidence, diagnose and fix the cause, and retry or resume. Continue later stages only after the failed prerequisite passes.

Reuse existing pipeline status/configuration/log files. Add one concise `run-status.md` only when needed to explain multiple stages or recovery; avoid a separate tracking framework or duplicate reports.

Enable the CLI's global `--log-file` before the stage subcommand, using one path under `logs/` per stage/attempt, never inside an artifact target. Share that path and the live-tail command from the command reference. Text progress includes completed counts, training metrics, elapsed time and estimated remaining time; counts are local to each named phase, not a whole-pipeline percentage. Continue capturing process stdout/stderr as well, since third-party output is not all routed through the progress logger.

## Scientific invariants

- Preserve frame/reference pixel geometry and FPS. SEA-RAFT flow and shared FFT are full-image and unsmoothed. Existing masks may remain dependencies of old static/prepared artifacts; manual-subject preparation uses visible rendered contributions instead. Do not add segmentation or masking services.
- COLMAP sees sampled sweep RGB and one reference per fixed view, with full-image features. Semantic masks classify static FG/BG only. Keep raw and normalized camera conventions intact. Current grouping shares intrinsics within sweep and within references; flag incompatible input cameras instead of silently accepting them.
- Static training uses fixed cameras, separate FG/BG parameter domains, joint depth-ordered rasterization, direct RGB, RGB L1 + SSIM, and the accepted eroded semantic-mask loss. BG densification stops earlier than FG and obeys its configured hard count cap. Depth supervision is still disabled until the aligned-depth artifact is specified; do not invent depth inputs or restore scale/dynamic/track state or Shape-of-Motion dependencies.
- Freeze final foreground indexing. Preserve static-scene, foreground, reference-camera, flow, and downstream identities. Never repair a mismatch by rewriting hashes or mixing incompatible inputs.
- Preserve view order and artifact mode slots everywhere, including historical greedy slots. Display sorting never reorders stored arrays. Cache exports bind exact bins/Hz and shared-grid identity; mismatches fail rather than recompute DFT.
- Preserve old dense rFFT ancestors and current shared caches. New spectra use original-length temporal-mean detrend and symmetric Hann before zero padding, negative-exponent FFT, unnormalized complex64 and `(u,v)` order, with no mask/clamp. Computing a spectrum is necessary only when requested or when the chosen new-frequency cache does not yet exist; reusing selected inputs does not require a full spectrum.
- Preserve fixed complex alpha alignment, neural geometry/interpolation, structure losses, support roles and unresolved zeros. Donor/follower propagation is part of the current neural field; do not append historical post-training fragment propagation, rigid trust or legacy fill. When explicitly running a historical rigid pipeline, preserve that pipeline's own trust/fill contract.
- A rendered-design or manual preview does not require fitting time-dependent coordinates. Manual display uses `means + Re(sum(q*phi))` with user-controlled sinusoidal gain/phase. Do not report a fitted-video flow R² when no coordinates were fitted.
- Preserve existing complete results and their historical coordinates. They are optional compatibility inputs, not a reason to repeat coordinate fitting in future runs.

## Autonomous repair and recovery

**Experiment error -> preserve evidence -> diagnose -> repair -> resume.** An authorized pipeline run includes fixing implementation and execution problems needed to finish that run. Exceptions, assertions, API/type/shape/import errors, native crashes, and failures in preflight, QA, validation, or readiness checks do not require another user confirmation. Apply the same recovery loop to new failures after a retry; an initially unknown cause calls for investigation, not an automatic end to the turn.

1. Preserve the failed attempt's command, traceback, logs, checkpoints, outputs, and live-process details. Give a concise progress update while continuing work; do not end the turn merely to announce an error. Record the failed attempt and the stage's active `repairing` status in `run-status.md`, with a concrete diagnostic or repair next action.
2. Inspect the actual failure and make a small, evidence-backed code, invocation, or environment repair within the existing run contract and execution permissions. Preserve user edits, scientific settings, and accepted baselines. Use focused development checks or a small reproduction when they help verify the fix.
3. Retry the failed compute stage using compatible checkpoints and unaffected ancestors. If outputs or checkpoints are incompatible, use a fresh attempt path and rebuild the affected suffix. Do not add prerequisite revalidation; preserve actual input/I/O and no-overwrite errors rather than reporting a failed computation as successful.
4. Record each repair, changed code/config, retry command, execution outcome, and artifact identity. Do not repeatedly rerun an unchanged deterministic failure; use new evidence to revise the diagnosis. For transient failures, use bounded retries, then investigate if they persist.

Use [validation-recovery.md](references/validation-recovery.md) for stage-specific resume support. Retain useful logs and QA; remove only agent-created disposable files after checking their exact paths. This workflow does not require a new permanent test suite.

Ask for user input only when progress depends on something that cannot be resolved with the available data, tools, resources, and authorization, such as an inaccessible required input, missing access, or a necessary change to the scientific objective. First complete independent work that can still proceed, then state the concrete blocker, preserved progress, attempted fixes, and smallest required decision. Ordinary program errors and uncertainty about their cause are not such blockers. Resource limits are not permission to silently reduce K, resolution, training, or view count. Run authorization does not itself authorize commit, push, upload, package downgrade, input deletion, or broad cache cleanup.

Changing output-affecting code invalidates the affected artifact and its descendants even if old hashes still validate. Keep unaffected ancestors. Use new attempt paths, or a fresh run root for a broad upstream change; update every downstream link. Never mix experiments under one apparent result.

## Finish with evidence

For the default run, completion means the mode-generation command succeeds and records `modes_ready`; validation is disabled. Do not add an independent strict load/replay after successful publication. No rendered-design, modal coordinates, full spectrum, materialized video result or Viser initialization is needed to satisfy this endpoint.

Hand off the output path, requested frequencies and the launch command when applicable. Mention actual failures or limitations and a few useful metrics already in the logs; do not compute a new comparison/evaluation report unless requested.

For an explicitly requested preview, stop after preview publication without validation. Record `preview_ready / viewer_not_started / visualization_checked: false`; give the exact `viewer --preview` launch command. Do not run headless/manual-deformation/image checks. The user will launch and evaluate the result.

Distinguish **execution completed**, **viewer visually tested**, and **scientific result approved**. Never silently replace the accepted baseline. Continue repairing in-scope failures until the chosen endpoint passes, rather than expanding the run to coordinate fitting or another endpoint.
