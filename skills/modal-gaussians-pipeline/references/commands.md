# Current pipeline commands

Use these commands for SEA-RAFT/shared-FFT frequency work with existing reference
metadata and prepared geometry. They are examples, not a script to execute in
full. Run only the requested stages and reuse completed matching artifacts.
[The skill's no-validation policy](../SKILL.md#禁止-validate--no-experiment-validation)
applies throughout. Historical bootstrap and research commands are archived in
[legacy/commands.md](legacy/commands.md); they do not define the current pipeline.

Examples below use Anaconda Prompt from the repository root in the
`modal-gaussian` environment. `modal-gaussians` is equivalent to
`python -m modal_gaussians.cli`. On another host, resolve input/output paths and
use its actual interpreter. Do not assume the old example dataset or frame rate.

## Flow only: compute once or reuse

Corn already has complete flow at `outputs/corn1_sea_raft_0225_001` and
`outputs/corn2_sea_raft_0225_001`. Reuse it. When inference is requested:

```bat
modal-gaussians flow compute --images data\prepared\images\corn1 --reuse-stabilization outputs\corn_local_001\flow\view1 --output outputs\corn1_sea_raft_flow_trial_001
```

The inherited artifact supplies frame order, FPS, reference and already
stabilized images when present; its Farneback arrays and spectra are not read.
The model repository and local weights default to `outputs/third_party/SEA-RAFT`
and `outputs/models/sea-raft-M`; override with `--sea-raft-repo` and `--model-dir`.
No model download, mask clipping, smoothing or temporal transform occurs here.
Use a new output path; preserve the accepted flow and its ancestors.

## Shared-grid spectrum and manual selection

All views must share FPS and Nfft, with Nfft covering every sequence length.
Original-length mean subtraction and symmetric Hann precede padding. This
example is Corn at 20 fps, Nfft 1600, bins 0–800 over 0–10 Hz:

```bat
modal-gaussians --log-file outputs\corn_spectrum_001.log spectrum build --view view1 outputs\corn1_sea_raft_0225_001 --view view2 outputs\corn2_sea_raft_0225_001 --scene outputs\corn_subject_selection_001\static_scene --nfft 1600 --output outputs\corn_spectrum_001
```

The completed matching cache is reusable. All image pixels are stored; the
manual subject box only chooses the default curve/display region. No Gaussian
weight loading, training, optical-flow recomputation or validation is added.

When the user requests a GUI, hand off its command without starting it:

```bat
modal-gaussians spectrum viewer --input outputs\corn_spectrum_001 --work-dir outputs\corn_spectrum_001\work --host 127.0.0.1 --port 8110
```

The local URL is `http://127.0.0.1:8110/` after launch. Select actual plotted bins,
step bins or enter exact grid frequencies. There is no automatic nearest-bin or
peak snapping; off-grid input leaves selection unchanged. DC is display-only.
Save produces a selection JSON. GUI Export also writes per-bin/view modal images
under its new session's `modal_images` directory. The equivalent CLI is:

```bat
modal-gaussians spectrum export --input outputs\corn_spectrum_001 --selection PATH_TO_SELECTION_JSON --output outputs\corn_selected_modes_trial_001\modal_images
```

CLI export writes `bin_XXXX/view1` and other views directly inside `--output`.
This copies cache slices; a missing or mismatched cache is an error, not permission
to fall back to DFT. Frequency 0.225 Hz is bin 18 in this example.

For requested uniform/greedy comparisons, use `spectrum select --input CACHE
--method uniform --count 60 --output NEW_SELECTION_DIR`, or `--method greedy
--topology EXISTING_TOPOLOGY`. Both produce `selection.json` for `spectrum export`.
Greedy additionally saves its ordered gains and sufficient statistics, fitting
SEA-RAFT targets on the existing topology samples without DFT. These fits are the
selection objective, not an added validation pass. Uniform bins span `(0, Nyquist]`;
resolve the actual frame rate from source metadata. Bush is 30 fps; Corn is 20 fps.

## Selected-modal preparation and matching graph

The exported fields can introduce a frequency absent from the parent snapshot.
Preparation preserves geometry/timing and recomputes complex view alignment and
normalization; it does not inherit another mode's alignment or specialized graph.

```bat
modal-gaussians motion prepare-selected-modal --prepared outputs\corn_subject_selection_001\prepared --view view1 outputs\corn_selected_modes_trial_001\modal_images\bin_0018\view1 --view view2 outputs\corn_selected_modes_trial_001\modal_images\bin_0018\view2 --frequency-hz 0.225 --output outputs\corn_prepared_trial_001
modal-gaussians graph build-modal-similarity --prepared outputs\corn_prepared_trial_001 --geometry-graph PATH_TO_UNFILTERED_GEOMETRY_CACHE --view view1 outputs\corn_selected_modes_trial_001\modal_images\bin_0018\view1 --view view2 outputs\corn_selected_modes_trial_001\modal_images\bin_0018\view2 --frequency 0.225 --soft-weights --minimum-edge-factor 0.05 --output outputs\corn_soft_graph_trial_001
```

Resolve `PATH_TO_UNFILTERED_GEOMETRY_CACHE` from the matching scene/prepared
metadata. Do not substitute a pruned graph or another scene's cache. Keep baseline
K=16/radius 0.08 and soft factors 1/0.05. Controls/support retain original
geometric distances; soft propagation attenuates interpolation without increasing
control count. [BASELINE.md](../../../BASELINE.md) records exact accepted paths
and numerical settings. Modal graph files remain frequency-specific, but prepared
component-field training shares geometry independently of weights and observations.

Before a batch, import compatible existing controls without sampling them again:

```bat
modal-gaussians motion prepare-shared-controls --prepared outputs\bush_neural_modal_similarity_0744_001\prepared --geometry-graph outputs\_cache\geometry\d8037d88fd6390382ad02ca899fb5f6ce7009e3a33814209164047b77be43126 --controls-from outputs\_cache\trained_modes\50a4e5e3a235247ed38405eeea55adbea7d811c0e6c5069db83cd556989fea32 --config configs\neural_component_field.json
```

This reads the existing KNN and v16 layout, fills material distances on saved
supports once, and publishes `control_geometry` in the prepared cache directory.
Alternatively, pass an existing compatible `control_geometry` directory through
`--controls-from` to import both layout and support distances without recomputing
them. This is useful after a code change; source cache identities remain intact.
No network replay, training or validation occurs. Later `iterate-neural` calls
reuse it automatically; only soft propagation/control weights are recomputed
under `control_weights`, with observation-dependent donor roles kept separate.
Without a compatible import/cache, layout construction runs once. Changing
topology, geometry, control radius/budget or learning-component gates creates a
different shared cache. Old completed artifacts and single-frequency inputs remain
readable. A stopped run created before this code change needs a new experiment
directory; preserve its checkpoint rather than rewriting its contract.

An applied manual subject scene may be passed to preparation using `--scene`.
It rebuilds observations using visible selected-Gaussian contributions, without
old fine-mask gating. A new scene without existing reference metadata/prepared
geometry needs a separately resolved bootstrap. Do not silently launch the
archived Farneback chain to manufacture these inputs.

## Train modes, optionally publish a manual preview

```bat
modal-gaussians motion iterate-neural --prepared outputs\corn_prepared_trial_001 --config configs\neural_component_field.json --geometry-graph outputs\corn_soft_graph_trial_001 --frequency-hz 0.225 --output outputs\corn_modes_trial_001\experiment --stage modes
```

The default endpoint is `modes_ready`, without validation or per-frame coordinates.
The preset uses rigidity 0.03 and control-rotation loss 0; local motion rotations
and Viewer ellipsoid rotation remain enabled. The accepted component-field donor
rules are unchanged. For exact historical reproduction, use its frozen config.

When a preview is requested, use the same settings and `--stage preview` instead.
After publication, provide this launch command without executing it:

```bat
modal-gaussians viewer --preview outputs\corn_modes_trial_001\experiment\preview --work-dir outputs\corn_modes_trial_001\work\viewer --host 127.0.0.1 --port 8108
```

A preview does not imply that Viser was started or the result was visually tested.
Do not append `--stage full`, coordinate fitting, PNG exports or quality checks.

## Logs, failures and compatibility

The current config sets `neural.max_iterations=5000`, counting all prior updates
when resuming. For an iteration-cap increase, stop dispatch and let active work
finish, then use a new batch output and `--continue-from OLD_BATCH` (not
`--resume-from`, which skips completed frequencies). The same option on
`motion iterate-neural` takes an old experiment directory. Only the cap may
increase; data/graph/loss/optimizer settings must match. Published prepared
inputs and graph caches are reused. Compatible checkpoints retain model,
optimizer, RNG, history and patience; already-converged checkpoints need no
additional updates. Missing/incompatible numerical-input checkpoints restart
with a logged reason. Old results and their identities remain unchanged.

For an explicitly authorized parallel batch from an already exported selection:

```bat
modal-gaussians --log-file outputs\bush_uniform60_modes_001\batch.log motion batch-neural --modal-images outputs\bush_uniform60_modes_001\modal_images --prepared outputs\bush_neural_modal_similarity_0744_001\prepared --geometry-graph outputs\_cache\geometry\d8037d88fd6390382ad02ca899fb5f6ce7009e3a33814209164047b77be43126 --config configs\neural_component_field.json --output outputs\bush_uniform60_modes_001 --cpu-workers 3 --gpu-workers 2 --threads-per-worker 2 --experiment-name experiment_shared_001
```

This separates CPU preparation from GPU training. CPU slots run selected-modal
preparation, graph weighting and `motion prepare-control-weights`; the latter
populates the same cache consumed by training, without loading a scene or GPU.
A small `control_weights_ready.json` receipt is published after cache completion.
GPU slots start only for ready frequencies while CPU slots prepare later ones.
Completed pre-split modes are skipped without recomputing CPU work. It creates
`batch_state.json`, `batch_workers.json` and five-second `gpu_usage.csv` samples.
The initial counts are used only when the control file does not yet exist;
change that file atomically to adjust the live limits, for example
`{"cpu_workers":3,"gpu_workers":2}`. Zero pauses new launches in that queue,
without interrupting active work. `batch_state.json` includes `active_cpu`,
`active_gpu` and `ready_for_gpu`. A subprocess failure prevents further launches;
existing children finish their current stages. The same command resumes matching
artifacts/checkpoints without validation; changed model code/config requires a
new batch contract/output. Use resource observations to choose concurrency, not
scientific changes to iteration limits, resolution or modes. Do not restart an
already active batch: its OS lock prevents duplicate scheduling.

Soft propagation uses a separate CPU process pool per frequency. The batch flag
`--propagation-workers 4` (default 4) parallelizes control-source searches while
preserving exact adaptive Dijkstra distances, original support indices and final
normalization. Each child receives the read-only graph once; no GPU computation
or extra dependency is introduced. Three CPU preparation slots therefore use up to
twelve propagation processes, separate from `--threads-per-worker` library
thread limits. For a standalone single-frequency command, set the environment
variable `MODAL_GAUSSIANS_PROPAGATION_WORKERS=4`; its default is serial.
To adjust a running batch without discarding active work, add or update the
positive integer `propagation_workers` in `batch_workers.json`, for example
`{"cpu_workers":4,"gpu_workers":2,"propagation_workers":6}`. Each new CPU
weight-preparation stage reads that file before creating its pool. Existing
pools keep their previous count. Schedulers started before live pool settings
were implemented still show their original pool default in `batch_state.json`;
the control file and each child's `soft propagation: N CPU workers` log record
the effective setting. No scientific cache identities change.

To migrate an old unsplit scheduler, first set its control file to `{"workers":0}`,
let its active frequencies finish, and stop that idle scheduler. Only then replace
the file with the two queue limits. Scheduling-only changes preserve scientific
identities and can resume the same batch directory.

To continue after a model/geometry code change, first let active frequencies finish and stop the
old scheduler. Use a new `--output` plus `--resume-from OLD_BATCH`. Matching
completed modes are inherited through their original `result_dir`; there is no
network replay or identity rewrite. Inputs, selected bins and scientific config
must match. Unfinished frequencies run under the new code; import the existing
shared geometry cache before launching if its code key changed. Subsequent
resumes use the same complete command, including `--resume-from`.

Place `--log-file PATH` before the subcommand, outside the new artifact directory.
Keep the failing process's output/checkpoint, repair an actual failure and resume
with unchanged scientific settings. Use exit status and existing publication
metadata to report completion; do not read back or hash-scan the output as a check.
Avoid duplicate training processes and broad environment changes.

Old flow/mode/result readers preserve artifact identities and ancestors. Explicit
historical computation uses `legacy flow analyze`, `legacy frequency select` and
`legacy frequency export-modes`; ordinary work must not execute them. Coordinate
and static/bootstrap compatibility commands remain available when specifically
needed. Do not delete historical dependencies referenced by accepted outputs.
