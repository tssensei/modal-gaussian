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
and numerical settings. Current graphs remain frequency-specific; shared topology
with per-frequency weights is a next direction, not an implemented shortcut.

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
