# Modal Gaussians

Reconstruct frequency-specific 3D Gaussian motion from asynchronous fixed-view
videos. The current input pipeline is **SEA-RAFT reference-to-frame flow → shared
FFT cache → manual frequency selection → cached modal-image export**. Training
uses complex U/V modal images, soft graph weights, and a neural component field.

The accepted **Corn 0.225 Hz** and **Bush 0.744 Hz** results, exact source paths,
configuration and launch commands are recorded in [BASELINE.md](BASELINE.md).
Earlier preparation and research recipes are archived in
[docs/legacy/pipeline.md](docs/legacy/pipeline.md); they are not the default workflow.

## Environment

From the repository root:

```bat
conda env create --file environment.yml
conda activate modal-gaussian
```

The environment uses Python 3.11, PyTorch CUDA 12.8, gsplat 1.5.3 and Viser 1.1.0.
SEA-RAFT inference additionally uses the locally installed upstream repository
and downloaded Hugging Face weights. The existing locations are
`outputs/third_party/SEA-RAFT` and `outputs/models/sea-raft-M`; inference does not
download a model. The first gsplat render may compile its CUDA extension.
The standalone spectrum GUI does not load Gaussian weights or render with CUDA.

## 1. Reuse or compute SEA-RAFT flow

Reuse completed SEA-RAFT flow whenever the frames, reference and inference
settings are unchanged. Corn already has complete flows at
`outputs/corn1_sea_raft_0225_001` and `outputs/corn2_sea_raft_0225_001`.

For a new flow computation with an existing reference/timing artifact:

```bat
modal-gaussians flow compute --images data\prepared\images\corn1 --reuse-stabilization outputs\corn_local_001\flow\view1 --output outputs\corn1_sea_raft_flow_trial_001
```

`--reuse-stabilization` supplies frame order, sample rate, reference frame and
already stabilized images when present. It does not read the old Farneback flow
or spectrum. `--sea-raft-repo` and `--model-dir` override the local locations above.
The output is full-image reference-to-frame flow without fine-mask clipping,
smoothing, FFT or selected-frequency DFT.

This migration reuses existing reference metadata and prepared geometry. It is
not a new mask-free, from-scratch static reconstruction/bootstrap pipeline.
Historical artifact readers remain available for those dependencies.

## 2. Build one shared FFT cache

Use the same sample rate and `Nfft` for every view. `Nfft` must cover the longest
input sequence. Each sequence is mean-subtracted and multiplied by its own
symmetric Hann window before zero padding. The cache stores unnormalized
complex64 U/V for **all pixels**, including background.

Corn uses 20 fps and `Nfft=1600`: **801 bins over 0–10 Hz**, spaced by
**0.0125 Hz**; 0.225 Hz is bin 18.

```bat
modal-gaussians --log-file outputs\corn_spectrum_001.log spectrum build --view view1 outputs\corn1_sea_raft_0225_001 --view view2 outputs\corn2_sea_raft_0225_001 --scene outputs\corn_subject_selection_001\static_scene --nfft 1600 --output outputs\corn_spectrum_001
```

The completed Corn cache already exists at `outputs/corn_spectrum_001`. A matching
complete cache is reused. Zarr compression and tiled computation bound memory;
the two complex spectra are about 22 GiB before compression. The manual subject
box chooses the default statistics/display region; it does not crop cached data.

## 3. Inspect, select and export

```bat
modal-gaussians spectrum viewer --input outputs\corn_spectrum_001 --work-dir outputs\corn_spectrum_001\work --host 127.0.0.1 --port 8110
```

Open [http://127.0.0.1:8110/](http://127.0.0.1:8110/) after starting the server.
The left panel contains the spectrum and shared selection list; the right panel
shows reference RGB, joint amplitude and complex U/V previews. Switch views or
between the projected subject box and full frame, zoom the spectrum, step bins,
or inspect individual pixel values. U/V previews use a common display scale.

**No automatic peak search or snapping:** click an actual plotted sample or enter
an exact cached frequency. Off-grid input reports the legal interval and keeps
the current selection. DC can be inspected but cannot be exported for training.

Save writes a selection JSON. GUI Export writes a new selection and
`modal_images/bin_XXXX/view1` (and `view2`) directories containing
`modal_image.npy` and provenance. The CLI writes `bin_XXXX/view1` directly under
its `--output`; to use the same directory layout:

```bat
modal-gaussians spectrum export --input outputs\corn_spectrum_001 --selection PATH_TO_SELECTION_JSON --output outputs\corn_selected_modes_trial_001\modal_images
```

Export copies cached slices directly. It never reruns FFT/DFT, estimates a nearby
frequency, or silently falls back to the old pipeline.

## 4. Prepare and train selected modes

For each exported frequency, pass its per-view directories to
`motion prepare-selected-modal`. The following example assumes bin 18 was
exported with the command above:

```bat
modal-gaussians motion prepare-selected-modal --prepared outputs\corn_subject_selection_001\prepared --view view1 outputs\corn_selected_modes_trial_001\modal_images\bin_0018\view1 --view view2 outputs\corn_selected_modes_trial_001\modal_images\bin_0018\view2 --frequency-hz 0.225 --output outputs\corn_prepared_trial_001
```

Preparation reuses the parent's geometry and timing, recomputes cross-view
complex alignment and target normalization, and can add a frequency absent from
the parent. It does not inherit that old frequency's alignment or specialized
graph. For an applied manual subject selection, pass its scene with `--scene`;
visible selected-Gaussian contributions replace the old fine-mask supervision.

Build the matching soft graph from the **unfiltered** KNN candidate cache:

```bat
modal-gaussians graph build-modal-similarity --prepared outputs\corn_prepared_trial_001 --geometry-graph PATH_TO_UNFILTERED_GEOMETRY_CACHE --view view1 outputs\corn_selected_modes_trial_001\modal_images\bin_0018\view1 --view view2 outputs\corn_selected_modes_trial_001\modal_images\bin_0018\view2 --frequency 0.225 --soft-weights --minimum-edge-factor 0.05 --output outputs\corn_soft_graph_trial_001
modal-gaussians motion iterate-neural --prepared outputs\corn_prepared_trial_001 --config configs\neural_component_field.json --geometry-graph outputs\corn_soft_graph_trial_001 --frequency-hz 0.225 --output outputs\corn_modes_trial_001\experiment --stage modes
```

Resolve `PATH_TO_UNFILTERED_GEOMETRY_CACHE` from the matching prepared scene's
saved geometry; it is not a graph from another scene or frequency. The current
baseline keeps all K=16 candidates within radius 0.08, with motion-rejected edges
weighted at 0.05 of their original value. Control placement uses original
geometric distances; weak paths attenuate existing interpolation weights without
adding controls. Gaussian rigidity is 0.03 and control-rotation loss is 0.
Local rotations and Viewer ellipsoid rotation remain active. See
[BASELINE.md](BASELINE.md) for the remaining parameters and accepted graphs.

Use `--stage preview` only when a manual-oscillator preview is requested, then
launch it separately with `modal-gaussians viewer --preview ... --work-dir ...`.
Ordinary mode generation does not fit per-frame coefficients, train a video
reconstruction, start Viser or run experiment validation.

Shared topology with per-frequency weights is the recorded next direction;
complete multi-frequency graph reuse is not yet implemented. Current graph
artifacts and their modal inputs must still match the requested frequency.

## Compatibility and execution policy

Farneback plus per-view rFFT, greedy frequency selection and selected exact-DFT
execution are available only under `modal-gaussians legacy` for explicitly
requested historical work. Shared artifact readers preserve existing results,
identities, cached geometry and reference metadata. Do not delete historical
ancestors still referenced by accepted results.

New outputs use new directories; accepted results remain immutable. Follow the
user's **禁止 validate** policy: no real-data readback scans, network-replay
comparisons, extra renders or automatic evaluation. Basic input checks and
safe publication remain in place; focused synthetic development checks are
separate from experiment validation. A successful export is not a visual approval.

Add `--log-file PATH` before the subcommand to retain progress and errors. Do not
place the log inside a new artifact's output directory. For code organization,
see [motion/README.md](src/modal_gaussians/motion/README.md); pipeline operators
can use the [current command reference](skills/modal-gaussians-pipeline/references/commands.md).
