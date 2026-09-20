# Modal Gaussians

Reconstruct frequency-specific 3D Gaussian motion from asynchronous fixed-view
videos. The current input pipeline is **SEA-RAFT reference-to-frame flow → shared
FFT cache → manual frequency selection → cached modal-image export**. Training
uses complex U/V modal images, soft graph weights, and a neural component field.

The accepted **Corn 0.225 Hz** and **Bush 0.744 Hz** results, exact source paths,
configuration and launch commands are recorded in [BASELINE.md](BASELINE.md).
Earlier preparation and research recipes are archived in
[docs/legacy/pipeline.md](docs/legacy/pipeline.md); they are not the default workflow.

## Scene-owned storage

Current reusable Bush/Corn data now lives in `scene_library/`. Before a new
experiment, run `modal-gaussians storage list --scene bush` (or `corn`).
See [SCENE_STORAGE.md](SCENE_STORAGE.md) for the directory layout, cache reuse,
accepted-result lookup and classification of old outputs. Existing historical
paths below remain provenance aliases; new outputs belong under the scene's
`experiments/` directory.

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

For explicitly requested selection comparisons, `spectrum select` saves a list
without copying the full-resolution fields. Uniform selection covers `(0, Nyquist]`;
greedy selection uses existing topology pixels and the original equal-view flow
reconstruction R² objective, with paired real/imaginary cached modal columns.
It reads SEA-RAFT targets but never recomputes DFT. Greedy order and prefix gains
are retained in `selection.json` and `report.json`:

```bat
modal-gaussians spectrum select --input outputs\bush_spectrum_001 --method uniform --count 60 --output outputs\bush_spectrum_uniform60_001
modal-gaussians spectrum select --input outputs\bush_spectrum_001 --method greedy --count 60 --topology outputs\bush_neural_dense_controls_001\topology --output outputs\bush_spectrum_greedy60_001
```

For scenes without a manual 3D box, `spectrum build` accepts repeated
`--region LABEL BOOL_NPY` instead of `--scene`. These regions affect curves only;
the FFT cache still includes every pixel. The GUI remains manual, without snapping.

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

Prepared component-field training shares control layout, adjacency, owners and
material support distances across frequencies. Control edge weights and soft
interpolation attenuation have a separate cache; modal graph artifacts and inputs
must still match each frequency. To reuse controls from an existing compatible
v16 result, run `motion prepare-shared-controls --prepared ... --geometry-graph
PATH_TO_UNFILTERED_KNN_CACHE --controls-from COMPLETED_MODES --config
configs/neural_component_field.json` once. Subsequent training uses this cache
automatically. This imports the layout and fills missing support distances without
rebuilding KNN, resampling controls or replaying a network. See the
[command reference](skills/modal-gaussians-pipeline/references/commands.md).
`--controls-from` also accepts an existing compatible `control_geometry` cache;
its saved support distances are imported directly, without recomputation.

For multiple exported frequencies, `motion batch-neural --modal-images EXPORT
--prepared PARENT --geometry-graph KNN_CACHE --config CONFIG --output RUN_ROOT
--cpu-workers 3 --gpu-workers 2` maintains separate preparation and training queues.
Alpha synchronization defaults to CuPy (`--alpha-backend cupy`), including geometry
decomposition and the bounded TRF optimizer. One resident GPU worker prepares alpha
for all required frequencies while CPU workers build ready modal graphs. It then
releases alpha storage and performs CuPy float64 soft propagation, reusing its
topology and workspace. After all weights finish, that worker exits and the GNN
queue starts. Use `--stage weights` to stop after weights; the same output with
`--stage modes` then continues to training. Control placement and supports stay
unchanged. Install `pip install -e ".[gpu-propagation]"` for the GPU mainline.
The old CPU search is in `motion/legacy/neural/control_propagation.py`; explicit
`--propagation-backend cpu` is for historical runs/comparisons only, with no
automatic fallback. Its `--propagation-workers` setting has no effect on GPU.
`batch_state.json` records progress; `gpu_weights.log` and
`propagation_status.json` describe the resident worker. `gpu_usage.csv` samples
resources every five seconds. Edit `batch_workers.json` atomically to change
`cpu_workers` and `gpu_workers` independently (0–8 each). Any positive GPU count
enables only one alpha/propagation worker; later it controls GNN concurrency. Zero
pauses new launches while active work finishes. See
[GPU usage and measurements](docs/gpu-soft-propagation.md).
See [GPU alpha and geometry caching](docs/gpu-alpha.md) for the solver contract,
timings and explicit `--alpha-backend cpu` compatibility option. Neither GPU backend
silently falls back to CPU. Existing results retain their original alpha metadata.

Fixed-geometry modal training can optionally cache gsplat projection and sorted
tile intersections. In a **copy of your existing training config**, add
`"modal_projection_backend": "cached"` inside `neural`, keeping the other settings.
Pass that config to `motion iterate-neural` or `motion batch-neural` with `--config`.
Set the value to `"dynamic"` to restore the original rendering path; omission in
historical configs means dynamic. This option is recorded in the resolved training,
batch and checkpoint contracts. Switching backends requires a new experiment;
the existing continuation rule still allows only an increased iteration limit.

The cache is in GPU memory for each scene/camera projector and is shared across
its training steps/modes. It is rebuilt when a new training process starts, with
no disk cache or changes to immutable scene data. It caches projection geometry,
tile sorting and radial sampling coordinates; pixel compositing and feature
backward still use gsplat. Foreground/background occlusion and alpha normalization
are preserved. Geometry/camera updates, parameter replacement or enabled geometry
gradients fail explicitly rather than using stale projection data. Rebuild the
prepared observations/projector after geometry changes; use the dynamic renderer
for future geometry optimization. The current neural trainer still freezes geometry
with either setting. Do not mutate tensors through `.data`, which bypasses PyTorch
version tracking. Original RGB/static rendering paths are unchanged.

Synthetic CUDA checks compare forward values and feature gradients for pinhole,
radial and occluded-subject rendering, verify reuse, and reject stale geometry:
`python -m unittest discover -s tests -p test_modal_projection.py -v`.
An authorized Bush comparison measured the three-view projection/image-loss and
field-backward segment at 57.66 ms dynamic versus 12.02 ms cached (4.80x), with
0.099 s cache construction and 56.9 MiB additional live memory. This excludes GNN,
regularizers and optimizer work; full-training speedup remains unmeasured. See
[the benchmark method and results](docs/modal-projection-benchmark-20260920.md).

Failures stop further launches, preserving logs/checkpoints.
Re-running the same command resumes matching work and skips published modes.
New runs default to 5,000 total updates per frequency, retaining patience 50 and
relative tolerance `1e-6`. To raise a stopped batch's cap, use a fresh output with
`--continue-from OLD_BATCH` and the larger-cap config. This reuses published
prepared inputs/graphs and continues each compatible checkpoint with its optimizer,
RNG and early-stop state. Already-converged modes retain their learned result;
missing or incompatible checkpoints start from initialization with an explicit log.
Only the iteration cap may change. Parent artifacts are preserved. The same
`--continue-from OLD_EXPERIMENT` option is available on `motion iterate-neural`.
The initial batch needs a fresh experiment directory name (default
`experiment_shared_001`); use `--experiment-name` to preserve older attempts.
After a code change, use a new output directory and `--resume-from OLD_BATCH` to
carry completed frequencies with matching inputs/configuration. The source batch
must be stopped. Old mode identities/files remain unchanged; each state's
`result_dir` points to its actual experiment. Unfinished frequencies start under
the new code, while repeated launches of that new batch resume its checkpoints.

## 5. Experimental fixed-mode RGB coefficients

`coordinates fit-rgb` fits independent complex coefficients for every frame of
each recording. It freezes the static scene, appearance, cameras, displacement
modes and rotation modes. Initialization uses the existing direct-coordinate
artifact: subtract its reference coefficient, fit a common RGB pose offset,
then optimize all modes jointly per frame. The objective is full-image
`0.8 L1 + 0.2 (1 - SSIM)` plus a weak, decaying anchor in the direct solver's
pair-normalized units. There is no zero-mean or temporal/oscillator constraint.

```bat
modal-gaussians storage run --scene bush -- coordinates fit-rgb --scene @static --modes @experiments/coefficient40_rgb_001/mode_bank --input @experiments/coefficient40_rgb_001/direct_coordinates --config configs/rgb_coordinates.json --output @experiments/coefficient40_rgb_001/rgb_coordinates
modal-gaussians storage run --scene bush -- result materialize --scene @static --modes @experiments/coefficient40_rgb_001/mode_bank --coordinates @experiments/coefficient40_rgb_001/rgb_coordinates --output @experiments/coefficient40_rgb_001/result
```

RGBs are discovered from the direct artifact's recorded image directory, using
the stabilized images when applicable. `--images LABEL DIRECTORY` overrides
one view explicitly; frames must retain the recorded PNG names, dimensions and
camera geometry. Gaussian centers and orientations both follow the fixed
complex modes. Isotropic image pyramids preserve the camera distortion model.
The JSON config controls scales, epochs, learning rates, anchor weights and
batch size. Defaults are an initial experiment budget, not a quality guarantee.

Add `--view view2` to `coordinates fit-rgb` to fit only that recording using the
existing shared inputs. The output contains only its coefficients and can be
materialized and exported normally. Omit `--view` to fit all recordings.

Outputs contain only coefficients, source identities, image checksums and
training-loss history; fitting does not launch a Viewer or compute evaluation
metrics. Use a new output directory. For separate single-frequency results,
`coordinates prepare --scene bush --expected-modes 40 --output ...` collects
indexed fixed modes, creates a rendered design, and initializes coefficients
from saved SEA-RAFT flow. It never starts RGB fitting or resumes mode training.
See [COEFFICIENT_FITTING.md](COEFFICIENT_FITTING.md) for the exact scene-library
storage layout, stage reuse, commands and supported Viewer features.

`result export-video --result ... --view view1 --output ...` explicitly exports
one RGB-fitted recording as `comparison.mp4`: original fitting frames above,
reconstruction below, at recorded resolution and FPS. It uses a new directory
and never runs automatically after fitting. See the offline export command in
[COEFFICIENT_FITTING.md](COEFFICIENT_FITTING.md).

Synthetic CPU checks (no scene data or GPU needed):

```bat
python -m unittest discover -s tests -p "test_coefficient_rgb_*.py"
```

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
