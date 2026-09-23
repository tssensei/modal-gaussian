# Modal Gaussians

Recover frequency-specific complex 3D vibration modes from **asynchronous
fixed-view videos**, then use modal coefficients to reconstruct recorded motion
or synthesize user-controlled motion. A sweep supplies static reconstruction
inputs; separately recorded fixed views observe a shared spatial mode basis.
They are **not synchronized multi-view frames**: each recording has its own
coefficients.

```text
Sweep + fixed-view camera/reference preparation
  → static 3D Gaussians and reusable geometry
Fixed-view videos + render-matched motion references
  → SEA-RAFT reference-to-frame flow → shared-grid FFT → selected U/V modal images
  → per-frequency view alignment and soft graph weights → GNN/control fields
  → saved complex 3D displacement and rotation modes           [default endpoint]
      ↳ optional manual-mode viewing / synthesis
  → fixed-mode flow initialization → RGB coefficient fitting  [when requested]
  → result binding → offline video export / interactive playback
```

## Agent starting point

1. Read [SCENE_STORAGE.md](SCENE_STORAGE.md), then the scene's
   `scene_library/<scene>/catalog.json` and
   `scene_library/<scene>/results/index.json` to locate inputs.
   Reuse artifacts whose scene, reference, frequency and configuration contracts match.
2. Use the current policy at the top of [BASELINE.md](BASELINE.md) and
   [configs/neural_component_field.json](configs/neural_component_field.json).
   Later historical sections describe earlier experiments, not current defaults.
3. Identify the requested stage and stop there. **Do not automatically start
   training, coefficient fitting, Viser, export or real-data validation.**
   Mode-generation requests normally stop at `--stage modes`.
4. Put new work under `scene_library/<scene>/experiments/<new_name>/`.
   Preserve raw `data/`, immutable arrays/models, manifests and cache identities.
   Resolve historical paths with `modal_gaussians.scene_store.resolve_path()`;
   never rewrite provenance or bypass a contract mismatch.
5. Report completion from exit status, logs and publication metadata. Do not add
   real-data readback scans, network-replay comparisons, extra renders or metric
   passes as completion checks. Required input checks and atomic publication stay
   enabled; focused synthetic development checks are separate.

Read-only discovery, from the repository root:

```bat
modal-gaussians storage list --scene bush
modal-gaussians storage path --scene bush --asset spectrum
modal-gaussians storage path --scene bush --asset candidate_graph
```

`storage run --scene NAME -- COMMAND` expands `@asset` aliases for that scene.
Aliases are interpreted by this wrapper, not by the shell or standalone commands.
Use `modal-gaussians COMMAND --help` for flags before constructing a run.

## Environment

Reuse the existing `modal-gaussian` Conda environment. For a new installation:

```bat
conda env create --file environment.yml
conda activate modal-gaussian
pip install -e ".[gpu-propagation]"
```

[environment.yml](environment.yml) and [pyproject.toml](pyproject.toml) define the
runtime: Python 3.11, PyTorch 2.7.1/CUDA 12.8, gsplat 1.5.3, Viser 1.1.0 and CuPy for
GPU alpha/soft propagation. These GPU backends fail explicitly; there is no
silent CPU fallback. The first gsplat render may compile its CUDA extension.
SEA-RAFT uses local code at `scene_library/_shared/tools/third_party/SEA-RAFT`
and weights at `scene_library/_shared/tools/models/sea-raft-M`; inference does
not download them. Offline video export requires FFmpeg.

## Pipeline stages and contracts

All commands below are under `modal-gaussians`. Existing scenes enter at the
first missing stage; do not rerun the whole chain.

| Stage | Input → output | Entry point / boundary |
| --- | --- | --- |
| Static preparation | Sweep, camera calibration, subject selection and video preparation → static Gaussians, cameras, reference/timing metadata and prepared geometry | Existing Bush/Corn assets are registered. New-scene bootstrap must be resolved separately; see [historical preparation](docs/legacy/pipeline.md). |
| Motion reference | Static camera render + video masks → selected frame and overlay review images | `flow select-reference`; review the candidate before computing new flow. |
| Optical flow | Selected motion reference + recorded/stabilized frames → full-image reference-to-frame flow | `flow compute --reference-selection ... --reuse-stabilization ...`; no FFT or mask clipping. |
| Shared spectrum | Per-view flow sequences → cached complex U/V for every pixel and frequency bin | `spectrum build`; all views share FPS and Nfft, with Nfft covering the longest sequence. |
| Frequency selection | Cached spectrum → selection JSON → per-bin/per-view modal images | `spectrum viewer` when requested, or `spectrum select` for requested uniform/greedy selection; `spectrum export` copies cached slices. |
| Mode preparation | Selected modal images + frozen geometry → aligned supervision and matching soft graph | `motion prepare-selected-modal`, then `graph build-modal-similarity --soft-weights`; alignment and weights are frequency-specific. |
| Spatial mode training | Prepared supervision + soft graph → complex displacement and rotation fields | `motion iterate-neural --stage modes`; `motion batch-neural --stage modes` orchestrates preparation, weights and training for multiple frequencies. |
| Coefficient preparation | Indexed saved modes + matching SEA-RAFT flow → mode bank, rendered design and direct coefficients | `coordinates prepare`; does not train modes or start RGB fitting. |
| RGB fitting | Fixed mode bank + direct initialization + RGB frames → per-recording/per-frame coefficients | `coordinates fit-rgb`; `--view` selects one recording, omission fits all. |
| Delivery | Static scene + modes + coefficients → bound result and optional comparison video | `result materialize`, then explicit `result export-video`; `viewer --input` also accepts models or batch indexes directly. |

**Reference and FFT rules.** Geometry reference and motion reference are distinct.
Keep the geometry reference, stabilized image grid and original sequence time
origin unchanged. New motion references invalidate flow, FFT, modal exports,
alignment and frequency weights; compatible KNN/control geometry remains reusable.
Reference-selection masks rank candidates only; they do not clip supervision.
Each sequence is mean-subtracted and Hann-windowed before zero padding. Shared
FFT grids do not synchronize recordings. Export exact cached bins: no snapping,
new DFT or DC training mode. Display/statistics regions do not crop the FFT cache.

**Spatial mode rules.** Frozen geometry is shared across frequencies: KNN
candidates, control layout, owners and geometric supports. Complex view alignment
(alpha), soft edge weights and interpolation attenuation belong to each frequency;
a single mode's graph is not a general-purpose graph. The GNN learns control
fields, with fixed interpolation and donor transfer composing the Gaussian field
before modal-image supervision. See [component fields](docs/component-field.md).

The current recipe is **new motion references + per-view RMS normalization +
deformation weight 0.03 + control-rotation weight 0**, with up to **5,000 updates
per frequency**. Local rotations remain active. Use the full current preset when
reusing older preparations; partial configs inherit omitted prepared settings.
CuPy alpha and soft propagation finish before batch GNN training starts.

**Coefficient rules.** Static geometry, appearance, cameras, displacement modes
and rotation modes remain fixed. Flow ridge initialization is followed by
reference-coefficient subtraction, a shared pose offset per recording, then
joint per-frame coefficient optimization against RGB:

```text
x(t) = x_static + Re(sum_k(q_k(t) * phi_k))
coordinates.npy: complex64 [sum of selected recording frame counts, mode count]
RGB loss: 0.8 L1 + 0.2 (1 - SSIM) + weak decaying initialization anchor
```

Rotation uses the saved angular fields through an exponential map. RGB fitting
uses image pyramids and recorded PNGs (normally stabilized), with no temporal
smoothing, oscillator constraint or frequency locking. A mode's spatial frequency
label does not restrict its coefficient's temporal spectrum. Training losses are
not formal PSNR/SSIM/LPIPS evaluation. See [COEFFICIENT_FITTING.md](COEFFICIENT_FITTING.md).

## Registered data snapshot (2026-09-23)

The catalogs/indexes are the source for current paths and execution status.
This snapshot records completed work, not visual acceptance or quality metrics.

| Scene | Current FFT grid | Completed mode batch | Exact result-index status |
| --- | --- | --- | --- |
| Bush, 3 views | 30 FPS, Nfft 1250, 0.024 Hz spacing | 20 modes, 0.240–4.992 Hz, including 0.744 Hz; `@uniform20_batch` | `completed_uniform20_newref_0to5` |
| Corn, 2 views | 20 FPS, Nfft 1600, 0.0125 Hz spacing | 10 modes, 0.25–2.5 Hz; `@uniform10_batch` | `completed_uniform10_newref_0to2p5` |

Both scenes also retain new-reference comparison runs. Bush registers
`@coefficient20_mode_bank` and `@coefficient20_view1`; their presence is not a
fresh integrity check or reconstruction-quality evaluation. Old-reference
baselines, the old Bush 40-mode batch and its coefficient experiment were deleted.
Do not use their historical paths or user evaluations for current results.

## Common task examples

These are **independent examples, not a script to execute in full**. Run only
requested stages, reuse matching completed inputs, and choose fresh output names.
Commands use Anaconda Prompt syntax from the repository root. Some linked
reference documents still contain historical paths/counts; resolve current assets
and use the explicit statuses above.

### Train an explicitly selected set of frequencies

Given a selection JSON for the current Bush spectrum, replace `SELECTION_JSON`
and `NEW_RUN`. Skip export if those exact modal slices already exist.

```bat
modal-gaussians storage run --scene bush -- spectrum export --input @spectrum --selection SELECTION_JSON --output @experiments/NEW_RUN/modal_images
modal-gaussians storage run --scene bush -- motion batch-neural --modal-images @experiments/NEW_RUN/modal_images --prepared @prepared --geometry-graph @candidate_graph --config configs/neural_component_field.json --output @experiments/NEW_RUN/batch --cpu-workers 3 --gpu-workers 2 --stage modes
```

The batch takes the **unfiltered candidate graph**, then builds each frequency's
soft weights. Single-frequency `iterate-neural` instead takes its prepared,
frequency-matched soft graph. Shared control caches are reused when compatible.

### Fit and export one recording with fixed modes

For a new Bush 20-mode experiment, replace `NEW_FIT`. Always pass `--status`:
the CLI's historical default is `completed_uniform60`. If a compatible mode bank
and direct initialization already exist, reuse them and start at `fit-rgb` with
a new RGB output directory.

```bat
modal-gaussians storage run --scene bush -- coordinates prepare --scene bush --status completed_uniform20_newref_0to5 --expected-modes 20 --output @experiments/NEW_FIT
modal-gaussians storage run --scene bush -- coordinates fit-rgb --scene @static --modes @experiments/NEW_FIT/mode_bank --input @experiments/NEW_FIT/direct_coordinates --view view1 --config configs/rgb_coordinates.json --output @experiments/NEW_FIT/rgb_coordinates_view1
modal-gaussians storage run --scene bush -- result materialize --scene @static --modes @experiments/NEW_FIT/mode_bank --coordinates @experiments/NEW_FIT/rgb_coordinates_view1 --output @experiments/NEW_FIT/result_view1
modal-gaussians storage run --scene bush -- result export-video --result @experiments/NEW_FIT/result_view1 --view view1 --output @experiments/NEW_FIT/exports/view1_001
```

Export preserves input FPS and panel resolution: fitting input above,
reconstruction below. The H.264 comparison is for viewing, not metric evaluation.

### View saved modes or coefficients

When a Viewer launch is requested, replace `MODEL_OR_RESULTS_INDEX` and
`VIEWER_DIR` with resolved paths:

```bat
modal-gaussians viewer --input MODEL_OR_RESULTS_INDEX --work-dir VIEWER_DIR
```

No combined bank is needed for viewing a batch. Add `--coordinates COORDINATES_DIR`
for fitted playback; a bound result already supplies its coefficients. Unfitted
views retain manual oscillation. Spectrum uses saved alpha and modal images;
missing projections are computed on demand into `cache/viewer_projection/`.
It does not recompute FFT, refit alpha or fit coefficients.

## Resume and change boundaries

| Situation | Supported action |
| --- | --- |
| Interrupted matching mode batch | Rerun the same command; reuse published stages and matching checkpoints. |
| Increase only the iteration cap | New output + `--continue-from OLD_BATCH` (or old single-frequency experiment); compatible optimizer/RNG/early-stop state is retained. |
| Changed-code batch attempt | Stop the source batch, then use new output + `--resume-from OLD_BATCH`; matching completed modes retain their identities, unfinished modes run under the new code. |
| Interrupted coefficient preparation | Same command + `--resume`; completed stages are reused, incomplete stages restart. |
| New or interrupted RGB fit | New RGB output directory; there is no optimizer-resume checkpoint. |
| Changed references, inputs, loss or backend | New experiment with matching dependencies; never edit old manifests to force reuse. |

Place `--log-file PATH` before the subcommand and outside a newly published
artifact directory. Preserve failure logs/checkpoints. Farneback and old DFT
execution are historical `legacy` workflows; do not invoke them to fill missing
current inputs automatically.

## Code and detailed documentation

| Area | Start here |
| --- | --- |
| CLI and scene identity/path resolution | [cli.py](src/modal_gaussians/cli.py), [scene_store.py](src/modal_gaussians/scene_store.py), [storage contracts](SCENE_STORAGE.md) |
| Flow, references and FFT | [flow/](src/modal_gaussians/flow/), [spectrum_cache.py](src/modal_gaussians/spectrum_cache.py) |
| GNN, geometry, donors and publication | [motion code map](src/modal_gaussians/motion/README.md), [batch.py](src/modal_gaussians/motion/neural/batch.py) |
| GPU preparation and optional modal projection cache | [GPU alpha](docs/gpu-alpha.md), [GPU propagation](docs/gpu-soft-propagation.md), [projection cache](docs/modal-projection-benchmark-20260920.md) |
| Fixed-mode preparation and RGB optimization | [coefficient_preparation.py](src/modal_gaussians/coefficient_preparation.py), [rgb_coordinates.py](src/modal_gaussians/rgb_coordinates.py), [rgb_fitting.py](src/modal_gaussians/rgb_fitting.py), [rgb_rendering.py](src/modal_gaussians/rgb_rendering.py) |
| Playback and offline export | [vis/](src/modal_gaussians/vis/), [result_video.py](src/modal_gaussians/result_video.py) |
| Additional command recipes | [Command reference](skills/modal-gaussians-pipeline/references/commands.md) (resolve historical examples through current catalogs) |

For coefficient code changes, focused synthetic CPU checks are available via
`python -m unittest discover -s tests -p "test_coefficient_rgb_*.py"`.
Do not substitute real-scene validation for development checks.
