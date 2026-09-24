# Modal Gaussians

Recover complex 3D vibration modes from separately recorded fixed-view videos,
then reconstruct each recording or synthesize user-controlled motion. A sweep
provides static geometry. Fixed-view recordings **are not synchronized** and
must have independent temporal coefficients.

## Start here

For code changes, read [AGENTS.md](AGENTS.md): stage ownership, interface changes,
cache invalidation and development checks.

1. Read [SCENE_STORAGE.md](SCENE_STORAGE.md), the scene's `catalog.json` and
   `results/index.json`. Reuse only inputs with matching identities/contracts.
2. Read [BASELINE.md](BASELINE.md) for the current numerical recipe and
   [REBUILD.md](REBUILD.md) for the breaking cleanup and affected Bush/Corn data.
3. Run only the requested stages. Training, RGB fitting, real-scene validation,
   exports and Viser launches each require task authorization. Synthetic
   development tests are separate from real experiment validation.
4. Publish new experiments under `scene_library/<scene>/experiments/<new_name>/`.
   Preserve raw data and immutable artifacts. Never edit a manifest to bypass a mismatch.

The code now contains one pipeline. Old rigid/attachment models, repeated DFT,
CPU alpha/propagation backends, physics-coordinate fitting, preview wrappers and
old CLI aliases are removed. Existing data was not migrated or deleted during
cleanup; **the cleaned pipeline has not been rerun on Bush or Corn**.

## Pipeline and source map

```text
Sweep + fixed-view videos
  -> frames / masks / optional stabilization -> sequence references
  -> COLMAP + static 3DGS + subject partition
       |                       |
       |                       -> reusable observation geometry + KNN candidates
       -> render-matched motion references -> SEA-RAFT flow
           -> shared FFT -> select bins -> complex U/V modal images
               -> per-frequency complex view gains + soft graph weights
               -> controls + GNN -> saved complex 3D displacement / angular fields
                   |                       |
                   -> manual synthesis     -> mode bank -> flow ridge initialization
                                               -> fixed-mode RGB coefficient fitting
                                                   -> bound result -> offline video / playback
                                                   -> optional prepare-refinement -> refine-scene
                                                       -> new scene / baked modes / coefficients
                                                       -> bound result -> offline video / playback
```

| Stage | Source directory | Main entry points | Disk boundary |
| --- | --- | --- | --- |
| Frames, masks, stabilization | [`preprocessing/`](src/modal_gaussians/preprocessing/) | `frames.py`, `reference.py`, `stabilization.py` | PNGs, sequence reference |
| Cameras, static Gaussians, subject selection | [`geometry/`](src/modal_gaussians/geometry/) | `colmap.py`, `training.py`, `partition.py`, `selection.py` | COLMAP, static scene |
| Motion reference and optical flow | [`flow/`](src/modal_gaussians/flow/) | `reference_selection.py`, `sea_raft.py` | selection, `flow.zarr` |
| Frequency analysis | [`spectrum/`](src/modal_gaussians/spectrum/) | `cache.py`, `selection.py`, `transform.py` | shared FFT, bin selection, modal images |
| Spatial mode learning | [`motion/`](src/modal_gaussians/motion/) | `prepared.py`, `selected_modal.py`, `batch.py`, `training.py`, `network.py` | prepared observations, soft graphs, controls, single-frequency models |
| Temporal coefficients | [`coordinates/`](src/modal_gaussians/coordinates/) | `preparation.py`, `direct.py`, `rgb.py`, `sweep.py`, `fitting.py`, `rendering.py` | fixed mode bank, design, fixed-view/sweep RGB coefficients |
| Optional joint scene refinement | [`coordinates/`](src/modal_gaussians/coordinates/) | `reference.py`, `sequences.py`, `refinement_artifacts.py`, `refinement.py`; `motion/reference_field.py`, `geometry/density.py` | frozen reference inputs, frame/camera bindings, resumable work, derived scene/modes/coefficients |
| Result binding, evaluation and video | [`results/`](src/modal_gaussians/results/) | `artifact.py`, `evaluation.py`, `video.py` | result manifest, metrics/CSV, comparison MP4 |
| Interactive inspection | [`vis/`](src/modal_gaussians/vis/) | `inputs.py`, `viewer.py`, `spectrum.py` | explicit viewer/projection work directory |
| Shared infrastructure | [`common/`](src/modal_gaussians/common/) | `scene_store.py`, `cache.py`, camera math, array I/O | path resolution, cache contracts, atomic publication |

[`cli.py`](src/modal_gaussians/cli.py) only dispatches commands.
`motion/observations/` owns correspondence topology and GPU complex-gain fitting;
`motion/common/` owns projection, graph and donor-transfer math. `_vendor/` holds
licensed XMem inference code. Stage producers own artifact I/O; numerical kernels
operate on arrays/tensors. Disk boundaries remain deliberate development checkpoints.
There is no new in-memory scheduler or GPU-resident pipeline layer.

Registered sweep PNGs/cameras also feed `coordinates fit-sweep`. Its coefficients
join selected fixed-view RGB fits and a verified motion reference at
`prepare-refinement`; the sweep never enters the modal FFT observation set.

## Commands

Use `modal-gaussians <group> <command> --help` for exact arguments.
[Command recipes](skills/modal-gaussians-pipeline/references/commands.md) cover
cold preparation through mode training; [coefficient fitting](COEFFICIENT_FITTING.md)
covers the optional reconstruction branch.

| Stage | Command |
| --- | --- |
| Inspect registered paths | `storage list`, `storage path`, `storage run` |
| Frames/masks; bind video timing/grid | `prepare gui`, `prepare reference` |
| Static reconstruction/partition | `colmap prepare`, `static train`, `static repartition`, `static apply-selection` |
| Match motion reference; infer flow | `flow select-reference`, `flow compute` |
| Cache/select/export FFT bins | `spectrum build`, `spectrum select`, `spectrum export` |
| Prepare geometry once | `motion prepare-neural` |
| Run selected frequencies | `motion batch-neural` |
| Individual frequency stages | `motion prepare-selected-modal`, `graph build-modal-similarity`, `motion prepare-control-weights`, `motion iterate-neural` |
| Fit each video's free coefficients | `coordinates prepare`, `coordinates fit-rgb` |
| Fit moving-camera sweep coefficients | `coordinates fit-sweep --fps 30`, `coordinates downsample-sweep` |
| Refine using selected recordings and optional sweep | `coordinates prepare-refinement`, `coordinates refine-scene` |
| Bind/evaluate/export | `result materialize`, `result evaluate`, `result export-video` |
| Explicit interactive inspection | `viewer`, `spectrum viewer` |

A completed mode batch publishes `index.json`, directly usable by `viewer --input`
and `coordinates prepare --index ... --status complete --expected-modes N`.
`--stage weights` stops a batch before GNN training. Mode training never starts
coefficient fitting or a viewer.

Use `viewer --no-spectrum` for 3D/manual/coefficient playback without loading FFT
sources or the Spectrum panel. Default viewing still validates those sources.

Joint refinement starts after RGB fitting covers the selected recordings (`--view`,
repeatable; default all bank views). Optional sweep supervision uses registered
per-frame cameras and independently fitted coefficients. Fixed-view and sweep groups
have equal loss weight. Original modal observation views remain intact for Spectrum.
Controls and the reference graph stay fixed; live foreground Gaussians can change.
Canonical centers stay near their permanent root's original KNN segments;
children inherit this shape bound and the frequency-specific propagation rules.
The default is two rounds at full resolution (1.0): sparse geometry/coefficient
updates (fixed views 5 FPS, sweep 10 FPS), then frozen-geometry coefficient updates
on every reconstruction frame. Sweep fitting, evaluation and export use a real
30 FPS subset; view1 stays at 30 FPS. Density changes are limited to round 1's
geometry phase. Each coefficient phase reuses one baked GPU basis.
The explicit `tools/import_refinement_reference.py` imports v16 component fields
into a verified immutable motion reference, without retraining or enabling old
training loaders. Current v18 sources use the same reference builder. Training
never runs a GNN. See
[refinement commands and contracts](COEFFICIENT_FITTING.md#joint-scene-refinement).

## Environment and checks

Use the project's CUDA-capable Python environment (Python >=3.11, PyTorch,
CUDA/gsplat, CuPy). Install the package with `python -m pip install -e .` after
configuring PyTorch for the local GPU. COLMAP and FFmpeg are external executables;
SEA-RAFT repository/weights live under `scene_library/_shared/tools/` by default.
Mask preparation additionally uses the `mask` extra. See `pyproject.toml` for dependencies.

```sh
python -m unittest discover -s tests
```

`tests/` is local and Git-ignored, so it may be absent in a fresh clone.
Tests use temporary synthetic data; CUDA-specific tests skip without a GPU.
They do not load scene-library experiments or start Viser. Test success does not
establish real-scene reconstruction quality. No formal reconstruction metrics are
computed automatically. Explicit `result evaluate` measures native-resolution input
PNGs against uncompressed renders (PSNR, SSIM, RMSE; add `--lpips` for LPIPS-Alex).
Install the `evaluation` extra for LPIPS. An MP4 is not a metric input.
