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

The accepted stabilization baseline is now **background camera-pose estimation
against a fixed COLMAP map, followed by depth-aware reprojection to one camera**.
See [BASELINE](BASELINE.md#video-stabilization-baseline--accepted-2026-09-24).
`prepare reference --scene STATIC --view LABEL` uses it by default. Only explicitly
confirmed tripod recordings use `--tripod`. The old `--stabilize` flag and 2D
homography implementation are removed. The moving sweep remains a COLMAP input.
The new method requires COLMAP and static-scene depth before stabilized references
can be produced; [REBUILD](REBUILD.md) records this ordering for the next 540p run.

The motion subject is defined by the **union of user-saved 3D boxes on the current static scene**:
`viewer --scene STATIC --select-subject --work-dir SELECTION_WORK`, followed by
`static apply-selection --scene STATIC --selection SAVED_NPZ --output SUBJECT_SCENE`.
Use that subject scene for downstream graph/control preparation and modal learning.
XMem masks still serve bootstrap/stabilization; they do not override the selected
3D motion subject. Pause for the user's box if no matching selection exists.
In the selection viewer, `Add box` duplicates the active box; choose `Active box`
to move/rotate/resize it, or `Remove active box` to remove it (at least one remains).
`Save selection` saves every box and the union's Gaussian indices.
Initial/reset boxes use foreground coordinate percentiles 5–95 with a 10% margin;
this is an editing starting point, not point removal. Size sliders have separate
per-axis ranges (twice the active box size when loaded/switched/reset) and fine
steps (0.01% of that size). Saved box geometry is preserved when reopened.

The static bootstrap now uses a coarse **3,000-update, batch-4 SH** recipe:
`static train --iterations 3000`. SH grows from degree 0 to 3 and is retained by
coefficient fitting, refinement and playback. See [BASELINE](BASELINE.md) for
the author-aligned learning rates/density rules and retained project differences.
Posed Depth Anything 3 supervision is available through `static prepare-depth`
and `static train --depth DEPTH`; it uses the existing COLMAP cameras. Old direct-RGB scenes require a new static
run and downstream rebuild; no data is migrated automatically.
Static training uses full-frame RGB L1 + 0.2 DSSIM, plus optional depth L2; the mask loss and
`--mask-weight` option are removed. Masks remain inputs for initial partitioning,
stabilization and modal observation support.

## Pipeline and source map

```text
Sweep + fixed-view videos
  -> raw frames / masks -> COLMAP -> optional DA3 depth -> static 3DGS + subject partition
       -> background poses + depth stabilization (explicit tripod bypass)
           -> sequence references + reusable observation geometry / KNN
           -> render-matched motion references -> SEA-RAFT flow
           -> shared FFT -> select bins -> complex U/V modal images
               -> per-frequency complex view gains + soft graph weights
               -> controls + GNN -> saved complex 3D displacement / angular fields
                   |                       |
                   -> manual synthesis     -> mode bank
                                               -> flow ridge -> fixed-mode RGB fitting
                                                   -> bound result -> video / playback
                                               -> prepare-refinement + recorded videos
                                                   -> zero-q warmup -> refine-motion
                                                   -> fixed scene + refined modes / coefficients
                                                   -> bound result -> video / playback
```

| Stage | Source directory | Main entry points | Disk boundary |
| --- | --- | --- | --- |
| Frames, masks, stabilization | [`preprocessing/`](src/modal_gaussians/preprocessing/) | `frames.py`, `reference.py`, `stabilization.py` | PNGs, sequence reference |
| Cameras, depth, static Gaussians, subject selection | [`geometry/`](src/modal_gaussians/geometry/) | `colmap.py`, `depth.py`, `training.py`, `partition.py`, `selection.py` | COLMAP, DA3 targets, static scene |
| Motion reference and optical flow | [`flow/`](src/modal_gaussians/flow/) | `reference_selection.py`, `sea_raft.py` | selection, `flow.zarr` |
| Frequency analysis | [`spectrum/`](src/modal_gaussians/spectrum/) | `cache.py`, `selection.py`, `transform.py` | shared FFT, bin selection, modal images |
| Spatial mode learning | [`motion/`](src/modal_gaussians/motion/) | `prepared.py`, `selected_modal.py`, `batch.py`, `training.py`, `network.py` | prepared observations, soft graphs, controls, single-frequency models |
| Temporal coefficients | [`coordinates/`](src/modal_gaussians/coordinates/) | `preparation.py`, `direct.py`, `rgb.py`, `sweep.py`, `fitting.py`, `rendering.py` | fixed mode bank, design, fixed-view/sweep RGB coefficients |
| Optional joint motion refinement | [`coordinates/`](src/modal_gaussians/coordinates/) | `reference.py`, `sequences.py`, `refinement_artifacts.py`, `refinement.py`; `motion/reference_field.py`, `motion/fixed_field.py` | fixed reference/operator, frame/camera bindings, resumable work, modes/coefficients |
| Result binding, evaluation and video | [`results/`](src/modal_gaussians/results/) | `artifact.py`, `evaluation.py`, `video.py` | result manifest, metrics/CSV, comparison MP4 |
| Interactive inspection | [`vis/`](src/modal_gaussians/vis/) | `inputs.py`, `viewer.py`, `spectrum.py` | explicit viewer/projection work directory |
| Shared infrastructure | [`common/`](src/modal_gaussians/common/) | `scene_store.py`, `cache.py`, camera math, array I/O | path resolution, cache contracts, atomic publication |

[`cli.py`](src/modal_gaussians/cli.py) only dispatches commands.
`motion/observations/` owns correspondence topology and GPU complex-gain fitting;
`motion/common/` owns projection, graph and donor-transfer math. `_vendor/` holds
licensed XMem inference code. Stage producers own artifact I/O; numerical kernels
operate on arrays/tensors. Disk boundaries remain deliberate development checkpoints.
There is no new in-memory scheduler or GPU-resident pipeline layer.

Registered sweep PNGs/cameras also feed standalone `coordinates fit-sweep`.
Motion refinement accepts the videos directly with zero-start coefficient warmup;
its preparation never fits q. Sweep never enters the modal FFT observation set.

## Commands

Use `modal-gaussians <group> <command> --help` for exact arguments.
[Command recipes](skills/modal-gaussians-pipeline/references/commands.md) cover
cold preparation through mode training; [coefficient fitting](COEFFICIENT_FITTING.md)
covers the optional reconstruction branch.

The independent `tools/compare_flow_coordinates.py` diagnostic compares reference-only
and reference-plus-adjacent flow coefficients with a shared RGB reference offset.
Its stages, assumptions and outputs are described in [coefficient fitting](COEFFICIENT_FITTING.md#reference--adjacent-flow-diagnostic).
It does not replace production coefficient fitting or motion refinement.
Its reference-only solution can explicitly initialize motion refinement with
`coordinates refine-motion --flow-initialization DIAGNOSTIC_ROOT`. This selects
the diagnostic's fixed-view prefix, preserves its shared offset and normalization,
and skips zero-q warmup. See the reconstruction document for validation/recovery.

| Stage | Command |
| --- | --- |
| Inspect registered paths | `storage list`, `storage path`, `storage run` |
| Frames/masks; bind video timing/grid | `prepare gui`, `prepare reference` |
| Static reconstruction/partition | `colmap prepare`, `static prepare-depth`, `static train`, `static repartition`, `static apply-selection` |
| Match motion reference; infer flow | `flow select-reference`, `flow compute` |
| Cache/select/export FFT bins | `spectrum build`, `spectrum select`, `spectrum export` |
| Prepare geometry once | `motion prepare-neural` |
| Run selected frequencies | `motion batch-neural` |
| Individual frequency stages | `motion prepare-selected-modal`, `graph build-modal-similarity`, `motion prepare-control-weights`, `motion iterate-neural` |
| Fit each video's free coefficients | `coordinates prepare`, `coordinates fit-rgb` |
| Fit moving-camera sweep coefficients | `coordinates fit-sweep --fps 30`, `coordinates downsample-sweep` |
| Refine using selected recordings and optional sweep | `coordinates prepare-refinement`, `coordinates refine-motion` |
| Bind/evaluate/export | `result materialize`, `result evaluate`, `result export-video` |
| Explicit interactive inspection | `viewer`, `spectrum viewer` |

A completed mode batch publishes `index.json`, directly usable by `viewer --input`
and `coordinates prepare --index ... --status complete --expected-modes N`.
`--stage weights` stops a batch before GNN training. Mode training never starts
coefficient fitting or a viewer.

Use `viewer --no-spectrum` for 3D/manual/coefficient playback without loading FFT
sources or the Spectrum panel. Default viewing still validates those sources.

Motion refinement freezes the entire static scene and learns control-point modal
corrections plus independent per-frame q. Preparation accepts selected recordings
(`--view`, repeatable; default all) and optional sweep metadata, with no prior RGB
fit required. Use a verified fixed reference; v16 sources go through the explicit
importer. Original modal observation views remain intact. The default is ten
full-frame zero-q warmup passes, then two rounds of sparse joint updates (fixed
5 FPS, sweep 10 FPS) and exhaustive coefficient passes, all at resolution 1.0.
Sweep is a real 30 FPS subset. Dynamic KNN regularizers preserve local motion;
control-field anchors limit drift from original modes. The final bank/coordinates
reference the unchanged static scene. This is distinct from fixed-mode fitting.
See [motion refinement](COEFFICIENT_FITTING.md#fixed-scene-motion-refinement).
Optional config-driven early stopping checks full-sequence RGB loss, retains the
best state and publishes it on plateau or budget exhaustion; it is disabled by
default. This measures training-set convergence, not held-out quality.

## Environment and checks

Use the project's CUDA-capable Python environment (Python >=3.11, PyTorch,
CUDA/gsplat, CuPy). Install the package with `python -m pip install -e .` after
configuring PyTorch for the local GPU. COLMAP and FFmpeg are external executables;
SEA-RAFT repository/weights live under `scene_library/_shared/tools/` by default.
Mask preparation additionally uses the `mask` extra. See `pyproject.toml` for dependencies.
DA3 inference uses a separate environment via `static prepare-depth --python`:
its upstream NumPy < 2 requirement conflicts with this project's NumPy >= 2.
Use a local multi-view DA3 snapshot (for example DA3-LARGE-1.1). The command records
weight/source hashes; it does not install packages, download models, or start training.
See the [command recipes](skills/modal-gaussians-pipeline/references/commands.md).

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
