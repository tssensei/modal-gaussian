# Fixed-mode coefficient experiments

Follow [SCENE_STORAGE.md](SCENE_STORAGE.md). Preparation selects records by
exact status from the registered scene's result index, checks the expected count,
and sorts by frequency. For Bush, `completed_uniform60` selects the 40 completed
modes and excludes the accepted 0.744 Hz baseline. It never resumes mode training.

## Storage contract

Use a **new immediate child** of the scene's `experiments/` directory. The
following example is created only when the corresponding commands are run:

```text
C:/Users/zitengsong/Documents/school/research/modal-gaussian/scene_library/bush/experiments/coefficient40_rgb_001/
├── preparation.json              # source/config/code contract and completed stages
├── mode_bank/                    # derived input collection, not newly trained modes
│   ├── manifest.json             # v17 completed_modes; source IDs/slots/frequencies
│   ├── phi.npy                   # complex64 [K,G,3], unchanged saved displacement
│   ├── rotation.npy              # complex64 [K,G,3], cached angular fields
│   └── support.npz               # g_points, support_class[K,G], observation_view_mask[K,G,V]
├── rendered_design/
│   ├── manifest.json             # scene/bank/camera/flow identities and settings
│   ├── design.npy                # float32 [P,2,2K], view samples concatenated
│   └── samples.npz               # pixels, view offsets, image shapes, foreground alpha
├── direct_coordinates/
│   ├── manifest.json             # v2 direct coordinates; frame map and SEA bindings
│   ├── coordinates.npy           # complex64 [sum(T_view),K], ridge initialization
│   └── diagnostics.npz           # mode_pair_scales[V,K] only; no evaluation metrics
├── rgb_coordinates/              # created separately by fit-rgb
│   ├── manifest.json             # RGB settings, source identities and training history
│   └── coordinates.npy           # complex64 [sum(T_view),K], fitted coefficients
├── result/                       # optional, created separately by result materialize
│   └── manifest.json             # links scene, bank, design and fitted coordinates
└── exports/                      # optional, explicit offline exports only
    └── view1_001/                # new directory for each export
        ├── comparison.mp4        # original RGB above reconstructed RGB
        ├── manifest.json         # result identity, view, frame order, FPS and encoding
        └── encode.log            # FFmpeg errors, normally empty
```

For current Bush inputs: `K=40`, `G=231761`, `V=3`. Video lengths are 1170,
1013 and 1020, giving coefficient arrays of `[3203,40]`. `P` depends on image
sampling settings. Phi and rotation together require about 424 MiB; the design
requires `P * 2 * 80 * 4` bytes, plus sample/support metadata.

Original models remain in `results/models/<hash>/`, and original training
checkpoints remain in `checkpoints/<hash>/`. The bank is an experiment input
cache. Preparation neither changes these original files/identities nor publishes
replacement trained modes. It does not copy frames, flow, FFT arrays, frequency
graphs, or networks into the experiment. Historical paths use `resolve_path()`.

## Prepare and reuse

Run from the repository:

```bat
modal-gaussians storage run --scene bush -- coordinates prepare --scene bush --status completed_uniform60 --expected-modes 40 --output @experiments/coefficient40_rgb_001
```

Preparation checks scene/camera/reference identities and exact Gaussian order.
It copies saved Phi without normalization, realignment or retraining. The v16
loader evaluates each saved network once to extract its angular field, with
forensic validation disabled. Saving rotation in the bank avoids repeating this
field extraction when fitting coefficients or loading the result.

The existing rasterized displacement projection supplies the linear design at
the reference pose. Rotation and visibility nonlinearities are handled later by
RGB fitting. The existing pair-normalized ridge solver reads sampled SEA-RAFT
flow pixels, solves all modes jointly per frame, and returns independent,
mean-zero initial coordinates per recording. Its reconstruction and spectral
evaluation passes are skipped.

Both `sea_raft_flow` and `sea_raft_selected_frequency_experiment` are supported;
the latter still stores a full flow sequence. Historical reference metadata
binds geometry and masks; deleted Farneback flow/spectrum arrays are never
opened. SEA motion data receives its own identity, separate from the historical
geometry-reference identity. Stabilized image timing and location must match SEA.

Defaults: pixel stride 2, alpha minimum 0.05, mask erosion 1, 8 modes per render
batch, ridge 1e-4 and 64 frames per flow chunk. See `coordinates prepare --help`.

Existing experiments are rejected unless `--resume` is supplied. Resume checks
source IDs, selected modes, camera/flow bindings, configuration and implementation
revisions before reusing published stages. Each stage publishes atomically.
An interrupted incomplete stage is recomputed; there is no intra-stage checkpoint.
An expanded index or changed settings causes a mismatch, not a silent change of
the selected basis. Use a new experiment for different inputs.

For another RGB trial with the same basis and initialization, directly reference
the existing `mode_bank/` and `direct_coordinates/`. No preparation is necessary.
The registry/catalog and imported result index are not changed automatically;
`preparation.json` records completion for this new experiment.

## Fit and bind the result separately

Preparation does not start RGB fitting, a Viewer, video export or evaluation.
The next explicit commands are:

```bat
modal-gaussians storage run --scene bush -- coordinates fit-rgb --scene @static --modes @experiments/coefficient40_rgb_001/mode_bank --input @experiments/coefficient40_rgb_001/direct_coordinates --config configs/rgb_coordinates.json --output @experiments/coefficient40_rgb_001/rgb_coordinates
modal-gaussians storage run --scene bush -- result materialize --scene @static --modes @experiments/coefficient40_rgb_001/mode_bank --coordinates @experiments/coefficient40_rgb_001/rgb_coordinates --output @experiments/coefficient40_rgb_001/result
```

RGB fitting freezes the fields and scene and uses the matching stabilized PNGs.
The final coefficients already include the fitted pose offset. It saves training
history and final coefficients, without optimizer-resume checkpoints or an
additional PSNR/SSIM/LPIPS evaluation pass. New RGB trials require new output
directories. The historical `physics-fit` requires evaluated v1 direct inputs;
the new v2 direct artifact is intended for RGB initialization.

### Fit only one recording

Add `--view view2` to fit only that video; omitting `--view` retains all-view
fitting. The shared mode bank, rendered design and direct initialization can
remain unchanged with all three views. Only the selected recording's RGB frames
are opened and optimized, using its original initialization slice and mode-pair
scales. No source artifacts are rewritten and no modes are retrained.

```bat
modal-gaussians storage run --scene bush -- coordinates fit-rgb --scene @static --modes @experiments/coefficient40_rgb_001/mode_bank --input @experiments/coefficient40_rgb_001/direct_coordinates --view view2 --config configs/rgb_coordinates.json --output @experiments/coefficient40_rgb_001/rgb_coordinates_view2
modal-gaussians storage run --scene bush -- result materialize --scene @static --modes @experiments/coefficient40_rgb_001/mode_bank --coordinates @experiments/coefficient40_rgb_001/rgb_coordinates_view2 --output @experiments/coefficient40_rgb_001/result_view2
modal-gaussians storage run --scene bush -- result export-video --result @experiments/coefficient40_rgb_001/result_view2 --view view2 --output @experiments/coefficient40_rgb_001/exports/view2_001
```

`rgb_coordinates_view2/` contains the usual `manifest.json` and `coordinates.npy`,
but only `[T_view2,K]` coefficients (`[1013,40]` for the current Bush inputs).
Its view index and frame offset start at zero, while the original view label,
frame names and source identities stay bound to the original artifacts.
`result_view2/manifest.json` binds this single fitted recording; playback and
video export use its original camera. Geometry observation colors still describe
all views that contributed to the fixed modes. There are no additional input
caches. Use a new output directory for every trial.

The Python fitting API accepts `view_label="view2"`. Unknown labels fail before
optimization. With `--view`, any `--images` override must name that selected view.
This selection applies to RGB fitting, not the preceding `coordinates prepare`
stage, which continues to prepare the shared inputs for all source views.

The bank result supports RGB/manual-mode playback, observation colors and a
floating **Spectrum** panel. The panel resolves each bank entry to its original
model and exported modal image, checks the saved shared-FFT identity and exact
bin, and reads the cached frequency curve and reference image. Only the selected
view's display data is opened; switching frequency does not compute FFT, load
flow, replay a network or refit alignment. Saved training alphas align the fixed
mode's rendered projection with the U/V modal image and 3D phase colors.

All source cameras remain available for mode visualization even when RGB
coefficients were fitted for just one video. Coordinate playback still includes
only fitted recordings. The panel's Solo button and frequency/U/V selectors are
synchronized with the 3D controls. Brightness can be normalized per mode or
across **all saved modes**; this does not scan unselected FFT bins.

The full input spectrum averages the saved cache analysis region. The projected
mode curve averages rendered-design model-support samples. The panel labels
these different spatial domains explicitly; neither curve is a video-fit metric.
Select `Model support` for directly comparable input/projection image coverage,
or `Cached region` to see the input over the larger saved region. Dim pixels
outside projection support show the reference RGB, not zero-motion predictions.

The old flow/measurement-based Spectrum loader and display-only alpha fitting
have been removed. Legacy single-artifact viewers retain their 3D controls;
the new floating panel uses v17 mode banks. Per-frequency graph/control overlays
are still not merged into a single bank graph. No new disk cache or changed
scientific artifact is created by this panel.

## Export one fitted view offline

After RGB fitting and result materialization, explicitly run:

```bat
modal-gaussians storage run --scene bush -- result export-video --result @experiments/coefficient40_rgb_001/result --view view1 --output @experiments/coefficient40_rgb_001/exports/view1_001
```

`--view` selects exactly one recorded label (`view1`, `view2`, or `view3` for
Bush); use a new output directory for another view or export. Python callers
can use `modal_gaussians.result_video.export_result_video(result_dir=...,
view_label="view1", output_dir=..., device="cuda")`.

The upper panel uses the exact PNG inputs recorded by RGB fitting, including
any `--images` override. By default these are **stabilized source video frames**,
not the unwarped raw video. The lower panel renders the same camera and frame's
final coefficients, including the fitted offset, with frozen displacement and
rotation fields and the static background. Frames keep their recorded order and
FPS, independently for each recording. Export does not fit or modify anything.

Each panel retains native resolution: current Bush output is 1920x2160 at
30 FPS. Odd widths receive one black column on the right for H.264 compatibility.
Encoding uses H.264 CRF 18, yuv420p, no audio; this lossy comparison video is for
viewing, not an input for reconstruction-quality metrics. FFmpeg must be on PATH
or in the active Conda environment. Frames stream directly to the encoder;
no PNG sequence is stored. Successful completion publishes the directory
atomically, and existing outputs are rejected. No export or metrics run is
automatically triggered by fitting or preparation.

Synthetic CPU checks (including tiny FFmpeg encode/decode), without real scene data or CUDA:

```bat
python -m unittest discover -s tests -p "test_coefficient_rgb_*.py"
```
