# Modal Gaussians

Standalone reconstruction tools for asynchronous multi-view modal Gaussian
analysis. The first migrated vertical slice validates ordered image/mask
sequences, optionally stabilizes them to a reference frame, computes dense
reference-to-frame Farneback flow, and evaluates the per-pixel temporal FFT.

The current user-selected Corn motion baseline is
`corn_neural_fragment_propagation_001`: a neural complex modal field with all
spatial mutual-KNN edges retained, deformation/rotation penalties both 0.1,
and post-training local fragment motion propagation. The accepted artifact
contains one frequency, **0.225 Hz**. See [BASELINE.md](BASELINE.md) for its
result identity, exact parameters, metrics, launch command, and historical references.

## Code organization

Motion implementations are grouped under `src/modal_gaussians/motion/`:

- [`neural/`](src/modal_gaussians/motion/neural): geometry/control graphs, GNN,
  neural training/artifacts, and fragment propagation.
- [`rigid/`](src/modal_gaussians/motion/rigid): component graphs, rigid solves,
  sequential fill, motion-basis fitting, and green refinement.
- [`common/`](src/modal_gaussians/motion/common): unified completed-modes loading,
  source validation, frequency mapping, projection, and geometry helpers.

See the [motion code map](src/modal_gaussians/motion/README.md) for entry points
and dependencies. Static reconstruction, observations, coordinates, result
packaging, and Viewer remain shared. The former root-level motion aliases have
been removed; Python imports now use the `motion/` packages. CLI commands,
artifact formats, and experiment parameters are unchanged.

## Environment

Create the project Conda environment from the repository root:

```powershell
conda env create --file environment.yml
conda activate modal-gaussian
```

`environment.yml` fixes Python 3.11, the CUDA Toolkit at exactly 12.8, and
PyTorch 2.7.1 for CUDA 12.8. `pyproject.toml` contains the ordinary Python package
dependencies. COLMAP remains an external executable: it may be available on
`PATH`, or its absolute path may be passed with `--colmap-command`.
The environment also installs the pinned `gsplat` rasterizer used by the
independent static-3DGS stage. The COLMAP preparation command itself does not
import PyTorch or require a Python COLMAP module.

On Windows, the first static render compiles gsplat's CUDA extension once. The
loader discovers Conda's Ninja and Visual Studio 2022 C++ Build Tools, and
translates the two GCC-only flags emitted by gsplat 1.5.3. Linux cluster runs do
not use this compatibility path. Viewer performs this compiler/backend setup
synchronously before opening its server, so a setup problem fails at startup
instead of leaving a connected browser whose background renders repeatedly fail.

## Live text progress

CLI stages report start/completion/failure immediately. Static training reports
step/epoch/batch, loss, PSNR, SSIM, FG/BG Gaussian counts, and density-control
events. Offline QA reports completed cameras; flow/FFT, greedy selection, dense
DFT, rigid/basis/fill, rendered design, and coordinate loops report their
completed units. COLMAP output streams as the executable emits lines.

Progress is normally throttled to five seconds, with immediate first/last-unit
and epoch/mode reports. Elapsed time and ETA refer to the named phase, not the
entire pipeline; resumed training estimates speed only from newly completed
steps. One phase reaching 100% does not bypass artifact validation/publication.
No new rendering previews or viewer are started for monitoring.

Optionally append these progress messages and Python failure tracebacks to an
external UTF-8 log. Place `--log-file` **before** the subcommand and outside the
artifact target (which must not already exist):

```powershell
modal-gaussians --log-file C:\outputs\scene_run\logs\static_train_attempt01.log static train `
  --input C:\data\joint_colmap `
  --work-dir C:\outputs\scene_run\work\static `
  --output C:\outputs\scene_run\static_scene
```

After the log is created, follow it from another PowerShell terminal:

```powershell
Get-Content -LiteralPath C:\outputs\scene_run\logs\static_train_attempt01.log -Tail 20 -Wait
```

Ctrl+C stops only the log follower. Logs append; use separate files for separate
stage attempts. This progress log is not a full stdout/stderr capture of every
third-party library, so retain process output too when debugging. Logging does
not change scientific configuration, array ordering, or resume state.

## Optional video and mask preparation

The local preparation app preserves SAM ViT-H prompting and XMem-s012
forward/backward tracking. It is independent of Shape-of-Motion and does not
change the analysis pipeline: analysis still starts from prepared PNGs/masks.
No mask dependencies are imported by the other commands.

Install the optional tools in the **same** Conda environment. First inspect the
installation plans; do not replace the existing PyTorch/CUDA/gsplat versions:

```powershell
conda activate modal-gaussian
conda install -c conda-forge ffmpeg --freeze-installed --dry-run
python -m pip install --dry-run --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[mask]"

conda install -c conda-forge ffmpeg --freeze-installed
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[mask]"
python -m pip check
```

The extra pins Gradio 6.21.0, torchvision 0.22.1+cu128 (matching
PyTorch 2.7.1+cu128), and official SAM at commit
`dca509fe793f601edb92606367a655c15ac00fdf`. FFmpeg supplies both `ffmpeg`
and `ffprobe`. No second Conda environment is needed. The vendored XMem
inference code and third-party notices are under `src/modal_gaussians/_vendor/xmem`.

Place these two official model downloads directly in `checkpoints/masking/`:

- [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- [XMem-s012.pth](https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth)

The app checks both files before launch, records their SHA-256 in sequence
metadata, and never downloads weights during a tracking job. XMem loads the
complete checkpoint strictly, without extra pretrained-ResNet downloads.

From the repository root, launch:

```powershell
modal-gaussians prepare gui `
  --root-dir data/prepared `
  --checkpoint-dir checkpoints/masking `
  --port 8890
```

Open `http://127.0.0.1:8890` locally. The server is loopback-only, with sharing
and Gradio usage analytics disabled; no video upload is needed. CUDA is the default;
`--device cpu` is an explicit, slower alternative. This is a single-editor app:
do not edit the same prompt in multiple browser tabs.

Extraction and mask tracking share one page. Wide windows show three columns:
video extraction on the left, frame selection and point prompting in the middle,
and the current mask preview plus tracking controls on the right. Narrow windows
wrap the columns automatically. The middle image stays unmodified; clicks update
the mask overlay and point markers in the right-hand preview.

1. **Extract video (optional):** enter a local MOV/MP4 path (for example
   `C:\Users\zitengsong\Documents\school\research\modal-gaussian\data\videos\corn1.mov`),
   a sequence name such as `corn1`, and the desired output FPS. FPS is required;
   choose it for your experiment, not a guessed default. Blank end uses the
   rest of the video; blank height preserves resolution. FFmpeg honors rotation
   metadata, preserves aspect ratio when resizing, and performs no crop.
2. **Load sequence:** enter the sequence name; leave the image path blank to
   use `root/images/name`, or enter an external PNG directory to use it in place.
   Extracted sequences recover FPS from metadata; external sequences need their
   actual FPS. A read-only line shows exactly which input and mask output are
   currently bound. Editing the input fields takes effect only after Load.
3. Select a clear prompting frame and click **Get SAM features**. Use positive
   foreground and negative background clicks. **Add foreground / start another**
   unions the current object with previous foregrounds; **Clear current points**
   discards only the current object; **Clear all masks** starts over. Changing
   the frame or loaded sequence clears all prompts and features.
4. Click **Track / replace masks**. XMem tracks every frame forward/backward
   from the prompt, saving masks incrementally into a temporary directory.
   SAM VRAM is released before tracking; XMem memory resets between directions.
   Cancel takes effect between frames. There is no whole-video preview or
   extra Save Masks step; the output is published after complete validation.

The SAM prompting frame is **not** implicitly the later optical-flow reference.
Choose that reference separately when running `flow analyze`.

```text
data/prepared/
├── images/<sequence>/00001.png, 00002.png, ...
├── masks/<sequence>/00001.png, 00002.png, ...
└── metadata/<sequence>.json
```

Inputs and masks must contain at least three equal-sized, lexicographically
ordered PNGs with matching names. Masks are single-channel uint8, background
0 / foreground 255. Metadata (kept outside PNG directories) records source,
clip interval, output FPS, size/count, paths, prompting frame and model hashes.
External image directories are neither copied nor modified.

**Overwrite rules:** same-name preparation reruns need no renaming or overwrite
flag. The app replaces whole directories so shorter reruns leave no stale
frames. Successful re-extraction **deletes that sequence's old masks** and marks
masks pending; retracking replaces masks only. Other sequences, original videos,
external images and model weights are never removed. Generation/cancellation or
a caught publication failure preserves the previously published result. During
publication, same-filesystem renames and temporary rollback copies are used
(multi-directory replacement is not a single atomic operation on Windows).
Do not read the same outputs in another pipeline while replacing them.

If the OS/process is killed during publication, `.prepare.lock` and a
`.prepare-*` directory can remain. Inspect `publication.json` and `previous-*`
rollback copies before recovery; do not blindly delete these directories. There
is no automatic mid-track checkpoint/resume. Normal successful or cancelled
operations clean their temporary files and do not keep historical backups.

Overwriting inputs referenced by an old experiment may invalidate identities or
change its reference-image visualization. Archive those inputs yourself if an
old experiment must remain reproducible. The core flow/COLMAP/modal artifacts
retain their existing non-overwrite rules. `data/`, `checkpoints/` and `outputs/`
are Git-ignored; original raw videos stay where you placed them. Type checking
covers the project and mask wrappers; the untyped third-party XMem snapshot is
explicitly excluded from diagnostics to preserve its inference implementation.

## Current command

```powershell
modal-gaussians flow analyze `
  --images C:\path\to\images `
  --masks C:\path\to\masks `
  --fps 30 `
  --reference-frame 000120 `
  --output C:\path\to\flow_analysis
```

Add `--stabilize` to run the accepted reference-anchored background homography
stage. Add `--smoothing weighted-gaussian` to apply the migrated
Davis-inspired contrast-weighted spatial flow filter before the FFT. Both are
disabled by default and their settings are recorded in `manifest.json`.

The command accepts zero-padded PNG frames and matching binary PNG masks only.
Image filenames are sorted lexicographically to define temporal order, and mask
stems must match them exactly. The command never overwrites a completed output
directory.

### Lossless chunked flow storage

New flow artifacts use format version 7:

```text
flow_analysis/
├── manifest.json
├── flow.zarr/       # [T,H,W,2] float32
├── spectrum.zarr/   # [floor(T/2)+1,H,W,2] complex64
└── mask_union.npy   # [H,W] bool
```

The Zarr v3 directories use lossless Zstd compression and sharding (multiple
chunks per file). They retain **every pixel, frequency and numerical bit**;
there is no float16 conversion, resolution reduction, ROI crop or new masking.
The existing smoothing option still applies its original mask rule. Compression
savings depend on the data; unsmoothed background flow is generally less
compressible. Copy each entire `.zarr` directory, not just `zarr.json`.

PNG decoding, Farneback and smoothing now run frame-by-frame in small batches.
Optional stabilization uses temporary compressed arrays. Full rFFT and selected
exact-DFT exports read spatial tiles with a target input budget of 32 MiB per
tile, plus transform/codec working memory. Downstream solvers and the spectrum
panel read only their requested pixels; they do not expand the complete store.
The full rFFT remains available to the GUI, and selected dense mode files remain
ordinary memory-mapped `.npy` arrays.

Use `load_flow_analysis_artifact(path)` to obtain lazy `arrays.flow` and
`arrays.spectrum` handles; slice them instead of calling `np.asarray` on the
complete array. For paired `(x,y)` samples use `flow.storage.read_pixels`, which
supports both storage formats and preserves the legacy sampling layout.
The manifest records shape, dtype, chunk/shard layout and per-store SHA-256.
Incomplete or damaged stores fail validation; publication remains atomic and
non-overwriting.

Existing version-6 `flow.npy` / `spectrum.npy` artifacts and their identities
remain readable without conversion, including existing downstream results.
Existing experiments are not modified or reduced in size automatically. New
version-7 artifacts have new identities: do not replace an ancestor of an old
result with a newly generated flow store or rewrite its hashes to match.
CLI arguments and experiment JSON settings are unchanged. Zarr 3.1.6 is pinned
in `pyproject.toml` and uses the existing Python 3.11 environment.

## Joint COLMAP preparation

Before static 3D Gaussian training, prepare one joint COLMAP reconstruction from
sampled moving-camera sweep frames and exactly one reference from each fixed
modal view:

```powershell
modal-gaussians colmap prepare `
  --frames C:\data\sweep\images `
  --frame-masks C:\data\sweep\masks `
  --reference view1 C:\data\view1\ref.png C:\data\view1\ref_mask.png `
  --reference view2 C:\data\view2\ref.png C:\data\view2\ref_mask.png `
  --sample-stride 10 `
  --output C:\data\joint_colmap
```

The command naturally sorts the PNG sweep sequence, samples every Nth frame,
and stages the sampled frames and references in one COLMAP reconstruction. It
runs feature extraction, exhaustive matching, and sparse mapping. A result is
published only if one sparse component registers every sampled frame and every
reference.

COLMAP extracts features from the complete RGB images so that its sparse cloud
contains both foreground and background geometry. Semantic foreground masks
are copied into the result, but are used only later to divide sparse points
between the two static Gaussian sets.

The confirmed experiment branch is fixed in this command: COLMAP uses the
`SIMPLE_RADIAL` model, the sweep shares one set of intrinsics, and all
references share a second set. This matches a moving sweep camera plus fixed
views captured with one other unchanged camera.

The output is deliberately small:

```text
joint_colmap/
├── images/
│   ├── sweep/                    # sampled sweep RGB frames
│   └── references/               # reference RGB frames
├── masks/                        # same relative layout as images
├── sparse/0/                     # selected binary COLMAP model
├── cameras.json                  # K, full camera parameters, w2c and c2w
├── point_cloud.ply               # sparse COLMAP point cloud
└── colmap.log
```

`cameras.json` separates sweep and reference records. Its poses and PLY use the
raw COLMAP world coordinates, whose scale is arbitrary. For distorted models
such as `SIMPLE_RADIAL`, the 3x3 K matrix is not the complete projection model;
the JSON therefore also preserves the COLMAP model name and full parameter
vector.

On the cluster, activate the existing `colmap` Conda environment so that the
executable is on `PATH`, or pass its absolute path through `--colmap-command`.
No Python COLMAP module is required.

## Static foreground/background 3DGS

Train the independent static scene from a completed joint-COLMAP directory:

```powershell
modal-gaussians static train `
  --input C:\data\joint_colmap `
  --work-dir C:\outputs\scene_static_work `
  --output C:\outputs\static_scene
```

The accepted defaults are 100 epochs, batch size 8, at most 40,000 initial
foreground points, at most 80,000 initial background points, and random seed
42. Foreground densification stops at step 4,000; background densification
stops earlier at step 1,000 and background growth is capped at 160,000
Gaussians. With the planned full-20-FPS COLMAP training set this is expected to
produce roughly 4,800 optimizer steps, leaving about 800 final steps without
densification.

Both sampled sweep frames and registered reference frames participate in the
RGB loss. Their COLMAP K and poses remain fixed. The semantic masks supervise a
jointly depth-ordered foreground-membership channel with weight 1.0, using the
same 7×7 erosion kernel and 98% trimmed L1 convention as the accepted old static
run. RGB is ignored only in the uncertain eroded mask boundary. Depth inputs
and the inverse-depth losses are deliberately not connected yet; all depth
weights remain zero until the aligned-depth artifact is specified.

The work directory stores `resume.pt` and, after training, an offline `qa/`
render set. Resume only when the joint-COLMAP input and all resolved training
settings are unchanged:

```powershell
modal-gaussians static train `
  --input C:\data\joint_colmap `
  --work-dir C:\outputs\scene_static_work `
  --output C:\outputs\static_scene_retry `
  --resume
```

The published `static_scene/` contains only `manifest.json`, a class-free
`tensors.pt`, and `training_summary.json`. It contains separate, stable final
foreground and background index domains and no trajectory, motion-basis, or
modal placeholder state. Render the stored sweep/reference cameras again with:

```powershell
modal-gaussians static render `
  --scene C:\outputs\static_scene `
  --output C:\outputs\static_scene_qa `
  --role all
```

`SIMPLE_RADIAL` parameters remain recorded for provenance, but this first
static implementation intentionally preserves the accepted pinhole-K rendering
convention used by the downstream projection logic.

The corresponding Python interface is intentionally direct:

```python
from modal_gaussians.static import cameras_from_scene_manifest, load_static_scene

scene = load_static_scene("C:/outputs/static_scene", "cuda")
camera = cameras_from_scene_manifest(scene.manifest)[0]
result = scene.render(
    camera,
    composition="all",  # all | foreground | background
    outputs=("rgb", "alpha", "expected_depth"),
)
```

## Pixel-to-Gaussian observation topology

Build one mode-independent topology after accepting a static scene. Each
`--view` binds a reference-camera label from the static bundle to the matching
fixed-view flow artifact:

```powershell
modal-gaussians topology build `
  --scene C:\outputs\static_scene `
  --view view1 C:\outputs\view1_flow `
  --view view2 C:\outputs\view2_flow `
  --output C:\outputs\observation_topology
```

For every sampled pixel inside the eroded flow mask, the command requires valid
foreground alpha and expected depth, unprojects a canonical surface point,
preselects nearby foreground Gaussians, and ranks them by opacity-weighted 3D
Mahalanobis contribution. It retains up to four positive-depth contributors,
normalizes their weights per pixel, and stores the pinhole projection Jacobian
for each contributor. Background Gaussian indices never enter this artifact.

The output is deliberately limited to two files:

```text
observation_topology/
├── manifest.json
└── topology.npz
```

The NPZ uses ragged `sample_offsets` rather than duplicating pixel data for
every contributor. The manifest binds the topology to the exact static-scene,
foreground-Gaussian, reference-camera, and flow-artifact identities. All
scientific thresholds are included in the topology identity.

## Automatic greedy frequency selection

Select one shared ordered prefix of `K` frequencies from an inclusive candidate
grid. The `--view` entries must use the same labels and order as the topology:

```powershell
modal-gaussians frequency select `
  --topology C:\outputs\observation_topology `
  --view view1 C:\outputs\view1_flow `
  --view view2 C:\outputs\view2_flow `
  --min-hz 0.2 `
  --max-hz 4.0 `
  --step-hz 0.025 `
  --count 20 `
  --output C:\outputs\frequency_selection
```

For every view, selection samples raw reference-to-frame flow at the topology
pixels, removes its temporal mean, applies the symmetric Hann window, and
evaluates an exact DFT at every candidate frequency. One frequency is a paired
real/imaginary group. At each step, greedy selection adds the candidate with
the largest equal-view mean R2; ties select the lower frequency. The result is
kept in greedy order rather than sorted by frequency.

```text
frequency_selection/
├── manifest.json
└── selection.npz
```

This stage intentionally has no manual peak picking, GUI, editable shortlist,
regional plots, or 3D modal solve. The artifact is bound to the exact topology
and per-view flow identities.

## Dense complex 2D modal fields

Export the full-image complex `(u,v)` field at every selected frequency. Views
must follow the exact label order recorded by the selection artifact:

```powershell
modal-gaussians frequency export-modes `
  --selection C:\outputs\frequency_selection `
  --view view1 C:\outputs\view1_flow `
  --view view2 C:\outputs\view2_flow `
  --output C:\outputs\complex_2d_modes
```

The exporter repeats the accepted temporal-mean detrend and symmetric Hann
window, then evaluates the negative-exponent exact DFT at the selected greedy
prefix. It does not apply the foreground mask, normalize amplitudes, clamp
outliers, or sort frequencies. Each view is written directly as a memory-
mappable `[K,H,W,2] complex64` array:

```text
complex_2d_modes/
├── manifest.json
├── view_000.npy
├── view_001.npy
└── ...
```

The dense rFFT remains in each flow artifact for future full-spectrum GUI use.
The selected exact-DFT arrays are the scientific inputs for later Gaussian
measurement construction.

## Gaussian measurement bank

Sample the dense complex fields only at the pixels retained by the observation
topology:

```powershell
modal-gaussians measurements build `
  --topology C:\outputs\observation_topology `
  --modes C:\outputs\complex_2d_modes `
  --output C:\outputs\gaussian_measurements
```

```text
gaussian_measurements/
├── manifest.json
└── measurements.npy
```

`measurements.npy` is a memory-mappable `[K,P,2] complex64` array. `K` follows
the greedy mode order and `P` follows the topology sample order; the last axis
is complex `(u,v)` displacement. Sampling is an exact integer-pixel lookup with
no interpolation, mask, normalization, or amplitude clamp.

The bank deliberately does not duplicate Gaussian indices, contributor
weights, or projection Jacobians. Those remain in the topology artifact. A
later 3D solver joins the two artifacts by sample index, so one measured pixel
may constrain several Gaussian contributors while one Gaussian may receive
measurements from many pixels and views.

## Observed structure graph

Build the mode-independent local-rigid structure graph after accepting the
static scene and observation topology:

```powershell
modal-gaussians graph build `
  --scene C:\outputs\static_scene `
  --topology C:\outputs\observation_topology `
  --output C:\outputs\observed_structure_graph
```

The defaults are the accepted rigid-components settings: mutual 8-NN,
normalized-world maximum distance `0.008`, color/depth MAD multiplier `3`, five
depth-profile samples, one required supporting view, and minimum component size
of four nodes and three edges.

```text
observed_structure_graph/
├── manifest.json
└── graph.npz
```

Observed nodes are foreground Gaussians with a positive topology contribution.
Candidate edges pass, in order, the absolute distance limit, adaptive Lab-color
threshold, rendered foreground endpoint-depth and line-profile depth-jump
thresholds, and shared-view support requirement. Small retained components lose
their edges but their observed nodes remain explicitly isolated.

The compact graph stores only Gaussian indices, view support, retained edge
weights and diagnostics, and connected-component labels. Static positions and
colors remain owned by the static scene. The artifact is published as
`candidate_unapproved`; graph construction never approves it automatically.
The rigid solver may consume this candidate, but its result remains unapproved
too. Component extent, cross-layer edges, holes, isolated nodes, and solved
motion are approved together by the later unified visualization stage.

## Alpha synchronization and rigid modal displacement

Solve every selected frequency using the static foreground, topology-aligned
measurements, and observed graph:

```powershell
modal-gaussians rigid solve `
  --scene C:\outputs\static_scene `
  --topology C:\outputs\observation_topology `
  --measurements C:\outputs\gaussian_measurements `
  --graph C:\outputs\observed_structure_graph `
  --work-dir C:\outputs\rigid_work `
  --output C:\outputs\rigid_modes
```

For one mode, topology contributor rows express the observation model
`y_view = alpha_view * J_view * phi`. The reference view fixes `alpha=1`.
Shared Gaussians first synchronize every other identifiable view with one
bounded complex alpha: its angle is the temporal phase offset and its magnitude
is the relative response gain. The accepted gain interval is `[0.25,4]`, at
least 16 shared Gaussians are required, and numerically unidentifiable views are
excluded instead of silently assigned an offset.

After synchronization, every connected graph component is represented by one
complex infinitesimal SE(3) twist. Its complex translation and rotation produce
one `[G,3] complex64` displacement field `phi`; this guarantees zero edge-length
change to first order. Because playback is additive rather than a finite SE(3)
transform, the solver also measures finite edge-length drift over 64 phases.
Only components with adequate multi-view support, full-rank singular ratio, and
finite drift are copied into `trusted_phi`.

The work directory keeps one identity-checked `mode_XXX.npz` checkpoint per
greedy mode, so an interrupted K-mode solve resumes without recomputing finished
modes. The final non-overwriting artifact is:

```text
rigid_modes/
├── manifest.json
└── rigid_modes.npz
```

The NPZ contains raw and trusted `phi`, per-view complex alphas, component
twists, rank/residual/support diagnostics, first-order rigidity errors, and
finite-drift statistics. Values outside their accompanying masks are zero and
must not be interpreted as solved motion. No spatial motion fill is applied in
this stage. The artifact status is `solver_candidate_unapproved` until the
later unified visualization is reviewed.

## Motion-basis blend

The current experimental route treats the rigid solve as a **basis extractor**,
not as the final piecewise-rigid field. Fit one soft blend over the complete
foreground after `rigid solve`:

```powershell
modal-gaussians motion fit-bases `
  --scene C:\outputs\static_scene `
  --topology C:\outputs\observation_topology `
  --measurements C:\outputs\gaussian_measurements `
  --graph C:\outputs\observed_structure_graph `
  --rigid C:\outputs\rigid_modes `
  --work-dir C:\outputs\motion_basis_work `
  --rigid-basis-count 6 `
  --local-rigid-basis-count 4 `
  --output C:\outputs\basis_completed_modes
```

Only rigid components retained at every selected frequency become nonzero
motion bases. A zero-motion basis is appended implicitly so local motion may
attenuate without negative cancellation. For selected component `c`, its
complex first-order SE(3) field is evaluated at every foreground Gaussian:

```text
B[k,g,c] = translation[k,c]
           + cross(rotation[k,c], means[g] - centroid[c])
phi[k,g] = sum_c weight[g,c] * B[k,g,c]
```

By default (`--weight-sharing shared`), every Gaussian owns one real,
non-negative simplex weight vector shared across all modes. Consequently, even
Gaussians that originally belonged to a rigid
component are no longer hard-assigned to that component's twist. The joint
pixel-composited FISTA solve uses the synchronized multi-view complex
measurements for directly supported Gaussians and a full-foreground geometric
graph penalty to extend the same weight field through unobserved regions. An
unreachable graph component is assigned the zero basis explicitly rather than
being reported as a solved moving region.

The first experiment uses six global rigid bases, at most four nearby rigid
bases per Gaussian, an 8-neighbor full-foreground graph, maximum edge distance
`0.008`, distance temperature `0.02`, smoothness weight `0.01`, and prior
weight `0.001`. Device selection defaults to `auto`; `--resume` reuses the work
checkpoint only when all source identities and resolved settings match. These
are solver inputs recorded in the artifact identity; changing them requires a
new output and a matching work directory.

The output retains the downstream `modal_gaussians.completed_modes` contract
but uses artifact version 3. The manifest dispatch key is
`completion_method: shared_motion_basis_blend`; the more specific solver
description is `semantics.method: joint_pixel_composited_motion_basis_fista`.
`completed_modes.npz` contains the final `[K,G_fg,3] complex64` `phi`, shared
`[G_fg,B] float32` `weights`, selected component twists, the spatial prior,
candidate-basis mask, full-foreground graph, and fit diagnostics. Its mutually
exclusive, exhaustive
`measurement_supported_mask`, `graph_propagated_mask`, and
`zero_fallback_mask` arrays describe how every foreground Gaussian obtained its
weights. They deliberately do not impersonate the anchor/fill masks of the
older sequential algorithm.

### Per-frequency weights

Use `--weight-sharing per-frequency` with separate work and output directories
to fit an independent simplex weight vector for every frequency and Gaussian:

```powershell
modal-gaussians motion fit-bases `
  --scene C:\outputs\static_scene `
  --topology C:\outputs\observation_topology `
  --measurements C:\outputs\gaussian_measurements `
  --graph C:\outputs\observed_structure_graph `
  --rigid C:\outputs\rigid_modes `
  --weight-sharing per-frequency `
  --work-dir C:\outputs\frequency_motion_basis_work `
  --rigid-basis-count 6 `
  --local-rigid-basis-count 4 `
  --output C:\outputs\frequency_basis_completed_modes
```

The field becomes `phi[k,g] = sum_c weight[k,g,c] * B[k,g,c]`. Basis
selection, nearest-component candidates, zero basis, non-negative simplex
constraints, spatial graph, and pixel-composited observation model stay the
same. The change allows each frequency to use a different spatial blend.
Smoothness and prior penalties are averaged over modes to preserve their scale
relative to the shared-weight objective; the per-frequency experiment can use
the same `MotionBasisConfig` settings. Work checkpoints and artifact identities
remain specific to the method and settings, so a shared-weight checkpoint
cannot resume a per-frequency run.

This alternative uses completed-modes version 4 and
`completion_method: per_frequency_motion_basis_blend`. Its `weights` and
`spatial_prior_weights` arrays have shape `[K,G_fg,B]`; weight entropy, dominant
basis, and the three support masks have shape `[K,G_fg]`. Candidate bases remain a shared
`[G_fg,B]` mask, and `phi` keeps shape `[K,G_fg,3]`. The Viewer displays support
roles for the selected mode. Version 3 shared-weight and version 1/2 sequential
artifacts retain their existing contracts and loading behavior.

The per-frequency builder and strict loader are in
`src/modal_gaussians/motion/rigid/motion_basis_frequency.py`:
`build_frequency_motion_basis_modes_artifact` and
`load_frequency_motion_basis_modes`. They reuse the geometric and observation
operators in `motion_basis.py`; the existing shared-weight builder remains
unchanged. This is a separate experimental route and does not replace the
first shared-weight experiment or the sequential baseline.

### Trust components independently at each frequency

Use `--basis-selection-policy trusted_per_mode --weight-sharing per-frequency`
to retain each component only at frequencies where its original rigid trust
checks pass. Omit `--rigid-basis-count`: the builder uses the union of these
trusted components automatically, without ranking or truncating that union.
For example, use the same `motion fit-bases` inputs as above with:

```powershell
  --weight-sharing per-frequency `
  --basis-selection-policy trusted_per_mode `
  --local-rigid-basis-count 4 `
  --work-dir C:\outputs\per_mode_trusted_work `
  --output C:\outputs\per_mode_trusted_modes
```

This produces version 6: the active basis mask is `[K,B]`, candidates are
`[K,G_fg,B]`, and each Gaussian uses up to four active nearby rigid bases plus
the zero basis. All input frequencies remain present. Green refinement of a
version 6 parent produces version 7 and preserves these per-frequency masks.
In the Viewer, graph colors follow **Selected frequency (Hz)** in Gaussian
color or the synchronized Spectrum selection: active components keep distinct,
stable colors; inactive components are gray. Blue/green/purple point roles
retain their existing meaning and their separate Modal role mode control.

### Refine green points with blue points fixed

For an experiment that bypasses component trust filtering, add
`--basis-selection-policy all_rigid_components --rigid-basis-count 53` to
`motion fit-bases` when the rigid source contains 53 solved components. The
count is an exact check against that source, not a top-ranked subset. The
default policy remains `trusted_all_modes`. Keep
`--local-rigid-basis-count 4` to use four nearby rigid bases plus the zero basis
per Gaussian. Use fresh work/output directories and rebuild downstream
artifacts; the original rigid trust masks and graph coloring still describe
which components passed the original checks.

To make graph-propagated (green) points follow the motion of nearby
measurement-supported (blue) points, refine a completed version 4 or 6 artifact in
separate work and output directories:

```powershell
modal-gaussians motion refine-green `
  --input C:\outputs\frequency_basis_completed_modes `
  --work-dir C:\outputs\green_refinement_work `
  --output C:\outputs\green_refined_completed_modes `
  --blue-green-multiplier 1 `
  --green-prior-multiplier 1 `
  --max-iterations 3000
```

The refinement fixes all blue weights and zero-fallback weights to their parent
values, and optimizes only green weights independently for each frequency.
It retains the parent's bases, candidates, simplex constraints, support roles,
and spatial graph. By default, blue-green and green-green edges retain the
parent's graph penalty, and the distance-prior penalty on green points retains
the parent's strength. With parent weights of 0.01 and 0.001, these coefficients
are 0.01 for both edge groups and 0.001 for the green distance prior. The original
total edge-weight sum and foreground-point count remain the graph and prior
denominators, so changing these multipliers does not silently rescale other
terms.

The topology observation prediction is unchanged because its contributing blue
weights remain fixed. Full rendered modal images and fitted time coordinates
can still change because the full renderer includes green points. Compare the
spatial fields first with the parent's fixed time coordinates, then rebuild
`coordinates render-design`, `coordinates solve-direct`, and `result materialize`
for the final result.

With a version 4 parent, this route uses completed-modes version 5 with
`completion_method: fixed_observation_green_basis_refinement`. It keeps the
version 4 `[K,G_fg,B]` weight and `[K,G_fg]` support-role shapes; the Viewer keeps
blue, green, and purple role colors. A version 6 parent instead produces
version 7 with the same completion method and retains per-mode eligibility.
Its builder and strict loader are
`build_green_refined_motion_basis_artifact` and
`load_green_refined_motion_basis_modes` in
`src/modal_gaussians/motion/rigid/motion_basis_green.py`. Existing version 1-4 artifacts and
the default `motion fit-bases` behavior are unchanged.

Compare all three spatial-field methods using **direct coordinates**:

```text
rigid solve -> motion fit-bases -> coordinates render-design
            -> coordinates solve-direct -> result materialize
```

Do not apply `coordinates physics-fit` in a controlled motion-basis comparison.
The legacy and basis results must all use direct coordinates so that a temporal
post-fit cannot be mistaken for a spatial-field improvement.

## Neural full-foreground complex displacement fields

`motion fit-neural` constructs completed-modes v8 with
`completion_method: neural_complex_displacement_field`. It exports a fixed
`phi[K,G_fg,3]` complex64 field in the static foreground's Gaussian order.
Playback uses this exported field and the existing modal coordinate convention;
it does not run the neural network inside Viser.

The default `--graph-edge-filter depth` preserves depth/path filtering.
For a spatial-graph ablation, `--graph-edge-filter none` retains every mutual-KNN
candidate within the configured distance, skips depth/alpha path rejection,
and uses edge weights `1/sqrt(degree_i*degree_j)`. These edges are recorded as
unverified spatial priors, not depth-supported connections. The option does not
disable alpha/mask filtering of the modal-image supervision pixels. It requires
a new geometry/control/interpolation build and a new work directory; old v8
artifacts retain their original identities and remain loadable.

```powershell
modal-gaussians motion fit-neural `
  --scene outputs/static_scene `
  --topology outputs/observation_topology `
  --measurements outputs/gaussian_measurements `
  --graph outputs/observed_structure_graph `
  --alignment-from outputs/rigid_modes `
  --work-dir outputs/neural_work `
  --output outputs/neural_completed_modes
```

These paths are placeholders for matching upstream artifacts. `--alignment-from`
binds the rigid artifact's existing cross-view complex alignment; rigid motion
bases and their trust filtering do not define the neural displacement field.
The modal-image objective uses dense complex 2D modes and the full foreground
Gaussian feature renderer, with foreground-alpha normalization. Direct pixel
support therefore follows actual rendered contributions, rather than membership
in the older topology contributor set. Invisible points can still lack direct
image supervision.

Each frequency owns an independent residual graph network. Its input is the
normalized control-point position; three 64-wide SiLU message layers share
information along the fixed control graph. The zero-initialized output head
also sees the original normalized position and produces complex displacement
`d_a` and infinitesimal rotation `omega_a` (12 real channels). No per-Gaussian
learnable residual or learned interpolation weights are added.

The geometry graph covers all foreground Gaussians. RGB is not an input. A
view supplies support, a contradiction when both endpoints are visible, or
unknown evidence. Supported edges require no contradictory view; wholly unknown
edges use the shorter radius and explicit weaker weight. Occlusion is not a
contradiction. Small disconnected components and isolated nodes are retained.
Graph-distance farthest-point sampling covers the foreground within `h=0.03L`,
where `L` is its bounding-box diagonal. Every control inside graph distance
`2h` contributes through a normalized Wendland kernel; there is no top-four
truncation or interpolation across disconnected components.

The exported field is

```text
Phi_i = sum_a N_ia * (d_a + omega_a cross (x_i - c_a))
R_i   = sum_a N_ia * omega_a
```

The deformation penalty measures the final Gaussian edge residual
`Phi_j - Phi_i - 0.5*(R_i+R_j) cross (x_j-x_i)`, divided by edge length.
The second structural term measures differences of control rotations along
their material graph paths. Both terms use fixed dimensionless scaling;
constant infinitesimal rigid motion has zero structural loss. Finite bending,
twisting and stretching remain expressible with a soft penalty. These are
geometric priors, not calibrated material stiffness or evidence of true modes.
The observation term is a radial complex-vector Huber loss with each view's
fixed energy normalization. The total is `L_modal + L_deformation + 0.1*L_rotation`.
Amplitude scales come from measured modal energy and fixed rasterized projection
sensitivity; export restores the original raw DFT units. Those coefficients
must not be interpreted as physical displacement amplitudes without temporal
coordinates and the appropriate Fourier convention.

The defaults are: geometry neighbors 8 and maximum distance 0.008; unknown-region
maximum distance 0.004 and edge weight 0.1; control-radius fraction 0.03 with at
most 2048 controls; hidden width 64 and 3 message layers. Pixel stride is 2,
foreground-alpha minimum 0.05, and mask erosion is 1 iteration. The loss uses
energy-floor fraction 0.05, Huber delta 1, deformation weight 1, rotation weight
0.1, and rotation-length fraction 0.05. Optimization uses learning rate 0.001,
at most 2000 iterations, gradient clipping 1, seed 1729, convergence patience 50,
relative tolerance 1e-6, and checkpoints every 100 iterations. Every setting is
available as a CLI option; see `motion fit-neural --help`. `--resume` requires
matching source identities and configuration. Production feature rendering
requires CUDA; CPU synthetic checks do not provide a production CPU renderer.
Work directories retain the original fixed observations, graph, renderer
contribution masses, support masks and normalization scales. Resume uses these
persisted inputs, because CUDA gradient reductions are not bitwise deterministic.
It still checks source/configuration/runtime identities and the current rendered
alpha within tolerance. Latest model/optimizer/RNG state is saved separately
from the best model used to export `phi`.

For a shorter preview after stopping a run, `motion export-neural-prefix`
accepts the same five source paths and `--work-dir`, plus `--count 5` and a
fresh `--output`. It exports only the first five **complete** checkpoints,
without training or changing their network weights, alpha, geometry or original
normalization. Missing or unfinished checkpoints are rejected. The v8 preview
records explicit source-mode indices and retains the original full-frequency
normalization provenance; downstream coordinates are fitted for the preview's
five modes. Keep the original work directory to resume the larger run later.
Do not compare its five-mode flow fit to a twenty-mode baseline as if they used
the same number of modes.

The output carries its own geometry graph and per-mode supervision roles.
In Viser, blue means directly image-supervised, green means structure-inferred,
and purple means unresolved. Geometry component colors show connectivity;
they are not rigid trust classifications. The Spectrum panel reads the bound
dense 2D modes directly. Its displayed complex alignment is fitted for comparison,
as for earlier methods, and is separate from the alignment used by the objective.

Continue through `coordinates render-design`, `coordinates solve-direct`, and
`result materialize` using the new completed modes. Earlier v1-v7 artifacts retain
their loaders and Viewer conventions. Implementation and synthetic verification
do not change the accepted [Corn baseline](BASELINE.md) or constitute a new plant
experiment.

Development checks: run `python -m unittest discover -s tests -v` in the project
environment. They use synthetic geometry and images, including a tiny actual
CUDA feature-rendering check when CUDA is available; they do not fit the Corn
dataset or start Viser. The first five-frequency Corn preview was removed during
the user-requested output cleanup on 2026-09-05. The retained final non-neural
reference and accepted neural baseline are listed in [BASELINE.md](BASELINE.md);
their shared dependencies are documented in [outputs/CLEANUP.md](outputs/CLEANUP.md).

## Sequential full-foreground motion fill (accepted baseline)

The migrated sequential route remains available as the piecewise-rigid
baseline. Complete its rigid candidate over the full foreground Gaussian set:

```powershell
modal-gaussians motion fill `
  --scene C:\outputs\static_scene `
  --topology C:\outputs\observation_topology `
  --measurements C:\outputs\gaussian_measurements `
  --graph C:\outputs\observed_structure_graph `
  --rigid C:\outputs\rigid_modes `
  --work-dir C:\outputs\motion_fill_work `
  --output C:\outputs\completed_modes
```

When the selected K is an **upper limit**, append `--valid-modes-only` to
`motion fill`. This omits only modes with zero trusted rigid Gaussian seeds;
it does not relax alpha/rigidity thresholds or select replacements to reach K.
Without the flag, every selected mode must have trusted seeds. An empty usable
subset always fails before completion begins.

Completion format v2 records `mode_selection` (policy, original count, retained
source slots and rejected-mode reasons). Its local `mode_slot=0..K_valid-1`
records include `source_mode_slot`, while candidate indices/frequencies and
their relative greedy order stay unchanged. Original dense DFT, measurements,
and rigid artifacts are retained without copying or editing. The downstream
design, coordinates and result carry these mode records; the spectrum panel
uses the source-slot mapping to read the original dense fields. Loaders check
the mapping and exact trusted seed values against the bound rigid parent.
Legacy format v1 full-prefix completion artifacts remain readable. Changed
selection policies/maps invalidate work-directory identity and cannot resume
an incompatible checkpoint.

This migrates only the accepted sequential rigid branch. It first revisits
rigid components that have one supported view but failed the trusted-seed gate.
The reliably observable twist directions are kept; weak or viewing-ray-
dominated directions are filled jointly from neighboring trusted components.
Components whose additive playback exceeds the finite-drift threshold are
rejected. The remaining component fields become additional fixed anchors.

The second stage constructs a distance-pruned union 8-NN graph over every
foreground Gaussian, including those absent from the observation topology. It
solves independent complex 3D Gaussian motion with real/imaginary LSMR systems,
while preserving all trusted and promoted anchors exactly. By default, only
points within eight graph hops of an anchor are filled. Disconnected or hop-
limited points remain zero with an explicit unresolved mask; zero values are
never interpreted as successful completion.

The full-foreground graph is stored inside the completion artifact instead of
requiring another preprocessing directory:

```text
completed_modes/
├── manifest.json
└── completed_modes.npz
```

The NPZ contains `[K,G_fg,3] complex64` completed `phi`, trusted/promoted/
pointwise/unresolved support classes, completion and connectivity masks,
per-point measurement residuals, component-promotion diagnostics, LSMR
diagnostics, and the shared fill graph. One identity-bound checkpoint per mode
is retained in the work directory for resume. The output remains
`completion_candidate_unapproved` until the unified visualization is reviewed.

## Rendered modal design

Project the completed complex 3D modes through the actual foreground 3DGS:

```powershell
modal-gaussians coordinates render-design `
  --scene C:\outputs\static_scene `
  --modes C:\outputs\completed_modes `
  --view view1 C:\outputs\view1_flow `
  --view view2 C:\outputs\view2_flow `
  --output C:\outputs\rendered_design
```

`--modes` accepts either the version-3 shared-basis result or a legacy
sequential-fill baseline. This stage no longer uses the observation topology's
short contributor lists.
For every selected mode and fixed view it evaluates the pinhole projection
Jacobian at every foreground Gaussian, rasterizes all projected features with
the same Gaussian geometry, opacity, transmittance, and depth ordering as the
static renderer, then divides by rendered foreground alpha. Background
Gaussians are excluded and contribute zero features. Legacy completed-mode
entries that remain unresolved are retained as explicit zero displacement
while still participating in foreground alpha; version-3 zero-fallback entries
are likewise explicit zero-basis blends.

Candidate pixels come from the eroded flow mask, foreground-alpha threshold,
and an internal stride grid. The output columns use the fixed real packing

```text
design[p,:,2k]   = Re(J phi_k)
design[p,:,2k+1] = -Im(J phi_k)
```

so multiplying by `[Re(q_k), Im(q_k)]` produces
`Re(q_k * J phi_k)`. A randomized direct-feature render verifies this sign and
packing convention for every view before publication.

```text
rendered_design/
├── manifest.json
├── design.npy
└── samples.npz
```

`design.npy` is mmap-loadable `float32 [P,2,2K]`. `samples.npz` stores the
ordered view shapes, contiguous per-view sample offsets, pixel coordinates,
and foreground alpha. The manifest binds the static scene, foreground,
completed modes, cameras, flow artifacts, mode order, settings, diagnostics,
and file hashes through one path-independent identity. Publication is
non-overwriting and atomic. Temporal coordinate fitting is intentionally not
part of this stage.

## Direct modal coordinates

Fit the per-frame complex modal activations independently for every fixed
view:

```powershell
modal-gaussians coordinates solve-direct `
  --design C:\outputs\rendered_design `
  --view view1 C:\outputs\view1_flow `
  --view view2 C:\outputs\view2_flow `
  --output C:\outputs\direct_coordinates
```

Each view may have its own frame count and FPS. Its rendered design is flattened
to `[2P,2K]`; the two columns of each mode are scaled together by their RMS.
The solver forms one normalized Gram matrix, adds the default relative ridge
`1e-4`, computes one Cholesky factorization, and reuses it for all frame chunks
of that view. It does not impose a sinusoidal time equation at the selected
frequencies—the frequencies remain mode labels during this direct fit.

Flow is always interpreted relative to the flow artifact's reference frame.
After solving, the reference-frame coordinate is subtracted and a per-view
temporal mean-zero gauge is applied. Consequently reconstructed flow uses
`q(t) - q(reference)` even though the stored `q(t)` has zero temporal mean.

```text
direct_coordinates/
├── manifest.json
├── coordinates.npy
└── diagnostics.npz
```

`coordinates.npy` stores the concatenated per-view
`complex64 [sum(T_view),K]` coordinates in declared view/local-frame order.
The manifest retains every view's frame names, FPS, reference frame, frame
offset, flow identity, solver settings, and overall/per-view flow metrics.
`diagnostics.npz` stores frame indices/times, mode-pair scales, rank and
condition data, per-frame RMSE/relative residual/R2, coordinate magnitudes,
frequency-energy summaries, and cross-mode correlations. The artifact is
identity-bound, mmap-loadable, non-overwriting, and atomically published.

## Physics coordinate post-fit

Apply the accepted damped-oscillator temporal prior to the direct coordinates:

```powershell
modal-gaussians coordinates physics-fit `
  --input C:\outputs\direct_coordinates `
  --output C:\outputs\physics_coordinates `
  --damping-ratio 0.05 `
  --forcing-weight 0.1
```

For mode frequency `f`, the solver uses
`D2 + 2*zeta*(2*pi*f)*D1 + (2*pi*f)^2*I` on the actual per-view timestamps.
The same sparse linear system is applied independently to the real and
imaginary coordinate channels, with an explicit temporal mean-zero constraint.
The accepted mainline leaves adjacent-force regularization disabled; it can be
enabled with `--forcing-difference-weight`.

This stage changes only `q(t)`. Static Gaussians, completed 3D modes, cameras,
and the rendered design remain fixed. Input and fitted coordinates are both
re-evaluated against the exact bound flow artifacts through the same rendered
design used by the direct solve. The implementation rejects publication if its
recomputed input metrics disagree with the direct artifact.

```text
physics_coordinates/
├── manifest.json
├── coordinates.npy
└── diagnostics.npz
```

`coordinates.npy` is mmap-loadable `complex64 [sum(T_view),K]` in the unchanged
view/frame/mode order. Diagnostics retain before/after flow metrics, coordinate
retention, first/second derivatives, normalized latent-force energy,
assigned-frequency energy, cross-mode correlation, and sparse-system condition
estimates. The artifact is published as an unapproved candidate for the later
unified visualization.

## Result materialization

Bind the fixed static scene, completed 3D modal fields, and the chosen temporal
coordinates into one Viewer-ready result:

```powershell
modal-gaussians result materialize `
  --scene C:\outputs\static_scene `
  --modes C:\outputs\completed_modes `
  --coordinates C:\outputs\physics_coordinates `
  --output C:\outputs\modal_result
```

`--coordinates` accepts either the direct or physics-coordinate artifact and
records which one was selected. The accepted pipeline above uses physics
coordinates; the first shared-basis A/B experiment instead materializes direct
coordinates for both candidates, as documented in its separate section.
Materialization performs no fitting, rendering, or tensor conversion. It
reloads the entire identity chain and requires the same static foreground
ordering, completed-mode ordering, rendered design,
view labels, flow identities, frame names, reference frames, FPS values, and
coordinate shape.

```text
modal_result/
└── manifest.json
```

The result is deliberately a linked immutable artifact: the large static
Gaussian tensors, completed `phi`, and coordinates remain in their existing
directories and are not copied. The manifest stores their absolute paths and
path-independent identities. Loading the result revalidates every source, so a
changed, missing, or mismatched source fails before visualization. Move the
source artifact directories only by rematerializing the result at their new
paths.

For frame `t` in view `v`, the runtime deformation is
`sum_k Re(q[v,t,k] * phi[k])` on foreground Gaussian means; background
Gaussians remain static. Stored coordinates use the temporal mean-zero gauge.
Reference-to-frame flow comparisons use `q(t) - q(reference)`. The materialized
result remains `modal_result_candidate_unapproved` until the unified
visualization is reviewed.

## Unified Viser inspection

Launch the migrated result Viewer from the single materialized-result entry
point:

```powershell
modal-gaussians viewer `
  --result C:\outputs\modal_result `
  --work-dir C:\outputs\modal_viewer_work `
  --port 8080
```

The Viewer reloads and verifies the complete linked identity chain before it
opens. Rendering requires CUDA because deformed foreground Gaussians and the
static background are concatenated into one `gsplat` rasterization, preserving
their shared depth order and occlusion.

The accepted old modal-Viewer functions are available together:

- fixed-view playback with view selection, frame stepping, play/pause, FPS,
  and canonical-pose display;
- stored direct/physics coordinates or a manual oscillator with motion scale,
  per-mode enable, gain, phase, solo, disable-all, and enable-all controls;
- RGB, calibrated projected modal phase, and unique-topology-view observation-
  count Gaussian coloring;
- artifact-aware display-role point clouds. Version-3 through version-7 basis results
  show Measurement-supported, Graph-propagated, and Zero-fallback in blue,
  green, and purple. These are the exact basis-fit support masks and are not
  presented as legacy anchors or fills. Version-1/2 sequential results retain
  their existing Anchor, Filled, and Unobserved interpretation;
- calibrated camera frustums, camera jumps, orbit-center reset, foreground-only
  rendering, and complete-render hiding;
- component graph edges with distinct, stable colors for trusted components
  and gray for the others. Version-6/7 `trusted_per_mode` results use the current
  Selected frequency; older results retain the all-source-frequency trust
  intersection. These graph colors are separate from blue/green/purple point roles;
- synchronized original dense-rFFT and reconstructed selected-mode spectra,
  U/V modal images, view/mode selection, and per-mode or entire-spectrum
  amplitude normalization;
- editable camera paths with keyframes, spline/loop controls, preview playback,
  timing/FPS settings, JSON load/save, and up-direction reset. Camera-path JSON
  is written below `viewer_work/camera_paths/`.

Viser is pinned to 1.1.0. `Spectrum` uses its standalone floating-panel API,
matching the old Viewer: initial position (16, 16), size 720 x 800 CSS pixels,
with dragging, resizing, and docking. `Render` remains in the main control
panel. This layout does not change scientific artifacts. Restart the
Viewer after upgrading and hard-refresh the browser to load the matching client.
The main controls follow the final old `run_rendering.py` Viewer order:
Rendering → Time → Cameras → Gaussian color → Modal playback → Debug points
→ Render. Debug points uses the frequency-labelled `Modal role mode` dropdown
(only when there is more than one mode) and the loaded artifact's role legend:
three basis roles for versions 3-7, or the accepted legacy roles for
version 1/2. `Hide background` appears only when background Gaussians exist.
The dark theme, yellow accent, medium control width, and default Viewer Res
of 2048 also match the old Viewer. Explicit `--viewer-res` settings still take
precedence, so an existing run configured with 1024 keeps that resolution.
The obsolete Shape-of-Motion dynamic
track overlay is intentionally absent: the standalone modal result contains no
dynamic trajectories or track state.

## Migrated pipeline boundary

The migrated core currently reaches an unapproved materialized modal result:

1. discover zero-padded image names in lexicographic order and validate matching
   masks, FPS, and the reference frame;
2. optionally stabilize every frame to the reference background with
   Shi-Tomasi features, forward/backward LK tracking, and a RANSAC homography;
3. build the union foreground analysis mask;
4. compute fixed-parameter Farneback flow from the reference to every frame;
5. optionally apply Davis-inspired contrast-weighted Gaussian smoothing;
6. subtract each pixel's temporal mean, apply a symmetric Hann window, and
   compute its temporal real FFT;
7. train and accept a static foreground/background 3DGS;
8. map fixed-view flow pixels to foreground Gaussian contributors and projection
   Jacobians;
9. greedily select the requested first K shared exact-DFT frequencies by
   equal-view macro R2;
10. export the selected frequencies as dense complex `(u,v)` modal fields for
    every fixed view;
11. sample those fields at topology pixels to form the compact Gaussian
    measurement bank without duplicating topology data;
12. build a color/depth-filtered observed Gaussian graph candidate;
13. synchronize each view's bounded complex phase/gain offset from shared
    Gaussian observations;
14. solve one complex first-order rigid-body twist per connected component and
    retain only components passing the multi-view, rank, and finite-drift gates;
15. promote finite-safe single-view component solutions, then propagate fixed
    rigid motion through the hop-limited full-foreground union-KNN graph;
16. rasterize the completed full-foreground complex 3D modes into an
    alpha-normalized, flow-pixel modal design with `Re/-Im` column packing.
17. fit independent per-view complex modal coordinates with mode-pair
    normalization, reference subtraction, ridge, and a temporal mean-zero gauge.
18. post-fit each per-view/mode coordinate with the accepted damped-oscillator
    operator and latent-force penalty, then re-evaluate input/output flow through
    the unchanged rendered design;
19. materialize one lightweight, identity-bound result manifest that links the
    static scene, completed 3D modes, and the selected direct or physics
    coordinates without duplicating their large arrays.
20. inspect and approve that bound result in the unified Viser playback,
    spectrum, support-role, camera, and camera-path interface.

Every stage writes a separate non-overwriting artifact. Frequency selection
uses the topology pixels and raw flow arrays directly; the cached rFFT remains
the earlier dense per-pixel spectrum product and does not constrain the exact
candidate grid.
The shared motion-basis experiment branches after step 14: it replaces step 15
with a global per-Gaussian basis fit and deliberately stops after direct
coordinates for its controlled A/B comparison. It does not replace the
accepted sequential-plus-physics pipeline above. Graph, rigid, completion,
rendered-design, and direct-coordinate candidates are intentionally not
auto-approved; their combined inspection is performed by the unified Viser
stage. Physics-coordinate candidates inherit the same unapproved state, and
result materialization preserves it until the user reviews the Viewer output.
# Fragment motion propagation (v9)

`modal-gaussians motion propagate-fragments --parent <v8 completed_modes> --output <new completed_modes>`
derives a new result without retraining the neural field. It selects complete
components with at most 16 nodes and bounding-box diagonal at most 0.016, then
searches for a nearby nonfragment host with at least four times as many nodes.
Host anchors come from its 3-core, within one graph-distance patch of radius
0.008; each fragment point must be within 0.008 of a supported anchor. A competing
anchor within 1.25 times the nearest distance, but outside the seed's 0.016 graph
neighborhood, marks the attachment ambiguous and leaves the original motion.

All anchors in the local patch and within 0.008 of the fragment centroid use a
normalized Wendland kernel. At least three supported anchors are required per
mode. Shared fragment weights transfer complex displacement and the existing
local infinitesimal rotation; hosts remain bitwise unchanged. All thresholds are
CLI parameters. These are geometric heuristics, not identified material properties.

The v9 loader validates its immutable v8 parent, replays geometry selection and
motion transfer, and checks copied networks, configuration, sources and arrays.
Attachments do not alter the parent geometry graph. Viewer debug roles mark
propagated fragments yellow and retain original image-contribution counts
separately. Rebuild rendered design, direct coordinates and the final result from
the new v9 modes before playback. No automatic baseline replacement occurs.
