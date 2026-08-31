# Modal Gaussians

Standalone reconstruction tools for asynchronous multi-view modal Gaussian
analysis. The first migrated vertical slice validates ordered image/mask
sequences, optionally stabilizes them to a reference frame, computes dense
reference-to-frame Farneback flow, and evaluates the per-pixel temporal FFT.

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
not use this compatibility path.

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
foreground points, at most 100,000 initial background points, and random seed
42. Both sampled sweep frames and registered reference frames participate in
the RGB loss. Their COLMAP K and poses remain fixed. Masks affect only the
initial foreground/background point classification; there is no mask, alpha,
or depth supervision.

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

## Sequential full-foreground motion fill

Complete the rigid candidate over the full foreground Gaussian set:

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

This stage no longer uses the observation topology's short contributor lists.
For every selected mode and fixed view it evaluates the pinhole projection
Jacobian at every foreground Gaussian, rasterizes all projected features with
the same Gaussian geometry, opacity, transmittance, and depth ordering as the
static renderer, then divides by rendered foreground alpha. Background
Gaussians are excluded and contribute zero features. Completed-mode entries
that remain unresolved are retained as explicit zero displacement while still
participating in foreground alpha.

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
records which one was selected. Materialization performs no fitting, rendering,
or tensor conversion. It reloads the entire identity chain and requires the
same static foreground ordering, completed-mode ordering, rendered design,
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
- support-role point clouds for trusted rigid, promoted single-view rigid,
  pointwise KNN fill, and unresolved foreground Gaussians, including mode,
  role, count, and point-size filters;
- calibrated camera frustums, camera jumps, orbit-center reset, foreground-only
  rendering, and complete-render hiding;
- synchronized original dense-rFFT and reconstructed selected-mode spectra,
  U/V modal images, view/mode selection, and per-mode or entire-spectrum
  amplitude normalization;
- editable camera paths with keyframes, spline/loop controls, preview playback,
  timing/FPS settings, JSON load/save, and up-direction reset. Camera-path JSON
  is written below `viewer_work/camera_paths/`.

Viser 1.0.30 does not expose the custom floating-panel extension used by the
old experiment environment, so the same spectrum controls are hosted in the
standard `Spectrum` tab. This changes only panel placement, not the data,
controls, or synchronization behavior. The obsolete Shape-of-Motion dynamic
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
    rigid motion through the hop-limited full-foreground union-KNN graph.
16. rasterize the completed full-foreground complex 3D modes into an
    alpha-normalized, flow-pixel modal design with `Re/-Im` column packing.
17. fit independent per-view complex modal coordinates with mode-pair
    normalization, reference subtraction, ridge, and a temporal mean-zero gauge.
18. post-fit each per-view/mode coordinate with the accepted damped-oscillator
    operator and latent-force penalty, then re-evaluate input/output flow through
    the unchanged rendered design.
19. materialize one lightweight, identity-bound result manifest that links the
    static scene, completed 3D modes, and the selected direct or physics
    coordinates without duplicating their large arrays.
20. inspect and approve that bound result in the unified Viser playback,
    spectrum, support-role, camera, and camera-path interface.

Every stage writes a separate non-overwriting artifact. Frequency selection
uses the topology pixels and raw flow arrays directly; the cached rFFT remains
the earlier dense per-pixel spectrum product and does not constrain the exact
candidate grid.
Graph, rigid, completion, rendered-design, and direct-coordinate candidates are
intentionally not auto-approved; their combined inspection is performed by the
unified Viser stage. Physics-coordinate candidates inherit the same unapproved
state, and result materialization preserves it until the user reviews the
Viewer output.
