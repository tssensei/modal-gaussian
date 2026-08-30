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

## Migrated pipeline boundary

This slice deliberately stops before peak selection and visualization:

1. discover zero-padded image names in lexicographic order and validate matching
   masks, FPS, and the reference frame;
2. optionally stabilize every frame to the reference background with
   Shi-Tomasi features, forward/backward LK tracking, and a RANSAC homography;
3. build the union foreground analysis mask;
4. compute fixed-parameter Farneback flow from the reference to every frame;
5. optionally apply Davis-inspired contrast-weighted Gaussian smoothing;
6. subtract each pixel's temporal mean, apply a symmetric Hann window, and
   compute its temporal real FFT.

The output is one non-overwriting directory containing raw flow, per-frame valid
masks, the union analysis mask, frame times, the RGB reference frame, complex
per-pixel spectra, the frequency axis, a global amplitude summary, and a
structurally validated `manifest.json`. When stabilization is enabled, the
derived PNG sequence, homographies, settings, and diagnostics are stored
alongside the analysis. The manifest records source directories and active
scientific settings but does not compute content identities or file hashes.
