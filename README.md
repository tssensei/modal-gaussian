# Modal Gaussians

Standalone reconstruction tools for asynchronous multi-view modal Gaussian
analysis. The first migrated vertical slice validates ordered image/mask
sequences, optionally stabilizes them to a reference frame, computes dense
reference-to-frame Farneback flow, and evaluates the per-pixel temporal FFT.

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
