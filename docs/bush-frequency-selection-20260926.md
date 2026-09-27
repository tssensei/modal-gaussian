# Bush: uniform versus greedy frequency selection — 2026-09-26

Completed **original 2D modal-image capacity diagnostic** on Bush view1.
Greedy 20 frequencies explain more flow than uniform 60, both before and after
subtracting each pixel's temporal mean. This is evidence that frequency selection
limits this 2D fit; it is not a measured improvement to the 3D reconstruction or
the 6000-step refined model. The production recipe remains unchanged.

## Results

All primary results use the same pixel-pair RMS normalization and ridge `1e-4`.
Explained energy is `1 - sum(residual^2) / sum(observed_flow^2)`, pooled over frames,
pixels and x/y components. The centered metric subtracts each pixel's temporal
mean separately from observed and predicted flow before computing that ratio.
RMSE is per component, not vector endpoint error.

| Frequencies | Uniform total | Greedy total | Uniform centered | Greedy centered | Uniform RMSE (px) | Greedy RMSE (px) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 62.87% | **92.41%** | 55.82% | **85.95%** | 1.0117 | 0.4574 |
| 40 | 71.46% | **96.33%** | 69.50% | **93.53%** | 0.8869 | 0.3181 |
| 60 | 82.91% | **97.92%** | 82.25% | **96.12%** | 0.6864 | 0.2393 |

## Experimental settings

- Same view1 frames 0–299 at 30 FPS (0–9.9667 s), 960×540 images, and 75,445 valid
  reference pixels as the uniform experiment. Reference frame: index 853,
  timestamp 28.4333 s. Validity is preserved.
- Original cached complex FFT fields; 208 positive bins in `(0, 5] Hz`, with
  NFFT=1250 and spacing 0.024 Hz (actual candidates 0.024–4.992 Hz). No DC.
- Uniform selection uses `round(j * 208 / N)`, `j=1..N`, with NumPy's rounding.
- Greedy treats each frequency's `[Re, -Im]` columns as a pair. At each step it
  chooses the addition giving the highest unregularized total-flow OLS explained
  energy; ties within `1e-12` choose the lower frequency. Rank once to 60, then
  evaluate the nested first 20/40/60 prefixes, not the lowest N frequencies.
- Reused the numerical greedy core from `src/modal_gaussians/frequency.py` at
  Git commit `c55efa6f6a3143f87286d53d74fcc9f13333ceed`. The historical objective
  averages views equally; this diagnostic has only view1. Current cached FFT
  fields are used, without restoring the historical DFT production path.
- Float64 fitting; independent free complex coefficients for every frame.
  No alpha alignment, 3D projection, frequency locking or temporal smoothing.
  Greedy selection optimizes total OLS energy; the primary comparison then fits
  with the same ridge as uniform. Centering is a diagnostic, not a second fit.

First 20 frequencies in greedy order (Hz):

```text
0.024, 0.048, 0.072, 1.008, 1.512, 1.536, 1.104, 0.096, 0.192, 0.240,
0.120, 0.624, 1.128, 1.080, 1.488, 1.344, 0.144, 0.168, 0.600, 1.152
```

The largest selected frequency among 60 is 2.328 Hz. Uniform 20 starts at
0.240 Hz and uniform 60 at 0.072 Hz, missing some low-frequency fields that
greedy chooses first. No ablation isolated how much gain those fields explain.

## Interpretation and checks

Selection and fitting use the same 300 frames, and the FFT source also includes
those frames. These are **in-sample capacity measurements**, not held-out scores.
The centered improvement includes time-varying flow, beyond a constant pose
offset. Free coefficients need not oscillate at the FFT label's frequency, so
selected fields cannot automatically be interpreted as physical vibration modes.
Transfer to 3D modes, RGB reconstruction and other clips remains untested.

| Greedy count | Real rank | Normalized design condition number | OLS total | OLS centered |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 40 | 329.28 | 92.56% | 86.28% |
| 40 | 80 | 799.48 | 96.56% | 94.08% |
| 60 | 120 | 2529.32 | 98.22% | 96.78% |

All three designs are full rank, but conditioning worsens with added fields.
Keep the ridge results separate from the OLS selection scores.

Passed synthetic complex-packing, ridge, direct-OLS and centering checks;
historical greedy matched exhaustive direct least squares for single-view and
equal-weight two-view cases, including duplicate-column rank and tie handling.
Full-candidate statistics reproduced the old uniform OLS scores, and final SVD
agreed with greedy scores. Published output identity, all 21 output checksums,
14 source checksums and unique nested prefixes were verified.

Recorded runtime: input reads/checks 8.044 s, sufficient statistics 0.529 s,
greedy ranking 4.686 s, 20/40/60 fits 1.601/2.259/3.117 s, total 20.241 s.
Total excludes startup synthetic checks, final checksumming/publication and
subsequent report verification. No 3D training, RGB refinement or video export ran.

## Local artifacts and reproduction

This document is outside Git-ignored directories so the findings can be committed.
The following scene-owned artifacts and scripts remain local and Git-ignored;
they are not supplied by a fresh clone:

- [Detailed Chinese report](../scene_library/bush/experiments/raw2d_greedy_20_40_60_20260926_001_REPORT.md).
- [Greedy experiment](../scene_library/bush/experiments/raw2d_greedy_20_40_60_20260926_001/):
  `summary.json`, `greedy_order.csv` (all 60 ranks and marginal gains),
  `per_frame.csv`, per-count selections/coordinates/diagnostics, sampled fields,
  sufficient statistics, timings, manifest, execution script and numerical snapshots.
- [Uniform experiment](../scene_library/bush/experiments/raw2d_capacity_20_40_60_20260926_001/).
- Source flow diagnostic: `scene_library/bush/experiments/flow_coordinates540_view1_10s_20260925_001/`.
- Source spectrum: `scene_library/bush/experiments/pipeline540_box0744_20260924_001/spectrum/`.

Run from the repository root with the original environment, matching local inputs
and the historical Git commit available. Replace `NEW_RUN` with a new, nonexistent
scene-owned experiment directory; never overwrite published outputs:

```powershell
python scene_library/bush/experiments/raw2d_greedy_20_40_60_20260926_001/run_experiment.py --uniform scene_library/bush/experiments/raw2d_capacity_20_40_60_20260926_001 --output scene_library/bush/experiments/NEW_RUN
```

Provenance identifiers:

```text
Greedy experiment: 1d9e604a9a818ec3a269503145e37165e206bb7f15e9cc10f8a1943f1a94e087
Uniform parent:    a3630f7100aaa132c2784c33efb8ce892f7314f6cb2791a0e83a48b1ad71146f
Spectrum:          6b002e7ffe82ae8da7421596e9bddbe746521233fc9589fd69334456acd22255
Historical source SHA256:
de8c3614f0ef13cc01bb032187c496c9142602582bb7f8e4485be722049f6d55
```

Recording this result changes documentation only. No cache invalidation,
downstream rebuild or catalog update is required.
