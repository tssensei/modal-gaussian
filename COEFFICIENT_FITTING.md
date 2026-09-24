# Fixed-mode coefficient fitting

Spatial mode learning and temporal fitting are separate stages. Input modes are
immutable complex displacement and angular fields. The static scene, appearance,
cameras, Gaussian order and both fields remain frozen throughout RGB fitting.
Videos were recorded separately; fit independent coefficients for each video.

```text
completed batch index
  -> fixed mode bank + rendered linear projection design
  -> SEA-RAFT ridge initialization
  -> shared RGB pose offset + per-frame complex coefficients
  -> result binding -> explicit video export
```

`coordinates.npy` is `complex64 [total_selected_frames, mode_count]`.
Displacement is `Re(sum_k(q_k(t) * phi_k))`. Saved angular fields drive the
exponential-map rotation of static Gaussian orientations.

## Commands

Replace uppercase paths with current, contract-compatible outputs. All new paths
must be inside the scene's new experiment. Catalogs may still point to artifacts
listed in [REBUILD.md](REBUILD.md).

```sh
modal-gaussians coordinates prepare --scene bush --index BATCH/index.json --status complete --expected-modes 20 --output FIT
modal-gaussians coordinates fit-rgb --scene STATIC --modes FIT/mode_bank --input FIT/direct_coordinates --view view1 --config configs/rgb_coordinates.json --output FIT/rgb_coordinates_view1
modal-gaussians result materialize --scene STATIC --modes FIT/mode_bank --coordinates FIT/rgb_coordinates_view1 --output FIT/result_view1
modal-gaussians result export-video --result FIT/result_view1 --view view1 --output FIT/exports/view1_001
```

`prepare` produces `preparation.json`, `mode_bank/`, `rendered_design/` and
`direct_coordinates/`. Use `--resume` only for an identical preparation contract.
It never starts RGB fitting. `--status` and `--expected-modes` are explicit, so an
old experiment count/status cannot be selected accidentally.

`fit-rgb --view LABEL` reads and fits only that video while reusing shared mode
inputs. Omit `--view` to fit all recordings independently. Initialization subtracts
the reference coefficient, estimates a shared RGB pose offset, then optimizes
per-frame coefficients over multiple resolutions. Loss is
`0.8 * L1 + 0.2 * (1 - SSIM)` plus a weak, decaying initialization anchor.
There is no optimizer-resume checkpoint for RGB fitting.

Targets are the actual inference-grid PNGs, which may be stabilized. Export keeps
input FPS and each panel's resolution: target above, reconstruction below. Output
contains `comparison.mp4`, `manifest.json`, and `encode.log`.

## Limits

- No temporal smoothing, oscillator constraint or hard frequency locking.
- Labeled high-frequency spatial modes may help fit lower-frequency movement.
- GPU fitting currently uses per-frame renders with gradient accumulation.
  Image loading/caching and synchronization remain potential optimization work.
- Training losses and compressed comparison videos are not formal reconstruction
  metrics. RMSE/PSNR/SSIM/LPIPS reports require a separate requested evaluation.
- Fitting does not start exports, metrics or Viser automatically.
