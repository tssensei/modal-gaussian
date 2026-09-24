# Current numerical recipe

Configuration: [`configs/neural_component_field.json`](configs/neural_component_field.json).
Pipeline/entry points: [README](README.md). Cleanup compatibility: [REBUILD](REBUILD.md).

## Spatial modes

- Static geometry, appearance and cameras remain fixed during mode learning.
- Fixed views are independent recordings. Cross-view complex gains align the
  spatial modal observations; they do not synchronize video frames.
- Use explicit render-matched motion-reference selections on the fixed camera grid.
  Geometry reference and motion reference may be different frames.
- SEA-RAFT computes full-frame reference-to-frame flow in input pixels.
- Shared FFT uses temporal-mean detrending, a symmetric Hann window and a common
  zero-padded RFFT grid. Export exact cached bins; do not recompute selected DFTs.
- Candidate geometry is mutual KNN: K=16, maximum normalized-world distance 0.08.
  Each frequency has its own modal-similarity weights, with minimum factor 0.05.
  Shared control geometry can be reused; one frequency's soft weights cannot.
- GPU alpha fitting and GPU soft propagation use CuPy. There is no CPU fallback.
  The resident preparation worker exits before GNN training starts.
- Control radius fraction 0.015; maximum 32,768 controls. GNN hidden dimension 256,
  local feature dimension 32, three message layers, maximum 5,000 updates.
- Per-view RMS-normalized modal-image loss, deformation weight 0.03,
  control-rotation regularization weight 0. Angular displacement is still learned
  and saved; zero regularization weight does not disable rotation.
- The default modal projector is dynamic. `modal_projection_backend="cached"`
  is an optional execution optimization for fixed geometry, with the same math.

Component fields require >=10 Gaussians and >=2 controls, plus effective
observations at the selected frequency. A whole eligible component keeps its own
field, including weak or occluded members. Propagation donors additionally require
>=101 component points, >=2 controls and >=3 reliable observed points. Recipients
without eligible donors remain zero/unresolved. See [component fields](docs/component-field.md).

Automatic mask partitions use foreground/mask sampling. Applied manual selections
use full-scene visibility and subject contribution, without inherited mask erosion.
The prepared artifact records the sampling policy and per-view relative depth
visibility tolerance explicitly. Tolerances are calibration inputs, not a hidden
application of an old graph's thresholds.

## Temporal reconstruction

`x(t) = x_static + Re(sum_k(q_k(t) * phi_k))`.
Angular fields use the same coefficients and an exponential-map rotation.
RGB fitting freezes static Gaussians, appearance, cameras and both spatial fields.
Only complex coefficients change. Each video has its own coefficients.

There is currently no temporal smoothing, oscillator model or frequency locking.
A frequency label identifies the learned spatial mode, not a constraint on fitted
coefficient time series. Training losses do not substitute for reconstruction metrics.

## Resume boundaries

| Situation | Action |
| --- | --- |
| Interrupted batch, identical code/input/config | Rerun the same command/output. |
| Increase only the training iteration cap | New output with `--continue-from`; compatible optimizer/RNG state is retained. |
| New batch attempt inheriting complete modes | New output with `--resume-from`; only matching completed results are inherited. |
| Interrupted coefficient preparation | Same command with `--resume`; exact published stages are reused. |
| Interrupted RGB optimization | New output directory; no optimizer checkpoint exists. |
| Changed identity, schema, loss, reference or code contract | Rebuild affected dependencies into new outputs. |

These mechanisms apply to the current schema. They do not upgrade pre-cleanup artifacts.
