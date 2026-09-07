# v11 implementation validation

Validated on 2026-09-07. No bush optimization, modal-coordinate fitting, PNG
generation, Viewer server, new plant experiment directory, or baseline change
was performed.

## Development checks

91 tests passed across surface attachments, legacy training fill, neural
artifacts/interfaces, iteration caches, CUDA rendering/pipeline recovery,
neural field mathematics, geometry and module layout.

Coverage includes near isolated Gaussian support, far-point rejection even with
large covariance, anisotropic support direction, boundary/single-anchor
fallback, host ambiguity/unknown visibility, depth tie resolution, per-point
weight variation and smoothing, preserved legacy patches, control budgets,
complex rigid reproduction, finite-difference fragment gradients, source
covariance mismatch, v11 publication/reconstruction, cache reuse/invalidation,
and Viewer role synchronization with unsorted source frequencies.

The actual CUDA check uses 4 foreground Gaussians at 32x32 pixels and two
synthetic training steps. It verifies interruption/resume with byte-identical
fixed inputs and reconstruction of the exported field from saved weights.

## Existing bush geometry preflight

Input: `outputs/bush_neural_dense_controls_001`, existing v10 completed identity
`fb282fad24d8ce2caa8fd4a070010f671ea4a7d5478037118e3d1ac47d7ab74a`.
The existing v10 artifact passed the current strict loader. Only its static
geometry, Gaussian sizes, fixed observation support and depth inputs were used
to construct the proposed v11 attachment operator. Its trained motion was not
modified or used as a new v11 result.

| Item | Count |
|---|---:|
| Foreground Gaussians | 231,761 |
| Independent host components | 155 |
| Host Gaussians | 197,220 |
| Host controls | 2,419 |
| Attachment candidates | 34,541 |
| Geometrically attached candidates | 24,088 |
| Preserved legacy component patches | 858 |
| Geometrically unresolved candidates | 10,453 |

| Unresolved reason | Gaussians |
|---|---:|
| Outside `0.03L` search radius | 3,454 |
| Projected support gap exceeds 0.008 | 5,111 |
| Multiple hosts remain ambiguous | 1,779 |
| Outside the selected component host's support | 109 |

With existing fixed alignment/observation masks, propagated support is available
to 20,623 points at 0.357 Hz and 24,088 at 0.744 Hz. There are respectively 24
and zero unsupported host components. These are support counts, **not trained
motion quality or fit metrics**.

The uncached geometry construction took approximately 27.6 seconds; historical
loading plus the preflight took 53.6 seconds. These are one-off development
timings, not an end-to-end training benchmark. Raw diagnostic output remains in
`tmp/bush_surface_attachment_preflight.json`.

The support gap is an approximation, and nearby different branches can still
be ambiguous. Visual coherence, 2D modal fit, and remaining black gaps require a
new training run and inspection before this strategy can become a baseline.
