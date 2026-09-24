# Cleanup and required rebuilds — 2026-09-23

This cleanup deliberately breaks old module imports, commands and artifact schemas.
It preserves the current component-field numerical recipe and fixed-mode RGB fitting.
No scene-library artifact, raw video, model, cache or catalog was changed. No real
experiment was run to validate the cleaned pipeline. Synthetic checks are development checks.

## What was removed

- Farneback/legacy flow analysis, selected-frequency repeated DFT and greedy selection.
- Rigid mode completion; surface, pointwise, guarded and refinement training branches.
- CPU alpha fitting and CPU soft-propagation alternatives.
- Physics/oscillator coordinate fitting and automatic post-training coordinate stages.
- Old model readers, network replay just to recover runtime rotation, preview artifacts
  and old viewer aliases. Models now save displacement, angular fields and control displacement.
- Old migration/benchmark scripts, historical recipes and tests dedicated to removed branches.

Modules moved into `preprocessing`, `geometry`, `flow`, `spectrum`, `motion`,
`coordinates`, `results`, `vis` and `common`; no import shims were added.
Git history contains the removed code. Restoring it would require reverting or
cherry-picking the relevant changes with their dependencies.

## Artifact boundary changes

| Artifact | Current contract | Treatment of existing data |
| --- | --- | --- |
| Raw RGB/masks; existing stabilized PNGs | pixel files unchanged | Reuse; do not re-extract/re-stabilize solely for cleanup. |
| Sequence reference | `sequence_reference` v1 | New lightweight metadata/mask union; replaces `flow_analysis` reference artifacts. |
| Static scene | v2/v3 with SIMPLE_RADIAL projection | Retain supported scenes; pinhole-only static bundles require retraining. |
| Motion reference selection | v1 bound to new reference identity | Select again for the current static scene/grid. |
| SEA-RAFT flow | v2 with explicit selection binding | Recompute into new directories. |
| Shared FFT and selected modal images | current transform + new flow/reference identities | Rebuild after flow; export exact bins. |
| Geometry/prepared observations | `neural_prepared` v2; KNN cache v2 | Rebuild directly from static scene and references. No old observed graph is required. |
| Per-frequency soft graph | `modal_similarity_graph` v2 | Rebuild for the matching prepared frequency. |
| Controls/alpha/work checkpoints | current input/code contracts | Rebuild; pre-cleanup checkpoints cannot resume into this implementation. |
| Single-frequency completed mode | v18, baked angular/control fields | Retrain; readers reject the old schemas. |
| Mode bank, design, direct/RGB coordinates, bound result/video | existing formats with new source identities | Rebuild downstream of the new modes. |

To reuse already stabilized pixels, point `prepare reference --images ... --masks ...`
at those PNG directories, specify the original FPS/reference frame, and omit
`--stabilize`. This creates a new reference without decoding an old flow format.
It does not reuse old downstream flow/FFT identities.

## Bush and Corn

Read-only manifest inspection confirmed the following catalog targets. Some newer
experiment subdirectories denied reads; their complete contents/quality were not audited.

| Scene | Static target | Required work |
| --- | --- | --- |
| Bush | `bush/geometry/static`: v3, full-scene visible-mask partition, distortion applied | Static scene can be reused. New references -> selected motion references -> flow -> FFT -> geometry/observations -> 20 modes -> coefficient fit/result/export. |
| Corn | `corn/geometry/static`: v3 manual selection, distortion **not** applied | Retrain a distortion-aware static scene, redo its subject selection, then rebuild downstream references/flow/FFT/geometry/modes. Existing COLMAP calibration/raw inputs can be reused if their input contract still matches. |

Existing Bush targets include `uniform20_0to5_newref_20260920/batch` and
`coefficient20_0to5_newref_20260920/`. Corn's target is
`uniform10_0to2p5_newref_20260921/batch`. These remain on disk as historical results;
the cleaned code must not treat their existence as reusable current output.
Choose fresh experiment names and update catalog pointers only after the new runs finish.

Preserve the intended FFT grids unless changing the experiment: Bush 30 FPS,
NFFT=1250, delta-f=0.024 Hz; Corn 20 FPS, NFFT=1600, delta-f=0.0125 Hz.
The historical Bush selection included 0.744 Hz. Inspect the saved bin list if
exact reproduction is required; a new uniform selection need not reproduce that list.

`motion prepare-neural --view LABEL REFERENCE RELATIVE_DEPTH_TOLERANCE` now makes
visibility calibration explicit. Readable ancestor graph manifests contained:

| Scene | view1 | view2 | view3 |
| --- | --- | --- | --- |
| Bush | 0.01824313479256816 | 0.022121385373454542 | 0.014344444496370852 |
| Corn | 0.010864540798950474 | 0.00919846558852587 | — |

Sources: `bush/geometry/ancestors/bush_neural_dense_controls_001/observed_graph/manifest.json`
and `corn/geometry/ancestors/corn_static_20fps_001/observed_graph/manifest.json`,
`thresholds.views[].endpoint_gap.threshold`. They are recorded calibration values,
not new measurements. Reassess them when geometry/projection changes, especially Corn.
The runtime does not read these old graphs.

The old Bush 40-mode experiment was previously deleted. Its favorable user review
is not an acceptance result for the 20-mode data or this cleanup.
