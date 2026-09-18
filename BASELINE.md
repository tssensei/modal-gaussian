# Current neural motion baseline

On **2026-09-18**, the user accepted the **Corn 0.225 Hz** result below as the
new baseline, following visual acceptance of the matching **Bush 0.744 Hz**
recipe. Both use SEA-RAFT modal images, soft graph weights with fixed control
sampling, **Gaussian rigidity 0.03**, and **control-rotation loss 0**.
The earlier hard-cut Bush result is retained as a historical comparison.

## Accepted Corn and Bush results

| | Corn | Bush |
| --- | --- | --- |
| Frequency | **0.225 Hz** | **0.744 Hz** |
| Experiment | `outputs/corn_neural_soft_rigidity003_rotation0_0225_001/experiment` | `outputs/bush_neural_soft_rigidity003_rotation0_0744_001/experiment` |
| Manual preview | `outputs/corn_neural_soft_rigidity003_rotation0_0225_001/experiment/preview` | `outputs/bush_neural_soft_rigidity003_rotation0_0744_001/experiment/preview` |
| Prepared supervision | `outputs/corn_subject_selection_001/prepared` | `outputs/bush_neural_modal_similarity_0744_001/prepared` |
| Soft graph | `outputs/corn_subject_selection_001/graph_modal_soft_002` | `outputs/bush_neural_soft_fixed_controls_0744_001/graph` |
| Frozen configuration | [Corn config](outputs/corn_neural_soft_rigidity003_rotation0_0225_001/config.json) | [Bush config](outputs/bush_neural_soft_rigidity003_rotation0_0744_001/config.json) |
| Resolved contract | [Corn iteration](outputs/corn_neural_soft_rigidity003_rotation0_0225_001/experiment/iteration.json) | [Bush iteration](outputs/bush_neural_soft_rigidity003_rotation0_0744_001/experiment/iteration.json) |

Both are v16 `neural_component_field_with_stable_donors` results. Corn retains
the saved manual subject selection in `outputs/corn_subject_selection_001/static_scene`
and its visible-subject supervision; do not substitute the older fine-mask
preparation. Corn uses two views and Bush uses three. Their frozen normalization,
view alignment and reference geometry remain scene-specific.

## Default experiment recipe

Use SEA-RAFT reference-to-frame flow and selected-frequency exact DFT modal
images for both graph construction and training. The Bush sources are
`outputs/bush1_sea_raft_0744_001`, `outputs/bush2_sea_raft_0744_001`, and
`outputs/bush3_sea_raft_0744_001`; Corn uses `outputs/corn1_sea_raft_0225_001`
and `outputs/corn2_sea_raft_0225_001`. When preparing new inputs, compute complex view alignment and target
normalization through `motion prepare-selected-modal`; inherited Farneback flow
metadata supplies reference geometry/timing only. No full spectrum is required.

Start with unfiltered mutual-KNN candidates, **K=16**, maximum radius **0.08**
in scene units. Apply `graph build-modal-similarity --soft-weights` with similarity threshold
**0.20**, conflict threshold **0.30**, amplitude floor fraction **0.02**, and
maximum projected endpoint distance **32 pixels**. Other settings remain:
amplitude percentile 99, patch radius 1, patch relative dispersion maximum 0.30,
and alpha minimum 0.05. **Retain every candidate edge**. Edges supported by at
least one reliable view and without a reliable conflict keep their original
weight; edges the hard filter would reject, including unknown edges, receive
**0.05 times** that weight. New graph builds use soft weights;
historical hard-cut graphs remain readable.
Depth/alpha gate visibility, with no preliminary depth-discontinuity cut.

Pass the saved graph explicitly with `motion iterate-neural --geometry-graph`;
the neural config uses `graph_edge_filter=none` to avoid another filter.
Control sampling, coverage, owners and support within `2h` use the **original
geometric shortest-path distance**, keeping the original control layout.
Propagation costs `edge_length / edge_factor` only attenuate existing Wendland
interpolation weights by geometric distance divided by propagation distance,
followed by per-Gaussian normalization. They do not drive control sampling or
create extra controls. Merely using the new
K/radius defaults without `--geometry-graph` does **not** reproduce this baseline.
Graphs remain frequency-specific; use matching inputs/graphs for other scenes
or frequencies rather than reusing either accepted graph.

Keep width **256**, local features **32**, **3** message layers, control radius
**0.015L**, maximum 32,768 controls, and new-run deformation/rotation weights **0.03/0**.
Keep learning rate 0.001, maximum 2,000 steps, patience 50, relative tolerance
1e-6, seed 1729, and the existing `component_field` donor rules. The reusable
numeric preset is [neural_component_field.json](configs/neural_component_field.json).
Donor propagation is unchanged. Local rotation in the displacement field and
Viewer ellipsoid rotation remain active; only the control-rotation loss is off.
Explicit configurations take priority. Custom partial `--config` files inherit
omitted fields from their prepared snapshot: include `"deformation_weight": 0.03`
and `"rotation_weight": 0` when reusing older preparations, or use the current preset.

This acceptance records the user's visual judgment of **Corn at 0.225 Hz and
Bush at 0.744 Hz**; it does not establish performance on other frequencies or scenes. No
experiment validation or modal-coordinate fitting was run. Keep the accepted
artifacts and their ancestors immutable. This document records the later user
approval; original `preview_candidate_unapproved`/execution-status fields retain
their publication-time values.

Launch from the repository in Anaconda Prompt:

```bat
modal-gaussians viewer --preview "outputs\corn_neural_soft_rigidity003_rotation0_0225_001\experiment\preview" --work-dir "outputs\corn_neural_soft_rigidity003_rotation0_0225_001\work\viewer" --host 127.0.0.1 --port 8108
modal-gaussians viewer --preview "outputs\bush_neural_soft_rigidity003_rotation0_0744_001\experiment\preview" --work-dir "outputs\bush_neural_soft_rigidity003_rotation0_0744_001\work\viewer" --host 127.0.0.1 --port 8107
```

## Historical Bush hard-cut baseline (2026-09-18)

The earlier SEA-RAFT/modal-similarity result was visually accepted before the
soft-weight experiments. Preserve it with its original **0.1/0.1**
deformation/rotation loss weights:

- Experiment: `outputs/bush_neural_modal_similarity_0744_001/experiment`.
- Manual preview: `outputs/bush_neural_modal_similarity_0744_001/experiment/preview`.
- Prepared inputs: `outputs/bush_neural_modal_similarity_0744_001/prepared` (also reused above).
- Hard-cut graph: `outputs/bush_graph_modal_similarity_0744_002`.
- Frozen configuration: [config.json](outputs/bush_neural_modal_similarity_0744_001/config.json).
- Resolved contract: [iteration.json](outputs/bush_neural_modal_similarity_0744_001/experiment/iteration.json).
- Immutable trained modes: `outputs/_cache/trained_modes/0628e6cfefb69e82eacf5973ec0ee345d537f6d98745ae0e985e278af90a087c`.
- Completed-modes identity: `2be1f71b1b35410deca8713e37581a9d91ceef59571d4b9fb98b99eb81fba517`.
- Preview identity: `c5ef66b709c373669f855a5b1dac57f0db4e9688835b1fa31262ddbcbd524170`.
- Frequency 0.744 Hz, selected-bundle slot 0, original candidate index 1; all three views participated.
- 770,364 retained edges, 1,382 controls, 645 training steps; 105,214 of 231,761 Gaussians received donor motion.

This recipe kept only candidates with reliable motion-similarity support and
no reliable conflict, deleting unknown edges as well. Its thresholds were
0.20/0.30 with the same K=16, radius 0.08, 32-pixel gate and 256/32/3 network.
Controls and interpolation followed the pruned graph. Its saved artifacts remain
available for comparison; new graph construction uses the soft-weight recipe above.

## Historical Bush features32 baseline (2026-09-07)

On 2026-09-07, the user selected **only increasing the local feature dimension to
32**, with width **256** and **three** message-passing layers, as the new baseline
after comparing the bush capacity experiments. This is the `features32` result,
not the combined 32-feature/six-layer experiment. The choice is based on the
user's visual assessment, rather than the lowest training loss.

### Historical result and experiment recipe

- Experiment: `outputs/bush_neural_capacity_0744_001/features32`.
- Manual preview: `outputs/bush_neural_capacity_0744_001/features32/preview`.
- Frequency: **0.744 Hz**, source slot 1 of the prepared `[0.357, 0.744]` Hz inputs;
  the preparation's original normalization is retained.
- Completed modes: v16, `neural_component_field_with_stable_donors`.
- Completed-modes identity: `83dece66f94ab721a1504c25db41b609eed79be2811b83ba8b31fd44aedd3268`.
- Immutable trained modes: `outputs/_cache/trained_modes/8c653354f2b2e433f43aa568855f426b7e52adafb006dac9c5c56ec96da9d069`.
- Prepared inputs: `outputs/bush_neural_dense_controls_001/prepared`.
- Frozen experiment config: [features32.json](outputs/bush_neural_capacity_0744_001/configs/features32.json).
- Resolved configuration and source contract: [iteration.json](outputs/bush_neural_capacity_0744_001/features32/iteration.json).
- The frozen configuration above preserves the historical defaults; the current
  `configs/neural_component_field.json` now follows the 2026-09-18 recipe.

The accepted settings are width **256**, local features **32**, **3** message
layers, control coverage radius `0.015L`, maximum 32,768 controls, and deformation
and rotation-variation weights **0.1 / 0.1**. Keep Adam learning rate `0.001`,
maximum 2,000 steps, early-stop patience **50**, relative tolerance `1e-6`, and
seed 1729. The suggested longer patience was not applied to this baseline.

Use the whole-component field strategy: components need at least **10 Gaussians**,
**two controls**, and some effective image support to learn their own motion.
Reliable points in components with at least **101 Gaussians** and **two controls**
may supply propagation; other components receive pointwise motion from these
donors. Keep the existing donor checks, four-neighbor transfer, geometry,
interpolation and full modal-image supervision. Observation refinement is off.
See [component-field.md](docs/component-field.md) for the exact classification.

Historical capacity comparisons should use this frozen recipe and matching data,
frequency, gain, phase and motion scale. This acceptance covers the bush 0.744 Hz
result; it did not imply that the 32-feature recipe had already been trained on
corn or validated at other frequencies. Historical results and immutable saved
configs, quality gates and logs remain unchanged. No modal coordinates were fit
for this preview. The user performs visual evaluation.

## Current execution endpoint

On 2026-09-07, the user requested that future runs stop after obtaining the final
3D modes at the requested frequencies. Use `motion iterate-neural --stage modes`
(the default), retaining the accepted fragment propagation and 2D modal-image
supervision. Do not automatically fit per-frame modal coordinates to optical
flow, run physics post-fit, compute coordinate-dependent flow R², or run full
evaluation as a completion check. A later preview request uses `--stage preview`
and manual oscillation, without coordinate fitting. `--stage full` requires a
separate explicit request for coordinate/video reconstruction. Existing baseline
coordinates and historical metrics below are preserved records, not requirements
for new experiments.

The current v16 implementation uses **whole-component fields with separate
propagation donors**. Fixed pointwise propagation is composed into the final
Gaussian field before full modal-image supervision; there is no observation
post-refinement. The selected Corn and Bush results have completed training and
manual preview preparation.

## Historical corn neural baseline: accepted result and scope

On 2026-09-05, the user selected `corn_neural_fragment_propagation_001` as the
baseline at that time. Its original recipe, metrics and launch command below are
preserved as historical records, not the defaults for new experiments.

- Result: `outputs/corn_neural_fragment_propagation_001/modal_result`
- Result identity: `a60774d885e5e70047d6dbb2b2c09c40d34db5f1fc24a544d9d2598f3b2fc5bb`
- Completed modes: v9, `neural_fragment_motion_propagation`.
- Completed-modes identity: `f103cee158f9b316f92a97c7aae86de5e73515977a3b68a75bcf7c1f83452716`.
- Accepted artifact contains **one frequency: 0.225 Hz**, source slot 0, candidate index 7. The other 19 source frequencies have not been evaluated with this final recipe. Original full-20-frequency observation normalization is retained.
- Report: [experiment-report.md](outputs/corn_neural_fragment_propagation_001/experiment-report.md)
- Frozen run contract: [run-spec.json](outputs/corn_neural_fragment_propagation_001/run-spec.json)
- Validation and metrics: [run-status.md](outputs/corn_neural_fragment_propagation_001/run-status.md), [comparison_001.json](outputs/corn_neural_fragment_propagation_001/logs/comparison_001.json).
- Coordinates: direct, two views and 2,378 frames; no physics postfit.
- All 11 execution/check stages passed, including headless Viser data readiness. Visual baseline acceptance is the user's subsequent confirmation in this conversation. Immutable artifact quality-gate fields and original execution logs are preserved; this document records the later baseline decision.

## Historical corn recipe

- Static 3DGS remains fixed. Each frequency has an independent neural complex displacement field with shared control-node interpolation within the geometry graph.
- Geometry covers all 44,251 foreground Gaussians. Keep all mutual-KNN candidates: 8 neighbors, distance limit 0.008, `graph_edge_filter=none`; no RGB or depth/path edge filtering. Degree-normalized spatial weights remain enabled.
- Parent neural field uses A-group regularization: deformation weight **0.1**, rotation-variation weight **0.1**. Control radius is `0.03L`; interpolation support is `0.06L`; network width 64, three message-passing layers; learning rate 0.001, maximum 2,000 iterations, seed 1729. The exact remaining settings are in the frozen run contract.
- Full foreground modal-image rendering supplies supervision with fixed complex alpha; no rigid-basis fitting or green refinement is used.
- After training, classify whole components with at most **16 Gaussians** and bounding-box diagonal at most **0.016** as fragment candidates. Hosts must be nonfragment components with at least **4 times** the fragment's node count.
- Host anchors come from the **3-core**, using attachment distance **0.008**, local graph patch radius **0.008**, at least **3 supported anchors** per mode, and ambiguity ratio **1.25**.
- Transfer the parent's complex displacement and local rotation using common normalized Wendland weights for each fragment. All hosts and unselected Gaussians retain their parent mode values bitwise; there is no retraining or cascading between fragments. Geometric graph edges remain unchanged; attachment metadata is stored separately.
- Actual propagation: **250 components / 361 Gaussians**, including 193 of the original 197 singleton components. Seven candidate components / 21 Gaussians retain original motion due to ambiguity or lack of a host. The 33-point component also retains its original motion.
- Viewer roles: directly supervised = blue; structure inferred = green; fragment propagated = yellow; unresolved = purple. Original image contribution counts remain separately available.

The immutable v8 parent is `outputs/corn_neural_no_depth_edges_001/neural_completed_modes`, identity `ff739ec80f4311ae84ecd2bb1436a4a1cc263de1ae9141164e74ac82637f2760`. Preserve this parent and all linked upstream data with the accepted v9 result.

## Historical corn metrics and comparison convention

At 0.225 Hz:

| Metric | Accepted baseline |
|---|---:|
| Overall direct-flow R² | 0.4948244913741727 |
| Topology modal NRMSE | 0.5174500528710997 |
| Full-render modal NRMSE, fixed alpha | 0.5221506559211886 |
| Fragment-to-anchor displacement-difference RMS, 64 unit DFT phases | 0.0022375977883458127 |

On the same 361 fragment-to-anchor connections, the last metric decreased from 0.09370473887544951 before propagation, approximately 97.6%. These phase diagnostics use the raw DFT field at fixed amplitude, not calibrated physical strain.

Compare future candidates at matching cameras, source frequencies, and stated amplitudes. Use manual oscillator with matching phase, gain and motion scale to isolate mode changes; direct coordinates are separately refit and can alter video-driven trajectories. The accepted single-frequency R² must not be compared directly against a 20-mode result as an equal-capacity experiment.

Launch in Anaconda Prompt:

```bat
call "C:\Users\zitengsong\Documents\school\research\modal-gaussian\outputs\corn_neural_fragment_propagation_001\launch-viewer.cmd"
```

Expected URL after launch: `http://127.0.0.1:8087`.

## Historical rigid-basis reference

The previous accepted baseline, `corn_motion_basis_per_mode_trusted_002`, is retained as a historical 20-frequency reference. It is no longer the primary baseline for new representation experiments.

- Result: `outputs/corn_motion_basis_per_mode_trusted_002/modal_result`
- Result identity: `f95971dd62c6171bddd16c881283489b1b27b6b3490e61c3a05ff6366dc3ce98`
- Report: [experiment-report.md](outputs/corn_motion_basis_per_mode_trusted_002/experiment-report.md)
- Frozen run contract: [run-spec.json](outputs/corn_motion_basis_per_mode_trusted_002/run-spec.json)
- Basis policy: `trusted_per_mode`, independent weights per frequency, 20 frequencies, 23-component union, 7–23 trusted components per frequency, four local rigid bases plus zero.
- Base fit: graph penalty `0.01`, distance prior `0.001`.
- Green refinement: blue-green multiplier `1`, green-prior multiplier `1`; blue weights and modal displacements remain fixed to the v6 parent.
- Coordinates: direct; no physics postfit.
- Completed modes: v7; reuses the v6 parent in `outputs/corn_motion_basis_per_mode_trusted_001/basis_completed_modes`.
- Metrics: flow R² `0.8291204243942694`; topology NRMSE `0.6042767901685501`; full-render modal NRMSE with fixed alpha `0.668217890921227`.
- Execution state: all 12 checks passed; headless Viser readiness passed. User acceptance concerns the baseline choice and improved green-point appearance. Existing artifact quality-gate fields are preserved.

On 2026-09-05, the user requested removal of old outputs, retaining this final non-neural reference and the accepted v9 neural baseline. Earlier experiment results, including the initial five-frequency preview, were deleted. Shared input artifacts and the immutable v6/v8 parents remain at their original paths. See [the output cleanup record](outputs/CLEANUP.md) for the retained directory roles and verification.
