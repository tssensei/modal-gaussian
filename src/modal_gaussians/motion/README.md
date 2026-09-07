# Motion code map

The selected baseline is v16 `neural_component_field_with_stable_donors`.
See [BASELINE.md](../../../BASELINE.md) for the accepted result and
[the cleanup report](../../../docs/code-cleanup.md) for this refactor.
The [earlier audit](../../../docs/baseline-code-audit.md) describes the code before cleanup.

## Current neural pipeline

Read these modules in this order:

| Module | Responsibility |
|---|---|
| [baseline.py](neural/baseline.py) | Preset for new experiments: width 256, features 32, three message layers, current component-field strategy |
| [geometry_graph.py](neural/geometry_graph.py) | Foreground mutual-KNN graph, graph-distance controls, fixed sparse interpolation |
| [component_field.py](neural/component_field.py) | Whole-component learning eligibility; separate reliable donors and fixed pointwise transfer |
| [neural_field.py](neural/neural_field.py) | GNN, complex control field, differentiable transfer, modal/structure losses and single-frequency optimization |
| [neural_modes.py](neural/neural_modes.py) | Frozen observations, training/checkpoints, mode selection and artifact publication |
| [artifacts.py](neural/artifacts.py) | Persisted-array/source validation and network replay for v8/v10/v11/v12/v14/v16 |
| [prepared.py](neural/prepared.py) | Immutable observations and independent geometry/control caches |
| [iteration.py](neural/iteration.py) | Resolve configuration and run modes, optional preview, or explicitly requested full evaluation |
| [preview.py](neural/preview.py) | Bind modes, rendered design and sources for manual preview without video coordinates |
| [strategies.py](neural/strategies.py) | Lazy version/strategy dispatch; only selected implementations contribute to training dependencies |

The default is `modes`. `preview` additionally builds rendered modal-image
comparisons and manual oscillation inputs. Only explicit `full` imports and runs
coordinate fitting/result packaging. Current training fills follower motion
before the full-foreground observation loss; it does not run historical
post-training propagation or observation refinement.

## Configuration and compatibility

`fit-neural`, preparation from explicit sources, and `iterate-neural` without
`--config` select [baseline.py](neural/baseline.py). The baseline override keeps
the preparation's observation units and sampling. An explicit iteration config
continues to overlay the preparation's recorded defaults, so use
[neural_component_field.json](../../../configs/neural_component_field.json) as
the starting config for baseline parameter studies.

Historical decoding stays separate: `NeuralModesConfig` and `NeuralFieldConfig`
retain early defaults, and old v16 configs without `min_learning_controls`
retain their original meaning. Model state-dict keys, array names, identities
and saved `phi` semantics are unchanged. Saved results need no migration.

The new iteration contract is version 2 and code revisions reflect the new
module boundaries. Start changed-code experiments in a new output directory;
do not reuse an old revision's directory as a resume. Existing result loading
and Viewer commands remain supported.

## Shared functions and sources

| Module | Responsibility |
|---|---|
| [completed_modes.py](common/completed_modes.py) | Unified lazy dispatch to the appropriate strict format loader |
| [sources.py](common/sources.py) | Shared source identities and the fixed-alpha alignment adapter |
| [projection.py](common/projection.py) | Calibrated Jacobians, sampling and foreground feature projection |
| [mode_mapping.py](common/mode_mapping.py) | Explicit exported-to-source frequency mapping |
| [graph_ops.py](common/graph_ops.py) | Induced host subgraphs, shared across strategies |
| [point_transfer.py](common/point_transfer.py) | Nearest donor component and fixed per-point transfer weights |
| [visibility.py](common/visibility.py) | Frozen depth/alpha visibility with calibrated cameras |
| [geometry_ops.py](common/geometry_ops.py) | Shared image sampling with preserved numerical conventions |

The existing rigid alignment and observed-graph formats remain source contracts.
The neural path consumes fixed complex alpha and identifiability, plus depth
tolerances for donor visibility. It does not consume rigid motion or trust.
These dependencies still require the corresponding old loaders.

## Historical implementations

- [legacy/neural](legacy/neural/README.md): fragment, surface, pointwise, guarded
  and observation-refinement strategies. They import shared helpers instead of
  defining copies. No forwarding files remain at their former `neural/` paths.
- [rigid](rigid): rigid components, sequential fill, motion-basis fitting,
  per-frequency candidates and green refinement. This existing boundary stays
  intact because old results and alignment sources still use its loaders.

Use canonical imports under `motion.neural`, `motion.common`, `motion.rigid`,
or `motion.legacy.neural`. Archived scripts referencing the moved Python
modules need import-path updates; artifact paths and CLI command names do not.

## I/O boundaries

A public load still verifies its on-disk data. Within one producer call, retain
validated arrays/objects across atomic publication and downstream preview/design
construction instead of loading them again. Reused inputs must match the
requested paths and linked identities. A competing cache publisher's output
is independently validated. Only source-code hashes use a process-local stat
cache; user data never bypass content validation on that basis.
