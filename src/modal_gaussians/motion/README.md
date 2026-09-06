# Motion code map

Both motion pipelines use the same static scene and observation sources, and
export complex foreground displacement `phi[K,G,3]` for the shared rendered
design, coordinate fitting, result packaging, and Viewer.

## Neural field pipeline

Read these modules in order:

1. [geometry_graph.py](neural/geometry_graph.py): all-foreground geometry graph,
   graph-distance control selection, control graph, and fixed sparse interpolation.
2. [neural_field.py](neural/neural_field.py): `PerFrequencyModalGNN`, control-motion
   composition, structural/modal losses, and `train_single_frequency`.
3. [neural_modes.py](neural/neural_modes.py): frozen observation renderer, complete
   training orchestration, checkpoints, prefix export, and strict v8 loading.
4. [fragment_propagation.py](neural/fragment_propagation.py): local post-training
   fragment attachment and motion transfer; strict derived v9 loading.

The accepted baseline uses deformation weight `0.1`, rotation weight `0.1`,
`graph_edge_filter=none`, and fragment propagation. CLI defaults still use
deformation weight `1.0` and depth filtering. Directory reorganization does not
change these defaults or the accepted [baseline](../../../BASELINE.md).

## Rigid and motion-basis pipeline

1. [structure_graph.py](rigid/structure_graph.py): original observation-supported
   component graph and its format validator.
2. [rigid.py](rigid/rigid.py): complex component motion, trust checks, and rigid artifacts.
3. [motion_basis.py](rigid/motion_basis.py): basis candidates and shared weight fitting.
4. [motion_basis_frequency.py](rigid/motion_basis_frequency.py): independent weights
   and trusted basis selection per frequency.
5. [motion_basis_green.py](rigid/motion_basis_green.py): fixed-blue green refinement.

[motion_fill.py](rigid/motion_fill.py) also retains the older sequential
promotion/fill implementation and the v1/v2 validator.

## Shared boundaries

| Module | Responsibility |
|---|---|
| [completed_modes.py](common/completed_modes.py) | Common artifact type and lazy version dispatch to strict v1–v9 validators |
| [mode_mapping.py](common/mode_mapping.py) | Exact local-to-source frequency mapping; never infer ordering from frequency values |
| [sources.py](common/sources.py) | Shared scene/topology/measurement identity checks; fixed-alpha compatibility adapter |
| [projection.py](common/projection.py) | Projection Jacobian, sampling configuration, mask/alpha pixel selection, foreground feature sampling |
| [geometry_ops.py](common/geometry_ops.py) | Geometric image sampling, preserving historical float32/float64 arithmetic separately |

The neural pipeline still accepts the existing rigid artifact as
`--alignment-from`. The adapter validates that artifact but exposes only complex
alpha and identifiability arrays to neural code. It neither initializes from
rigid motion nor applies rigid trust. The observed-graph identity remains part
of the existing source contract, including when neural depth filtering is disabled.

Static reconstruction, `flow/`, topology, frequency selection, measurements,
synchronization, rendered design, coordinates, and `vis/` remain shared at the
package root. Rendering and result code import the neutral completed-modes
interface. Viewer uses method-specific graph display adapters after loading it.

## Imports and saved-result compatibility

Use canonical imports in new code:

```python
from modal_gaussians.motion.neural.neural_field import PerFrequencyModalGNN
from modal_gaussians.motion.rigid.rigid import solve_rigid_components
from modal_gaussians.motion.common.completed_modes import load_completed_modes
```

The ten previous root module aliases have been removed. Import motion modules
from `motion.neural`, `motion.rigid`, or `motion.common`; for example, replace
`modal_gaussians.neural_field` with `modal_gaussians.motion.neural.neural_field`.
Archived experiment scripts and source snapshots under `outputs/` retain their
original imports for provenance; update those imports before rerunning an archived
helper against the current package. Current CLI commands and saved-result loading
use the canonical paths and do not need these aliases.

`rendered_design` still re-exports its former projection helpers and configuration.
Within `motion.rigid`, `motion_fill.load_completed_modes` forwards to the common
loader; shared consumers should import the common loader directly.

CLI names/options, default values, artifact format strings, identities, source
paths, saved network state dictionaries, and numerical algorithms are unchanged.
No result migration or retraining is required. Run development checks with
`python -m unittest discover -s tests -v` from the repository root.
