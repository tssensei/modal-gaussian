# Whole-component fields with separate propagation donors (v16)

The v14 reliability gates could switch individual members of a learning component
to pure propagation. They also replaced small-component fields with projected
residuals around nearby stable motion. v16 separates **how a component moves**
from **which of its points may supply motion to other components**.

| Component at the current frequency | Motion representation | External donors |
| --- | --- | --- |
| At least 101 points, at least two controls, some effective image support | Own control field for every member | Only qualifying reliable points |
| 10–100 points, at least two controls, some effective image support | Own control field for every member | None |
| Fewer than two controls at the configured coverage radius | Pointwise propagation for the whole component | None |
| 1–9 points | Pointwise propagation | None |
| Entire component without effective image support | Pointwise propagation | None |

Effective support is the existing full-render contribution test
`mass > max(1e-12, 1e-8 * maximum_view_mass)`, intersected with the frequency's
alpha-identifiable views. One supported member activates the whole eligible
component. A weak or occluded member retains its component's field; it is not
replaced by a field from a nearby component.

Donor eligibility additionally requires a component of at least 101 points,
at least two controls, and at least three reliable members at this frequency.
Each actual donor must have an identifiable supported view with contribution
mass at least 0.05 and pass the existing center visibility/depth/alpha test in
that same view. Failing any donor condition does not disable the component's
own field. These defaults are all explicit in the new config.

Each propagation recipient selects the nearest qualifying donor, chooses that
donor's component, then interpolates at most four eligible donor displacements
within it with the existing inverse-square weights. There is no distance cutoff,
rotation extrapolation, propagation cascade or small-component donor. A whole
unobserved component is consistently a propagation recipient, but its members
are filled pointwise. If no donor exists at a frequency, recipients remain
explicitly unresolved/zero while observed components keep their own fields.

## Training and storage

The control count is first measured on size-eligible components using the fixed
graph-distance coverage radius. Components with fewer than
`min_learning_controls=2` controls are then removed from the network's control
domain in their entirety. No artificial extra control is added to make a tiny
component qualify. The graph-distance sampling radius and interpolation kernel
are unchanged. Controls of wholly unobserved components are inactive per frequency;
the saved ordering is stable when selecting a subset of frequencies. Restoring
previously rejected learning components can increase the actual control count;
the configured budget is still enforced.

All learning-component geometric edges participate in the deformation loss,
including direct-to-inferred and inferred-to-inferred edges. Pure-propagation
components do not contribute independent structural edges. Full-foreground
rendering still occurs after differentiable propagation, so follower image
residuals can supervise donors. Alpha, image support, energy normalization and
the renderer retain their existing meaning.

The network directly predicts the component field, with no neighbor-residual
parameterization, absolute neighbor penalty or per-point observable projector.
Different components can have different translations, rotations and phases;
the structure loss does not penalize distinct component-wise infinitesimal rigid
motions. This restores capacity; it does not guarantee physically correct motion
from weak or inconsistent observations.

The current baseline override uses width 256, local features 32, three message layers,
control radius 0.015L, maximum 32768 controls and deformation/rotation weights
0.1/0.1. On 2026-09-07, the user selected
`outputs/bush_neural_capacity_0744_001/features32` at 0.744 Hz after visual
comparison. Only local feature dimension changed from the preceding 16-feature
reference; the combined six-layer experiment is not the selected baseline.
See [BASELINE.md](../BASELINE.md) for the frozen configuration. Other settings
inherit from the preparation. Use a **new** experiment:

```text
modal-gaussians motion iterate-neural --prepared <prepared> --config configs/neural_component_field.json --output <new-experiment> --frequency-hz 0.744 --stage preview
```

`--stage preview` prepares the manual Viser preview without modal-coordinate
fitting or launching the viewer. Omit it to stop at 3D modes. Do not supply
`--refine-observations`: v16 explicitly rejects that option before training.

The strategy is `component_field`, completed-modes version 16, method
`neural_component_field_with_stable_donors`. Arrays separately record
`u_own_field_mask[K,G]`, `u_donor_mask[K,G]`, component support, control counts,
frozen donor visibility and pointwise transfer indices/weights. `u_control_count`
records the count before the minimum-control filter, so a rejected component's
single control is auditable even though it is absent from `c_positions`. They participate
in artifact identity, cache keys and deterministic reconstruction checks. The
old strategies, configs, artifacts and baselines remain available. Earlier v16
configs without `min_learning_controls` load with the original value of one,
preserving their exact identities. New experiments use the explicit value two;
changing this value requires a new experiment and cannot resume old weights.

Viewer roles retain their actual meaning: blue = directly supervised own field,
green = structurally inferred own field, yellow = propagated, purple = unresolved.
Donor eligibility is stored separately and does not change these colors.

## Development validation

Focused synthetic checks cover the 9/10/100/101 boundaries, weak and occluded
members, single-control large components, wholly unobserved components, no-donor
fallback, frequency selection, independent translations, structural penalties
on inferred members, gradients and exact resume. A four-Gaussian CUDA scene
checks actual full-render training, interruption/resume and exported-field
reconstruction. Loading and tamper rejection, cache invalidation, role mapping,
and representative v8/v14 loading are also checked. The subsequent two-control
learning gate has targeted synthetic coverage for component-wide rejection,
network input removal, retained two-control fields and old v16 identities.
Subsequent bush and corn training runs used this two-control gate. The current
user-selected bush baseline uses 32 local features; earlier experiments retain
their original configurations. Visual acceptance of this result is not a claim
of validation across other datasets or frequencies.

Core implementation: `motion/neural/component_field.py`; tensor assembly:
`motion/neural/neural_modes.py::_field_geometry`. The neural network and loss
implementation in `neural_field.py` have not been changed for v16.
