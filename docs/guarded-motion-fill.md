# Stable donors and constrained small-component learning

Implemented only; not yet evaluated on a new bush or corn training run. Use
`configs/neural_guarded.json` for the next experiment. Existing pointwise v12/v13
configurations and artifacts retain their original behavior for reproducibility.
The accepted baseline has not been replaced.

## Classification

| Component / point | New behavior |
| --- | --- |
| At least 101 Gaussians and at least 2 provisional controls | Can be a stable host, subject to reliable observation support |
| 10–100 Gaussians | Can learn a constrained residual; never a propagation donor |
| 1–9 Gaussians, including 6–9 | No independent learning or controls; pointwise propagation |
| Weak or occluded point, including one inside a large component | Pointwise propagation; no independent displacement correction |
| Large component with only one provisional control | Propagation, matching the earlier stable-host eligibility rule |

“Stable” identifies an eligibility class, not a guarantee of physical accuracy.
The geometry, control density, network width 256, local feature dimension 16,
and existing modal/deformation/rotation losses are otherwise inherited from the
prepared experiment. Host motion is learned in the new run; old motion arrays are
not used as initialization or frozen targets.

Reliable point/view evidence requires all of:

- The original selected-pixel renderer support and an identifiable fixed alpha.
- Positive projectable depth, in-frame projection, foreground alpha above the
  existing observation threshold, and center depth consistent with the rendered
  surface using the old graph's endpoint tolerance. Camera distortion is retained.
- Accumulated normalized feature contribution at least `min_observation_mass=0.05`.
  This is contribution summed over sampled pixels, not per-pixel opacity.

Each learning component needs at least `min_observed_points=3` reliable points
in the current frequency. A 10–100 point component additionally needs at least
`min_residual_observed_fraction=0.2` of its points reliable. Only its reliable
points receive residual corrections; its weak points propagate directly.
These are explicit initial thresholds, not measured optimal settings.

Static control domains use visibility/mass from all views so selecting fewer
frequencies does not alter saved control order. Active controls, donor points,
residual masks and ranks are determined independently per frequency. A control
can remain active when its position is weakly observed but its interpolation
influences a reliably observed point in the same eligible component.

## Motion and losses

Every non-donor selects the component of its nearest reliable **stable donor**,
then up to four donor points in that component, with normalized inverse-square
distance weights. There is no distance cutoff, cascade, or rotation extrapolation.
Small-component learned residuals never feed this lookup or another point's prior.

For a qualified small point:

\[
  \Phi_i=\underbrace{\sum_j\beta_{ij}\Phi_j}_{\text{stable neighbor motion}}
         +P_i\psi_i.
\]

`psi` comes from the existing GNN/control interpolation, now interpreted as a
residual on small components. `P` projects onto reliable observation directions:
visible-view Jacobians are normalized per view, their Gram matrices weighted by
relative contribution mass, and directions below `observable_rtol=0.1` times the
largest singular value are dropped. This preserves the neighbor motion exactly
in the retained numerical nullspace, for both real and imaginary displacement.

The added loss is

\[
 L_{neighbor}=\frac1{|S|}\sum_{i\in S}\|P_i\psi_i/s_k\|^2,
 \qquad \lambda_{neighbor}=1.0,
\]

where `S` is the fixed residual-point set and `s_k` is the existing frozen modal
amplitude scale. Unlike the earlier structure-only constraint, this penalizes a
small component's anomalous rigid translation as well as deformation. Stable-host
absolute displacement is not penalized. Pure followers have zero residual.
Full-image supervision still includes their rendered contribution and trains
their shared donor motion; it does not create follower parameters.

Final displacement and blended local rotation still enter the existing structure
loss on reliable learning edges. Network, alpha, interpolation and masks are
saved so the exported field can be reconstructed and checked on load.

## Optional observation refinement

On guarded artifacts, `--refine-observations` updates **only the same qualified
small residual points**, with the same fixed projectors. Stable donors, 1–9 point
components and weak/interior followers remain fixed. It penalizes the **total**
deviation from stable neighbor motion, including the trained residual, rather
than treating the trained result as a new unconstrained origin. The effective
prior weight is the maximum of the requested refinement prior and the parent's
neighbor prior (default 1.0). Both requested and effective settings are recorded.

Base guarded modes use completed_modes v14; guarded refinement uses v15. Legacy
v12/v13 loading and correction semantics are unchanged. Viewer blue includes
reliable small residual points, yellow means unchanged stable-neighbor propagation,
and purple means no motion source. No stable source in a requested frequency is
an explicit failure, not permission to use small components as donors.

## Subsequent use

Run in a new experiment directory, never resume the old pointwise experiment with
changed settings. For the prepared bush data, choose `configs/neural_guarded.json`
with `--frequency-hz 0.744`; default stage ends at 3D modes. Add `--stage preview`
only when a manual Viser preview is requested. Optional `--refine-observations`
uses the guarded rules above. This implementation did not run that experiment,
fit modal coordinates, generate PNGs or start a Viewer.
