# Component fields

The current single-frequency model format is v18. It stores complex displacement,
angular displacement and control displacement, so viewers and coefficient fitting
can load the fields directly without replaying the training network.

## Field and donor domains

| Component | Field | May donate to other components |
| --- | --- | --- |
| >=101 points, >=2 controls, effective observation | Own control field over every member | Reliable members only |
| 10–100 points, >=2 controls, effective observation | Own control field over every member | No |
| Too small, fewer than two controls, or entirely unobserved | Propagation from eligible donors | No |

Effective support is positive full-render contribution above
`max(1e-12, 1e-8 * maximum_view_mass)`, intersected with alpha-identifiable views.
One supported member activates an eligible component. Donor gates do not remove
weak or occluded members from that component's own learned field.

A donor component also needs at least three reliable observed points. Reliability
uses saved visibility, observation mass and identifiability. Up to four donors
supply a recipient; unmatched recipients stay zero and are marked unresolved.
Support classes distinguish donors, other own-field points, recipients and unresolved points.

## Geometry and optimization

The shared mutual-KNN geometry determines connected components, controls and
material support distances. Each frequency supplies its own soft edge costs and
interpolation attenuation; these do not resample controls. The GNN predicts
complex control motion. Gaussian deformation and rotations are derived from this
field, and the modal projector compares it to aligned complex U/V observations.

The current loss uses per-view RMS normalization, deformation weight 0.03 and
control-rotation weight 0. Rotation is still present in rendered motion. See
[BASELINE.md](../BASELINE.md) and [the config](../configs/neural_component_field.json)
for exact defaults. Static geometry, camera parameters and appearance stay fixed.

Implementation: [`component_field.py`](../src/modal_gaussians/motion/component_field.py),
[`network.py`](../src/modal_gaussians/motion/network.py),
[`training.py`](../src/modal_gaussians/motion/training.py).
