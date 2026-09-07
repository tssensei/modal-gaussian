# Surface attachments (completed modes v11)

This implementation changes the spatial control domain and fixed attachment
operator. It has development validation only; it does not replace an approved
baseline or claim that the bush artifacts have been retrained.

## Host selection and motion

At the existing scene scale `L` and control radius `h`, components with at least
`min_host_nodes=101` Gaussians are provisionally sampled. A component becomes a
host only if it also receives at least `min_host_controls=2` controls. This is an
experimental selection heuristic, not a physical test of whether an object may
move independently. Provisional sampling cannot consume the final control
budget; final host sampling still enforces the configured hard budget.

All other components are attachment candidates, regardless of extent. They
receive no independent network inputs or control parameters. The original
foreground geometry graph and Gaussian ordering are retained.

The host network outputs complex translation and local rotation coefficients.
Host interpolation produces `Phi_j, R_j`. An attached point uses

```
Phi_i = sum_j beta_ij * (Phi_j + cross(R_j, x_i-x_j))
N_i   = sum_j beta_ij * N_j
```

`beta` and the composed interpolation `N` are fixed before optimization. The
full foreground renderer includes these points, so their image losses
backpropagate into host controls. Network width, depth, learning rate, loss
definitions and normalizations are inherited from the experiment configuration.
The fine graph receives no attachment edges; coupling is through interpolation.

## Attachment decisions

1. Search host centers within `search_radius_fraction * L` (default `0.03L`).
   No giant Gaussian can bypass this hard distance cap.
2. Compute each Gaussian's covariance from its activated scales and quaternion.
   Along the center-to-center direction `e`, its projected support radius is
   `support_sigma * sqrt(e.T @ covariance @ e)` (default `2 sigma`). The gap
   between these projected supports must not exceed `max_support_gap=0.008`.
   This is a Gaussian footprint approximation, not an exact surface-distance
   calculation or a guarantee that two real branches touch.
3. For each candidate component, rank hosts by how many member points they can
   cover, then median distance of those points to their nearest accepted host
   center. Only one host is chosen per candidate component. Host-size-ratio and
   minimum-three-anchor gates from v10 do not apply.
4. Equally covering hosts within `ambiguity_ratio=1.25` of the nearest host's
   median distance are alternatives. Actual foreground contribution and
   endpoint depth agreement identify visible points. Co-visible endpoint pairs
   at compatible camera depth vote for a host. A unique best vote resolves a
   tie; otherwise it remains explicitly ambiguous. Occluded/out-of-frame views
   contribute no vote and never veto a sole geometric candidate. RGB is unused.
5. Accepted points get anchors in a `patch_radius=0.008` geodesic neighborhood
   of their nearest accepted host point. Boundary points and one-anchor patches
   are allowed. Interior 3-core points get a soft `core_preference=1.25` weight.
   Distance weights are proportional to
   `exp(-((distance-min_distance)/patch_radius)^2)` times the core preference.
6. Average geometric weight rows along the original candidate component's
   edges, four steps with neighbor strength `0.25`. Reapply the search/support
   limits and normalize after each step. Rows never cross chosen hosts, and
   unassigned points do not acquire weights by smoothing. A member outside its
   component's chosen host support stays unresolved with its own reason.

For prior v10 fragments, the old deterministic assignment is replayed with
`legacy_config`. Successful patches whose host remains eligible and whose
anchors satisfy the new support limits retain their original rows exactly.
Those rows are not smoothed. Legacy replay does not use trained motion.

## Status and validation

`a_point_status` records host, attached, outside search radius, excessive support
gap, ambiguous host, or outside the selected host's support. Detailed per-point
host IDs, nearest-host distances, sparse anchor rows, preserved legacy flags,
provisional control counts, covariance and view evidence are saved in v11.

`attachment_preflight.json` is written in the training work directory before
optimization. It reports geometry coverage/failure reasons and per-frequency
unsupported hosts. Unassigned geometry and whole host groups without valid
frequency-specific observation support remain exactly zero and purple;
foreground labels are never changed by attachment failure.

The loader replays classification, host selection and interpolation, checks
the static covariance source, and reconstructs exported `phi` from saved
network weights. All new arrays/configuration participate in artifact and
resume identities. Cached control inputs include covariance and view-evidence
identity; loss-only changes reuse them. Observation and fine-graph caches stay
independent of attachment parameters. Old v8/v9/v10 files retain their loaders
and historical interpretation.

## Enabling the strategy

Fresh `fit-neural` and preparation from explicit sources default to surface
attachments. Explicit historical fragment JSON without a `strategy` key still
selects v10 for compatibility. For new direct-fit fragment JSON, include
`"strategy": "surface"`; other settings may be omitted to use their defaults.

Existing prepared snapshots keep their recorded strategy. Switch one experiment
with the supplied **iteration override**, keeping all neural/loss settings:

```bat
modal-gaussians motion iterate-neural ^
  --prepared <existing-prepared-directory> ^
  --config configs\neural_surface_attachments.json ^
  --output <new-experiment-directory>
```

The switch copies the snapshot's old fragment settings into `legacy_config`.
It creates new controls and a new training run; it cannot resume a v10 run as
v11. The command stops at validated modes by default. Add `--stage preview`
only when a manual preview is requested. Neither form starts Viser, fits modal
coordinates, or generates PNGs. The supplied override is for `iterate-neural`,
not the flat `fit-neural --fragment-config` schema.

To train only an existing prepared frequency, add `--frequency-hz 0.744`.
Repeat the option for multiple frequencies; results retain source order. Missing
or duplicate frequencies fail before training. This reuses the exact DFT, fixed
alpha and full-source normalization snapshot without rebuilding flow or spectra.
The experiment and model cache bind the selected source slots. Only those slots
are optimized and published; the completed artifact records `completed_selection`,
the original source list and checkpoint hashes. Network seeds retain their source
slot, and preview maps the selected mode back to the correct dense modal image.
Omitting the option retains the existing all-prepared-modes behavior.

Viewer supports v11 graph/roles and baked motion without running the network.
Solo playback and selecting a phase frequency also select that source
frequency's role colors, including when frequency display order is sorted.
When multiple modes play, motion is their sum while color still describes the
one frequency shown in `Modal role mode`.
