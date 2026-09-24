# Maintaining the mainline pipeline

These rules apply to changes throughout this repository. Follow the user's task
scope; routine implementation and recovery within that scope need no extra approval.

## Read before changing code

Use [README.md](README.md) for stage ownership and entry points,
[BASELINE.md](BASELINE.md) for the current numerical recipe, and
[SCENE_STORAGE.md](SCENE_STORAGE.md) for artifact and cache rules. For reconstruction,
also read [COEFFICIENT_FITTING.md](COEFFICIENT_FITTING.md).
Before using scene data, inspect its catalog/index and [REBUILD.md](REBUILD.md).

Trace the changed function's callers, the stage's producer and its downstream
consumers. Identify whether the change affects computation, artifact format,
cache identity, or only presentation. Do this before choosing a new abstraction.

## Keep code in its owning stage

- Use the existing `preprocessing`, `geometry`, `flow`, `spectrum`, `motion`,
  `coordinates`, `results` and `vis` packages. The README owns the detailed map.
- Keep numerical work out of `cli.py`; it parses arguments and dispatches.
  Training, preparation and loaders must not depend on viewer initialization.
- Keep stage-specific helpers beside their consumers. Put a helper in `common`
  only when multiple stages actually share it; use `motion/common` for motion math.
- Reuse existing loaders, path resolution, hashing and atomic-publication helpers.
  Avoid duplicate implementations, one-implementation interfaces and speculative
  plugin/config systems. Leave vendored code and its licenses intact unless needed
  for the task.
- Maintain one supported mainline. Remove replaced implementations and their
  obsolete commands/docs. Do not add legacy readers, import shims or silent backend
  fallbacks to keep historical experiments runnable; Git preserves that history.
  A retained alternative must serve a concrete current requirement.

## Preserve stage boundaries

Stage orchestration owns loading, validation and publication. Numerical kernels
consume arrays/tensors and explicit settings. Keep scientific parameters in the
existing configuration structures, with validation; keep scene paths in catalogs
and command inputs rather than scene-specific branches in `src`.

Preserve explicit disk checkpoints and independently callable stages. A stage
must not automatically launch later training, fitting, export or viewing stages.
Batch orchestration composes the stage APIs; update it when a CLI or API changes.
Keep producer/consumer agreement on array shape, dtype, order, units and complex
sign convention, and on camera, reference, frequency and source identities.

Performance work should start with a measured bottleneck and a focused numerical
equivalence check, including gradients when relevant. GPU-resident execution can
reuse the existing kernels and contracts when needed; do not build a second
pipeline or remove recovery checkpoints speculatively.

## Protect scientific and storage contracts

- Recordings are asynchronous. Keep per-video temporal coefficients independent.
  RGB coefficient fitting keeps static geometry, appearance, cameras, displacement
  modes and angular modes fixed. Changing these assumptions is an explicit method
  change, not routine cleanup.
- `coordinates refine-scene` is the explicit joint-refinement method. Keep it
  separate from fixed-mode `fit-rgb`. Freeze reference graph nodes/edges, per-mode
  propagation costs, control positions and control fields; protect control
  Gaussians from position updates and density operations. Only non-control live
  Gaussians may move/split/cull; children inherit permanent reference roots.
  Reference graph indices and live Gaussian indices are different domains.
  Preserve support and the fixed-root incident-segment shape bound with checked
  rollback. Derive its radius from original unweighted edge lengths, never from
  moved children or frequency weights. Cancel splits with any invalid child;
  validate the bound on restore/publication and bake final fields before publication.
  Alternate sparse joint updates with exhaustive coefficient-only passes at full
  resolution. During coefficient passes freeze every Gaussian parameter and reuse
  one detached GPU basis; invalidate it after geometry changes or checkpoint load.
  Use the geometry-update clock for Gaussian LR/density/newborn age and the total
  update clock for coefficient LR. Keep each frame's Adam state across both phases.
- Keep original modal observation views separate from supervised sequences. Sweep
  has per-frame calibrated cameras and its own extraction clock; it is not an FFT
  observation or a synchronized recording. Use `coordinates/sequences.py` bindings
  throughout training, evaluation, export and playback. Average fixed views within
  their group, then weight fixed and sweep groups equally; undo the actual sample
  weight for density statistics. Never share per-frame coefficient Adam state.
  Sweep fitting/refinement uses an actual 30 FPS frame subset. Preserve source
  indices, timestamps, PNG hashes and cameras together; never relabel 60 FPS as 30.
- The explicit v16 reference importer in `tools/` is the authorized legacy boundary.
  It validates sources and field equivalence, publishes new fixed inputs, and never
  edits old manifests or invokes old training. General model loaders remain current.
- Preserve the current recipe unless the task changes it. Distinguish a numerical
  optimization from a change to the loss, sampling, normalization or model.
- Reuse only matching cache contracts. Share static/control geometry when valid;
  keep frequency-specific gains, soft weights and donor roles frequency specific.
- Resolve stored paths with `common.scene_store.resolve_path()`. Preserve raw data
  and published artifacts. Publish new outputs atomically into new scene-owned
  paths; never edit identities or bypass mismatch checks to reuse an artifact.
- A changed format or interpretation needs an explicit version/contract update
  and matching producer/consumer changes. Output-affecting code must participate
  in the appropriate cache revision. Check existing revision rules even for a
  refactor before claiming old caches or checkpoints remain reusable.
- Record affected stages, reusable ancestors and the required downstream rebuilds
  in `REBUILD.md`. Mark unknown/unreadable data as unverified. Do not run rebuilds
  or redirect catalog entries merely because the cleanup requires them.

## Verify and finish

Use the smallest relevant synthetic check for nontrivial logic. For interface
changes, check the producer-to-consumer handoff and affected CLI/batch commands.
For numerical changes, check the intended result and failure cases. Run broader
checks when the change crosses stages; check `git diff --check` before delivery.

`tests/` is intentionally local and Git-ignored. Do not force-add tests or restore
ignore exceptions. It may be absent in a fresh clone: use an available focused
check or create a small local synthetic reproduction, and report what actually ran.
The full local suite, when present, runs with `python -m unittest discover -s tests`.

Code/documentation work does not authorize real-scene training, fitting,
validation, exports or Viser. Execute only the task-authorized stages and recover
in-scope failures. Synthetic checks do not establish real-scene quality; distinguish
them from experiment completion, visual review and reconstruction metrics.

Update only the documentation affected by the change: README for interfaces and
ownership, BASELINE for the recipe, SCENE_STORAGE for storage contracts,
COEFFICIENT_FITTING for reconstruction, and REBUILD for invalidated artifacts.
Keep the repository pipeline skill's command recipes consistent with changed CLI
arguments. Report what changed, what was checked and what still requires a run.
