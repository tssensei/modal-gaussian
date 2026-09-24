# Scene storage

All active inputs, reusable caches and experiments belong to `scene_library/`.
Start with `scene_library/<scene>/catalog.json` and `results/index.json`.
The root `registry.json` owns physical relocation and asset aliases.
`MODAL_GAUSSIANS_LIBRARY` can select another library root.

```text
scene_library/
  registry.json
  _shared/tools/                  # external tools and weights
  <scene>/
    catalog.json                 # mutable asset pointers
    data/                        # raw recordings/frames, where cataloged
    references/                  # video timing, masks, fixed pixel grids
    geometry/                    # COLMAP/static/prepared geometry
    flow/                        # reusable reference flow, where cataloged
    cache/                       # identity-keyed numerical caches
    checkpoints/                 # resumable work
    results/index.json           # curated completed-result bindings
    experiments/<new_name>/      # new immutable stage outputs and mutable run state
```

Actual locations come from the catalog; an experiment can own its own flow,
spectrum and geometry. Do not relocate files just to resemble this diagram.
Keep raw inputs under existing `data/` locations. Never modify originals in place.

```sh
modal-gaussians storage list --scene bush
modal-gaussians storage path --scene bush --asset static
modal-gaussians storage run --scene bush -- viewer --input @mode:0.744 --work-dir @experiments/NEW_VIEWER
```

The last command starts a viewer and is only an example for an authorized viewing task.
`@asset/suffix` resolves a catalog asset before dispatch.

## Identity and publication

- Resolve all manifest paths using
  `modal_gaussians.common.scene_store.resolve_path()`. Stored old paths can be
  relocation aliases; do not rewrite identity-bearing metadata.
- Published arrays, model tensors and source manifests are immutable. New inputs,
  reference selections, schemas or numerical settings require new output paths.
- A cache key binds its inputs, numerical settings and implementation revision.
  Existing files alone do not establish a valid cache hit.
- Share KNN/control geometry only across matching static identity and Gaussian
  order. Modal weights, view gains, donor eligibility and fields are frequency specific.
- Mutable run files include logs, status, scheduler limits and checkpoints.
  Atomic publication prevents partially written results from appearing complete.
- `motion batch-neural` publishes a local `index.json` after all requested modes
  finish. Pass this file to viewing/fitting. Catalog/curated-index updates are
  separate bookkeeping; a new batch does not silently redirect existing aliases.

The path resolver stays because current immutable data depends on relocation.
It does not load removed model/flow schemas. [REBUILD.md](REBUILD.md) lists the
pre-cleanup artifacts that now require new outputs.
