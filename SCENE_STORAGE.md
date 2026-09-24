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

## Joint refinement artifacts

```text
experiments/NEW_REFINEMENT/
  motion_reference/ # fixed motion reference v1; explicit old-model import or current builder
  sweep_coordinates/ # sweep RGB coordinates v1 at 30 FPS; fitted or parent-row subset
  baseline_result/   # original scene, fixed-view + sweep RGB; result v3
  baseline_evaluation/ # explicit native PNG metrics
  prepared/       # preparation v2; immutable reference.npz, initial.npz, manifest.json
  work/           # checkpoint.pt, training.jsonl, run.json, process lock
  refined/
    manifest.json
    scene/        # static scene v4; tensors.pt and identity_map.npz
    mode_bank/    # completed modes v19; phi.npy, rotation.npy, support.npz
    coordinates/  # refined RGB coordinates v2; coordinates.npy
  result/         # explicit materialization, modal result v3
  evaluation/     # explicit evaluation, with per-frame CSV
  exports/        # explicit export-video
```

The reference graph owns its original node coordinates/order and frequency
propagation costs. Live foreground rows own `uid`, `root_id`, `protected` and
`birth_step` (geometry-update clock); children inherit roots but receive new UIDs. Removing a live row
does not remove a reference node. Prepared path tables remain sparse on disk.
Queries cache the immutable tables on the selected device, expand sparse indices
there, and use bounded Gaussian blocks with up to four frequencies per group and
activation recomputation. Device caches are transient and excluded from checkpoints.
Within an update, sampled sequences share each queried field block and one query
backward pass. Dynamic-position/angular gradient buffers are also transient;
no shared query graph is saved or reused after a parameter/density update.

Shape neighborhoods and median incident edge lengths are derived from this same
immutable unweighted graph. Their CPU/GPU adjacency caches are transient; no new
reference file or artifact version is needed. `shape_radius_fraction` is recorded
in training settings/run identity and the published coordinate settings. Children
inherit the original root's fixed segment neighborhood and tolerance, not a new
neighborhood around their parent. Checkpoint loading and publication reject centers
outside that bound. A changed fraction or implementation requires a new work
directory; original graph/path caches remain reusable.

Scene v4 has fresh foreground/tensor identities and ranges, unchanged cameras
and normalization, and explicit parent-scene/refinement provenance. A parent's
manual partition remains provenance; it is not a partition of the new rows.
Mode v19 stores final displacement/angular fields in the new foreground order,
plus inherited observation roles and original Spectrum sources. It renders
directly without reopening source GNN models. Reference overlays use a separate
original-node index domain.

Refined coordinates bind the new scene/modes and the original input PNG records.
The old RGB/design artifacts are initialization provenance. Modal result v3 accepts
multiple disjoint coordinate artifacts with identical scene/mode identities and
mode order. Linear designs are required only for ordinary direct/RGB coordinates.
Result v1/v2 is not read; materialize into a new directory and preserve old baselines.
Publication checksums and identities are generated
for the new outputs; old flow/reference manifests are never rewritten.

`result evaluate` publishes an immutable sibling `evaluation/` directory containing
`manifest.json`, `metrics.json` and `per_frame.csv`. Its identity binds the result,
actual input PNG hashes, native-resolution metric protocol, implementation and
dependency versions. Keep the pre-refinement result/evaluation beside a short
baseline record; do not redirect or overwrite the old model/coordinate artifacts.
Formal evaluation hashes source tensors, baked fields and each input image.

`--resume` requires the same prepared identity, configuration, implementation
revision and device. Checkpoints atomically contain all parameters, optimizer
rows, mappings, density statistics, sampler cursors/permutations and RNG states;
checkpoints also store round/phase, total/geometry/coefficient update counters,
per-round sparse selections and exhaustive coefficient permutations. Save at every
phase boundary and every 200 total updates. Old checkpoints fail the implementation
contract. A coefficient phase keeps one detached complex64 `[K,G,3]` displacement/
angular pair on GPU, excluded from checkpoints. Resume rebuilds it once; geometry
changes invalidate it, and final publication uses it directly. Disk stage boundaries
remain explicit. A failed run keeps
the last complete checkpoint and does not publish a final bundle.

Prepared/refined sequence records bind each coefficient row to its PNG hash,
camera identity, original frame index and timestamp. Cameras remain authoritative
in the scene. Sweep coordinates bind extraction metadata and their fixed-view
normalization source, without flow/reference/FFT identities. All registered sweep
frames must lie on a uniform extraction grid; sparse uniform strides retain their
timestamps and adjust playback FPS; irregular grids and noninteger ratios are
rejected. `fit-sweep --fps 30` selects before optimization. The pure
`downsample-sweep` stage selects coefficient rows and all bindings from an existing
artifact, records parent identity/selected rows and reports zero newly optimized
frames. Source artifacts and PNGs remain unchanged. Bush selects rows 0,2,...,722
from 724 frames at 60 FPS, producing 362 rows at 30 FPS; view1 remains 1170 rows at
30 FPS. Never change only an FPS label. The mode bank still records the original
three modal observation views, independent of which sequences supervise refinement.
