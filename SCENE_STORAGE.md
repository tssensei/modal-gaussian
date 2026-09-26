# Scene storage

Manual subject selections now use selection **v3**, with `box_positions [B,3]`,
`box_wxyz [B,4]`, `box_dimensions [B,3]` and `B >= 1`. A Gaussian is selected only
when its center is inside any box. The NPZ records the box count, union
rule, source identities and selected indices. Applied static scenes remain v6,
with partition method `manual_subject_selection_v3` and all box arrays retained.
Spectrum regions project the union using a separate ray-depth interval per box;
overlapping pixels are counted once. A single box uses
the same v3 format. Old v1/v2 selections/partitions are not read through a legacy
branch; preserve them and explicitly save/apply a new selection before reuse.

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

Motion-refinement early stopping, when enabled, stores latest optimizer/sampler
state and progress history in `work/checkpoint.pt`, checked best states under
`work/best/checkpoint_STEP.pt`, and readable history in `work/early_stopping.json`.
The latest checkpoint binds the selected best file's hash; unreferenced newer
files after interruption do not change selection. Final coordinates include
`training_selection` provenance. Actual training steps can exceed published steps.

The reference/adjacent-flow diagnostic owns `contract.json`, `design/`, `pairs/`,
`solution/`, `comparison/`, `curves/` and mutable `logs/` under a new experiment.
Each completed stage publishes atomically with source/output hashes. Pair flow is
explicit forward/backward adjacent flow, never a replacement SEA-RAFT reference
artifact. Solution arrays are original-unit complex64, with an explicit shared
reference offset and no temporal centering. These experiment products are not
accepted as formal RGB coordinates or modal results. They preserve original
scene/bank/flow/PNG inputs and do not redirect catalog entries.
`refine-motion --flow-initialization` is the explicit checked import boundary for
the reference-only solution. The run/checkpoint binds source identities, consumed
array hashes and the exact supervised prefix; final coordinates retain this
provenance. The original prepared identity remains a parent identity, while output
views/images identify the actual subset. No source artifact is rewritten.

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

## SH static scene artifacts

Current static scene versions are **v5** (trained base), **v6** (repartitioned or
manually selected). Historical geometry-refined v7 scenes are no longer accepted.
Each partition stores `means`,
`quaternions`, `log_scales`, `opacity_logits`, `sh_dc [G,3]` and
`sh_rest [G,15,3]`. The manifest declares `spherical_harmonics_world` and the active
SH degree (0..3), which also participates in scene identity. Repartitioning and
density changes preserve/remap every SH row; motion refinement preserves the whole scene.
Static resume v4 binds tensors, optimizers, sampler state, iteration budget, optional
depth identity and
training/scene/density/radial-render implementation hashes. Pixel and device-grid
caches are transient process memory and are not checkpoint or artifact fields.
Earlier direct-RGB scenes and resume
v1/v2/v3 are not current inputs. Training summary v3 and epoch/checkpoint statistics
contain RGB and unweighted depth losses, with no removed mask-loss fields. Scene tensor formats
remain v5/v6. Preserve old outputs as described in
[REBUILD](REBUILD.md); never rewrite their manifests.

## Static depth targets

`static prepare-depth` atomically publishes `modal_gaussians.static_depth` v1 in
a new experiment's `depth/` directory. Each camera has a hashed NPZ containing
float32 `depth [H,W]`, float32 `confidence [H,W]` and bool `valid [H,W]` on the
original training pixel grid. Invalid values are zero. Depth is camera Z in
normalized scene units, not ray distance or inverse depth.

The content-hashed manifest binds the full static dataset identity, camera/image records,
normalization, grouping, model file hashes, inference settings, DA3 source/runtime
and producer hashes. It records processed K/grid and confidence thresholds.
The loader validates every file and requires complete camera coverage; changed
pixels, poses, normalization, missing arrays or corrupted targets fail before
training. `inference.log` is retained; intermediate undistorted PNGs are temporary.
`static train --depth` records the target identity/path in `depth_supervision` and
the target identity in checkpoint v4. Changing it forbids resume. Base scene
tensor format and identity calculation remain v5; rendered content changes produce
new scene identities through the tensor hashes, as before.

## Stabilized recording artifacts

Sequence reference v2 explicitly declares stabilized or tripod capture. Default
preparation requires a static scene and registered raw reference of matching
resolution/hash. Its `mask_union.npy` excludes pixels outside `valid_mask.npy`
(common support). Stabilized sequences v2 own `images/`, `masks/`, `valid/`,
`geometry.npz` and a manifest under `stabilized_sequence/`. Geometry records contain
poses, target camera, depth/completion flags, point IDs, holdouts and timestamps.
Identity binds source pixels, settings/code, static camera/scene and COLMAP hashes.
Missing pixels remain black and invalid; no source camera/manifest is rewritten.

SEA-RAFT v3 owns time-common trajectory support; spectrum/selected exports v2
and neural preparation v3 carry it. Fixed RGB coordinates v2 and refinement preparation/coordinates v5 bind
the common valid-mask path/checksum in each fixed recording's image record.
Sweep has native full-frame support. Evaluation v2 records support identities and
rejects comparisons with different supports. Old artifacts remain historical;
rebuild requirements are in [REBUILD](REBUILD.md).

## Motion refinement artifacts

```text
experiments/NEW_REFINEMENT/
  motion_reference/ # reusable fixed reference v1, explicit v16 boundary if needed
  prepared/         # preparation v5: reference.npz, operator.npz, initial.npz, manifest
  work/             # checkpoint.pt, training.jsonl, run.json, process lock
  refined/
    manifest.json   # references original STATIC; no scene tensor copy
    mode_bank/      # completed modes v20: phi.npy, rotation.npy, support.npz, operator.npz
    coordinates/    # refined RGB v5: coordinates.npy + sequence/image bindings
  result/           # explicit modal result v3
  evaluation/       # explicit metrics and per-frame CSV
  exports/          # explicit videos
```

Preparation binds selected fixed sequences directly and an optional genuine 30 FPS
sweep subset. There is no fitted-coordinate input. `initial.npz` contains frozen
pixel-pair scales; q starts at zero inside training. Original modal observation views
remain complete and distinct from supervised sequences. Reference identity and
operator checksum participate in preparation identity. Each frame keeps its PNG
hash, authoritative static-scene camera, source index, timestamp and coefficient row.

`operator.npz` has an explicit internal `version=2`: per-mode CSR `ptr [K,G+1]`
(global entry offsets), `control [NNZ]`, float32 `weight [NNZ]` and unweighted
canonical `lever [NNZ,3]`. A second CSR (`query_ptr [K,G+1]`, `query_root`,
`query_weight`) preserves the original own/donor two-stage float32 sum order.
Coalesced version-1 operators are rejected: reassociation can violate the original
field tolerance. All modal weights/valid controls remain frequency specific.
Preparation verifies both fields at original positions. No
query graph, optimizer graph or dense `[K,G,C]` array is published.

All Gaussian tensors/order/counts and the original static identity remain exact.
Mode v20 stores final complex64 `[K,G,3]` displacement and angular fields, original
and final complex control fields, fixed operator, control validity, reference graph
and inherited observation roles/alpha. It directly references the original scene;
the parent manual partition remains valid because no rows change. Viewer renders
the final fields, without expanding source GNN models. Spectrum retains original
modal images; projection caches use the new completed-mode identity.

Coordinates v5 bind original STATIC, new mode identity and exact input frames.
Result v3 needs no linear design for these coordinates; ordinary direct/RGB paths
retain their existing validation. All publications are atomic and immutable.
Old scene-refinement v19 banks, preparation/coordinate v3 and carrier v4 are not
silently upgraded. Retain files and baselines; prepare/run/materialize into new paths.

Checkpoints contain control corrections, q and all Adam states, warmup-end anchor,
warmup/joint/coefficient progress, sparse selections, permutations/cursors and RNG
states. Save every 200 total updates and phase boundaries. Exact input/config/code/
device identities gate resume. Detached baked GPU fields are transient and rebuilt
after load; joint changes invalidate them. Non-finite loss or gradient leaves the
last complete checkpoint and no final output. Gaussian tensors are checked unchanged
at publication.

Standalone sweep RGB v1 and fixed RGB v2 remain fixed-mode baseline tools. They are
not prerequisites for motion refinement. A 60-to-30 FPS sweep subset selects actual
rows/images/cameras together (Bush: 724 to 362), never just changing an FPS label.
Evaluation still hashes exact PNGs and renders at native resolution on valid support;
save independent metrics/CSV and keep historical results untouched. No catalog is
redirected by implementation or preparation. See COEFFICIENT_FITTING.md for the
scientific equations and REBUILD.md for affected dependencies.
