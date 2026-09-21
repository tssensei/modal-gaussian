# Scene library

The active storage root is `scene_library/`, grouped by scene. It is local data
and is ignored by Git. `registry.json` is required: it maps original
identity-bearing paths to their new physical locations without changing saved
manifests, cache contracts or model identities.

| Directory within each scene | Contents |
| --- | --- |
| `references/` | Reference/timing metadata and stabilized frames; no old Farneback arrays |
| `experiments/reference_flow_*/flow/` (Bush), `experiments/reference_modes_*/flow/` (Corn) | Current new-reference SEA-RAFT flow; resolve via `flow1`, etc. |
| Current input experiment `spectrum/` | New-reference shared FFT; resolve via `spectrum` |
| Current input experiment `selection.json`, `modal_images/` | New-reference frequency choices and complex U/V slices |
| `geometry/` | Static scene, prepared observations, manual subject selection and required ancestors |
| Current prepared experiment `graph/` | Frequency-specific soft graph |
| `cache/` | Content-addressed KNN geometry, control layouts, per-frequency interpolation and controls |
| `results/models/` | Published 3D mode artifacts, with original identities |
| `checkpoints/` | Independently copied optimizer/RNG checkpoints and fixed inputs |
| `experiments/` | New-reference inputs, comparisons, previews, logs and future experiments |

`_shared/tools/` contains SEA-RAFT code/weights. `_shared/history/` retains the
storage audit and small historical scripts. Raw input images under `data/`
remain external inputs and must be retained.

## Find data before starting another experiment

New flow datasets follow **static Gaussian → motion-reference selection and
overlays → SEA-RAFT → shared FFT → modal preparation/training**. Use
`flow select-reference`, review its candidate, then pass `--reference-selection`
to `flow compute`. Preserve the original geometry reference, camera/grid and time
origin; do not edit old manifests or reinterpret old flow/FFT as new-reference
data. Existing completed frequency work still reuses its original caches.

The 2026-09-20 default recipe is **new motion references + per-view RMS +
deformation weight 0.03**. See [the current policy and reusable input paths](BASELINE.md#current-experiment-policy-2026-09-20).
After the user-authorized 2026-09-20 cleanup, `@spectrum`, `@prepared`, `@graph`
and `@flow1` etc. point to the new-reference inputs. `@geometry_prepared` retains
the original fixed geometry snapshot needed by preparation. Old `@baseline`,
`@batch40`, `@uniform60` and `@greedy60` aliases were removed. Preview aliases
are explicit (`@preview_absolute`, `@preview_absolute_deform0p3`, and Corn
`@preview_rms`). Bush now has the completed new-reference RMS uniform-20 batch;
`@uniform20_batch` and `@uniform20_selection` locate its records.

Corn's 2026-09-20 reviewed motion references are view1 **00103 / 5.10 s** and view2
**00078 / 3.85 s**. Their selection contracts are indexed as `motion_reference1`
and `motion_reference2`; the original comparison images remain under
`corn/experiments/reference_selection_20260920/`. The green silhouette recipe is
selection-only and does not restore fine-mask training. See [README.md](README.md)
for commands, non-green subjects and cache invalidation rules.

From the repository in Anaconda Prompt:

```bat
modal-gaussians storage list --scene bush
modal-gaussians storage path --scene bush --asset candidate_graph
modal-gaussians storage path --scene corn --asset mode:0.225
modal-gaussians storage path --scene corn --asset prepared
```

`catalog.json` lists retained cache contracts and new-reference results.
`results/index.json` lists two Bush comparisons, twenty newly completed Bush
RMS modes and three Corn comparisons, including normalization and deformation
settings. `mode:0.225` resolves the Corn RMS model; Bush `mode:<frequency>`
resolves the new RMS batch, while comparisons use explicit preview aliases.
The old 40-frequency batch, baselines and coefficient fit have been deleted. Bush's new FFT grid is 0.024 Hz (Nfft 1250 at 30 fps), so
new frequency selections must use that grid, not the deleted 0.0125 Hz grid.

`storage run` expands explicit `@asset` references and delegates to the existing
CLI. It does not create a second pipeline or automatically launch work.

```bat
modal-gaussians storage run --scene corn -- spectrum viewer --input @spectrum --work-dir @experiments/spectrum_viewer --host 127.0.0.1 --port 8110
modal-gaussians storage run --scene bush -- viewer --preview @preview_absolute --work-dir @experiments/baseline_viewer --host 127.0.0.1 --port 8091
```

For new frequency work, first reuse `@spectrum` and its exported slices, then
`@prepared` and `@candidate_graph`. Reuse a matching existing frequency graph
when its scene/frequency/configuration match; a baseline soft graph is not
generic across frequencies. Put new outputs under `@experiments/<new_name>`.
Prepared training automatically uses the registered scene's cache directory;
published modes and mutable checkpoints go to their respective directories.
Existing cache contract checks remain in force. The imported result index is a
curated snapshot, not permission to reuse a mode based on frequency alone.

Original absolute references are resolved by the supported artifact readers.
Direct third-party scripts that open raw strings from old JSON must use
`modal_gaussians.scene_store.resolve_path` or take paths from this registry.
Changing the library root is supported through `MODAL_GAUSSIANS_LIBRARY` for
the imported snapshot; keep `registry.json` with the library.

## Current cleanup (2026-09-20)

Deleted 217 old-reference experiment, model/checkpoint and cache directories,
including old full flow/FFT, Bush 40-frequency results and coefficient fitting.
Removed logical payload: **206.98 GiB**. Kept new-reference flow/FFT, five
models/checkpoints, raw `data/`, geometry/reference snapshots, shared tools
and matching controls. Original manifests and identity mappings were not rewritten.
Some retained geometry snapshots contain historical targets/provenance; they
are preparation dependencies, not active old-reference experiments.

Audit and old index snapshots:
`scene_library/_shared/history/cleanup_new_references_20260920/`.
Only metadata/path checks were run, with no training or experiment validation.

## Historical import and old-output cleanup (superseded by the cleanup above)

The 2026-09-19 import preserved **47,541 files / 221.025 GiB**:

| Scene | Logical size |
| --- | ---: |
| Bush | 184.118 GiB |
| Corn | 36.818 GiB |
| Shared tooling/history | 0.089 GiB |

Same-volume hard links hold immutable arrays, images and model payloads.
They are independent directory entries: deleting the old entry leaves the new
one valid. They share underlying bytes, so never edit these payloads in place.
Mutable checkpoints and metadata were copied separately. Added disk usage is
about **6.10 GiB**, rather than another 221 GiB. Hard links are not a backup
against disk failure.

On 2026-09-19, the user authorized cleanup. All **90,702 reviewed files** in
the old `outputs/` tree were deleted, releasing approximately **79.42 GiB** of
disk space. `outputs/` is now empty. `scene_library/` and raw `data/` were kept;
both accepted baselines, all 40 completed Bush modes and their required metadata
references remained present at that time. The execution record is
`scene_library/cleanup_execution_20260919.json`.

`scene_library/old_outputs_classification.csv` records the deleted entries:

- `relocated_copy_or_link`: retained content now present in the scene library;
  redundant old entries have been removed.
- `previously_reviewed_legacy`: obsolete experiments/caches from the earlier
  cleanup analysis, about **73.19 GiB**, not imported.
- `unreviewed_do_not_delete`: new/unclassified files, if any. None at import time.

`import_plan.json` records every transfer and whether it was linked or copied;
`migration_accounting.json` records filename/size and indexed-result-path checks.
At import time, both baselines and all 40 completed Bush frequencies had their
model, prepared, graph and checkpoint directories present. This is storage accounting, not
numerical or visual validation. No flow, FFT, graph or training was rerun.

Storage-only code edits change the broad training source revision. Old results
retain their original identities; only retained artifacts remain loadable. Do not rewrite
their keys or bypass a revision guard to resume an old experiment in place;
use the existing explicit continuation/new-attempt mechanisms when requested.
The shared geometry/control algorithms and their code keys were left intact.

The one-time importer is `scripts/organize_scene_library.py`; it consumes the
reviewed retention analysis and metadata inventory, publishes only after import,
and never deletes sources. New experiments should use the library directly,
rather than repeat the import.

## Coefficient fitting intermediates

Fixed-mode reconstruction uses a new scene experiment with `mode_bank/`,
`rendered_design/`, `direct_coordinates/`, and later `rgb_coordinates/`.
`preparation.json` binds the sources/configuration and records completed stages.
These are derived inputs and coefficients, not replacement trained modes or
training checkpoints. Original models, graphs and checkpoints remain immutable.
See [COEFFICIENT_FITTING.md](COEFFICIENT_FITTING.md) for formats and commands.

Viewer input is independent of these coefficient-fitting intermediates. Use
`viewer --input <model-or-results_index.json> --work-dir <viewer-directory>` to
open one or many modes directly. Existing previews/results remain readable.
Missing Spectrum projections are cached per model slot and view in
`cache/viewer_projection/<hash>/`, using geometry, camera, sampling and projection
implementation identities. No aggregated model arrays or combined preview are
written. For an unregistered scene the cache lives under `work-dir/cache/`.
