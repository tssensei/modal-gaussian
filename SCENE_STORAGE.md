# Scene library

The active storage root is `scene_library/`, grouped by scene. It is local data
and is ignored by Git. `registry.json` is required: it maps original
identity-bearing paths to their new physical locations without changing saved
manifests, cache contracts or model identities.

| Directory within each scene | Contents |
| --- | --- |
| `references/` | Reference/timing metadata and stabilized frames; no old Farneback arrays |
| `flow/` | Full SEA-RAFT flow and existing single-frequency modal images |
| `spectrum/main/` | Complete shared-grid complex FFT cache, including background |
| `selections/`, `modal_images/` | Saved frequency choices and exported complex U/V slices |
| `geometry/` | Static scene, prepared observations, manual subject selection and required ancestors |
| `graphs/` | Accepted frequency-specific soft graphs |
| `cache/` | Content-addressed KNN geometry, control layouts, per-frequency interpolation and controls |
| `results/models/` | Published 3D mode artifacts, with original identities |
| `checkpoints/` | Independently copied optimizer/RNG checkpoints and fixed inputs |
| `experiments/` | Baseline, batch status/configuration, previews, logs and future experiments |

`_shared/tools/` contains SEA-RAFT code/weights. `_shared/history/` retains the
storage audit and small historical scripts. Raw input images under `data/`
remain external inputs and must be retained.

## Find data before starting another experiment

From the repository in Anaconda Prompt:

```bat
modal-gaussians storage list --scene bush
modal-gaussians storage path --scene bush --asset candidate_graph
modal-gaussians storage path --scene bush --asset mode:0.25
modal-gaussians storage path --scene corn --asset prepared
```

`catalog.json` in each scene lists imported cache contracts. `results/index.json`
lists accepted baselines and the 40 completed Bush frequencies. The Bush batch
remains stopped; unfinished frequencies are not listed as completed results.
Additional older model/checkpoint artifacts are continuation ancestors, retained
under their existing hashes, rather than additional accepted baselines.

`storage run` expands explicit `@asset` references and delegates to the existing
CLI. It does not create a second pipeline or automatically launch work.

```bat
modal-gaussians storage run --scene corn -- spectrum viewer --input @spectrum --work-dir @experiments/spectrum_viewer --host 127.0.0.1 --port 8110
modal-gaussians storage run --scene bush -- viewer --preview @baseline --work-dir @experiments/baseline_viewer --host 127.0.0.1 --port 8091
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

## Import and old-output cleanup

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
references remain present. The execution record is
`scene_library/cleanup_execution_20260919.json`.

`scene_library/old_outputs_classification.csv` records the deleted entries:

- `relocated_copy_or_link`: retained content now present in the scene library;
  redundant old entries have been removed.
- `previously_reviewed_legacy`: obsolete experiments/caches from the earlier
  cleanup analysis, about **73.19 GiB**, not imported.
- `unreviewed_do_not_delete`: new/unclassified files, if any. None at import time.

`import_plan.json` records every transfer and whether it was linked or copied;
`migration_accounting.json` records filename/size and indexed-result-path checks.
Both baselines and all 40 completed Bush frequencies have their model, prepared,
graph and checkpoint directories present. This is storage accounting, not
numerical or visual validation. No flow, FFT, graph or training was rerun.

Storage-only code edits change the broad training source revision. Old results
retain their original identities and remain directly loadable. Do not rewrite
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
