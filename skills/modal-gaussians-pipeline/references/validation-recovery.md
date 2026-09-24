# Recovery and verification boundaries

Do not automatically run real-scene validation, metric reports or visual checks.
Normal input validation, cache-contract checks and publication checks remain part
of executing a stage. Synthetic development checks are separate from experiment
validation; do not present them as evidence of reconstruction quality.

## Before an authorized run

Inspect the scene catalog/index and exact requested endpoint. Confirm source
paths, FPS, image/mask dimensions, static camera/Gaussian order, FFT bins and free
resources. Use the existing CUDA-capable environment, local SEA-RAFT weights,
COLMAP and FFmpeg. Do not install or downgrade packages speculatively.

Flow storage is at least `8*T*H*W` bytes per view; a shared complex RFFT is
`16*(NFFT/2+1)*H*W` bytes. Include temporary arrays and atomic-publication headroom.
Use an explicit scene cache/output path. Do not reduce inputs to fit resources
without a scientific decision from the user.

## Resume

| Interrupted work | Recovery |
| --- | --- |
| Batch, unchanged code and contract | Rerun the same command/output; completed stages are reused. |
| Neural optimization | The matching work directory restores model, Adam, RNG and patience state. |
| Only increase iteration cap | New output plus `--continue-from OLD_EXPERIMENT` or `OLD_BATCH`. |
| Reuse completed modes in a new matching batch | `--resume-from OLD_BATCH`. |
| Coefficient preparation | Same inputs/output with `--resume`. |
| RGB coefficient optimization | No optimizer checkpoint; use a new output. |
| Changed schema, source, reference or output-affecting code | Rebuild the affected suffix into new paths. |

Preserve failure logs and checkpoints. Diagnose the actual exception or native
error, make a bounded repair, and retry using compatible ancestors. A runtime
error during an authorized run is not itself a reason to request permission again.
Do not repeatedly rerun a deterministic failure without new evidence. Never edit
identities, bypass mismatch checks, or mark an incomplete directory complete.

[REBUILD.md](../../../REBUILD.md) records the cleanup's required rebuilds. Current
resume logic does not migrate pre-cleanup artifacts. Keep catalogs unchanged until
new outputs finish and the task calls for updating their pointers.

## Report

Provide the completed output/index path, frequencies and already recorded timing.
State actual failures and limitations. Mode publication does not imply viewer
inspection, metric evaluation or user acceptance. Stop at the requested endpoint.
