# GPU alpha synchronization

New `motion prepare-selected-modal` and `motion batch-neural` commands default
to `--alpha-backend cupy`. Install the existing `gpu-propagation` extra; no
additional solver dependency is needed. `--alpha-backend cpu` explicitly selects
the retained SciPy reference. A GPU initialization/allocation/numerical failure
does not fall back to CPU. Historical rigid synchronization explicitly uses CPU.
Reading existing prepared/model artifacts does not initialize CuPy.

## Computation and numerical contract

The GPU implementation covers grouped geometry SVD, observation constraints,
per-Gaussian profiled 3D solves, normalized block-Huber residuals, batched two-point
finite differences, and the outer bounded trust-region-reflective optimizer.
All solver arrays use float64/complex128. Output dtypes remain unchanged.

`_alpha_trf.py` adapts the dense exact-SVD branch of SciPy 1.17.1's bounded TRF.
Its BSD license is included in `licenses/scipy.txt` and distribution metadata.
It preserves finite-difference bound adjustment, strictly feasible iterates,
Coleman–Li scaling, reflected steps, trust-radius updates, active-bound reporting,
unit parameter scale, tolerances of 1e-8 and max_nfev=500. Numerical-difference
calls are separate from that evaluation limit. The block-Huber transformation is
applied once, inside the residual; outer least squares has linear loss.

Residuals, Jacobians and SVD/eigh arrays stay on CUDA. Python controls iteration
and reads scalar decisions. Geometry cache publication downloads fixed arrays;
final view decisions and diagnostics download small results. CPU/GPU arithmetic
need not produce bitwise-identical iterates, SVD bases or termination iterations.
Reference alpha, gain bounds, rank cutoffs and exclusion rules remain unchanged.

## Geometry cache and identities

Geometry is cached under `scene_library/<scene>/cache/alpha_geometry/<hash>/`
using the existing atomic cache mechanism. Contracts include original topology
identity, view order/subset, weight convention, precision, decomposition thresholds
and implementation revision. Fixed row grouping, Jacobians/weights, nullspace
factors and per-view Gram matrices are reusable across frequencies. Observations,
RHS, normalization, informative constraints, Huber scale and final alpha are not.
Each encountered view subset receives its own decomposition.

A resident workspace retains only the current geometry and its used subsets.
Gaussian and parameter batches respect a 2 GiB reserve; temporary allocation
failure halves the batch. If a single block or required resident solver arrays
cannot fit, preparation fails explicitly. There is no precision reduction or
observation dropping. The first construction still groups variable-size rows
on CPU; expensive factorizations are on GPU.

New alignment/prepared and batch contracts include `alpha_backend` with algorithm
and source revision. Missing metadata means historical CPU. Old identities are
never rewritten. A new GPU attempt cannot silently claim a CPU prepared artifact
as its own. Explicit checkpoint continuation retains its prepared alpha provenance,
which is also recorded per job in `batch_state.json`.

## Batch lifecycle

One resident GPU process handles two request operations using the existing
`propagation_request.json` / `propagation_status.json` protocol:

1. Prepare each frequency's alpha/observations on GPU; CPU graph construction can overlap.
2. After all required prepared artifacts exist, close the alpha workspace and
   construct the propagation workspace. Compute frequency weights as graphs become ready.
3. After all required weights exist, exit the worker and release its CUDA context;
   then allow the configured GNN training concurrency.

GPU preparation uses one worker regardless of positive `gpu_workers`; zero pauses
new launches after active work finishes. CPU graph slots retain their existing
meaning. `--stage weights` stops before GNN; a later identical `--stage modes`
request reuses published inputs. Failure stops new launches and keeps completed
artifacts. Worker logs remain `gpu_weights.log`; status includes the operation.
The legacy CPU alpha/propagation switches preserve their explicit compatibility
paths and do not enable an automatic fallback.

New experiments belong under the scene's `experiments/<new_name>/`. Resolve saved
historical paths through the registry. Do not resume the stopped Bush batch merely
because defaults changed.

## Development checks and timing

Initial development used synthetic arrays, mocked scheduler jobs and temporary
artifacts. The separately authorized real-input timing below subsequently tested
one Bush frequency. No flow/FFT/graph rebuild, training, preview or stopped-batch
restart was performed.

```bat
python -m unittest discover -s tests -p test_alpha_gpu.py -v
python -m unittest discover -s tests -p test_gpu_weight_batch.py -v
```

The two focused test files are tracked explicitly despite the general `/tests/`
ignore rule. They cover
cached/subset geometry, CPU/GPU residual and alpha agreement, finite differences,
bounds, outliers, degeneracy, allocation retry, no CPU fallback, old metadata,
phase barriers, pause, failure and resume. The CPU pipeline and selected-modal
synthetic checks also remain applicable.

The focused run passed 28 checks. A broader run also reached two pre-existing
`test_selected_modal_spectrum.py` failures: those tests patch the removed
Viewer symbols `_completed_dense_modes` and `load_entry`. The symbols are absent
in the unchanged HEAD implementation too; this upgrade does not change that Viewer.

Each new prepared artifact has `alpha_timings.json`: cold workspace initialization,
geometry/cache work, upload, constraints, residual/derivative work, outer TRF,
diagnostics, download, cache/prepared publication, allocation retries and sampled
memory peaks. TRF includes residual time; nested times must not be added together.
Cold kernel compilation can also appear in the first operation using that kernel.
GPU stage timing synchronizes at its boundaries and uses wall time.

## Authorized Bush timing, 2026-09-20

A single 0.25 Hz frequency was measured on the RTX 5090 using all three views,
231,761 Gaussians, 120,267 sampled pixels and 465,419 contributor rows. Both
backends consumed the same existing topology and modal images, with the saved
alpha configuration and float64/complex128 computation. Source prepared identity:
`46ecea55d991ff33078d4465ce759c332bd4c5f4f04cdf4fbcb27d06d114edca`.

| Alpha stage | Wall time | CPU time / GPU time |
| --- | ---: | ---: |
| CPU reference | 11.240 s | 1.00 |
| GPU first geometry build/publication | 58.854 s | 0.19 (slower) |
| GPU geometry reloaded from disk | 8.904 s | 1.26 |
| GPU geometry retained in workspace | 6.959 s | 1.62 |

Times include observation preparation and the complete alpha solve. Common input
loading took 0.150 s and is excluded; GPU workspace initialization took another
0.065 s (first-run total including that initialization: 58.919 s). Existing CuPy
kernel disk cache was retained. Disk-cache and resident runs used the same process
and warm CUDA libraries; disk-cache timing is not a fresh-process startup test.
Each condition ran once, in table order; these are quick measurements, not medians
or a claim about all frequencies or complete prepared publication/training.

First-run geometry loading/building took 42.969 s, including 16.995 s of cache
publication. Resident TRF took 6.256 s, including 1.462 s in residual/derivative
evaluation. The remaining TRF work and scalar control overhead merit profiling;
this test does not attribute that difference to a single operation. Warm sampled
memory peaks were 6.18 GiB in the CuPy pool and 8.09 GiB device-wide; the latter
includes other applications. Nested stage times must not be summed.

All three GPU runs matched CPU alpha within `rtol=1e-5, atol=1e-7`; maximum absolute
difference was `1.1921e-7`. Identifiable masks, exclusion reasons, gain-bound masks
and optimizer termination status matched (successful status 2). This is a limited
numerical comparison, not scientific/visual acceptance. The 19 focused synthetic
checks above also passed again before commit.

Reproduction script, raw timings and logs are local, excluded experiment data:
`scene_library/bush/experiments/alpha_gpu_benchmark_20260920/`. The script explicitly
solves alpha only and writes no prepared/model/checkpoint replacements. Existing
baselines, completed frequencies and the stopped batch were left unchanged.
