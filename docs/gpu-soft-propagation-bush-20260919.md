# Bush float64 GPU soft-propagation comparison — 2026-09-19

The CuPy backend passed the three-frequency comparison. CPU was the default at
measurement time; the user subsequently promoted GPU propagation to the mainline.
Neither a GNN nor the stopped Bush batch was started. No flow, FFT,
modal graph, control sampling or geometric shortest paths were recomputed.

## Environment and scope

- Native Windows, Python 3.11.16, RTX 5090 (compute capability 12.0).
- CuPy 14.2.0, NumPy 2.4.6, SciPy 1.17.1; `cupy-cuda12x` was installed in the
  existing `modal-gaussian` environment.
- Installed Toolkit and actual NVRTC: **12.8**. CuPy reports its linked/build
  runtime as 12.9 and the local runtime as 12.8; no Toolkit upgrade was made.
  CUDA driver API reports 13.3.
- 3,428 controls, 231,119 host Gaussians, 2,300,540 original support entries per
  frequency. All numerical distances, costs, attenuation and weights are float64.
- Existing Bush 0.25, 5.0 and 10.25 Hz graphs and saved shared support distances.
  CPU: four spawned propagation processes, one frequency at a time. GPU: one
  resident workspace, one full-size warmup, three measured runs per frequency.
- Whole-device memory sampled approximately once per second. Ordinary small
  development checks and code editing ran during the CPU reference phase;
  no other scene job or GPU benchmark overlapped those CPU searches.

## Measured wall times

| Frequency | CPU search | GPU search median | Search speedup | CPU complete soft propagation | GPU complete soft propagation median | Soft-propagation speedup |
|---|---:|---:|---:|---:|---:|---:|
| 0.25 Hz | 244.063 s | 0.449 s | 543.4× | 246.387 s | 2.652 s | 92.9× |
| 5.0 Hz | 249.266 s | 0.131 s | 1901.2× | 251.529 s | 2.406 s | 104.6× |
| 10.25 Hz | 240.378 s | 0.107 s | 2250.2× | 242.247 s | 2.526 s | 95.9× |

Search time is wall time, including GPU frontier-counter reads and wave
synchronization. CPU search includes pool setup/teardown and result collection.
The complete soft-propagation measurement includes input checks, transfers,
distance initialization/search/extraction, and the unchanged CPU attenuation,
Wendland normalization and control-edge aggregation. Numerical comparison time
is excluded. It does not include loading graph files or writing the final cache.

Adding the separately measured graph-loading and canonical-cache publication
stages gives the following **stage sums**, not a claim about complete pipeline
or GNN training time:

| Frequency | CPU load + propagation + publish | GPU load + propagation + publish | Ratio |
|---|---:|---:|---:|
| 0.25 Hz | 249.262 s | 5.509 s | 45.25× |
| 5.0 Hz | 254.808 s | 5.675 s | 44.90× |
| 10.25 Hz | 245.543 s | 5.773 s | 42.53× |

The same measured load time is included on both sides. Canonical CPU/GPU cache
publication was measured separately after numerical comparison, approximately
0.96–1.01 s per cache. Shared layout loading (0.102 s), one-time imports and
dependency/kernel initialization are reported separately in the raw records.
These sums exclude metadata-ready publication and the synthetic checks.

Typical GPU phase times were upload 0.0022–0.0025 s, initialization
0.0065–0.0154 s, and support extraction/download 0.0168–0.0198 s. CPU weight
composition took 1.87–2.41 s and now dominates this operation. The large search
speedups apply specifically to the existing adaptive SciPy implementation,
including its repeated whole-graph searches; they are not a comparison against
every possible CPU shortest-path implementation.

## Correctness and GPU capacity

All original support distances and both final weight arrays were compared for
every measured GPU run against the matching CPU reference at `rtol=1e-9`,
`atol=1e-11`. **Maximum absolute distance error and interpolation-weight error
were zero in all nine runs.** Control-edge weights also passed. This does not
promise bitwise equality for all future graphs.

Every run used one batch containing all 3,428 controls. The four large buffers
allocated **22,183,726,096 bytes = 20.660 GiB**. No allocation retry or reduced
control batch was needed. Whole-device sampled memory peaked at
**26,462 MiB = 25.842 GiB**, including unrelated display/application allocations.

| Frequency | Propagation waves over three runs | Processed active items per run |
|---|---:|---:|
| 0.25 Hz | 210–211 | 182.86–183.48 million |
| 5.0 Hz | 141–145 | 41.98–42.12 million |
| 10.25 Hz | 152–154 | 37.52–37.62 million |

Different scheduling can change the intermediate frontiers and wave count;
queue exhaustion yielded the same final distances here. No dense control-by-node
distance matrix or predecessor array was written to disk.

The first benchmark workspace took 0.294 s with the existing CuPy compile cache.
A separate fresh disk-cache compilation took **0.459 s including CuPy/workspace
initialization**, with **0.167 s for module compilation/loading**. NVRTC reported
12.8 in that fresh-cache measurement. These times are excluded from warm medians.

## Cache integration and tests

The existing shared layout and support distances were imported with the existing
`prepare-shared-controls` mechanism under current geometry code identity
`b66549a870de865a34e36bad5d5649453ae8251b55cdc44e8d2a2a7e49423ffc`.
No controls were added. Measured GPU arrays were published under new, distinct
backend/kernel identities in `scene_library/bush/cache/control_weights`.
The normal weight-preparation entry then hit the geometry and frequency-weight
caches for all three frequencies and published ready markers; it did not run
shortest paths again or start training.

Four new numerical GPU checks passed: exact support distances and batching,
competing frontier updates, invalid/unreachable inputs, and normalized weights
with subsequent training-cache lookup. Five scheduler/protocol checks passed:
all-weights barrier, weights-only termination, zero-worker pause, worker failure
without training, and resident request/shutdown. Existing CPU propagation,
shared-control, preparation and batch checks also passed. Tests remain ignored
by the existing `/tests/` rule.

## Artifacts and use

Raw measurements, support-only diagnostics, GPU memory samples, new comparison
caches, cold-compilation record and ready markers:

`scene_library/bush/experiments/gpu_soft_propagation_001/`

Key files: `report.json`, `cold_compile.json`, `gpu_usage.csv`,
`ready_0.25.json`, `ready_5.json`, `ready_10.25.json`.
`report.json` lists the immutable source graphs and canonical GPU cache paths.

See [installation and batch commands](gpu-soft-propagation.md). CuPy now defaults
on; `--propagation-backend cupy` remains accepted. `--stage weights` prepares all weights;
the same batch with `--stage modes` subsequently enables GNN after the workspace
process exits. The benchmark did not commit, push, change the accepted baseline,
resume old training, or extend to the other frequencies.
