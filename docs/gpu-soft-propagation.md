# Mainline float64 GPU soft propagation

CuPy is the default on `motion prepare-control-weights`, `motion iterate-neural`,
and `motion batch-neural`; `--propagation-backend cupy` is optional. No CPU fallback
is performed on a CuPy error. The old CPU search lives in
`motion/legacy/neural/control_propagation.py`. Explicit `--propagation-backend cpu`
is retained for historical work/comparison only. Reading old artifacts and Viewer
use do not require a CuPy workspace. Install the GPU extra for mainline training.

## Installation (Anaconda Prompt, activated environment)

```bat
python -m pip install "cupy-cuda12x>=14.2"
```

This implementation uses CuPy RawKernel functions from one NVRTC RawModule;
it does not require a C++ extension build, cuGraph or WSL. CUDA Toolkit 12.8
and CuPy 14.2.0 were used for the Bush comparison. See the official
[installation instructions](https://docs.cupy.dev/en/stable/install.html) and
[RawKernel interface](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html).
The equivalent package extra is `pip install -e ".[gpu-propagation]"`.

## Computation and memory

Only shortest paths move to the GPU. Geometry, control layout, saved support
ordering, double-precision attenuation/Wendland normalization, control edge
weights, donor and GNN behavior are unchanged. The GPU searches the entire host
graph within each control's safe distance bound; paths can leave the support
domain. First arrival at a target is not a stopping condition. Every frontier
must be exhausted; invalid/unreachable required targets raise an error before
publication.

The workspace uses exactly `28 * host_nodes * control_batch` bytes for distances,
two uint64 queues and uint32 deduplication marks, plus CSR/costs/support extraction
and counters. It tries all controls, reserving 2 GiB of currently free memory.
Only insufficient memory or allocation failure reduces the control batch.
Each batch still searches the complete host graph in float64. Kernel launches
process at most 2^20 frontier items. No full distance matrix is saved.

Backend and kernel-source revision enter frequency-weight, upper-control and
iteration identities. Existing caches and artifact identities are not rewritten.
Shared control geometry is independent of the backend. Following a source revision,
import the existing layout once instead of resampling it:

```bat
modal-gaussians storage run --scene bush -- motion prepare-shared-controls --prepared @prepared --geometry-graph @candidate_graph --controls-from @control_geometry --config configs/neural_component_field.json
```

## Two-stage batch execution

The following command is an example for a **new full uniform60 experiment**;
it is not executed by the benchmark. It does not resume the stopped batch.
Existing flow, FFT exports, candidate graph and imported controls are reused;
normal per-frequency preparation/graph stages run or hit their exact caches.

```bat
modal-gaussians storage run --scene bush -- motion batch-neural --modal-images @modal_images --prepared @prepared --geometry-graph @candidate_graph --config configs/neural_component_field.json --output @experiments/uniform60_cupy_001 --propagation-backend cupy --cpu-workers 3 --gpu-workers 2 --stage modes
```

CPU workers prepare future frequencies while one resident GPU subprocess computes
weights in ready-job order. All required weight caches must finish before this
subprocess exits and the GNN queue starts. GPU weights never share a GPU phase
with GNN training. `gpu_workers` specifies normal training concurrency; any
positive value enables only **one** propagation worker. Setting it to zero in
`batch_workers.json` finishes the current frequency and pauses further launches.
`propagation_workers` is unused and reported as null on the mainline. Its flag and
live setting apply only when explicitly selecting the legacy CPU backend.

To prepare weights only, change `--stage modes` to `--stage weights`. Later run
the same command/output directory with `--stage modes`; the completed caches are
reused. Interrupted work also resumes in the same output directory. Different
configuration/backend/code requires a new experiment. Weights-only execution
does not accept training checkpoint continuation or cross-batch result resume.

Monitor `batch_state.json`, `gpu_weights.log`, `propagation_status.json`, and
`gpu_usage.csv` in that output directory. Completed entries are atomically
published. A failed or interrupted in-progress search is recomputed, never
presented as a completed cache. The scheduler manages only its own children.

## Explicit benchmark

```bat
python scripts/benchmark_soft_propagation.py --output scene_library/bush/experiments/gpu_soft_propagation_002
```

Choose a fresh numbered directory. This command runs only the three authorized
Bush frequencies (0.25, 5.0, 10.25 Hz): one exclusive CPU reference with four
processes per frequency, then one full-size GPU warmup and three measured GPU
runs per frequency. Every saved support distance and final interpolation/control
edge weight is compared at rtol=1e-9, atol=1e-11. It neither builds modal graphs
nor starts GNN. `report.json` separates search, allocation, upload, initialization,
download, CPU weight composition and publication, with whole-device memory
samples in `gpu_usage.csv`. Benchmark diagnostics contain support distances only.

Local development checks (ignored under the existing `/tests/` rule):

```bat
python -m unittest discover -s tests -p test_control_propagation_gpu.py -v
python -m unittest discover -s tests -p test_gpu_weight_batch.py -v
```

The timings are measured wall time, including frontier counter synchronization;
they are not kernel-event-only estimates. No speedup is guaranteed for a different
scene or graph. Cold initialization/compilation and warm steady-state timings
must be reported separately.
