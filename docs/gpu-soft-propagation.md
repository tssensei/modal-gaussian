# GPU soft propagation

[`control_propagation_gpu.py`](../src/modal_gaussians/motion/control_propagation_gpu.py)
uses CuPy RawKernel/NVRTC for float64 shortest paths. CuPy is a main dependency;
there is no CPU backend or automatic fallback. SciPy remains useful for CPU graph
construction and the independent synthetic-test oracle.

Shared geometry caches hold component/control order and geometric support distances.
Each frequency's modal graph changes edge costs and produces separate control
weights. Soft costs attenuate Wendland interpolation without changing the control
layout. Donor eligibility is computed from that frequency's observations later.

Each source searches its whole host graph within its safe distance bound. Paths
may leave the support region; finding one target does not terminate the frontier.
Unreachable or invalid required targets fail before publication. The workspace
uses `28 * host_nodes * control_batch` bytes for distance/frontier/mark buffers,
plus graph, extraction and counters. It reserves 2 GiB and reduces batch size on
allocation failure, preserving full-graph searches and precision. No full distance
matrix is published.

Cache contracts include geometry/order, graph costs, configuration and kernel
revision. Reuse only exact matches. A frequency's weights cannot serve another
frequency. Current shared geometry is reused automatically; no import/migration
command is needed.

The batch has three phases: GPU alpha with overlapping CPU graph construction,
GPU weights, then GNN training after the resident preparation process exits.
`batch_workers.json` controls CPU/GPU concurrency between launches; logs and
request/status files remain in the experiment. See [GPU alpha](gpu-alpha.md) and
[command recipes](../skills/modal-gaussians-pipeline/references/commands.md).
