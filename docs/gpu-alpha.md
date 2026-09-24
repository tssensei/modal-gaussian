# GPU complex view gains

[`motion/observations/alpha.py`](../src/modal_gaussians/motion/observations/alpha.py)
coordinates per-frequency complex gain estimation. `alpha_gpu.py` and `_alpha_trf.py`
implement the CuPy solver. There is one backend and failures propagate to the caller.
These gains align spatial observations from independent recordings; they do not
synchronize frames or constrain the later RGB coefficient time series.

Reference-view alpha is one. Other views have bounded gains and phases. Grouped
geometry SVD, profiled 3D solves, block-Huber residuals, two-point differences and
bounded trust-region steps use float64/complex128. Final fields retain their
published dtypes. Rank, information, shared-point and bound checks decide which
views are identifiable; excluded views do not supervise the spatial mode.

The optimizer adapts SciPy's dense exact-SVD TRF algorithm. Its license is retained
in [`licenses/scipy.txt`](../licenses/scipy.txt). Defaults are 1e-8 tolerances and
500 function evaluations. Python reads scalar iteration decisions; solver arrays
stay on CUDA. Numerical differences count separately from function evaluations.

The `alpha_geometry` cache binds topology, view order/subset, precision, thresholds
and implementation revision. Row grouping, fixed Jacobians and decomposition
factors are shared across frequencies. Observations, RHS, informative constraints,
Huber scale and optimized gains are recomputed. Allocation failures reduce chunk
size while retaining precision and all observations; a single block that cannot
fit raises an error. The workspace reserves 2 GiB.

A batch's resident GPU worker prepares alpha for all frequencies while CPU graph
construction can overlap. It then releases alpha state, computes frequency-specific
control weights, exits, and only then permits GNN training. Zero GPU workers pauses
new launches. `--stage weights` stops before GNN training. See
[GPU propagation](gpu-soft-propagation.md) and [recovery](../BASELINE.md#resume-boundaries).
