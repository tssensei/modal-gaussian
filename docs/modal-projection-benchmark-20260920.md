# Fixed modal projection comparison, 2026-09-20

The user authorized a short real-scene comparison after adding the optional
`neural.modal_projection_backend="cached"` implementation. This benchmark uses
existing Bush 0.25 Hz geometry, prepared observations and a saved mode shape;
it does not run the GNN, update parameters, regenerate upstream data, publish
models/checkpoints, export images or resume the stopped batch.

## Input and method

- RTX 5090, 231,761 foreground Gaussians, three 1920 x 1080 reference cameras.
- 205,381 / 158,059 / 124,070 supervision pixels; all three views identifiable.
- Each camera uses the existing radial coefficient `-0.0002737025019895056`.
- Prepared identity: `46ecea55d991ff33078d4465ce759c332bd4c5f4f04cdf4fbcb27d06d114edca`.
- Same saved complex64 mode field, Jacobians, targets, view gains, confidence
  normalization and radial-Huber image objective for both backends.
- View-by-view backward to the field leaf matches the image-supervision part of
  training. GNN forward/backward, field composition, regularizers, optimizer,
  early stopping and checkpoints are excluded.
- Each backend has its first call and three additional warmup steps, then 60
  measured steps in six blocks of ten. Backend order alternates between blocks.
- Wall time synchronizes CUDA before/after each complete three-view step. CUDA
  events additionally delimit forward/loss and backward within each view.
  Event intervals can include device idle time while Python prepares launches.

## Results

| Work per three-view step | Dynamic median | Cached median | Ratio |
| --- | ---: | ---: | ---: |
| Forward plus image loss, CUDA events | 48.580 ms | 3.956 ms | 12.28x |
| Backward to field, CUDA events | 8.988 ms | 7.825 ms | 1.15x |
| **Complete step, wall time** | **57.659 ms** | **12.023 ms** | **4.80x** |

Wall-time means were 57.752 / 12.093 ms. The median reduction is 79.15%, or
45.637 ms per three-view step. The cached path reuses projection, tile sorting
and the radial sampling grid, and warps only the selected supervision pixels.
Pixel compositing and feature backward still use gsplat. This comparison does
not isolate how much each cached operation contributes.

Cache construction after renderer warmup took **0.0989 s** for all three cameras.
Additional live Torch allocation was **59,627,008 bytes (56.9 MiB)**. At the measured
per-step saving, construction amortizes after approximately three steps. Raw peak
memory records contain both projector sets in one process and must not be treated
as separate-backend total memory measurements.

Common scene/observation loading took 0.356 s. The first dynamic call took 2.712 s,
including lazy renderer initialization; the first cached call after construction
took 14.426 ms. Neither first-call value is part of the steady-state ratio, and
the latter is not an independent cold-process comparison.

In the limited comparison included in this benchmark, predicted values matched
exactly and the maximum absolute field-gradient difference was `1.4317e-9`.
The checks used `rtol=2e-5, atol=2e-6`. This is not a visual/scientific acceptance.

Extrapolating this isolated segment to 5,000 identical steps saves about **228 s**,
including cache construction. This is not a measured full-training speedup;
other training work and changes in machine load remain outside the comparison.

Reproduction script, raw timing samples and log are local experiment data:
`scene_library/bush/experiments/modal_projection_benchmark_20260920/`.
The existing models, baselines, prepared arrays and stopped batch were unchanged.
