# Adaptive SfM comparison: Phase 2

This phase adds wall-clock control, safe checkpoints and short training checks.
It does not run the 300/900-second comparison, rank reconstruction quality, export
videos or import a frozen Adaptive scene. Phase 1 inputs remain immutable.

## Shared time definition

Use monotonic `time.perf_counter()`, with CUDA synchronization at complete update
boundaries. The training clock starts immediately before sampling the first batch.
It includes sampling/image loading, transfer, loss/backward/optimizer updates,
density statistics, spatial/temporal density changes, culling, scalar logging,
periodic checkpoints and time-milestone checkpoints. It is not CPU process time
or the rasterizer's isolated CUDA event duration.

Initialization (input validation, point initialization and explicit kernel warmup),
final checkpoint saving, optional final reports and scene publication are measured
separately. An outer process timer records command wall time, including startup.
The training limit is not a limit on the entire command's wall time.
Native initialization is retained: Adaptive explicitly warms Taichi kernels;
lazy loading or compilation first encountered during training is charged to the
training clock. Report both training and command time, and control cache state
when running the formal comparison; these short runs are not throughput rankings.

The limit is checked after a complete update and all associated density work.
Every positive budget permits at least one complete update.
Checkpoint writes during training count against the limit, with another check
after writing. One expensive event can overshoot; report requested and actual
seconds. A state captured after 300 seconds is not a strict <=300-second result
merely because its milestone label is 300.

Milestones capture the first complete state crossing each threshold. Metadata
distinguishes capture time, completed updates and save-completion time. Multiple
thresholds crossed during one event may share a state; no intermediate state is
fabricated inside an optimizer or density update.

## Execution interfaces

Modal uses execution arguments outside the scientific config:

```text
static train ... --time-budget-seconds 900 --time-milestones-seconds 300
```

Adaptive keeps its native underscore argument style:

```text
scripts/train.py ... --time_budget_seconds 900 --time_milestones_seconds 300
```

Limits must be positive and finite; milestones are ascending, unique and no
greater than the limit. Time-controlled runs start fresh: budget plus resume is
rejected. Existing non-budget StaticTrainer resume remains available with matching
inputs/code/config. Adaptive snapshots include model and Adam state but omit
sampler/RNG/density recovery state, so they are not exact resumes. Use one
continuous run for multiple budget points.
Adaptive also rejects a nonempty output directory before writing budgeted outputs.

Keep a sufficiently large iteration cap, e.g. 30000, to preserve the intended
LR/density schedule. Time control does not change objectives, sampling,
initialization, Gaussian limits, keyframe thresholds or regularization. Adaptive
now honors the exact iteration cap instead of rounding up to an entire epoch;
the update trajectory before that cap is unchanged by this stopping fix.
For 318 frames and a 30000-update cap, this means 30000 updates instead of the
upstream 30210. Periodic saves now trigger and record completed update counts:
the 7000 checkpoint is captured after 7000 updates rather than 7001.

Budget mode disables all Adaptive image reports, including the former final
held-out report, and rejects GUI pauses. Non-budget default reporting is retained.
Training and evaluation remain separate stages.

## Storage and recovery

Each trainer writes `gaussian_training_timing` v1 `training_timing.json`: in the
Static work directory or Adaptive model directory. It records actual updates,
stop reason, budget/overshoot, phase timings, density events, Gaussian counts,
peak CUDA allocation/reservation and milestones. The aggregate training clock
also includes unclassified bookkeeping; component timings need not sum exactly.

Static keeps the existing complete resume layout and publishes scene v5 at the
actual stopping step. A partial epoch retains its sampler cursor and accumulator
in the checkpoint; publishing a partial summary must not reset them. Final
publication timing is written outside the immutable scene bundle.

Adaptive writes a temporary checkpoint beside `gaussian_model.pth` and atomically
replaces that file only after serialization succeeds. Consumers open that exact
filename rather than choosing an arbitrary directory entry. Final, milestone
and periodic metadata all record completed updates.

Static source changes participate in the existing implementation hash. Earlier
checkpoints require their original code; do not rewrite identities. Published
scenes, Phase 1 inputs and existing depth/motion ancestors remain unchanged.

## Validation protocol

- Synthetic fake-clock checks cover expiry during optimization, complete density
  events, periodic saves and milestone saves, without six real-scene epochs just
  to test control flow.
- Check invalid limits, suppressed image reporting, fresh-run restrictions,
  exact iteration caps, checkpoint loading and partial-epoch state.
- Run two real 540p smoke checks serially with cap 30000, a short time limit and
  an interior milestone. These are pipeline checks, not method rankings.
- Reload final and milestone states; check finite model/Adam tensors and agreement
  between state counts, saved steps and timing reports.
- Preserve input hashes, commands, process timings, logs and source patches in
  the existing experiment's `phase2/` directory.

At 318 training frames, Adaptive's first spatial density event is around update
1272 and its first temporal event around 1908. Short real runs normally report
zero density events; synthetic checks establish the associated control flow.

Two upstream observations remain unchanged: the error-statistics loop iterates
the original sampled camera list after constructing a limited list, and uses the
loader time convention rather than the training remap. These are recorded method
implementation limitations, not silently changed by the timing patch. Formal
image metrics will use the common evaluator in Phase 4.

## Completed short-run checks

On the prepared 540p dataset, serial fresh runs used a 12-second budget and a
3-second milestone, keeping the 30000-update schedule and seed 0:

| Method | Updates | Training seconds | Command seconds | Final Gaussian count |
| --- | ---: | ---: | ---: | ---: |
| Modal static, batch 4 | 352 | 12.001 | 22.924 | 80,973 |
| Adaptive, batch 1 | 170 | 12.019 | 18.239 | 225,421 |

Both final states and milestones load successfully; all model/Adam tensors checked
are finite. Static published tensors equal its final checkpoint. Adaptive native
queries at normalized times 0, .5 and 1 succeed. Neither short run reached a real
density event, and Adaptive still had one keyframe per Gaussian. Static retained
its native foreground/background initialization subsampling; the two update counts
are therefore not equivalent units of work.

Thirteen Static checks, seven Adaptive fake-clock scenarios and nine invalid
configuration checks passed, together with atomic-save/loader/output-directory
checks. All 1010 Phase 1 input hashes match. Detailed logs, commands, patches and
results are in the scene-owned experiment's `phase2/PHASE2_REPORT.md`. Formal
300/900-second training, held-out metrics and deblurring conclusions remain later
phases.
