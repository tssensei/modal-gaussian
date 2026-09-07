from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Literal, Sequence, cast

from modal_gaussians.colmap import (
    ReferenceInput,
    prepare_colmap,
)
from modal_gaussians.flow.pipeline import (
    SMOOTHING_METHODS,
    FlowAnalysisConfig,
    run_flow_analysis,
)
from modal_gaussians.progress import progress_log, report_progress


def _positive_float(value: str) -> float:
    """Parse one finite positive CLI float."""

    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _non_negative_float(value: str) -> float:
    """Parse one finite non-negative CLI float."""

    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _positive_int(value: str) -> int:
    """Parse one positive CLI integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    """Parse one non-negative CLI integer."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the unified analysis, static-scene, and modal-data commands."""

    parser = argparse.ArgumentParser(prog="modal-gaussians")
    parser.add_argument(
        "--log-file", type=Path,
        help="Append live progress and failures to this external log (before subcommand)",
    )
    command_parsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = command_parsers.add_parser("prepare", help="Optional video/SAM/XMem preparation")
    prepare_commands = prepare_parser.add_subparsers(dest="prepare_command", required=True)
    mask_gui = prepare_commands.add_parser("gui", help="Local interactive frame/mask preparation")
    mask_gui.add_argument("--root-dir", required=True, type=Path)
    mask_gui.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/masking"))
    mask_gui.add_argument("--port", type=_positive_int, default=8890)
    mask_gui.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    flow_parser = command_parsers.add_parser(
        "flow", help="Dense 2D image-plane motion analysis"
    )
    flow_commands = flow_parser.add_subparsers(dest="flow_command", required=True)
    analyze = flow_commands.add_parser(
        "analyze",
        help="Compute reference-to-frame flow and its temporal rFFT",
    )
    analyze.add_argument("--images", required=True, type=Path)
    analyze.add_argument("--masks", required=True, type=Path)
    analyze.add_argument("--fps", required=True, type=_positive_float)
    analyze.add_argument("--reference-frame", required=True)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument(
        "--stabilize",
        action="store_true",
        help="Apply the accepted reference-background homography stabilization",
    )
    analyze.add_argument(
        "--smoothing",
        choices=SMOOTHING_METHODS,
        default="none",
        help="Optional spatial smoothing before the temporal FFT",
    )
    analyze.add_argument(
        "--sigma-b-px",
        type=_positive_float,
        default=3.0,
        help="Weighted Gaussian displacement blur sigma in pixels",
    )
    analyze.add_argument(
        "--sigma-c-px",
        type=_non_negative_float,
        default=0.0,
        help="Pre-gradient contrast blur sigma in pixels",
    )
    colmap_parser = command_parsers.add_parser(
        "colmap", help="Joint COLMAP preparation for static Gaussian training"
    )
    colmap_commands = colmap_parser.add_subparsers(
        dest="colmap_subcommand", required=True
    )
    prepare = colmap_commands.add_parser(
        "prepare",
        help="Register sampled sweep frames and one reference per fixed view",
    )
    prepare.add_argument("--frames", required=True, type=Path)
    prepare.add_argument("--frame-masks", required=True, type=Path)
    prepare.add_argument(
        "--reference",
        required=True,
        action="append",
        nargs=3,
        metavar=("LABEL", "IMAGE", "MASK"),
        help="Reference label, RGB path, and mask path; repeat for more views",
    )
    prepare.add_argument("--sample-stride", required=True, type=_positive_int)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--colmap-command", dest="colmap_executable", default="colmap")
    static_parser = command_parsers.add_parser(
        "static", help="Independent static foreground/background 3DGS"
    )
    static_commands = static_parser.add_subparsers(
        dest="static_command", required=True
    )
    train = static_commands.add_parser(
        "train", help="Train and export a pure-tensor static 3DGS bundle"
    )
    train.add_argument("--input", required=True, type=Path)
    train.add_argument("--work-dir", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--epochs", type=_positive_int, default=100)
    train.add_argument("--batch-size", type=_positive_int, default=8)
    train.add_argument("--num-fg", type=_positive_int, default=40_000)
    train.add_argument("--num-bg", type=_positive_int, default=80_000)
    train.add_argument("--seed", type=_non_negative_int, default=42)
    train.add_argument("--mask-weight", type=_non_negative_float, default=1.0)
    train.add_argument(
        "--fg-densify-stop-step", type=_positive_int, default=4_000
    )
    train.add_argument(
        "--bg-densify-stop-step", type=_positive_int, default=1_000
    )
    train.add_argument("--max-bg-gaussians", type=_positive_int, default=160_000)
    train.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when input and resolved training config identities match",
    )
    render = static_commands.add_parser(
        "render", help="Render stored sweep/reference cameras for offline QA"
    )
    render.add_argument("--scene", required=True, type=Path)
    render.add_argument("--output", required=True, type=Path)
    render.add_argument(
        "--role", choices=("all", "sweep", "reference"), default="all"
    )
    repartition = static_commands.add_parser(
        "repartition", help="Reclassify trained Gaussians using visibility and per-frame masks"
    )
    repartition.add_argument("--scene", required=True, type=Path)
    repartition.add_argument("--output", required=True, type=Path)
    repartition.add_argument("--dataset-root", type=Path, help="Optional relocated source root; mask hashes must match")
    repartition.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    repartition.add_argument("--mask-dilation-pixels", type=_non_negative_int, default=10,
                             help="Dilate foreground masks by this radius in original-image pixels; no erosion")
    repartition.add_argument("--min-visible-mass", type=_positive_float, default=0.5)
    repartition.add_argument("--min-visible-groups", type=_positive_int, default=2)
    repartition.add_argument("--class-fraction", type=_positive_float, default=0.8)
    repartition.add_argument("--view-angle-degrees", type=_positive_float, default=10.0)
    repartition.add_argument("--view-position-fraction", type=_positive_float, default=0.05)
    topology_parser = command_parsers.add_parser(
        "topology", help="Pixel-to-foreground-Gaussian observation topology"
    )
    topology_commands = topology_parser.add_subparsers(
        dest="topology_command", required=True
    )
    topology_build = topology_commands.add_parser(
        "build", help="Build one reusable mode-independent topology artifact"
    )
    topology_build.add_argument("--scene", required=True, type=Path)
    topology_build.add_argument(
        "--view",
        required=True,
        action="append",
        nargs=2,
        metavar=("LABEL", "FLOW_ARTIFACT"),
        help="Reference-camera label and matching flow artifact; repeat per view",
    )
    topology_build.add_argument("--output", required=True, type=Path)
    topology_build.add_argument("--pixel-stride", type=_positive_int, default=4)
    topology_build.add_argument("--candidate-count", type=_positive_int, default=4)
    topology_build.add_argument("--preselect-count", type=_positive_int, default=32)
    topology_build.add_argument("--alpha-min", type=_non_negative_float, default=0.05)
    topology_build.add_argument(
        "--min-contribution", type=_non_negative_float, default=1e-12
    )
    topology_build.add_argument(
        "--mask-erode-iters", type=_non_negative_int, default=1
    )
    frequency_parser = command_parsers.add_parser(
        "frequency", help="Automatic shared modal-frequency selection"
    )
    frequency_commands = frequency_parser.add_subparsers(
        dest="frequency_command", required=True
    )
    frequency_select = frequency_commands.add_parser(
        "select", help="Greedily select the first K shared exact-DFT frequencies"
    )
    frequency_select.add_argument("--topology", required=True, type=Path)
    frequency_select.add_argument(
        "--view",
        required=True,
        action="append",
        nargs=2,
        metavar=("LABEL", "FLOW_ARTIFACT"),
        help="Topology view label and matching flow artifact; repeat in view order",
    )
    frequency_select.add_argument("--min-hz", required=True, type=_positive_float)
    frequency_select.add_argument("--max-hz", required=True, type=_positive_float)
    frequency_select.add_argument("--step-hz", required=True, type=_positive_float)
    frequency_select.add_argument("--count", required=True, type=_positive_int)
    frequency_select.add_argument("--output", required=True, type=Path)
    frequency_export = frequency_commands.add_parser(
        "export-modes",
        help="Export dense complex 2D fields at the selected frequencies",
    )
    frequency_export.add_argument("--selection", required=True, type=Path)
    frequency_export.add_argument(
        "--view",
        required=True,
        action="append",
        nargs=2,
        metavar=("LABEL", "FLOW_ARTIFACT"),
        help="Selection view label and matching flow artifact; repeat in view order",
    )
    frequency_export.add_argument("--output", required=True, type=Path)
    measurements_parser = command_parsers.add_parser(
        "measurements",
        help="Topology-aligned complex pixel measurement bank",
    )
    measurements_commands = measurements_parser.add_subparsers(
        dest="measurements_command", required=True
    )
    measurements_build = measurements_commands.add_parser(
        "build",
        help="Sample dense complex modes at the topology pixels",
    )
    measurements_build.add_argument("--topology", required=True, type=Path)
    measurements_build.add_argument("--modes", required=True, type=Path)
    measurements_build.add_argument("--output", required=True, type=Path)
    graph_parser = command_parsers.add_parser(
        "graph",
        help="Observed foreground-Gaussian structure graph",
    )
    graph_commands = graph_parser.add_subparsers(
        dest="graph_command", required=True
    )
    graph_build = graph_commands.add_parser(
        "build",
        help="Build an unapproved rigid-component graph candidate",
    )
    graph_build.add_argument("--scene", required=True, type=Path)
    graph_build.add_argument("--topology", required=True, type=Path)
    graph_build.add_argument("--output", required=True, type=Path)
    graph_build.add_argument("--max-neighbors", type=_positive_int, default=8)
    graph_build.add_argument("--max-distance", type=_positive_float, default=0.008)
    graph_build.add_argument(
        "--color-mad-multiplier", type=_non_negative_float, default=3.0
    )
    graph_build.add_argument(
        "--depth-mad-multiplier", type=_non_negative_float, default=3.0
    )
    graph_build.add_argument("--depth-samples", type=_positive_int, default=5)
    graph_build.add_argument("--min-shared-views", type=_positive_int, default=1)
    graph_build.add_argument("--min-component-nodes", type=_positive_int, default=4)
    graph_build.add_argument("--min-component-edges", type=_positive_int, default=3)
    rigid_parser = command_parsers.add_parser(
        "rigid",
        help="Bounded-complex view synchronization and rigid modal solve",
    )
    rigid_commands = rigid_parser.add_subparsers(
        dest="rigid_command", required=True
    )
    rigid_solve = rigid_commands.add_parser(
        "solve",
        help="Solve an unapproved rigid modal candidate for every selected mode",
    )
    rigid_solve.add_argument("--scene", required=True, type=Path)
    rigid_solve.add_argument("--topology", required=True, type=Path)
    rigid_solve.add_argument("--measurements", required=True, type=Path)
    rigid_solve.add_argument("--graph", required=True, type=Path)
    rigid_solve.add_argument("--work-dir", required=True, type=Path)
    rigid_solve.add_argument("--output", required=True, type=Path)
    motion_parser = command_parsers.add_parser(
        "motion", help="Full-foreground modal motion construction"
    )
    motion_commands = motion_parser.add_subparsers(
        dest="motion_command", required=True
    )
    propagate_fragments = motion_commands.add_parser(
        "propagate-fragments", help="Attach compact fragments to fixed local v8 neural motions",
    )
    for name in ("parent", "output"):
        propagate_fragments.add_argument(f"--{name}", required=True, type=Path)
    for name, default in (("max-fragment-nodes", 16), ("core-degree", 3), ("min-anchors", 3)):
        propagate_fragments.add_argument(f"--{name}", type=_positive_int, default=default)
    for name, default in (("max-fragment-extent", 0.016), ("attachment-distance", 0.008),
                          ("patch-radius", 0.008), ("host-size-ratio", 4.0), ("ambiguity-ratio", 1.25)):
        propagate_fragments.add_argument(f"--{name}", type=_positive_float, default=default)
    fit_neural = motion_commands.add_parser(
        "fit-neural", help="Fit full-foreground complex displacement fields",
    )
    prepare_neural = motion_commands.add_parser("prepare-neural", help="Freeze reusable neural observations and geometry caches")
    for name in ("from-result", "scene", "topology", "measurements", "graph", "alignment-from", "config"):
        prepare_neural.add_argument(f"--{name}", type=Path)
    prepare_neural.add_argument("--cache-dir", type=Path, default=Path("outputs/_cache"))
    prepare_neural.add_argument("--output", type=Path, required=True)
    iterate_neural = motion_commands.add_parser("iterate-neural", help="Produce 3D modes; optionally prepare a preview or fit coordinates")
    iterate_neural.add_argument("--prepared", type=Path, required=True)
    iterate_neural.add_argument("--config", type=Path)
    iterate_neural.add_argument("--refine-observations", action="store_true",
                                help="After v12 propagation, refine visible followers while keeping hosts fixed; reuse training")
    iterate_neural.add_argument("--refinement-config", type=Path,
                                help="Optional flat observation-refinement JSON; requires --refine-observations")
    iterate_neural.add_argument("--frequency-hz", type=float, action="append",
                                help="Train only this exact prepared frequency; repeat for a subset, preserving source order and normalization")
    iterate_neural.add_argument("--output", type=Path, required=True)
    iterate_neural.add_argument("--stage", choices=("modes", "preview", "full"), default="modes",
                                help="Stop after final 3D modes by default; preview adds display data, full explicitly fits video coordinates")
    for name in ("scene", "topology", "measurements", "graph", "alignment-from", "work-dir", "output"):
        fit_neural.add_argument(f"--{name}", required=True, type=Path)
    for name, default in (
        ("graph-neighbors", 8), ("max-controls", 2048), ("hidden-dim", 64),
        ("message-layers", 3), ("pixel-sample-stride", 2),
        ("max-iterations", 2000), ("convergence-patience", 50),
        ("checkpoint-every", 100),
    ):
        fit_neural.add_argument(f"--{name}", type=_positive_int, default=default)
    for name, default in (("mask-erosion-iterations", 1), ("seed", 1729)):
        fit_neural.add_argument(f"--{name}", type=_non_negative_int, default=default)
    fit_neural.add_argument("--local-feature-dim", type=_non_negative_int, default=0,
                            help="Learn this many features per control and frequency; 0 preserves the coordinate-only network")
    for name, default in (
        ("graph-max-distance", 0.008), ("unknown-max-distance", 0.004),
        ("unknown-edge-weight", 0.1), ("control-radius-fraction", 0.03),
        ("alpha-minimum", 0.05), ("energy-floor-fraction", 0.05),
        ("huber-delta", 1.0), ("rotation-length-fraction", 0.05),
        ("learning-rate", 0.001), ("gradient-clip", 1.0),
    ):
        fit_neural.add_argument(f"--{name}", type=_positive_float, default=default)
    for name, default in (
        ("deformation-weight", 1.0), ("rotation-weight", 0.1),
        ("relative-tolerance", 1.0e-6),
    ):
        fit_neural.add_argument(f"--{name}", type=_non_negative_float, default=default)
    fit_neural.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    fit_neural.add_argument("--fragment-treatment", choices=("in-training", "post-training"), default="in-training",
                            help="Default: host controls with fixed fragment fill before the loss")
    fit_neural.add_argument("--fragment-config", type=Path,
                            help="Fragment JSON: strategy=component_field selects v16, guarded v14, pointwise v12, surface v11; no strategy v10")
    fit_neural.add_argument("--graph-edge-filter", choices=("depth", "none"), default="depth",
                            help="Use depth/path filtering, or retain every spatial mutual-KNN candidate")
    fit_neural.add_argument(
        "--resume", action="store_true",
        help="Resume only when sources, model, geometry and optimizer settings match",
    )
    export_neural_prefix = motion_commands.add_parser(
        "export-neural-prefix", help="Export completed prefix checkpoints without training",
    )
    for name in ("scene", "topology", "measurements", "graph", "alignment-from", "work-dir", "output"):
        export_neural_prefix.add_argument(f"--{name}", required=True, type=Path)
    export_neural_prefix.add_argument("--count", type=_positive_int, required=True)
    fit_bases = motion_commands.add_parser(
        "fit-bases",
        help="Fit per-Gaussian weights over selected rigid motion bases",
    )
    fit_bases.add_argument("--scene", required=True, type=Path)
    fit_bases.add_argument("--topology", required=True, type=Path)
    fit_bases.add_argument("--measurements", required=True, type=Path)
    fit_bases.add_argument("--graph", required=True, type=Path)
    fit_bases.add_argument("--rigid", required=True, type=Path)
    fit_bases.add_argument("--work-dir", required=True, type=Path)
    fit_bases.add_argument("--output", required=True, type=Path)
    fit_bases.add_argument(
        "--weight-sharing",
        choices=("shared", "per-frequency"),
        default="shared",
        help="Share weights across modes or fit a separate weight field per mode",
    )
    fit_bases.add_argument(
        "--rigid-basis-count", type=_positive_int, default=None,
        help="Exact pool size (legacy policies default to 6); omit for trusted_per_mode",
    )
    fit_bases.add_argument(
        "--basis-selection-policy",
        choices=("trusted_all_modes", "all_rigid_components", "trusted_per_mode"),
        default="trusted_all_modes",
        help="Select trust across all modes, independently per mode, or all solved components",
    )
    fit_bases.add_argument(
        "--local-rigid-basis-count", type=_positive_int, default=4
    )
    fit_bases.add_argument("--graph-neighbors", type=_positive_int, default=8)
    fit_bases.add_argument(
        "--graph-max-distance", type=_positive_float, default=0.008
    )
    fit_bases.add_argument(
        "--distance-temperature", type=_positive_float, default=0.02
    )
    fit_bases.add_argument(
        "--zero-prior-score", type=_positive_float, default=0.05
    )
    fit_bases.add_argument(
        "--smooth-weight", type=_non_negative_float, default=0.01
    )
    fit_bases.add_argument(
        "--prior-weight", type=_non_negative_float, default=0.001
    )
    fit_bases.add_argument(
        "--energy-floor-fraction", type=_positive_float, default=0.05
    )
    fit_bases.add_argument("--max-iterations", type=_positive_int, default=1_000)
    fit_bases.add_argument(
        "--relative-tolerance", type=_non_negative_float, default=1.0e-8
    )
    fit_bases.add_argument(
        "--projected-gradient-tolerance",
        type=_non_negative_float,
        default=1.0e-6,
    )
    fit_bases.add_argument(
        "--convergence-patience", type=_positive_int, default=10
    )
    fit_bases.add_argument(
        "--initial-lipschitz", type=_positive_float, default=1.0
    )
    fit_bases.add_argument(
        "--backtracking-factor", type=_positive_float, default=2.0
    )
    fit_bases.add_argument("--checkpoint-every", type=_positive_int, default=25)
    fit_bases.add_argument(
        "--mode-chunk-size", type=_positive_int, default=4
    )
    fit_bases.add_argument(
        "--sample-chunk-size", type=_positive_int, default=16_384
    )
    fit_bases.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    fit_bases.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when every input and resolved solver setting matches",
    )
    refine_green = motion_commands.add_parser(
        "refine-green",
        help="Refine graph-propagated weights with observation-supported weights fixed",
    )
    refine_green.add_argument("--input", required=True, type=Path)
    refine_green.add_argument("--work-dir", required=True, type=Path)
    refine_green.add_argument("--output", required=True, type=Path)
    refine_green.add_argument(
        "--blue-green-multiplier", type=_positive_float, default=1.0,
        help="Multiply the parent's graph penalty on blue-green edges",
    )
    refine_green.add_argument(
        "--green-prior-multiplier", type=_non_negative_float, default=1.0,
        help="Multiply the parent's distance-prior penalty on green points",
    )
    refine_green.add_argument("--max-iterations", type=_positive_int, default=3_000)
    refine_green.add_argument(
        "--relative-tolerance", type=_non_negative_float, default=1.0e-8
    )
    refine_green.add_argument(
        "--projected-gradient-tolerance", type=_non_negative_float, default=1.0e-6
    )
    refine_green.add_argument(
        "--convergence-patience", type=_positive_int, default=10
    )
    refine_green.add_argument("--checkpoint-every", type=_positive_int, default=50)
    refine_green.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    refine_green.add_argument(
        "--resume", action="store_true",
        help="Resume only when the parent identity and refinement settings match",
    )
    motion_fill = motion_commands.add_parser(
        "fill",
        help="Promote single-view rigid components then fill independent Gaussians",
    )
    motion_fill.add_argument("--scene", required=True, type=Path)
    motion_fill.add_argument("--topology", required=True, type=Path)
    motion_fill.add_argument("--measurements", required=True, type=Path)
    motion_fill.add_argument("--graph", required=True, type=Path)
    motion_fill.add_argument("--rigid", required=True, type=Path)
    motion_fill.add_argument("--work-dir", required=True, type=Path)
    motion_fill.add_argument("--output", required=True, type=Path)
    motion_fill.add_argument(
        "--valid-modes-only", action="store_true",
        help="Treat selected K as an upper limit; omit modes with no trusted rigid seeds",
    )
    motion_fill.add_argument("--neighbors", type=_positive_int, default=8)
    motion_fill.add_argument("--max-distance", type=_positive_float, default=0.008)
    motion_fill.add_argument("--max-anchor-hops", type=_positive_int, default=8)
    motion_fill.add_argument(
        "--observable-ratio", type=_positive_float, default=1.0e-2
    )
    motion_fill.add_argument(
        "--ray-direction-fraction", type=_non_negative_float, default=0.8
    )
    motion_fill.add_argument(
        "--max-finite-drift", type=_non_negative_float, default=2.0
    )
    coordinates_parser = command_parsers.add_parser(
        "coordinates", help="Rasterized modal design and temporal coordinates"
    )
    coordinates_commands = coordinates_parser.add_subparsers(
        dest="coordinates_command", required=True
    )
    render_design = coordinates_commands.add_parser(
        "render-design",
        help="Rasterize completed foreground modes into flow-space columns",
    )
    render_design.add_argument("--scene", required=True, type=Path)
    render_design.add_argument("--modes", required=True, type=Path)
    render_design.add_argument(
        "--view",
        required=True,
        action="append",
        nargs=2,
        metavar=("LABEL", "FLOW_ARTIFACT"),
        help="Completed-mode view label and exact flow artifact; repeat in order",
    )
    render_design.add_argument("--output", required=True, type=Path)
    render_design.add_argument("--pixel-stride", type=_positive_int, default=2)
    render_design.add_argument("--alpha-min", type=_positive_float, default=0.05)
    render_design.add_argument(
        "--mask-erode-iters", type=_non_negative_int, default=1
    )
    render_design.add_argument("--modes-per-batch", type=_positive_int, default=8)
    solve_direct = coordinates_commands.add_parser(
        "solve-direct",
        help="Fit independent per-view modal coordinates to optical flow",
    )
    solve_direct.add_argument("--design", required=True, type=Path)
    solve_direct.add_argument(
        "--view",
        required=True,
        action="append",
        nargs=2,
        metavar=("LABEL", "FLOW_ARTIFACT"),
        help="Rendered-design view label and exact flow artifact; repeat in order",
    )
    solve_direct.add_argument("--output", required=True, type=Path)
    solve_direct.add_argument(
        "--ridge-relative", type=_positive_float, default=1.0e-4
    )
    solve_direct.add_argument("--frame-chunk-size", type=_positive_int, default=64)
    physics_fit = coordinates_commands.add_parser(
        "physics-fit",
        help="Post-fit direct coordinates with a damped oscillator",
    )
    physics_fit.add_argument("--input", required=True, type=Path)
    physics_fit.add_argument("--output", required=True, type=Path)
    physics_fit.add_argument(
        "--damping-ratio", type=_non_negative_float, default=0.05
    )
    physics_fit.add_argument(
        "--forcing-weight", type=_non_negative_float, default=0.1
    )
    physics_fit.add_argument(
        "--forcing-difference-weight", type=_non_negative_float, default=0.0
    )
    physics_fit.add_argument(
        "--assigned-band-half-width-hz", type=_positive_float, default=0.1
    )
    physics_fit.add_argument("--frame-chunk-size", type=_positive_int, default=64)
    result_parser = command_parsers.add_parser(
        "result", help="Bind one immutable static/mode/coordinate result"
    )
    result_commands = result_parser.add_subparsers(
        dest="result_command", required=True
    )
    result_materialize = result_commands.add_parser(
        "materialize",
        help="Validate and link a complete modal result without copying tensors",
    )
    result_materialize.add_argument("--scene", required=True, type=Path)
    result_materialize.add_argument("--modes", required=True, type=Path)
    result_materialize.add_argument("--coordinates", required=True, type=Path)
    result_materialize.add_argument("--output", required=True, type=Path)
    viewer = command_parsers.add_parser(
        "viewer", help="Inspect one complete modal result in Viser"
    )
    viewer_input = viewer.add_mutually_exclusive_group(required=True)
    viewer_input.add_argument("--result", type=Path)
    viewer_input.add_argument("--preview", type=Path)
    viewer.add_argument("--work-dir", required=True, type=Path)
    viewer.add_argument("--host", default="0.0.0.0")
    viewer.add_argument("--port", type=_positive_int, default=8080)
    viewer.add_argument("--viewer-res", type=_positive_int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Log one CLI stage without hiding failures, retrying, or starting others."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    with progress_log(args.log_file):
        started_at = time.perf_counter()
        report_progress(f"START {parser.prog} {arguments!r}")
        try:
            result = _dispatch(parser, args, arguments)
        except KeyboardInterrupt:
            report_progress("INTERRUPTED by user")
            raise
        except BaseException:
            report_progress(f"FAILED\n{traceback.format_exc()}")
            raise
        report_progress(
            f"{'COMPLETE' if result == 0 else 'FAILED'} {args.command}"
            f" | elapsed={time.perf_counter() - started_at:.1f}s exit_code={result}"
        )
        return result


def _dispatch(
    parser: argparse.ArgumentParser, args: argparse.Namespace, arguments: list[str]
) -> int:
    """Dispatch the parsed command and preserve existing CLI error semantics."""

    try:
        if args.command == "prepare" and args.prepare_command == "gui":
            try:
                from modal_gaussians.mask_gui import run_mask_gui
            except ImportError as error:
                raise RuntimeError("Mask preparation requires the optional [mask] dependencies; see README") from error
            run_mask_gui(root_dir=args.root_dir, checkpoint_dir=args.checkpoint_dir,
                         port=args.port, device=args.device)
            return 0
        if args.command == "flow" and args.flow_command == "analyze":
            artifact = run_flow_analysis(
                image_dir=args.images,
                mask_dir=args.masks,
                fps_hz=args.fps,
                reference_frame_name=args.reference_frame,
                output_dir=args.output,
                config=FlowAnalysisConfig(
                    stabilize=bool(args.stabilize),
                    smoothing=str(args.smoothing),
                    sigma_b_px=float(args.sigma_b_px),
                    sigma_c_px=float(args.sigma_c_px),
                ),
                command=[parser.prog, *arguments],
            )
            print(f"flow artifact: {artifact.path.resolve()}")
            print(f"manifest: {(artifact.path / 'manifest.json').resolve()}")
            return 0
        if args.command == "colmap" and args.colmap_subcommand == "prepare":
            output = prepare_colmap(
                frames_dir=args.frames,
                frame_masks_dir=args.frame_masks,
                references=tuple(
                    ReferenceInput(
                        label=label,
                        image_path=Path(image_path),
                        mask_path=Path(mask_path),
                    )
                    for label, image_path, mask_path in args.reference
                ),
                sample_stride=int(args.sample_stride),
                output_dir=args.output,
                colmap_command=str(args.colmap_executable),
            )
            print(f"COLMAP output: {output.resolve()}")
            print(f"cameras: {(output / 'cameras.json').resolve()}")
            print(f"point cloud: {(output / 'point_cloud.ply').resolve()}")
            return 0
        if args.command == "static" and args.static_command == "train":
            from modal_gaussians.static_training import (
                StaticTrainConfig,
                run_static_training,
            )

            output = run_static_training(
                input_dir=args.input,
                work_dir=args.work_dir,
                output_dir=args.output,
                config=StaticTrainConfig(
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    num_foreground=int(args.num_fg),
                    num_background=int(args.num_bg),
                    seed=int(args.seed),
                    mask_loss_weight=float(args.mask_weight),
                    foreground_densify_stop_step=int(args.fg_densify_stop_step),
                    background_densify_stop_step=int(args.bg_densify_stop_step),
                    max_background_gaussians=int(args.max_bg_gaussians),
                ),
                resume=bool(args.resume),
            )
            print(f"static scene: {output.resolve()}")
            print(f"manifest: {(output / 'manifest.json').resolve()}")
            print(f"tensors: {(output / 'tensors.pt').resolve()}")
            return 0
        if args.command == "static" and args.static_command == "repartition":
            from modal_gaussians.static_partition import PartitionConfig, repartition_static_scene

            output = repartition_static_scene(
                scene_dir=args.scene, output_dir=args.output, device=str(args.device),
                dataset_root=args.dataset_root,
                config=PartitionConfig(
                    mask_dilation_pixels=int(args.mask_dilation_pixels),
                    minimum_visible_mass=float(args.min_visible_mass),
                    minimum_visible_groups=int(args.min_visible_groups),
                    class_fraction=float(args.class_fraction),
                    view_angle_degrees=float(args.view_angle_degrees),
                    view_position_fraction=float(args.view_position_fraction),
                ),
            )
            print(f"repartitioned static scene: {output.resolve()}")
            print(f"classification summary: {(output / 'partition-summary.json').resolve()}")
            return 0
        if args.command == "static" and args.static_command == "render":
            from modal_gaussians.static_training import render_static_bundle

            output = render_static_bundle(
                scene_dir=args.scene,
                output_dir=args.output,
                role=cast(
                    Literal["all", "sweep", "reference"], str(args.role)
                ),
            )
            print(f"static QA: {output.resolve()}")
            print(f"metrics: {(output / 'metrics.json').resolve()}")
            return 0
        if args.command == "topology" and args.topology_command == "build":
            from modal_gaussians.topology import (
                TopologyConfig,
                TopologyViewInput,
                build_observation_topology_artifact,
            )

            artifact = build_observation_topology_artifact(
                scene_dir=args.scene,
                views=tuple(
                    TopologyViewInput(label=label, flow_artifact=Path(flow_artifact))
                    for label, flow_artifact in args.view
                ),
                output_dir=args.output,
                config=TopologyConfig(
                    pixel_sample_stride=int(args.pixel_stride),
                    pixel_candidate_count=int(args.candidate_count),
                    pixel_preselect_count=int(args.preselect_count),
                    foreground_alpha_minimum=float(args.alpha_min),
                    minimum_contribution=float(args.min_contribution),
                    mask_erosion_iterations=int(args.mask_erode_iters),
                ),
                command=[parser.prog, *arguments],
            )
            print(f"observation topology: {artifact.path.resolve()}")
            print(f"identity: {artifact.manifest['topology_identity']}")
            return 0
        if args.command == "frequency" and args.frequency_command == "select":
            from modal_gaussians.frequency import (
                FrequencySelectionConfig,
                FrequencyViewInput,
                build_frequency_selection_artifact,
            )

            artifact = build_frequency_selection_artifact(
                topology_dir=args.topology,
                views=tuple(
                    FrequencyViewInput(
                        label=label, flow_artifact=Path(flow_artifact)
                    )
                    for label, flow_artifact in args.view
                ),
                output_dir=args.output,
                config=FrequencySelectionConfig(
                    minimum_hz=float(args.min_hz),
                    maximum_hz=float(args.max_hz),
                    step_hz=float(args.step_hz),
                    count=int(args.count),
                ),
                command=[parser.prog, *arguments],
            )
            selected = artifact.arrays.selected_frequencies_hz
            print(f"frequency selection: {artifact.path.resolve()}")
            print("selected Hz: " + ", ".join(f"{value:.9g}" for value in selected))
            print(f"identity: {artifact.manifest['frequency_selection_identity']}")
            return 0
        if args.command == "frequency" and args.frequency_command == "export-modes":
            from modal_gaussians.modes import (
                ComplexModeViewInput,
                build_complex_2d_modes_artifact,
            )

            artifact = build_complex_2d_modes_artifact(
                selection_dir=args.selection,
                views=tuple(
                    ComplexModeViewInput(
                        label=label, flow_artifact=Path(flow_artifact)
                    )
                    for label, flow_artifact in args.view
                ),
                output_dir=args.output,
                command=[parser.prog, *arguments],
            )
            print(f"complex 2D modes: {artifact.path.resolve()}")
            print(f"views: {len(artifact.view_modes)}")
            print(f"modes: {len(artifact.manifest['modes'])}")
            print(f"identity: {artifact.manifest['complex_2d_modes_identity']}")
            return 0
        if (
            args.command == "measurements"
            and args.measurements_command == "build"
        ):
            from modal_gaussians.measurements import (
                build_gaussian_measurements_artifact,
            )

            artifact = build_gaussian_measurements_artifact(
                topology_dir=args.topology,
                modes_dir=args.modes,
                output_dir=args.output,
                command=[parser.prog, *arguments],
            )
            print(f"Gaussian measurements: {artifact.path.resolve()}")
            print(f"modes: {artifact.measurements.shape[0]}")
            print(f"samples: {artifact.measurements.shape[1]}")
            print(
                "identity: "
                f"{artifact.manifest['gaussian_measurements_identity']}"
            )
            return 0
        if args.command == "graph" and args.graph_command == "build":
            from modal_gaussians.motion.rigid.structure_graph import (
                ObservedStructureGraphConfig,
                build_observed_structure_graph_artifact,
            )

            artifact = build_observed_structure_graph_artifact(
                scene_dir=args.scene,
                topology_dir=args.topology,
                output_dir=args.output,
                config=ObservedStructureGraphConfig(
                    max_neighbors=int(args.max_neighbors),
                    max_distance=float(args.max_distance),
                    color_mad_multiplier=float(args.color_mad_multiplier),
                    depth_mad_multiplier=float(args.depth_mad_multiplier),
                    depth_samples=int(args.depth_samples),
                    min_shared_views=int(args.min_shared_views),
                    min_component_nodes=int(args.min_component_nodes),
                    min_component_edges=int(args.min_component_edges),
                ),
                command=[parser.prog, *arguments],
            )
            counts = artifact.manifest["counts"]
            print(f"observed graph candidate: {artifact.path.resolve()}")
            print(
                "nodes/edges/components: "
                f"{counts['node_count']}/{counts['retained_edges']}/"
                f"{counts['components']}"
            )
            print(f"isolated nodes: {counts['isolated_nodes']}")
            print("quality gate: manual approval required")
            print(
                "identity: "
                f"{artifact.manifest['observed_structure_graph_identity']}"
            )
            return 0
        if args.command == "rigid" and args.rigid_command == "solve":
            from modal_gaussians.motion.rigid.rigid import build_rigid_modes_artifact

            artifact = build_rigid_modes_artifact(
                scene_dir=args.scene,
                topology_dir=args.topology,
                measurements_dir=args.measurements,
                graph_dir=args.graph,
                work_dir=args.work_dir,
                output_dir=args.output,
                command=[parser.prog, *arguments],
            )
            counts = artifact.manifest["counts"]
            print(f"rigid modal candidate: {artifact.path.resolve()}")
            print(
                "modes/views/components: "
                f"{counts['modes']}/{counts['views']}/"
                f"{counts['rigid_components']}"
            )
            print(
                "trusted Gaussian seeds per mode: "
                + ", ".join(
                    str(value)
                    for value in counts["trusted_seed_gaussians_per_mode"]
                )
            )
            print("quality gate: unified visualization approval required")
            print(f"identity: {artifact.manifest['rigid_modes_identity']}")
            return 0
        if args.command == "motion" and args.motion_command == "propagate-fragments":
            from modal_gaussians.motion.neural.fragment_propagation import (
                FragmentPropagationConfig,
                build_fragment_modes,
            )

            names = ("max_fragment_nodes", "core_degree", "min_anchors", "max_fragment_extent",
                     "attachment_distance", "patch_radius", "host_size_ratio", "ambiguity_ratio")
            artifact = build_fragment_modes(
                parent_dir=args.parent, output_dir=args.output,
                config=FragmentPropagationConfig(**{name: getattr(args, name) for name in names}),
                command=[parser.prog, *arguments],
            )
            print(f"fragment-propagated modal candidate: {artifact.path}")
            print(f"identity: {artifact.manifest['completed_modes_identity']}")
            print(json.dumps(artifact.manifest["diagnostics"]))
            return 0
        if args.command == "motion" and args.motion_command == "prepare-neural":
            from modal_gaussians.motion.neural.prepared import prepare_neural
            overrides = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            artifact = prepare_neural(from_result=args.from_result, scene_dir=args.scene, topology_dir=args.topology,
                measurements_dir=args.measurements, graph_dir=args.graph, alignment_from=args.alignment_from,
                cache_dir=args.cache_dir, output_dir=args.output, config_overrides=overrides)
            print(f"prepared: {artifact.path}")
            print(f"prepared_identity: {artifact.manifest['prepared_identity']}")
            return 0
        if args.command == "motion" and args.motion_command == "iterate-neural":
            from modal_gaussians.motion.neural.iteration import iterate_neural
            root = iterate_neural(prepared_dir=args.prepared, config_path=args.config,
                                  output_dir=args.output, stage=args.stage, frequencies_hz=args.frequency_hz,
                                  refine_observations=args.refine_observations, refinement_config_path=args.refinement_config)
            print(f"iteration: {root}")
            return 0
        if args.command == "motion" and args.motion_command == "fit-neural":
            from modal_gaussians.motion.neural.neural_modes import (
                NeuralModesConfig,
                build_neural_modes_artifact,
            )

            config_names = (
                "graph_neighbors", "graph_max_distance", "graph_edge_filter", "unknown_max_distance",
                "unknown_edge_weight", "control_radius_fraction", "max_controls",
                "hidden_dim", "message_layers", "local_feature_dim", "pixel_sample_stride", "alpha_minimum",
                "mask_erosion_iterations", "energy_floor_fraction", "huber_delta",
                "deformation_weight", "rotation_weight", "rotation_length_fraction",
                "learning_rate", "max_iterations", "gradient_clip", "seed",
                "convergence_patience", "relative_tolerance", "checkpoint_every", "device",
            )
            training_fragments = None
            if args.fragment_treatment == "in-training":
                from modal_gaussians.motion.neural.fragment_propagation import FragmentPropagationConfig
                from modal_gaussians.motion.neural.surface_attachments import SurfaceAttachmentConfig
                fragment_values = json.loads(args.fragment_config.read_text(encoding="utf-8")) if args.fragment_config else {}
                from modal_gaussians.motion.neural.training_fragments import config_class
                cls = config_class(fragment_values) if args.fragment_config else SurfaceAttachmentConfig
                training_fragments = cls(**fragment_values).to_dict()
            elif args.fragment_config:
                raise ValueError("--fragment-config requires --fragment-treatment in-training")
            artifact = build_neural_modes_artifact(
                scene_dir=args.scene, topology_dir=args.topology,
                measurements_dir=args.measurements, graph_dir=args.graph,
                alignment_from=args.alignment_from, work_dir=args.work_dir,
                output_dir=args.output, resume=bool(args.resume),
                config=NeuralModesConfig(**{name: getattr(args, name) for name in config_names},
                                         training_fragment_config=training_fragments),
                command=[parser.prog, *arguments],
            )
            counts = artifact.manifest["counts"]
            print(f"neural modal candidate: {artifact.path.resolve()}")
            print(f"modes/foreground: {counts['modes']}/{counts['foreground_gaussians']}")
            print("quality gate: unified visualization approval required")
            print(f"identity: {artifact.manifest['completed_modes_identity']}")
            return 0
        if args.command == "motion" and args.motion_command == "export-neural-prefix":
            from modal_gaussians.motion.neural.neural_modes import export_neural_prefix_artifact

            artifact = export_neural_prefix_artifact(
                scene_dir=args.scene, topology_dir=args.topology,
                measurements_dir=args.measurements, graph_dir=args.graph,
                alignment_from=args.alignment_from, work_dir=args.work_dir,
                output_dir=args.output, count=args.count, command=[parser.prog, *arguments],
            )
            counts = artifact.manifest["counts"]
            print(f"neural prefix candidate: {artifact.path.resolve()}")
            print(f"modes/foreground: {counts['modes']}/{counts['foreground_gaussians']}")
            print("quality gate: unified visualization approval required")
            print(f"identity: {artifact.manifest['completed_modes_identity']}")
            return 0
        if args.command == "motion" and args.motion_command == "fit-bases":
            from modal_gaussians.motion.rigid.motion_basis import (
                MotionBasisConfig,
                build_motion_basis_modes_artifact,
            )

            if args.basis_selection_policy == "trusted_per_mode":
                if args.weight_sharing != "per-frequency":
                    parser.error("trusted_per_mode requires --weight-sharing per-frequency")
                if args.rigid_basis_count is not None:
                    parser.error("trusted_per_mode determines its pool automatically; omit --rigid-basis-count")
                rigid_basis_count = None
            else:
                rigid_basis_count = 6 if args.rigid_basis_count is None else int(args.rigid_basis_count)
            if args.weight_sharing == "per-frequency":
                from modal_gaussians.motion.rigid.motion_basis_frequency import build_frequency_motion_basis_modes_artifact

                build_basis_artifact = build_frequency_motion_basis_modes_artifact
            else:
                build_basis_artifact = build_motion_basis_modes_artifact
            artifact = build_basis_artifact(
                scene_dir=args.scene,
                topology_dir=args.topology,
                measurements_dir=args.measurements,
                observed_graph_dir=args.graph,
                rigid_modes_dir=args.rigid,
                work_dir=args.work_dir,
                output_dir=args.output,
                resume=bool(args.resume),
                config=MotionBasisConfig(
                    rigid_basis_count=rigid_basis_count,
                    basis_selection_policy=str(args.basis_selection_policy),
                    local_rigid_basis_count=int(args.local_rigid_basis_count),
                    graph_neighbors=int(args.graph_neighbors),
                    graph_max_distance=float(args.graph_max_distance),
                    distance_temperature=float(args.distance_temperature),
                    zero_prior_score=float(args.zero_prior_score),
                    smooth_weight=float(args.smooth_weight),
                    prior_weight=float(args.prior_weight),
                    energy_floor_fraction=float(args.energy_floor_fraction),
                    max_iterations=int(args.max_iterations),
                    relative_tolerance=float(args.relative_tolerance),
                    projected_gradient_tolerance=float(
                        args.projected_gradient_tolerance
                    ),
                    convergence_patience=int(args.convergence_patience),
                    initial_lipschitz=float(args.initial_lipschitz),
                    backtracking_factor=float(args.backtracking_factor),
                    checkpoint_every=int(args.checkpoint_every),
                    mode_chunk_size=int(args.mode_chunk_size),
                    sample_chunk_size=int(args.sample_chunk_size),
                    device=cast(
                        Literal["auto", "cpu", "cuda"], str(args.device)
                    ),
                ),
                command=[parser.prog, *arguments],
            )
            counts = artifact.manifest["counts"]
            print(f"motion-basis modal candidate: {artifact.path.resolve()}")
            print(f"weight sharing: {args.weight_sharing}")
            print(
                "modes/foreground/rigid bases/total bases: "
                f"{counts['modes']}/{counts['foreground_gaussians']}/"
                f"{counts['rigid_bases']}/{counts['bases']}"
            )
            if "basis_active_mask" in artifact.arrays:
                print("trusted rigid bases per source mode: " + ", ".join(
                    str(int(value)) for value in artifact.arrays["basis_active_mask"][:, :-1].sum(axis=1)
                ))
            role_count_scope = (
                " (union over modes)" if args.weight_sharing == "per-frequency" else ""
            )
            print(
                "measurement-supported/graph-propagated/zero-fallback Gaussians"
                f"{role_count_scope}: "
                f"{counts['measurement_supported_gaussians']}/"
                f"{counts['graph_propagated_gaussians']}/"
                f"{counts['zero_fallback_gaussians']}"
            )
            print(f"full-foreground spatial edges: {counts['spatial_edges']}")
            print("quality gate: unified visualization approval required")
            print(f"identity: {artifact.manifest['completed_modes_identity']}")
            return 0
        if args.command == "motion" and args.motion_command == "refine-green":
            from modal_gaussians.motion.rigid.motion_basis_green import (
                GreenRefinementConfig,
                build_green_refined_motion_basis_artifact,
            )

            artifact = build_green_refined_motion_basis_artifact(
                parent_dir=args.input,
                work_dir=args.work_dir,
                output_dir=args.output,
                resume=bool(args.resume),
                config=GreenRefinementConfig(
                    blue_green_multiplier=float(args.blue_green_multiplier),
                    green_prior_multiplier=float(args.green_prior_multiplier),
                    max_iterations=int(args.max_iterations),
                    relative_tolerance=float(args.relative_tolerance),
                    projected_gradient_tolerance=float(
                        args.projected_gradient_tolerance
                    ),
                    convergence_patience=int(args.convergence_patience),
                    checkpoint_every=int(args.checkpoint_every),
                    device=cast(Literal["auto", "cpu", "cuda"], str(args.device)),
                ),
                command=[parser.prog, *arguments],
            )
            counts = artifact.manifest["counts"]
            print(f"green-refined modal candidate: {artifact.path.resolve()}")
            print(
                "modes/foreground: "
                f"{counts['modes']}/{counts['foreground_gaussians']}"
            )
            print("measurement-supported weights: fixed to parent")
            print(
                "fixed measurement-supported/refined graph-propagated/"
                "zero-fallback Gaussians (union over modes): "
                f"{counts['measurement_supported_gaussians']}/"
                f"{counts['graph_propagated_gaussians']}/"
                f"{counts['zero_fallback_gaussians']}"
            )
            print("quality gate: unified visualization approval required")
            print(f"identity: {artifact.manifest['completed_modes_identity']}")
            return 0
        if args.command == "motion" and args.motion_command == "fill":
            from modal_gaussians.motion.rigid.motion_fill import (
                MotionFillConfig,
                build_completed_modes_artifact,
            )

            artifact = build_completed_modes_artifact(
                scene_dir=args.scene,
                topology_dir=args.topology,
                measurements_dir=args.measurements,
                observed_graph_dir=args.graph,
                rigid_modes_dir=args.rigid,
                work_dir=args.work_dir,
                output_dir=args.output,
                valid_modes_only=bool(args.valid_modes_only),
                config=MotionFillConfig(
                    neighbors=int(args.neighbors),
                    max_distance=float(args.max_distance),
                    max_anchor_hops=int(args.max_anchor_hops),
                    observable_singular_ratio_minimum=float(
                        args.observable_ratio
                    ),
                    ray_direction_minimum_fraction=float(
                        args.ray_direction_fraction
                    ),
                    maximum_finite_drift=float(args.max_finite_drift),
                ),
                command=[parser.prog, *arguments],
            )
            counts = artifact.manifest["counts"]
            print(f"completed modal candidate: {artifact.path.resolve()}")
            print(
                "modes/foreground/fill edges: "
                f"{counts['modes']}/{counts['foreground_gaussians']}/"
                f"{counts['fill_graph_edges']}"
            )
            print(
                "unresolved Gaussians per mode: "
                + ", ".join(
                    str(value)
                    for value in counts["unresolved_gaussians_per_mode"]
                )
            )
            print("quality gate: unified visualization approval required")
            print(f"identity: {artifact.manifest['completed_modes_identity']}")
            return 0
        if (
            args.command == "coordinates"
            and args.coordinates_command == "render-design"
        ):
            from modal_gaussians.rendered_design import (
                RenderedDesignConfig,
                RenderedDesignViewInput,
                build_rendered_modal_design_artifact,
            )

            artifact = build_rendered_modal_design_artifact(
                scene_dir=args.scene,
                completed_modes_dir=args.modes,
                views=tuple(
                    RenderedDesignViewInput(
                        label=label, flow_artifact=Path(flow_artifact)
                    )
                    for label, flow_artifact in args.view
                ),
                output_dir=args.output,
                config=RenderedDesignConfig(
                    pixel_sample_stride=int(args.pixel_stride),
                    alpha_minimum=float(args.alpha_min),
                    mask_erosion_iterations=int(args.mask_erode_iters),
                    modes_per_batch=int(args.modes_per_batch),
                ),
                command=[parser.prog, *arguments],
            )
            counts = artifact.manifest["counts"]
            print(f"rendered modal design: {artifact.path.resolve()}")
            print(
                "views/modes/samples: "
                f"{counts['views']}/{counts['modes']}/{counts['samples']}"
            )
            print(
                "identity: "
                f"{artifact.manifest['rendered_design_identity']}"
            )
            return 0
        if (
            args.command == "coordinates"
            and args.coordinates_command == "solve-direct"
        ):
            from modal_gaussians.direct_coordinates import (
                DirectCoordinateConfig,
                DirectCoordinateViewInput,
                build_direct_modal_coordinates_artifact,
            )

            artifact = build_direct_modal_coordinates_artifact(
                rendered_design_dir=args.design,
                views=tuple(
                    DirectCoordinateViewInput(
                        label=label, flow_artifact=Path(flow_artifact)
                    )
                    for label, flow_artifact in args.view
                ),
                output_dir=args.output,
                config=DirectCoordinateConfig(
                    ridge_relative=float(args.ridge_relative),
                    frame_chunk_size=int(args.frame_chunk_size),
                ),
                command=[parser.prog, *arguments],
            )
            counts = artifact.manifest["counts"]
            print(f"direct modal coordinates: {artifact.path.resolve()}")
            print(
                "views/frames/modes: "
                f"{counts['views']}/{counts['frames']}/{counts['modes']}"
            )
            print(f"flow R2: {artifact.manifest['overall']['flow_r2']:.9g}")
            print(
                "identity: "
                f"{artifact.manifest['direct_coordinates_identity']}"
            )
            return 0
        if (
            args.command == "coordinates"
            and args.coordinates_command == "physics-fit"
        ):
            from modal_gaussians.physics_coordinates import (
                PhysicsCoordinateConfig,
                build_physics_modal_coordinates_artifact,
            )

            artifact = build_physics_modal_coordinates_artifact(
                direct_coordinates_dir=args.input,
                output_dir=args.output,
                config=PhysicsCoordinateConfig(
                    damping_ratio=float(args.damping_ratio),
                    forcing_weight=float(args.forcing_weight),
                    forcing_difference_weight=float(
                        args.forcing_difference_weight
                    ),
                    assigned_band_half_width_hz=float(
                        args.assigned_band_half_width_hz
                    ),
                    frame_chunk_size=int(args.frame_chunk_size),
                ),
                command=[parser.prog, *arguments],
            )
            counts = artifact.manifest["counts"]
            overall = artifact.manifest["overall"]
            print(f"physics modal coordinates: {artifact.path.resolve()}")
            print(
                "views/frames/modes: "
                f"{counts['views']}/{counts['frames']}/{counts['modes']}"
            )
            print(
                "flow R2: "
                f"{overall['input_flow_r2']:.9g} -> "
                f"{overall['output_flow_r2']:.9g}"
            )
            print(
                "identity: "
                f"{artifact.manifest['physics_coordinates_identity']}"
            )
            return 0
        if args.command == "result" and args.result_command == "materialize":
            from modal_gaussians.result import materialize_modal_result

            artifact = materialize_modal_result(
                scene_dir=args.scene,
                completed_modes_dir=args.modes,
                coordinates_dir=args.coordinates,
                output_dir=args.output,
                command=[parser.prog, *arguments],
            )
            counts = artifact.manifest["counts"]
            coordinate_source = artifact.manifest["coordinate_source"]
            print(f"modal result: {artifact.path.resolve()}")
            print(
                "foreground/background/modes/views/frames: "
                f"{counts['foreground_gaussians']}/"
                f"{counts['background_gaussians']}/"
                f"{counts['modes']}/{counts['views']}/{counts['frames']}"
            )
            print(f"coordinates: {coordinate_source['kind']}")
            print("quality gate: unified visualization approval required")
            print(f"identity: {artifact.manifest['modal_result_identity']}")
            return 0
        if args.command == "viewer":
            from modal_gaussians.vis.viewer import run_modal_viewer

            run_modal_viewer(
                result_dir=args.preview or args.result,
                work_dir=args.work_dir,
                host=str(args.host),
                port=int(args.port),
                viewer_resolution=int(args.viewer_res),
                preview=args.preview is not None,
            )
            return 0
        parser.error("unsupported command")
    except (
        FileExistsError,
        FileNotFoundError,
        FloatingPointError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
