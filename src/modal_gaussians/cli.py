from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Sequence

from modal_gaussians.colmap import (
    ReferenceInput,
    prepare_colmap,
)
from modal_gaussians.flow.pipeline import (
    SMOOTHING_METHODS,
    FlowAnalysisConfig,
    run_flow_analysis,
)


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
    """Build the unified flow, static-scene, topology, and frequency commands."""

    parser = argparse.ArgumentParser(prog="modal-gaussians")
    command_parsers = parser.add_subparsers(dest="command", required=True)
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
    train.add_argument("--num-bg", type=_positive_int, default=100_000)
    train.add_argument("--seed", type=_non_negative_int, default=42)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one CLI invocation and report user-facing validation errors."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
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
        if args.command == "static":
            from modal_gaussians.static_training import (
                StaticTrainConfig,
                render_static_bundle,
                run_static_training,
            )
        if args.command == "static" and args.static_command == "train":
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
                ),
                resume=bool(args.resume),
            )
            print(f"static scene: {output.resolve()}")
            print(f"manifest: {(output / 'manifest.json').resolve()}")
            print(f"tensors: {(output / 'tensors.pt').resolve()}")
            return 0
        if args.command == "static" and args.static_command == "render":
            output = render_static_bundle(
                scene_dir=args.scene,
                output_dir=args.output,
                role=str(args.role),
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
