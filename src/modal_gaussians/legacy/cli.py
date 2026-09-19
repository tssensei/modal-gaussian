"""Explicit opt-in commands for the retired Farneback / exact-DFT workflow."""
from pathlib import Path


def add_legacy_commands(command_parsers, _positive_float, _non_negative_float, _positive_int):
    legacy = command_parsers.add_parser("legacy", help="Retired Farneback / greedy exact-DFT commands")
    legacy_commands = legacy.add_subparsers(dest="legacy_command", required=True)
    flow_parser = legacy_commands.add_parser("flow", help="Retired Farneback flow and native FFT")
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
        choices=("none", "weighted-gaussian"),
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
    frequency_parser = legacy_commands.add_parser(
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


def dispatch_legacy(parser, args, arguments):
    if args.legacy_command == "flow" and args.flow_command == "analyze":
        from modal_gaussians.legacy.flow.pipeline import FlowAnalysisConfig, run_flow_analysis

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
    if args.legacy_command == "frequency" and args.frequency_command == "select":
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
    if args.legacy_command == "frequency" and args.frequency_command == "export-modes":
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
    raise ValueError("Unknown legacy command")
