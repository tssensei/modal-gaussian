from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Sequence

from modal_gaussians.flow.pipeline import (
    SMOOTHING_METHODS,
    FlowAnalysisConfig,
    run_flow_analysis,
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.command != "flow" or args.flow_command != "analyze":
        parser.error("unsupported command")
    try:
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
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    print(f"flow artifact: {artifact.path.resolve()}")
    print(f"manifest: {(artifact.path / 'manifest.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
