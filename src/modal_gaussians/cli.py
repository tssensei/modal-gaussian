from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Literal, Sequence, cast

from modal_gaussians.geometry.colmap import ReferenceInput, prepare_colmap
from modal_gaussians.common.progress import progress_log, report_progress


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
    storage = command_parsers.add_parser("storage", help="Discover and use scene-owned reusable inputs/results")
    storage_commands = storage.add_subparsers(dest="storage_command", required=True)
    storage_list = storage_commands.add_parser("list")
    storage_list.add_argument("--scene")
    storage_path = storage_commands.add_parser("path")
    storage_path.add_argument("--scene", required=True)
    storage_path.add_argument("--asset", required=True)
    storage_run = storage_commands.add_parser("run", help="Run an existing command with @scene-asset paths")
    storage_run.add_argument("--scene", required=True)
    storage_run.add_argument("arguments", nargs=argparse.REMAINDER)
    prepare_parser = command_parsers.add_parser("prepare", help="Optional video/SAM/XMem preparation")
    prepare_commands = prepare_parser.add_subparsers(dest="prepare_command", required=True)
    mask_gui = prepare_commands.add_parser("gui", help="Local interactive frame/mask preparation")
    mask_gui.add_argument("--root-dir", required=True, type=Path)
    mask_gui.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/masking"))
    mask_gui.add_argument("--port", type=_positive_int, default=8890)
    mask_gui.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    reference = prepare_commands.add_parser('reference', help='Bind timing, masks and the fixed camera pixel grid')
    for name in ('images', 'masks', 'output'):
        reference.add_argument('--' + name, type=Path, required=True)
    reference.add_argument('--fps', type=_positive_float, required=True)
    reference.add_argument('--reference-frame', required=True)
    reference.add_argument('--tripod', action='store_true', help='Explicitly declare a stationary tripod recording; skip stabilization')
    reference.add_argument('--scene', type=Path, help='Static scene with registered raw reference and COLMAP background map (required unless --tripod)')
    reference.add_argument('--view', help='Registered reference camera label (required unless --tripod)')
    reference.add_argument('--config', type=Path, help='StabilizationSettings JSON overrides')
    flow_parser = command_parsers.add_parser(
        "flow", help="SEA-RAFT reference-to-frame flow (FFT is a separate cache step)"
    )
    spectrum_parser = command_parsers.add_parser("spectrum", help="Shared-grid FFT cache and manual peak inspection")
    spectrum_commands = spectrum_parser.add_subparsers(dest="spectrum_command", required=True)
    spectrum_build = spectrum_commands.add_parser("build", help="Cache a common FFT grid from existing SEA-RAFT flows")
    spectrum_build.add_argument("--view", action="append", nargs=2, required=True, metavar=("LABEL", "SEA_FLOW"))
    spectrum_region = spectrum_build.add_mutually_exclusive_group(required=True)
    spectrum_region.add_argument("--scene", type=Path)
    spectrum_region.add_argument("--region", action="append", nargs=2, metavar=("LABEL", "BOOL_NPY"),
                                 help="Analysis region only; full-image FFT data is always saved")
    spectrum_build.add_argument("--nfft", required=True, type=_positive_int)
    spectrum_build.add_argument("--output", required=True, type=Path)
    spectrum_viewer = spectrum_commands.add_parser("viewer", help="Standalone browser spectrum GUI (no automatic snapping)")
    spectrum_viewer.add_argument("--input", required=True, type=Path)
    spectrum_viewer.add_argument("--work-dir", required=True, type=Path)
    spectrum_viewer.add_argument("--host", default="127.0.0.1")
    spectrum_viewer.add_argument("--port", type=_positive_int, default=8110)
    spectrum_export = spectrum_commands.add_parser("export", help="Copy selected cached bins into modal-image inputs")
    spectrum_export.add_argument("--input", required=True, type=Path)
    spectrum_export.add_argument("--selection", required=True, type=Path)
    spectrum_export.add_argument("--output", required=True, type=Path)
    spectrum_select = spectrum_commands.add_parser("select", help="Save evenly spaced cached FFT bins")
    spectrum_select.add_argument("--input", required=True, type=Path)
    spectrum_select.add_argument("--count", required=True, type=_positive_int)
    spectrum_select.add_argument("--max-frequency", type=_positive_float)
    spectrum_select.add_argument("--output", required=True, type=Path)
    flow_commands = flow_parser.add_subparsers(dest="flow_command", required=True)
    flow_select = flow_commands.add_parser("select-reference", help="Choose a motion reference after static geometry; save overlays")
    flow_select.add_argument("--scene", required=True, type=Path)
    flow_select.add_argument("--view-label", required=True)
    flow_select.add_argument("--reference", required=True, type=Path)
    flow_select.add_argument("--target-mask", choices=("alpha", "green"), default="alpha",
                             help="Selection silhouette only; green reproduces the reviewed Corn recipe")
    flow_select.add_argument("--workers", type=_positive_int, default=6)
    flow_select.add_argument("--output", required=True, type=Path)
    flow_compute = flow_commands.add_parser("compute", help="Compute full-frame SEA-RAFT flow using local M weights")
    flow_compute.add_argument("--images", required=True, type=Path)
    flow_compute.add_argument("--reference", required=True, type=Path,
                              help="Prepared sequence reference: timing and fixed pixel grid")
    flow_compute.add_argument("--output", required=True, type=Path)
    flow_compute.add_argument("--reference-selection", type=Path, required=True,
                              help="Reviewed select-reference output")
    from modal_gaussians.common.scene_store import library_root
    flow_compute.add_argument("--sea-raft-repo", type=Path, default=library_root() / "_shared/tools/third_party/SEA-RAFT")
    flow_compute.add_argument("--model-dir", type=Path, default=library_root() / "_shared/tools/models/sea-raft-M")
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
    train.add_argument("--iterations", type=_positive_int, default=3_000)
    train.add_argument("--batch-size", type=_positive_int, default=4)
    train.add_argument("--num-fg", type=_positive_int, default=40_000)
    train.add_argument("--num-bg", type=_positive_int, default=80_000)
    train.add_argument("--seed", type=_non_negative_int, default=42)
    train.add_argument(
        "--fg-densify-stop-step", type=_positive_int, default=9_000
    )
    train.add_argument(
        "--bg-densify-stop-step", type=_positive_int, default=9_000
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
    apply_selection = static_commands.add_parser(
        "apply-selection", help="Publish a new static foreground/background split from a saved 3D selection")
    apply_selection.add_argument("--scene", required=True, type=Path)
    apply_selection.add_argument("--selection", required=True, type=Path)
    apply_selection.add_argument("--output", required=True, type=Path)
    graph_parser = command_parsers.add_parser(
        "graph",
        help="Observed foreground-Gaussian structure graph",
    )
    graph_commands = graph_parser.add_subparsers(
        dest="graph_command", required=True
    )
    similarity_graph = graph_commands.add_parser(
        "build-modal-similarity", help="Assign soft weights to KNN candidates using local modal similarity")
    for name in ("prepared", "geometry-graph", "output"):
        similarity_graph.add_argument(f"--{name}", required=True, type=Path)
    similarity_graph.add_argument("--view", required=True, action="append", nargs=2,
                                  metavar=("LABEL", "MODAL_IMAGE_DIR"))
    similarity_graph.add_argument("--frequency", required=True, type=_positive_float)
    for name, default in (("similarity-threshold", 0.20), ("difference-threshold", 0.30),
                          ("amplitude-floor-fraction", 0.02), ("max-pixel-distance", 32.0)):
        similarity_graph.add_argument(f"--{name}", type=_positive_float, default=default)
    similarity_graph.add_argument("--minimum-edge-factor", type=_positive_float, default=0.05,
                                  help="Weight fraction for rejected edges; accepted edges keep weight one")
    motion_parser = command_parsers.add_parser(
        "motion", help="Full-foreground modal motion construction"
    )
    motion_commands = motion_parser.add_subparsers(
        dest="motion_command", required=True
    )
    prepare_neural = motion_commands.add_parser("prepare-neural", help="Freeze reusable neural observations and geometry caches")
    prepare_neural.add_argument('--scene', type=Path, required=True)
    prepare_neural.add_argument('--view', action='append', nargs=3, required=True,
                                metavar=('LABEL', 'REFERENCE', 'DEPTH_TOLERANCE'))
    prepare_neural.add_argument('--config', type=Path)
    prepare_neural.add_argument('--cache-dir', type=Path, required=True)
    prepare_neural.add_argument('--output', type=Path, required=True)
    prepare_selected = motion_commands.add_parser(
        "prepare-selected-modal", help="Prepare selected SEA-RAFT modes, optionally rebuilding a manually selected subject")
    prepare_selected.add_argument("--prepared", type=Path, required=True)
    prepare_selected.add_argument("--view", required=True, action="append", nargs=2,
                                  metavar=("LABEL", "MODAL_IMAGE_DIR"))
    prepare_selected.add_argument("--frequency-hz", type=_positive_float, required=True)
    prepare_selected.add_argument("--output", type=Path, required=True)
    prepare_weights = motion_commands.add_parser("prepare-control-weights", help="Populate exact soft weights before GPU training")
    for name in ("prepared", "geometry-graph", "config", "output"):
        prepare_weights.add_argument(f"--{name}", type=Path, required=True)
    prepare_weights.add_argument("--frequency-hz", type=_positive_float, required=True)
    batch_neural = motion_commands.add_parser("batch-neural", help="Run independent CPU preparation and GPU training queues")
    for name in ("modal-images", "prepared", "geometry-graph", "config", "output"):
        batch_neural.add_argument(f"--{name}", type=Path, required=True)
    batch_neural.add_argument("--cpu-workers", type=_positive_int, default=3,
                              help="Concurrent graph preparation stages")
    batch_neural.add_argument("--gpu-workers", type=_non_negative_int, default=2,
                              help="Concurrent GPU training stages")
    batch_neural.add_argument("--threads-per-worker", type=_positive_int, default=2)
    batch_neural.add_argument("--resume-from", type=Path,
                              help="Carry completed frequencies from a stopped batch into a new attempt")
    batch_neural.add_argument("--continue-from", type=Path,
                              help="Extend a stopped batch's iteration cap, reusing prepared inputs and checkpoints")
    batch_neural.add_argument("--experiment-name", default="experiment_shared_001")
    iterate_neural = motion_commands.add_parser("iterate-neural", help="Produce 3D modes")
    iterate_neural.add_argument("--prepared", type=Path, required=True)
    iterate_neural.add_argument("--config", type=Path)
    iterate_neural.add_argument("--continue-from", type=Path,
                                help="Continue this experiment with a larger iteration cap into a new output")
    iterate_neural.add_argument("--geometry-graph", type=Path,
                                help="Use this saved modal graph; reuse shared control geometry and update its weights")
    iterate_neural.add_argument("--frequency-hz", type=float, action="append",
                                help="Train only this exact prepared frequency; repeat for a subset, preserving source order and normalization")
    iterate_neural.add_argument("--output", type=Path, required=True)
    batch_neural.add_argument("--stage", choices=("weights", "modes"), default="modes")
    coordinates_parser = command_parsers.add_parser(
        "coordinates", help="Rasterized modal design and temporal coordinates"
    )
    coordinates_commands = coordinates_parser.add_subparsers(
        dest="coordinates_command", required=True
    )
    prepare_coordinates = coordinates_commands.add_parser(
        "prepare", help="Collect fixed modes and initialize coordinates from saved SEA-RAFT flow",
    )
    prepare_coordinates.add_argument("--scene", required=True, help="Registered scene name")
    prepare_coordinates.add_argument("--output", required=True, type=Path, help="New scene experiment directory")
    prepare_coordinates.add_argument("--index", type=Path, help="Result index (default: scene results/index.json)")
    prepare_coordinates.add_argument("--status", required=True, help="Exact indexed result status to include")
    prepare_coordinates.add_argument("--expected-modes", required=True, type=_positive_int)
    prepare_coordinates.add_argument("--resume", action="store_true", help="Reuse published stages with an identical contract")
    prepare_coordinates.add_argument("--pixel-stride", type=_positive_int, default=2)
    prepare_coordinates.add_argument("--alpha-min", type=_positive_float, default=0.05)
    prepare_coordinates.add_argument("--mask-erode-iters", type=_non_negative_int, default=1)
    prepare_coordinates.add_argument("--modes-per-batch", type=_positive_int, default=8)
    prepare_coordinates.add_argument("--ridge-relative", type=_positive_float, default=1e-4)
    prepare_coordinates.add_argument("--frame-chunk-size", type=_positive_int, default=64)
    rgb_fit = coordinates_commands.add_parser(
        "fit-rgb", help="Refine direct coefficients against RGB with fixed spatial modes",
    )
    rgb_fit.add_argument("--scene", required=True, type=Path)
    rgb_fit.add_argument("--modes", required=True, type=Path)
    rgb_fit.add_argument("--input", required=True, type=Path, help="Direct-coordinate artifact")
    rgb_fit.add_argument("--output", required=True, type=Path)
    rgb_fit.add_argument("--config", type=Path, help="RGBFitConfig JSON (defaults if omitted)")
    rgb_fit.add_argument("--view", help="Fit only this recorded video label; default: all views")
    rgb_fit.add_argument(
        "--images", action="append", nargs=2, metavar=("LABEL", "DIRECTORY"),
        help="Override a view's recorded/stabilized RGB directory; preserve frame names and geometry",
    )
    rgb_fit.add_argument("--device", default="cuda")
    sweep_fit = coordinates_commands.add_parser("fit-sweep", help="Fit independent sweep coefficients using registered per-frame cameras")
    for name in ("scene", "modes", "scale-source", "metadata", "output"):
        sweep_fit.add_argument("--" + name, required=True, type=Path)
    sweep_fit.add_argument("--config", type=Path)
    sweep_fit.add_argument("--device", default="cuda")
    sweep_fit.add_argument("--fps", type=float, default=30)
    sweep_subset = coordinates_commands.add_parser("downsample-sweep", help="Publish an integer FPS subset of fitted sweep coefficients")
    sweep_subset.add_argument("--input", required=True, type=Path)
    sweep_subset.add_argument("--output", required=True, type=Path)
    sweep_subset.add_argument("--fps", type=float, default=30)
    refinement_prepare = coordinates_commands.add_parser("prepare-refinement", help="Freeze reference graph and validate RGB initialization")
    refinement_prepare.add_argument("--scene", required=True, type=Path)
    refinement_prepare.add_argument("--modes", required=True, type=Path)
    refinement_prepare.add_argument("--coordinates", required=True, action="append", type=Path)
    refinement_prepare.add_argument("--output", required=True, type=Path)
    refinement_prepare.add_argument("--view", action="append")
    refinement_prepare.add_argument("--sweep-coordinates", type=Path)
    refinement_prepare.add_argument("--reference", type=Path)
    refinement_fit = coordinates_commands.add_parser("refine-scene", help="Joint foreground and per-recording coefficient refinement")
    refinement_fit.add_argument("--prepared", required=True, type=Path)
    refinement_fit.add_argument("--config", type=Path)
    refinement_fit.add_argument("--work-dir", required=True, type=Path)
    refinement_fit.add_argument("--output", required=True, type=Path)
    refinement_fit.add_argument("--resume", action="store_true")
    refinement_fit.add_argument("--device", default="cuda")
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
    result_materialize.add_argument("--coordinates", required=True, action="append", type=Path)
    result_materialize.add_argument("--output", required=True, type=Path)
    result_video = result_commands.add_parser(
        "export-video", help="Export one RGB-fitted view: original above reconstruction"
    )
    result_video.add_argument("--result", required=True, type=Path)
    result_video.add_argument("--view", required=True, help="Recorded view label, e.g. view1")
    result_video.add_argument("--output", required=True, type=Path, help="New export directory")
    result_video.add_argument("--device", default="cuda")
    result_evaluate = result_commands.add_parser(
        "evaluate", help="Measure full-resolution reconstruction against fitted input PNGs"
    )
    result_evaluate.add_argument("--result", required=True, type=Path)
    result_evaluate.add_argument("--output", required=True, type=Path)
    result_evaluate.add_argument("--view", action="append", help="Repeat to select views; default all fitted views")
    result_evaluate.add_argument("--lpips", action="store_true", help="Also measure pretrained LPIPS-Alex (evaluation extra)")
    result_evaluate.add_argument("--device", default="cuda")
    result_evaluate.add_argument("--baseline", type=Path, help="Matching evaluation directory; save per-frame metric deltas")
    viewer = command_parsers.add_parser(
        "viewer", help="Inspect modal results, geometry graphs, or select a 3D subject in Viser"
    )
    viewer_input = viewer.add_mutually_exclusive_group(required=True)
    viewer_input.add_argument("--input", type=Path, help="Model, batch index, mode bank or result")
    viewer.add_argument("--coordinates", type=Path, help="Optional fitted coordinates for --input")
    viewer_input.add_argument("--scene", type=Path, help="Inspect static geometry without modal results")
    viewer.add_argument("--geometry-graph", type=Path,
                        help="Geometry cache entry directory (required with --scene unless --select-subject)")
    viewer.add_argument("--select-subject", action="store_true",
                        help="Select full-scene Gaussian centers with a movable 3D box (requires --scene)")
    viewer.add_argument("--selection", type=Path,
                        help="Resume a saved subject selection (requires --select-subject)")
    viewer.add_argument("--work-dir", required=True, type=Path)
    viewer.add_argument("--host", default="0.0.0.0")
    viewer.add_argument("--port", type=_positive_int, default=8080)
    viewer.add_argument("--viewer-res", type=_positive_int, default=2048)
    viewer.add_argument("--no-spectrum", action="store_true",
                        help="Show 3D motion without loading the optional Spectrum panel or FFT sources")
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
        from modal_gaussians.common.scene_store import asset_path, expand_arguments, list_scene, resolve_path
        if args.command == "storage":
            if args.storage_command == "list":
                print(json.dumps(list_scene(args.scene), indent=2, ensure_ascii=False))
            elif args.storage_command == "path":
                print(asset_path(args.scene, args.asset))
            else:
                command = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
                if not command or command[0] == "storage":
                    raise ValueError("Specify an existing pipeline command after --")
                try:
                    return main(expand_arguments(args.scene, command))
                except SystemExit as error:
                    return int(error.code or 0)
            return 0
        # Resolve explicitly typed paths before delegating to existing commands.
        for name, value in vars(args).items():
            if isinstance(value, Path):
                setattr(args, name, resolve_path(value))
        if args.command == "flow" and args.flow_command == "select-reference":
            from modal_gaussians.flow.reference_selection import select_reference
            output = select_reference(scene_dir=args.scene, label=args.view_label,
                reference=args.reference, output_dir=args.output,
                target_mask=args.target_mask, workers=args.workers, command=[parser.prog, *arguments])
            print(f"Motion reference selection: {output}")
            return 0
        if args.command == "flow" and args.flow_command == "compute":
            from modal_gaussians.flow.sea_raft import compute_flow
            output = compute_flow(images=args.images, reference=args.reference,
                                  output_dir=args.output, sea_raft_repo=args.sea_raft_repo,
                                  model_dir=args.model_dir, command=[parser.prog, *arguments],
                                  reference_selection=args.reference_selection)
            print(f"SEA-RAFT flow: {output}")
            return 0
        if args.command == "spectrum":
            from modal_gaussians.spectrum.cache import build_spectrum, export_selection, load_spectrum
            if args.spectrum_command == "build":
                cache = build_spectrum(views=args.view, scene_dir=args.scene,
                                       fft_length=args.nfft, output_dir=args.output, region_paths=args.region)
                print(f"Spectrum: {cache.path}")
                print(f"{len(cache.frequencies)} bins | step={cache.manifest['fps_hz'] / cache.manifest['fft_length']:g} Hz")
            elif args.spectrum_command == "viewer":
                from modal_gaussians.vis.spectrum_viewer import run_spectrum_viewer
                run_spectrum_viewer(spectrum_dir=args.input, work_dir=args.work_dir,
                                    host=args.host, port=args.port)
            elif args.spectrum_command == "select":
                from modal_gaussians.spectrum.selection import select_spectrum
                output = select_spectrum(spectrum_dir=args.input, count=args.count,
                                         max_frequency_hz=args.max_frequency, output_dir=args.output)
                print(f"Frequency selection: {output}")
            else:
                output = export_selection(load_spectrum(args.input), args.selection, args.output)
                print(f"Selected modal images: {output}")
            return 0
        if args.command == 'prepare' and args.prepare_command == 'reference':
            from modal_gaussians.preprocessing.reference import prepare_reference
            from modal_gaussians.preprocessing.stabilization import StabilizationSettings
            artifact = prepare_reference(images=args.images, masks=args.masks, fps=args.fps,
                reference_frame=args.reference_frame, output_dir=args.output, tripod=args.tripod,
                scene_dir=args.scene, label=args.view,
                settings=(StabilizationSettings(
                    **json.loads(args.config.read_text(encoding='utf-8'))) if args.config else None))
            print(artifact.path)
            return 0
        if args.command == "prepare" and args.prepare_command == "gui":
            try:
                from modal_gaussians.preprocessing.mask_gui import run_mask_gui
            except ImportError as error:
                raise RuntimeError("Mask preparation requires the optional [mask] dependencies; see README") from error
            run_mask_gui(root_dir=args.root_dir, checkpoint_dir=args.checkpoint_dir,
                         port=args.port, device=args.device)
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
            from modal_gaussians.geometry.training import StaticTrainConfig, run_static_training

            output = run_static_training(
                input_dir=args.input,
                work_dir=args.work_dir,
                output_dir=args.output,
                config=StaticTrainConfig(
                    iterations=int(args.iterations),
                    batch_size=int(args.batch_size),
                    num_foreground=int(args.num_fg),
                    num_background=int(args.num_bg),
                    seed=int(args.seed),
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
            from modal_gaussians.geometry.partition import PartitionConfig, repartition_static_scene

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
        if args.command == "static" and args.static_command == "apply-selection":
            from modal_gaussians.geometry.partition import apply_subject_selection

            output = apply_subject_selection(scene_dir=args.scene, selection_path=args.selection,
                output_dir=args.output, command=[parser.prog, *arguments])
            print(f"selected subject static scene: {output.resolve()}")
            print("Gaussian parameters preserved; rebuild observations and graph for this foreground")
            return 0
        if args.command == "static" and args.static_command == "render":
            from modal_gaussians.geometry.training import render_static_bundle

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
        if args.command == "graph" and args.graph_command == "build-modal-similarity":
            from modal_gaussians.motion.modal_similarity import ModalSimilarityConfig
            from modal_gaussians.motion.modal_similarity_artifact import build_modal_similarity_graph_artifact

            manifest = build_modal_similarity_graph_artifact(
                prepared_dir=args.prepared, geometry_graph_dir=args.geometry_graph,
                views=args.view, frequency_hz=args.frequency, output_dir=args.output,
                config=ModalSimilarityConfig(similarity_threshold=args.similarity_threshold,
                    difference_threshold=args.difference_threshold,
                    amplitude_floor_fraction=args.amplitude_floor_fraction,
                    max_pixel_distance=args.max_pixel_distance,
                    minimum_edge_factor=args.minimum_edge_factor),
                command=[parser.prog, *arguments])
            print(f"Modal-similarity graph: {args.output.resolve()}")
            print(json.dumps(manifest["summary"], indent=2))
            return 0
        if args.command == "motion" and args.motion_command == "prepare-neural":
            from modal_gaussians.motion.prepared import prepare_neural
            overrides = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            from modal_gaussians.motion.training import NeuralModesConfig
            from modal_gaussians.motion.component_field import ComponentFieldConfig
            from modal_gaussians.motion.iteration import resolve_config
            defaults = {'neural': NeuralModesConfig().to_dict(), 'fragment': ComponentFieldConfig().to_dict()}
            overrides = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
            settings = resolve_config(defaults, overrides)
            artifact = prepare_neural(scene_dir=args.scene, views=args.view, output_dir=args.output,
                cache_dir=args.cache_dir, config=NeuralModesConfig.from_dict(settings['neural']))
            print(f'Prepared geometry: {artifact.path}')
            print(f'Candidate graph: {artifact.manifest["geometry_graph"]}')
            return 0
        if args.command == "motion" and args.motion_command == "prepare-selected-modal":
            from modal_gaussians.motion.selected_modal import prepare_selected_modal
            artifact = prepare_selected_modal(prepared_dir=args.prepared, views=args.view,
                frequency_hz=args.frequency_hz, output_dir=args.output,
                alpha_backend="cupy")
            print(f"Selected-modal prepared: {artifact.path}")
            print(f"prepared_identity: {artifact.manifest['prepared_identity']}")
            return 0
        if args.command == "motion" and args.motion_command == "batch-neural":
            from modal_gaussians.motion.batch import run_batch
            path = run_batch(modal_images=args.modal_images, prepared_dir=args.prepared,
                geometry_graph_dir=args.geometry_graph, config_path=args.config, output_dir=args.output,
                cpu_workers=args.cpu_workers, gpu_workers=args.gpu_workers, threads_per_worker=args.threads_per_worker,
                 resume_from=args.resume_from,
                continue_from=args.continue_from,
                propagation_backend="cupy", alpha_backend="cupy", stage=args.stage,
                experiment_name=args.experiment_name)
            print(f"Batch {args.stage} ready: {path}")
            return 0
        if args.command == "motion" and args.motion_command == "prepare-control-weights":
            from modal_gaussians.motion.control_preparation import prepare_control_weights
            path = prepare_control_weights(prepared_dir=args.prepared, geometry_graph_dir=args.geometry_graph,
                config_path=args.config, frequency_hz=args.frequency_hz, output_path=args.output,
                backend="cupy")
            print(f"Control weights ready: {path}")
            return 0
        if args.command == "motion" and args.motion_command == "iterate-neural":
            from modal_gaussians.motion.iteration import iterate_neural
            root = iterate_neural(prepared_dir=args.prepared, config_path=args.config,
                                  output_dir=args.output, frequencies_hz=args.frequency_hz,
                                  geometry_graph_dir=args.geometry_graph,
                                  continue_from=args.continue_from,
                                  propagation_backend="cupy")
            print(f"iteration: {root}")
            return 0
        if args.command == "coordinates" and args.coordinates_command == "prepare":
            from modal_gaussians.coordinates.preparation import prepare_coefficient_inputs
            from modal_gaussians.coordinates.design import RenderedDesignConfig
            from modal_gaussians.coordinates.direct import DirectCoordinateConfig

            artifact = prepare_coefficient_inputs(
                scene=args.scene, output_dir=args.output, index_path=args.index, status=args.status,
                expected_modes=args.expected_modes, resume=args.resume,
                design_config=RenderedDesignConfig(pixel_sample_stride=args.pixel_stride,
                    alpha_minimum=args.alpha_min, mask_erosion_iterations=args.mask_erode_iters,
                    modes_per_batch=args.modes_per_batch),
                direct_config=DirectCoordinateConfig(ridge_relative=args.ridge_relative,
                    frame_chunk_size=args.frame_chunk_size),
            )
            print(f"Coefficient inputs prepared: {artifact.path.parent}")
            print("RGB fitting has not been started.")
            return 0
        if args.command == "coordinates" and args.coordinates_command == "fit-rgb":
            from dataclasses import fields
            from modal_gaussians.coordinates.rgb import build_rgb_modal_coordinates_artifact
            from modal_gaussians.coordinates.fitting import RGBFitConfig

            payload = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            if not isinstance(payload, dict) or set(payload) - {field.name for field in fields(RGBFitConfig)}:
                raise ValueError("RGB config must be an object containing only RGBFitConfig fields")
            image_pairs = args.images or []
            if len({label for label, _ in image_pairs}) != len(image_pairs):
                raise ValueError("RGB image override labels must be unique")
            artifact = build_rgb_modal_coordinates_artifact(
                scene_dir=args.scene, completed_modes_dir=args.modes,
                direct_coordinates_dir=args.input, output_dir=args.output,
                config=RGBFitConfig(**payload), image_directories=dict(image_pairs),
                view_label=args.view,
                device=args.device, command=[parser.prog, *arguments],
            )
            counts = artifact.manifest["counts"]
            print(f"RGB modal coordinates: {artifact.path.resolve()}")
            print(f"views/frames/modes: {counts['views']}/{counts['frames']}/{counts['modes']}")
            print(f"identity: {artifact.manifest['rgb_coordinates_identity']}")
            return 0
        if args.command == "coordinates" and args.coordinates_command == "fit-sweep":
            from dataclasses import fields
            from modal_gaussians.coordinates.sweep import fit_sweep
            from modal_gaussians.coordinates.fitting import RGBFitConfig
            payload = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            if not isinstance(payload, dict) or set(payload) - {f.name for f in fields(RGBFitConfig)}:
                raise ValueError('RGB config must contain only RGBFitConfig fields')
            artifact = fit_sweep(scene_dir=args.scene, completed_modes_dir=args.modes,
                scale_source=args.scale_source, metadata_path=args.metadata, output_dir=args.output,
                config=RGBFitConfig(**payload), device=args.device, fps=args.fps)
            print(f"Sweep RGB coordinates: {artifact.path}")
            return 0
        if args.command == "coordinates" and args.coordinates_command == "downsample-sweep":
            from modal_gaussians.coordinates.sweep import downsample_sweep
            artifact = downsample_sweep(input_dir=args.input, output_dir=args.output, fps=args.fps)
            print(f"Sweep frame subset: {artifact.path}")
            return 0
        if args.command == "coordinates" and args.coordinates_command == "prepare-refinement":
            from modal_gaussians.coordinates.refinement_artifacts import prepare_refinement
            output = prepare_refinement(scene_dir=args.scene, completed_modes_dir=args.modes,
                coordinates_dirs=args.coordinates, output_dir=args.output, view_labels=args.view,
                sweep_coordinates_dir=args.sweep_coordinates, reference_dir=args.reference)
            print(f"Refinement inputs prepared: {output}")
            return 0
        if args.command == "coordinates" and args.coordinates_command == "refine-scene":
            from dataclasses import fields
            from modal_gaussians.coordinates.refinement import RefinementConfig, refine_scene
            payload = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            if not isinstance(payload, dict) or set(payload) - {f.name for f in fields(RefinementConfig)}:
                raise ValueError("Refinement config must contain only RefinementConfig fields")
            output = refine_scene(prepared_dir=args.prepared, config=RefinementConfig(**payload),
                work_dir=args.work_dir, output_dir=args.output, resume=args.resume, device=args.device)
            print(f"Refined scene bundle: {output}")
            return 0
        if args.command == "result" and args.result_command == "materialize":
            from modal_gaussians.results.artifact import materialize_modal_result

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
        if args.command == "result" and args.result_command == "export-video":
            from modal_gaussians.results.video import export_result_video

            video = export_result_video(result_dir=args.result, view_label=args.view,
                                        output_dir=args.output, device=args.device)
            print(f"comparison video: {video}")
            return 0
        if args.command == "result" and args.result_command == "evaluate":
            from modal_gaussians.results.evaluation import evaluate_result

            output = evaluate_result(result_dir=args.result, output_dir=args.output,
                                     view_labels=args.view, with_lpips=args.lpips, device=args.device, baseline_dir=args.baseline)
            print(f"reconstruction metrics: {output / 'metrics.json'}")
            return 0
        if args.command == "viewer":
            if args.coordinates is not None and args.scene is not None:
                parser.error("--coordinates requires modal input")
            if args.selection is not None and not args.select_subject:
                parser.error("--selection requires --select-subject")
            if args.select_subject:
                if args.scene is None:
                    parser.error("--select-subject requires --scene instead of --input")
                if args.geometry_graph is not None:
                    parser.error("--select-subject cannot be combined with --geometry-graph")
                from modal_gaussians.vis.subject_selection_viewer import run_subject_selection_viewer

                run_subject_selection_viewer(
                    scene_dir=args.scene, work_dir=args.work_dir, selection_path=args.selection,
                    host=str(args.host), port=int(args.port),
                    viewer_resolution=int(args.viewer_res),
                )
                return 0
            if (args.scene is None) != (args.geometry_graph is None):
                parser.error("--scene and --geometry-graph must be supplied together")
            if args.scene is not None:
                from modal_gaussians.vis.graph_viewer import run_graph_viewer

                run_graph_viewer(scene_dir=args.scene, graph_dir=args.geometry_graph,
                                 work_dir=args.work_dir, host=str(args.host), port=int(args.port),
                                 viewer_resolution=int(args.viewer_res))
                return 0
            from modal_gaussians.vis.viewer import run_modal_viewer

            run_modal_viewer(
                with_spectrum=not args.no_spectrum,
                result_dir=args.input,
                work_dir=args.work_dir,
                host=str(args.host),
                port=int(args.port),
                viewer_resolution=int(args.viewer_res),
                coordinates=args.coordinates,
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
