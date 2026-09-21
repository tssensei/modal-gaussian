"""Render static reference views and preview one-way SEA-RAFT registration.

Run from the repository's Python environment. This experiment does not update
the scene, modal observations, alpha, graphs, or training artifacts.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from modal_gaussians.iteration_cache import atomic_json
from modal_gaussians.progress import progress_log, report_progress
from modal_gaussians.scene_store import asset_path, library_root, resolve_path


def warp_reference(reference, flow):
    """Gather reference at p + flow_G_to_R(p); never negate a forward flow."""
    if reference.ndim != 3 or reference.shape[2] != 3 or flow.shape != (*reference.shape[:2], 2):
        raise ValueError("Expected RGB [H,W,3] and flow [H,W,2] on the same grid")
    if not np.isfinite(flow).all():
        raise ValueError("Registration flow contains non-finite values")
    h, w = reference.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    map_x, map_y = x + flow[..., 0], y + flow[..., 1]
    valid = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    warped = cv2.remap(reference, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, valid


def flow_colors(flow, limit, colorize):
    """Use SEA-RAFT's color wheel with one shared scale; clip only display."""
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Display limit must be positive and finite")
    magnitude = np.linalg.norm(flow, axis=-1)
    normalized = flow / np.maximum(limit, magnitude)[..., None]
    return colorize(normalized[..., 0], normalized[..., 1])


def write_rgb(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write {path}")


def comparison(render, reference, flow_rgb, warped, valid, label, limit, colorize):
    h, w = render.shape[:2]
    overlay = np.rint(.5 * render.astype(np.float32) + .5 * warped).astype(np.uint8)
    y, x = np.indices((h, w))
    outside = ~valid
    checker = ((x // 8 + y // 8) % 2).astype(bool)
    overlay[outside & checker] = [255, 0, 255]
    overlay[outside & ~checker] = [40, 40, 40]

    def panel(rgb, title):
        result = cv2.copyMakeBorder(rgb, 42, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(result, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .72, (245, 245, 245), 1, cv2.LINE_AA)
        return result

    top = np.concatenate((panel(render, "Static 3DGS RGB (G)"),
                          panel(reference, "Video reference RGB (R)")), axis=1)
    bottom = np.concatenate((panel(flow_rgb, "SEA-RAFT: G -> R | displacement in pixels"),
        panel(overlay, "50% G + 50% R(p + flow) | magenta checker = out of bounds")), axis=1)
    sheet = np.concatenate((top, bottom), axis=0)
    footer = np.full((186, 2 * w, 3), 30, np.uint8)
    # Direction wheel uses image coordinates: +x right, +y down.
    yy, xx = np.mgrid[-64:65, -64:65].astype(np.float32)
    wheel = colorize(xx / 64, yy / 64)
    wheel[np.hypot(xx, yy) > 64] = 30
    footer[25:154, 50:179] = wheel
    for text, pos in (("-y", (100, 18)), ("+y", (100, 175)),
                      ("-x", (12, 94)), ("+x", (186, 94))):
        cv2.putText(footer, text, pos, cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
    ramp = np.zeros((22, 460, 2), np.float32)
    ramp[..., 0] = np.linspace(0, limit, 460)
    footer[70:92, 250:710] = flow_colors(ramp, limit, colorize)
    for text, pos in (
        (f"{label} | shared saturation scale: {limit:.2f} px", (250, 35)),
        ("0 px", (250, 115)), (f"{limit / 2:.2f}", (453, 115)), (f">= {limit:.2f} px", (670, 115)),
        ("Hue = direction; saturation = magnitude; white = zero displacement.", (250, 145)),
        ("Single direction only. In-bounds is NOT a confidence or occlusion mask.", (250, 173))):
        cv2.putText(footer, text, pos, cv2.FONT_HERSHEY_SIMPLEX, .63, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate((sheet, footer), axis=0)


def run(args):
    import torch
    from modal_gaussians.flow.sea_raft import load_model, read_image
    from modal_gaussians.static import cameras_from_scene_manifest, load_static_scene

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"format": "modal_gaussians.reference_registration_preview", "version": 1,
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "scene": args.scene, "direction": "3dgs_render_to_video_reference",
        "flow_units": "pixels", "warp": "reference(p + flow_G_to_R(p))",
        "reverse_flow_computed": False, "confidence_computed": False,
        "validation": False, "views": []}
    atomic_json(output / "manifest.json", manifest)
    with progress_log(output / "run.log"):
        try:
            report_progress(f"START {sys.argv}")
            if not torch.cuda.is_available():
                raise RuntimeError("This experiment requires CUDA")
            scene_path = asset_path(args.scene, "static")
            scene = load_static_scene(scene_path, device="cuda", validate=False).eval()
            scene.requires_grad_(False)
            cameras = [c for c in cameras_from_scene_manifest(scene.manifest) if c.role == "reference"]
            if not cameras:
                raise ValueError("Static scene has no reference cameras")
            repo = library_root() / "_shared/tools/third_party/SEA-RAFT"
            weights = library_root() / "_shared/tools/models/sea-raft-M"
            model = load_model(repo, weights)
            from utils.flow_viz import flow_uv_to_colors

            manifest.update(static_scene=str(scene_path), static_scene_identity=scene.manifest["static_scene_identity"],
                model={"repository": str(repo), "weights": str(weights), "config": vars(model.args)},
                rendering="all foreground and background; unchanged saved reference camera convention",
                spatial_processing="native resolution; no crop, resize, image mask, or flow smoothing")
            products = []
            for camera in cameras:
                if not camera.label or not camera.label.startswith("view") or not camera.label[4:].isdigit():
                    raise ValueError(f"Expected viewN camera label: {camera.label!r}")
                label = camera.label
                source = asset_path(args.scene, "flow" + label[4:])
                metadata = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
                reference_path = resolve_path(metadata["reference_image"], strict=True)
                target_cpu = read_image(reference_path)
                if tuple(target_cpu.shape) != (1, 3, camera.height, camera.width):
                    raise ValueError(f"Reference shape differs from saved camera: {label}")
                reference = target_cpu[0].permute(1, 2, 0).numpy().astype(np.uint8)
                destination = output / label
                destination.mkdir()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.no_grad():
                    rendered = scene.render(camera.to("cuda"), composition="all", outputs=("rgb",))["rgb"]
                    render = (rendered.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                    source_gpu = torch.from_numpy(render).permute(2, 0, 1)[None].float().cuda()
                    torch.cuda.synchronize()
                    render_seconds = time.perf_counter() - tick
                    report_progress(f"{label}: rendered {camera.width}x{camera.height}; running G -> R")
                    tick = time.perf_counter()
                    result = model(source_gpu, target_cpu.cuda(), iters=model.args.iters, test_mode=True)
                    flow = result["final"][0].permute(1, 2, 0).float().cpu().numpy()
                    torch.cuda.synchronize()
                    inference_seconds = time.perf_counter() - tick
                    del result, source_gpu, rendered
                warped, valid = warp_reference(reference, flow)
                with (destination / "flow.npy").open("xb") as f:
                    np.save(f, flow, allow_pickle=False)
                with (destination / "in_bounds.npy").open("xb") as f:
                    np.save(f, valid, allow_pickle=False)
                write_rgb(destination / "render_rgb.png", render)
                products.append((label, render, reference, flow, warped, valid))
                manifest["views"].append({"label": label, "reference_image": str(reference_path),
                    "reference_frame_name": metadata["reference_frame_name"],
                    "camera": camera.to_manifest_record(), "shape_hw": [camera.height, camera.width],
                    "flow_file": f"{label}/flow.npy", "flow_dtype": str(flow.dtype),
                    "in_bounds_file": f"{label}/in_bounds.npy", "render_seconds": render_seconds,
                    "inference_seconds": inference_seconds})
                atomic_json(output / "manifest.json", manifest)
                report_progress(f"{label}: flow saved; inference {inference_seconds:.3f}s")

            limit = args.display_max_px
            if limit is None:
                limit = max(1., *(float(np.percentile(np.linalg.norm(p[3], axis=-1), 99)) for p in products))
            manifest["display"] = {"max_displacement_px": limit,
                "scale_rule": "explicit" if args.display_max_px else "max of per-view full-frame P99; minimum 1 px",
                "color_wheel": "SEA-RAFT Middlebury; white zero; display-only saturation above limit",
                "overlay": "50% rendered RGB + 50% sampled reference; out-of-bounds magenta checker"}
            for label, render, reference, flow, warped, valid in products:
                colors = flow_colors(flow, limit, flow_uv_to_colors)
                write_rgb(output / label / "flow_color.png", colors)
                write_rgb(output / label / "comparison.png",
                          comparison(render, reference, colors, warped, valid, label, limit, flow_uv_to_colors))
            manifest.update(status="complete", pending_user_review=True,
                            elapsed_seconds=time.perf_counter() - started)
            atomic_json(output / "manifest.json", manifest)
            report_progress(f"COMPLETE {output}; display limit {limit:.2f} px; pending user review")
        except BaseException as error:
            manifest.update(status="failed", error=str(error), elapsed_seconds=time.perf_counter() - started)
            atomic_json(output / "manifest.json", manifest)
            report_progress(f"FAILED {error}")
            raise


def self_test():
    """Synthetic translation checks direction, bilinear sampling and bounds."""
    image = np.repeat(np.arange(8, dtype=np.float32)[None, :, None], 4, axis=0)
    image = np.repeat(image, 3, axis=2)
    flow = np.zeros((4, 8, 2), np.float32)
    warped, valid = warp_reference(image, flow)
    np.testing.assert_array_equal(warped, image)
    assert valid.all()
    flow[..., 0] = 1.5
    warped, valid = warp_reference(image, flow)
    np.testing.assert_allclose(warped[:, :6, 0], np.tile(np.arange(6) + 1.5, (4, 1)))
    assert valid[:, :6].all() and not valid[:, 6:].any()
    flow[..., 0], flow[..., 1] = 0, -1
    _, valid = warp_reference(image, flow)
    assert not valid[0].any() and valid[1:].all()
    normalized = []
    def colorize(u, v):
        normalized.append(np.stack((u, v), -1))
        return np.zeros((*u.shape, 3), np.uint8)
    vectors = np.array([[[0., 0.], [3., 4.], [6., 8.]]], np.float32)
    original = vectors.copy()
    flow_colors(vectors, 5., colorize)
    np.testing.assert_allclose(normalized[0], [[[0., 0.], [.6, .8], [.6, .8]]])
    np.testing.assert_array_equal(vectors, original)
    print("Synthetic warp direction, interpolation, bounds and display clipping checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="corn", choices=("corn", "bush"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--display-max-px", type=float, help="Shared display saturation limit in pixels")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.output is None:
            parser.error("--output is required")
        if args.display_max_px is not None and (not np.isfinite(args.display_max_px) or args.display_max_px <= 0):
            parser.error("--display-max-px must be positive and finite")
        run(args)
