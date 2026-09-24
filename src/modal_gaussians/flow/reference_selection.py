"""Choose a motion reference on the fixed camera grid, independently of geometry."""
from concurrent.futures import ThreadPoolExecutor
import csv
import time

import cv2
import numpy as np

from modal_gaussians.common.cache import atomic_json, identity
from modal_gaussians.common.progress import report_progress
from modal_gaussians.common.scene_store import resolve_path

FORMAT = "modal_gaussians.motion_reference_selection"
KERNEL = np.ones((3, 3), np.uint8)


def sequence_metadata(reference_dir, images=None):
    from modal_gaussians.preprocessing.reference import load_reference
    reference = load_reference(reference_dir)
    root, source = reference.path, reference.manifest
    raw = resolve_path(source['inputs']['sequence']['image_directory'], strict=True)
    if images is not None and resolve_path(images) != raw:
        raise ValueError('Images differ from the recorded sequence')
    if source['stabilized_sequence'] is not None:
        image_dir, mask_dir = root / 'stabilized_sequence/images', root / 'stabilized_sequence/masks'
    else:
        image_dir = raw
        mask_dir = resolve_path(source['inputs']['sequence']['mask_directory'], strict=True)
    names = source['frame_names']
    if (sorted(p.stem for p in image_dir.glob('*.png')) != names
            or sorted(p.stem for p in mask_dir.glob('*.png')) != names):
        raise ValueError('Sequence image/mask inventory differs')
    return dict(root=root, source=source, images=raw, image_dir=image_dir, mask_dir=mask_dir,
                names=names, fps=float(source['fps_hz']), reference=source['reference_frame_index'],
                shape_hw=source['shape_hw'], valid_mask=reference.arrays.valid_mask)


def make_contract(sequence, scene, view, selected, settings):
    return {"format": FORMAT, "version": 1, "implementation": "silhouette_distance_v1",
        "geometry_source_identity": identity(sequence["source"]),
        "static_scene_identity": scene["static_scene_identity"],
        "foreground_identity": scene["foreground_identity"],
        "label": view["label"], "camera_identity": view["camera_identity"],
        "shape_hw": list(sequence["shape_hw"]), "fps_hz": sequence["fps"],
        "frame_count": len(sequence["names"]),
        "reference_frame_name": selected["frame"], "reference_frame_index": int(selected["index"]),
        "settings": settings}


def reference_binding(selection):
    contract = selection["contract"]
    if (selection.get("format") != FORMAT or selection.get("version") != 1
            or selection.get("status") != "complete" or contract.get("format") != FORMAT
            or contract.get("version") != 1 or selection.get("selection_identity") != identity(contract)):
        raise ValueError("Incomplete or incompatible motion-reference selection")
    return {"identity": selection["selection_identity"], "contract": contract}


def motion_reference(manifest, frozen, *, view=None, static_scene_identity=None):
    """Check the explicit motion-reference binding to the fixed pixel grid."""
    name, index = manifest["reference_frame_name"], manifest["reference_frame_index"]
    names = frozen["frame_names"]
    if type(index) is not int or not 0 <= index < len(names) or names[index] != name:
        raise ValueError("Motion reference is outside the original sequence")
    binding = manifest.get("reference_selection")
    if binding is None:
        raise ValueError("A motion-reference selection is required")
    contract = binding.get("contract", {})
    if (binding.get("identity") != identity(contract) or contract.get("format") != FORMAT
            or contract.get("version") != 1 or contract.get("implementation") != "silhouette_distance_v1"
            or contract.get("geometry_source_identity") != identity(frozen)
            or contract.get("reference_frame_name") != name or contract.get("reference_frame_index") != index
            or contract.get("fps_hz") != frozen["fps_hz"] or contract.get("frame_count") != len(names)
            or contract.get("shape_hw") != frozen["shape_hw"]):
        raise ValueError("Motion reference does not match its geometry/sequence contract")
    if view is not None and any(contract.get(k) != view[k] for k in ("label", "camera_identity", "shape_hw")):
        raise ValueError("Motion reference camera/view differs")
    if static_scene_identity is not None and contract.get("static_scene_identity") != static_scene_identity:
        raise ValueError("Motion reference static geometry differs; select again")
    return name, index


def _read(path, gray=False):
    if path is None:
        raise ValueError("Reference selection requires existing per-frame subject masks")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot read {path}")
    return image


def _write(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write {path}")


def selection_mask(rgb, alpha, mode):
    """Selection-only approximation; never used to clip flow or training pixels."""
    mask = alpha >= .05
    if mode == "green":
        # Corn's manual Gaussian selection includes soil. These are the reviewed
        # Corn thresholds; use alpha for flowers or non-green subjects.
        r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
        mask &= (g-r >= 3) & (g-b >= 3) & (2*g-r-b >= 12)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, KERNEL)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.zeros(count, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 50
        mask = keep[labels]
    elif mode != "alpha":
        raise ValueError("Target mask must be alpha or green")
    return mask.astype(bool)


def boundary(mask):
    return mask & (cv2.erode(mask.astype(np.uint8), KERNEL,
                  borderType=cv2.BORDER_CONSTANT, borderValue=0) == 0)


def target_data(mask):
    edge = boundary(mask)
    if not edge.any():
        raise ValueError("Empty selection silhouette")
    yy, xx = np.nonzero(edge)
    distance = cv2.distanceTransform(np.uint8(~edge), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return mask, edge, distance, yy, xx


def score_mask(mask, target):
    fixed, _, fixed_distance, yy, xx = target
    if mask.shape != fixed.shape or not mask.any():
        raise ValueError("Empty mask or mask dimensions differ")
    edge = boundary(mask)
    y, x = np.nonzero(edge)
    x0, x1 = min(x.min(), xx.min()), max(x.max(), xx.max()) + 1
    y0, y1 = min(y.min(), yy.min()), max(y.max(), yy.max()) + 1
    distance = cv2.distanceTransform(np.uint8(~edge[y0:y1, x0:x1]), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    forward, backward = fixed_distance[edge], distance[yy-y0, xx-x0]
    return {"boundary_mean_px": float(.5 * (forward.mean() + backward.mean())),
        "boundary_p95_px": float(np.percentile(np.concatenate((forward, backward)), 95)),
        "iou": float((mask & fixed).sum() / (mask | fixed).sum())}


def _panel(image, title):
    output = cv2.copyMakeBorder(image, 38, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
    cv2.putText(output, title, (9, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (245, 245, 245), 1, cv2.LINE_AA)
    return output


def write_previews(output, render_rgb, fixed, old_rgb, new_rgb, old, selected, old_mask, new_mask):
    """Pure pixel overlays: RGB input/output, no optical flow or image warping."""
    rgb = [render_rgb, old_rgb, new_rgb]
    if any(im.shape != (*fixed.shape, 3) for im in rgb):
        raise ValueError("Reference RGB dimensions differ from Gaussian render")
    g, old_rgb, new_rgb = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in rgb]
    a, b = cv2.addWeighted(g, .5, old_rgb, .5, 0), cv2.addWeighted(g, .5, new_rgb, .5, 0)
    for name, im in (("render_rgb.png", g), ("old_reference.png", old_rgb),
                     ("selected_reference.png", new_rgb), ("old_overlay.png", a), ("selected_overlay.png", b)):
        _write(output / name, im)
    _write(output / "selection_render_mask.png", np.uint8(fixed)*255)
    _write(output / "selected_mask.png", np.uint8(new_mask)*255)
    _write(output / "comparison_full.png", np.concatenate((
        np.concatenate((_panel(g, "Static Gaussian RGB"), _panel(new_rgb, f"Selected {selected['frame']}")), axis=1),
        np.concatenate((_panel(a, f"Old {old['frame']} + G | 50/50; no warp"),
                        _panel(b, f"New {selected['frame']} + G | 50/50; no warp")), axis=1)), axis=0))
    yy, xx = np.nonzero(fixed)
    crop = np.s_[max(0, yy.min()-40):min(fixed.shape[0], yy.max()+41),
                 max(0, xx.min()-40):min(fixed.shape[1], xx.max()+41)]
    detail = np.concatenate((
        np.concatenate((_panel(old_rgb[crop], f"Old {old['frame']} | {old['time_seconds']:.2f}s"),
                        _panel(new_rgb[crop], f"New {selected['frame']} | {selected['time_seconds']:.2f}s")), axis=1),
        np.concatenate((_panel(a[crop], "Old + Gaussian | 50/50; no warp"),
                        _panel(b[crop], "New + Gaussian | 50/50; no warp")), axis=1)), axis=0)
    _write(output / "comparison_subject.png", detail)
    panels = []
    fixed_edge = boundary(fixed)
    for record, im, mask in ((old, old_rgb, old_mask), (selected, new_rgb, new_mask)):
        im = (im*.55).astype(np.uint8)
        edge = boundary(mask)
        im[fixed_edge], im[edge], im[fixed_edge & edge] = (0,255,0), (255,0,255), (255,255,255)
        panels.append(_panel(im[crop], f"{record['frame']} | edge gap {record['boundary_mean_px']:.2f}px"))
    _write(output / "contours.png", np.concatenate(panels, axis=1))


def select_reference(*, scene_dir, label, reference, output_dir, target_mask="alpha", workers=6,
                     command=None):
    started = time.perf_counter()
    from modal_gaussians.geometry.scene import load_static_scene, cameras_from_scene_manifest
    import torch

    destination = resolve_path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if type(workers) is not int or workers < 1 or target_mask not in ("alpha", "green"):
        raise ValueError("Invalid selector settings")
    seq = sequence_metadata(reference)
    if seq["mask_dir"] is None:
        raise ValueError("Reference selection requires existing per-frame subject masks")
    scene = load_static_scene(scene_dir, device="cuda", validate=False).eval()
    scene.requires_grad_(False)
    camera = next((c for c in cameras_from_scene_manifest(scene.manifest)
                   if c.role == "reference" and c.label == label), None)
    if camera is None or [camera.height, camera.width] != seq["shape_hw"]:
        raise ValueError("Reference camera label/shape differs")
    stable = seq['source']['stabilized_sequence']
    if stable is not None and (stable['static_scene_identity'] != scene.manifest['static_scene_identity']
            or stable['target_camera']['camera_identity'] != camera.to_manifest_record()['camera_identity']):
        raise ValueError('Stabilization target camera/static scene differs; prepare again')
    # One required render supplies RGB and visible subject coverage together.
    with torch.no_grad():
        rendered, _ = scene.render_batch([camera.to("cuda")], return_foreground_mask=True)
        rgb = (rendered["rgb"][0].clamp(0,1)*255).round().to(torch.uint8).cpu().numpy()
        alpha = rendered["foreground_mask"][0].cpu().numpy()
    fixed = selection_mask(rgb, alpha, target_mask) & seq['valid_mask']
    target = target_data(fixed)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"format": FORMAT, "version": 1, "status": "running", "command": command,
        "scene": str(resolve_path(scene_dir)), "geometry_source": str(seq["root"]),
        "images": str(seq["images"]), "inference_images": str(seq["image_dir"]),
        "masks": str(seq["mask_dir"]), "pending_user_review": True,
        "ranking": "symmetric mean boundary distance in native pixels; IoU tie-break",
        "contours_legend": "green=Gaussian, magenta=video, white=overlap",
        "no_geometric_warp": True, "reference_binding_changed": False}
    atomic_json(destination / "manifest.json", manifest)
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        def evaluate(item):
            index, name = item
            mask = (_read(seq["mask_dir"] / f"{name}.png", True) > 0) & seq['valid_mask']
            return {"index": index, "frame": name, "time_seconds": index/seq["fps"], **score_mask(mask, target)}
        records = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for record in pool.map(evaluate, enumerate(seq["names"])):
                records.append(record)
                if len(records) % 200 == 0:
                    report_progress(f"{label}: ranked {len(records)}/{len(seq['names'])} frames")
        records.sort(key=lambda r: (r["boundary_mean_px"], -r["iou"], r["index"]))
        selected = records[0]
        old = next(r for r in records if r["index"] == seq["reference"])
        with (destination / "scores.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["rank", *selected])
            writer.writeheader()
            writer.writerows({"rank": i+1, **r} for i, r in enumerate(records))
        old_rgb, new_rgb = [cv2.cvtColor(_read(seq["image_dir"] / f"{r['frame']}.png"), cv2.COLOR_BGR2RGB)
                            for r in (old, selected)]
        old_mask, new_mask = [_read(seq["mask_dir"] / f"{r['frame']}.png", True)>0 for r in (old, selected)]
        write_previews(destination, rgb, fixed, old_rgb, new_rgb, old, selected, old_mask, new_mask)
        contract = make_contract(seq, scene.manifest, {"label": label,
            "camera_identity": camera.to_manifest_record()["camera_identity"]}, selected,
            {"target_mask": target_mask, "alpha_minimum": .05, "mask_recipe": "reviewed_corn_v1" if target_mask=="green" else "alpha_v1"})
        manifest.update(status="complete", contract=contract, selection_identity=identity(contract),
            selected=selected, old=old, old_rank=records.index(old)+1, frames_scored=len(records),
            elapsed_seconds=time.perf_counter()-started)
        atomic_json(destination / "manifest.json", manifest)
        report_progress(f"{label}: selected {selected['frame']} ({selected['time_seconds']:.2f}s); review {destination}")
        return destination
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        atomic_json(destination / "manifest.json", manifest)
        raise
    finally:
        cv2.setNumThreads(previous_threads)
