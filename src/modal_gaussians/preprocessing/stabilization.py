"""Fixed-map background PnP and depth reprojection, with no temporal smoothing."""
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from modal_gaussians.common.cache import atomic_json, sha256, module_revision
from modal_gaussians.common.progress import Progress
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.preprocessing.sequence import read_color_image, read_binary_mask


@dataclass(frozen=True)
class StabilizationSettings:
    """Pixel thresholds are calibrated at 540p and scale with input height."""
    mask_margin: float = 5.
    point_cell: float = 8.
    point_border: float = 12.
    map_error_native: float = 2.
    min_track_length: int = 3
    reprojection_error: float = 1.5
    forward_backward_error: float = .75
    lk_error: float = 20.
    lk_window: int = 31
    lk_levels: int = 4
    min_points: int = 40
    min_inliers: int = 25
    depth_alpha: float = .5
    seed: int = 1729

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid stabilization setting: {key}")
        for key in ('min_track_length', 'lk_window', 'lk_levels', 'min_points', 'min_inliers', 'seed'):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"Stabilization {key} must be an integer")
        if self.depth_alpha > 1 or self.min_inliers < 6 or self.min_points < self.min_inliers:
            raise ValueError('Invalid depth threshold or PnP point counts')


def project(points, pose, K, distortion):
    return cv2.projectPoints(points, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], K, distortion)[0].reshape(-1, 2)


def estimate_pose(points, pixels, initial, K, distortion, threshold, minimum):
    """Each frame uses the same world map. Failure never copies another pose."""
    ok, r, t, inliers = cv2.solvePnPRansac(np.asarray(points, np.float64), np.asarray(pixels, np.float64), K, distortion,
        rvec=cv2.Rodrigues(initial[:3, :3])[0], tvec=initial[:3, 3].copy(), useExtrinsicGuess=True,
        iterationsCount=200, reprojectionError=threshold, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < minimum:
        raise ValueError('Insufficient static-background PnP consensus')
    keep = inliers.ravel()
    for _ in range(2):
        r, t = cv2.solvePnPRefineLM(points[keep], pixels[keep], K, distortion, r, t)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(r)[0], t.ravel()
        errors = np.linalg.norm(project(points, pose, K, distortion)-pixels, axis=1)
        keep = np.flatnonzero(errors < threshold)
        if len(keep) < minimum:
            raise ValueError('Refinement lost static-background PnP consensus')
    return pose, keep


def depth_world_grid(depth, pose, K, distortion):
    """Unproject target depth; OpenCV integer pixel-center convention throughout."""
    yy, xx = np.mgrid[:depth.shape[0], :depth.shape[1]].astype(np.float32)
    xy = cv2.undistortPoints(np.stack((xx, yy), -1).reshape(-1, 1, 2), K, distortion).reshape(*depth.shape, 2)
    camera = np.concatenate((xy, np.ones((*depth.shape, 1), np.float32)), -1)*depth[..., None]
    return cv2.transform(camera, np.column_stack((pose[:3, :3].T, -pose[:3, :3].T@pose[:3, 3])))


def reprojection_map(world, pose, K, distortion):
    camera = cv2.transform(world, pose[:3].astype(world.dtype))
    z = camera[..., 2]
    x, y = camera[..., 0]/np.maximum(z, 1e-8), camera[..., 1]/np.maximum(z, 1e-8)
    radial = 1+float(distortion[0])*(x*x+y*y)
    mx = (x*radial*K[0, 0]+K[0, 2]).astype(np.float32)
    my = (y*radial*K[1, 1]+K[1, 2]).astype(np.float32)
    h, w = world.shape[:2]
    valid = np.isfinite(mx)&np.isfinite(my)&(z > 0)&(mx >= 0)&(my >= 0)&(mx <= w-2)&(my <= h-2)
    return mx, my, valid


def complete_depth(depth, alpha, points, target, K, distortion, threshold):
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    valid = np.isfinite(depth)&(depth > 1e-6)&(alpha >= threshold)
    camera = points@target[:3, :3].T+target[:3, 3]
    positive = camera[:, 2] > 0
    pixels = project(points[positive], target, K, distortion)
    inverse = 1/camera[positive, 2]
    if len(pixels) < 4:
        raise ValueError('Insufficient background depth support')
    result = depth.copy()
    yy, xx = np.nonzero(~valid)
    if len(xx):
        values = LinearNDInterpolator(pixels, inverse)(xx, yy)
        outside = ~np.isfinite(values)
        values[outside] = NearestNDInterpolator(pixels, inverse)(xx[outside], yy[outside])
        result[yy, xx] = 1/values
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError('Depth completion failed')
    return result.astype(np.float32), ~valid


def stabilize_image_sequence(sequence, output, *, scene_dir, label, settings=None, device='cuda'):
    """Write into the caller's unpublished directory; never start downstream stages."""
    import sys
    import torch
    from modal_gaussians.geometry import scene as scene_module
    from modal_gaussians.geometry.scene import load_static_scene, cameras_from_scene_manifest, registered_image_points, SceneNormalization

    settings = settings or StabilizationSettings()
    settings.validate()
    scene = load_static_scene(scene_dir, device=device, validate=True).eval()
    scene.requires_grad_(False)
    matches = [c for c in cameras_from_scene_manifest(scene.manifest) if c.role == 'reference' and c.label == label]
    if len(matches) != 1:
        raise ValueError('Select one registered fixed-view camera with --view')
    camera = matches[0]
    h, w = sequence.height, sequence.width
    ref_index = sequence.reference_frame_index
    if ([camera.height, camera.width] != [h, w] or sha256(sequence.image_paths[ref_index]) != camera.image_sha256):
        raise ValueError('Raw reference PNG/size differs from registered camera; register this resolution before stabilization')
    map_root = resolve_path(scene.manifest['dataset']['input_root'], strict=True)
    map_files = {n:sha256(map_root/n) for n in
                 ('cameras.json', 'sparse/0/cameras.bin', 'sparse/0/images.bin', 'sparse/0/points3D.bin')}
    if any(scene.manifest['dataset']['file_identities'].get(n) != digest for n, digest in map_files.items()):
        raise ValueError('COLMAP map differs from the static-scene source')
    ids, pixels, points, errors, lengths = registered_image_points(map_root, camera.name)
    # COLMAP stores corner-coordinate intrinsics; OpenCV indexes pixel centers.
    K = camera.K.cpu().double().numpy().copy()
    K[:2, 2] -= .5
    pixels = pixels-.5
    distortion = np.array([camera.radial_distortion, 0., 0., 0., 0.])
    normalization = SceneNormalization.from_dict(scene.manifest['scene_normalization'])
    points = normalization.normalize_points(points)
    target = camera.world_to_camera.cpu().double().numpy()
    factor = h/540.
    margin = max(1, round(settings.mask_margin*factor))
    kernel = np.ones((2*margin+1, 2*margin+1), np.uint8)
    def background(index):
        mask = read_binary_mask(sequence.mask_paths[index], (h, w))
        return cv2.dilate(mask.astype(np.uint8), kernel) == 0
    bg = background(ref_index)
    xy = np.rint(pixels).astype(int)
    border = settings.point_border*factor
    valid = (xy[:, 0] >= border)&(xy[:, 0] < w-border)&(xy[:, 1] >= border)&(xy[:, 1] < h-border)
    valid &= bg[np.clip(xy[:, 1], 0, h-1), np.clip(xy[:, 0], 0, w-1)]
    residual = np.linalg.norm(project(points, target, K, distortion)-pixels, axis=1)
    valid &= (errors <= settings.map_error_native)&(lengths >= settings.min_track_length)&(residual <= settings.reprojection_error*factor)
    valid &= (points@target[:3, :3].T+target[:3, 3])[:, 2] > 0
    used, keep = set(), []
    for i in np.argsort(residual):
        cell = tuple((pixels[i]/(settings.point_cell*factor)).astype(int))
        if valid[i] and cell not in used:
            keep.append(i); used.add(cell)
    points, pixels, ids = points[keep], pixels[keep], ids[keep]
    holdout = np.zeros(len(points), bool)
    holdout[np.random.default_rng(settings.seed).permutation(len(points))[::5]] = True
    if np.sum(~holdout) < settings.min_points or holdout.sum() < 8:
        raise ValueError('Insufficient background 3D coverage; improve masks/map before stabilization')
    root = Path(output)
    root.mkdir()
    for folder in ('images', 'masks', 'valid'):
        (root/folder).mkdir()
    gray_ref = cv2.cvtColor(read_color_image(sequence.image_paths[ref_index]), cv2.COLOR_BGR2GRAY)
    p0 = pixels.astype(np.float32).reshape(-1, 1, 2)
    window = max(3, round(settings.lk_window*factor)//2*2+1)
    lk = dict(winSize=(window, window), maxLevel=settings.lk_levels,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 60, .005))
    poses, diagnostics = [], []
    progress = Progress('background camera poses', sequence.frame_count, unit='frames')
    for i in range(sequence.frame_count):
        gray = cv2.cvtColor(read_color_image(sequence.image_paths[i]), cv2.COLOR_BGR2GRAY)
        p, st, error = cv2.calcOpticalFlowPyrLK(gray_ref, gray, p0, None, **lk)
        back, sb, _ = cv2.calcOpticalFlowPyrLK(gray, gray_ref, p, None, **lk)
        observed = p[:, 0].astype(np.float64)
        good = st[:, 0].astype(bool)&sb[:, 0].astype(bool)&np.isfinite(observed).all(1)
        good &= (np.linalg.norm(back[:, 0]-p0[:, 0], axis=1) < settings.forward_backward_error*factor)&(error[:, 0] < settings.lk_error)
        xy = np.rint(np.nan_to_num(observed)).astype(int)
        good &= (xy[:, 0] >= 1)&(xy[:, 0] < w-1)&(xy[:, 1] >= 1)&(xy[:, 1] < h-1)
        good &= background(i)[np.clip(xy[:, 1], 0, h-1), np.clip(xy[:, 0], 0, w-1)]
        train, test = good&~holdout, good&holdout
        if train.sum() < settings.min_points or test.sum() < 8:
            raise ValueError(f'Insufficient background tracks: {sequence.frame_names[i]}')
        cv2.setRNGSeed(settings.seed+i)
        pose, inliers = estimate_pose(points[train], observed[train], target, K, distortion,
                                      settings.reprojection_error*factor, settings.min_inliers)
        errors = np.linalg.norm(project(points[test], pose, K, distortion)-observed[test], axis=1)
        poses.append(pose)
        diagnostics.append(dict(frame=sequence.frame_names[i], inliers=len(inliers), heldout=int(test.sum()),
                                median_px=float(np.median(errors)), p95_px=float(np.quantile(errors, .95))))
        progress.update(i+1)
    # The target is the already registered static camera: all consumers share it.
    with torch.no_grad():
        rendered = scene.render(camera.to(device))
    depth, filled = complete_depth(rendered['expected_depth'].cpu().numpy(), rendered['alpha'].cpu().numpy(),
                                  points[~holdout], target, K, distortion, settings.depth_alpha)
    world = depth_world_grid(depth, target, K, distortion)
    union, common = np.zeros((h, w), bool), np.ones((h, w), bool)
    files = {}
    progress = Progress('depth reprojection', sequence.frame_count, unit='frames')
    for i, pose in enumerate(poses):
        mx, my, valid = reprojection_map(world, pose, K, distortion)
        image = cv2.remap(read_color_image(sequence.image_paths[i]), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.remap(read_binary_mask(sequence.mask_paths[i], (h, w)).astype(np.uint8), mx, my,
                         cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)&valid
        image[~valid] = 0
        name = sequence.frame_names[i]+'.png'
        for folder, data in (('images', image), ('masks', mask.astype(np.uint8)*255), ('valid', valid.astype(np.uint8)*255)):
            path = root/folder/name
            if not cv2.imwrite(str(path), data):
                raise OSError(f'Cannot publish stabilized pixels: {path}')
            files[f'{folder}/{name}'] = sha256(path)
        union |= mask
        common &= valid
        progress.update(i+1)
    if not (union&common).any():
        raise ValueError('No common valid subject support after stabilization')
    np.savez_compressed(root/'geometry.npz', poses=np.asarray(poses), target_pose=target, K=K,
        distortion=distortion, depth=depth, filled_depth=filled, point_ids=ids, holdout=holdout,
        timestamps=np.arange(sequence.frame_count)/sequence.fps_hz)
    files['geometry.npz'] = sha256(root/'geometry.npz')
    record = dict(format='modal_gaussians.stabilized_image_mask_sequence', version=2,
        method='background_pnp_reference_depth_v1', settings=asdict(settings),
        implementation=module_revision(sys.modules[__name__], scene_module),
        static_scene_identity=scene.manifest['static_scene_identity'], static_scene=str(resolve_path(scene_dir)),
        target_camera=camera.to_manifest_record(), map_files=map_files,
        depth_completion='linear_inverse_background_depth; nearest_outside_hull; heldout_excluded',
        temporal_smoothing=False, border='black_invalid; no_crop_no_replication', diagnostics=diagnostics, files=files)
    atomic_json(root/'manifest.json', record)
    return record, union&common, common
