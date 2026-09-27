"""Publish a shared, undistorted sweep benchmark without refitting SfM.

The Adaptive dataset contains every frame and its explicit paper split. The
StaticDataset contains only training frames, with the same pixels and raw poses.
This tool never trains a model or modifies its source dataset.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile

import cv2
import numpy as np

from modal_gaussians.common.cache import atomic_json, identity, publish_directory, sha256
from modal_gaussians.common import camera_geometry
from modal_gaussians.common.camera_geometry import distort_normalized, undistort_normalized
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.geometry.colmap import _qvec_to_rotation
from modal_gaussians.geometry import colmap, scene
from modal_gaussians.geometry.scene import _read_c_string, _read_exact, load_static_dataset

OBSERVATION = np.dtype([('xy', '<f8', (2,)), ('id', '<i8')])
FORMAT = 'modal_gaussians.adaptive_sfm_benchmark'


def split_indices(count: int, hold: int = 8, segment_length: int = 4):
    if count < 2 or hold < 2 or segment_length < 1:
        raise ValueError('Need at least two frames, hold >= 2 and segment_length >= 1')
    indices = np.arange(count)
    test = ((indices // segment_length) % hold == hold - 1) & (count - indices > 3)
    return indices[~test], indices[test]


def read_model(root: Path):
    """Read full observations/tracks using the existing strict binary readers."""
    cameras, images, points = {}, {}, {}
    path = root / 'cameras.bin'
    with path.open('rb') as stream:
        count, = struct.unpack('<Q', _read_exact(stream, 8, path))
        for _ in range(count):
            key, model, width, height = struct.unpack('<iiQQ', _read_exact(stream, 24, path))
            n = {0: 3, 1: 4, 2: 4}.get(model)
            if n is None or key in cameras:
                raise ValueError('Unsupported or duplicate COLMAP camera')
            params = struct.unpack('<' + 'd' * n, _read_exact(stream, 8*n, path))
            cameras[key] = (model, width, height, params)
    path = root / 'images.bin'
    with path.open('rb') as stream:
        count, = struct.unpack('<Q', _read_exact(stream, 8, path))
        for _ in range(count):
            row = struct.unpack('<i7di', _read_exact(stream, 64, path))
            name = _read_c_string(stream, path)
            n, = struct.unpack('<Q', _read_exact(stream, 8, path))
            obs = np.frombuffer(_read_exact(stream, 24*n, path), OBSERVATION).copy()
            if row[0] in images or row[-1] not in cameras:
                raise ValueError('Duplicate image or missing COLMAP camera')
            images[row[0]] = dict(id=row[0], q=np.array(row[1:5]), t=np.array(row[5:8]),
                                  camera_id=row[-1], name=name, observations=obs)
    path = root / 'points3D.bin'
    with path.open('rb') as stream:
        count, = struct.unpack('<Q', _read_exact(stream, 8, path))
        for _ in range(count):
            row = struct.unpack('<QdddBBBdQ', _read_exact(stream, 51, path))
            track = np.frombuffer(_read_exact(stream, 8*row[-1], path), '<i4').reshape(-1, 2).copy()
            if row[0] in points:
                raise ValueError('Duplicate COLMAP point')
            points[row[0]] = dict(id=row[0], xyz=np.array(row[1:4]), rgb=tuple(row[4:7]),
                                  error=row[7], track=track)
    validate_tracks(images, points)
    return cameras, images, points


def validate_tracks(images, points):
    """Check both directions; empty tracks deliberately retain SfM initialization."""
    seen = set()
    for key, point in points.items():
        if not np.isfinite(point['xyz']).all() or not np.isfinite(point['error']):
            raise ValueError('Nonfinite COLMAP point')
        for image_id, feature_id in point['track']:
            image = images.get(int(image_id))
            pair = (int(image_id), int(feature_id))
            if (image is None or not 0 <= feature_id < len(image['observations'])
                    or image['observations']['id'][feature_id] != key or pair in seen):
                raise ValueError(f'Inconsistent COLMAP track: {key}, {pair}')
            seen.add(pair)
    for image_id, image in images.items():
        if not np.isfinite(image['observations']['xy']).all():
            raise ValueError('Nonfinite image observations')
        for feature_id in np.flatnonzero(image['observations']['id'] >= 0):
            if (image_id, int(feature_id)) not in seen:
                raise ValueError('COLMAP image observation has no reciprocal track')


def centered_maps(K, k, input_size, output_size, focal):
    """Inverse warp: COLMAP pixel centers are array indices + 0.5."""
    width, height = output_size
    y, x = np.mgrid[:height, :width]
    rays = (np.stack((x, y), -1) + .5 - [width/2, height/2]) / focal
    source = distort_normalized(rays, k) * np.diag(K)[:2] + K[:2, 2] - .5
    iw, ih = input_size
    valid = ((source[..., 0] >= 0) & (source[..., 0] <= iw-1)
             & (source[..., 1] >= 0) & (source[..., 1] <= ih-1))
    if k < 0:
        valid &= 1 + 3*k*np.sum(rays*rays, -1) > 0
    return source[..., 0].astype(np.float32), source[..., 1].astype(np.float32), valid


def common_focal(records, output_size):
    """Find a conservative common FoV, retaining every bilinear source footprint."""
    width, height = output_size
    # Testing every boundary pixel avoids assuming radial edge extrema are corners.
    x = np.arange(width) + .5 - width/2
    y = np.arange(height) + .5 - height/2
    boundary = np.concatenate((np.column_stack((x, np.full(width, y[0]))),
        np.column_stack((x, np.full(width, y[-1]))),
        np.column_stack((np.full(height, x[0]), y)),
        np.column_stack((np.full(height, x[-1]), y))))
    intrinsics = {(tuple(np.asarray(r['K']).ravel()), r['camera_parameters'][3],
                   r['image_width'], r['image_height']) for r in records}
    def fits(focal):
        rays = boundary / focal
        for packed, k, iw, ih in intrinsics:
            K = np.array(packed).reshape(3, 3)
            pixels = distort_normalized(rays, k)*np.diag(K)[:2] + K[:2, 2] - .5
            if (np.any(pixels < 1e-4) or np.any(pixels > np.array([iw-1, ih-1])-1e-4)
                    or (k < 0 and np.any(1 + 3*k*np.sum(rays*rays, -1) <= 0))):
                return False
        return True
    lower = max(max(r['K'][0][0]*width/r['image_width'],
                    r['K'][1][1]*height/r['image_height']) for r in records)
    if fits(lower):
        return float(lower)
    upper = lower
    for _ in range(32):
        upper *= 1.1
        if fits(upper):
            break
    else:
        raise ValueError('Cannot find a fully valid centered pinhole camera')
    for _ in range(48):
        midpoint = (lower + upper)/2
        if fits(midpoint):
            upper = midpoint
        else:
            lower = midpoint
    return float(upper*(1+1e-7))


def transform_model(images, points, records, output_size, focal):
    width, height = output_size
    transformed, feature_maps = {}, {}
    max_roundtrip = 0.0
    for index, record in enumerate(records):
        old = images[int(record['image_id'])]
        K = np.asarray(record['K'])
        k = record['camera_parameters'][3]
        obs = old['observations']
        rays = undistort_normalized((obs['xy']-K[:2, 2])/np.diag(K)[:2], k)
        xy = rays*focal + [width/2, height/2]
        if len(xy):
            restored = distort_normalized(rays, k)*np.diag(K)[:2] + K[:2, 2]
            max_roundtrip = max(max_roundtrip, float(np.max(np.abs(restored-obs['xy']))))
        keep = ((xy >= [.5, .5]) & (xy <= [width-.5, height-.5])).all(1)
        mapping = np.full(len(obs), -1, np.int64)
        mapping[keep] = np.arange(np.count_nonzero(keep))
        feature_maps[old['id']] = mapping
        new_obs = obs[keep].copy()
        new_obs['xy'] = xy[keep]
        transformed[old['id']] = {**old, 'camera_id': 1, 'name': f'{index:06d}.png',
                                   'observations': new_obs}
    output_points = filter_tracks(points, transformed, feature_maps)
    # Error in points3D is in pixels, so recompute after the camera change.
    errors = {key: [] for key in output_points}
    for image in transformed.values():
        obs = image['observations']
        linked = np.flatnonzero(obs['id'] >= 0)
        if not len(linked):
            continue
        xyz = np.array([points[int(key)]['xyz'] for key in obs['id'][linked]])
        camera_xyz = xyz @ _qvec_to_rotation(image['q'].copy()).T + image['t']
        if np.any(camera_xyz[:, 2] <= 0):
            raise ValueError('A retained SfM observation has nonpositive camera depth')
        projected = focal*camera_xyz[:, :2]/camera_xyz[:, 2:] + [width/2, height/2]
        for key, error in zip(obs['id'][linked], np.linalg.norm(projected-obs['xy'][linked], axis=1)):
            errors[int(key)].append(float(error))
    for key, values in errors.items():
        output_points[key]['error'] = float(np.mean(values)) if values else 0.0
    validate_tracks(transformed, output_points)
    return transformed, output_points, max_roundtrip


def filter_tracks(points, images, feature_maps=None):
    result = {}
    for key, point in points.items():
        track = []
        for image_id, feature_id in point['track']:
            if int(image_id) in images:
                new_id = int(feature_id) if feature_maps is None else int(feature_maps[int(image_id)][feature_id])
                if new_id >= 0:
                    track.append((int(image_id), new_id))
        result[key] = {**point, 'track': np.array(track, dtype='<i4').reshape(-1, 2)}
    return result


def write_model(root, images, points, output_size, focal, *, static=False):
    root.mkdir(parents=True)
    width, height = output_size
    with (root/'cameras.bin').open('wb') as stream:
        stream.write(struct.pack('<QiiQQ', 1, 1, 2 if static else 0, width, height))
        params = [focal, width/2, height/2] + ([0.0] if static else [])
        stream.write(struct.pack('<'+'d'*len(params), *params))
    with (root/'images.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(images)))
        for image in images.values():
            stream.write(struct.pack('<i7di', image['id'], *image['q'], *image['t'], 1))
            name = ('sweep/' if static else '') + image['name']
            stream.write(name.encode('utf-8')+b'\0')
            stream.write(struct.pack('<Q', len(image['observations'])))
            stream.write(image['observations'].tobytes())
    with (root/'points3D.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(points)))
        for point in points.values():
            stream.write(struct.pack('<QdddBBBdQ', point['id'], *point['xyz'], *point['rgb'],
                                     point['error'], len(point['track'])))
            stream.write(point['track'].tobytes())


def write_ply(path, points):
    # Both readers consume XYZ/RGB regardless of track length; doubles preserve SfM.
    values = np.empty(len(points), dtype=[('x','<f8'), ('y','<f8'), ('z','<f8'),
                                           ('red','u1'), ('green','u1'), ('blue','u1')])
    for index, point in enumerate(points.values()):
        values[index] = (*point['xyz'], *point['rgb'])
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(points))+
              '\nproperty double x\nproperty double y\nproperty double z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(values.tobytes())


def prepare(input_root, output_root, *, width=960, height=540, fps=30.0,
            expected_frames=362, hold=8, segment_length=4):
    source = resolve_path(input_root, strict=True)
    destination = resolve_path(output_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Output and source must be separate directory trees')
    if width < 2 or height < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid dimensions or FPS')
    dataset = load_static_dataset(source)
    source_manifest = json.loads((source/'cameras.json').read_text(encoding='utf-8'))
    records = sorted(source_manifest['frames'], key=lambda row: row['timestamp_seconds'])
    if len(records) != expected_frames or any(r['role'] != 'sweep' for r in records):
        raise ValueError('Sweep frame count/role differs from the requested contract')
    timestamps = np.array([r['timestamp_seconds'] for r in records], dtype=np.float64)
    indices = np.array([r['source_index'] for r in records], dtype=np.int64)
    if (not np.isfinite(timestamps).all() or not np.allclose(np.diff(timestamps), 1/fps, atol=1e-8, rtol=1e-6)
            or np.any(np.diff(indices) <= 0) or len(set(np.diff(indices).tolist())) != 1):
        raise ValueError('Expected a genuine uniform sweep frame/time subset')
    cameras, images, points = read_model(source/'sparse/0')
    for r in records:
        image = images[int(r['image_id'])]
        camera = cameras[image['camera_id']]
        f, cx, cy, _ = r['camera_parameters']
        expected_K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        expected_w2c = np.eye(4)
        expected_w2c[:3, :3] = _qvec_to_rotation(image['q'].copy())
        expected_w2c[:3, 3] = image['t']
        if (image['name'] != r['image_name'] or camera[:3] != (2, r['image_width'], r['image_height'])
                or r['camera_id'] != image['camera_id']
                or not np.allclose(r['K'], expected_K, atol=1e-10, rtol=0)
                or not np.allclose(r['world_to_camera'], expected_w2c, atol=1e-8, rtol=0)
                or not np.allclose(camera[3], r['camera_parameters'], atol=1e-10, rtol=0)
                or not np.allclose(image['q'], r['qvec_wxyz'], atol=1e-10, rtol=0)
                or not np.allclose(image['t'], r['tvec'], atol=1e-10, rtol=0)):
            raise ValueError('Camera manifest differs from the authoritative binary model')
    output_size = (width, height)
    focal = common_focal(records, output_size)
    K_new = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]])
    transformed, all_points, roundtrip = transform_model(images, points, records, output_size, focal)
    if not np.isfinite(roundtrip) or roundtrip > 1e-5:
        raise ValueError(f'Observation undistortion roundtrip exceeded 1e-5 pixels: {roundtrip}')
    train, test = split_indices(len(records), hold, segment_length)
    train_ids = {int(records[i]['image_id']) for i in train}
    train_images = {key: image for key, image in transformed.items() if key in train_ids}
    train_points = filter_tracks(all_points, train_images)
    validate_tracks(train_images, train_points)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.adaptive-sfm-', dir=destination.parent))
    try:
        adaptive, static = temporary/'dataset_adaptive', temporary/'dataset_static_train'
        (adaptive/'images').mkdir(parents=True)
        for parent in ('images', 'masks'):
            for role in ('sweep', 'references'):
                (static/parent/role).mkdir(parents=True)
        mapping, static_records, map_cache = [], [], {}
        for index, r in enumerate(records):
            image = transformed[int(r['image_id'])]
            name = image['name']
            map_key = (tuple(np.asarray(r['K']).ravel()), tuple(r['camera_parameters']), r['image_width'], r['image_height'])
            if map_key not in map_cache:
                mx, my, valid = centered_maps(np.asarray(r['K']), r['camera_parameters'][3],
                    (r['image_width'], r['image_height']), output_size, focal)
                if not valid.all():
                    raise ValueError('Chosen pinhole has invalid bilinear source pixels')
                support = cv2.remap(np.ones((r['image_height'], r['image_width']), np.float32),
                                   mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                if not np.all(support == 1):
                    raise ValueError('Pinhole includes padding')
                map_cache[map_key] = mx, my
            mx, my = map_cache[map_key]
            source_image, source_mask = source/'images'/r['image_name'], source/'masks'/r['image_name']
            rgb = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
            if rgb is None or mask is None:
                raise FileNotFoundError(source_image)
            rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            if not cv2.imwrite(str(adaptive/'images'/name), rgb):
                raise OSError('Failed to write undistorted RGB')
            row = dict(index=index, name=name, split='train' if image['id'] in train_ids else 'test',
                source_index=int(r['source_index']), source_frame_name=r['source_frame_name'],
                source_image_name=r['image_name'], timestamp_seconds=float(timestamps[index]),
                adaptive_raw_time=index/len(records), normalized_time=index/(len(records)-1),
                image_id=image['id'], source_camera_id=r['camera_id'],
                source_image_sha256=dataset.file_identities['images/'+r['image_name']],
                source_mask_sha256=dataset.file_identities['masks/'+r['image_name']],
                image_sha256=sha256(adaptive/'images'/name), K=K_new.tolist(),
                raw_world_to_camera=r['world_to_camera'], source_K=r['K'],
                source_camera_parameters=r['camera_parameters'])
            if image['id'] in train_ids:
                shutil.copyfile(adaptive/'images'/name, static/'images/sweep'/name)
                if not cv2.imwrite(str(static/'masks/sweep'/name), mask):
                    raise OSError('Failed to write undistorted mask')
                row['mask_sha256'] = sha256(static/'masks/sweep'/name)
                record = copy.deepcopy(r)
                record.update(image_name='sweep/'+name, camera_id=1, sample_index=index,
                    source_image=str(destination/'dataset_static_train/images/sweep'/name),
                    source_mask=str(destination/'dataset_static_train/masks/sweep'/name),
                    camera_model='SIMPLE_RADIAL', camera_parameters=[focal, width/2, height/2, 0.0],
                    image_width=width, image_height=height, K=K_new.tolist())
                static_records.append(record)
            mapping.append(row)
            if index % 25 == 0 or index == len(records)-1:
                print(f'Undistorted {index+1}/{len(records)} frames', flush=True)
        write_model(adaptive/'sparse/0', transformed, all_points, output_size, focal)
        write_model(static/'sparse/0', train_images, train_points, output_size, focal, static=True)
        write_ply(adaptive/'sparse/0/points3D.ply', all_points)
        shutil.copyfile(adaptive/'sparse/0/points3D.ply', static/'point_cloud.ply')
        source_record = dict(path=str(source), dataset_identity=dataset.dataset_identity,
                             files=dict(dataset.file_identities), derivation=source_manifest.get('derivation'))
        derivation = dict(method='centered_pinhole_undistortion_and_training_subset_v1',
            parent=str(source), parent_cameras_sha256=sha256(source/'cameras.json'),
            world_coordinates_unchanged=True, colmap_refit=False, fps_hz=fps,
            resolution=list(output_size), all_pixels_valid=True,
            rgb_filter='opencv_INTER_LINEAR', mask_filter='opencv_INTER_NEAREST',
            half_pixel='COLMAP centers=array indices+0.5; subtract0.5 only for cv2.remap')
        atomic_json(static/'cameras.json', dict(format='modal_gaussians_colmap', version=1,
            camera_model='SIMPLE_RADIAL', camera_grouping='sweep_only', colmap_uses_foreground_masks=False,
            frames=static_records, references=[], derivation=derivation))
        check = load_static_dataset(static)
        if (len(check.cameras) != len(train) or not np.array_equal(check.raw_points, dataset.raw_points)
                or not np.array_equal(check.point_colors, dataset.point_colors)):
            raise ValueError('StaticDataset handoff changed point geometry or frame count')
        for root in (adaptive, static):
            _, checked_images, checked_points = read_model(root/'sparse/0')
            if len(checked_points) != len(points):
                raise ValueError('Output dropped SfM points')
        unchanged = all(sha256(source/name) == digest for name, digest in dataset.file_identities.items())
        if not unchanged:
            raise ValueError('Source changed during adaptation')
        atomic_json(temporary/'frame_mapping.json', mapping)
        split = dict(llffhold=hold, segment_length=segment_length, final_three_train=True,
                     train_indices=train.tolist(), test_indices=test.tolist())
        atomic_json(temporary/'split.json', split)
        report = dict(frame_count=len(records), training_frames=len(train), test_frames=len(test),
            width=width, height=height, fps_hz=fps, K=K_new.tolist(), source_unchanged=unchanged,
            max_observation_roundtrip_pixels=roundtrip, static_dataset_identity=check.dataset_identity,
            sfm_points=len(points), adaptive_empty_tracks=sum(len(p['track']) == 0 for p in all_points.values()),
            static_empty_tracks=sum(len(p['track']) == 0 for p in train_points.values()),
            source_xyz_float32_sha256=hashlib.sha256(dataset.raw_points.astype('<f4').tobytes()).hexdigest(),
            source_rgb_float32_sha256=hashlib.sha256(dataset.point_colors.astype('<f4').tobytes()).hexdigest(),
            empty_track_policy='retain all XYZ/RGB for shared initialization; no-track error=0; not a new SfM map',
            point_error_policy='recomputed in pinhole pixels on all retained sweep tracks before train subset',
            static_loader_verified=True, adaptive_python_loader_verified=False,
            sfm_scope='source SfM includes held-out sweep frames and fixed references; no RGB optimization on holdouts')
        atomic_json(temporary/'verification.json', report)
        files = {p.relative_to(temporary).as_posix(): sha256(p) for p in sorted(temporary.rglob('*')) if p.is_file()}
        producer = {Path(m.__file__).name: sha256(Path(m.__file__)) for m in (camera_geometry, colmap, scene)}
        producer[Path(__file__).name] = sha256(Path(__file__))
        manifest = dict(format=FORMAT, version=1, source=source_record, derivation=derivation,
                        split=split, producer_sha256=sha256(Path(__file__)), producer_files=producer,
                        files=files, verification=report)
        manifest['identity'] = identity(manifest)
        atomic_json(temporary/'manifest.json', manifest)
        publish_directory(temporary, destination)
        return manifest
    finally:
        if temporary.exists():
            if temporary.resolve().parent != destination.parent.resolve() or not temporary.name.startswith('.adaptive-sfm-'):
                raise RuntimeError('Unsafe staging cleanup path')
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--expected-frames', type=int, default=362)
    parser.add_argument('--llffhold', type=int, default=8)
    parser.add_argument('--segment-length', type=int, default=4)
    args = parser.parse_args()
    manifest = prepare(args.input, args.output, width=args.width, height=args.height, fps=args.fps,
        expected_frames=args.expected_frames, hold=args.llffhold, segment_length=args.segment_length)
    print(json.dumps({'output': str(resolve_path(args.output)), 'identity': manifest['identity'],
                      'verification': manifest['verification']}, indent=2))


if __name__ == '__main__':
    main()
