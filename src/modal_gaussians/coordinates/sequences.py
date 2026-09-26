"""Frame/camera bindings shared by fitting, refinement, results and playback."""
import copy
import math
from pathlib import Path
from modal_gaussians.common.scene_store import resolve_path

from modal_gaussians.geometry.scene import cameras_from_scene_manifest


def fixed_sequences(scene, bank, labels):
    """Bind only selected recordings; preserve the bank's complete observation set."""
    from .sources import load_coordinate_flow
    from .rgb import _image_directory
    from modal_gaussians.common.cache import sha256
    cameras = {c.label:c for c in cameras_from_scene_manifest(scene.manifest) if c.role == 'reference'}
    available = [v['label'] for v in bank.manifest['views']]
    labels = available if labels is None else labels
    if not labels or len(set(labels)) != len(labels) or not set(labels) <= set(available):
        raise ValueError('Select unique fixed recordings from the mode bank')
    views,images = [],[]
    for i,original in enumerate(bank.manifest['views']):
        if original['label'] not in labels:
            continue
        camera = cameras.get(original['label'])
        if camera is None or camera.to_manifest_record()['camera_identity'] != original['camera_identity']:
            raise ValueError('Fixed recording camera identity differs')
        flow_path = bank.manifest['flow_artifacts'][i]
        flow = load_coordinate_flow(flow_path)
        try:
            if flow.identity != original['flow_identity']:
                raise ValueError('Fixed recording flow identity differs')
            names = flow.manifest['frame_names']
            if any(Path(name).name != name or not name for name in names):
                raise ValueError('Invalid fixed frame name')
            view = dict(original, camera_name=camera.name, frame_names=names, frame_count=len(names),
                        fps_hz=flow.manifest['fps_hz'], frame_offset=0,
                        reference_frame_name=flow.manifest['reference_frame_name'],
                        reference_frame_index=flow.manifest['reference_frame_index'])
        finally:
            flow.arrays.flow.store.close()
        directory,validity = _image_directory(flow_path,view)
        image = dict(label=view['label'],directory=str(directory),validity=validity,
                     files=[dict(name=f'{n}.png',sha256=sha256(directory/f'{n}.png')) for n in names])
        views.append(bind_fixed_view(view,image)); images.append(image)
    return reindex_views(views),images


def bind_fixed_view(view, image):
    """Normalize a validated fixed-camera coordinate view without changing its source."""
    result = copy.deepcopy(view)
    result['kind'] = 'fixed'
    result['frames'] = [dict(name=name, camera_name=view['camera_name'],
        camera_identity=view['camera_identity'], shape_hw=list(view['shape_hw']),
        source_index=i, timestamp_seconds=i / view['fps_hz'],
        image_sha256=record['sha256'], coefficient_index=view['frame_offset'] + i)
        for i, (name, record) in enumerate(zip(view['frame_names'], image['files']))]
    return result


def reindex_views(views):
    result, offset = [], 0
    for i, view in enumerate(views):
        v = copy.deepcopy(view)
        v.update(index=i, frame_offset=offset)
        for j, frame in enumerate(v['frames']):
            frame['coefficient_index'] = offset + j
        offset += v['frame_count']
        result.append(v)
    return result


def frame_camera(cameras, view, index):
    """Return the exact recorded camera; never reuse the first sweep camera."""
    frame = view['frames'][index]
    camera = cameras[frame['camera_name']]
    if camera.to_manifest_record()['camera_identity'] != frame['camera_identity']:
        raise ValueError('Frame camera identity differs')
    return camera


def frame_stride(source_fps, target_fps):
    """Integer decimation only: changing a playback label is not resampling."""
    if not all(math.isfinite(v) and v > 0 for v in (source_fps, target_fps)):
        raise ValueError('Frame rates must be finite and positive')
    ratio = source_fps / target_fps
    stride = round(ratio)
    if stride < 1 or not math.isclose(ratio, stride, rel_tol=0, abs_tol=1e-8):
        raise ValueError('Target FPS must divide source FPS by an integer; no upsampling')
    return stride


def validate_time_grid(view):
    frames, fps = view['frames'], view['fps_hz']
    frame_stride(fps, fps)
    if not frames or len(frames) != view['frame_count']:
        raise ValueError('Invalid sequence time grid')
    start = frames[0]['timestamp_seconds']
    stride = frames[1]['source_index'] - frames[0]['source_index'] if len(frames) > 1 else 1
    if stride <= 0:
        raise ValueError('Source frame indices must increase')
    for i, frame in enumerate(frames):
        if (not math.isfinite(frame['timestamp_seconds'])
                or not math.isclose(frame['timestamp_seconds'], start + i / fps, rel_tol=0, abs_tol=1e-7)
                or frame['source_index'] != frames[0]['source_index'] + i * stride):
            raise ValueError('Sequence must have a uniform source/time grid matching FPS')


def downsample_sequence(view, image, fps):
    """Subset every frame binding together, preserving original acquisition times."""
    validate_time_grid(view)
    stride = frame_stride(view['fps_hz'], fps)
    indices = list(range(0, view['frame_count'], stride))
    if len(image['files']) != view['frame_count']:
        raise ValueError('Sequence image count differs')
    result, source = copy.deepcopy(view), copy.deepcopy(image)
    result.update(fps_hz=float(fps), frame_count=len(indices),
                  frames=[result['frames'][i] for i in indices],
                  frame_names=[result['frame_names'][i] for i in indices])
    source['files'] = [source['files'][i] for i in indices]
    return reindex_views([result])[0], source, indices


def validate_sequences(views, images, scene_manifest, *, frame_count=None):
    cameras = {c.name: c for c in cameras_from_scene_manifest(scene_manifest)}
    if not views or len({v['label'] for v in views}) != len(views):
        raise ValueError('Sequence labels must be nonempty and unique')
    if [v['label'] for v in views] != [s['label'] for s in images]:
        raise ValueError('Sequence image coverage/order differs')
    offset = 0
    for i, (v, source) in enumerate(zip(views, images)):
        if v.get('kind') == 'fixed':
            from .rgb import load_valid_mask
            if source.get('validity') is None:
                raise ValueError('Fixed RGB sequence requires an explicit valid-pixel binding')
            load_valid_mask(source, v['shape_hw'])
        validate_time_grid(v)
        count, fps = v['frame_count'], v['fps_hz']
        if (type(count) is not int or count < 1 or v['index'] != i or v['frame_offset'] != offset
                or not math.isfinite(fps) or fps <= 0 or v.get('kind') not in ('fixed', 'sweep')
                or len(v['frames']) != count or len(v['frame_names']) != count
                or len(set(v['frame_names'])) != count or len(source['files']) != count):
            raise ValueError('Invalid sequence layout')
        if (v['camera_name'] != v['frames'][0]['camera_name']
                or v['camera_identity'] != v['frames'][0]['camera_identity']):
            raise ValueError('Sequence first camera differs')
        previous_index, previous_time = -1, -math.inf
        for j, (f, file) in enumerate(zip(v['frames'], source['files'])):
            camera = frame_camera(cameras, v, j)
            if (f['name'] != v['frame_names'][j] or file['name'] != f"{f['name']}.png"
                    or Path(file['name']).name != file['name']
                    or f['image_sha256'] != file['sha256'] or len(file['sha256']) != 64
                    or f['coefficient_index'] != offset + j
                    or f['shape_hw'] != [camera.height, camera.width] or f['shape_hw'] != v['shape_hw']
                    or type(f['source_index']) is not int or f['source_index'] <= previous_index
                    or not math.isfinite(f['timestamp_seconds']) or f['timestamp_seconds'] <= previous_time):
                raise ValueError('Invalid frame/image/camera binding')
            if v['kind'] == 'fixed':
                if camera.role != 'reference' or camera.label != v['label'] or camera.name != v['camera_name']:
                    raise ValueError('Fixed recording camera differs')
            elif (camera.role != 'sweep' or camera.image_sha256 != file['sha256']
                  or resolve_path(source['directory']) / file['name'] !=
                     resolve_path(scene_manifest['dataset']['input_root']) / camera.image_relative_path):
                raise ValueError('Sweep image/camera binding differs')
            previous_index, previous_time = f['source_index'], f['timestamp_seconds']
        offset += count
    if frame_count is not None and offset != frame_count:
        raise ValueError('Sequence coefficient count differs')


def sequence_weights(views):
    """Equal fixed/sweep group weights; average fixed views within their group."""
    kinds = [v.get('kind', 'fixed') for v in views]
    if any(k not in ('fixed', 'sweep') for k in kinds) or kinds.count('sweep') > 1:
        raise ValueError('Expected fixed recordings and at most one sweep')
    groups = len(set(kinds))
    return [1 / (groups * kinds.count(k)) for k in kinds]
