"""Fit a moving-camera recording against an immutable scene and baked modes."""
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np

from modal_gaussians.common.cache import sha256, module_revision
from modal_gaussians.common.progress import Progress
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.geometry.scene import cameras_from_scene_manifest
from .fitting import RGBFitConfig, solve_rgb_coordinates_view
from .preparation import _publish
from .rendering import make_sequence_renderer, resize_rgb
from .rgb import RGBModalCoordinatesArtifact, load_rgb_frame
from .sequences import validate_sequences, frame_camera, downsample_sequence, validate_time_grid

SWEEP_FORMAT = 'modal_gaussians.sweep_rgb_coordinates'


def sweep_sequence(scene, metadata_path, fps=30):
    """Only registered cameras, exact training PNGs and the extraction time grid."""
    metadata_path = resolve_path(metadata_path, strict=True)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    target_fps, fps = fps, float(metadata['fps_hz'])
    if (metadata.get('format') != 'modal-gaussians-preparation' or metadata.get('version') != 1
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError('Invalid sweep extraction metadata')
    root = resolve_path(scene.manifest['dataset']['input_root'], strict=True)
    camera_file = root/'cameras.json'
    if sha256(camera_file) != scene.manifest['dataset']['file_identities']['cameras.json']:
        raise ValueError('Sweep camera source changed')
    original = json.loads(camera_file.read_text(encoding='utf-8'))
    cameras = [c for c in cameras_from_scene_manifest(scene.manifest) if c.role == 'sweep']
    records = {v['image_name']:v for v in original['frames'] if v['role'] == 'sweep'}
    if (len(records) != sum(v['role'] == 'sweep' for v in original['frames'])
            or not cameras or any(c.name not in records for c in cameras)):
        raise ValueError('Missing registered sweep frame records')
    cameras.sort(key=lambda c:records[c.name]['source_index'])
    frames, files, directories = [], [], set()
    extraction_root = resolve_path(metadata['images'], strict=True)
    for i,c in enumerate(cameras):
        record = records[c.name]; index = record['source_index']
        path = root/c.image_relative_path
        if (type(index) is not int or not 0 <= index < metadata['frame_count']
                or resolve_path(record['source_image']).parent != extraction_root
                or Path(record['source_image']).name != record['source_frame_name']
                or Path(record['source_frame_name']).name != path.name
                or [c.height,c.width] != [metadata['height'],metadata['width']]
                or sha256(path) != c.image_sha256):
            raise ValueError(f'Sweep extraction/camera/image differs: {c.name}')
        directories.add(path.parent)
        files.append(dict(name=path.name,sha256=c.image_sha256))
        frames.append(dict(name=path.stem, camera_name=c.name,
            camera_identity=c.to_manifest_record()['camera_identity'], shape_hw=[c.height,c.width],
            source_index=index, timestamp_seconds=float(metadata['clip_start_seconds'])+index/fps,
            image_sha256=c.image_sha256, coefficient_index=i))
    if len(directories) != 1:
        raise ValueError('Sweep frames must share their registered image directory')
    if len(frames) > 1:
        strides = np.diff([f['source_index'] for f in frames])
        if not np.all(strides == strides[0]) or strides[0] <= 0:
            raise ValueError('Registered sweep frames must have a uniform extraction time grid')
        fps /= int(strides[0])
    view = dict(index=0,label='sweep',kind='sweep',frame_offset=0,frame_count=len(frames),
        frame_names=[f['name'] for f in frames],frames=frames,shape_hw=frames[0]['shape_hw'],fps_hz=fps,
        camera_name=cameras[0].name,camera_identity=frames[0]['camera_identity'])
    image = dict(label='sweep',directory=str(next(iter(directories))),files=files)
    validate_sequences([view],[image],scene.manifest,frame_count=len(frames))
    view, image, rows = downsample_sequence(view, image, target_fps)
    return view,image,dict(path=str(metadata_path),sha256=sha256(metadata_path),
                           cameras_sha256=sha256(camera_file), registered_rows=rows, source_fps_hz=fps)


def load_sweep_coordinates(path):
    from .refinement_artifacts import read_manifest
    from .direct import _validate_modes
    root,m = read_manifest(path,SWEEP_FORMAT,1,'sweep_coordinates_identity')
    _validate_modes(m['modes'])
    RGBFitConfig(**m['settings']).validate()
    q = np.load(root/'coordinates.npy',allow_pickle=False)
    n,k = m['counts']['frames'],len(m['modes'])
    scales = np.asarray(m['pair_scales'])
    if (q.dtype != np.complex64 or q.shape != (n,k) or not np.isfinite(q).all()
            or m['counts'] != dict(frames=n,modes=k,views=1) or len(m['views']) != 1
            or m['views'][0].get('kind') != 'sweep' or scales.shape != (k,)
            or not np.isfinite(scales).all() or np.any(scales<=0)
            or set(m['checksums']) != {'coordinates.npy'}):
        raise ValueError('Invalid sweep coordinate artifact')
    validate_time_grid(m['views'][0])
    if m['views'][0]['frame_count'] != n:
        raise ValueError('Sweep coefficient/frame count differs')
    return RGBModalCoordinatesArtifact(root,m,q)


def downsample_sweep(*, input_dir, output_dir, fps=30):
    """Publish a frame subset of an existing fit; no optimization or image rewriting."""
    from .refinement_artifacts import write_manifest
    from modal_gaussians.geometry.scene import load_static_scene
    from modal_gaussians.motion.common.completed_modes import load_completed_modes
    source = load_sweep_coordinates(input_dir)
    m = dict(source.manifest)
    scene = load_static_scene(m['static_scene'], validate=True)
    bank = load_completed_modes(m['completed_modes'], validate=True)
    if (m['static_scene_identity'] != scene.manifest['static_scene_identity']
            or bank.manifest['static_scene_identity'] != m['static_scene_identity']
            or bank.manifest['foreground_identity'] != scene.manifest['foreground_identity']
            or bank.manifest['completed_modes_identity'] != m['completed_modes_identity']
            or bank.manifest['modes'] != m['modes']):
        raise ValueError('Sweep subset scene/mode source identity differs')
    validate_sequences(m['views'], m['images'], scene.manifest, frame_count=len(source.coordinates))
    view, image, rows = downsample_sequence(m['views'][0], m['images'][0], fps)
    destination = resolve_path(output_dir)
    if any(destination.is_relative_to(resolve_path(p)) for p in
           (input_dir, m['static_scene'], m['completed_modes'], image['directory'])):
        raise ValueError('Sweep subset output must be outside immutable inputs')
    m.pop('sweep_coordinates_identity')
    m.update(views=[view], images=[image], counts=dict(frames=len(rows), views=1, modes=len(m['modes'])),
        parent_coordinates=dict(path=str(source.path), identity=source.manifest['sweep_coordinates_identity'],
                                selected_rows=rows, source_fps_hz=source.manifest['views'][0]['fps_hz']),
        training=dict(operation='frame_subset', optimized_frames=0),
        implementation=module_revision(sys.modules[__name__], sys.modules[downsample_sequence.__module__]))
    with _publish(destination) as work:
        np.save(work/'coordinates.npy', source.coordinates[rows], allow_pickle=False)
        m['checksums'] = {'coordinates.npy':sha256(work/'coordinates.npy')}
        write_manifest(work,m,'sweep_coordinates_identity')
    return load_sweep_coordinates(destination)


def fit_sweep(*,scene_dir,completed_modes_dir,scale_source,metadata_path,output_dir,
              config=RGBFitConfig(),device='cuda',fps=30):
    from modal_gaussians.results.artifact import _load_sources
    from .refinement_artifacts import write_manifest
    from . import fitting, rendering, sequences
    from modal_gaussians.geometry.scene import load_static_scene
    from modal_gaussians.motion.common.completed_modes import load_completed_modes
    config.validate()
    load_static_scene(scene_dir, validate=True)
    load_completed_modes(completed_modes_dir, validate=True)
    _,scene,bank,kind,rgb,_,direct,views = _load_sources(scene_dir=scene_dir,
        completed_modes_dir=completed_modes_dir,coordinates_dir=scale_source)
    if kind != 'rgb' or len(views) != 1:
        raise ValueError('Sweep scale source must be a single fixed-view RGB fit')
    destination = resolve_path(output_dir)
    view,image,metadata = sweep_sequence(scene,metadata_path,fps)
    if any(destination.is_relative_to(resolve_path(p)) for p in
           (scene_dir,completed_modes_dir,scale_source,image['directory'])):
        raise ValueError('Sweep output must be outside immutable inputs')
    if destination.exists(): raise FileExistsError(destination)
    original = next(v for v in direct.manifest['views'] if v['label']==views[0]['label'])
    scales = np.asarray(direct.diagnostics['mode_pair_scales'][original['index']],np.float32)
    cameras = {c.name:c for c in cameras_from_scene_manifest(scene.manifest)}
    render = make_sequence_renderer(scene,[frame_camera(cameras,view,i) for i in range(view['frame_count'])],
                                   bank.arrays['phi'],bank.rotation,device)
    @lru_cache(maxsize=8)
    def target(i,scale):
        record=image['files'][i]
        value,_=load_rgb_frame(Path(image['directory'])/record['name'],view['shape_hw'],record['sha256'])
        return resize_rgb(value,scale).to(device)
    progress=Progress('Sweep RGB fitting',config.offset_steps+config.epochs_per_scale*len(config.scales),unit='stages')
    def report(row):
        progress.update(row['step'] if row['phase']=='offset' else config.offset_steps+row['epoch'])
    q,summary=solve_rgb_coordinates_view(np.zeros((view['frame_count'],len(bank.manifest['modes'])),np.complex64),
        scales,None,None,target,config,device,render_frame=render,on_progress=report)
    with _publish(destination) as work:
        np.save(work/'coordinates.npy',q,allow_pickle=False)
        write_manifest(work,dict(format=SWEEP_FORMAT,version=1,static_scene=str(resolve_path(scene_dir)),
            static_scene_identity=scene.manifest['static_scene_identity'],
            completed_modes=str(bank.path),completed_modes_identity=bank.manifest['completed_modes_identity'],
            modes=bank.manifest['modes'],views=[view],images=[image],pair_scales=scales.tolist(),
            scale_source=dict(path=str(rgb.path),identity=rgb.manifest['rgb_coordinates_identity'],label=views[0]['label']),
            metadata=metadata,settings=config.to_dict(),training=summary,
            implementation=module_revision(sys.modules[__name__],fitting,rendering,sequences),
            counts=dict(frames=len(q),views=1,modes=q.shape[1]),
            quality_gate=dict(status='sweep_rgb_candidate_unapproved'),
            checksums={'coordinates.npy':sha256(work/'coordinates.npy')}),'sweep_coordinates_identity')
    return load_sweep_coordinates(destination)
