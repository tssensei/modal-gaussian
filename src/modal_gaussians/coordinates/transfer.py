"""Explicit frozen-coordinate bindings to a newly published scene and mode bank."""
import numpy as np

from modal_gaussians.common.cache import sha256
from modal_gaussians.common.scene_store import resolve_path
from .preparation import _publish
from .refinement_artifacts import read_manifest, write_manifest
from .rgb import RGBModalCoordinatesArtifact

TRANSFER_FORMAT = 'modal_gaussians.transferred_rgb_coordinates'


def load_transferred_coordinates(path):
    root,m = read_manifest(path,TRANSFER_FORMAT,1,'transferred_coordinates_identity')
    from .direct import _validate_modes
    from .sequences import validate_time_grid
    _validate_modes(m['modes'])
    q = np.load(root/'coordinates.npy',allow_pickle=False)
    offset = 0
    for i,v in enumerate(m['views']):
        validate_time_grid(v)
        if v['index'] != i or v['frame_offset'] != offset:
            raise ValueError('Invalid transferred view layout')
        offset += v['frame_count']
    if (q.dtype != np.complex64 or q.shape != (offset,len(m['modes'])) or not np.isfinite(q).all()
            or m['counts'] != dict(frames=offset,modes=q.shape[1],views=len(m['views']))
            or m['operation'] not in ('frozen_transfer','zero_ablation')
            or set(m['checksums']) != {'coordinates.npy'}
            or (m['operation']=='zero_ablation' and np.any(q != 0))):
        raise ValueError('Invalid transferred coordinates')
    if m['operation']=='frozen_transfer' and m['checksums']['coordinates.npy'] != m['parent']['coordinates_sha256']:
        raise ValueError('Frozen coordinate values changed')
    return RGBModalCoordinatesArtifact(root,m,q)


def transfer_coordinates(*, source_dir, scene_dir, modes_dir, output_dir, motion='fitted'):
    from modal_gaussians.results.artifact import _load_coordinate_artifact
    from modal_gaussians.geometry.scene import load_static_scene
    from modal_gaussians.motion.common.completed_modes import load_completed_modes
    from .sequences import validate_sequences
    if motion not in ('fitted','zero'): raise ValueError('Expected fitted or zero motion')
    kind,source = _load_coordinate_artifact(resolve_path(source_dir,strict=True))
    scene = load_static_scene(scene_dir,validate=True)
    bank = load_completed_modes(modes_dir,validate=True)
    zero_baseline=bank.manifest['version']==20 and motion=='zero'
    parent = dict(identity=bank.manifest['completed_modes_identity'],static_scene_identity=bank.manifest['static_scene_identity']) if zero_baseline else bank.manifest.get('parent_modes',{})
    if ((bank.manifest['version'] != 21 and not zero_baseline) or kind not in ('refined_rgb','sweep_rgb')
            or source.manifest['completed_modes_identity'] != parent['identity']
            or source.manifest['static_scene_identity'] != parent['static_scene_identity']
            or source.manifest['modes'] != bank.manifest['modes']
            or scene.manifest['static_scene_identity'] != bank.manifest['static_scene_identity']):
        raise ValueError('Frozen coordinate transfer source differs')
    validate_sequences(source.manifest['views'],source.manifest['images'],scene.manifest,
                       frame_count=len(source.coordinates))
    output=resolve_path(output_dir)
    protected=[resolve_path(p,strict=True) for p in (source_dir,scene_dir,modes_dir)]
    protected.extend(resolve_path(v['directory'],strict=True) for v in source.manifest['images'])
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Coordinate transfer output overlaps immutable inputs')
    with _publish(output) as work:
        write_transfer(work,source,kind,bank.path,bank.manifest,scene.manifest['static_scene_identity'],motion)
    return load_transferred_coordinates(output_dir)


def write_transfer(work,source,kind,bank_path,bank_manifest,scene_identity,motion):
    """Write a validated binding inside its caller's atomic publication directory."""
    import shutil
    from modal_gaussians.results.artifact import COORDINATE_IDENTITY_NAMES
    if motion=='fitted': shutil.copyfile(source.path/'coordinates.npy',work/'coordinates.npy')
    else: np.save(work/'coordinates.npy',np.zeros_like(source.coordinates),allow_pickle=False)
    m = dict(format=TRANSFER_FORMAT,version=1,static_scene_identity=scene_identity,
        completed_modes=str(bank_path),completed_modes_identity=bank_manifest['completed_modes_identity'],
        modes=bank_manifest['modes'],views=source.manifest['views'],images=source.manifest['images'],
        operation='frozen_transfer' if motion=='fitted' else 'zero_ablation',
        parent=dict(path=str(source.path),kind=kind,identity=source.manifest[COORDINATE_IDENTITY_NAMES[kind]],
            static_scene_identity=source.manifest['static_scene_identity'],
            completed_modes_identity=source.manifest['completed_modes_identity'],
            coordinates_sha256=sha256(source.path/'coordinates.npy')),
        counts=dict(frames=len(source.coordinates),modes=source.coordinates.shape[1],views=len(source.manifest['views'])),
        quality_gate=dict(status='scene_refinement_candidate_unapproved'),
        checksums={'coordinates.npy':sha256(work/'coordinates.npy')})
    write_manifest(work,m,'transferred_coordinates_identity')
    return m
