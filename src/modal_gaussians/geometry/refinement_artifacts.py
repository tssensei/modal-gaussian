"""Publication of a new canonical scene with frozen reference-domain motion."""
import copy
import json
from pathlib import Path
import numpy as np
import torch

from modal_gaussians.common.cache import atomic_json, sha256
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.coordinates.preparation import _publish
from modal_gaussians.coordinates.refinement_artifacts import read_manifest, write_manifest, _archive
from .scene import tensor_dictionary_identity, PROJECTION_CONVENTION


def validate_scene_lineage(path, m, tensors):
    root = resolve_path(path,strict=True)
    p = m['scene_refinement']
    if sha256(root/'lineage.npz') != p['lineage_sha256']:
        raise ValueError('Scene lineage checksum differs')
    a = _archive(root/'lineage.npz')
    if set(a) != {'foreground','background'}: raise ValueError('Invalid scene lineage inventory')
    for part,rows in a.items():
        count = len(tensors[part+'.means'])
        if (rows.dtype != np.int64 or rows.shape != (count,) or np.any(rows<0)
                or np.any(rows>=p['source_counts'][part]) or m['counts'][part] != count):
            raise ValueError('Invalid scene lineage domain')
    return p


def load_scene_refined_modes(path):
    from modal_gaussians.motion.common.completed_modes import CompletedModesArtifact
    from modal_gaussians.coordinates.direct import _validate_modes
    root,m = read_manifest(path,'modal_gaussians.completed_modes',21,'completed_modes_identity')
    _validate_modes(m['modes'])
    if set(m['checksums']) != {'phi.npy','rotation.npy','support.npz'}:
        raise ValueError('Invalid scene-refined mode inventory')
    a = _archive(root/'support.npz')
    phi = np.load(root/'phi.npy',mmap_mode='r',allow_pickle=False)
    omega = np.load(root/'rotation.npy',mmap_mode='r',allow_pickle=False)
    k,g,c = len(m['modes']),m['counts']['foreground_gaussians'],len(a['c_positions'])
    roots = a['reference_root']
    if (phi.shape != (k,g,3) or omega.shape != phi.shape or phi.dtype != np.complex64 or omega.dtype != np.complex64
            or not np.isfinite(phi).all() or not np.isfinite(omega).all() or a['g_points'].shape != (g,3)
            or roots.shape != (g,) or roots.dtype != np.int64 or np.any(roots<0)
            or np.any(roots>=m['reference_count']) or a['support_class'].shape != (k,g)
            or a['observation_view_mask'].shape != (k,g,len(m['views']))
            or a['c_positions'].shape != (c,3) or a['control_displacement'].shape != (k,c,3)
            or a['control_angular'].shape != (k,c,3) or 'c_control_point_index' in a
            or any(not np.isfinite(v).all() for v in a.values() if v.dtype.kind in 'fc')):
        raise ValueError('Invalid independent reference/rendering mode domains')
    a['phi'] = phi
    return CompletedModesArtifact(root,m,a,omega,a['control_displacement'])


def publish_scene_refinement(destination, trainer, inputs):
    from .training import _sha256_json
    source,bank,coordinates,prepared = inputs
    parent = source.manifest
    final = resolve_path(destination)
    with _publish(final) as work:
        scene_path,mode_path = work/'scene',work/'mode_bank'
        scene_path.mkdir();mode_path.mkdir()
        tensors = trainer.scene.tensor_dictionary()
        torch.save(tensors,scene_path/'tensors.pt')
        np.savez(scene_path/'lineage.npz',**trainer.lineage)
        m = copy.deepcopy(parent)
        for key in ('partition','partition_identity','partition_source_path'): m.pop(key,None)
        m.update(version=8,tensors_sha256=sha256(scene_path/'tensors.pt'),
            foreground_identity=tensor_dictionary_identity(tensors,'foreground.'),
            background_identity=tensor_dictionary_identity(tensors,'background.'),
            counts=dict(foreground=trainer.scene.foreground.count,background=trainer.scene.background.count),
            training_config=trainer.config.resolved(),depth_supervision=None,
            scene_refinement=dict(source=str(source.path) if hasattr(source,'path') else trainer.source_scene,
                source_scene_identity=parent['static_scene_identity'],source_counts=parent['counts'],
                source_modes_identity=bank.manifest['completed_modes_identity'],
                run_identity=trainer.run_identity,published_step=trainer.global_step,motion=trainer.motion,
                lineage_sha256=sha256(scene_path/'lineage.npz'),partition_rule='inherit_parent_partition'))
        fg,bg = m['counts']['foreground'],m['counts']['background']
        m['representation'].update(foreground_local_index_domain=[0,fg],background_local_index_domain=[0,bg],
            combined_foreground_index_domain=[0,fg],combined_background_index_domain=[fg,fg+bg])
        payload=dict(dataset_identity=m['dataset']['dataset_identity'],foreground_identity=m['foreground_identity'],
            background_identity=m['background_identity'],normalization=m['scene_normalization'],representation='vanilla_3dgs_sh3',
            sh_degree=m['representation']['sh_degree'],camera_identities=[v['camera_identity'] for v in m['cameras']],
            projection_convention=PROJECTION_CONVENTION,scene_refinement=m['scene_refinement'])
        m['static_scene_identity']=_sha256_json(payload)
        atomic_json(scene_path/'manifest.json',m)
        positions=trainer.scene.foreground.params['means'].detach()
        roots=trainer.lineage['foreground']
        shape=(trainer.field.mode_count,len(roots),3)
        arrays=[np.lib.format.open_memmap(mode_path/name,mode='w+',dtype=np.complex64,shape=shape)
                for name in ('phi.npy','rotation.npy')]
        for k,values in enumerate(trainer.field.bake(positions,roots)):
            for target,value in zip(arrays,values): target[k]=value.cpu().numpy()
        for a in arrays: a.flush();a._mmap.close()
        support=bank.arrays
        np.savez(mode_path/'support.npz',g_points=positions.cpu().numpy(),reference_root=roots,
            reference_points=trainer.field.arrays['points'],reference_edges=trainer.field.arrays['edges'],
            support_class=support['support_class'][:,roots],observation_view_mask=support['observation_view_mask'][:,roots],
            alphas=support['alphas'],alpha_identifiable_mask=support['alpha_identifiable_mask'],
            c_positions=support['c_positions'],control_displacement=support['control_displacement'],
            control_angular=support['control_angular'],control_valid=support['control_valid'])
        b=dict(format='modal_gaussians.completed_modes',version=21,completion_method='scene_refinement_frozen_material_field',
            static_scene=str(final/'scene'),static_scene_identity=m['static_scene_identity'],foreground_identity=m['foreground_identity'],
            modes=bank.manifest['modes'],views=bank.manifest['views'],mode_sources=bank.manifest['mode_sources'],
            counts=dict(modes=shape[0],foreground_gaussians=shape[1],views=len(bank.manifest['views'])),
            reference_count=len(trainer.field.arrays['points']),reference_prepared=bank.manifest['prepared'],
            parent_modes=dict(path=str(bank.path),identity=bank.manifest['completed_modes_identity'],
                static_scene_identity=bank.manifest['static_scene_identity']),run_identity=trainer.run_identity,
            quality_gate=dict(status='scene_refinement_candidate_unapproved'),
            semantics=dict(observation_roles='inherited_reference_roots',controls='independent_immutable_reference_domain'),
            checksums={name:sha256(mode_path/name) for name in ('phi.npy','rotation.npy','support.npz')})
        write_manifest(mode_path,b,'completed_modes_identity')
        from modal_gaussians.coordinates.transfer import write_transfer
        coordinate_path=work/'coordinates';coordinate_path.mkdir()
        q=write_transfer(coordinate_path,coordinates,'sweep_rgb',final/'mode_bank',b,
            m['static_scene_identity'],trainer.motion)
        atomic_json(work/'manifest.json',dict(format='modal_gaussians.scene_refinement',version=1,
            run_identity=trainer.run_identity,scene=str(final/'scene'),modes=str(final/'mode_bank'),
            static_scene_identity=m['static_scene_identity'],completed_modes_identity=b['completed_modes_identity'],
            coordinates=str(final/'coordinates'),transferred_coordinates_identity=q['transferred_coordinates_identity'],
            published_step=trainer.global_step))
    return final
