"""Immutable preparation and publication for fixed-scene motion refinement."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from modal_gaussians.common.cache import atomic_json, identity, module_revision, sha256
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.geometry.scene import cameras_from_scene_manifest, load_static_scene
from modal_gaussians.motion.common.completed_modes import CompletedModesArtifact, load_completed_modes
from .preparation import _publish
from .rgb import RGBModalCoordinatesArtifact
from .sequences import reindex_views, validate_sequences

PREPARED_FORMAT = "modal_gaussians.refinement_preparation"
COORDINATES_FORMAT = "modal_gaussians.refined_rgb_coordinates"


def manifest_identity(m, name):
    return identity({k: v for k, v in m.items() if k != name})


def write_manifest(path, manifest, name):
    manifest[name] = manifest_identity(manifest, name)
    atomic_json(Path(path) / "manifest.json", manifest)


def read_manifest(path, format_name, version, name):
    root = resolve_path(path, strict=True)
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (m.get("format") != format_name or m.get("version") != version
            or m.get(name) != manifest_identity(m, name)):
        raise ValueError(f"Invalid {format_name} manifest or identity")
    for filename, digest in m.get("checksums", {}).items():
        if Path(filename).name != filename or sha256(root / filename) != digest:
            raise ValueError(f"Artifact checksum differs: {filename}")
    return root, m


def _archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


@torch.no_grad()
def projection_scales(scene, bank, views, device):
    from .sources import load_coordinate_flow
    from .direct import mode_pair_scales, SAMPLE_BLOCK_SIZE
    from .design import RenderedDesignConfig, prepare_modal_projection, project_modal_features
    scene = scene.to(device).requires_grad_(False)
    cameras = {c.name:c.to(device) for c in cameras_from_scene_manifest(scene.manifest)}
    settings, scales = RenderedDesignConfig(), []
    originals = {v['label']:i for i,v in enumerate(bank.manifest['views'])}
    for view in views:
        flow = load_coordinate_flow(bank.manifest['flow_artifacts'][originals[view['label']]])
        try:
            camera = cameras[view['camera_name']]
            pixels,alpha,jacobian,_ = prepare_modal_projection(scene,camera,
                flow.arrays.mask_union,settings,flow.arrays.valid_mask)
            diagonal = []
            for k in range(0,len(bank.manifest['modes']),settings.modes_per_batch):
                design = project_modal_features(scene,camera,
                    bank.arrays['phi'][k:k+settings.modes_per_batch],pixels,alpha,jacobian)
                gram = np.zeros((design.shape[-1],design.shape[-1]),np.float64)
                for lo in range(0,len(design),SAMPLE_BLOCK_SIZE):
                    block = np.asarray(design[lo:lo+SAMPLE_BLOCK_SIZE],np.float64).reshape(-1,design.shape[-1])
                    gram += block.T @ block
                diagonal.extend(np.diag(gram))
            scales.append(mode_pair_scales(diagonal,len(pixels)))
        finally:
            flow.arrays.flow.store.close()
    return np.asarray(scales,np.float32)


def prepare_refinement(*, scene_dir, completed_modes_dir, output_dir,
                       path_backend='cupy', view_labels=None, sweep_metadata=None, reference_dir=None, device='cuda'):
    from .reference import build_reference, load_reference
    from .sequences import fixed_sequences
    from .sweep import sweep_sequence
    from modal_gaussians.motion import fixed_field
    from . import reference, sequences, direct, design, sources, sweep, rgb
    from modal_gaussians.motion import reference_field
    from modal_gaussians.motion.common import projection
    from modal_gaussians.geometry import scene as scene_module
    from modal_gaussians.common import camera_rendering
    started = time.perf_counter()
    destination = resolve_path(output_dir)
    paths = [resolve_path(p,strict=True) for p in (scene_dir,completed_modes_dir)]
    if destination.exists():
        raise FileExistsError(destination)
    scene = load_static_scene(paths[0],validate=True)
    bank = load_completed_modes(paths[1],validate=True)
    if (bank.manifest.get('version') != 17
            or bank.manifest['static_scene_identity'] != scene.manifest['static_scene_identity']
            or bank.manifest['foreground_identity'] != scene.manifest['foreground_identity']):
        raise ValueError('Refinement requires a matching original scene and mode bank')
    views,images = fixed_sequences(scene,bank,view_labels)
    sweep_source = None
    if sweep_metadata is not None:
        view,image,sweep_source = sweep_sequence(scene,sweep_metadata,30)
        views.append(view); images.append(image)
    views = reindex_views(views)
    protected = paths+[resolve_path(v['directory']) for v in images]
    protected += [resolve_path(v['path']) for v in bank.manifest['sources']]
    if reference_dir is not None:
        protected.append(resolve_path(reference_dir))
    if any(destination.is_relative_to(p) for p in protected):
        raise ValueError('Preparation must be outside immutable sources')
    validate_sequences(views,images,scene.manifest)
    if reference_dir is None:
        arrays,modes,checks = build_reference(scene,bank,source_loader=load_completed_modes,path_backend=path_backend)
        reference_source_identity = None
    else:
        arrays,modes,checks,reference_source_identity = load_reference(reference_dir,scene,bank)
    fixed_field.ControlCorrection(arrays,'cpu')  # Validate the original modal gauge before preparing tables.
    operator = fixed_field.prepare_operator(arrays)
    field = fixed_field.FixedField(arrays,operator)
    validation = []
    with torch.no_grad():
        d,o = torch.from_numpy(arrays['displacement']),torch.from_numpy(arrays['angular'])
        # Bounded blocks avoid staging a second full basis in host memory.
        for k in range(field.mode_count):
            errors = [0.,0.]
            for lo in range(0,len(arrays['points']),field.block_size):
                hi = min(lo+field.block_size,len(arrays['points']))
                values = field.block(d,o,k,lo,hi)
                for j,(value,saved,norm) in enumerate(zip(values,
                        (bank.arrays['phi'][k,lo:hi],bank.rotation[k,lo:hi]),
                        (arrays['amplitude_scale'][k],arrays['angular_scale'][k]))):
                    errors[j] = max(errors[j],float(np.max(np.abs(value.numpy()-saved))/norm))
            if not np.isfinite(errors).all() or max(errors)>1e-5:
                raise ValueError(f"Frequency {bank.manifest['modes'][k]['frequency_hz']} fixed operator error: {errors}")
            validation.append(errors)
    fixed_views = [v for v in views if v['kind']=='fixed']
    scales = projection_scales(scene,bank,fixed_views,device)
    if sweep_source is not None:
        scales = np.concatenate((scales,scales[:1]),axis=0)
    if scales.shape != (len(views),len(modes)) or not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError('Invalid coefficient normalization scales')
    with _publish(destination) as work:
        np.savez(work/'reference.npz',**arrays)
        np.savez(work/'operator.npz',**operator)
        np.savez(work/'initial.npz',scales=scales)
        m = dict(format=PREPARED_FORMAT,version=5,static_scene=str(paths[0]),
            static_scene_identity=scene.manifest['static_scene_identity'],
            mode_bank=str(paths[1]),completed_modes_identity=bank.manifest['completed_modes_identity'],
            mode_sources=modes,modes=bank.manifest['modes'],views=views,images=images,
            original_views=bank.manifest['views'],initialization='zero_per_frame_complex',
            sweep_source=sweep_source,sweep_scale_view=None if sweep_source is None else fixed_views[0]['label'],
            reference_source=None if reference_dir is None else str(resolve_path(reference_dir)),
            reference_source_identity=reference_source_identity,reference_validation=checks,
            operator_validation=validation,normalization='original_rendered_design_pixel_pair_rms',
            implementation=module_revision(sys.modules[__name__],reference,sequences,direct,design,fixed_field,
                reference_field,sources,sweep,rgb,projection,scene_module,camera_rendering),
            preparation_seconds=time.perf_counter()-started,
            checksums={f:sha256(work/f) for f in ('reference.npz','operator.npz','initial.npz')})
        m['reference_identity']=m['checksums']['reference.npz']
        m['operator_identity']=m['checksums']['operator.npz']
        write_manifest(work,m,'preparation_identity')
    return destination


def load_prepared(path):
    from modal_gaussians.motion.fixed_field import FixedField
    root,m = read_manifest(path,PREPARED_FORMAT,5,'preparation_identity')
    if (set(m['checksums']) != {'reference.npz','operator.npz','initial.npz'}
            or m['reference_identity'] != m['checksums']['reference.npz']
            or m['operator_identity'] != m['checksums']['operator.npz']):
        raise ValueError('Prepared reference/operator inventory differs')
    scene = load_static_scene(m['static_scene'],validate=True)
    if scene.manifest['static_scene_identity'] != m['static_scene_identity']:
        raise ValueError('Prepared parent scene changed')
    arrays,initial,operator = (_archive(root/f) for f in ('reference.npz','initial.npz','operator.npz'))
    if not np.array_equal(arrays['points'],scene.foreground.params['means'].detach().numpy()):
        raise ValueError('Fixed scene/field order differs')
    FixedField(arrays,operator)
    scales = initial['scales']
    if (set(initial) != {'scales'} or scales.shape != (len(m['views']),len(m['modes']))
            or not np.isfinite(scales).all() or np.any(scales<=0)):
        raise ValueError('Invalid frozen coefficient scales')
    validate_sequences(m['views'],m['images'],scene.manifest)
    return root,m,scene,arrays,initial,operator


def load_flow_initialization(path, prepared, scene):
    """Bind the reference-only diagnostic's prefix and absolute q to refinement.

    The prepared artifact stays unchanged; the returned training selection and
    initialization provenance participate in the run and published identities.
    """
    root = resolve_path(path, strict=True)
    contract = json.loads((root/'contract.json').read_text(encoding='utf-8'))
    solution = root/'solution'
    record = json.loads((solution/'manifest.json').read_text(encoding='utf-8'))
    if (contract.get('kind') != 'reference_adjacent_flow_diagnostic' or contract.get('version') != 1
            or record.get('kind') != 'flow_comparison_solution' or record.get('version') != 1
            or record.get('identity') != manifest_identity(record, 'identity')
            or record.get('contract_identity') != identity(contract)):
        raise ValueError('Invalid reference-flow initialization contract')
    for key in ('preparation_identity','static_scene_identity','completed_modes_identity','modes'):
        if contract.get(key) != prepared.get(key):
            raise ValueError(f'Flow initialization {key} differs from preparation')
    bank = load_completed_modes(prepared['mode_bank'], validate=True)
    if (bank.manifest['completed_modes_identity'] != prepared['completed_modes_identity']
            or bank.manifest['static_scene_identity'] != prepared['static_scene_identity']
            or bank.manifest['modes'] != prepared['modes']):
        raise ValueError('Flow initialization mode source differs')
    frames = contract.get('frames', [])
    n, k = len(frames), len(prepared['modes'])
    candidates = [v for v in prepared['views'] if v['kind']=='fixed' and n>0
                  and v['frames'][:n]==frames and n<=v['frame_count']]
    if len(candidates) != 1:
        raise ValueError('Flow initialization must bind one exact fixed-view prefix')
    view = candidates[0]
    images = next(i for i in prepared['images'] if i['label']==view['label'])
    camera = next(c for c in cameras_from_scene_manifest(scene.manifest) if c.name==view['camera_name'])
    ref = view['reference_frame_index']
    if (contract['camera'] != camera.to_manifest_record() or contract['shape_hw'] != view['shape_hw']
            or contract['fps_hz'] != view['fps_hz'] or contract['validity'] != images['validity']
            or contract['reference_frame'] != view['frames'][ref]):
        raise ValueError('Flow initialization camera/clock/reference/validity differs')
    directory = resolve_path(images['directory'], strict=True)
    for i in sorted(set(range(n)) | {ref}):
        entry = images['files'][i]
        if (Path(entry['name']).name != entry['name']
                or entry['name'] != view['frames'][i]['name']+'.png'
                or sha256(directory/entry['name']) != view['frames'][i]['image_sha256']):
            raise ValueError('Flow initialization PNG binding differs')
    values, hashes = {}, {}
    for name in ('reference_only','reference_only_relative','reference_offset','scales'):
        filename = name+'.npy'
        hashes[filename] = sha256(solution/filename)
        if hashes[filename] != record['checksums'].get(filename):
            raise ValueError(f'Damaged flow initialization {filename}')
        values[name] = np.load(solution/filename, allow_pickle=False)
    for name, shape in (('reference_only',(n,k)),('reference_only_relative',(n,k)),('reference_offset',(k,))):
        a = values[name]
        if a.dtype != np.complex64 or a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f'Invalid flow initialization {name}')
    q, scales = values['reference_only'], values['scales']
    if (not np.array_equal(q, values['reference_only_relative']+values['reference_offset'])
            or scales.shape != (k,) or scales.dtype.kind != 'f'
            or not np.isfinite(scales).all() or np.any(scales<=0)):
        raise ValueError('Flow initialization offset/normalization differs')
    selection = reindex_views([dict(view, frame_count=n, frames=frames, frame_names=view['frame_names'][:n])])
    selected_images = [dict(images, files=images['files'][:n])]
    validate_sequences(selection, selected_images, scene.manifest, frame_count=n)
    provenance = dict(method='reference_only_flow_plus_shared_rgb_offset', directory=str(root),
        contract_identity=identity(contract), solution_identity=record['identity'], checksums=hashes,
        source_rows=[f['coefficient_index'] for f in frames], offset_already_included=True,
        normalization='diagnostic_rendered_design_pixel_pair_rms', warmup_skipped=True)
    return (dict(prepared, views=selection, images=selected_images,
                 initialization=provenance['method'], initialization_source=provenance),
            dict(scales=scales[None].astype(np.float32), coordinates=q))


def load_refined_coordinates(path):
    root, m = read_manifest(path, COORDINATES_FORMAT, 5, "refined_coordinates_identity")
    q = np.load(root / "coordinates.npy", allow_pickle=False)
    from .direct import _validate_modes
    _validate_modes(m["modes"])
    offset = 0
    for i, v in enumerate(m["views"]):
        if (v["index"] != i or v["frame_offset"] != offset or v["frame_count"] < 1
                or len(v["frame_names"]) != v["frame_count"] or not np.isfinite(v["fps_hz"]) or v["fps_hz"] <= 0
                or len(set(v["frame_names"])) != v["frame_count"]
                or len(v["frames"]) != v["frame_count"]):
            raise ValueError("Invalid refined recording layout")
        offset += v["frame_count"]
    if (q.dtype != np.complex64 or q.shape != (offset, len(m["modes"])) or not np.isfinite(q).all()
            or not m["views"] or len({v["label"] for v in m["views"]}) != len(m["views"])
            or m["counts"] != {"frames": offset, "views": len(m["views"]), "modes": q.shape[1]}
            or set(m["checksums"]) != {"coordinates.npy"}):
        raise ValueError("Invalid refined coefficients")
    images = m["images"]
    if [v["label"] for v in images] != [v["label"] for v in m["views"]]:
        raise ValueError("Refined image-source order differs")
    for view, image in zip(m["views"], images):
        if [f["name"] for f in image["files"]] != [f"{name}.png" for name in view["frame_names"]]:
            raise ValueError("Refined image frames differ")
    return RGBModalCoordinatesArtifact(root, m, q)


def load_refined_modes(path):
    from modal_gaussians.motion.fixed_field import FixedField
    root,m = read_manifest(path,'modal_gaussians.completed_modes',20,'completed_modes_identity')
    from .direct import _validate_modes
    _validate_modes(m['modes'])
    if (set(m['checksums']) != {'phi.npy','rotation.npy','support.npz','operator.npz'}
            or m.get('operator_identity') != m['checksums']['operator.npz']
            or m.get('completion_method') != 'fixed_scene_motion_refinement'):
        raise ValueError('Refined mode file inventory differs')
    a = _archive(root/'support.npz')
    k,g,v = len(m['modes']),m['counts']['foreground_gaussians'],len(m['views'])
    if m['counts'] != dict(modes=k,foreground_gaussians=g,views=v):
        raise ValueError('Refined mode counts differ')
    for name in ('phi','rotation'):
        a[name] = np.load(root/f'{name}.npy',mmap_mode='r',allow_pickle=False)
        if a[name].shape != (k,g,3) or a[name].dtype != np.complex64 or not np.isfinite(a[name]).all():
            raise ValueError('Invalid refined field')
    controls = a['c_control_point_index']
    if (a['g_points'].shape != (g,3) or not np.array_equal(a['g_points'],a['reference_points'])
            or a['support_class'].shape != (k,g) or a['support_class'].dtype != np.int8
            or np.any((a['support_class']<0)|(a['support_class']>3))
            or a['observation_view_mask'].shape != (k,g,v) or a['observation_view_mask'].dtype != bool
            or a['alphas'].shape != (k,v) or a['alpha_identifiable_mask'].shape != (k,v)
            or a['control_valid'].shape != (k,len(controls)) or a['control_valid'].dtype != bool
            or a['control_length_scale'].shape != () or not float(a['control_length_scale'])>0
            or controls.dtype != np.int64 or controls.ndim != 1 or np.any(controls<0) or np.any(controls>=g)
            or len(np.unique(controls)) != len(controls)
            or not np.array_equal(a['c_positions'],a['g_points'][controls])
            or a['reference_edges'].ndim != 2 or a['reference_edges'].shape[1]!=2
            or np.any(a['reference_edges']<0) or np.any(a['reference_edges']>=g)
            or len(m['mode_sources']) != k
            or any(a[key].shape != (k,len(controls),3) or a[key].dtype != np.complex64
                   for key in ('control_displacement','control_angular','original_displacement','original_angular'))
            or any(not np.isfinite(value).all() for value in a.values() if value.dtype.kind in 'fc')):
        raise ValueError('Invalid fixed-scene mode support')
    FixedField(dict(points=a['g_points'],controls=controls,displacement=a['control_displacement']),_archive(root/'operator.npz'))
    return CompletedModesArtifact(root,m,a,a.pop('rotation'),a['control_displacement'])


def publish_refinement(destination, prepared, scene, field, coordinates, *, controls, run_identity, settings, baked_fields):
    import shutil
    destination = resolve_path(destination)
    root,source = prepared
    original = load_static_scene(source['static_scene'],validate=True)
    if (scene.manifest['static_scene_identity'] != source['static_scene_identity']
            or any(not torch.equal(v.detach().cpu(),original.state_dict()[key]) for key,v in scene.state_dict().items())):
        raise ValueError('Motion refinement changed the frozen scene')
    a = field.a
    shape = (field.mode_count,scene.foreground.count,3)
    with _publish(destination) as work:
        bank_path,coord_path = work/'mode_bank',work/'coordinates'
        bank_path.mkdir(); coord_path.mkdir()
        for name,value in zip(('phi','rotation'),baked_fields):
            if value.shape != shape or value.dtype != torch.complex64 or not torch.isfinite(value).all():
                raise ValueError('Invalid final baked motion')
            np.save(bank_path/f'{name}.npy',value.detach().cpu().numpy(),allow_pickle=False)
        d,o = (value.detach().cpu().numpy().astype(np.complex64) for value in controls)
        np.savez(bank_path/'support.npz',g_points=a['points'],support_class=a['support_class'],
            observation_view_mask=a['observation_view_mask'],alphas=a['alphas'],
            alpha_identifiable_mask=a['alpha_identifiable_mask'],reference_points=a['points'],
            reference_edges=a['edges'],reference_controls=a['controls'],
            reference_weights=a['lengths'][None]/a['propagation'],
            c_control_point_index=a['controls'],c_positions=a['points'][a['controls']],
            control_valid=a['control_valid'],control_displacement=d,control_angular=o,
            original_displacement=a['displacement'],original_angular=a['angular'],control_length_scale=a['radius'])
        shutil.copyfile(root/'operator.npz',bank_path/'operator.npz')
        b = dict(format='modal_gaussians.completed_modes',version=20,completion_method='fixed_scene_motion_refinement',
            static_scene=source['static_scene'],static_scene_identity=source['static_scene_identity'],
            foreground_identity=scene.manifest['foreground_identity'],reference_identity=source['reference_identity'],
            operator_identity=source['operator_identity'],prepared=str(root),preparation_identity=source['preparation_identity'],
            run_identity=run_identity,modes=source['modes'],views=source['original_views'],mode_sources=source['mode_sources'],
            semantics=dict(observation_roles='inherited_mode_sources',motion='rgb_refined_control_fields'),
            counts=dict(modes=shape[0],foreground_gaussians=shape[1],views=len(source['original_views'])),
            checksums={f:sha256(bank_path/f) for f in ('phi.npy','rotation.npy','support.npz','operator.npz')},
            quality_gate=dict(status='refined_motion_candidate_unapproved'))
        write_manifest(bank_path,b,'completed_modes_identity')
        np.save(coord_path/'coordinates.npy',np.asarray(coordinates,np.complex64),allow_pickle=False)
        c = dict(format=COORDINATES_FORMAT,version=5,static_scene_identity=source['static_scene_identity'],
            completed_modes=str(destination/'mode_bank'),completed_modes_identity=b['completed_modes_identity'],
            modes=source['modes'],views=source['views'],images=source['images'],settings=settings,
            initialization=source['initialization'],preparation_identity=source['preparation_identity'],run_identity=run_identity,
            quality_gate=dict(status='refined_rgb_candidate_unapproved'),
            counts=dict(views=len(source['views']),frames=len(coordinates),modes=shape[0]),
            checksums={'coordinates.npy':sha256(coord_path/'coordinates.npy')})
        if 'initialization_source' in source:
            c['initialization_source'] = source['initialization_source']
        if 'training_selection' in source:
            c['training_selection'] = source['training_selection']
        write_manifest(coord_path,c,'refined_coordinates_identity')
        write_manifest(work,dict(format='modal_gaussians.refined_motion',version=1,static_scene=source['static_scene'],
            static_scene_identity=source['static_scene_identity'],completed_modes_identity=b['completed_modes_identity'],
            refined_coordinates_identity=c['refined_coordinates_identity'],preparation_identity=source['preparation_identity'],
            run_identity=run_identity),'refinement_identity')
    return destination
