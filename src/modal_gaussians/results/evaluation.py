"""Explicit, immutable RGB reconstruction measurements on fitted input frames."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from importlib.metadata import version
import math
from pathlib import Path
import sys
import time

import torch

from modal_gaussians.common.cache import atomic_json, identity, module_revision, sha256
from modal_gaussians.common.progress import Progress
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.coordinates.preparation import _publish
from modal_gaussians.coordinates.rgb import load_rgb_frame, load_valid_mask
from modal_gaussians.coordinates.fitting import rgb_ssim
from modal_gaussians.coordinates.rendering import make_sequence_renderer
from modal_gaussians.coordinates.sequences import frame_camera
from modal_gaussians.geometry.scene import cameras_from_scene_manifest, load_static_scene
from modal_gaussians.geometry.training import _qa_metrics
from modal_gaussians.motion.common.completed_modes import load_completed_modes
from modal_gaussians.results.artifact import load_modal_result


def frame_metrics(prediction, target, perceptual=None, valid=None):
    """HWC float RGB in [0,1]; use the existing static-QA PSNR/SSIM convention."""
    if (prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3
            or min(prediction.shape[:2]) < 11
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()
            or target.min() < 0 or target.max() > 1):
        raise ValueError("Metrics require matching finite RGB images, targets in [0,1], H/W >=11")
    prediction = prediction.clamp(0, 1)
    squared = (prediction-target).square()
    mse = float((squared.mean() if valid is None else squared[valid].mean()).item())
    if valid is None:
        psnr, ssim = _qa_metrics(prediction, target)
    else:
        psnr, ssim = -10*math.log10(max(mse, 1e-12)), float(rgb_ssim(prediction, target, valid))
    values = dict(mse=mse, rmse=math.sqrt(mse), psnr_db=psnr, ssim=ssim)
    if perceptual is not None:
        if valid is not None:
            prediction = prediction*valid[..., None]
            target = target*valid[..., None]
        score = perceptual(prediction.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None], normalize=True)
        if valid is None:
            values['lpips'] = float(score.mean())
        else:
            if score.numel() == 1:
                raise ValueError('Masked LPIPS requires spatial=True')
            from torch.nn import functional as F
            support = F.interpolate(valid.float()[None, None], size=score.shape[-2:], mode='area')
            values['lpips'] = float((score*support).sum()/support.sum())
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Non-finite reconstruction metric")
    return values


def _summary(rows):
    keys = ('psnr_db', 'ssim', 'rmse') + (('lpips',) if 'lpips' in rows[0] else ())
    result = {f'mean_{k}': sum(r[k] for r in rows) / len(rows) for k in keys}
    mse = sum(r['mse'] * r['pixels'] for r in rows) / sum(r['pixels'] for r in rows)
    return dict(frames=len(rows), **result, pooled_rmse=math.sqrt(mse),
                pooled_psnr_db=-10 * math.log10(max(mse, 1e-12)))


@torch.inference_mode()
def evaluate_result(*, result_dir, output_dir, view_labels=None, with_lpips=False, device='cuda', baseline_dir=None):
    """Evaluate every selected frame without resizing, fitting, or video compression."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    started = time.perf_counter()
    result = load_modal_result(result_dir)
    if result.manifest['coordinate_source']['kind'] not in ('rgb', 'refined_rgb', 'sweep_rgb', 'mixed_rgb'):
        raise ValueError("Evaluation requires RGB-fitted coordinates with input PNG records")
    # Result loading verifies bindings; a formal measurement additionally hashes tensors/fields.
    load_static_scene(result.manifest['sources']['static_scene']['path'], 'cpu', validate=True)
    load_completed_modes(result.completed_modes.path, validate=True)
    by_label = {v['label']: v for v in result.manifest['views']}
    labels = list(by_label) if view_labels is None else list(view_labels)
    if not labels or len(set(labels)) != len(labels) or any(v not in by_label for v in labels):
        raise ValueError(f"Select unique available views: {list(by_label)}")
    sources = []
    protected = [result.path] + [resolve_path(s['path'], strict=True)
                                for s in result.manifest['sources'].values()]
    for label in labels:
        view = by_label[label]
        matches = [s for s in result.coordinates.manifest['images'] if s['label'] == label]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one RGB source for {label}")
        source = matches[0]
        directory = resolve_path(source['directory'], strict=True)
        protected.append(directory)
        if len(source['files']) != view['frame_count']:
            raise ValueError(f"RGB source frame count differs: {label}")
        for name, record in zip(view['frame_names'], source['files']):
            if record['name'] != f'{name}.png' or Path(record['name']).name != record['name']:
                raise ValueError(f"RGB source frame order differs: {label}")
        sources.append((view, source, directory))
    if any(destination.is_relative_to(p) for p in protected):
        raise ValueError("Evaluation output must be outside immutable inputs")

    protocol = dict(task='fitted_recording_reconstruction', region='recorded_common_valid_support; full_frame_for_native_sweep',
                    resolution='native_input_png', color='RGB stored values / 255; no linearization',
                    prediction='float32 clamped [0,1]; no quantization', coefficient='saved q unchanged',
                    psnr='-10*log10(max(mean_RGB_squared_error,1e-12)); ceiling 120 dB',
                    ssim='Gaussian window 11, sigma 1.5, data_range 1, K=(0.01,0.03); fully valid windows only',
                    aggregation='arithmetic frame means; pooled MSE weighted by pixel count; equal-view macro also reported',
                    lpips=None)
    packages = {name: version(name) for name in ('torch', 'gsplat', 'pytorch-msssim', 'numpy')}
    perceptual = None
    if with_lpips:
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError("Install modal-gaussians[evaluation] for --lpips") from error
        perceptual = lpips.LPIPS(net='alex', version='0.1', spatial=True, verbose=False).eval().requires_grad_(False)
        digest = hashlib.sha256()
        for name, tensor in sorted(perceptual.state_dict().items()):
            digest.update(name.encode()); digest.update(tensor.numpy().tobytes())
        protocol['lpips'] = dict(network='alex', version='0.1', pretrained=True,
                                 input='native RGB, normalize=True; invalid RGB zero on both sides; spatial score weighted by valid support (boundary context remains)',
                                 state_sha256=digest.hexdigest())
        packages.update({name: version(name) for name in ('lpips', 'torchvision')})
        perceptual = perceptual.to(device)
    from modal_gaussians.coordinates import rendering, rgb, sequences
    from modal_gaussians.geometry import scene, training
    revision = module_revision(sys.modules[__name__], rendering, rgb, sequences, scene, training)
    protocol_identity = identity(dict(protocol=protocol, packages=packages, code_revision=revision))
    support_identities = {v['label']:s.get('validity', {}).get('sha256') for v,s,_ in sources}
    baseline = None
    if baseline_dir is not None:
        baseline_path = resolve_path(baseline_dir, strict=True)
        if destination.is_relative_to(baseline_path):
            raise ValueError('Evaluation output must be outside the immutable baseline')
        baseline = json.loads((baseline_path / 'manifest.json').read_text(encoding='utf-8'))
        if (baseline.get('format') != 'modal_gaussians.result_evaluation' or baseline.get('version') != 2
                or baseline['evaluation_identity'] != identity({k:v for k,v in baseline.items() if k != 'evaluation_identity'})
                or baseline['protocol_identity'] != protocol_identity
                or baseline.get('support_identities') != support_identities
                or any(Path(f).name != f or sha256(baseline_path/f) != h for f,h in baseline['files'].items())):
            raise ValueError('Baseline checksum/protocol differs; re-evaluate with the current evaluator')
        def frame_bindings(views):
            return {v['label']: (v['fps_hz'], v['shape_hw'], [
                {k:f[k] for k in ('name','image_sha256','camera_identity','source_index','timestamp_seconds')}
                for f in v['frames']]) for v in views}
        if frame_bindings(baseline['views']) != frame_bindings([v for v,_,_ in sources]):
            raise ValueError('Baseline sequence/frame/camera bindings differ')
        with (baseline_path/'per_frame.csv').open(encoding='utf-8',newline='') as stream:
            baseline_rows = {(r['view'],int(r['frame_index']),r['frame_name']):r for r in csv.DictReader(stream)}
    cameras = {c.name: c for c in cameras_from_scene_manifest(result.scene.manifest)}
    rows = []
    with _publish(destination) as work:
        columns = ['view', 'frame_index', 'frame_name', 'camera_name', 'timestamp_seconds', 'pixels', 'mse', 'rmse', 'psnr_db', 'ssim']
        if with_lpips:
            columns.append('lpips')
        with (work / 'per_frame.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for view, source, directory in sources:
                valid = load_valid_mask(source, view['shape_hw'], device=device)
                frame_cameras = [frame_camera(cameras, view, i) for i in range(view['frame_count'])]
                camera = frame_cameras[0]
                if [camera.height, camera.width] != view['shape_hw']:
                    raise ValueError("Camera and recorded image dimensions differ")
                render = make_sequence_renderer(result.scene, frame_cameras, result.completed_modes.arrays['phi'],
                                           result.completed_modes.rotation, device)
                progress = Progress(f"Evaluate {view['label']}", view['frame_count'], unit='frames')
                for index, record in enumerate(source['files']):
                    target, _ = load_rgb_frame(directory / record['name'], view['shape_hw'], record['sha256'])
                    q = torch.tensor(result.coordinates.coordinates[view['frame_offset'] + index],
                                     dtype=torch.complex64, device=device)
                    values = frame_metrics(render(index, q, 1.0), target.to(device), perceptual, valid)
                    row = dict(view=view['label'], frame_index=index, frame_name=record['name'],
                               camera_name=frame_cameras[index].name, timestamp_seconds=view['frames'][index]['timestamp_seconds'],
                               pixels=camera.height * camera.width if valid is None else int(valid.sum()), **values)
                    rows.append(row); writer.writerow(row)
                    progress.update(index + 1)
                del render
        per_view = {label: _summary([r for r in rows if r['view'] == label]) for label in labels}
        macro = {k: sum(v[k] for v in per_view.values()) / len(labels)
                 for k in next(iter(per_view.values())) if k.startswith('mean_')}
        metrics = dict(per_view=per_view, all_frames=_summary(rows), equal_view_mean=macro)
        files = ['metrics.json', 'per_frame.csv']
        if baseline is not None:
            keys = ('psnr_db','ssim','rmse') + (('lpips',) if with_lpips else ())
            differences = []
            if len(baseline_rows) != len(rows): raise ValueError('Baseline metric row count differs')
            for row in rows:
                previous = baseline_rows[row['view'],row['frame_index'],row['frame_name']]
                differences.append({**{k:row[k] for k in columns[:5]},
                    **{'delta_'+k:row[k]-float(previous[k]) for k in keys}})
            with (work/'per_frame_delta.csv').open('w',encoding='utf-8',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(differences[0]))
                writer.writeheader();writer.writerows(differences)
            metrics['delta_from_baseline'] = {label: {'mean_delta_'+k:sum(r['delta_'+k] for r in differences if r['view']==label)/per_view[label]['frames']
                for k in keys} for label in labels}
            files.append('per_frame_delta.csv')
        atomic_json(work / 'metrics.json', metrics)
        manifest = dict(format='modal_gaussians.result_evaluation', version=2,
                        created_utc=datetime.now(timezone.utc).isoformat(), result=str(result.path),
                        modal_result_identity=result.manifest['modal_result_identity'],
                        coordinate_source=result.manifest['coordinate_source'],
                        protocol=protocol, protocol_identity=protocol_identity, code_revision=revision,
                        support_identities=support_identities,
                        packages=packages, device=str(device), cuda=torch.version.cuda,
                        device_name=torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else 'CPU',
                        views=[v for v, _, _ in sources], images=[s for _, s, _ in sources],
                        seconds=time.perf_counter() - started,
                        baseline=None if baseline is None else dict(path=str(baseline_path),identity=baseline['evaluation_identity']),
                        files={name: sha256(work / name) for name in files})
        manifest['evaluation_identity'] = identity(manifest)
        atomic_json(work / 'manifest.json', manifest)
    return destination
