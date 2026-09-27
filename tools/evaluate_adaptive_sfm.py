"""Independent static rendering, common metrics and review exports for the SfM benchmark.

Run each subcommand explicitly; none starts training or a later stage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from modal_gaussians.common.cache import atomic_json, identity, sha256
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.coordinates.preparation import _publish
from modal_gaussians.geometry.scene import Camera, SceneNormalization, load_static_dataset
from modal_gaussians.geometry.training import _load_resume_payload, _scene_from_tensor_dictionary
from modal_gaussians.preprocessing.frames import media_executable, _process_options
from modal_gaussians.results.evaluation import frame_metrics, _summary


def read_inputs(path):
    root = resolve_path(path, strict=True)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('format') != 'modal_gaussians.adaptive_sfm_benchmark'
            or manifest.get('version') != 1
            or identity({k: v for k, v in manifest.items() if k != 'identity'}) != manifest['identity']):
        raise ValueError('Invalid benchmark input contract')
    for name, digest in manifest['files'].items():
        if not (root / name).resolve().is_relative_to(root) or sha256(root / name) != digest:
            raise ValueError(f'Changed benchmark input: {name}')
    mapping = json.loads((root / 'frame_mapping.json').read_text(encoding='utf-8'))
    test = [r for r in mapping if r['split'] == 'test']
    if [r['index'] for r in test] != manifest['split']['test_indices']:
        raise ValueError('Held-out membership/order differs')
    return root, manifest, mapping, test


def camera_from_record(record, normalization, width, height):
    K = np.asarray(record['K'], dtype=np.float64)
    raw = np.asarray(record['raw_world_to_camera'], dtype=np.float64)
    if not np.allclose(K, [[K[0, 0], 0, width/2], [0, K[0, 0], height/2], [0, 0, 1]]):
        raise ValueError('Expected the prepared centered pinhole intrinsics')
    return Camera(name=Path(record['name']).stem, role='sweep', label=None,
        width=width, height=height, K=torch.tensor(K, dtype=torch.float32),
        raw_world_to_camera=torch.tensor(raw, dtype=torch.float32),
        world_to_camera=torch.tensor(normalization.normalize_world_to_camera(raw), dtype=torch.float32),
        camera_model='SIMPLE_RADIAL', camera_parameters=(float(K[0, 0]), width/2, height/2, 0.),
        image_relative_path=record['name'], mask_relative_path=record['name'] if 'mask_sha256' in record else '',
        image_sha256=record['image_sha256'], mask_sha256=record.get('mask_sha256', ''),
        distortion_applied=True)


def write_rgb(path, rgb):
    image = np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(path)


@torch.inference_mode()
def render_static(args):
    started = time.perf_counter()
    root, manifest, mapping, test = read_inputs(args.input)
    dataset = load_static_dataset(root / 'dataset_static_train')
    checkpoint = resolve_path(args.checkpoint, strict=True)
    payload = _load_resume_payload(checkpoint)
    if payload['dataset_identity'] != dataset.dataset_identity or payload['depth_identity'] is not None:
        raise ValueError('Static checkpoint is not bound to the prepared RGB-only training set')
    # Validate against the already published normalization; never derive one from held-out cameras.
    published = json.loads(resolve_path(args.scene_manifest, strict=True).read_text(encoding='utf-8'))
    if (published['dataset']['dataset_identity'] != dataset.dataset_identity
            or published['scene_normalization'] != dataset.normalization.to_dict()):
        raise ValueError('Published scene normalization differs from the original training inputs')
    width, height = manifest['derivation']['resolution']
    known = {c.name: c for c in dataset.cameras}
    max_error = 0.
    for r in mapping:
        camera = camera_from_record(r, dataset.normalization, width, height)
        if camera.name in known:
            old = known[camera.name]
            if (old.image_sha256 != camera.image_sha256 or not torch.equal(old.K, camera.K)
                    or not torch.equal(old.world_to_camera, camera.world_to_camera)
                    or not torch.equal(old.raw_world_to_camera, camera.raw_world_to_camera)):
                raise ValueError(f'Training camera reconstruction differs: {camera.name}')
        raw = dataset.raw_points[::97].astype(np.float64)
        normalized = dataset.normalization.normalize_points(raw)
        normalized_camera = normalized @ camera.world_to_camera.numpy()[:3, :3].T + camera.world_to_camera.numpy()[:3, 3]
        raw_camera = raw @ camera.raw_world_to_camera.numpy()[:3, :3].T + camera.raw_world_to_camera.numpy()[:3, 3]
        max_error = max(max_error, float(np.max(np.abs(normalized_camera - raw_camera/dataset.normalization.scale))))
    if max_error >= 1e-4:
        raise ValueError('Normalized camera projection mismatch')
    scene = _scene_from_tensor_dictionary(payload['scene_tensors']).to('cuda').eval()
    scene.sh_degree = min(payload['global_step']//1000, 3)
    destination = resolve_path(args.output)
    if destination.is_relative_to(root) or destination.is_relative_to(checkpoint.parent):
        raise ValueError('Render output must be outside immutable inputs/checkpoints')
    with _publish(destination) as work:
        folder = work / 'dynamic'; folder.mkdir()
        records = []
        for i, r in enumerate(test):
            camera = camera_from_record(r, dataset.normalization, width, height)
            rgb = scene.render_batch([camera.to('cuda')], include_depth=False)[0]['rgb'][0].cpu().numpy()
            if rgb.shape != (height, width, 3) or not np.isfinite(rgb).all():
                raise ValueError(f'Invalid rendered frame: {r["name"]}')
            np.save(folder / f'{r["index"]:06d}.npy', rgb.astype(np.float32), allow_pickle=False)
            write_rgb(folder / r['name'], rgb)
            records.append(dict(index=r['index'], name=r['name'], source_index=r['source_index'],
                timestamp_seconds=r['timestamp_seconds'], normalized_time=r['normalized_time'],
                image_sha256=r['image_sha256'], camera=camera.to_manifest_record()))
            if i % 10 == 0: print(f'Static render {i+1}/{len(test)}', flush=True)
        record = dict(format='modal_gaussians.adaptive_sfm_static_render', version=1,
            data_identity=manifest['identity'], checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
            actual_updates=payload['global_step'], gaussians=scene.count, sh_degree=scene.sh_degree,
            background=[1, 1, 1], normalization=dataset.normalization.to_dict(),
            normalization_max_error=max_error, shape_hwc=[height, width, 3],
            float_output='unquantized float32 RGB; metrics clamp [0,1]', frames=records,
            code_sha256=sha256(Path(__file__)), elapsed_seconds=time.perf_counter()-started,
            files={p.relative_to(work).as_posix(): sha256(p) for p in folder.iterdir()})
        atomic_json(work / 'render_manifest.json', record)


def load_render(path, data_identity, test):
    path = resolve_path(path, strict=True)
    record = json.loads((path / 'render_manifest.json').read_text(encoding='utf-8'))
    if record.get('format') not in ('modal_gaussians.adaptive_sfm_static_render', 'modal_gaussians.adaptive_sfm_renders') or record.get('version') != 1:
        raise ValueError('Unsupported benchmark render contract')
    if record['data_identity'] != data_identity:
        raise ValueError('Render input identity differs')
    for name, digest in record['files'].items():
        if not (path / name).resolve().is_relative_to(path) or sha256(path / name) != digest:
            raise ValueError(f'Changed render: {name}')
    frames = [r for r in record['frames'] if r.get('variant', 'dynamic') == 'dynamic']
    if [r['index'] for r in frames] != [r['index'] for r in test]:
        raise ValueError('Rendered held-out frame order differs')
    for a, b in zip(frames, test):
        if a['image_sha256'] != b['image_sha256'] or not math.isclose(a['normalized_time'], b['normalized_time'], abs_tol=1e-12):
            raise ValueError('Rendered frame identity/time differs')
        camera = a.get('camera', a)
        if (not np.allclose(camera['K'], b['K'], atol=2e-5, rtol=0)
                or not np.allclose(camera['raw_world_to_camera'], b['raw_world_to_camera'], atol=1e-6, rtol=0)):
            raise ValueError('Rendered camera differs from the common benchmark camera')
    return path, record


@torch.inference_mode()
def metrics(args):
    root, manifest, _, test = read_inputs(args.input)
    destination = resolve_path(args.output)
    perceptual = None
    lpips_protocol = None
    if args.torch_home is not None:
        torch_home = resolve_path(args.torch_home, strict=True)
        backbone = torch_home / 'hub/checkpoints/alexnet-owt-7be5be79.pth'
        if not backbone.is_file(): raise FileNotFoundError('Local AlexNet weights required; no download is attempted')
        os.environ['TORCH_HOME'] = str(torch_home)
        import lpips
        perceptual = lpips.LPIPS(net='alex', version='0.1', spatial=True, verbose=False).eval().requires_grad_(False)
        digest = hashlib.sha256()
        for name, value in sorted(perceptual.state_dict().items()):
            digest.update(name.encode()); digest.update(value.numpy().tobytes())
        lpips_protocol = dict(network='alex', version='0.1', spatial=True,
            state_sha256=digest.hexdigest(), backbone_sha256=sha256(backbone), native_resolution=True)
        perceptual = perceptual.cuda()
    sources = {}
    for item in args.render:
        label, path = item.split('=', 1)
        if label in sources: raise ValueError('Duplicate render label')
        sources[label] = load_render(path, manifest['identity'], test)
    if destination.is_relative_to(root) or any(destination.is_relative_to(p) for p, _ in sources.values()):
        raise ValueError('Metrics output must be outside inputs')
    rows = []
    for label, (path, record) in sources.items():
        variants = ['dynamic'] + sorted(p.name for p in path.glob('frozen_*') if p.is_dir())
        for variant in variants:
            for r in test:
                target = cv2.cvtColor(cv2.imread(str(root / 'dataset_adaptive/images' / r['name'])), cv2.COLOR_BGR2RGB)
                prediction = np.load(path / variant / f'{r["index"]:06d}.npy', allow_pickle=False)
                if prediction.dtype != np.float32:
                    raise ValueError('Metrics require float32 unquantized renders')
                scores = frame_metrics(torch.from_numpy(prediction).cuda(), torch.from_numpy(target.astype(np.float32)/255).cuda(), perceptual)
                rows.append(dict(method=label, variant=variant, frame_index=r['index'],
                    frame_name=r['name'], timestamp_seconds=r['timestamp_seconds'],
                    normalized_time=r['normalized_time'], image_sha256=r['image_sha256'],
                    pixels=target.shape[0]*target.shape[1], **scores))
            print(f'Metrics complete: {label}/{variant}', flush=True)
    protocol = dict(resolution='native 960x540', pixels='full common valid frame',
        color='stored RGB/255, no exposure alignment or linearization', prediction='float32 clamped [0,1]; no quantization',
        psnr='MSE floor 1e-12, ceiling 120 dB', ssim='pytorch-msssim Gaussian 11 sigma 1.5 data_range=1',
        lpips=lpips_protocol, matched_time='index/(N-1)',
        frozen='preselected t=.5 primary static snapshot; t=0/1 sensitivity, cross-time residual not a blur metric',
        aggregation='arithmetic frame means and pixel-weighted pooled MSE')
    summaries = {}
    for label in sources:
        summaries[label] = {variant: _summary([r for r in rows if r['method']==label and r['variant']==variant])
            for variant in sorted({r['variant'] for r in rows if r['method']==label})}
    with _publish(destination) as work:
        with (work / 'per_frame.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        atomic_json(work / 'metrics.json', dict(protocol=protocol, summaries=summaries))
        atomic_json(work / 'manifest.json', dict(format='modal_gaussians.adaptive_sfm_evaluation', version=1,
            data_identity=manifest['identity'], frame_indices=[r['index'] for r in test], protocol=protocol,
            renders={label: dict(path=str(path), manifest_sha256=sha256(path/'render_manifest.json')) for label,(path,_) in sources.items()},
            code_sha256=sha256(Path(__file__)), metric_code_sha256=sha256(Path(sys.modules[frame_metrics.__module__].__file__)),
            files={p.name:sha256(p) for p in work.iterdir() if p.is_file()}))


def panel(rgb, title):
    bar = np.zeros((44, rgb.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, rgb], axis=0)


def visuals(args):
    root, manifest, _, test = read_inputs(args.input)
    renders = {}
    for item in args.render:
        label, path = item.split('=', 1)
        renders[label] = load_render(path, manifest['identity'], test)[0]
    if set(renders) != {'static_300','static_900','adaptive_300','adaptive_900'}:
        raise ValueError('Visuals require the four named budget render sets')
    groups = {}
    for budget in (300, 900):
        groups[f'heldout_{budget}s'] = [('Input held-out PNG', None, None),
            (f'Static {budget}s', f'static_{budget}', 'dynamic'),
            (f'Adaptive {budget}s frame time', f'adaptive_{budget}', 'dynamic')]
        groups[f'frozen_{budget}s'] = [(f'Static {budget}s',f'static_{budget}','dynamic')]+[
            (f'Adaptive frozen t={t}',f'adaptive_{budget}',v) for t,v in [('0','frozen_0'),('.5','frozen_0p5'),('1','frozen_1')]]
    destination = resolve_path(args.output)
    if destination.is_relative_to(root) or any(destination.is_relative_to(p) for p in renders.values()):
        raise ValueError('Visual output must be outside immutable inputs')
    with _publish(destination) as work:
        probes = {}
        for name, entries in groups.items():
            width=960*len(entries); height=584
            command=[media_executable('ffmpeg'),'-hide_banner','-loglevel','error','-nostdin','-n',
                '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate','2',
                '-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(work/f'{name}.mp4')]
            with (work/f'{name}_encode.log').open('wb') as log:
                process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,**_process_options())
                try:
                    for j,r in enumerate(test):
                        images=[]
                        for title,label,variant in entries:
                            path=root/'dataset_adaptive/images'/r['name'] if label is None else renders[label]/variant/r['name']
                            image=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
                            images.append(image)
                        caption=f"frame {r['index']} / {r['timestamp_seconds']:.3f}s"
                        combined=np.concatenate([panel(im, f'{title} | {caption}') for im,(title,_,_) in zip(images,entries)],axis=1)
                        process.stdin.write(combined.tobytes())
                        if j in (0,len(test)//2,len(test)-1):
                            cv2.imwrite(str(work/f'{name}_{r["index"]:06d}.png'),cv2.cvtColor(combined,cv2.COLOR_RGB2BGR))
                            detail=np.concatenate([panel(im[140:380,320:640],title) for im,(title,_,_) in zip(images,entries)],axis=1)
                            cv2.imwrite(str(work/f'{name}_detail_{r["index"]:06d}.png'),cv2.cvtColor(detail,cv2.COLOR_RGB2BGR))
                    process.stdin.close()
                    if process.wait()!=0: raise RuntimeError(f'Encoding failed: {name}')
                finally:
                    if process.poll() is None: process.kill(); process.wait()
            probe=json.loads(subprocess.check_output([media_executable('ffprobe'),'-v','error','-select_streams','v:0',
                '-count_frames','-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(work/f'{name}.mp4')],**_process_options()))['streams'][0]
            if (int(probe['nb_read_frames'])!=len(test) or probe['width']!=width or probe['height']!=height or probe['r_frame_rate']!='2/1'):
                raise ValueError(f'Encoded review video differs: {name}')
            probes[name]=probe
        atomic_json(work/'manifest.json',dict(data_identity=manifest['identity'], review_fps=2,
            note='Held-out review slideshow with discontinuous source timestamps; not a 30 FPS reconstructed sequence',
            frozen_note='Different canonical poses; no matching deblurred ground truth',
            detail_crop_xyxy=[320,140,640,380], detail_selection='fixed center crop selected using input image before inspecting reconstructions',
            frames=[{k:r[k] for k in ('index','timestamp_seconds','normalized_time')} for r in test],
            panels=groups, video_probes=probes, files={p.name:sha256(p) for p in work.iterdir() if p.is_file()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    for stage in ('render-static','metrics','visuals'):
        p=sub.add_parser(stage); p.add_argument('--input',required=True,type=Path); p.add_argument('--output',required=True,type=Path)
        if stage=='render-static':
            p.add_argument('--checkpoint',required=True,type=Path); p.add_argument('--scene-manifest',required=True,type=Path)
        else:
            p.add_argument('--render',action='append',required=True,help='LABEL=render-directory')
        if stage=='metrics': p.add_argument('--torch-home',type=Path,help='Local cached AlexNet weights; omit to explicitly skip LPIPS')
    args=parser.parse_args()
    {'render-static':render_static,'metrics':metrics,'visuals':visuals}[args.stage](args)


if __name__=='__main__': main()
