"""Explicit stages for a fixed-view reference/adjacent-flow diagnostic experiment.

No production coordinate/result format is published. Run each stage explicitly:
design, pairs, solve, export, plot. Outputs are immutable; logs are mutable.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np


def context(args, *, gpu=False):
    import torch
    from modal_gaussians.common.cache import atomic_json, identity, sha256
    from modal_gaussians.common.scene_store import resolve_path
    from modal_gaussians.coordinates.refinement_artifacts import read_manifest, PREPARED_FORMAT
    from modal_gaussians.coordinates.sources import load_coordinate_flow
    from modal_gaussians.coordinates.rgb import load_valid_mask
    from modal_gaussians.geometry.scene import load_static_scene, cameras_from_scene_manifest
    from modal_gaussians.motion.common.completed_modes import load_completed_modes

    torch.set_num_threads(4)
    prepared, m = read_manifest(args.prepared, PREPARED_FORMAT, 5, 'preparation_identity')
    view = next(v for v in m['views'] if v['label'] == args.view)
    images = next(v for v in m['images'] if v['label'] == args.view)
    scene = load_static_scene(m['static_scene'], validate=True)
    bank = load_completed_modes(m['mode_bank'], validate=True)
    flow = load_coordinate_flow(args.flow)
    n, ref = args.frames, view['reference_frame_index']
    if not 2 <= n <= view['frame_count'] or view['kind'] != 'fixed':
        raise ValueError('Need a fixed recording with at least the requested frames')
    camera = next(c for c in cameras_from_scene_manifest(scene.manifest) if c.name == view['camera_name'])
    original = next(v for v in bank.manifest['views'] if v['label'] == args.view)
    if (scene.manifest['static_scene_identity'] != m['static_scene_identity']
            or bank.manifest['static_scene_identity'] != m['static_scene_identity']
            or bank.manifest['completed_modes_identity'] != m['completed_modes_identity']
            or bank.manifest['modes'] != m['modes']
            or camera.to_manifest_record()['camera_identity'] != view['camera_identity']
            or original['camera_identity'] != view['camera_identity']
            or flow.manifest['reference_selection']['identity'] != original['motion_reference']['selection_identity']
            or flow.manifest['reference_selection']['contract']['static_scene_identity'] != m['static_scene_identity']
            or flow.reference_identity != original['geometry_reference_flow_identity']
            or flow.manifest['reference_frame_index'] != ref
            or flow.manifest['reference_frame_name'] != view['reference_frame_name']
            or flow.manifest['frame_names'] != view['frame_names']
            or flow.manifest['fps_hz'] != view['fps_hz']):
        raise ValueError('Scene/bank/camera/reference/sequence binding differs')
    directory = resolve_path(images['directory'], strict=True)
    if directory != flow.image_directory:
        raise ValueError('Flow was computed on different images')
    frames = view['frames'][:n]
    if not np.allclose(np.diff([f['timestamp_seconds'] for f in frames]), 1/view['fps_hz']):
        raise ValueError('Nonuniform frame clock')
    checked = {}
    for i in sorted(set(range(n)) | {ref}):
        entry = images['files'][i]
        if entry['name'] != view['frame_names'][i]+'.png' or Path(entry['name']).name != entry['name']:
            raise ValueError('Image name differs')
        path = directory/entry['name']
        if sha256(path) != entry['sha256'] or entry['sha256'] != view['frames'][i]['image_sha256']:
            raise ValueError(f'Image checksum differs: {path}')
        checked[str(path)] = entry['sha256']
    preview = resolve_path(args.rgb_preview, strict=True)
    old = json.loads((preview/'manifest.json').read_text(encoding='utf-8'))
    if (old['static_scene_identity'] != m['static_scene_identity']
            or old['completed_modes_identity'] != m['completed_modes_identity']
            or old['frames'] != frames or old['modes'] != m['modes']
            or sha256(preview/'coordinates.npy') != old['coordinates']['sha256']):
        raise ValueError('RGB comparison does not use the same scene, modes and frames')
    old_q = np.load(preview/'coordinates.npy', allow_pickle=False)
    if old_q.shape != (n, len(m['modes'])) or old_q.dtype != np.complex64 or not np.isfinite(old_q).all():
        raise ValueError('Invalid comparison coordinates')
    valid = load_valid_mask(images, view['shape_hw']).numpy()
    weights = resolve_path(flow.manifest['model']['weights'], strict=True)
    source_files = [prepared/'manifest.json', bank.path/'manifest.json',
                    resolve_path(m['static_scene'])/'manifest.json', flow.path/'manifest.json',
                    preview/'manifest.json', preview/'coordinates.npy', weights]
    source_hashes = {str(p):sha256(p) for p in source_files}
    contract = dict(kind='reference_adjacent_flow_diagnostic', version=1,
        preparation_identity=m['preparation_identity'], static_scene_identity=m['static_scene_identity'],
        completed_modes_identity=m['completed_modes_identity'], modes=m['modes'],
        camera=camera.to_manifest_record(), shape_hw=view['shape_hw'], fps_hz=view['fps_hz'],
        frames=frames, reference_frame=view['frames'][ref], image_hashes=checked,
        validity=images['validity'], flow_identity=flow.identity, flow_source=str(flow.path),
        source_hashes=source_hashes, rgb_preview=str(preview),
        gauge='q = shared_reference_RGB_offset + reference_relative_flow_q; no temporal centering',
        approximation='fixed canonical projection; no angular-covariance term in flow design')
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'contract.json').exists():
        if json.loads((output/'contract.json').read_text(encoding='utf-8')) != contract:
            raise ValueError('Existing experiment input contract differs')
    else:
        atomic_json(output/'contract.json', contract)
    if gpu:
        scene = scene.to('cuda').eval().requires_grad_(False)
        camera = camera.to('cuda')
    return dict(output=output, contract=contract, contract_identity=identity(contract), scene=scene,
                scene_path=m['static_scene'], bank_path=m['mode_bank'],
                bank=bank, flow=flow, camera=camera, view=view, images=images,
                directory=directory, valid=valid, old_q=old_q)


def publish_record(work, ctx, extra):
    from modal_gaussians.common.cache import atomic_json, identity, sha256
    from modal_gaussians.flow.storage import storage_sha256
    source_root = Path(__file__).resolve().parents[1]/'src/modal_gaussians'
    implementations = ['coordinates/flow_constraints.py', 'coordinates/direct.py',
        'coordinates/rendering.py', 'coordinates/fitting.py', 'motion/common/projection.py',
        'motion/common/geometry_ops.py', 'geometry/scene.py', 'flow/sea_raft.py', 'flow/storage.py']
    checksums = {p.name:storage_sha256(p) if p.is_dir() else sha256(p)
                 for p in sorted(work.iterdir()) if p.name != 'manifest.json'}
    record = dict(kind='flow_comparison_'+extra.pop('stage'), version=1,
                  contract_identity=ctx['contract_identity'], checksums=checksums,
                  implementation={str(Path(__file__).name):sha256(Path(__file__)),
                                  **{p:sha256(source_root/p) for p in implementations}}, **extra)
    record['identity'] = identity(record)
    atomic_json(work/'manifest.json', record)


def read_stage(ctx, name):
    from modal_gaussians.common.cache import identity, sha256
    from modal_gaussians.flow.storage import storage_sha256
    path = ctx['output']/name
    record = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    saved = record.pop('identity')
    if identity(record) != saved or record['contract_identity'] != ctx['contract_identity']:
        raise ValueError(f'Invalid {name} contract')
    for filename, digest in record['checksums'].items():
        p = path/filename
        if Path(filename).name != filename or (storage_sha256(p) if p.is_dir() else sha256(p)) != digest:
            raise ValueError(f'Damaged {name}/{filename}')
    record['identity'] = saved
    return path, record


def design_stage(args):
    from modal_gaussians.coordinates.preparation import _publish
    from modal_gaussians.motion.common.projection import (
        RenderedDesignConfig, prepare_modal_projection, project_modal_features, uses_visible_subject)
    ctx = context(args, gpu=True)
    flow, scene, bank, camera = (ctx[k] for k in ('flow','scene','bank','camera'))
    settings = RenderedDesignConfig()
    if not uses_visible_subject(scene):
        raise ValueError('This experiment requires the saved manual 3D subject selection')
    with _publish(ctx['output']/'design') as work:
        pixels, alpha, jacobian, _ = prepare_modal_projection(
            scene, camera, flow.arrays.mask_union, settings, flow.arrays.valid_mask)
        design = np.lib.format.open_memmap(work/'design.npy', mode='w+', dtype=np.float32,
                                          shape=(len(pixels),2,2*len(bank.manifest['modes'])))
        for start in range(0, len(bank.manifest['modes']), settings.modes_per_batch):
            stop = min(start+settings.modes_per_batch, len(bank.manifest['modes']))
            design[:,:,2*start:2*stop] = project_modal_features(
                scene, camera, bank.arrays['phi'][start:stop], pixels, alpha, jacobian)
        design.flush(); del design
        np.save(work/'pixels.npy',pixels,allow_pickle=False)
        publish_record(work,ctx,dict(stage='design',settings=settings.to_dict(),
            sampling='visible manual-box subject, original reference-flow common support', samples=len(pixels)))
    flow.arrays.flow.store.close()


def pairs_stage(args):
    import torch
    from modal_gaussians.coordinates.preparation import _publish
    from modal_gaussians.flow.sea_raft import load_model, read_image
    from modal_gaussians.flow.storage import create_array
    ctx = context(args)
    source = ctx['flow'].manifest['model']
    model = load_model(source['repository'], Path(source['weights']).parent)
    if vars(model.args) != source['config']:
        raise ValueError('SEA-RAFT inference configuration differs from reference flow')
    names = ctx['view']['frame_names'][:args.frames]
    with _publish(ctx['output']/'pairs') as work:
        shape=(len(names)-1,*ctx['view']['shape_hw'],2)
        forward, backward = (create_array(work/(name+'.zarr'),shape,np.float32) for name in ('forward','backward'))
        block = int(forward.shards[0])
        with torch.inference_mode():
            previous = read_image(ctx['directory']/(names[0]+'.png')).cuda()
            for start in range(0,len(names)-1,block):
                end=min(start+block,len(names)-1)
                buffers=[np.empty((end-start,*shape[1:]),np.float32) for _ in range(2)]
                for t in range(start,end):
                    current = read_image(ctx['directory']/(names[t+1]+'.png')).cuda()
                    for values,(a,b) in zip(buffers,((previous,current),(current,previous))):
                        value=model(a,b,iters=model.args.iters,test_mode=True)['final'][0].permute(1,2,0).float().cpu().numpy()
                        if value.shape != shape[1:] or not np.isfinite(value).all():
                            raise ValueError('Invalid SEA-RAFT output')
                        values[t-start]=value
                    previous=current
                forward[start:end],backward[start:end]=buffers
                print(f'Adjacent pairs {end}/{len(names)-1}',flush=True)
        forward.store.close(); backward.store.close()
        publish_record(work,ctx,dict(stage='pairs',pairs=len(names)-1,model=source,
            direction=['t_to_t_plus_1','t_plus_1_to_t'],units='input_pixels'))
    ctx['flow'].arrays.flow.store.close()


def solve_stage(args):
    import cv2
    import torch
    from modal_gaussians.common.cache import atomic_json
    from modal_gaussians.coordinates.preparation import _publish
    from modal_gaussians.coordinates.flow_constraints import (
        FlowConstraintConfig, valid_positions, sample_adjacent_flow, flow_normal_equations,
        solve_flow_system, flow_residuals)
    from modal_gaussians.coordinates.rendering import make_rgb_renderer
    from modal_gaussians.coordinates.fitting import _rgb_loss
    from modal_gaussians.flow.storage import open_array, read_pixels
    ctx=context(args,gpu=True)
    design_path,dm=read_stage(ctx,'design'); pairs_path,pm=read_stage(ctx,'pairs')
    d=np.load(design_path/'design.npy',mmap_mode='r'); pixels=np.load(design_path/'pixels.npy')
    forward,backward=(open_array(pairs_path/(name+'.zarr')) for name in ('forward','backward'))
    n,p,k=args.frames,len(pixels),d.shape[-1]//2
    config=FlowConstraintConfig()
    with _publish(ctx['output']/'solution') as work:
        reference=np.lib.format.open_memmap(work/'reference_samples.npy',mode='w+',dtype=np.float32,shape=(n,p,2))
        adjacent=np.lib.format.open_memmap(work/'adjacent_samples.npy',mode='w+',dtype=np.float32,shape=(n-1,p,2))
        rvalid=np.zeros((n,p),bool); avalid=np.zeros((n-1,p),bool)
        flow=ctx['flow'].arrays.flow
        zero=read_pixels(flow,ctx['view']['reference_frame_index'],pixels)
        if not np.isfinite(zero).all() or np.max(np.abs(zero))>1e-6:
            raise ValueError('Reference-to-itself flow is not zero')
        for start in range(0,n,32):
            stop=min(n,start+32)
            reference[start:stop]=read_pixels(flow,slice(start,stop),pixels)-zero
        for t in range(n):
            rvalid[t]=valid_positions(pixels+reference[t],ctx['valid'])
            if t:
                adjacent[t-1],avalid[t-1]=sample_adjacent_flow(pixels,reference[t-1],
                    np.asarray(forward[t-1]),np.asarray(backward[t-1]),ctx['valid'],ctx['valid'],config)
            if (t+1)%50==0: print(f'Flow observations {t+1}/{n}',flush=True)
        forward.store.close(); backward.store.close(); flow.store.close()
        np.save(work/'reference_valid.npy',rvalid);np.save(work/'adjacent_valid.npy',avalid)
        reference.flush();adjacent.flush()
        started=time.perf_counter()
        system=flow_normal_equations(d,reference,adjacent,rvalid,avalid)
        relative=[solve_flow_system(system,FlowConstraintConfig(adjacent_weight=weight)) for weight in (0.,1.)]
        solve_seconds=time.perf_counter()-started
        np.save(work/'scales.npy',system['scales'])
        np.savez(work/'normal_equations.npz',**system)
        render=make_rgb_renderer(ctx['scene'],ctx['camera'],ctx['bank'].arrays['phi'],ctx['bank'].rotation)
        ref=ctx['view']['reference_frame_index']
        bgr=cv2.imread(str(ctx['directory']/(ctx['view']['frame_names'][ref]+'.png')))
        target=torch.tensor(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),device='cuda',dtype=torch.float32)/255
        valid=torch.tensor(ctx['valid'],device='cuda')
        scales=torch.tensor(system['scales'],dtype=torch.float32,device='cuda')
        parameter=torch.nn.Parameter(torch.zeros((k,2),device='cuda'))
        optimizer=torch.optim.Adam([parameter],lr=.01)
        best_loss=float('inf'); best=None; history=[]
        started=time.perf_counter()
        for step in range(201):
            optimizer.zero_grad(set_to_none=True)
            q=torch.view_as_complex(parameter)/scales
            loss=_rgb_loss(render(q,1.),target,valid)
            if not torch.isfinite(loss): raise ValueError('Non-finite reference RGB loss')
            value=float(loss.detach());history.append(dict(step=step,rgb_loss=value))
            if value<best_loss: best_loss=value;best=q.detach().cpu().numpy().copy()
            if step==200: break
            loss.backward()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError('Non-finite/disconnected reference offset gradient')
            optimizer.param_groups[0]['lr']=.01+(.001-.01)*step/199
            optimizer.step()
            if (step+1)%50==0: print(f'Reference offset {step+1}/200 loss={value:.8f}',flush=True)
        offset_seconds=time.perf_counter()-started
        np.save(work/'reference_offset.npy',best)
        residuals={}
        for name,rel in zip(('reference_only','reference_adjacent'),relative):
            np.save(work/(name+'_relative.npy'),rel)
            np.save(work/(name+'.npy'),(rel+best).astype(np.complex64))
            residuals[name]=flow_residuals(d,rel,reference,adjacent,rvalid,avalid)
        atomic_json(work/'residuals.json',residuals)
        atomic_json(work/'offset_history.json',history)
        del reference,adjacent
        publish_record(work,ctx,dict(stage='solution',settings=asdict(config),design_identity=dm['identity'],
            pairs_identity=pm['identity'],reference_counts=rvalid.sum(axis=1).tolist(),
            adjacent_counts=avalid.sum(axis=1).tolist(),linear_solve_seconds=solve_seconds,
            offset_seconds=offset_seconds,offset_best_loss=best_loss,offset_initial_loss=history[0]['rgb_loss'],
            offset_settings=dict(steps=200,lr_start=.01,lr_end=.001,scale=1.,initial='zero',best_includes_zero=True)))


def export_stage(args):
    import cv2
    import torch
    from modal_gaussians.common.cache import atomic_json, sha256
    from modal_gaussians.coordinates.preparation import _publish
    from modal_gaussians.coordinates.rendering import make_rgb_renderer
    from modal_gaussians.coordinates.fitting import _rgb_loss
    from modal_gaussians.preprocessing.frames import media_executable, _process_options
    ctx=context(args,gpu=True)
    solution,sm=read_stage(ctx,'solution')
    names=['existing_RGB','reference_only','reference_adjacent']
    coordinates=[ctx['old_q']]+[np.load(solution/(name+'.npy')) for name in names[1:]]
    render=make_rgb_renderer(ctx['scene'],ctx['camera'],ctx['bank'].arrays['phi'],ctx['bank'].rotation)
    h,w=ctx['view']['shape_hw'];fps=ctx['view']['fps_hz']
    valid=torch.tensor(ctx['valid'],device='cuda');metrics={name:[] for name in names}
    with _publish(ctx['output']/'comparison') as work:
        command=[media_executable('ffmpeg'),'-hide_banner','-loglevel','error','-nostdin','-n',
            '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{2*w}x{2*h}',
            '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','18',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(work/'comparison.mp4')]
        with (work/'encode.log').open('wb') as log:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,**_process_options())
            try:
                with torch.inference_mode():
                    for t in range(args.frames):
                        bgr=cv2.imread(str(ctx['directory']/(ctx['view']['frame_names'][t]+'.png')))
                        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                        target=torch.tensor(rgb,device='cuda',dtype=torch.float32)/255
                        panels=[rgb.copy()]
                        for name,qs in zip(names,coordinates):
                            pred=render(torch.tensor(qs[t],device='cuda'),1.)
                            if not torch.isfinite(pred).all(): raise ValueError('Non-finite render')
                            mse=float((pred-target)[valid].square().mean())
                            metrics[name].append(dict(frame=t,rmse=mse**.5,psnr=-10*np.log10(max(mse,1e-15)),
                                rgb_loss=float(_rgb_loss(pred,target,valid))))
                            panels.append(pred.clamp(0,1).mul(255).round().byte().cpu().numpy())
                        for panel,label in zip(panels,['INPUT',*names]):
                            cv2.putText(panel,label,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4,cv2.LINE_AA)
                            cv2.putText(panel,label,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
                        grid=np.concatenate((np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)),axis=0)
                        process.stdin.write(grid.tobytes())
                        if t in (0,106,139,299): cv2.imwrite(str(work/f'frame_{t:04d}.png'),cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))
                        if (t+1)%50==0: print(f'Comparison render {t+1}/{args.frames}',flush=True)
                process.stdin.close()
                if process.wait()!=0: raise RuntimeError('FFmpeg failed')
            except BaseException:
                if process.poll() is None: process.kill()
                process.wait();raise
        cap=cv2.VideoCapture(str(work/'comparison.mp4'));count=0
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS)!=fps: raise ValueError('Video clock differs')
        while True:
            ok,frame=cap.read()
            if not ok: break
            if frame.shape!=(2*h,2*w,3): raise ValueError('Video shape differs')
            count+=1
        cap.release()
        if count!=args.frames: raise ValueError('Video is incomplete')
        summary={name:{metric:float(np.mean([r[metric] for r in rows])) for metric in ('rmse','psnr','rgb_loss')}
                 for name,rows in metrics.items()}
        atomic_json(work/'metrics.json',dict(per_frame=metrics,summary=summary,
            region='stabilization-valid full image',render_values='unclipped float RGB before video encoding',
            interpretation='Used-frame reconstruction diagnostics; not an equal-budget or novel-view benchmark'))
        # Reload source artifacts to recheck the original arrays after all computation.
        from modal_gaussians.geometry.scene import load_static_scene
        from modal_gaussians.motion.common.completed_modes import load_completed_modes
        load_static_scene(ctx['scene_path'],validate=True)
        load_completed_modes(ctx['bank_path'],validate=True)
        for path,digest in {**ctx['contract']['source_hashes'],**ctx['contract']['image_hashes']}.items():
            if sha256(Path(path))!=digest: raise ValueError(f'Source changed: {path}')
        publish_record(work,ctx,dict(stage='comparison',solution_identity=sm['identity'],frames=count,
            fps_hz=fps,shape_hw=[2*h,2*w],layout=['input','existing_RGB','reference_only','reference_adjacent'],
            metrics=summary,source_hashes_unchanged=True))
    ctx['flow'].arrays.flow.store.close()


def plot_stage(args):
    import os
    import tempfile
    output=Path(args.output).resolve()
    destination=output/'curves'
    if destination.exists(): raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix='.curves.',dir=output) as temporary:
        _plot_stage(args,Path(temporary))
        os.rename(temporary,destination)


def _plot_stage(args, destination):
    # Separate plotting interpreter may be used; this stage does not import Torch.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import csv
    from modal_gaussians.common.cache import atomic_json, identity, sha256
    output=Path(args.output).resolve();contract=json.loads((output/'contract.json').read_text(encoding='utf-8'))
    ctx=dict(output=output,contract=contract,contract_identity=identity(contract))
    # Validate only file outputs here, avoiding the Torch-dependent flow storage import.
    solution=output/'solution';m=json.loads((solution/'manifest.json').read_text(encoding='utf-8'))
    digest=m.pop('identity')
    if identity(m)!=digest or m['contract_identity']!=identity(contract): raise ValueError('Solution identity differs')
    for name,digest in m['checksums'].items():
        if sha256(solution/name)!=digest: raise ValueError(f'Solution checksum differs: {name}')
    names=['existing_RGB','reference_only','reference_adjacent']
    preview=Path(contract['rgb_preview'])
    if sha256(preview/'coordinates.npy')!=contract['source_hashes'][str(preview/'coordinates.npy')]:
        raise ValueError('RGB comparison coefficients changed')
    q=[np.load(Path(contract['rgb_preview'])/'coordinates.npy')]+[np.load(solution/(n+'.npy')) for n in names[1:]]
    scales=np.load(solution/'scales.npy');times=np.array([f['timestamp_seconds'] for f in contract['frames']])
    summary={}
    with PdfPages(destination/'curves.pdf') as pdf:
        for component in ('real','imag','magnitude','normalized_change'):
            fig,axes=plt.subplots(5,4,figsize=(17,15),sharex=True)
            for k,ax in enumerate(axes.flat):
                if k>=len(scales): ax.set_visible(False);continue
                for name,values in zip(names,q):
                    curve={'real':values[:,k].real,'imag':values[:,k].imag,'magnitude':abs(values[:,k]),
                           'normalized_change':abs(np.diff(values[:,k]))*scales[k]}[component]
                    ax.plot(times[1:] if component=='normalized_change' else times,curve,label=name,linewidth=.8)
                ax.set_title(f"{contract['modes'][k]['frequency_hz']:.3f} Hz");ax.grid(alpha=.2)
                ax.set_xlabel('seconds')
            axes.flat[0].legend(fontsize=7)
            fig.suptitle(component+' — original coefficients, no smoothing')
            fig.tight_layout(rect=(0,0,1,.975));fig.savefig(destination/(component+'.png'),dpi=130);pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(13,4))
        for name,values in zip(names,q):
            change=np.sqrt(np.mean(abs(np.diff(values,axis=0)*scales)**2,axis=1))
            ax.plot(times[1:],change,label=name,linewidth=1)
            summary[name]=dict(mean_normalized_change=float(change.mean()),max_normalized_change=float(change.max()),
                peak_destination_frame=int(change.argmax()+1))
        ax.set(xlabel='seconds',ylabel='RMS |scale * delta q|');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(destination/'normalized_change_summary.png',dpi=150);pdf.savefig(fig);plt.close(fig)
    with (destination/'coefficients.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['method','frame','seconds','mode','Hz','real','imag','magnitude','normalized_delta'])
        for name,values in zip(names,q):
            for t in range(len(values)):
                for k in range(len(scales)):
                    value=values[t,k]
                    writer.writerow([name,t,times[t],k,contract['modes'][k]['frequency_hz'],value.real,value.imag,abs(value),
                                     abs(value-values[t-1,k])*scales[k] if t else ''])
    atomic_json(destination/'summary.json',summary)
    atomic_json(destination/'manifest.json',dict(kind='flow_comparison_curves',contract_identity=identity(contract),
        files={p.name:sha256(p) for p in destination.iterdir()},solution_identity=json.loads((solution/'manifest.json').read_text())['identity']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['design','pairs','solve','export','plot'])
    parser.add_argument('--prepared',required=True)
    parser.add_argument('--flow',required=True)
    parser.add_argument('--rgb-preview',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--view',default='view1')
    parser.add_argument('--frames',type=int,default=300)
    args=parser.parse_args()
    from modal_gaussians.common.cache import atomic_json
    started=time.perf_counter();state=dict(stage=args.stage,status='running')
    log=Path(args.output)/'logs'/(args.stage+'.json')
    if (Path(args.output)/{'export':'comparison','plot':'curves','solve':'solution'}.get(args.stage,args.stage)).exists():
        raise FileExistsError('Stage output already exists; choose a new experiment')
    atomic_json(log,state)
    try:
        globals()[args.stage+'_stage'](args)
        state['status']='complete'
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        state['elapsed_seconds']=time.perf_counter()-started
        atomic_json(log,state)
        print(json.dumps(state),flush=True)


if __name__=='__main__': main()
