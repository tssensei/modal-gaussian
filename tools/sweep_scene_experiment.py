"""Explicit A/B orchestration; each stage is independently restartable."""
import argparse
import csv
import importlib.util
import json
import shutil
from pathlib import Path
import subprocess
import sys
import time
from fractions import Fraction

import cv2
import numpy as np
import torch

from modal_gaussians.common.cache import atomic_json,sha256,identity,exclusive_work
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.coordinates.preparation import _publish
from modal_gaussians.coordinates.transfer import transfer_coordinates
from modal_gaussians.coordinates.rendering import make_sequence_renderer
from modal_gaussians.coordinates.sequences import frame_camera
from modal_gaussians.geometry.refinement import SceneRefineConfig,refine_scene
from modal_gaussians.geometry.scene import cameras_from_scene_manifest
from modal_gaussians.preprocessing.frames import media_executable,_process_options
from modal_gaussians.results.artifact import materialize_modal_result,load_modal_result
from modal_gaussians.results.evaluation import evaluate_result
from modal_gaussians.vis.inputs import load_viewer_input


def bind(scene,modes,coordinates,path):
    if not path.exists():
        materialize_modal_result(scene_dir=scene,completed_modes_dir=modes,coordinates_dir=coordinates,output_dir=path)
    result=load_modal_result(path)
    expected=[resolve_path(v,strict=True) for v in (scene,modes,coordinates)]
    actual=[resolve_path(result.manifest['sources'][name]['path'],strict=True) for name in ('static_scene','completed_modes','coordinates_0')]
    if actual!=expected:raise ValueError('Existing result sources differ')
    return path


def early_stop_due(evaluations,policy):
    """Use only exhaustive evaluations; small improvements may accumulate."""
    if (policy['patience']<1 or policy['minimum_step']<0
            or not 0<policy['relative_improvement']<1):
        raise ValueError('Invalid experiment stopping policy')
    significant=float('inf');stale=0
    for row in evaluations:
        value=row['rgb_loss']
        if not np.isfinite(value):raise ValueError('Non-finite evaluation')
        if value<significant*(1-policy['relative_improvement']):
            significant=value;stale=0
        else:stale+=1
    return bool(evaluations and evaluations[-1]['step']>=policy['minimum_step'] and stale>=policy['patience'])


def finish_early(recipe,root):
    """Publish a stopped, complete checkpoint without altering its resume contract."""
    from modal_gaussians.geometry.refinement import SceneRefineTrainer
    from modal_gaussians.geometry.refinement_artifacts import publish_scene_refinement
    from modal_gaussians.coordinates.refinement_artifacts import load_prepared
    from modal_gaussians.results.artifact import _load_sources
    from modal_gaussians.geometry.scene import load_static_scene
    from modal_gaussians.motion.common.completed_modes import load_completed_modes
    policy=json.loads((root/'early_stopping_policy.json').read_text())
    forced=policy.get('mode')=='user_requested_checkpoint_stop'
    for arm,motion in (('A_zero','zero'),('B_fitted','fitted')):
        output=root/arm/'published';work=root/arm/'work'
        if output.exists():continue
        # A live trainer still holds this lock: stop it only after a full checkpoint.
        with exclusive_work(work/'run.lock'):
            config=SceneRefineConfig(**recipe['config'])
            contract=json.loads((work/'contract.json').read_text())
            if contract['config']!=config.resolved() or contract['motion']!=motion:
                raise ValueError('Stopped checkpoint implementation/config differs')
            load_static_scene(recipe['scene'],validate=True);load_completed_modes(recipe['modes'],validate=True)
            _,scene,bank,kind,coordinates,*_=_load_sources(scene_dir=recipe['scene'],completed_modes_dir=recipe['modes'],coordinates_dir=recipe['coordinates'])
            _,prepared,_,reference,_,_=load_prepared(bank.manifest['prepared'])
            expected=dict(static_scene_identity=scene.manifest['static_scene_identity'],modes_identity=bank.manifest['completed_modes_identity'],
                coordinates_identity=coordinates.manifest['sweep_coordinates_identity'],preparation_identity=prepared['preparation_identity'])
            if kind!='sweep_rgb' or any(contract[k]!=v for k,v in expected.items()):raise ValueError('Stopped input identity differs')
            trainer=SceneRefineTrainer(scene,bank,coordinates,reference,config,work,'cuda',motion,identity(contract),recipe['scene'])
            trainer.restore(work/'checkpoint.pt')
            elapsed=trainer.elapsed_seconds_before_resume
            if not forced and (not trainer.evaluations or trainer.evaluations[-1]['step']!=trainer.global_step
                    or not early_stop_due(trainer.evaluations,policy)):
                raise ValueError('Complete checkpoint has not met the stopping policy')
            if forced and trainer.evaluations[-1]['step']!=trainer.global_step:
                if not (work/'checkpoint_at_stop.pt').exists():shutil.copyfile(work/'checkpoint.pt',work/'checkpoint_at_stop.pt')
                trainer.evaluate();trainer.save()  # Read-only evaluation; no Gaussian update.
            rows=[json.loads(line) for line in (work/'history.jsonl').read_text().splitlines()]
            completed=max(r['step'] for r in rows)
            state=dict(status='complete',run_identity=trainer.run_identity,motion=motion,
                actual_steps=trainer.global_step,completed_updates_before_stop=completed,
                discarded_updates_after_checkpoint=completed-trainer.global_step,
                published_step=trainer.best_step,best_loss=trainer.best_loss,
                elapsed_seconds=elapsed,support_rejections=trainer.support_rejections,
                peak_allocated_bytes=None,early_stopped=True,early_stopping_policy=policy)
            state['peak_memory_protocol']='original live allocator peak unavailable after external checkpoint stop; no further training performed'
            trainer.restore(work/f"best_{state['published_step']:07d}.pt")
            atomic_json(work/'run.json',dict(state,status='publishing_best_after_early_stop'))
            publish_scene_refinement(output,trainer,(scene,bank,coordinates,prepared))
            atomic_json(work/'run.json',state)
            print(f"Early-stopped {arm}: checkpoint {state['actual_steps']}, published {state['published_step']}",flush=True)


def local_lpips():
    return (importlib.util.find_spec('lpips') is not None and
        any((Path(torch.hub.get_dir())/'checkpoints').glob('alexnet-*.pth')))


@torch.inference_mode()
def videos(results,destination):
    """One lossless raw stream feeds four independent encodes and the comparison."""
    if destination.exists():return
    loaded=[load_modal_result(p) for p in results]
    view=loaded[0].manifest['views'][0]
    if any(r.manifest['views']!=loaded[0].manifest['views'] for r in loaded):
        raise ValueError('Video frame bindings differ')
    source=loaded[0].coordinates.manifest['images'][0]
    directory=resolve_path(source['directory'],strict=True)
    renderers=[]
    for r in loaded:
        cameras={c.name:c for c in cameras_from_scene_manifest(r.scene.manifest)}
        renderers.append(make_sequence_renderer(r.scene,[frame_camera(cameras,view,i) for i in range(view['frame_count'])],
            r.completed_modes.arrays['phi'],r.completed_modes.rotation))
    h,w=view['shape_hw'];count=view['frame_count']
    names=['original','coarse','A_zero','B_fitted','comparison_4way']
    with _publish(destination) as work:
        filters='[0:v]split=5[s0][s1][s2][s3][v4];'+';'.join(f'[s{i}]crop={w}:{h}:{i*w}:0[v{i}]' for i in range(4))
        command=[media_executable('ffmpeg'),'-hide_banner','-loglevel','error','-nostdin','-n',
            '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{4*w}x{h}','-framerate',str(view['fps_hz']),
            '-i','pipe:0','-filter_complex',filters]
        for i,name in enumerate(names):
            command+=['-map',f'[v{i}]','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(work/f'{name}.mp4')]
        with (work/'encode.log').open('wb') as log:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,**_process_options())
            try:
                for i,record in enumerate(source['files']):
                    file=directory/record['name']
                    if sha256(file)!=record['sha256']:raise ValueError('Original sweep image changed')
                    original=cv2.cvtColor(cv2.imread(str(file)),cv2.COLOR_BGR2RGB)
                    panels=[original]
                    for r,render in zip(loaded,renderers):
                        q=torch.tensor(r.coordinates.coordinates[i],device='cuda')
                        panels.append(render(i,q,1.).clamp(0,1).mul(255).round().to(torch.uint8).cpu().numpy())
                    row=np.concatenate(panels,axis=1);process.stdin.write(row.tobytes())
                    if i in (0,count//2,count-1):
                        cv2.imwrite(str(work/f'frame_{i:04d}.png'),cv2.cvtColor(row,cv2.COLOR_RGB2BGR))
                        crop=np.concatenate([p[h//4:3*h//4,w//4:3*w//4] for p in panels],axis=1)
                        cv2.imwrite(str(work/f'detail_{i:04d}.png'),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
                    if i%50==0:print(f'Four-way export {i}/{count}',flush=True)
                process.stdin.close()
                if process.wait()!=0:raise RuntimeError('Video encoder failed; inspect encode.log')
            finally:
                if process.poll() is None:process.kill();process.wait()
        for name in names:
            probe=json.loads(subprocess.check_output([media_executable('ffprobe'),'-v','error','-select_streams','v:0',
                '-count_frames','-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(work/f'{name}.mp4')],**_process_options()))['streams'][0]
            if (int(probe['nb_read_frames'])!=count or probe['height']!=h
                    or probe['width']!=(4*w if name=='comparison_4way' else w)
                    or float(Fraction(probe['r_frame_rate']))!=view['fps_hz']):
                raise ValueError('Exported video shape/count differs')
        atomic_json(work/'manifest.json',dict(frames=count,fps=view['fps_hz'],panel_order=names[:4],
            baseline='coarse scene with the original fitted sweep q',results=[str(p) for p in results],
            files={f.name:sha256(f) for f in work.iterdir() if f.is_file()}))


def deliver(recipe,root):
    started=time.perf_counter();perceptual=local_lpips()
    baseline=root/'baseline';baseline.mkdir(exist_ok=True)
    zero=baseline/'zero_coordinates'
    if not zero.exists():transfer_coordinates(source_dir=recipe['coordinates'],scene_dir=recipe['scene'],modes_dir=recipe['modes'],output_dir=zero,motion='zero')
    results={
        'coarse_zero':bind(recipe['scene'],recipe['modes'],zero,baseline/'zero_result'),
        'coarse_fitted':bind(recipe['scene'],recipe['modes'],recipe['coordinates'],baseline/'fitted_result'),
        'view1_coarse':bind(recipe['scene'],recipe['modes'],recipe['view1_coordinates'],baseline/'view1_result')}
    for arm in ('A_zero','B_fitted'):
        output=root/arm/'published'
        results[arm]=bind(output/'scene',output/'mode_bank',output/'coordinates',root/arm/'result')
        viewq=root/arm/'view1_coordinates'
        if not viewq.exists():transfer_coordinates(source_dir=recipe['view1_coordinates'],scene_dir=output/'scene',modes_dir=output/'mode_bank',output_dir=viewq)
        results[arm+'_view1']=bind(output/'scene',output/'mode_bank',viewq,root/arm/'view1_result')
    for name,path in results.items():
        target=root/'evaluations'/name
        if not target.exists():evaluate_result(result_dir=path,output_dir=target,with_lpips=perceptual)
        if name in ('A_zero','B_fitted','A_zero_view1','B_fitted_view1'):
            viewer=load_viewer_input(path)
            # Validate independent controls/reference graph without starting a server.
            from modal_gaussians.vis.viewer import ModalViewerData
            data=object.__new__(ModalViewerData);data.result=viewer;data.scene=viewer.scene
            data.select_mode(0);data._load_graph_display(0)
    videos([results['coarse_fitted'],results['A_zero'],results['B_fitted']],root/'delivery')
    commands=[]
    for port,name in enumerate(('A_zero','B_fitted','A_zero_view1','B_fitted_view1'),8091):
        commands.append(f'& "{sys.executable}" -m modal_gaussians.cli viewer --input "{results[name]}" --work-dir "{root/name/"viewer"}" --port {port}')
    (root/'VISER_COMMANDS.ps1').write_text('\n'.join(commands)+'\n',encoding='utf-8')
    atomic_json(root/'delivery_status.json',dict(status='complete',elapsed_seconds=time.perf_counter()-started,lpips=perceptual))


def report(root):
    names=('coarse_zero','coarse_fitted','A_zero','B_fitted','view1_coarse','A_zero_view1','B_fitted_view1')
    metrics={n:json.loads((root/'evaluations'/n/'metrics.json').read_text())['all_frames'] for n in names}
    differences={}
    for newer,older in (('A_zero','coarse_zero'),('B_fitted','coarse_fitted'),('B_fitted','A_zero'),
                        ('A_zero_view1','view1_coarse'),('B_fitted_view1','view1_coarse')):
        def rows(name):
            with (root/'evaluations'/name/'per_frame.csv').open(newline='') as stream:return list(csv.DictReader(stream))
        first,second=rows(newer),rows(older)
        if len(first)!=len(second):raise ValueError('Comparison frame count differs')
        delta=[]
        for a,b in zip(first,second):
            keys=('view','frame_index','frame_name','camera_name','timestamp_seconds')
            if any(a[k]!=b[k] for k in keys):raise ValueError('Comparison frame identity differs')
            delta.append({**{k:a[k] for k in keys},**{f'delta_{k}':float(a[k])-float(b[k]) for k in ('psnr_db','ssim','rmse')}})
        name=f'{newer}_minus_{older}'
        with (root/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(delta[0]));writer.writeheader();writer.writerows(delta)
        differences[name]={k:sum(r[k] for r in delta)/len(delta) for k in ('delta_psnr_db','delta_ssim','delta_rmse')}
    training={}
    for arm in ('A_zero','B_fitted'):
        work=root/arm/'work';run=json.loads((work/'run.json').read_text())
        history=[json.loads(v) for v in (work/'history.jsonl').read_text().splitlines()]
        # Recovery can leave speculative rows after the checkpoint; keep last row per step.
        history=list({r['step']:r for r in history if r['step']<=run['actual_steps']}.values());history.sort(key=lambda r:r['step'])
        events=[r['density'] for r in history if r['density']]
        totals={part:{key:sum(e[part][key] for e in events) for key in ('clone','split','culled','limited','unsupported_split')} for part in ('foreground','background')}
        timings={key:sum(r[key] for r in history) for key in ('query_seconds','render_seconds','backward_seconds','support_density_seconds','seconds')}
        training[arm]=dict(**run,events=totals,update_timing_totals=timings)
        with (root/arm/'gaussian_history.csv').open('w',newline='') as stream:
            keys=['step','foreground','background','support_rejected','rgb_loss','seconds']
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in history)
    atomic_json(root/'comparison.json',dict(metrics=metrics,differences=differences,training=training))
    lines=['# Sweep Gaussian scene refinement A/B','',
        'Both arms use frozen reference controls/cameras and a maximum budget of 7000 Gaussian updates. A uses zero sweep q; B uses the original fitted sweep q. Published states are selected only by full-sweep RGB loss. Actual budgets and any user-authorized early stopping are recorded below.','',
        '| Result | Frames | PSNR (dB) | SSIM | RMSE |','|---|---:|---:|---:|---:|']
    for name,m in metrics.items():lines.append(f"| {name} | {m['frames']} | {m['mean_psnr_db']:.6f} | {m['mean_ssim']:.6f} | {m['mean_rmse']:.6f} |")
    lines+=['','View1 uses the original 300-frame q, without fitting or checkpoint selection. It is an untrained-view check, not a fully independent test set.','',
        'Full deltas and per-frame comparisons: comparison.json and *_minus_*.csv. Original/coarse/A/B independent videos and the four-column comparison: delivery/. Viser commands: VISER_COMMANDS.ps1. No server is launched.','']
    for arm,t in training.items():
        peak=(f"{t['peak_allocated_bytes']/2**30:.3f} GiB" if t['peak_allocated_bytes'] is not None else 'unavailable after external checkpoint stop')
        lines.append(f"{arm}: actual {t['actual_steps']}, published {t['published_step']}; active training {t['elapsed_seconds']:.2f} s; peak allocated {peak}; position support rejections {t['support_rejections']}. Density counts and component timings are in comparison.json.")
        if t.get('early_stopped'):
            lines.append(f"User-authorized early stop: {t['early_stopping_policy']}. Discarded updates after retained checkpoint: {t['discarded_updates_after_checkpoint']}. Memory protocol: {t['peak_memory_protocol']}. A/B training budgets differ.")
    status=json.loads((root/'delivery_status.json').read_text())
    lines+=['','Original is the coordinate-bound registered sweep PNG sequence at native resolution and its recorded time grid.',
        'LPIPS computed.' if status['lpips'] else 'LPIPS omitted: local dependency or pretrained weights unavailable.',
        'Metrics are computed before video compression. Lower training RGB loss is not by itself evidence of better geometry. Visual inspection and source-integrity verification must be recorded separately.']
    (root/'RUN_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recipe',required=True)
    parser.add_argument('--stage',required=True,choices=['smoke','train','finish-early','deliver','report'])
    args=parser.parse_args();recipe=json.loads(Path(args.recipe).read_text(encoding='utf-8-sig'))
    root=resolve_path(recipe['experiment'])
    for key in ('scene','modes','coordinates','view1_coordinates'):
        recipe[key]=str(resolve_path(recipe[key],strict=True))
    protected=[Path(recipe[k]) for k in ('scene','modes','coordinates','view1_coordinates')]
    for key in ('coordinates','view1_coordinates'):
        manifest=json.loads((Path(recipe[key])/'manifest.json').read_text())
        protected.extend(resolve_path(v['directory'],strict=True) for v in manifest['images'])
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError('Experiment must not overlap immutable sources')
    root.mkdir(parents=True,exist_ok=True)
    settings=SceneRefineConfig(**recipe['config'])
    saved=root/'recipe.json'
    if saved.exists() and json.loads(saved.read_text())!=recipe:raise ValueError('Experiment recipe differs')
    if not saved.exists():atomic_json(saved,recipe)
    if args.stage=='deliver':deliver(recipe,root);return
    if args.stage=='report':report(root);return
    if args.stage=='finish-early':finish_early(recipe,root);return
    for arm,motion in (('A_zero','zero'),('B_fitted','fitted')):
        path=root/arm;path.mkdir(exist_ok=True)
        if (path/'published').exists():continue
        work=path/'work'
        checkpoint=work/'checkpoint.pt'
        previous=torch.load(checkpoint,map_location='cpu',weights_only=False)['global_step'] if checkpoint.exists() else 0
        if args.stage=='smoke' and previous>=25:continue
        if args.stage=='train' and previous<25:raise ValueError('Run smoke before full training')
        atomic_json(path/'config.json',recipe['config'])
        started=time.perf_counter()
        refine_scene(scene_dir=recipe['scene'],completed_modes_dir=recipe['modes'],coordinates_dir=recipe['coordinates'],
            motion=motion,config=settings,work_dir=work,output_dir=path/'published',
            resume=checkpoint.exists(),stop_after=25 if args.stage=='smoke' else None)
        with (root/'stage_timings.jsonl').open('a') as stream:stream.write(json.dumps(dict(arm=arm,stage=args.stage,seconds=time.perf_counter()-started))+'\n')
        torch.cuda.empty_cache()


if __name__=='__main__':main()
