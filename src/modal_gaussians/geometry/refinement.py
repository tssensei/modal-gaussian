"""Optimize Gaussian geometry/appearance under frozen material motion and q."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from pytorch_msssim import ssim

from modal_gaussians.common.cache import atomic_json, identity, module_revision, sha256, exclusive_work
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.common.progress import report_progress
from modal_gaussians.coordinates.refinement_artifacts import load_prepared
from modal_gaussians.coordinates.rendering import apply_angular_rotation
from modal_gaussians.coordinates.sequences import frame_camera
from modal_gaussians.motion.material_field import MaterialField
from modal_gaussians.results.artifact import _load_sources
from .density import densify_parameters, remap_parameters, split_positions
from .scene import StaticDataset, SceneNormalization, cameras_from_scene_manifest
from .training import StaticTrainer, StaticTrainConfig, _learning_rates, _atomic_torch_save


@dataclass(frozen=True)
class SceneRefineConfig(StaticTrainConfig):
    iterations: int = 7000
    seed: int = 1729
    density_stop_step: int = 5000
    foreground_densify_stop_step: int = 5000
    background_densify_stop_step: int = 5000
    opacity_reset_every: int = 3000
    max_background_gaussians: int = 400000
    max_foreground_gaussians: int = 800000
    source_step: int = 3000
    evaluation_every_steps: int = 500
    query_block_size: int = 4096

    def validate(self):
        super().validate()
        for key in ('max_foreground_gaussians','evaluation_every_steps','query_block_size'):
            if type(getattr(self,key)) is not int or getattr(self,key)<=0:
                raise ValueError(f'Invalid scene refinement {key}')
        if type(self.source_step) is not int or self.source_step<0 or self.depth_weight != 0:
            raise ValueError('Scene refinement requires nonnegative source clock and RGB-only loss')
        if not (self.density_warmup_steps<self.opacity_reset_every<self.density_stop_step
                and self.opacity_reset_every%self.density_control_every==0):
            raise ValueError('Opacity reset must coincide with one density event')

    def resolved(self):
        import sys
        from . import density,scene,refinement_artifacts,training
        from modal_gaussians.motion import material_field,reference_field
        from modal_gaussians.common import camera_rendering
        from modal_gaussians.coordinates import transfer,rendering,sequences
        return dict(**asdict(self),method='frozen_material_gaussian_refinement_v1',
            loss='L1 + 0.2*(1-SSIM)',learning_rates=_learning_rates(),
            implementation=module_revision(sys.modules[__name__],density,scene,refinement_artifacts,
                material_field,reference_field,camera_rendering,transfer,training,rendering,sequences))


class SceneRefineTrainer(StaticTrainer):
    def __init__(self,scene,bank,coordinates,reference,config,work_dir,device,motion,run_identity,source_scene):
        config.validate()
        m=scene.manifest
        if scene.sh_degree!=3:raise ValueError('Scene refinement requires the published degree-3 SH scene')
        dataset=StaticDataset(resolve_path(m['dataset']['input_root'],strict=True),cameras_from_scene_manifest(m),
            np.empty((0,3)),np.empty((0,3)),SceneNormalization.from_dict(m['scene_normalization']),
            m['dataset']['dataset_identity'],m['dataset']['file_identities'])
        super().__init__(scene,dataset,config,torch.device(device),work_dir,dict(m['counts']))
        self.source_scene=str(resolve_path(source_scene));self.run_identity=run_identity;self.motion=motion
        self.field=MaterialField(reference,bank.arrays['control_displacement'],bank.arrays['control_angular'],config.query_block_size)
        self.lineage={p:np.arange(getattr(scene,p).count,dtype=np.int64) for p in ('foreground','background')}
        self.views=coordinates.manifest['views'];self.images=coordinates.manifest['images']
        if len(self.views)!=1 or self.views[0]['kind']!='sweep' or self.views[0]['fps_hz']!=30:
            raise ValueError('Scene refinement requires one actual 30 FPS sweep')
        cameras={c.name:c for c in dataset.cameras}
        self.frames=[frame_camera(cameras,self.views[0],i).to(device) for i in range(len(coordinates.coordinates))]
        q=np.array(coordinates.coordinates,copy=True)
        if motion=='zero':q.fill(0)
        self.q=torch.as_tensor(q,device=device)
        self.order=[];self.cursor=0;self.history=[];self.evaluations=[]
        self.best_loss=float('inf');self.best_step=0;self.best_sha256=None
        self.support_rejections=0;self._set_rates()

    def _configure_optimizers(self):
        values={}
        for part in ('foreground','background'):
            for key,p in getattr(self.scene,part).params.items():
                values[f'{part}.{key}']=torch.optim.Adam([p],lr=_learning_rates()[part][key],eps=1e-15)
        return values,{}

    def _set_rates(self):
        clock=self.config.source_step+self.global_step
        for part in ('foreground','background'):
            for key,rate in _learning_rates()[part].items():
                if key=='means':rate*=self.camera_extent*.01**min(clock/self.config.max_lr_steps,1.)
                self.optimizers[f'{part}.{key}'].param_groups[0]['lr']=rate

    def _next_batch(self):
        frames=[]
        while len(frames)<self.config.batch_size:
            if self.cursor>=len(self.order):
                self.order=np.random.permutation(len(self.frames)).tolist();self.cursor=0
            count=min(self.config.batch_size-len(frames),len(self.order)-self.cursor)
            frames.extend(self.order[self.cursor:self.cursor+count]);self.cursor+=count
        return frames

    def _target(self,index):
        return (self._camera_pixels(self.frames[index]).float()/255.).to(self.device)

    @staticmethod
    def _rgb(prediction,target):
        return (prediction-target).abs().mean()+.2*(1-ssim(prediction.permute(2,0,1)[None],
            target.permute(2,0,1)[None],data_range=1.,size_average=True))

    @torch.no_grad()
    def check_initial_field(self,bank):
        errors=[]
        x=self.scene.foreground.params['means']
        for k,values in enumerate(self.field.bake(x,self.lineage['foreground'])):
            for name,value in zip(('phi','rotation'),values):
                expected=bank.arrays['phi'][k] if name=='phi' else bank.rotation[k]
                e=torch.as_tensor(np.array(expected),device=self.device)
                scale=max(float(e.abs().square().mean().sqrt()),1e-12)
                error=float((value-e).abs().max())/scale
                errors.append(dict(mode=k,field=name,normalized_max_error=error))
                if error>1e-5:raise ValueError(f'Initial material field differs: {errors[-1]}')
        return errors

    def update(self):
        tick=time.perf_counter();indices=self._next_batch();self._set_rates()
        for opt in self.optimizers.values():opt.zero_grad(set_to_none=True)
        base=self.scene.foreground.params
        old=base['means'].detach().clone()
        query_start=time.perf_counter()
        if self.motion=='zero':
            deformed=base['means'][None].expand(len(indices),-1,-1)
            angles=None
        else:
            deformed,angles=self.field.deform(base['means'],self.lineage['foreground'],self.q[indices])
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        render_means=deformed.detach().requires_grad_(True)
        render_angles=None if angles is None else angles.detach().requires_grad_(True)
        query_seconds=time.perf_counter()-query_start
        losses=[];render_start=time.perf_counter()
        for j,i in enumerate(indices):
            orientations=base['quaternions'] if angles is None else apply_angular_rotation(base['quaternions'],render_angles[j])
            prediction,info=self.scene.render_deformed_with_info(self.frames[i],render_means[j],
                foreground_quaternions=orientations,retain_screen_grad=True,include_depth=False)
            loss=self._rgb(prediction['rgb'],self._target(i))
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite scene refinement loss')
            (loss/len(indices)).backward()
            # StaticTrainer removes batched-loss averaging; each sequential render has B=1.
            info['means2d'].grad.mul_(len(indices))
            self._accumulate_density_stats(info,[self.frames[i]])
            losses.append(float(loss.detach()))
        render_seconds=time.perf_counter()-render_start
        backward_start=time.perf_counter()
        if angles is None:torch.autograd.backward(deformed,render_means.grad)
        else:torch.autograd.backward((deformed,angles),(render_means.grad,render_angles.grad))
        for p in self.scene.parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise FloatingPointError('Missing/non-finite Gaussian gradient')
        for opt in self.optimizers.values():opt.step()
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        backward_seconds=time.perf_counter()-backward_start
        support_start=time.perf_counter()
        with torch.no_grad():
            bad=~self.field.supported(base['means'],self.lineage['foreground'])
            base['means'][bad]=old[bad]
            state=self.optimizers['foreground.means'].state[base['means']]
            for key in ('exp_avg','exp_avg_sq'):state[key][bad]=0
            rejected=int(bad.sum());self.support_rejections+=rejected
            if any(not torch.isfinite(p).all() for p in self.scene.parameters()):
                raise FloatingPointError('Non-finite updated Gaussian parameters')
        self.global_step+=1
        density=self.density_control()
        row=dict(step=self.global_step,rgb_loss=float(np.mean(losses)),frames=indices,
            foreground=self.scene.foreground.count,background=self.scene.background.count,
            support_rejected=rejected,query_seconds=query_seconds,render_seconds=render_seconds,
            backward_seconds=backward_seconds,support_density_seconds=time.perf_counter()-support_start,
            seconds=time.perf_counter()-tick,density=density)
        self.history.append(row)
        return row

    @torch.no_grad()
    def density_control(self):
        c=self.config;step=self.global_step
        if not (step>c.density_warmup_steps and step<c.density_stop_step and step%c.density_control_every==0):return None
        old_fg=self.scene.foreground.count
        scores=self.density_stats['screen_gradient_sum']/self.density_stats['visibility_count'].clamp_min(1)
        decision=dict(step=step)
        new_stats={key:[] for key in self.density_stats}
        for part_name,offset in (('foreground',0),('background',old_fg)):
            part=getattr(self.scene,part_name);count=part.count
            score=scores[offset:offset+count]
            high=score>c.densify_gradient_threshold
            large=part.active()['scales'].amax(-1)>c.densify_scale_threshold*self.camera_extent
            split,clone=high&large,high&~large
            stop=c.foreground_densify_stop_step if part_name=='foreground' else c.background_densify_stop_step
            if step>=stop:split.zero_();clone.zero_()
            cap=c.max_foreground_gaussians if part_name=='foreground' else c.max_background_gaussians
            split,clone,limited=self._limit_densify_candidates(split,clone,score,max(cap-count,0))
            children=split_positions(part,split)
            rejected=0
            if part_name=='foreground' and split.any():
                ids=split.nonzero().flatten().cpu().numpy()
                good=self.field.supported(children,np.repeat(self.lineage[part_name][ids],2)).reshape(-1,2).all(1)
                rejected=int((~good).sum())
                split[torch.as_tensor(ids,device=self.device)[~good]]=False
                children=children.reshape(-1,2,3)[good].reshape(-1,3)
            optimizers={key:self.optimizers[f'{part_name}.{key}'] for key in part.params}
            rows,newborn=densify_parameters(part,optimizers,split,clone,children=children)
            roots=self.lineage[part_name][rows.cpu().numpy()]
            stats={key:value[offset:offset+count][rows] for key,value in self.density_stats.items()}
            cull=part.active()['opacities']<c.cull_opacity_threshold
            if step>c.opacity_reset_every:
                cull|=part.active()['scales'].amax(-1)>c.cull_scale_threshold*self.camera_extent
                cull|=stats['maximum_screen_radius']>c.cull_screen_threshold
            keep=(~cull).nonzero().flatten()
            if len(keep)<2:raise RuntimeError(f'Density control would empty {part_name}')
            remap_parameters(part,optimizers,keep)
            self.lineage[part_name]=roots[keep.cpu().numpy()]
            for key in new_stats:new_stats[key].append(stats[key][keep])
            decision[part_name]=dict(before=count,after=part.count,split=int(split.sum()),clone=int(clone.sum()),
                culled=int(cull.sum()),limited=limited,unsupported_split=rejected)
        self.density_stats={key:torch.cat(value) for key,value in new_stats.items()}
        if step==c.opacity_reset_every:decision['opacity_reset']=self._reset_opacity()
        self.density_stats=self._new_density_stats()
        self.density_events.append(decision)
        return decision

    @torch.no_grad()
    def evaluate(self):
        # Bake once per check; fields are fixed across all evaluation frames.
        x=self.scene.foreground.params['means'];base=self.scene.foreground.params['quaternions']
        if self.motion=='fitted':
            fields=list(self.field.bake(x,self.lineage['foreground']))
            phi=torch.stack([v[0] for v in fields]);omega=torch.stack([v[1] for v in fields]);del fields
        total=0.
        for i,camera in enumerate(self.frames):
            if self.motion=='zero':means,rotation=x,base
            else:
                means=x+torch.einsum('k,kgc->gc',self.q[i],phi).real
                rotation=apply_angular_rotation(base,torch.einsum('k,kgc->gc',self.q[i],omega).real)
            outputs,_=self.scene.render_deformed_with_info(camera,means,foreground_quaternions=rotation,include_depth=False)
            total+=float(self._rgb(outputs['rgb'],self._target(i)))
        loss=total/len(self.frames)
        self.evaluations.append(dict(step=self.global_step,rgb_loss=loss,foreground=len(x),background=self.scene.background.count))
        if loss<self.best_loss:
            self.best_loss=loss;self.best_step=self.global_step
            self.best_sha256=None
            path=self.work_dir/f'best_{self.global_step:07d}.pt'
            _atomic_torch_save(self.payload(),path);self.best_sha256=sha256(path)
        atomic_json(self.work_dir/'evaluations.json',self.evaluations)
        return loss

    def payload(self):
        return dict(format='modal_gaussians.scene_refinement_checkpoint',version=1,run_identity=self.run_identity,
            scene_tensors=self.scene.tensor_dictionary(),sh_degree=self.scene.sh_degree,
            optimizers={k:v.state_dict() for k,v in self.optimizers.items()},global_step=self.global_step,
            lr_clock=self.config.source_step+self.global_step,lineage=self.lineage,
            density_stats={k:v.cpu() for k,v in self.density_stats.items()},density_events=self.density_events,
            rng_state=self._rng_state(),order=self.order,cursor=self.cursor,history=self.history,evaluations=self.evaluations,
            best_loss=self.best_loss,best_step=self.best_step,best_sha256=self.best_sha256,
            support_rejections=self.support_rejections,elapsed_seconds=self.elapsed_seconds())

    def save(self):
        _atomic_torch_save(self.payload(),self.work_dir/'checkpoint.pt')

    def restore(self,path):
        p=torch.load(path,map_location='cpu',weights_only=False)
        if (p.get('format')!='modal_gaussians.scene_refinement_checkpoint' or p.get('version')!=1
                or p['run_identity']!=self.run_identity or p['lr_clock']!=self.config.source_step+p['global_step']):
            raise ValueError('Scene refinement resume contract differs')
        for part in ('foreground','background'):
            for key in getattr(self.scene,part).params:
                getattr(self.scene,part).replace_parameter(key,p['scene_tensors'][f'{part}.{key}'].to(self.device))
        self.scene.sh_degree=p['sh_degree']
        self.optimizers,self.schedulers=self._configure_optimizers()
        for k,v in self.optimizers.items():v.load_state_dict(p['optimizers'][k])
        self.lineage=p['lineage'];self.global_step=p['global_step']
        for part,rows in self.lineage.items():
            if rows.dtype!=np.int64 or len(rows)!=getattr(self.scene,part).count:raise ValueError('Resume lineage shape differs')
        self.density_stats={k:v.to(self.device) for k,v in p['density_stats'].items()}
        if any(v.shape!=(self.scene.count,) for v in self.density_stats.values()):raise ValueError('Resume density shape differs')
        for key in ('density_events','order','cursor','history','evaluations','best_loss','best_step','best_sha256','support_rejections'):
            setattr(self,key,p[key])
        if self.best_sha256 is not None and sha256(self.work_dir/f'best_{self.best_step:07d}.pt')!=self.best_sha256:
            raise ValueError('Best scene checkpoint checksum differs')
        self.elapsed_seconds_before_resume=p['elapsed_seconds'];self.started_at=time.time()
        self._restore_rng_state(p['rng_state']);self._set_rates()


def refine_scene(*,scene_dir,completed_modes_dir,coordinates_dir,motion,config,work_dir,output_dir,
                 resume=False,device='cuda',stop_after=None):
    from .refinement_artifacts import publish_scene_refinement
    config.validate()
    if motion not in ('fitted','zero'):raise ValueError('Expected fitted or zero motion')
    work=resolve_path(work_dir);output=resolve_path(output_dir)
    if output.exists():raise FileExistsError(output)
    if not resume and work.exists() and any(work.iterdir()):raise FileExistsError(work)
    _,scene,bank,kind,coordinates,_,_,views=_load_sources(scene_dir=scene_dir,
        completed_modes_dir=completed_modes_dir,coordinates_dir=coordinates_dir)
    if bank.manifest['version']!=20 or kind!='sweep_rgb':raise ValueError('Expected refined v20 modes and independent sweep q')
    root,prepared,parent,reference,initial,operator=load_prepared(bank.manifest['prepared'])
    if (prepared['preparation_identity']!=bank.manifest['preparation_identity']
            or prepared['static_scene_identity']!=scene.manifest['static_scene_identity']
            or prepared['modes']!=bank.manifest['modes']):raise ValueError('Frozen reference source differs')
    del parent,initial,operator
    immutable=[resolve_path(p,strict=True) for p in (scene_dir,completed_modes_dir,coordinates_dir,root)]
    immutable.extend(resolve_path(v['directory'],strict=True) for v in coordinates.manifest['images'])
    if work==output or work.is_relative_to(output) or output.is_relative_to(work):raise ValueError('Work/output must be separate')
    if any(p.is_relative_to(v) or v.is_relative_to(p) for p in (work,output) for v in immutable):
        raise ValueError('Scene refinement must not overlap source artifacts')
    from .scene import load_static_scene
    from modal_gaussians.motion.common.completed_modes import load_completed_modes
    load_static_scene(scene_dir,validate=True);load_completed_modes(completed_modes_dir,validate=True)
    image=coordinates.manifest['images'][0]
    for entry in image['files']:
        if sha256(resolve_path(image['directory'],strict=True)/entry['name'])!=entry['sha256']:
            raise ValueError('Sweep training PNG changed')
    contract=dict(config=config.resolved(),motion=motion,static_scene_identity=scene.manifest['static_scene_identity'],
        modes_identity=bank.manifest['completed_modes_identity'],coordinates_identity=coordinates.manifest['sweep_coordinates_identity'],
        preparation_identity=prepared['preparation_identity'])
    run_identity=identity(contract)
    work.mkdir(parents=True,exist_ok=True)
    with exclusive_work(work/'run.lock'):
        random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(config.seed)
        trainer=SceneRefineTrainer(scene,bank,coordinates,reference,config,work,device,motion,run_identity,scene_dir)
        inputs=(scene,bank,coordinates,prepared)
        if resume:
            old=json.loads((work/'contract.json').read_text())
            if old!=contract:raise ValueError('Scene refinement work contract differs')
            trainer.restore(work/'checkpoint.pt')
        else:
            atomic_json(work/'contract.json',contract)
            atomic_json(work/'field_equivalence.json',trainer.check_initial_field(bank))
            trainer.evaluate();trainer.save()
        if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats()
        state=dict(status='running',run_identity=run_identity,motion=motion)
        atomic_json(work/'run.json',state)
        try:
            while trainer.global_step<config.iterations:
                row=trainer.update()
                with (work/'history.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
                if trainer.global_step%config.evaluation_every_steps==0 or trainer.global_step==config.iterations:
                    loss=trainer.evaluate();report_progress(f'Scene refine {motion}: step={trainer.global_step}, full RGB={loss:.8f}')
                if row['density'] or trainer.global_step%config.checkpoint_every_steps==0:trainer.save()
                if trainer.global_step==1 or trainer.global_step%25==0:
                    report_progress(f'Scene refine {motion}: step={trainer.global_step}/{config.iterations}, loss={row["rgb_loss"]:.6f}, '
                        f'fg={row["foreground"]}, bg={row["background"]}, seconds={row["seconds"]:.3f}')
                    atomic_json(work/'progress.json',row)
                if stop_after is not None and trainer.global_step>=stop_after:
                    trainer.save();state.update(status='paused_after_smoke',actual_steps=trainer.global_step)
                    atomic_json(work/'run.json',state);return None
            trainer.save()
            state.update(actual_steps=trainer.global_step,best_loss=trainer.best_loss,published_step=trainer.best_step,
                elapsed_seconds=trainer.elapsed_seconds(),support_rejections=trainer.support_rejections,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.device(device).type=='cuda' else 0)
            trainer.restore(work/f'best_{trainer.best_step:07d}.pt')
            publish_scene_refinement(output,trainer,inputs)
            state['status']='complete'
        except BaseException as error:
            state.update(status='failed',error=repr(error),actual_steps=trainer.global_step)
            raise
        finally:atomic_json(work/'run.json',state)
        return output
