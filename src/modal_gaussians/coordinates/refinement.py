"""Fixed-scene control-field refinement with independent per-frame coefficients."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from modal_gaussians.common.cache import atomic_json, exclusive_work, identity, module_revision, sha256
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.geometry.scene import cameras_from_scene_manifest
from modal_gaussians.motion import fixed_field
from modal_gaussians.motion.training import _atomic_torch
from .fitting import _rgb_loss
from .rendering import apply_angular_rotation, resize_rgb, scaled_camera
from .rgb import load_rgb_frame
from .refinement_artifacts import load_prepared, publish_refinement
from .sequences import frame_stride, sequence_weights


@dataclass(frozen=True)
class RefinementConfig:
    image_scale: float = 1.0
    warmup_passes: int = 10
    warmup_lr: float = 0.01
    warmup_final_lr: float = 0.001
    rounds: int = 2
    coefficient_epochs_per_round: int = 1
    fixed_joint_fps: float = 5.0
    sweep_joint_fps: float = 10.0
    coefficient_lr: float = 0.001
    coefficient_final_lr: float = 0.0001
    control_lr: float = 0.001
    control_final_lr: float = 0.0001
    anchor_weight: float = 1e-4
    field_anchor_weight: float = 1e-2
    rigidity_weight: float = 1e-3
    rotation_weight: float = 1e-3
    checkpoint_interval: int = 200
    query_block_size: int = 4096
    seed: int = 1729
    early_stop_patience: int = 0
    early_stop_joint_interval: int = 25
    early_stop_min_relative_improvement: float = 0.001

    def validate(self):
        if self.image_scale != 1.:
            raise ValueError('Motion refinement uses full resolution only')
        for key in ('warmup_passes','rounds','coefficient_epochs_per_round','checkpoint_interval','query_block_size','early_stop_joint_interval'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:
                raise ValueError(f'Invalid motion refinement {key}')
        if type(self.seed) is not int or not 0<=self.seed<2**32:
            raise ValueError('Invalid random seed')
        if type(self.early_stop_patience) is not int or self.early_stop_patience<0:
            raise ValueError('Invalid early-stop patience')
        if (type(self.early_stop_min_relative_improvement) not in (float,int)
                or not math.isfinite(self.early_stop_min_relative_improvement)
                or not 0<=self.early_stop_min_relative_improvement<1):
            raise ValueError('Invalid early-stop improvement threshold')
        for key in ('warmup_lr','warmup_final_lr','fixed_joint_fps','sweep_joint_fps',
                    'coefficient_lr','coefficient_final_lr','control_lr','control_final_lr',
                    'anchor_weight','field_anchor_weight','rigidity_weight','rotation_weight'):
            value = getattr(self,key)
            if type(value) not in (float,int) or not math.isfinite(value) or value<0 or (not key.endswith('weight') and value==0):
                raise ValueError(f'Invalid motion refinement {key}')


def update_early_stopping(state, loss, step, config):
    """Track the true minimum separately from cumulative significant progress."""
    if not math.isfinite(loss) or loss<0:
        raise FloatingPointError('Non-finite or negative early-stop RGB loss')
    improved = not state or loss<state['best_loss']
    if not state:
        state.update(best_loss=loss,best_step=step,progress_loss=loss,bad_checks=0,history=[])
    elif loss<state['progress_loss']*(1-config.early_stop_min_relative_improvement):
        state.update(progress_loss=loss,bad_checks=0)
    else:
        state['bad_checks']+=1
    if improved: state.update(best_loss=loss,best_step=step)
    state['stopped'] = state['bad_checks']>=config.early_stop_patience>0
    state['history'].append(dict(step=step,rgb_loss=loss,best_loss=state['best_loss'],bad_checks=state['bad_checks']))
    return improved


class RefinementTrainer:
    def __init__(self,scene,field,initial,views,config=RefinementConfig(),device='cuda'):
        config.validate()
        self.scene,self.field,self.views,self.config = scene.to(device),field,views,config
        self.scene.requires_grad_(False)
        self.device = torch.device(device)
        if any(v.get('kind')=='sweep' and v['fps_hz']!=30 for v in views):
            raise ValueError('Motion refinement requires a real 30 FPS sweep subset')
        self.weights = sequence_weights(views)
        self.scales = torch.as_tensor(initial['scales'],dtype=torch.float32,device=device)
        if self.scales.shape!=(len(views),field.mode_count) or not torch.isfinite(self.scales).all() or not (self.scales>0).all():
            raise ValueError('Invalid frozen coefficient normalization')
        self.q = [torch.nn.Parameter(torch.zeros(field.mode_count,2,device=device))
                  for _ in range(sum(v['frame_count'] for v in views))]
        if not self.q or [v['frame_offset'] for v in views]!=list(np.cumsum([0]+[v['frame_count'] for v in views[:-1]])):
            raise ValueError('Invalid sequence row layout')
        self.q0 = None
        if 'coordinates' in initial:
            values = np.asarray(initial['coordinates'])
            if values.dtype != np.complex64 or values.shape != (len(self.q),field.mode_count) or not np.isfinite(values).all():
                raise ValueError('Invalid initial complex coefficients')
            normalized = torch.as_tensor(values,device=device).clone()
            for view,scale in zip(views,self.scales):
                lo = view['frame_offset']
                normalized[lo:lo+view['frame_count']] *= scale
            if not torch.isfinite(normalized).all():
                raise ValueError('Non-finite normalized initial coefficients')
            with torch.no_grad():
                for p,value in zip(self.q,torch.view_as_real(normalized)):
                    p.copy_(value)
            self.q0 = torch.view_as_real(normalized).clone()
        self.qopt = [torch.optim.Adam([p],lr=config.warmup_lr if self.q0 is None else config.coefficient_lr) for p in self.q]
        self.controls = fixed_field.ControlCorrection(field.a,device)
        self.control_optimizer = torch.optim.Adam(self.controls.parameters(),lr=config.control_lr)
        self.rng = np.random.default_rng(config.seed)
        self.sparse_frames = [[self.select_joint_frames(v) for v in views] for _ in range(config.rounds)]
        self.joint_steps_per_round = max(map(len,self.sparse_frames[0]))
        self.coefficient_steps_per_round = len(self.q)*config.coefficient_epochs_per_round
        self.warmup_steps = len(self.q)*config.warmup_passes if self.q0 is None else 0
        self.round_steps = self.joint_steps_per_round+self.coefficient_steps_per_round
        self.total_steps = self.warmup_steps+config.rounds*self.round_steps
        self.step = self.joint_step = self.coefficient_step = 0
        self.baked_fields = None
        self.sampler_phase = None
        self.permutations,self.cursors,self.coefficient_order = [],[],np.empty(0,np.int64)
        # Each edge is evaluated from both endpoints, then averaged per supported node.
        a = field.a
        supported = np.any(np.diff(field.operator['query_ptr'],axis=1)>0,axis=0)
        edges = a['edges'][supported[a['edges']].all(axis=1)]
        lengths = a['lengths'][supported[a['edges']].all(axis=1)]
        if np.any(lengths<=0) or not np.isfinite(lengths).all():
            raise ValueError('Invalid fixed geometric edge lengths')
        degree = np.bincount(edges.ravel(),minlength=len(a['points']))
        weights = np.zeros(len(degree),np.float32)
        weights[degree>0] = 1/(degree[degree>0]*max(1,np.count_nonzero(degree)))
        self.edges = torch.as_tensor(edges,device=device,dtype=torch.long)
        self.lengths = torch.as_tensor(lengths,device=device,dtype=torch.float32)
        self.node_weights = torch.as_tensor(weights,device=device)
        self.enter_phase()

    def select_joint_frames(self,view):
        fps = self.config.sweep_joint_fps if view.get('kind')=='sweep' else self.config.fixed_joint_fps
        stride,count = frame_stride(view['fps_hz'],fps),view['frame_count']
        selected = np.asarray([self.rng.integers(lo,min(lo+stride,count)) for lo in range(0,count,stride)],np.int64)
        selected[0] = 0
        if len(selected)>1: selected[-1]=count-1
        return selected

    def phase_at(self,step):
        if step==self.total_steps: return (self.config.rounds,'complete',0)
        if step<self.warmup_steps: return (-1,'warmup',step)
        round_index,local = divmod(step-self.warmup_steps,self.round_steps)
        if local<self.joint_steps_per_round: return (round_index,'joint',local)
        return (round_index,'coefficient',local-self.joint_steps_per_round)

    @property
    def phase(self):
        return self.phase_at(self.step)[1]

    def enter_phase(self):
        key = self.phase_at(self.step)[:2]
        if key==self.sampler_phase: return
        self.sampler_phase = key
        self.control_optimizer.zero_grad(set_to_none=True)
        self.controls.requires_grad_(key[1]=='joint')
        self.permutations,self.cursors = [],[]
        self.coefficient_order = np.empty(0,np.int64)
        if key[1]=='joint':
            self.baked_fields = None
            self.permutations = [self.rng.permutation(p) for p in self.sparse_frames[key[0]]]
            self.cursors = [0]*len(self.views)
        elif key[1] in ('warmup','coefficient'):
            passes = self.config.warmup_passes if key[1]=='warmup' else self.config.coefficient_epochs_per_round
            self.coefficient_order = np.concatenate([self.rng.permutation(len(self.q)) for _ in range(passes)])

    def sample(self):
        result = []
        for v,view in enumerate(self.views):
            if self.cursors[v]==len(self.permutations[v]):
                self.permutations[v]=self.rng.permutation(self.sparse_frames[self.sampler_phase[0]][v]); self.cursors[v]=0
            frame = int(self.permutations[v][self.cursors[v]]); self.cursors[v]+=1
            result.append((v,frame,view['frame_offset']+frame))
        return result

    @torch.no_grad()
    def frozen_fields(self):
        if self.baked_fields is None:
            self.baked_fields = self.field.bake(*self.controls())
        return self.baked_fields

    def coefficient_rate(self):
        c=self.config
        if self.phase=='warmup':
            return c.warmup_lr+self.step/max(1,self.warmup_steps-1)*(c.warmup_final_lr-c.warmup_lr)
        return c.coefficient_lr+(self.step-self.warmup_steps)/max(1,self.total_steps-self.warmup_steps-1)*(c.coefficient_final_lr-c.coefficient_lr)

    def _loss(self,v,frame,index,current,angle,previous,previous_angle,render,target,valid_mask):
        started = time.perf_counter()
        base = self.scene.foreground.params['quaternions']
        rotation = apply_angular_rotation(base,angle)
        image,_ = render(v,frame,1.,current,rotation)
        rgb = _rgb_loss(image['rgb'],target(v,frame,1.),None if valid_mask is None else valid_mask(v,1.))
        rgb_seconds = time.perf_counter()-started
        started = time.perf_counter()
        zero = rgb*0
        anchor,rigid,relative = zero,zero,zero
        if self.phase!='warmup':
            anchor = (self.q[index]-self.q0[index]).square().sum(-1).mean()
            if frame>0:
                rigid,relative = fixed_field.temporal_graph_loss(current,previous,rotation,
                    apply_angular_rotation(base,previous_angle),self.edges,self.lengths,self.node_weights)
        c = self.config
        loss = rgb+c.anchor_weight*anchor+c.rigidity_weight*rigid+c.rotation_weight*relative
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite motion refinement loss')
        regularization_seconds = time.perf_counter()-started
        return loss,dict(label=self.views[v]['label'],frame=frame,rgb_loss=float(rgb.detach()),
                        coefficient_offset=float(anchor.detach()),rigidity_loss=float(rigid.detach()),rotation_loss=float(relative.detach()),
                        coefficient_magnitude=float(torch.view_as_complex(self.q[index].detach()).abs().mean()),
                        displacement_rms=float((current.detach()-self.scene.foreground.params['means']).square().mean().sqrt()),
                        angular_rms=float(angle.detach().square().mean().sqrt())),rgb_seconds,regularization_seconds

    def update(self,render,target,valid_mask=None):
        if self.step>=self.total_steps: raise RuntimeError('Refinement has completed')
        self.enter_phase()
        round_index,phase,phase_step = self.phase_at(self.step)
        joint = phase=='joint'
        if joint:
            samples = self.sample()
        else:
            index=int(self.coefficient_order[phase_step])
            v=next(v for v,view in enumerate(self.views) if view['frame_offset']<=index<view['frame_offset']+view['frame_count'])
            samples=[(v,index-self.views[v]['frame_offset'],index)]
        for _,_,index in samples:
            self.qopt[index].zero_grad(set_to_none=True)
            self.qopt[index].param_groups[0]['lr']=self.coefficient_rate()
        timing=dict(query_seconds=0.,query_backward_seconds=0.,render_seconds=0.,backward_seconds=0.,regularization_seconds=0.,bake_seconds=0.)
        begin=time.perf_counter()
        coefficients=[]
        pair_size = 1 if phase=='warmup' else 2
        for v,frame,index in samples:
            coefficients.append(torch.view_as_complex(self.q[index])/self.scales[v])
            if pair_size==2:
                coefficients.append(torch.view_as_complex(self.q[index-1 if frame>0 else index].detach())/self.scales[v])
        q=torch.stack(coefficients)
        if joint:
            self.control_optimizer.zero_grad(set_to_none=True)
            fraction=self.joint_step/max(1,self.config.rounds*self.joint_steps_per_round-1)
            self.control_optimizer.param_groups[0]['lr']=self.config.control_lr*(self.config.control_final_lr/self.config.control_lr)**fraction
            means,angular=self.field.deform(q,*self.controls())
            timing['query_seconds']=time.perf_counter()-begin
            render_means=means.detach().requires_grad_(True); render_angular=angular.detach().requires_grad_(True)
        else:
            phi,omega=self.frozen_fields()
            timing['bake_seconds']=time.perf_counter()-begin
            render_means=self.scene.foreground.params['means']+torch.einsum('sk,kgc->sgc',q,phi).real
            render_angular=torch.einsum('sk,kgc->sgc',q,omega).real
        losses=[]
        for s,(v,frame,index) in enumerate(samples):
            begin=time.perf_counter()
            current,previous = pair_size*s,pair_size*s+pair_size-1
            loss,record,rgb_seconds,regularization_seconds=self._loss(v,frame,index,render_means[current],render_angular[current],
                render_means[previous],render_angular[previous],render,target,valid_mask)
            weight=self.weights[v] if joint else self.weights[v]*len(self.q)/self.views[v]['frame_count']
            timing['render_seconds']+=rgb_seconds
            timing['regularization_seconds']+=regularization_seconds
            begin=time.perf_counter()
            (weight*loss).backward()
            timing['backward_seconds']+=time.perf_counter()-begin
            losses.append(record)
        field_penalty=0.
        if joint:
            begin=time.perf_counter()
            penalty=self.controls.penalty()
            (self.config.field_anchor_weight*penalty).backward()
            field_penalty=float(penalty.detach())
            timing['regularization_seconds']+=time.perf_counter()-begin
            begin=time.perf_counter()
            torch.autograd.backward((means,angular),(render_means.grad,render_angular.grad))
            timing['query_backward_seconds']=time.perf_counter()-begin
        parameters=[self.q[i] for _,_,i in samples]+(list(self.controls.parameters()) if joint else [])
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('Missing or non-finite motion refinement gradient')
        if joint:
            self.control_optimizer.step(); self.joint_step+=1; self.baked_fields=None
        else: self.coefficient_step+=1
        for _,_,index in samples: self.qopt[index].step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError('Non-finite motion refinement parameters')
        self.step+=1
        if self.step==self.warmup_steps:
            self.q0=torch.stack([p.detach().clone() for p in self.q])
        return dict(step=self.step,round=round_index,phase=phase,phase_step=phase_step+1,
                    phase_boundary=self.phase_at(self.step)[:2]!=self.sampler_phase,scale=1.,
                    joint_step=self.joint_step,coefficient_step=self.coefficient_step,
                    gaussian_count=self.scene.foreground.count,views=losses,field_anchor_loss=field_penalty,
                    control_correction_rms=float(self.controls.penalty().detach().sqrt()),**timing)

    def coordinates(self):
        rows=torch.stack([torch.view_as_complex(p.detach()) for p in self.q])
        for view,scale in zip(self.views,self.scales):
            lo=view['frame_offset']; rows[lo:lo+view['frame_count']]/=scale
        return rows.cpu().numpy().astype(np.complex64)

    @torch.no_grad()
    def mean_rgb_loss(self, render, target, valid_mask=None):
        """All bound frames, unchanged sampler/Adam; same group weights as training."""
        phi,omega=self.frozen_fields()
        base=self.scene.foreground.params
        scores=[]
        for v,view in enumerate(self.views):
            total=torch.zeros((),device=self.device)
            valid=None if valid_mask is None else valid_mask(v,1.)
            for frame in range(view['frame_count']):
                q=torch.view_as_complex(self.q[view['frame_offset']+frame])/self.scales[v]
                means=base['means']+torch.einsum('k,kgc->gc',q,phi).real
                angle=torch.einsum('k,kgc->gc',q,omega).real
                pred,_=render(v,frame,1.,means,apply_angular_rotation(base['quaternions'],angle))
                total+=_rgb_loss(pred['rgb'],target(v,frame,1.),valid)
            scores.append(float(total/view['frame_count']))
        if not all(math.isfinite(v) for v in scores):
            raise FloatingPointError('Non-finite early-stop sequence loss')
        return sum(w*s for w,s in zip(self.weights,scores)),scores

    def state_dict(self):
        r=np.random.get_state()
        return dict(step=self.step,joint_step=self.joint_step,coefficient_step=self.coefficient_step,
            sampler_phase=self.sampler_phase,sparse_frames=[[torch.from_numpy(p) for p in r] for r in self.sparse_frames],
            coefficient_order=torch.from_numpy(self.coefficient_order),permutations=[torch.from_numpy(p) for p in self.permutations],
            cursors=self.cursors,q=[p.detach().cpu() for p in self.q],qopt=[o.state_dict() for o in self.qopt],
            q0=None if self.q0 is None else self.q0.cpu(),controls=self.controls.state_dict(),
            control_optimizer=self.control_optimizer.state_dict(),rng=self.rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],
            numpy_rng=dict(name=r[0],keys=torch.from_numpy(r[1].astype(np.int64)),position=r[2],gauss=r[3],cached=r[4]))

    def load_state_dict(self,state):
        step=state['step']
        joint=max(0,step-self.warmup_steps)//self.round_steps*self.joint_steps_per_round+min(max(0,step-self.warmup_steps)%self.round_steps,self.joint_steps_per_round)
        if not 0<=step<=self.total_steps or state['joint_step']!=joint or state['coefficient_step']!=step-joint:
            raise ValueError('Invalid refinement checkpoint counters')
        key=tuple(state['sampler_phase'])
        if key not in (self.phase_at(step)[:2],self.phase_at(max(0,step-1))[:2]):
            raise ValueError('Invalid checkpoint phase')
        if (len(state['q'])!=len(self.q) or len(state['qopt'])!=len(self.qopt)
                or any(p.shape!=q.shape or not torch.isfinite(p).all() for p,q in zip(state['q'],self.q))
                or len(state['sparse_frames'])!=len(self.sparse_frames)
                or any(len(a)!=len(b) or any(not np.array_equal(x.numpy(),y) for x,y in zip(a,b))
                       for a,b in zip(state['sparse_frames'],self.sparse_frames))):
            raise ValueError('Invalid checkpoint coefficients/selection')
        if key[1]=='joint':
            if (len(state['permutations'])!=len(self.views) or len(state['cursors'])!=len(self.views)
                    or state['coefficient_order'].numel()!=0):
                raise ValueError('Invalid joint sampler')
            for permutation,cursor,selection in zip(state['permutations'],state['cursors'],self.sparse_frames[key[0]]):
                if not np.array_equal(permutation.sort().values.numpy(),selection) or not 0<=cursor<=len(selection):
                    raise ValueError('Invalid joint sampler order')
        else:
            passes=self.config.warmup_passes if key[1]=='warmup' else self.config.coefficient_epochs_per_round
            order=state['coefficient_order']
            if (order.shape!=(passes*len(self.q),) or state['permutations'] or state['cursors']
                    or any(not torch.equal(p.sort().values,torch.arange(len(self.q))) for p in order.split(len(self.q)))):
                raise ValueError('Invalid exhaustive sampler order')
        anchor=state['q0']
        if (step>=self.warmup_steps)!=(anchor is not None) or (anchor is not None and
                (anchor.shape!=(len(self.q),self.field.mode_count,2) or not torch.isfinite(anchor).all())):
            raise ValueError('Invalid post-warmup anchor')
        for name,value in self.controls.state_dict().items():
            saved=state['controls'][name]
            if saved.shape!=value.shape or not torch.isfinite(saved).all() or (name!='delta' and not torch.equal(saved.to(value.device),value)):
                raise ValueError('Checkpoint control prior differs')
        self.controls.load_state_dict(state['controls'])
        self.control_optimizer.load_state_dict(state['control_optimizer'])
        with torch.no_grad():
            for p,value,opt,saved in zip(self.q,state['q'],self.qopt,state['qopt']):
                p.copy_(value); opt.load_state_dict(saved)
        for optimizer in [self.control_optimizer,*self.qopt]:
            if any(not torch.isfinite(v).all() for values in optimizer.state.values() for v in values.values() if isinstance(v,torch.Tensor)):
                raise ValueError('Non-finite checkpoint Adam state')
        self.q0=None if anchor is None else anchor.to(self.device)
        self.step,self.joint_step,self.coefficient_step=step,state['joint_step'],state['coefficient_step']
        self.sampler_phase=key
        self.controls.requires_grad_(key[1]=='joint')
        self.coefficient_order=state['coefficient_order'].numpy().copy()
        self.permutations=[p.numpy().copy() for p in state['permutations']]; self.cursors=state['cursors']
        self.rng.bit_generator.state=state['rng']; torch.set_rng_state(state['torch_rng'].cpu())
        if self.device.type=='cuda': torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        r=state['numpy_rng']; np.random.set_state((r['name'],r['keys'].numpy().astype(np.uint32),r['position'],r['gauss'],r['cached']))
        self.baked_fields=None


def refine_motion(*, prepared_dir, work_dir, output_dir, config=RefinementConfig(), resume=False, device="cuda",
                  flow_initialization=None):
    config.validate()
    root, manifest, scene, arrays, initial, operator = load_prepared(prepared_dir)
    if flow_initialization is not None:
        from .refinement_artifacts import load_flow_initialization
        manifest, initial = load_flow_initialization(flow_initialization, manifest, scene)
    work, destination = resolve_path(work_dir), resolve_path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    immutable = [resolve_path(v['directory']) for v in manifest['images']]
    immutable.extend(resolve_path(v['path']) for v in manifest['mode_sources'] if v.get('path'))
    if manifest.get('mode_bank'): immutable.append(resolve_path(manifest['mode_bank']))
    if flow_initialization is not None: immutable.append(resolve_path(flow_initialization))
    if any(p.is_relative_to(source) for p in (work, destination) for source in immutable):
        raise ValueError('Refinement work/output must be outside immutable source artifacts and images')
    for a, b in ((work, root), (destination, root), (destination, work), (work, destination),
                 (work, resolve_path(manifest["static_scene"])), (destination, resolve_path(manifest["static_scene"]))):
        if a.is_relative_to(b):
            raise ValueError("Refinement work/output must be separate from immutable inputs and each other")
    from . import fitting, rendering, refinement_artifacts, rgb, sequences
    from modal_gaussians.geometry import scene as scene_module, density
    from modal_gaussians.motion import reference_field
    from modal_gaussians.common import camera_rendering
    contract = dict(preparation_identity=manifest["preparation_identity"], settings=asdict(config),
        supervision_weights=sequence_weights(manifest["views"]),
        implementation=module_revision(sys.modules[__name__], fixed_field, reference_field, density, fitting,
                                       rendering, refinement_artifacts, rgb, sequences, scene_module, camera_rendering), device=str(device))
    if flow_initialization is not None:
        contract.update(initialization_source=manifest['initialization_source'],
                        supervised_sequences=manifest['views'], effective_warmup_steps=0)
    run_identity = identity(contract)
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "checkpoint.pt"
    with exclusive_work(work / ".lock"):
        if not resume and ((work / "run.json").exists() or checkpoint.exists()):
            raise FileExistsError("Refinement work exists; use --resume or a new work directory")
        if resume and not checkpoint.is_file():
            raise FileNotFoundError("No complete refinement checkpoint to resume")
        torch.manual_seed(config.seed); np.random.seed(config.seed)
        trainer = RefinementTrainer(scene, fixed_field.FixedField(arrays, operator, config.query_block_size),
                                    initial, manifest["views"], config, device)
        stopping = {}
        if resume:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["run_identity"] != run_identity:
                raise ValueError("Resume contract/config/implementation differs")
            trainer.load_state_dict(saved["trainer"])
            stopping = saved.get('early_stopping',{})
            if stopping:
                best_path=work/'best'/f"checkpoint_{stopping['best_step']:07d}.pt"
                if sha256(best_path)!=stopping['best_checkpoint_sha256']:
                    raise ValueError('Best early-stop checkpoint differs')
            log_path = work / "training.jsonl"
            if log_path.exists():
                # A killed writer may leave an incomplete last line after the checkpoint.
                rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines(keepends=True)
                        if line.endswith("\n")]
                log_path.write_text("".join(json.dumps(row)+"\n" for row in rows if row["step"] <= trainer.step), encoding="utf-8")
        cameras = {c.name:c.to(device) for c in cameras_from_scene_manifest(scene.manifest)}
        @lru_cache(maxsize=None)
        def camera_at(name, scale):
            return scaled_camera(cameras[name], scale)
        image_records = {v["label"]:v for v in manifest["images"]}
        @lru_cache(maxsize=None)
        def valid_at(v, scale):
            from .rgb import load_valid_mask
            view = manifest['views'][v]
            return load_valid_mask(image_records[view['label']], view['shape_hw'], scale, device)
        def render(v, frame, scale, means, rotations):
            name = manifest['views'][v]['frames'][frame]['camera_name']
            camera = camera_at(name, scale)
            return scene.render_deformed(camera, means, foreground_quaternions=rotations), camera

        @lru_cache(maxsize=8)
        def target(v, frame, scale):
            view = manifest["views"][v]
            record = image_records[view["label"]]
            entry = record["files"][frame]
            if Path(entry["name"]).name != entry["name"] or entry["name"] != f"{view['frame_names'][frame]}.png":
                raise ValueError("RGB image order differs from prepared recording")
            path = resolve_path(record["directory"], strict=True) / entry["name"]
            image, _ = load_rgb_frame(path, view["shape_hw"], entry["sha256"])
            return resize_rgb(image, scale).to(device)

        def save():
            _atomic_torch(checkpoint, dict(run_identity=run_identity, contract=contract, trainer=trainer.state_dict(),early_stopping=stopping))
        def check_progress():
            begin=time.perf_counter()
            loss,views=trainer.mean_rgb_loss(render,target,valid_at)
            improved=update_early_stopping(stopping,loss,trainer.step,config)
            stopping['history'][-1].update(joint_step=trainer.joint_step,per_sequence_rgb=views,
                                          seconds=time.perf_counter()-begin)
            if improved:
                best_path=work/'best'/f"checkpoint_{trainer.step:07d}.pt"
                best_path.parent.mkdir(exist_ok=True)
                _atomic_torch(best_path,dict(run_identity=run_identity,trainer=trainer.state_dict(),rgb_loss=loss))
                stopping['best_checkpoint_sha256']=sha256(best_path)
            save()
            atomic_json(work/'early_stopping.json',stopping)
            print(f"Full-frame RGB {trainer.step}: {loss:.8f}; best {stopping['best_loss']:.8f}; "
                  f"plateau {stopping['bad_checks']}/{config.early_stop_patience}",flush=True)
        atomic_json(work / "run.json", dict(**contract, run_identity=run_identity, status="running"))
        if not resume:
            save()
        try:
            if config.early_stop_patience and trainer.step>=trainer.warmup_steps and not stopping:
                check_progress()
            with (work / "training.jsonl").open("a", encoding="utf-8") as log:
                while trainer.step < trainer.total_steps and not stopping.get('stopped',False):
                    row = trainer.update(render, target, valid_at)
                    log.write(json.dumps(row, allow_nan=False)+"\n"); log.flush()
                    check_due = config.early_stop_patience and trainer.step>=trainer.warmup_steps and (
                        not stopping or (row['phase']=='joint' and not row['phase_boundary']
                            and trainer.joint_step%config.early_stop_joint_interval==0)
                        or (row['phase']=='coefficient' and row['phase_boundary']))
                    if check_due:
                        check_progress()
                    elif trainer.step % config.checkpoint_interval == 0 or row['phase_boundary']:
                        save()
                    if trainer.step % 100 == 0:
                        print(f"refinement {trainer.step}/{trainer.total_steps}: {row['gaussian_count']} Gaussians", flush=True)
            actual_steps,actual_joint_steps=trainer.step,trainer.joint_step
            if stopping:
                best_path=work/'best'/f"checkpoint_{stopping['best_step']:07d}.pt"
                best=torch.load(best_path,map_location='cpu',weights_only=True)
                if sha256(best_path)!=stopping['best_checkpoint_sha256'] or best['run_identity']!=run_identity:
                    raise ValueError('Best early-stop checkpoint identity differs')
                trainer.load_state_dict(best['trainer'])
                manifest=dict(manifest,training_selection=dict(criterion='full_training_sequence_weighted_mean_rgb',
                    best_step=trainer.step,best_rgb_loss=stopping['best_loss'],actual_steps=actual_steps,
                    actual_joint_steps=actual_joint_steps,early_stopped=stopping['stopped']))
            publish_refinement(destination, (root, manifest), scene, trainer.field, trainer.coordinates(), controls=trainer.controls(), run_identity=run_identity, settings=asdict(config), baked_fields=trainer.frozen_fields())
        except BaseException as error:
            atomic_json(work / "run.json", dict(**contract, run_identity=run_identity, status="failed",
                                                error=str(error), last_complete_checkpoint=str(checkpoint)))
            raise
        atomic_json(work / "run.json", dict(**contract, run_identity=run_identity, status="complete", output=str(destination),
            actual_steps=actual_steps,actual_joint_steps=actual_joint_steps,published_step=trainer.step,
            stop_reason='rgb_plateau' if stopping.get('stopped') else 'budget_complete',early_stopping=stopping))
    return destination
