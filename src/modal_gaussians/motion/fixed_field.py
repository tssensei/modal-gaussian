"""Fixed sparse control interpolation and RGB-learned modal corrections."""
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from .reference_field import ReferenceField, _device_rows, _segment_sum


@torch.no_grad()
def prepare_operator(arrays, block_size=4096):
    """Cache canonical control stencils, retaining the original donor sum order."""
    field = ReferenceField(arrays, block_size)
    n, c = len(arrays['points']), len(arrays['controls'])
    ptrs, indices, weights, levers = [], [], [], []
    query_ptrs, query_roots, query_weights = [], [], []
    offset = query_offset = 0
    for k in range(field.mode_count):
        row_parts, control_parts, weight_parts, lever_parts = [], [], [], []
        for lo in range(0, n, block_size):
            roots = np.arange(lo, min(lo+block_size, n))
            counts, control, weight, lever, _ = field._own(
                torch.from_numpy(arrays['points'][roots]), roots, k, stencil=True)
            keep = weight.numpy() > 0
            row_parts.append(np.repeat(roots, counts.numpy())[keep])
            control_parts.append(control.numpy()[keep])
            weight_parts.append(weight.numpy()[keep])
            lever_parts.append(lever.numpy()[keep])
        rows, cols, w, lever = map(np.concatenate, (row_parts, control_parts, weight_parts, lever_parts))
        own = np.flatnonzero(arrays['own'][k])
        recipient, slot = np.nonzero((arrays['donor_weights'][k] > 0) & ~arrays['own'][k,:,None])
        outputs = np.r_[own,recipient]
        order = np.argsort(outputs,kind='stable')
        query_ptrs.append(np.r_[0,np.cumsum(np.bincount(outputs,minlength=n))]+query_offset)
        query_roots.append(np.r_[own,arrays['donor_ids'][k,recipient,slot]][order])
        query_weights.append(np.r_[np.ones(len(own)),arrays['donor_weights'][k,recipient,slot]][order].astype(np.float32))
        query_offset += len(outputs)
        ptrs.append(np.r_[0,np.cumsum(np.bincount(rows,minlength=n))]+offset)
        indices.append(cols.astype(np.int64)); weights.append(w.astype(np.float32))
        levers.append(lever.astype(np.float32))
        offset += len(cols)
    return dict(version=np.asarray(2,np.int64),ptr=np.stack(ptrs),control=np.concatenate(indices),
                weight=np.concatenate(weights),lever=np.concatenate(levers),query_ptr=np.stack(query_ptrs),
                query_root=np.concatenate(query_roots),query_weight=np.concatenate(query_weights))


class FixedField:
    """Bounded control-to-Gaussian composition; no path queries during training."""
    def __init__(self, arrays, operator, block_size=4096):
        self.a, self.operator, self.block_size = arrays, operator, block_size
        self.mode_count = len(arrays['displacement'])
        n, c = len(arrays['points']), len(arrays['controls'])
        if (not n or not c or arrays['points'].shape != (n,3) or not np.isfinite(arrays['points']).all()
                or arrays['controls'].dtype != np.int64 or np.any(arrays['controls']<0) or np.any(arrays['controls']>=n)
                or len(np.unique(arrays['controls'])) != c):
            raise ValueError('Invalid fixed control/point domains')
        for name in ('displacement','angular'):
            if name in arrays and (arrays[name].shape != (self.mode_count,c,3)
                    or arrays[name].dtype.kind != 'c' or not np.isfinite(arrays[name]).all()):
                raise ValueError('Invalid fixed control fields')
        if 'control_valid' in arrays and (arrays['control_valid'].shape != (self.mode_count,c) or arrays['control_valid'].dtype != bool):
            raise ValueError('Invalid control support')
        p, ids, w, lever = (operator[key] for key in ('ptr','control','weight','lever'))
        if (set(operator) != {'version','ptr','control','weight','lever','query_ptr','query_root','query_weight'}
                or np.asarray(operator.get('version')).shape != () or operator.get('version') != 2
                or type(block_size) is not int or block_size < 1 or p.dtype != np.int64
                or p.shape != (self.mode_count,n+1) or ids.dtype != np.int64
                or p[0,0] != 0 or p[-1,-1] != len(ids) or np.any(np.diff(p,axis=1)<0)
                or not np.array_equal(p[:-1,-1],p[1:,0]) or np.any(ids<0) or np.any(ids>=c)
                or ids.ndim != 1 or w.dtype != np.float32 or lever.dtype != np.float32
                or w.shape != (len(ids),) or lever.shape != (len(ids),3)
                or not np.isfinite(w).all() or not np.isfinite(lever).all() or np.any(w<=0)):
            raise ValueError('Invalid fixed interpolation operator')
        qp,qr,qw = (operator[key] for key in ('query_ptr','query_root','query_weight'))
        if (qp.dtype != np.int64 or qp.shape != p.shape or qp[0,0] != 0 or qp[-1,-1] != len(qr)
                or np.any(np.diff(qp,axis=1)<0) or not np.array_equal(qp[:-1,-1],qp[1:,0])
                or qr.dtype != np.int64 or qr.ndim != 1 or np.any(qr<0) or np.any(qr>=n)
                or qw.dtype != np.float32 or qw.shape != qr.shape or not np.isfinite(qw).all() or np.any(qw<=0)):
            raise ValueError('Invalid fixed donor interpolation operator')
        self.tables = {}

    def _tables(self, device):
        if device not in self.tables:
            self.tables[device] = {k:torch.as_tensor(v,device=device) for k,v in self.operator.items()}
        return self.tables[device]

    def block(self, displacement, angular, k, lo, hi):
        t = self._tables(displacement.device)
        queries, _ = _device_rows(t['query_ptr'][k], torch.arange(lo,hi,device=displacement.device))
        roots = t['query_root'][queries]
        ids, _ = _device_rows(t['ptr'][k],roots)
        counts = t['ptr'][k,roots+1]-t['ptr'][k,roots]
        controls = t['control'][ids]
        weight = t['weight'][ids,None].to(displacement.real.dtype)
        lever = t['lever'][ids].to(displacement.dtype)
        omega = angular[k,controls]
        # Coalescing donor/control weights reassociates float32 sums and can exceed
        # the field-equivalence tolerance when control rotations are large.
        phi = _segment_sum(weight*(displacement[k,controls]+torch.linalg.cross(omega,lever)),counts)
        angle = _segment_sum(weight*omega,counts)
        donor_weight = t['query_weight'][queries,None].to(displacement.real.dtype)
        donor_counts = t['query_ptr'][k,lo+1:hi+1]-t['query_ptr'][k,lo:hi]
        return (_segment_sum(donor_weight*phi,donor_counts),_segment_sum(donor_weight*angle,donor_counts))

    def deform(self, q, displacement, angular):
        if q.ndim not in (1,2) or q.shape[-1] != self.mode_count or not q.is_complex():
            raise ValueError('Expected complex q[K] or q[S,K]')
        if displacement.shape != self.a['displacement'].shape or angular.shape != displacement.shape:
            raise ValueError('Control motion dimensions differ')
        single = q.ndim == 1
        q = q[None] if single else q
        points = torch.as_tensor(self.a['points'],device=q.device,dtype=q.real.dtype)
        outputs = []
        for lo in range(0,len(points),self.block_size):
            hi = min(lo+self.block_size,len(points))
            dx = q.real.new_zeros((len(q),hi-lo,3)); angle = torch.zeros_like(dx)
            for start in range(0,self.mode_count,4):
                stop = min(start+4,self.mode_count)
                def query(d,o,coeff,lo=lo,hi=hi,start=start,stop=stop):
                    delta = coeff.real.new_zeros((len(coeff),hi-lo,3)); rot = torch.zeros_like(delta)
                    for k in range(start,stop):
                        phi, omega = self.block(d,o,k,lo,hi)
                        delta = delta+(coeff[:,k,None,None]*phi).real
                        rot = rot+(coeff[:,k,None,None]*omega).real
                    return delta,rot
                delta,rot = checkpoint(query,displacement,angular,q,use_reentrant=False) if torch.is_grad_enabled() else query(displacement,angular,q)
                dx,angle = dx+delta,angle+rot
            outputs.append((points[lo:hi]+dx,angle))
        result = tuple(torch.cat([part[i] for part in outputs],dim=1) for i in range(2))
        return tuple(v[0] for v in result) if single else result

    @torch.no_grad()
    def bake(self, displacement, angular):
        result = []
        for k in range(self.mode_count):
            blocks = [self.block(displacement,angular,k,lo,min(lo+self.block_size,len(self.a['points'])))
                      for lo in range(0,len(self.a['points']),self.block_size)]
            result.append(tuple(torch.cat([b[i] for b in blocks]) for i in range(2)))
        return tuple(torch.stack([r[i] for r in result]) for i in range(2))


class ControlCorrection(torch.nn.Module):
    """Normalize valid corrections and remove each mode's complex scale/phase gauge."""
    def __init__(self, arrays, device):
        super().__init__()
        self.length = float(arrays['radius'])
        base = np.concatenate((arrays['displacement'], self.length*arrays['angular']),axis=-1)
        valid = np.asarray(arrays['control_valid'],bool)
        if (valid.shape != base.shape[:2] or not np.isfinite(base).all()
                or not np.isfinite(self.length) or self.length<=0 or np.any(valid.sum(1)==0)):
            raise ValueError('Invalid control prior')
        rms = np.sqrt((np.abs(base)**2*valid[:,:,None]).sum((1,2))/(6*valid.sum(1)))
        if not np.isfinite(rms).all() or np.any(rms<=0):
            raise ValueError('Zero or non-finite original mode')
        self.register_buffer('base',torch.as_tensor(base,device=device))
        self.register_buffer('valid',torch.as_tensor(valid[:,:,None],device=device))
        self.register_buffer('scale',torch.as_tensor(rms[:,None,None],device=device,dtype=self.base.real.dtype))
        self.delta = torch.nn.Parameter(torch.zeros((*base.shape,2),device=device,dtype=self.base.real.dtype))

    def correction(self):
        b = self.base/self.scale*self.valid
        delta = torch.view_as_complex(self.delta)*self.valid
        projection = (b.conj()*delta).sum((1,2),keepdim=True)/(b.abs().square().sum((1,2),keepdim=True))
        return (delta-b*projection)*self.valid

    def forward(self):
        value = self.base+self.scale*self.correction()
        return value[...,:3],value[...,3:]/self.length

    def penalty(self):
        return (self.correction().abs().square().sum((1,2))/(6*self.valid.sum((1,2)))).mean()


def temporal_graph_loss(current, previous, rotation, previous_rotation, edges, lengths, node_weights):
    """Bidirectional, per-node-averaged local rigidity and sign-invariant rotation."""
    if not len(edges):
        zero = current.sum()*0
        return zero,zero
    from modal_gaussians.geometry.density import quaternion_to_rotation_matrix as quat_to_rotmat
    # Rotation matrices remove the quaternion sign ambiguity.
    R = quat_to_rotmat(rotation) @ quat_to_rotmat(previous_rotation).transpose(-1,-2)
    i,j = edges.T
    now,old = current[j]-current[i],previous[j]-previous[i]
    ri = torch.einsum('eji,ej->ei',R[i],now)-old
    rj = torch.einsum('eji,ej->ei',R[j],-now)+old
    rigid = ((ri.square().sum(-1)*node_weights[i]+rj.square().sum(-1)*node_weights[j])/lengths.square()).sum()
    rotation_loss = ((R[i]-R[j]).square().sum((-1,-2))/3*(node_weights[i]+node_weights[j])).sum()
    return rigid,rotation_loss
