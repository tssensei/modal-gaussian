"""Frozen reference motion queried by a changing set of rendering Gaussians."""
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .reference_field import ReferenceField


class MaterialField:
    def __init__(self, reference, displacement, angular, block_size=4096):
        self.arrays = dict(reference, displacement=np.asarray(displacement), angular=np.asarray(angular))
        self.query = ReferenceField(self.arrays, block_size)
        self.block_size = block_size
        self.mode_count = len(displacement)
        self.required = np.asarray(reference['support_class']) != 0
        self.roots = None
        self.layouts = {}
        self.margin = None
        if (displacement.shape != reference['displacement'].shape or angular.shape != displacement.shape
                or not np.isfinite(displacement).all() or not np.isfinite(angular).all()):
            raise ValueError('Invalid frozen material control fields')

    def _validate(self, positions, roots):
        roots = np.asarray(roots)
        if (roots.dtype != np.int64 or roots.shape != (len(positions),) or positions.shape != (len(roots),3)
                or np.any(roots < 0) or np.any(roots >= len(self.arrays['points']))):
            raise ValueError('Invalid rendering-to-reference root mapping')
        if self.roots is None or not np.array_equal(self.roots,roots):
            self.roots=roots.copy();self.layouts.clear()
        return roots

    def _layout(self, roots, lo, hi, start, stop, device):
        key=(lo,hi,start,stop,device)
        if key not in self.layouts:
            self.layouts[key]=self.query.prepare_query(np.tile(roots[lo:hi],stop-start),
                np.arange(start,stop).repeat(hi-lo),device)
        return self.layouts[key]

    @torch.no_grad()
    def supported(self, positions, roots):
        roots = self._validate(positions, roots)
        result = torch.ones(len(roots), dtype=torch.bool, device=positions.device)
        if self.margin is None:
            self.margin=self._support_margin()
        origin=torch.as_tensor(self.arrays['points'][roots],device=positions.device)
        margin=torch.as_tensor(self.margin[roots],device=positions.device)
        # A path length changes by at most ||x-x0||. Points inside this conservative
        # ball retain support; only the remaining points need the full path query.
        unsafe=((positions.double()-origin.double()).norm(dim=-1)>=margin).nonzero().flatten()
        if not len(unsafe):return result
        remaining=roots[unsafe.cpu().numpy()]
        positions=positions[unsafe]
        roots=remaining
        checked=torch.ones(len(roots),dtype=torch.bool,device=positions.device)
        for lo in range(0,len(roots),self.block_size):
            hi = min(lo+self.block_size,len(roots))
            for start in range(0,self.mode_count,20):
                stop=min(start+20,self.mode_count);count=stop-start
                required = torch.as_tensor(self.required[start:stop,roots[lo:hi]], device=positions.device)
                good = self.query._mode(positions[lo:hi].repeat(count,1),np.tile(roots[lo:hi],count),
                    np.arange(start,stop).repeat(hi-lo),support_only=True).reshape(count,hi-lo)
                checked[lo:hi] &= (good | ~required).all(0)
        result[unsafe]=checked
        return result

    def _support_margin(self):
        a=self.arrays;cp,qp=a['candidate_ptr'],a['query_ptr'];n=len(a['points'])
        costs=np.empty(len(a['candidate_control']),np.float64)
        for lo in range(0,n,self.block_size):
            hi=min(lo+self.block_size,n);ca,cb=cp[lo],cp[hi];qa,qb=qp[ca],qp[cb]
            nodes=np.repeat(np.repeat(np.arange(lo,hi),np.diff(cp[lo:hi+1])),np.diff(qp[ca:cb+1]))
            paths=a['query_path'][qa:qb][a['query_node'][qa:qb]==nodes]
            if len(paths)!=cb-ca:raise ValueError('Missing canonical reference portal')
            costs[ca:cb]=a['geometric'][paths]
        candidate_root=np.repeat(np.arange(n),np.diff(cp))
        margin=np.full(n,np.inf)
        for k in range(self.mode_count):
            distance=np.full(n,np.inf)
            valid=a['control_valid'][k,a['candidate_control']]
            np.minimum.at(distance,candidate_root,np.where(valid,costs,np.inf))
            own_margin=np.maximum(float(a['radius'])-distance,0.)
            current=np.where(a['own'][k],own_margin,np.inf)
            recipient,slot=np.nonzero((a['donor_weights'][k]>0)&~a['own'][k,:,None])
            np.minimum.at(current,recipient,own_margin[a['donor_ids'][k,recipient,slot]])
            margin=np.minimum(margin,np.where(self.required[k],current,np.inf))
        slack=32*np.finfo(np.float32).eps*(float(np.abs(a['points']).max())+float(a['radius'])+1.)
        return np.maximum(margin-slack,0.)

    def deform(self, positions, roots, q):
        roots = self._validate(positions, roots)
        if q.ndim != 2 or q.shape[1] != self.mode_count or not q.is_complex() or q.requires_grad:
            raise ValueError('Material motion requires frozen complex q[B,K]')
        means, angles = [], []
        for lo in range(0,len(roots),self.block_size):
            hi = min(lo+self.block_size,len(roots))
            ids = roots[lo:hi].copy()
            x = positions[lo:hi]
            delta = x.new_zeros((len(q),len(x),3)); angle = torch.zeros_like(delta)
            for start in range(0,self.mode_count,20):
                stop = min(start+20,self.mode_count)
                layout=self._layout(roots,lo,hi,start,stop,x.device)
                def compose(value,ids=ids,start=start,stop=stop,layout=layout):
                    dx = value.new_zeros((len(q),len(value),3)); omega = torch.zeros_like(dx)
                    count = stop-start
                    phi, rotation, _ = self.query._mode(value.repeat(count,1),None,None,prepared=layout)
                    phi, rotation = phi.reshape(count,len(ids),3), rotation.reshape(count,len(ids),3)
                    for j,k in enumerate(range(start,stop)):
                        dx = dx+(q[:,k,None,None]*phi[j]).real
                        omega = omega+(q[:,k,None,None]*rotation[j]).real
                    return dx,omega
                d,o = checkpoint(compose,x,use_reentrant=False) if x.requires_grad else compose(x)
                delta = delta+d; angle = angle+o
            means.append(x[None]+delta); angles.append(angle)
        return torch.cat(means,1),torch.cat(angles,1)

    @torch.no_grad()
    def bake(self, positions, roots):
        roots = self._validate(positions, roots)
        for k in range(self.mode_count):
            phi, rotation, good = self.query.mode(positions,roots,k)
            required = torch.as_tensor(self.required[k,roots],device=positions.device)
            if not good[required].all():
                raise ValueError('Published Gaussian left its material support')
            yield phi, rotation
