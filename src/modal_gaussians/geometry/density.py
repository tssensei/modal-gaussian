"""Gaussian row operations shared by static training and scene refinement."""
import math

import torch
from torch.nn import functional as F

from .scene import GAUSSIAN_FIELDS


def quaternion_to_rotation_matrix(quaternions):
    w, x, y, z = F.normalize(quaternions, dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


@torch.no_grad()
def split_positions(part, split):
    means = part.params["means"][split]
    offsets = torch.randn((len(means), 2, 3), device=means.device, dtype=means.dtype)
    offsets *= part.params["log_scales"][split].exp()[:, None]
    rotations = quaternion_to_rotation_matrix(part.params["quaternions"][split])
    return (means[:, None] + torch.einsum("nij,nkj->nki", rotations, offsets)).reshape(-1, 3)


@torch.no_grad()
def remap_parameters(part, optimizers, rows, newborn=None, overrides=None):
    """Gather parameter/Adam rows; new children inherit values but zero moments."""
    count = part.count
    if rows.ndim != 1 or len(rows) < 1 or bool(((rows < 0) | (rows >= count)).any()):
        raise ValueError("Invalid Gaussian row mapping")
    for field in GAUSSIAN_FIELDS:
        old = part.params[field]
        values = (overrides or {}).get(field, old.detach()[rows])
        new = part.replace_parameter(field, values)
        optimizer = optimizers[field]
        state = optimizer.state.pop(old, {})
        optimizer.param_groups[0]["params"] = [new]
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == count:
                value = value[rows].clone()
                if newborn is not None:
                    value[newborn] = 0
                state[key] = value
        if state:
            optimizer.state[new] = state


@torch.no_grad()
def densify_parameters(part, optimizers, split, duplicate, children=None):
    """Keep non-split parents, append clones, then two children per split."""
    ids = torch.arange(part.count, device=split.device)
    rows = torch.cat((ids[~split], ids[duplicate], ids[split].repeat_interleave(2)))
    newborn = torch.arange(len(rows), device=split.device) >= int((~split).sum())
    means = part.params["means"][rows].clone()
    scales = part.params["log_scales"][rows].clone()
    n = 2 * int(split.sum())
    if n:
        means[-n:] = split_positions(part, split) if children is None else children
        scales[-n:] -= math.log(1.6)
    remap_parameters(part, optimizers, rows, newborn, {"means": means, "log_scales": scales})
    return rows, newborn
