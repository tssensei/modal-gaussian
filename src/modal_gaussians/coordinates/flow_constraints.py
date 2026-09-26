"""Experimental fixed-design reference/adjacent flow constraints (no RGB fitting)."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import spsolve

from .direct import mode_pair_scales
from modal_gaussians.motion.common.geometry_ops import bilinear_sample, bilinear_valid


@dataclass(frozen=True)
class FlowConstraintConfig:
    adjacent_weight: float = 1.0
    ridge: float = 1e-4
    fb_absolute: float = 1.0
    fb_relative: float = .05

    def validate(self):
        values = (self.adjacent_weight, self.ridge, self.fb_absolute, self.fb_relative)
        if not all(np.isfinite(v) and v >= 0 for v in values) or self.ridge <= 0:
            raise ValueError('Flow weights/thresholds must be finite, nonnegative; ridge positive')


def valid_positions(points, mask):
    """Require a full bilinear footprint in observed image support."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError('Image support must be a boolean HW array')
    valid = bilinear_valid(points, *mask.shape)
    valid[valid] &= bilinear_sample(mask.astype(np.float32), np.asarray(points)[valid]) >= 1-1e-6
    return valid


def sample_adjacent_flow(pixels, reference_previous, forward, backward,
                         source_valid, target_valid, config=FlowConstraintConfig()):
    """Pull pair flow back to reference material samples, checking both endpoints."""
    config.validate()
    pixels, previous = np.asarray(pixels), np.asarray(reference_previous)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or previous.shape != pixels.shape:
        raise ValueError('Reference samples must be Px2')
    if (forward.shape != (*source_valid.shape, 2) or backward.shape != forward.shape
            or target_valid.shape != source_valid.shape
            or not np.isfinite(forward).all() or not np.isfinite(backward).all()):
        raise ValueError('Invalid adjacent flow shape or values')
    points = pixels + previous
    valid = valid_positions(points, source_valid)
    flow = np.zeros_like(points, dtype=np.float64)
    # geometry_ops.bilinear_sample is scalar-image sampling, one component at a time.
    for c in range(2):
        flow[valid, c] = bilinear_sample(forward[..., c], points[valid])
    ends = points + flow
    valid &= valid_positions(ends, target_valid)
    reverse = np.zeros_like(flow)
    for c in range(2):
        reverse[valid, c] = bilinear_sample(backward[..., c], ends[valid])
    valid &= valid_positions(ends + reverse, source_valid)
    error = np.linalg.norm(flow + reverse, axis=1)
    limit = config.fb_absolute + config.fb_relative * (
        np.linalg.norm(flow, axis=1) + np.linalg.norm(reverse, axis=1))
    valid &= error <= limit
    flow[~valid] = 0  # Explicit mask, never interpreted as a zero-motion observation.
    return flow, valid


def flow_normal_equations(design, reference, adjacent, reference_valid, adjacent_valid):
    """Normalize once; return small per-frame/pair Gram/RHS blocks in float64."""
    # ponytail: fixed canonical design; relinearize at deformed poses only if the
    # experiment exposes large-motion/visibility errors that this model cannot fit.
    d = np.asarray(design, dtype=np.float64)
    if d.ndim != 3 or d.shape[1] != 2 or d.shape[2] < 2 or d.shape[2] % 2 or not np.isfinite(d).all():
        raise ValueError('Design must be finite [P,2,2K]')
    t, p, c = len(reference), d.shape[0], d.shape[2]
    if (t < 2 or p < 1 or reference.shape != (t,p,2) or adjacent.shape != (t-1,p,2)
            or reference_valid.shape != (t,p) or adjacent_valid.shape != (t-1,p)
            or reference_valid.dtype != bool or adjacent_valid.dtype != bool):
        raise ValueError('Flow observations/masks do not match design and sequence')
    scales = mode_pair_scales(np.square(d).sum(axis=(0,1)), p)
    b = d / np.repeat(scales, 2)
    result = {'scales': scales}
    for name, values, masks in (('reference', reference, reference_valid), ('adjacent', adjacent, adjacent_valid)):
        grams, rhs, counts = np.zeros((len(values),c,c)), np.zeros((len(values),c)), masks.sum(axis=1)
        for i, (value, valid) in enumerate(zip(values, masks)):
            if not counts[i]:
                if name == 'reference':
                    raise ValueError(f'No valid reference observations at frame {i}')
                continue
            a = b[valid].reshape(-1,c)
            y = np.asarray(value[valid], np.float64).reshape(-1)
            if not np.isfinite(y).all():
                raise ValueError(f'Non-finite {name} observations at {i}')
            grams[i], rhs[i] = a.T @ a / len(y), a.T @ y / len(y)
        result[name+'_gram'], result[name+'_rhs'], result[name+'_counts'] = grams, rhs, counts
    return result


def solve_flow_system(system, config=FlowConstraintConfig()):
    """Sparse block-tridiagonal solve. Returns reference-relative complex64 q."""
    config.validate()
    diagonal = system['reference_gram'].copy()
    rhs = system['reference_rhs'].copy()
    t, c, _ = diagonal.shape
    edge = config.adjacent_weight * system['adjacent_gram']
    edge_rhs = config.adjacent_weight * system['adjacent_rhs']
    diagonal += config.ridge * np.eye(c)[None]
    diagonal[:-1] += edge
    diagonal[1:] += edge
    rhs[:-1] -= edge_rhs
    rhs[1:] += edge_rhs
    blocks, indices, indptr = [], [], [0]
    for i in range(t):
        if i:
            blocks.append(-edge[i-1]); indices.append(i-1)
        blocks.append(diagonal[i]); indices.append(i)
        if i+1 < t:
            blocks.append(-edge[i]); indices.append(i+1)
        indptr.append(len(indices))
    matrix = bsr_matrix((np.asarray(blocks), np.asarray(indices), np.asarray(indptr)), shape=(t*c,t*c)).tocsc()
    z = spsolve(matrix, rhs.reshape(-1)).reshape(t,c)
    if not np.isfinite(z).all() or not np.allclose(matrix @ z.ravel(), rhs.ravel(), rtol=1e-7, atol=1e-9):
        raise RuntimeError('Flow system failed its finite/residual check')
    raw = z / np.repeat(system['scales'], 2)
    return (raw[:,0::2] + 1j*raw[:,1::2]).astype(np.complex64)


def flow_residuals(design, relative, reference, adjacent, reference_valid, adjacent_valid):
    packed = np.stack((relative.real, relative.imag), axis=-1).reshape(len(relative), -1)
    records = []
    previous = None
    for t, q in enumerate(packed):
        prediction = np.asarray(design, np.float64) @ q
        error = prediction - reference[t]
        record = {'frame':t, 'reference_rmse_pixels':float(np.sqrt(np.mean(error[reference_valid[t]]**2)))}
        if t:
            valid = adjacent_valid[t-1]
            delta = prediction - previous - adjacent[t-1]
            record['adjacent_rmse_pixels'] = float(np.sqrt(np.mean(delta[valid]**2))) if valid.any() else None
        records.append(record)
        previous = prediction
    return records
