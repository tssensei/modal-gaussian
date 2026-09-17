"""Windowed phase evidence for candidate geometry edges, using observed motion."""
from __future__ import annotations

import numpy as np

from modal_gaussians.flow.spectrum import exact_dft_basis


def window_responses(
    values: np.ndarray,
    fps: float,
    frequency_hz: float,
    window_seconds: float = 10.0,
    hop_seconds: float = 2.0,
) -> np.ndarray:
    """Return complex64 [N,W,2] exact-frequency responses of [T,N,2] motion.

    Each complete window is mean-centered and multiplied by a symmetric Hann
    window before the negative-sign DFT. Window-local time starts at zero; its
    common phase offset cancels in same-window cross responses. No amplitude
    normalization is applied. A sequence shorter than one window is rejected.
    Non-finite samples make that point/component/window response unavailable.
    """
    if np.iscomplexobj(values):
        raise ValueError("Motion samples must be real [T,N,2]")
    samples = np.asarray(values, dtype=np.float32)
    if samples.ndim != 3 or samples.shape[2] != 2:
        raise ValueError("Motion samples must have shape [T,N,2]")
    if not all(np.isfinite(x) and x > 0 for x in (fps, window_seconds, hop_seconds)):
        raise ValueError("FPS, window duration and hop duration must be finite and positive")
    length, hop = round(window_seconds * fps), round(hop_seconds * fps)
    if hop < 1:
        raise ValueError("Hop duration must cover at least one sample")
    basis, hann = exact_dft_basis(length, fps, np.asarray([frequency_hz]))
    if samples.shape[0] < length:
        raise ValueError(f"Motion sequence has {samples.shape[0]} samples; a full window requires {length}")
    weighted_basis = basis[0] * hann
    starts = range(0, samples.shape[0] - length + 1, hop)
    result = np.empty((samples.shape[1], len(starts), 2), dtype=np.complex64)
    # Bound the temporary centered block instead of materializing all windows.
    point_block = max(1, 1_000_000 // (2 * length))
    for window, start in enumerate(starts):
        for first in range(0, samples.shape[1], point_block):
            last = first + point_block
            block = samples[start:start + length, first:last]
            with np.errstate(invalid="ignore", over="ignore"):
                centered = block - block.mean(axis=0, keepdims=True, dtype=np.float32)
                result[first:last, window] = np.einsum("t,tnd->nd", weighted_basis, centered)
    return result


def edge_phase_evidence(
    responses: np.ndarray,
    edges: np.ndarray,
    *,
    amplitude_floor_fraction: float = 0.05,
    minimum_windows: int = 3,
    minimum_effective_windows: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inconsistency_sum, evidence_weight, valid_count for each edge.

    Each point uses one fixed x/y component, chosen by its mean finite amplitude
    across windows. The amplitude floor is a fraction of the median positive,
    finite selected-component amplitude across all points and windows in this
    view. It rejects weak responses, without using phase variability as noise.

    For jointly valid windows, z=C_i*conj(C_j), R=abs(sum(z))/sum(abs(z)), and
    effective_count=sum(abs(z))**2/sum(abs(z)**2). A fixed phase difference,
    including pi, is coherent. Accepted edges return eta*(1-R), eta, count,
    where eta=count/W; otherwise both evidence values are zero. The defaults
    are heuristic amplitude/sample-support gates, not calibrated confidence;
    effective_count measures weight concentration, not window independence.
    """
    values, pairs = np.asarray(responses), np.asarray(edges)
    if values.ndim != 3 or values.shape[2] != 2 or not np.iscomplexobj(values):
        raise ValueError("Responses must be complex [N,W,2]")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("Edges must be integer [E,2]")
    if np.any(pairs < 0) or np.any(pairs >= len(values)):
        raise ValueError("Edge endpoints must index the response points")
    if not np.isfinite(amplitude_floor_fraction) or amplitude_floor_fraction < 0:
        raise ValueError("Amplitude floor fraction must be finite and nonnegative")
    if int(minimum_windows) != minimum_windows or minimum_windows < 1:
        raise ValueError("Minimum windows must be a positive integer")
    if not np.isfinite(minimum_effective_windows) or minimum_effective_windows <= 0:
        raise ValueError("Minimum effective windows must be finite and positive")
    inconsistency = np.zeros(len(pairs), dtype=np.float32)
    evidence = np.zeros(len(pairs), dtype=np.float32)
    counts = np.zeros(len(pairs), dtype=np.int32)
    windows = values.shape[1]
    if not len(pairs) or not windows:
        return inconsistency, evidence, counts

    amplitudes = np.abs(values)
    finite = np.isfinite(amplitudes)
    mean_amplitude = np.where(finite, amplitudes, 0).sum(axis=1, dtype=np.float64)
    mean_amplitude /= np.maximum(finite.sum(axis=1), 1)
    component = mean_amplitude.argmax(axis=1)
    selected = values[np.arange(len(values)), :, component]
    amplitudes = amplitudes[np.arange(len(values)), :, component]
    positive = np.isfinite(amplitudes) & (amplitudes > 0)
    if not np.any(positive):
        return inconsistency, evidence, counts
    floor = amplitude_floor_fraction * float(np.median(amplitudes[positive]))
    valid = positive & (amplitudes >= floor)
    edge_block = max(1, min(65_536, 1_000_000 // windows))
    for first in range(0, len(pairs), edge_block):
        last = first + edge_block
        left, right = pairs[first:last].T
        joint = valid[left] & valid[right]
        counts[first:last] = joint.sum(axis=1)
        # complex128 keeps products and effective-count sums safe for float32 data.
        a = np.where(joint, selected[left], 0).astype(np.complex128)
        b = np.where(joint, selected[right], 0).astype(np.complex128)
        cross = a * b.conj()
        weight = np.abs(cross)
        scale = weight.max(axis=1, keepdims=True)
        cross /= np.where(scale > 0, scale, 1)
        weight /= np.where(scale > 0, scale, 1)
        total = weight.sum(axis=1)
        square_sum = np.square(weight).sum(axis=1)
        effective = np.square(total) / np.where(square_sum > 0, square_sum, 1)
        accepted = ((counts[first:last] >= minimum_windows)
                    & (effective + 1e-12 >= minimum_effective_windows) & (total > 0))
        coherence = np.clip(np.abs(cross.sum(axis=1)) / np.where(total > 0, total, 1), 0, 1)
        eta = np.where(accepted, counts[first:last] / windows, 0)
        evidence[first:last] = eta
        inconsistency[first:last] = eta * (1 - coherence)
    return inconsistency, evidence, counts
