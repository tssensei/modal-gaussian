"""Shared FFT convention and one selected modal-image bundle."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import math
import numpy as np
TRANSFORM_CONVENTION = {
    "method": "shared_rfft", "exponent": "exp(-2j*pi*frequency_hz*frame_index/fps_hz)",
    "detrend": "temporal_mean", "window": "hann_symmetric", "normalization": "none",
    "amplitude_clamp": "none", "mask_application": "none",
}
@dataclass(frozen=True)
class Complex2DModesArtifact:
    """Represent validated per-view dense complex mode memmaps."""

    path: Path
    manifest: dict[str, Any]
    view_modes: tuple[np.ndarray, ...]

def _validate_mode_records(manifest: Mapping[str, Any]) -> np.ndarray:
    """Validate the ordered greedy slots and return their frequencies."""

    modes = manifest.get("modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError("Complex mode manifest must contain ordered modes")
    frequencies = np.empty(len(modes), dtype=np.float64)
    candidates: list[int] = []
    for expected_slot, mode in enumerate(modes):
        if not isinstance(mode, dict) or mode.get("mode_slot") != expected_slot:
            raise ValueError("Complex mode slots must be contiguous and ordered")
        candidate = mode.get("candidate_index")
        if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 0:
            raise ValueError("Complex mode candidate indices must be non-negative integers")
        frequency = float(mode.get("frequency_hz", np.nan))
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("Complex mode frequencies must be finite and positive")
        candidates.append(candidate)
        frequencies[expected_slot] = frequency
    if len(set(candidates)) != len(candidates):
        raise ValueError("Complex mode candidate indices must be unique")
    return frequencies
