from __future__ import annotations

import numpy as np


def symmetric_hann(sample_count: int) -> np.ndarray:
    """Return the symmetric Hann window used by every temporal transform."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    indices = np.arange(sample_count, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(
        2.0 * np.pi * indices / max(1, sample_count - 1)
    )


def exact_dft_basis(
    sample_count: int,
    fps_hz: float,
    frequencies_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the shared complex64 exact-DFT basis and float32 Hann window."""

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if sample_count < 3:
        raise ValueError("At least three temporal samples are required")
    if not np.isfinite(fps_hz) or fps_hz <= 0.0:
        raise ValueError("fps_hz must be finite and positive")
    if (
        frequencies.ndim != 1
        or frequencies.size == 0
        or not np.isfinite(frequencies).all()
        or np.any(frequencies < 0.0)
        or np.any(frequencies > 0.5 * float(fps_hz) + 1e-12)
    ):
        raise ValueError("Exact-DFT frequencies must be finite and within Nyquist")
    times = np.arange(sample_count, dtype=np.float64) / float(fps_hz)
    basis = np.exp(
        (-2j * np.pi) * frequencies[:, None] * times[None, :]
    ).astype(np.complex64, copy=False)
    window = symmetric_hann(sample_count).astype(np.float32)
    return basis, window


def temporal_rfft(
    flow: np.ndarray,
    *,
    block_width: int = 128,
) -> np.ndarray:
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError("flow must be [T,H,W,2]")
    if flow.shape[0] < 3:
        raise ValueError("At least three temporal samples are required")
    if not np.isfinite(flow).all():
        raise ValueError("flow contains NaN or Inf")
    if block_width <= 0:
        raise ValueError("block_width must be positive")

    sample_count, height, width, _ = flow.shape
    frequency_count = sample_count // 2 + 1
    spectrum = np.empty(
        (frequency_count, height, width, 2), dtype=np.complex64
    )
    window = symmetric_hann(sample_count).astype(np.float32)[:, None, None, None]
    width_block = min(int(block_width), width)
    for lower in range(0, width, width_block):
        upper = min(width, lower + width_block)
        values = np.asarray(flow[:, :, lower:upper, :], dtype=np.float32)
        values = values - values.mean(axis=0, keepdims=True, dtype=np.float32)
        values = values * window
        spectrum[:, :, lower:upper, :] = np.fft.rfft(
            values, axis=0
        ).astype(np.complex64, copy=False)

    return spectrum
