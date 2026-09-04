from __future__ import annotations

import numpy as np
from modal_gaussians.progress import Progress
from modal_gaussians.flow.storage import DenseArray, spatial_blocks


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
    flow: DenseArray,
    *,
    block_width: int = 128,
    output: DenseArray | None = None,
) -> DenseArray:
    """Transform spatial tiles; pass a disk-backed output to avoid a dense RAM result."""
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError("flow must be [T,H,W,2]")
    if flow.shape[0] < 3:
        raise ValueError("At least three temporal samples are required")
    if block_width <= 0:
        raise ValueError("block_width must be positive")

    sample_count, height, width, _ = flow.shape
    frequency_count = sample_count // 2 + 1
    shape = (frequency_count, height, width, 2)
    spectrum = np.empty(shape, dtype=np.complex64) if output is None else output
    if spectrum.shape != shape or spectrum.dtype != np.dtype(np.complex64):
        raise ValueError("rFFT output must be complex64 [F,H,W,2]")
    window = symmetric_hann(sample_count).astype(np.float32)[:, None, None, None]
    progress = Progress("flow dense rFFT", height * width, unit="pixels")
    completed = 0
    for rows, columns in spatial_blocks(flow.shape, block_width):
        values = np.asarray(flow[:, rows, columns, :], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("flow contains NaN or Inf")
        values = values - values.mean(axis=0, keepdims=True, dtype=np.float32)
        values = values * window
        spectrum[:, rows, columns, :] = np.fft.rfft(
            values, axis=0
        ).astype(np.complex64, copy=False)
        completed += (rows.stop - rows.start) * (columns.stop - columns.start)
        progress.update(completed)

    return spectrum
