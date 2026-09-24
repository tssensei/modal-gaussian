from __future__ import annotations

import numpy as np
from modal_gaussians.common.progress import Progress
from modal_gaussians.flow.storage import DenseArray, spatial_blocks


def symmetric_hann(sample_count: int) -> np.ndarray:
    """Return the symmetric Hann window used by every temporal transform."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    indices = np.arange(sample_count, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(
        2.0 * np.pi * indices / max(1, sample_count - 1)
    )


def temporal_rfft_tiles(flow: DenseArray, *, fft_length: int | None = None,
                        block_width: int = 128):
    """Yield bounded spatial tiles; window original samples before zero padding."""
    if flow.ndim != 4 or flow.shape[-1] != 2 or flow.shape[0] < 3:
        raise ValueError("flow must be [T,H,W,2] with at least three samples")
    sample_count = flow.shape[0]
    length = sample_count if fft_length is None else fft_length
    if not isinstance(length, (int, np.integer)) or length < sample_count:
        raise ValueError("FFT length must be an integer at least as large as the sample count")
    if block_width <= 0:
        raise ValueError("block_width must be positive")
    window = symmetric_hann(sample_count).astype(np.float32)[:, None, None, None]
    for rows, columns in spatial_blocks(flow.shape, block_width):
        values = np.asarray(flow[:, rows, columns, :], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("flow contains NaN or Inf")
        values = values - values.mean(axis=0, keepdims=True, dtype=np.float32)
        values *= window
        yield rows, columns, np.fft.rfft(values, n=length, axis=0).astype(np.complex64)


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
    progress = Progress("flow dense rFFT", height * width, unit="pixels")
    completed = 0
    for rows, columns, transformed in temporal_rfft_tiles(flow, block_width=block_width):
        spectrum[:, rows, columns, :] = transformed
        completed += (rows.stop - rows.start) * (columns.stop - columns.start)
        progress.update(completed)

    return spectrum
