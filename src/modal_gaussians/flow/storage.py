"""Lossless, bounded-I/O array storage shared by flow and its consumers."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr
from modal_gaussians.iteration_cache import read_bytes
from zarr.codecs import ZstdCodec
from zarr.storage import LocalStore


DenseArray = np.ndarray | zarr.Array
BLOCK_BYTES = 32 * 1024 * 1024


class _WindowsRetryLocalStore(LocalStore):
    """Retry transient Windows locks while retaining Zarr's atomic writes.

    A reader or scanner can briefly deny replacement of an existing shard.
    Retry the same buffer; persistent permissions and unrelated errors still
    fail. No data, chunk layout, compression or numerical settings change.
    """

    async def _set(self, key, value, exclusive=False):
        for attempt in range(7):
            try:
                return await super()._set(key, value, exclusive=exclusive)
            except OSError as error:
                if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 6:
                    raise
                await asyncio.sleep(min(0.05 * 2**attempt, 0.8))


def create_array(path: Path, shape: tuple[int, ...], dtype: Any) -> zarr.Array:
    """Create a sharded Zarr v3 array; even zero chunks are explicitly written."""

    chunks = tuple(min(size, limit) for size, limit in zip(shape, (8, 32, 128, 2)))
    shards = tuple(chunk * factor for chunk, factor in zip(chunks, (4, 4, 1, 1)))
    return zarr.create_array(
        _WindowsRetryLocalStore(path),
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=[ZstdCodec(level=3, checksum=True)],
        # Unwritten numerical chunks must fail finite validation, not look like zero motion.
        fill_value=np.nan if np.issubdtype(np.dtype(dtype), np.inexact) else False,
        config={"write_empty_chunks": True},
        overwrite=False,
    )


def open_array(path: Path) -> zarr.Array:
    """Open read-only; the artifact loader checks hashes and finite chunk contents."""

    return zarr.open_array(str(path), mode="r", zarr_format=3)


def array_blocks(array: DenseArray) -> Iterator[tuple[slice, ...]]:
    """Visit bounded slabs, including every element exactly once."""

    shape = array.shape
    rows = min(shape[1], 128)
    row_bytes = int(np.prod(shape[2:])) * np.dtype(array.dtype).itemsize
    frames = max(1, min(32, BLOCK_BYTES // (rows * row_bytes)))
    rows = max(1, min(rows, BLOCK_BYTES // (frames * row_bytes)))
    for start in range(0, shape[0], frames):
        for top in range(0, shape[1], rows):
            yield (
                slice(start, min(start + frames, shape[0])),
                slice(top, min(top + rows, shape[1])),
                *(slice(None) for _ in shape[2:]),
            )


def spatial_blocks(
    shape: tuple[int, ...], block_width: int
) -> Iterator[tuple[slice, slice]]:
    """Bound the input of each temporal transform while retaining all time samples."""

    frames, height, width, components = shape
    # Cap input at 32 MiB; transform temporaries are a small multiple of this.
    columns = max(1, min(width, block_width, BLOCK_BYTES // (frames * components * 4)))
    rows = max(1, min(32, BLOCK_BYTES // (frames * columns * components * 4)))
    for left in range(0, width, columns):
        for top in range(0, height, rows):
            yield (
                slice(top, min(top + rows, height)),
                slice(left, min(left + columns, width)),
            )


def validate_finite(array: DenseArray, label: str) -> None:
    """Reject non-finite values without a whole-array boolean allocation."""

    for selection in array_blocks(array):
        values = array[selection]
        read_bytes(values.nbytes)
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains NaN or Inf")


def copy_array(source: DenseArray, target: zarr.Array) -> None:
    """Copy numerical bytes in bounded slabs, preserving dtype and signed zeros."""

    if source.shape != target.shape or np.dtype(source.dtype) != np.dtype(target.dtype):
        raise ValueError("Chunked copy requires identical shapes and dtypes")
    for selection in array_blocks(source):
        target[selection] = source[selection]


def storage_sha256(path: Path) -> str:
    """Hash an NPY file or the exact paths and bytes of a Zarr directory."""

    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_symlink():
        raise ValueError(f"Array storage must not be a symlink: {path}")
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for entry in files:
        if entry.is_symlink():
            raise ValueError(f"Array storage must not contain symlinks: {entry}")
        if not entry.is_file():
            continue
        if entry != path:
            name = entry.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(name).to_bytes(8, "little"))
            digest.update(name)
            digest.update(entry.stat().st_size.to_bytes(8, "little"))
        with entry.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                read_bytes(len(block))
                digest.update(block)
    return digest.hexdigest()


def read_pixels(
    array: DenseArray,
    frames: slice | int,
    pixels_xy: np.ndarray,
    component: int | None = None,
) -> np.ndarray:
    """Read paired (x,y) coordinates as [T,P,C], [P,C], [T,P] or [P]."""

    pixels = np.asarray(pixels_xy, dtype=np.int64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("Pixel selection must be [P,2]")
    if (
        np.any(pixels < 0)
        or np.any(pixels[:, 0] >= array.shape[2])
        or np.any(pixels[:, 1] >= array.shape[1])
    ):
        raise ValueError("Pixel selection leaves the image")
    scalar_frame = isinstance(frames, int)
    if isinstance(frames, int):
        times = np.asarray([frames], dtype=np.int64)
    else:
        times = np.arange(*frames.indices(array.shape[0]))
    if np.any(times < 0) or np.any(times >= array.shape[0]):
        raise ValueError("Frame selection leaves the array")
    components = (
        np.arange(array.shape[3]) if component is None else np.asarray([component])
    )
    if np.any(components < 0) or np.any(components >= array.shape[3]):
        raise ValueError("Component selection leaves the array")
    # Match NumPy's legacy paired-index layout so float32 temporal reductions
    # retain their contiguous-axis (pairwise) summation order.
    result = np.empty(
        (len(pixels), len(times), len(components)), dtype=array.dtype
    ).transpose(1, 0, 2)
    # Bound coordinate broadcasting as well as decompressed data.
    batch_size = max(
        1, min(16, BLOCK_BYTES // max(1, len(pixels) * len(components) * 40))
    )
    for start in range(0, len(times), batch_size):
        selection = (
            times[start:start + batch_size, None, None],
            pixels[None, :, 1, None], pixels[None, :, 0, None],
            components[None, None, :],
        )
        values = array.vindex[selection] if isinstance(array, zarr.Array) else array[selection]
        result[start:start + batch_size] = values
    if scalar_frame:
        result = result[0]
    if component is not None:
        result = result[..., 0]
    return result
