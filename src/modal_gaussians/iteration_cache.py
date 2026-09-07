"""Content-checked, atomically published caches and iteration timings.

Caches are derived data, never authoritative replacements for input identities.
Readers reject damaged entries; concurrent writers publish only complete entries.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

import numpy as np

from modal_gaussians.numpy_io import save_named_arrays
from modal_gaussians.progress import report_progress

DEFAULT_CACHE = Path("outputs/_cache")
_active: ContextVar["Timings | None"] = ContextVar("iteration_timings", default=None)


def process_read_bytes() -> int | None:
    if os.name != "nt":
        return None
    import ctypes
    class Counters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.GetCurrentProcess.restype = ctypes.c_void_p
    library.GetProcessIoCounters.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters)]
    counters = Counters()
    if library.GetProcessIoCounters(library.GetCurrentProcess(), ctypes.byref(counters)):
        return int(counters.read_bytes)
    return None


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_bytes(count: int) -> None:
    timer = _active.get()
    if timer is not None:
        timer.bytes_read += int(count)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            read_bytes(len(block))
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def module_revision(*modules: Any) -> str:
    return identity({m.__name__: sha256(Path(m.__file__)) for m in modules})


@contextmanager
def exclusive_work(path: Path):
    """An OS lock releases on process death; leftover lock files are harmless."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


@dataclass
class Timings:
    records: list[dict[str, Any]] = field(default_factory=list)
    bytes_read: int = 0

    @contextmanager
    def stage(self, name: str, **details: Any):
        token = _active.set(self)
        started, before = time.perf_counter(), self.bytes_read
        io_before = process_read_bytes()
        report_progress(f"iteration: {name}")
        status = "complete"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            row = {"stage": name, "seconds": time.perf_counter() - started,
                   "bytes_read": self.bytes_read - before, "status": status, **details}
            io_after = process_read_bytes()
            row["process_read_transfer_bytes"] = (None if io_before is None or io_after is None else io_after - io_before)
            self.records.append(row)
            _active.reset(token)
            report_progress(f"iteration: {name} {status} in {row['seconds']:.3f}s")

    def save(self, path: Path) -> None:
        atomic_json(path, {"stages": self.records, "bytes_read": self.bytes_read,
                           "byte_accounting": "instrumented file hashing, cache payloads and dense-array reads"})


def load_entry(root: Path, contract: dict[str, Any]) -> dict[str, np.ndarray] | None:
    path = root / identity(contract)
    if not path.exists():
        return None
    if path.is_symlink() or not (path / "manifest.json").is_file():
        raise ValueError(f"Incomplete cache entry: {path}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != contract or manifest.get("version") != 1:
        raise ValueError(f"Cache contract differs: {path}")
    if sha256(path / "arrays.npz") != manifest.get("sha256"):
        raise ValueError(f"Cache payload checksum differs: {path}")
    read_bytes((path / "arrays.npz").stat().st_size)
    with np.load(path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return arrays


def put_entry(root: Path, contract: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / identity(contract)
    if destination.exists():
        return load_entry(root, contract)  # type: ignore[return-value]
    temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=root))
    save_named_arrays(temporary / "arrays.npz", arrays)
    # Validate the temporary payload before making its content address visible.
    with np.load(temporary / "arrays.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise ValueError("Written cache inventory differs")
        for name, expected in arrays.items():
            actual = archive[name]
            expected = np.asarray(expected)
            if (actual.shape != expected.shape or actual.dtype != expected.dtype
                    or not np.array_equal(np.ascontiguousarray(actual).reshape(-1).view(np.uint8),
                                          np.ascontiguousarray(expected).reshape(-1).view(np.uint8))):
                raise ValueError(f"Written cache array differs: {name}")
    atomic_json(temporary / "manifest.json", {"version": 1, "contract": contract,
                                               "sha256": sha256(temporary / "arrays.npz")})
    try:
        # rename (not replace) is intentional: never overwrite another publisher.
        os.rename(temporary, destination)
    except OSError:
        if not destination.exists():
            raise
        # Keep the completed competing temporary for inspection; it is not a hit.
    return load_entry(root, contract)  # type: ignore[return-value]


def cached(root: Path, contract: dict[str, Any], build: Callable[[], dict[str, np.ndarray]],
           timer: Timings, stage: str) -> dict[str, np.ndarray]:
    with timer.stage(stage):
        arrays = load_entry(root, contract)
        hit = arrays is not None
        if arrays is None:
            arrays = put_entry(root, contract, build())
    timer.records[-1]["cache_hit"] = hit
    return arrays
