"""Small, flushed text progress reports with optional append-only CLI logging."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
import time
from typing import Iterator


_logger = logging.getLogger("modal_gaussians.progress")
_logger.setLevel(logging.INFO)
_logger.propagate = False


def report_progress(message: str) -> None:
    """Show one timestamped message immediately and copy it to active run logs."""

    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    _logger.info(line)


@contextmanager
def progress_log(path: str | Path | None) -> Iterator[None]:
    """Append progress to one external log, closing only this call's handler."""

    if path is None:
        yield
        return
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(destination, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    try:
        yield
    finally:
        _logger.removeHandler(handler)
        handler.close()


def _duration(seconds: float) -> str:
    """Format elapsed/remaining seconds without rounding up unfinished work."""

    minutes, seconds_int = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}"


class Progress:
    """Report completed units, throttled to five seconds except at boundaries."""

    def __init__(
        self, label: str, total: int, *, unit: str = "items", initial: int = 0
    ) -> None:
        """Start a local work counter; resumed units do not inflate its speed."""

        self.label = label
        self.total = total
        self.unit = unit
        self.initial = initial
        self.started_at = time.perf_counter()
        self.last_report_at = self.started_at
        self.last_reported = initial
        self.update(initial, force=True)

    def update(self, completed: int, detail: str = "", *, force: bool = False) -> None:
        """Emit real completed work and a session-speed ETA, never advance it."""

        now = time.perf_counter()
        if not (
            force
            or completed == self.total
            or (completed > self.initial and self.last_reported == self.initial)
            or now - self.last_report_at >= 5.0
        ):
            return
        elapsed = now - self.started_at
        session_completed = completed - self.initial
        remaining = (
            _duration(elapsed * max(0, self.total - completed) / session_completed)
            if session_completed > 0
            else "--:--:--"
        )
        percent = 100.0 * completed / self.total if self.total else 100.0
        suffix = f" | {detail}" if detail else ""
        report_progress(
            f"{self.label}: {completed}/{self.total} {self.unit} ({percent:.1f}%)"
            f" | elapsed={_duration(elapsed)} eta={remaining}{suffix}"
        )
        self.last_report_at = now
        self.last_reported = completed
