from __future__ import annotations

import time
from datetime import datetime
from typing import Iterator, List, Tuple


class Timer:
    """Context manager and stand-alone timer for performance measurement."""

    def __init__(self) -> None:
        self._start: float | None = None
        self._elapsed: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        self._elapsed = time.perf_counter() - self._start
        self._start = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        return self._elapsed

    def __enter__(self) -> Timer:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


def format_duration(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{seconds:.2f}s"


def parse_timestamp(ts: str | float | int, fmt: str | None = None) -> datetime:
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts)
    if fmt:
        return datetime.strptime(ts, fmt)
    return datetime.fromisoformat(ts)


def sliding_window_indices(
    length: int,
    window_size: int,
    step: int = 1,
) -> List[Tuple[int, int]]:
    if window_size > length:
        return [(0, length)]
    indices: List[Tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + window_size, length)
        indices.append((start, end))
        if end == length:
            break
        start += step
    return indices


def to_unix_ms(dt: datetime | None = None) -> int:
    dt = dt or datetime.now()
    return int(dt.timestamp() * 1000)
