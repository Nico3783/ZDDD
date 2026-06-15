from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThroughputMeasurement:
    """Records a throughput measurement window."""

    operation: str
    start_time: float = field(default_factory=time.perf_counter)
    count: int = 0

    def increment(self, n: int = 1) -> None:
        """Increment the sample count."""
        self.count += n

    def elapsed_seconds(self) -> float:
        """Elapsed time since measurement started."""
        return time.perf_counter() - self.start_time

    def rate(self) -> float:
        """Compute throughput rate (samples/second)."""
        elapsed = self.elapsed_seconds()
        return self.count / elapsed if elapsed > 0 else 0.0


class ThroughputTracker:
    """Tracks throughput for detection pipeline operations.

    Measures samples processed per second across pipeline stages.
    """

    def __init__(self) -> None:
        self._rates: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}
        self._total_times: Dict[str, float] = {}
        logger.info("ThroughputTracker initialized")

    def start(self, operation: str) -> ThroughputMeasurement:
        """Start a throughput measurement.

        Args:
            operation: Name of the operation.

        Returns:
            ThroughputMeasurement to be updated.
        """
        return ThroughputMeasurement(operation=operation)

    def record_rate(self, operation: str, samples: int, elapsed_seconds: float) -> float:
        """Record a throughput rate directly.

        Args:
            operation: Name of the operation.
            samples: Number of samples processed.
            elapsed_seconds: Time taken in seconds.

        Returns:
            Throughput rate (samples/second).
        """
        rate = samples / elapsed_seconds if elapsed_seconds > 0 else 0.0

        if operation not in self._rates:
            self._rates[operation] = []
        self._rates[operation].append(rate)

        self._counts[operation] = self._counts.get(operation, 0) + samples
        self._total_times[operation] = self._total_times.get(operation, 0.0) + elapsed_seconds

        return rate

    def stop_and_record(self, measurement: ThroughputMeasurement) -> float:
        """Stop a measurement and record the rate.

        Args:
            measurement: Active ThroughputMeasurement.

        Returns:
            Throughput rate (samples/second).
        """
        elapsed = measurement.elapsed_seconds()
        return self.record_rate(measurement.operation, measurement.count, elapsed)

    def get_statistics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get throughput statistics.

        Args:
            operation: Specific operation, or None for all.

        Returns:
            Dictionary with throughput statistics per operation.
        """
        if operation:
            ops = [operation] if operation in self._rates else []
        else:
            ops = list(self._rates.keys())

        stats: Dict[str, Any] = {}
        for op in ops:
            rates = self._rates[op]
            if not rates:
                continue

            arr = np.array(rates)
            total_count = self._counts.get(op, 0)
            total_time = self._total_times.get(op, 0.0)

            stats[op] = {
                "measurements": len(rates),
                "total_samples": total_count,
                "total_time_s": total_time,
                "mean_rate": float(np.mean(arr)),
                "std_rate": float(np.std(arr)),
                "min_rate": float(np.min(arr)),
                "max_rate": float(np.max(arr)),
                "avg_rate": total_count / total_time if total_time > 0 else 0.0,
            }

        return stats

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._rates.clear()
        self._counts.clear()
        self._total_times.clear()
        logger.info("ThroughputTracker reset")

    def format_report(self) -> str:
        """Format throughput statistics as human-readable text.

        Returns:
            Formatted report string.
        """
        stats = self.get_statistics()

        lines = [
            "=" * 60,
            "  THROUGHPUT REPORT",
            "=" * 60,
            "",
            f"  {'Operation':<25} {'Total':>8} {'Time(s)':>10} {'Avg/s':>10} {'Max/s':>10}",
            "  " + "-" * 63,
        ]

        for op, s in sorted(stats.items()):
            lines.append(
                f"  {op:<25} {s['total_samples']:>8d} {s['total_time_s']:>9.2f} "
                f"{s['avg_rate']:>9.1f} {s['max_rate']:>9.1f}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)
