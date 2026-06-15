from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatencyMeasurement:
    """Records a single latency measurement."""

    operation: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0
    duration_ms: float = 0.0

    def stop(self) -> float:
        """Stop the timer and compute duration."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        return self.duration_ms


class LatencyTracker:
    """Tracks latency for detection pipeline operations.

    Records per-operation timing and computes aggregate statistics.
    """

    def __init__(self) -> None:
        self._measurements: Dict[str, List[float]] = {}
        logger.info("LatencyTracker initialized")

    def start(self, operation: str) -> LatencyMeasurement:
        """Start a latency measurement.

        Args:
            operation: Name of the operation.

        Returns:
            LatencyMeasurement instance to be stopped later.
        """
        return LatencyMeasurement(operation=operation)

    def record(self, operation: str, duration_ms: float) -> None:
        """Record a latency measurement directly.

        Args:
            operation: Name of the operation.
            duration_ms: Duration in milliseconds.
        """
        if operation not in self._measurements:
            self._measurements[operation] = []
        self._measurements[operation].append(duration_ms)

    def stop_and_record(self, measurement: LatencyMeasurement) -> float:
        """Stop a measurement and record it.

        Args:
            measurement: Active LatencyMeasurement.

        Returns:
            Duration in milliseconds.
        """
        duration = measurement.stop()
        self.record(measurement.operation, duration)
        return duration

    def get_statistics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get latency statistics.

        Args:
            operation: Specific operation, or None for all.

        Returns:
            Dictionary with latency statistics per operation.
        """
        if operation:
            ops = [operation] if operation in self._measurements else []
        else:
            ops = list(self._measurements.keys())

        stats: Dict[str, Any] = {}
        for op in ops:
            durations = self._measurements[op]
            if not durations:
                continue

            arr = np.array(durations)
            stats[op] = {
                "count": len(durations),
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "total_ms": float(np.sum(arr)),
            }

        return stats

    def get_overall_latency(self) -> Dict[str, Any]:
        """Get overall pipeline latency across all operations.

        Returns:
            Dictionary with overall latency stats.
        """
        all_durations = []
        for durations in self._measurements.values():
            all_durations.extend(durations)

        if not all_durations:
            return {"count": 0, "mean_ms": 0.0, "total_ms": 0.0}

        arr = np.array(all_durations)
        return {
            "count": len(all_durations),
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "total_ms": float(np.sum(arr)),
        }

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._measurements.clear()
        logger.info("LatencyTracker reset")

    def format_report(self) -> str:
        """Format latency statistics as human-readable text.

        Returns:
            Formatted report string.
        """
        stats = self.get_statistics()
        overall = self.get_overall_latency()

        lines = [
            "=" * 60,
            "  LATENCY REPORT",
            "=" * 60,
            "",
            f"  {'Operation':<25} {'Count':>6} {'Mean':>10} {'P95':>10} {'Max':>10}",
            "  " + "-" * 61,
        ]

        for op, s in sorted(stats.items()):
            lines.append(
                f"  {op:<25} {s['count']:>6d} {s['mean_ms']:>9.2f}ms "
                f"{s['p95_ms']:>9.2f}ms {s['max_ms']:>9.2f}ms"
            )

        lines.append("  " + "-" * 61)
        lines.append(
            f"  {'TOTAL':<25} {overall['count']:>6d} {overall['mean_ms']:>9.2f}ms "
            f"{overall.get('p95_ms', 0):>9.2f}ms {overall.get('max_ms', 0):>9.2f}ms"
        )
        lines.append("=" * 60)

        return "\n".join(lines)
