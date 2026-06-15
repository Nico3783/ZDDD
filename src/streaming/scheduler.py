from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.streaming.streamer import DatasetStreamer, StreamConfig, StreamMetrics

logger = logging.getLogger(__name__)


@dataclass
class ThroughputMeasurement:
    """Throughput measurement result."""

    samples_per_second: float
    batches_per_second: float
    total_samples: int
    total_batches: int
    measurement_duration_seconds: float


@dataclass
class LatencyMeasurement:
    """Latency measurement result."""

    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_measurements: int


class StreamScheduler:
    """Schedules and manages streaming simulation runs.

    Provides throughput and latency measurement capabilities.
    """

    def __init__(self, streamer: DatasetStreamer) -> None:
        """Initialize the scheduler.

        Args:
            streamer: Configured DatasetStreamer instance.
        """
        self.streamer = streamer
        self._throughput_history: List[ThroughputMeasurement] = []
        self._latency_history: List[LatencyMeasurement] = []
        logger.info("StreamScheduler initialized")

    def measure_throughput(
        self,
        processor: Optional[Any] = None,
    ) -> ThroughputMeasurement:
        """Run a streaming simulation and measure throughput.

        Args:
            processor: Optional batch processor function.

        Returns:
            ThroughputMeasurement result.
        """
        metrics = self.streamer.run(processor=processor)

        measurement = ThroughputMeasurement(
            samples_per_second=metrics.samples_per_second,
            batches_per_second=metrics.total_batches / max(metrics.total_time_seconds, 0.001),
            total_samples=metrics.total_samples,
            total_batches=metrics.total_batches,
            measurement_duration_seconds=metrics.total_time_seconds,
        )

        self._throughput_history.append(measurement)
        logger.info(
            "Throughput measurement: %.1f samples/sec, %.2f batches/sec",
            measurement.samples_per_second,
            measurement.batches_per_second,
        )

        return measurement

    def measure_latency(
        self,
        processor: Optional[Any] = None,
        n_iterations: int = 1,
    ) -> LatencyMeasurement:
        """Measure processing latency across multiple iterations.

        Args:
            processor: Optional batch processor function.
            n_iterations: Number of streaming iterations.

        Returns:
            LatencyMeasurement result.
        """
        all_latencies: List[float] = []

        for i in range(n_iterations):
            metrics = self.streamer.run(processor=processor)
            all_latencies.extend(metrics.batch_latencies)

        if not all_latencies:
            return LatencyMeasurement(
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                min_latency_ms=0.0,
                max_latency_ms=0.0,
                total_measurements=0,
            )

        sorted_latencies = sorted(all_latencies)
        n = len(sorted_latencies)

        measurement = LatencyMeasurement(
            avg_latency_ms=sum(sorted_latencies) / n,
            p50_latency_ms=sorted_latencies[int(n * 0.5)],
            p95_latency_ms=sorted_latencies[int(n * 0.95)],
            p99_latency_ms=sorted_latencies[int(n * 0.99)],
            min_latency_ms=sorted_latencies[0],
            max_latency_ms=sorted_latencies[-1],
            total_measurements=n,
        )

        self._latency_history.append(measurement)
        logger.info(
            "Latency measurement: avg=%.2fms, p95=%.2fms, p99=%.2fms",
            measurement.avg_latency_ms,
            measurement.p95_latency_ms,
            measurement.p99_latency_ms,
        )

        return measurement

    def benchmark(
        self,
        processor: Optional[Any] = None,
        n_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Run a full benchmark (throughput + latency).

        Args:
            processor: Optional batch processor function.
            n_iterations: Number of iterations for latency measurement.

        Returns:
            Dictionary with benchmark results.
        """
        throughput = self.measure_throughput(processor=processor)
        latency = self.measure_latency(processor=processor, n_iterations=n_iterations)

        return {
            "throughput": {
                "samples_per_second": throughput.samples_per_second,
                "batches_per_second": throughput.batches_per_second,
                "total_samples": throughput.total_samples,
                "duration_seconds": throughput.measurement_duration_seconds,
            },
            "latency": {
                "avg_ms": latency.avg_latency_ms,
                "p50_ms": latency.p50_latency_ms,
                "p95_ms": latency.p95_latency_ms,
                "p99_ms": latency.p99_latency_ms,
                "min_ms": latency.min_latency_ms,
                "max_ms": latency.max_latency_ms,
                "measurements": latency.total_measurements,
            },
        }

    def get_history(self) -> Dict[str, List]:
        """Get measurement history.

        Returns:
            Dictionary with throughput and latency history.
        """
        return {
            "throughput": [
                {
                    "samples_per_second": m.samples_per_second,
                    "total_samples": m.total_samples,
                    "duration": m.measurement_duration_seconds,
                }
                for m in self._throughput_history
            ],
            "latency": [
                {
                    "avg_ms": m.avg_latency_ms,
                    "p95_ms": m.p95_latency_ms,
                    "p99_ms": m.p99_latency_ms,
                    "measurements": m.total_measurements,
                }
                for m in self._latency_history
            ],
        }
