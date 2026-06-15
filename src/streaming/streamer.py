from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

import pandas as pd

from src.core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for the streaming simulation."""

    batch_size: int = 32
    replay_rate: float = 100.0
    max_samples: int = 100000
    loop: bool = False
    source: str = "dataset"


@dataclass
class StreamMetrics:
    """Metrics collected during streaming."""

    total_samples: int = 0
    total_batches: int = 0
    total_time_seconds: float = 0.0
    samples_per_second: float = 0.0
    avg_batch_latency_ms: float = 0.0
    min_batch_latency_ms: float = float("inf")
    max_batch_latency_ms: float = 0.0
    batch_latencies: List[float] = field(default_factory=list)


class DatasetStreamer:
    """Simulates real-time network traffic by streaming dataset records.

    Reads a preprocessed DataFrame and yields batches to simulate
    real-time traffic arrival patterns.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[StreamConfig] = None,
        on_batch_callback: Optional[Callable[[pd.DataFrame, int], None]] = None,
    ) -> None:
        """Initialize the dataset streamer.

        Args:
            data: Preprocessed DataFrame to stream from.
            config: Streaming configuration. If None, loads from settings.yaml.
            on_batch_callback: Optional callback function(batch_df, batch_index).
        """
        if config is None:
            cfg = get_config().load("settings")
            streaming_cfg = cfg.get("streaming", {})
            config = StreamConfig(
                batch_size=streaming_cfg.get("batch_size", 32),
                replay_rate=streaming_cfg.get("replay_rate", 100.0),
                max_samples=streaming_cfg.get("max_streaming_samples", 100000),
                loop=streaming_cfg.get("loop", False),
                source=streaming_cfg.get("source", "dataset"),
            )

        self.data = data
        self.config = config
        self.on_batch_callback = on_batch_callback
        self._metrics = StreamMetrics()

        logger.info(
            "DatasetStreamer initialized: %d samples, batch_size=%d, replay_rate=%.1f",
            len(self.data),
            self.config.batch_size,
            self.config.replay_rate,
        )

    def stream_batches(self) -> Iterator[tuple[pd.DataFrame, int]]:
        """Yield batches from the dataset.

        Yields:
            Tuple of (batch_dataframe, batch_index).
        """
        n_samples = min(len(self.data), self.config.max_samples)
        n_batches = (n_samples + self.config.batch_size - 1) // self.config.batch_size

        batch_idx = 0
        while True:
            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                batch = self.data.iloc[start:end].copy()

                yield batch, batch_idx

                if self.on_batch_callback:
                    self.on_batch_callback(batch, batch_idx)

                batch_idx += 1

                # Simulate real-time delay
                delay = self.config.batch_size / self.config.replay_rate
                time.sleep(delay)

            if not self.config.loop:
                break

    def run(self, processor: Optional[Callable[[pd.DataFrame], Any]] = None) -> StreamMetrics:
        """Run the streaming simulation with an optional processor.

        Args:
            processor: Optional function to process each batch.
                Receives a DataFrame, returns any result.

        Returns:
            StreamMetrics with collected performance metrics.
        """
        self._metrics = StreamMetrics()
        start_time = time.time()

        for batch, batch_idx in self.stream_batches():
            batch_start = time.time()

            if processor:
                processor(batch)

            batch_latency = (time.time() - batch_start) * 1000
            self._metrics.batch_latencies.append(batch_latency)
            self._metrics.total_samples += len(batch)
            self._metrics.total_batches += 1

        self._metrics.total_time_seconds = time.time() - start_time
        if self._metrics.total_time_seconds > 0:
            self._metrics.samples_per_second = (
                self._metrics.total_samples / self._metrics.total_time_seconds
            )
        if self._metrics.batch_latencies:
            self._metrics.avg_batch_latency_ms = sum(self._metrics.batch_latencies) / len(self._metrics.batch_latencies)
            self._metrics.min_batch_latency_ms = min(self._metrics.batch_latencies)
            self._metrics.max_batch_latency_ms = max(self._metrics.batch_latencies)

        logger.info(
            "Streaming complete: %d samples, %d batches, %.2f sec, %.1f samples/sec, avg_latency=%.2fms",
            self._metrics.total_samples,
            self._metrics.total_batches,
            self._metrics.total_time_seconds,
            self._metrics.samples_per_second,
            self._metrics.avg_batch_latency_ms,
        )

        return self._metrics

    @property
    def metrics(self) -> StreamMetrics:
        """Get current streaming metrics."""
        return self._metrics

    def get_sample(self, index: int) -> pd.Series:
        """Get a single sample by index.

        Args:
            index: Sample index.

        Returns:
            Series of feature values.
        """
        return self.data.iloc[index]

    def get_batch(self, start: int, end: int) -> pd.DataFrame:
        """Get a batch of samples by index range.

        Args:
            start: Start index.
            end: End index.

        Returns:
            DataFrame of features.
        """
        return self.data.iloc[start:end]

    @property
    def total_samples(self) -> int:
        """Total number of samples available for streaming."""
        return min(len(self.data), self.config.max_samples)

    @property
    def total_batches(self) -> int:
        """Total number of batches that will be streamed."""
        return (self.total_samples + self.config.batch_size - 1) // self.config.batch_size
