from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MicroBatch:
    """A micro-batch of samples."""

    batch_id: int
    data: pd.DataFrame
    created_at: float = field(default_factory=time.perf_counter)
    dispatched_at: float = 0.0
    completed_at: float = 0.0

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def queue_time_ms(self) -> float:
        if self.dispatched_at == 0.0:
            return 0.0
        return (self.dispatched_at - self.created_at) * 1000.0

    @property
    def processing_time_ms(self) -> float:
        if self.completed_at == 0.0 or self.dispatched_at == 0.0:
            return 0.0
        return (self.completed_at - self.dispatched_at) * 1000.0


class MicroBatchProcessor:
    """Accumulates individual samples and processes them in micro-batches.

    Buffers incoming samples and dispatches them when the batch size
    threshold is reached or a timeout occurs.
    """

    def __init__(
        self,
        batch_size: int = 32,
        timeout_ms: float = 100.0,
        processor: Optional[Callable[[pd.DataFrame], Any]] = None,
    ) -> None:
        """Initialize micro-batch processor.

        Args:
            batch_size: Target batch size before auto-dispatch.
            timeout_ms: Maximum wait time in ms before force dispatch.
            processor: Callable that processes a DataFrame batch.
        """
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.processor = processor

        self._buffer: List[pd.DataFrame] = []
        self._buffer_size: int = 0
        self._last_flush_time: float = time.perf_counter()
        self._batch_counter: int = 0
        self._completed_batches: List[MicroBatch] = []

        logger.info(
            "MicroBatchProcessor initialized: batch_size=%d, timeout_ms=%.1f",
            self.batch_size,
            self.timeout_ms,
        )

    def add_sample(self, sample: pd.DataFrame) -> Optional[MicroBatch]:
        """Add a sample to the buffer. Dispatches if batch is full.

        Args:
            sample: Single-row DataFrame to add.

        Returns:
            MicroBatch if dispatch occurred, None otherwise.
        """
        self._buffer.append(sample)
        self._buffer_size += len(sample)

        elapsed_ms = (time.perf_counter() - self._last_flush_time) * 1000.0

        if self._buffer_size >= self.batch_size or elapsed_ms >= self.timeout_ms:
            return self.flush()
        return None

    def flush(self) -> Optional[MicroBatch]:
        """Flush the current buffer as a micro-batch.

        Returns:
            MicroBatch if buffer had data, None if empty.
        """
        if not self._buffer:
            return None

        batch_data = pd.concat(self._buffer, ignore_index=True)
        self._batch_counter += 1

        batch = MicroBatch(
            batch_id=self._batch_counter,
            data=batch_data,
            dispatched_at=time.perf_counter(),
        )

        if self.processor:
            try:
                self.processor(batch_data)
            except Exception as e:
                logger.error("Micro-batch processing failed: %s", e)

        batch.completed_at = time.perf_counter()
        self._completed_batches.append(batch)

        logger.debug(
            "Micro-batch %d dispatched: %d samples, queue=%.2fms, process=%.2fms",
            batch.batch_id,
            batch.size,
            batch.queue_time_ms,
            batch.processing_time_ms,
        )

        self._buffer.clear()
        self._buffer_size = 0
        self._last_flush_time = time.perf_counter()

        return batch

    def get_stats(self) -> Dict[str, Any]:
        """Get micro-batch processing statistics.

        Returns:
            Dictionary with batch stats.
        """
        if not self._completed_batches:
            return {"batches": 0, "total_samples": 0}

        queue_times = [b.queue_time_ms for b in self._completed_batches]
        process_times = [b.processing_time_ms for b in self._completed_batches]
        sizes = [b.size for b in self._completed_batches]

        return {
            "batches": len(self._completed_batches),
            "total_samples": sum(sizes),
            "avg_batch_size": float(np.mean(sizes)),
            "avg_queue_time_ms": float(np.mean(queue_times)),
            "avg_processing_time_ms": float(np.mean(process_times)),
            "p95_queue_time_ms": float(np.percentile(queue_times, 95)) if queue_times else 0.0,
            "p95_processing_time_ms": float(np.percentile(process_times, 95)) if process_times else 0.0,
            "buffer_current_size": self._buffer_size,
        }

    def reset(self) -> None:
        """Clear buffer and stats."""
        self._buffer.clear()
        self._buffer_size = 0
        self._batch_counter = 0
        self._completed_batches.clear()
        self._last_flush_time = time.perf_counter()
        logger.info("MicroBatchProcessor reset")
