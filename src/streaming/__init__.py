"""Streaming simulation system.

Simulates real-time network traffic by streaming dataset records,
measuring throughput and latency for the detection pipeline.
"""

from src.streaming.dispatcher import BatchDispatcher, DispatchResult
from src.streaming.microbatch import MicroBatch, MicroBatchProcessor
from src.streaming.scheduler import (
    LatencyMeasurement,
    StreamScheduler,
    ThroughputMeasurement,
)
from src.streaming.simulator import StreamSimulator
from src.streaming.stream_reader import StreamStreamReader
from src.streaming.streamer import DatasetStreamer, StreamConfig, StreamMetrics

__all__ = [
    "BatchDispatcher",
    "DatasetStreamer",
    "DispatchResult",
    "LatencyMeasurement",
    "MicroBatch",
    "MicroBatchProcessor",
    "StreamConfig",
    "StreamMetrics",
    "StreamScheduler",
    "StreamSimulator",
    "StreamStreamReader",
    "ThroughputMeasurement",
]
