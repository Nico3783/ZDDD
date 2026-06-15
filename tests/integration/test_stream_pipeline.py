"""Integration tests for the streaming simulation pipeline.

Tests the DatasetStreamer, StreamScheduler, and StreamSimulator working together
to process synthetic traffic through the detection engine.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.detection_engine.engine import DetectionEngine, DetectionResult
from src.anomaly_detection.isolation_forest import IsolationForestModel
from src.classification.random_forest import RandomForestClassifierModel
from src.preprocessing.cleaner import clean_dataset
from src.preprocessing.encoder import encode_labels
from src.preprocessing.scaler import fit_transform_features
from src.streaming.streamer import DatasetStreamer, StreamConfig
from src.streaming.scheduler import StreamScheduler
from src.streaming.simulator import StreamSimulator


@pytest.fixture()
def small_engine(synthetic_flow_features: pd.DataFrame) -> DetectionEngine:
    """Return a DetectionEngine trained on a small synthetic dataset."""
    feature_cols = list(synthetic_flow_features.columns)
    iforest = IsolationForestModel()
    iforest.train(
        synthetic_flow_features,
        contamination=0.1,
        n_estimators=20,
        n_jobs=1,
        random_state=42,
    )

    # Create fake labels for RF training
    labels = np.where(
        synthetic_flow_features.index < len(synthetic_flow_features) // 2,
        "BENIGN",
        "ATTACK",
    )

    rf = RandomForestClassifierModel()
    rf.train(
        synthetic_flow_features,
        labels,
        n_estimators=20,
        n_jobs=1,
        random_state=42,
    )

    return DetectionEngine(anomaly_model=iforest, classifier_model=rf)


@pytest.fixture()
def small_df(synthetic_flow_features: pd.DataFrame) -> pd.DataFrame:
    """Return a small DataFrame for streaming tests."""
    return synthetic_flow_features.head(50).copy()


@pytest.fixture()
def sample_dataframe() -> pd.DataFrame:
    """Return a DataFrame with realistic network traffic columns."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame(
        {
            "Flow Duration": rng.uniform(0, 120, n),
            "Total Fwd Packets": rng.poisson(10, n).astype(float),
            "Total Bwd Packets": rng.poisson(5, n).astype(float),
            "Fwd Packet Length Mean": rng.exponential(100, n),
            "Bwd Packet Length Mean": rng.exponential(80, n),
            "Flow Bytes/s": rng.exponential(5000, n),
            "Flow Packets/s": rng.exponential(50, n),
            "Label": rng.choice(["BENIGN", "DDoS", "DoS"], n, p=[0.7, 0.2, 0.1]),
        }
    )


# ---------------------------------------------------------------------------
# DatasetStreamer
# ---------------------------------------------------------------------------
class TestDatasetStreamer:
    def test_stream_yields_batches(self, small_df: pd.DataFrame) -> None:
        config = StreamConfig(batch_size=16)
        streamer = DatasetStreamer(data=small_df, config=config)

        batches = list(streamer.stream_batches())
        assert len(batches) > 0
        for batch_df, idx in batches:
            assert isinstance(batch_df, pd.DataFrame)
            assert len(batch_df) <= 16

    def test_stream_total_samples(self, small_df: pd.DataFrame) -> None:
        config = StreamConfig(batch_size=16)
        streamer = DatasetStreamer(data=small_df, config=config)

        assert streamer.total_samples == len(small_df)

    def test_stream_total_batches(self, small_df: pd.DataFrame) -> None:
        config = StreamConfig(batch_size=16)
        streamer = DatasetStreamer(data=small_df, config=config)

        expected_batches = int(np.ceil(len(small_df) / 16))
        assert streamer.total_batches == expected_batches

    def test_stream_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        config = StreamConfig(batch_size=16)
        streamer = DatasetStreamer(data=df, config=config)

        batches = list(streamer.stream_batches())
        assert len(batches) == 0

    def test_stream_single_batch(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        config = StreamConfig(batch_size=100)
        streamer = DatasetStreamer(data=df, config=config)

        batches = list(streamer.stream_batches())
        assert len(batches) == 1
        batch_df, _idx = batches[0]
        assert len(batch_df) == 3


# ---------------------------------------------------------------------------
# StreamScheduler
# ---------------------------------------------------------------------------
class TestStreamScheduler:
    def test_benchmark_returns_throughput(
        self, small_df: pd.DataFrame, small_engine: DetectionEngine
    ) -> None:
        config = StreamConfig(batch_size=16)
        streamer = DatasetStreamer(data=small_df, config=config)
        scheduler = StreamScheduler(streamer)

        result = scheduler.benchmark(
            processor=small_engine.detect_batch, n_iterations=2
        )
        assert "throughput" in result
        assert result["throughput"]["samples_per_second"] > 0

    def test_get_history_returns_list(
        self, small_df: pd.DataFrame, small_engine: DetectionEngine
    ) -> None:
        config = StreamConfig(batch_size=16)
        streamer = DatasetStreamer(data=small_df, config=config)
        scheduler = StreamScheduler(streamer)

        scheduler.benchmark(
            processor=small_engine.detect_batch, n_iterations=2
        )
        history = scheduler.get_history()
        assert isinstance(history, dict)
        assert "throughput" in history


# ---------------------------------------------------------------------------
# StreamSimulator
# ---------------------------------------------------------------------------
class TestStreamSimulator:
    def test_generate_traffic(self) -> None:
        sim = StreamSimulator(random_state=42)
        df = sim.generate_traffic(n_packets=100, attack_ratio=0.2)
        assert len(df) == 100
        assert "Label" in df.columns or "label" in df.columns
        assert "timestamp" in df.columns

    def test_stream_from_dataframe(self, sample_dataframe: pd.DataFrame) -> None:
        sim = StreamSimulator(random_state=42)
        batches = list(sim.stream_from_dataframe(sample_dataframe, batch_size=50))
        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, pd.DataFrame)

    def test_attack_ratio(self) -> None:
        sim = StreamSimulator(random_state=42)
        df = sim.generate_traffic(n_packets=1000, attack_ratio=0.3)
        # Allow some variance
        n_attack = len(df[df.get("Label", df.get("label", pd.Series())) != "BENIGN"])
        assert 200 < n_attack < 400  # ~300 with ±10% variance

    def test_burst_traffic(self) -> None:
        sim = StreamSimulator(random_state=42)
        base_df = sim.generate_traffic(n_packets=50, attack_ratio=0.2)
        df = sim.simulate_bursts(base_df, burst_size_range=(5, 10))
        assert len(df) >= len(base_df)


# ---------------------------------------------------------------------------
# StreamSimulator + DetectionEngine integration
# ---------------------------------------------------------------------------
class TestSimulatorDetectionIntegration:
    def test_simulated_stream_detected(
        self, synthetic_flow_features: pd.DataFrame
    ) -> None:
        feature_cols = list(synthetic_flow_features.columns)
        iforest = IsolationForestModel()
        iforest.train(
            synthetic_flow_features,
            contamination=0.1,
            n_estimators=20,
            n_jobs=1,
            random_state=42,
        )
        rf = RandomForestClassifierModel()
        labels = np.where(
            synthetic_flow_features.index < len(synthetic_flow_features) // 2,
            "BENIGN",
            "ATTACK",
        )
        rf.train(
            synthetic_flow_features,
            labels,
            n_estimators=20,
            n_jobs=1,
            random_state=42,
        )

        engine = DetectionEngine(anomaly_model=iforest, classifier_model=rf)

        # Stream and process
        config = StreamConfig(batch_size=32)
        streamer = DatasetStreamer(data=synthetic_flow_features, config=config)

        total_detected = 0
        total_samples = 0
        for batch, idx in streamer.stream_batches():
            results = engine.detect_batch(batch)
            total_detected += sum(1 for r in results if r.is_anomaly)
            total_samples += len(batch)

        assert total_samples == len(synthetic_flow_features)
        assert total_detected > 0  # Some anomalies should be found
