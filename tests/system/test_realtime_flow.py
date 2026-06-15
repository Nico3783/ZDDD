"""End-to-end system test for real-time anomaly detection flow.

Tests the complete pipeline: DatasetStreamer -> Engine -> AlertManager -> Logger,
simulating real-time traffic streaming and alert generation.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.isolation_forest import IsolationForestModel
from src.alerting.logger import AlertLogger
from src.classification.random_forest import RandomForestClassifierModel
from src.detection_engine.alert_manager import AlertManager
from src.detection_engine.engine import DetectionEngine, DetectionResult
from src.preprocessing.cleaner import clean_dataset
from src.preprocessing.encoder import encode_labels
from src.preprocessing.scaler import fit_transform_features
from src.streaming.streamer import DatasetStreamer, StreamConfig
from src.streaming.simulator import StreamSimulator


@pytest.fixture()
def system_engine(synthetic_flow_features: pd.DataFrame) -> DetectionEngine:
    """Return a fully trained DetectionEngine for system tests."""
    feature_cols = list(synthetic_flow_features.columns)

    iforest = IsolationForestModel()
    iforest.train(
        synthetic_flow_features,
        contamination=0.1,
        n_estimators=20,
        n_jobs=1,
        random_state=42,
    )

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
def alert_manager(mock_config_loader: MagicMock) -> AlertManager:
    """Return a fresh AlertManager."""
    return AlertManager()


@pytest.fixture()
def log_dir(tmp_path: Path) -> Path:
    """Return a temporary log directory."""
    d = tmp_path / "logs"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Full real-time flow
# ---------------------------------------------------------------------------
class TestRealtimeFlow:
    def test_stream_detect_alert(
        self,
        synthetic_flow_features: pd.DataFrame,
        system_engine: DetectionEngine,
        alert_manager: AlertManager,
        log_dir: Path,
    ) -> None:
        """Stream -> detect -> alert -> log in a single pass."""
        config = StreamConfig(batch_size=32)
        streamer = DatasetStreamer(data=synthetic_flow_features, config=config)

        all_results: list[DetectionResult] = []
        for batch, idx in streamer.stream_batches():
            results = system_engine.detect_batch(batch)
            all_results.extend(results)

            for r in results:
                if r.is_anomaly:
                    alerts = alert_manager.process_detection_results([r])
                    for alert in alerts:
                        AlertLogger(log_dir=str(log_dir)).log_alert(alert)

        total = len(all_results)
        anomalies = [r for r in all_results if r.is_anomaly]

        assert total == len(synthetic_flow_features)
        assert len(anomalies) > 0

    def test_simulated_attack_detected(
        self,
        alert_manager: AlertManager,
    ) -> None:
        """Simulated traffic with attacks produces anomalies."""
        sim = StreamSimulator(random_state=42)
        df = sim.generate_traffic(n_packets=200, attack_ratio=0.4)

        feature_cols = [c for c in df.columns if c not in ("Label", "label", "timestamp")]
        X = df[feature_cols].select_dtypes(include=[np.number])

        # Train engine on simulator's column space
        iforest = IsolationForestModel()
        iforest.train(X, contamination=0.1, n_estimators=20, n_jobs=1, random_state=42)
        engine = DetectionEngine(anomaly_model=iforest)

        results = engine.detect_batch(X)
        anomalies = [r for r in results if r.is_anomaly]
        assert len(anomalies) > 0, "Expected anomalies in simulated attack traffic"

    def test_alert_rate_limiting(
        self,
        synthetic_flow_features: pd.DataFrame,
        system_engine: DetectionEngine,
        mock_config_loader: MagicMock,
    ) -> None:
        """AlertManager rate limiting prevents alert flooding."""
        am = AlertManager()
        am.cooldown_seconds = 0
        am.max_alerts_per_minute = 5
        results = system_engine.detect_batch(synthetic_flow_features)

        alerts = []
        for r in results:
            if r.is_anomaly:
                batch = am.process_detection_results([r])
                alerts.extend(batch)

        # Rate limit should cap alerts
        assert len(alerts) <= 10  # Allow small margin

    def test_detection_stats(
        self,
        synthetic_flow_features: pd.DataFrame,
        system_engine: DetectionEngine,
    ) -> None:
        """DetectionEngine.compute_stats returns correct structure."""
        results = system_engine.detect_batch(synthetic_flow_features)
        stats = system_engine.compute_stats(results)

        assert stats.total_samples == len(synthetic_flow_features)
        assert stats.anomalies_detected >= 0
        assert 0.0 <= stats.anomaly_rate <= 1.0

    def test_batch_prediction_consistency(
        self,
        synthetic_flow_features: pd.DataFrame,
        system_engine: DetectionEngine,
    ) -> None:
        """Batch predictions are deterministic for same input."""
        results1 = system_engine.detect_batch(synthetic_flow_features)
        results2 = system_engine.detect_batch(synthetic_flow_features)

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2, strict=True):
            assert r1.is_anomaly == r2.is_anomaly
            assert r1.anomaly_score == pytest.approx(r2.anomaly_score, abs=1e-6)
