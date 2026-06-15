"""System tests — end-to-end detection pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.isolation_forest import IsolationForestModel
from src.classification.random_forest import RandomForestClassifierModel
from src.detection_engine.engine import DetectionEngine, DetectionResult


@pytest.fixture()
def full_pipeline_engine() -> DetectionEngine:
    """Build a DetectionEngine with models trained on synthetic data."""
    rng = np.random.RandomState(42)
    n = 300

    # Generate synthetic features
    cols = [f"feat_{i}" for i in range(10)]
    X = pd.DataFrame(rng.rand(n, 10) * 1000, columns=cols)

    # Labels: 200 benign, 100 attack
    labels = ["BENIGN"] * 200 + ["DoS Hulk"] * 60 + ["DoS GoldenEye"] * 40
    y = pd.Series(labels)

    # Train Isolation Forest on benign only
    iforest = IsolationForestModel()
    iforest.train(X.iloc[:200], contamination=0.1, n_estimators=50, n_jobs=1, random_state=42)

    # Train Random Forest on all
    rf = RandomForestClassifierModel()
    rf.train(X, y, n_estimators=50, n_jobs=1, random_state=42)

    return DetectionEngine(anomaly_model=iforest, classifier_model=rf)


class TestEndToEndPipeline:
    def test_full_pipeline_detects_anomalies(self, full_pipeline_engine: DetectionEngine) -> None:
        rng = np.random.RandomState(99)
        # Generate some "attack-like" traffic (high values)
        attack_data = pd.DataFrame(rng.rand(10, 10) * 2000, columns=[f"feat_{i}" for i in range(10)])
        results = full_pipeline_engine.detect_batch(attack_data)
        assert len(results) == 10
        assert any(r.is_anomaly for r in results)

    def test_pipeline_handles_mixed_traffic(self, full_pipeline_engine: DetectionEngine) -> None:
        rng = np.random.RandomState(42)
        # Mix of normal and high-value samples
        normal = rng.rand(20, 10) * 500
        high = rng.rand(10, 10) * 3000
        mixed = np.vstack([normal, high])
        df = pd.DataFrame(mixed, columns=[f"feat_{i}" for i in range(10)])

        results = full_pipeline_engine.detect_batch(df)
        assert len(results) == 30
        # Should detect some anomalies in the high-value samples
        assert sum(1 for r in results if r.is_anomaly) > 0

    def test_pipeline_results_to_dataframe(self, full_pipeline_engine: DetectionEngine) -> None:
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.rand(20, 10) * 1000, columns=[f"feat_{i}" for i in range(10)])
        results = full_pipeline_engine.detect_batch(df)
        result_df = full_pipeline_engine.results_to_dataframe(results)
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 20
        assert "severity" in result_df.columns

    def test_pipeline_stats_are_consistent(self, full_pipeline_engine: DetectionEngine) -> None:
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.rand(50, 10) * 1000, columns=[f"feat_{i}" for i in range(10)])
        results = full_pipeline_engine.detect_batch(df)
        stats = full_pipeline_engine.compute_stats(results)
        assert stats.total_samples == 50
        assert stats.anomalies_detected + stats.normal_samples == 50

    def test_pipeline_severity_distribution(self, full_pipeline_engine: DetectionEngine) -> None:
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.rand(100, 10) * 1000, columns=[f"feat_{i}" for i in range(10)])
        results = full_pipeline_engine.detect_batch(df)
        stats = full_pipeline_engine.compute_stats(results)
        # All severity values should be valid
        for sev in stats.severity_counts:
            assert sev in ("low", "medium", "high", "critical")
