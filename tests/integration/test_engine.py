"""Integration tests for the DetectionEngine pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.isolation_forest import IsolationForestModel
from src.classification.random_forest import RandomForestClassifierModel
from src.detection_engine.engine import DetectionEngine, DetectionResult, DetectionStats


@pytest.fixture()
def engine_with_models(
    trained_isolation_forest: IsolationForestModel,
    trained_random_forest: RandomForestClassifierModel,
) -> DetectionEngine:
    """Create a DetectionEngine with trained models."""
    return DetectionEngine(
        anomaly_model=trained_isolation_forest,
        classifier_model=trained_random_forest,
    )


class TestDetectionEngineDetectBatch:
    def test_returns_list_of_detection_result(
        self, engine_with_models: DetectionEngine, synthetic_flow_features: pd.DataFrame
    ) -> None:
        results = engine_with_models.detect_batch(synthetic_flow_features.iloc[:10])
        assert isinstance(results, list)
        assert len(results) == 10
        assert all(isinstance(r, DetectionResult) for r in results)

    def test_results_have_required_fields(
        self, engine_with_models: DetectionEngine, synthetic_flow_features: pd.DataFrame
    ) -> None:
        results = engine_with_models.detect_batch(synthetic_flow_features.iloc[:5])
        for r in results:
            assert hasattr(r, "sample_id")
            assert hasattr(r, "is_anomaly")
            assert hasattr(r, "anomaly_score")
            assert hasattr(r, "predicted_class")
            assert hasattr(r, "severity")
            assert hasattr(r, "is_zero_day")

    def test_anomaly_score_is_float(
        self, engine_with_models: DetectionEngine, synthetic_flow_features: pd.DataFrame
    ) -> None:
        results = engine_with_models.detect_batch(synthetic_flow_features.iloc[:5])
        for r in results:
            assert isinstance(r.anomaly_score, float)

    def test_severity_is_valid_string(
        self, engine_with_models: DetectionEngine, synthetic_flow_features: pd.DataFrame
    ) -> None:
        results = engine_with_models.detect_batch(synthetic_flow_features.iloc[:10])
        for r in results:
            assert r.severity in ("low", "medium", "high", "critical")

    def test_custom_sample_ids(
        self, engine_with_models: DetectionEngine, synthetic_flow_features: pd.DataFrame
    ) -> None:
        ids = ["custom_0", "custom_1", "custom_2"]
        results = engine_with_models.detect_batch(synthetic_flow_features.iloc[:3], sample_ids=ids)
        assert [r.sample_id for r in results] == ids

    def test_default_sample_ids_generated(
        self, engine_with_models: DetectionEngine, synthetic_flow_features: pd.DataFrame
    ) -> None:
        results = engine_with_models.detect_batch(synthetic_flow_features.iloc[:3])
        assert [r.sample_id for r in results] == ["sample_0", "sample_1", "sample_2"]


class TestDetectionEngineComputeStats:
    def test_empty_results(self) -> None:
        engine = DetectionEngine()
        stats = engine.compute_stats([])
        assert stats.total_samples == 0
        assert stats.anomalies_detected == 0

    def test_stats_counts_anomalies(
        self, engine_with_models: DetectionEngine, synthetic_flow_features: pd.DataFrame
    ) -> None:
        results = engine_with_models.detect_batch(synthetic_flow_features.iloc[:20])
        stats = engine_with_models.compute_stats(results)
        assert stats.total_samples == 20
        assert stats.anomalies_detected + stats.normal_samples == 20
        assert stats.mean_anomaly_score >= 0.0

    def test_stats_severity_counts(
        self, engine_with_models: DetectionEngine, synthetic_flow_features: pd.DataFrame
    ) -> None:
        results = engine_with_models.detect_batch(synthetic_flow_features.iloc[:20])
        stats = engine_with_models.compute_stats(results)
        assert isinstance(stats.severity_counts, dict)
        total_severity = sum(stats.severity_counts.values())
        assert total_severity == 20


class TestDetectionEngineResultsToDataFrame:
    def test_conversion(self, engine_with_models: DetectionEngine, synthetic_flow_features: pd.DataFrame) -> None:
        results = engine_with_models.detect_batch(synthetic_flow_features.iloc[:10])
        df = engine_with_models.results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "sample_id" in df.columns
        assert "is_anomaly" in df.columns
        assert "severity" in df.columns


class TestDetectionEngineUpdateModels:
    def test_update_models(self, engine_with_models: DetectionEngine) -> None:
        new_iforest = IsolationForestModel()
        engine_with_models.update_models(anomaly_model=new_iforest)
        assert engine_with_models.anomaly_model is new_iforest

    def test_update_classifier(self, engine_with_models: DetectionEngine) -> None:
        new_rf = RandomForestClassifierModel()
        engine_with_models.update_models(classifier_model=new_rf)
        assert engine_with_models.classifier_model is new_rf


class TestDetectionEngineWithoutModels:
    def test_no_models_returns_no_anomalies(self, synthetic_flow_features: pd.DataFrame) -> None:
        engine = DetectionEngine()
        results = engine.detect_batch(synthetic_flow_features.iloc[:5])
        assert all(not r.is_anomaly for r in results)
        assert all(r.predicted_class == "BENIGN" for r in results)


class TestDetectionEngineSeverityComputation:
    def test_high_score_is_critical(self, engine_with_models: DetectionEngine) -> None:
        sev = engine_with_models._compute_severity(0.95, 0.9, False)
        assert sev == "critical"

    def test_medium_score_is_medium(self, engine_with_models: DetectionEngine) -> None:
        sev = engine_with_models._compute_severity(0.6, 0.8, False)
        assert sev == "medium"

    def test_low_score_is_low(self, engine_with_models: DetectionEngine) -> None:
        sev = engine_with_models._compute_severity(0.3, 0.9, False)
        assert sev == "low"

    def test_zero_day_high_confidence_is_critical(self, engine_with_models: DetectionEngine) -> None:
        sev = engine_with_models._compute_severity(0.75, 0.9, True)
        assert sev == "critical"
