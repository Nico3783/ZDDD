"""Unit tests for src/detection_engine/orchestrator.py (DetectionOrchestrator)."""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.detection_engine.orchestrator import DetectionOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_engine() -> MagicMock:
    """Return a mock DetectionEngine that produces a predictions DataFrame."""
    engine = MagicMock()
    n = 5
    predictions = pd.DataFrame({
        "is_anomaly": [True, False, True, False, False],
        "is_zero_day": [False, False, False, True, False],
        "severity": ["high", "low", "critical", "medium", "low"],
        "predicted_class": ["DoS Hulk", "BENIGN", "ZERO_DAY", "BENIGN", "BENIGN"],
        "anomaly_score": [0.8, 0.2, 0.95, 0.6, 0.3],
        "classification_confidence": [0.9, 1.0, 0.4, 0.8, 1.0],
        "sample_id": [f"sample_{i}" for i in range(n)],
    })
    engine.detect_batch.return_value = [MagicMock() for _ in range(n)]
    engine.results_to_dataframe.return_value = predictions
    return engine


@pytest.fixture()
def mock_alert_logger() -> MagicMock:
    """Return a mock AlertLogger."""
    return MagicMock()


@pytest.fixture()
def orchestrator(mock_engine: MagicMock, mock_alert_logger: MagicMock) -> DetectionOrchestrator:
    """Return a DetectionOrchestrator with mocked engine and logger."""
    return DetectionOrchestrator(engine=mock_engine, alert_logger=mock_alert_logger)


@pytest.fixture()
def empty_predictions_df() -> pd.DataFrame:
    """Return a DataFrame with zero rows matching expected prediction columns."""
    return pd.DataFrame({
        "is_anomaly": pd.Series(dtype=bool),
        "is_zero_day": pd.Series(dtype=bool),
        "severity": pd.Series(dtype=str),
        "predicted_class": pd.Series(dtype=str),
        "anomaly_score": pd.Series(dtype=float),
        "classification_confidence": pd.Series(dtype=float),
        "sample_id": pd.Series(dtype=str),
    })


@pytest.fixture()
def single_record() -> Dict[str, Any]:
    """Return a single network-traffic record dict."""
    return {
        "Fwd Packet Length Mean": 120.5,
        "Fwd Packet Length Std": 45.2,
        "Bwd Packet Length Mean": 80.0,
        "Bwd Packet Length Std": 30.1,
        "Flow Duration": 5000.0,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 8,
    }


@pytest.fixture()
def sample_records() -> List[Dict[str, Any]]:
    """Return a list of record dicts for batch processing."""
    return [
        {
            "Fwd Packet Length Mean": 120.5,
            "Fwd Packet Length Std": 45.2,
            "Bwd Packet Length Mean": 80.0,
            "Bwd Packet Length Std": 30.1,
            "Flow Duration": 5000.0,
            "Total Fwd Packets": 10,
            "Total Backward Packets": 8,
        }
        for _ in range(3)
    ]


# ---------------------------------------------------------------------------
# process_dataframe tests
# ---------------------------------------------------------------------------

class TestProcessDataframe:
    """Tests for DetectionOrchestrator.process_dataframe."""

    def test_returns_dict_with_expected_keys(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        result = orchestrator.process_dataframe(synthetic_flow_features)

        assert isinstance(result, dict)
        assert "total_samples" in result
        assert "total_alerts" in result
        assert "anomaly_count" in result
        assert "zero_day_count" in result
        assert "processing_time_seconds" in result
        assert "throughput_samples_per_sec" in result

    def test_total_samples_matches_input_rows(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        result = orchestrator.process_dataframe(synthetic_flow_features)

        assert result["total_samples"] == len(synthetic_flow_features)

    def test_alerts_and_counts_are_integers(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        result = orchestrator.process_dataframe(synthetic_flow_features)

        assert isinstance(result["total_alerts"], int)
        assert isinstance(result["anomaly_count"], int)
        assert isinstance(result["zero_day_count"], int)

    def test_processing_time_is_non_negative(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        result = orchestrator.process_dataframe(synthetic_flow_features)

        assert result["processing_time_seconds"] >= 0.0

    def test_throughput_is_non_negative(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        result = orchestrator.process_dataframe(synthetic_flow_features)

        assert result["throughput_samples_per_sec"] >= 0.0

    def test_alert_count_matches_anomaly_or_zero_day_rows(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        result = orchestrator.process_dataframe(synthetic_flow_features)

        # Mock engine produces 3 alerts (rows where is_anomaly or is_zero_day is True)
        assert result["total_alerts"] == 3

    def test_calls_engine_detect_batch(
        self,
        orchestrator: DetectionOrchestrator,
        mock_engine: MagicMock,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        orchestrator.process_dataframe(synthetic_flow_features)

        mock_engine.detect_batch.assert_called_once()

    def test_logs_each_alert(
        self,
        orchestrator: DetectionOrchestrator,
        mock_alert_logger: MagicMock,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        orchestrator.process_dataframe(synthetic_flow_features)

        assert mock_alert_logger.log_alert.call_count == 3

    def test_empty_dataframe_returns_zero_samples(
        self,
        orchestrator: DetectionOrchestrator,
        mock_engine: MagicMock,
    ) -> None:
        empty_df = pd.DataFrame(columns=[
            "Fwd Packet Length Mean", "Fwd Packet Length Std",
            "Bwd Packet Length Mean", "Bwd Packet Length Std",
            "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        ])
        empty_preds = pd.DataFrame({
            "is_anomaly": pd.Series(dtype=bool),
            "is_zero_day": pd.Series(dtype=bool),
            "severity": pd.Series(dtype=str),
            "predicted_class": pd.Series(dtype=str),
            "anomaly_score": pd.Series(dtype=float),
            "classification_confidence": pd.Series(dtype=float),
            "sample_id": pd.Series(dtype=str),
        })
        mock_engine.results_to_dataframe.return_value = empty_preds

        result = orchestrator.process_dataframe(empty_df)

        assert result["total_samples"] == 0
        assert result["total_alerts"] == 0
        assert result["anomaly_count"] == 0
        assert result["zero_day_count"] == 0

    def test_return_details_includes_predictions(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        result = orchestrator.process_dataframe(synthetic_flow_features, return_details=True)

        assert "predictions" in result
        assert isinstance(result["predictions"], list)
        # Mock engine returns 5 predictions regardless of input size
        assert len(result["predictions"]) == 5

    def test_return_details_false_excludes_predictions(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        result = orchestrator.process_dataframe(synthetic_flow_features, return_details=False)

        assert "predictions" not in result

    def test_engine_exception_raises_detection_error(
        self,
        orchestrator: DetectionOrchestrator,
        mock_engine: MagicMock,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        from src.core.exceptions import DetectionError

        mock_engine.detect_batch.side_effect = RuntimeError("model failure")

        with pytest.raises(DetectionError, match="Orchestration failed"):
            orchestrator.process_dataframe(synthetic_flow_features)

    def test_increments_processed_count(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        orchestrator.process_dataframe(synthetic_flow_features)

        stats = orchestrator.get_stats()
        assert stats["processed_count"] == len(synthetic_flow_features)

    def test_increments_alert_count(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        orchestrator.process_dataframe(synthetic_flow_features)

        stats = orchestrator.get_stats()
        assert stats["alert_count"] == 3


# ---------------------------------------------------------------------------
# process_batch tests
# ---------------------------------------------------------------------------

class TestProcessBatch:
    """Tests for DetectionOrchestrator.process_batch."""

    def test_returns_dict(
        self,
        orchestrator: DetectionOrchestrator,
        sample_records: List[Dict[str, Any]],
    ) -> None:
        result = orchestrator.process_batch(sample_records)

        assert isinstance(result, dict)

    def test_empty_list_returns_zero_counts(
        self,
        orchestrator: DetectionOrchestrator,
    ) -> None:
        result = orchestrator.process_batch([])

        assert result["total_samples"] == 0
        assert result["total_alerts"] == 0
        assert result["anomaly_count"] == 0
        assert result["zero_day_count"] == 0

    def test_empty_list_returns_zero_time(
        self,
        orchestrator: DetectionOrchestrator,
    ) -> None:
        result = orchestrator.process_batch([])

        assert result["processing_time_seconds"] == 0.0
        assert result["throughput_samples_per_sec"] == 0.0

    def test_non_empty_batch_delegates_to_process_dataframe(
        self,
        orchestrator: DetectionOrchestrator,
        mock_engine: MagicMock,
        sample_records: List[Dict[str, Any]],
    ) -> None:
        orchestrator.process_batch(sample_records)

        mock_engine.detect_batch.assert_called_once()
        call_args = mock_engine.detect_batch.call_args[0][0]
        assert isinstance(call_args, pd.DataFrame)
        assert len(call_args) == len(sample_records)

    def test_batch_returns_expected_keys(
        self,
        orchestrator: DetectionOrchestrator,
        sample_records: List[Dict[str, Any]],
    ) -> None:
        result = orchestrator.process_batch(sample_records)

        expected_keys = {
            "total_samples", "total_alerts", "anomaly_count",
            "zero_day_count", "processing_time_seconds", "throughput_samples_per_sec",
        }
        assert expected_keys.issubset(result.keys())

    def test_return_details_forwarded(
        self,
        orchestrator: DetectionOrchestrator,
        sample_records: List[Dict[str, Any]],
    ) -> None:
        result = orchestrator.process_batch(sample_records, return_details=True)

        assert "predictions" in result


# ---------------------------------------------------------------------------
# process_single tests
# ---------------------------------------------------------------------------

class TestProcessSingle:
    """Tests for DetectionOrchestrator.process_single."""

    def test_returns_dict(
        self,
        orchestrator: DetectionOrchestrator,
        single_record: Dict[str, Any],
    ) -> None:
        result = orchestrator.process_single(single_record)

        assert isinstance(result, dict)

    def test_calls_engine_detect_batch(
        self,
        orchestrator: DetectionOrchestrator,
        mock_engine: MagicMock,
        single_record: Dict[str, Any],
    ) -> None:
        orchestrator.process_single(single_record)

        mock_engine.detect_batch.assert_called_once()

    def test_single_record_input_is_one_row_dataframe(
        self,
        orchestrator: DetectionOrchestrator,
        mock_engine: MagicMock,
        single_record: Dict[str, Any],
    ) -> None:
        orchestrator.process_single(single_record)

        call_args = mock_engine.detect_batch.call_args[0][0]
        assert isinstance(call_args, pd.DataFrame)
        assert len(call_args) == 1

    def test_returns_first_prediction(
        self,
        orchestrator: DetectionOrchestrator,
        single_record: Dict[str, Any],
    ) -> None:
        result = orchestrator.process_single(single_record)

        # The mock engine returns a 5-row DF; process_single extracts the first row
        assert result["sample_id"] == "sample_0"
        assert result["is_anomaly"] is True

    def test_empty_predictions_returns_empty_dict(
        self,
        orchestrator: DetectionOrchestrator,
        mock_engine: MagicMock,
        single_record: Dict[str, Any],
    ) -> None:
        empty_preds = pd.DataFrame({
            "is_anomaly": pd.Series(dtype=bool),
            "is_zero_day": pd.Series(dtype=bool),
            "severity": pd.Series(dtype=str),
            "predicted_class": pd.Series(dtype=str),
            "anomaly_score": pd.Series(dtype=float),
            "classification_confidence": pd.Series(dtype=float),
            "sample_id": pd.Series(dtype=str),
        })
        mock_engine.results_to_dataframe.return_value = empty_preds

        result = orchestrator.process_single(single_record)

        assert result == {}


# ---------------------------------------------------------------------------
# register_response_handler tests
# ---------------------------------------------------------------------------

class TestRegisterResponseHandler:
    """Tests for DetectionOrchestrator.register_response_handler."""

    def test_accepts_callable(
        self,
        orchestrator: DetectionOrchestrator,
    ) -> None:
        handler = lambda alert: None

        # Should not raise
        orchestrator.register_response_handler(handler)

    def test_increments_handler_count(
        self,
        orchestrator: DetectionOrchestrator,
    ) -> None:
        handler = lambda alert: None
        orchestrator.register_response_handler(handler)

        stats = orchestrator.get_stats()
        assert stats["response_handler_count"] == 1

    def test_multiple_handlers_registered(
        self,
        orchestrator: DetectionOrchestrator,
    ) -> None:
        h1 = lambda alert: None
        h2 = lambda alert: None
        h3 = lambda alert: None
        orchestrator.register_response_handler(h1)
        orchestrator.register_response_handler(h2)
        orchestrator.register_response_handler(h3)

        stats = orchestrator.get_stats()
        assert stats["response_handler_count"] == 3

    def test_handler_called_on_alerts(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        received: List[Dict[str, Any]] = []
        handler = lambda alert: received.append(alert)
        orchestrator.register_response_handler(handler)

        orchestrator.process_dataframe(synthetic_flow_features)

        assert len(received) == 3  # 3 alerts from mock engine

    def test_handler_receives_alert_dict(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        received: List[Dict[str, Any]] = []
        handler = lambda alert: received.append(alert)
        orchestrator.register_response_handler(handler)

        orchestrator.process_dataframe(synthetic_flow_features)

        for alert in received:
            assert isinstance(alert, dict)
            assert "is_anomaly" in alert or "is_zero_day" in alert

    def test_handler_exception_does_not_break_processing(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        def bad_handler(alert: Dict[str, Any]) -> None:
            raise ValueError("handler exploded")

        orchestrator.register_response_handler(bad_handler)

        # Should not raise
        result = orchestrator.process_dataframe(synthetic_flow_features)
        assert result["total_samples"] == len(synthetic_flow_features)


# ---------------------------------------------------------------------------
# get_stats tests
# ---------------------------------------------------------------------------

class TestGetStats:
    """Tests for DetectionOrchestrator.get_stats."""

    def test_returns_dict(self, orchestrator: DetectionOrchestrator) -> None:
        stats = orchestrator.get_stats()

        assert isinstance(stats, dict)

    def test_expected_keys_present(self, orchestrator: DetectionOrchestrator) -> None:
        stats = orchestrator.get_stats()

        expected_keys = {
            "processed_count", "alert_count", "alert_rate", "response_handler_count",
        }
        assert expected_keys.issubset(stats.keys())

    def test_initial_counts_are_zero(self, orchestrator: DetectionOrchestrator) -> None:
        stats = orchestrator.get_stats()

        assert stats["processed_count"] == 0
        assert stats["alert_count"] == 0

    def test_alert_rate_is_zero_initially(self, orchestrator: DetectionOrchestrator) -> None:
        stats = orchestrator.get_stats()

        assert stats["alert_rate"] == 0.0

    def test_response_handler_count_zero_initially(self, orchestrator: DetectionOrchestrator) -> None:
        stats = orchestrator.get_stats()

        assert stats["response_handler_count"] == 0

    def test_alert_rate_calculation(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        orchestrator.process_dataframe(synthetic_flow_features)
        stats = orchestrator.get_stats()

        expected_rate = round(3 / len(synthetic_flow_features), 4)
        assert stats["alert_rate"] == expected_rate

    def test_stats_update_after_multiple_calls(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        orchestrator.process_dataframe(synthetic_flow_features)
        orchestrator.process_dataframe(synthetic_flow_features)

        stats = orchestrator.get_stats()
        assert stats["processed_count"] == len(synthetic_flow_features) * 2
        assert stats["alert_count"] == 6


# ---------------------------------------------------------------------------
# reset_stats tests
# ---------------------------------------------------------------------------

class TestResetStats:
    """Tests for DetectionOrchestrator.reset_stats."""

    def test_resets_processed_count(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        orchestrator.process_dataframe(synthetic_flow_features)
        orchestrator.reset_stats()

        stats = orchestrator.get_stats()
        assert stats["processed_count"] == 0

    def test_resets_alert_count(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        orchestrator.process_dataframe(synthetic_flow_features)
        orchestrator.reset_stats()

        stats = orchestrator.get_stats()
        assert stats["alert_count"] == 0

    def test_resets_alert_rate(
        self,
        orchestrator: DetectionOrchestrator,
        synthetic_flow_features: pd.DataFrame,
    ) -> None:
        orchestrator.process_dataframe(synthetic_flow_features)
        orchestrator.reset_stats()

        stats = orchestrator.get_stats()
        assert stats["alert_rate"] == 0.0


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestConstructor:
    """Tests for DetectionOrchestrator __init__."""

    def test_creates_default_engine_and_logger(self) -> None:
        with patch("src.detection_engine.orchestrator.DetectionEngine") as MockEngine, \
             patch("src.detection_engine.orchestrator.AlertLogger") as MockLogger:
            orch = DetectionOrchestrator()

            MockEngine.assert_called_once()
            MockLogger.assert_called_once()
            assert orch.engine is MockEngine.return_value
            assert orch.alert_logger is MockLogger.return_value

    def test_uses_provided_engine(self, mock_engine: MagicMock) -> None:
        with patch("src.detection_engine.orchestrator.AlertLogger"):
            orch = DetectionOrchestrator(engine=mock_engine)

            assert orch.engine is mock_engine

    def test_uses_provided_logger(self, mock_alert_logger: MagicMock) -> None:
        with patch("src.detection_engine.orchestrator.DetectionEngine"):
            orch = DetectionOrchestrator(alert_logger=mock_alert_logger)

            assert orch.alert_logger is mock_alert_logger

    def test_initializes_internal_state(self, mock_engine: MagicMock, mock_alert_logger: MagicMock) -> None:
        orch = DetectionOrchestrator(engine=mock_engine, alert_logger=mock_alert_logger)

        assert orch._processed_count == 0
        assert orch._alert_count == 0
        assert orch._response_handlers == []
