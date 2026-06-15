"""Unit tests for src.alerting.csv_logger.CSVAlertLogger.

CSVAlertLogger writes alerts as CSV rows with column headers, supports
batch logging, read-back, and rotation.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import pytest

from src.alerting.csv_logger import CSV_COLUMNS, CSVAlertLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alert(
    alert_id: str = "alert-001",
    severity: str = "high",
    predicted_class: str = "DoS Hulk",
    anomaly_score: float = 0.85,
    classification_confidence: float = 0.92,
    is_zero_day: bool = False,
    sample_id: str = "sample-001",
    message: str = "Known attack detected",
    timestamp: str | None = None,
) -> dict:
    alert: dict = {
        "alert_id": alert_id,
        "severity": severity,
        "predicted_class": predicted_class,
        "anomaly_score": anomaly_score,
        "classification_confidence": classification_confidence,
        "is_zero_day": is_zero_day,
        "sample_id": sample_id,
        "message": message,
    }
    if timestamp is not None:
        alert["timestamp"] = timestamp
    return alert


def _read_csv_rows(csv_path: str) -> list[dict]:
    """Read all data rows from a CSV file (skipping header)."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _read_csv_raw(csv_path: str) -> list[list[str]]:
    """Read raw rows including header."""
    with open(csv_path) as f:
        return [row for row in csv.reader(f)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "csv_logs"
    d.mkdir()
    return d


@pytest.fixture()
def logger(log_dir: Path) -> CSVAlertLogger:
    return CSVAlertLogger(log_dir=str(log_dir))


# ===================================================================
# Tests — log
# ===================================================================

class TestLog:
    """Tests for the log method."""

    def test_creates_csv_file(self, logger: CSVAlertLogger, log_dir: Path) -> None:
        logger.log(_make_alert())
        csv_files = list(log_dir.glob("alerts_*.csv"))
        assert len(csv_files) == 1

    def test_writes_correct_headers(self, logger: CSVAlertLogger, log_dir: Path) -> None:
        logger.log(_make_alert())
        csv_files = list(log_dir.glob("alerts_*.csv"))
        raw = _read_csv_raw(csv_files[0])
        assert raw[0] == CSV_COLUMNS

    def test_writes_data_row(self, logger: CSVAlertLogger, log_dir: Path) -> None:
        logger.log(_make_alert(alert_id="test-001"))
        csv_files = list(log_dir.glob("alerts_*.csv"))
        rows = _read_csv_rows(csv_files[0])
        assert len(rows) == 1
        assert rows[0]["alert_id"] == "test-001"

    def test_adds_timestamp_when_missing(self, logger: CSVAlertLogger) -> None:
        alert = _make_alert()
        assert "timestamp" not in alert
        logger.log(alert)
        alerts = logger.read_alerts()
        assert "timestamp" in alerts[0]
        assert len(alerts[0]["timestamp"]) > 0

    def test_preserves_provided_timestamp(self, logger: CSVAlertLogger) -> None:
        ts = "2026-06-15T14:30:00"
        logger.log(_make_alert(timestamp=ts))
        alerts = logger.read_alerts()
        assert alerts[0]["timestamp"] == ts

    def test_multiple_logs_append(self, logger: CSVAlertLogger, log_dir: Path) -> None:
        for i in range(5):
            logger.log(_make_alert(alert_id=f"csv-{i:03d}"))
        csv_files = list(log_dir.glob("alerts_*.csv"))
        rows = _read_csv_rows(csv_files[0])
        assert len(rows) == 5

    def test_headers_written_once(self, logger: CSVAlertLogger, log_dir: Path) -> None:
        for i in range(3):
            logger.log(_make_alert())
        csv_files = list(log_dir.glob("alerts_*.csv"))
        raw = _read_csv_raw(csv_files[0])
        header_count = sum(1 for row in raw if row[0] == "timestamp")
        assert header_count == 1

    def test_row_count_matches_logs(self, logger: CSVAlertLogger, log_dir: Path) -> None:
        """Data rows = total lines minus header."""
        for i in range(10):
            logger.log(_make_alert(alert_id=f"r-{i}"))
        csv_files = list(log_dir.glob("alerts_*.csv"))
        with open(csv_files[0]) as f:
            all_lines = [l for l in f if l.strip()]
        # 1 header + 10 data
        assert len(all_lines) == 11

    def test_flush_after_write(self, logger: CSVAlertLogger, log_dir: Path) -> None:
        logger.log(_make_alert())
        csv_files = list(log_dir.glob("alerts_*.csv"))
        assert csv_files[0].stat().st_size > 0

    def test_returns_none(self, logger: CSVAlertLogger) -> None:
        result = logger.log(_make_alert())
        assert result is None

    def test_all_csv_columns_in_header(self, logger: CSVAlertLogger, log_dir: Path) -> None:
        logger.log(_make_alert())
        csv_files = list(log_dir.glob("alerts_*.csv"))
        raw = _read_csv_raw(csv_files[0])
        assert raw[0] == [
            "timestamp",
            "alert_id",
            "severity",
            "predicted_class",
            "anomaly_score",
            "classification_confidence",
            "is_zero_day",
            "sample_id",
            "message",
        ]


# ===================================================================
# Tests — log_batch
# ===================================================================

class TestLogBatch:
    """Tests for the log_batch method."""

    def test_returns_count(self, logger: CSVAlertLogger) -> None:
        alerts = [_make_alert(alert_id=f"b-{i}") for i in range(7)]
        count = logger.log_batch(alerts)
        assert count == 7

    def test_empty_batch(self, logger: CSVAlertLogger) -> None:
        count = logger.log_batch([])
        assert count == 0

    def test_batch_all_logged(self, logger: CSVAlertLogger) -> None:
        alerts = [_make_alert(alert_id=f"b-{i}") for i in range(10)]
        logger.log_batch(alerts)
        loaded = logger.read_alerts()
        assert len(loaded) == 10

    def test_batch_values_in_file(self, logger: CSVAlertLogger) -> None:
        alerts = [_make_alert(alert_id=f"bv-{i}", severity="critical") for i in range(3)]
        logger.log_batch(alerts)
        loaded = logger.read_alerts()
        for a in loaded:
            assert a["severity"] == "critical"


# ===================================================================
# Tests — read_alerts
# ===================================================================

class TestReadAlerts:
    """Tests for the read_alerts method."""

    def test_empty_file(self, logger: CSVAlertLogger) -> None:
        alerts = logger.read_alerts()
        assert alerts == []

    def test_returns_list_of_dicts(self, logger: CSVAlertLogger) -> None:
        logger.log(_make_alert())
        alerts = logger.read_alerts()
        assert isinstance(alerts, list)
        assert isinstance(alerts[0], dict)

    def test_limit_parameter(self, logger: CSVAlertLogger) -> None:
        for i in range(10):
            logger.log(_make_alert(alert_id=f"lim-{i}"))
        limited = logger.read_alerts(limit=3)
        assert len(limited) == 3

    def test_limit_larger_than_available(self, logger: CSVAlertLogger) -> None:
        for i in range(2):
            logger.log(_make_alert())
        result = logger.read_alerts(limit=100)
        assert len(result) == 2

    def test_limit_none_returns_all(self, logger: CSVAlertLogger) -> None:
        for i in range(5):
            logger.log(_make_alert())
        result = logger.read_alerts(limit=None)
        assert len(result) == 5

    def test_read_after_rotation(self, logger: CSVAlertLogger) -> None:
        logger.log(_make_alert())
        logger._rotate()
        logger.log(_make_alert(alert_id="after-rotate"))
        alerts = logger.read_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_id"] == "after-rotate"

    def test_all_fields_preserved(self, logger: CSVAlertLogger) -> None:
        alert = _make_alert(
            alert_id="full-001",
            severity="critical",
            predicted_class="ZERO_DAY",
            anomaly_score=0.99,
            classification_confidence=0.45,
            is_zero_day=True,
            sample_id="s-full",
            message="Zero-day detected",
            timestamp="2026-03-15T08:00:00",
        )
        logger.log(alert)
        loaded = logger.read_alerts()
        assert loaded[0]["alert_id"] == "full-001"
        assert loaded[0]["severity"] == "critical"
        assert loaded[0]["anomaly_score"] == "0.99"  # CSV reads as strings
        assert loaded[0]["is_zero_day"] == "True"

    def test_limit_zero(self, logger: CSVAlertLogger) -> None:
        """limit=0 is falsy in the source, so all alerts are returned."""
        logger.log(_make_alert())
        result = logger.read_alerts(limit=0)
        assert len(result) == 1


# ===================================================================
# Tests — get_stats
# ===================================================================

class TestGetStats:
    """Tests for the get_stats method."""

    def test_returns_expected_keys(self, logger: CSVAlertLogger) -> None:
        stats = logger.get_stats()
        assert "current_file" in stats
        assert "write_count" in stats
        assert "max_file_size_mb" in stats

    def test_write_count_increments(self, logger: CSVAlertLogger) -> None:
        stats_before = logger.get_stats()
        assert stats_before["write_count"] == 0
        logger.log(_make_alert())
        logger.log(_make_alert())
        stats_after = logger.get_stats()
        assert stats_after["write_count"] == 2

    def test_current_file_path(self, logger: CSVAlertLogger) -> None:
        stats = logger.get_stats()
        assert stats["current_file"] is not None
        assert stats["current_file"].endswith(".csv")

    def test_max_file_size_mb_default(self, logger: CSVAlertLogger) -> None:
        stats = logger.get_stats()
        assert stats["max_file_size_mb"] == 50

    def test_custom_max_file_size(self, log_dir: Path) -> None:
        custom = CSVAlertLogger(log_dir=str(log_dir), max_file_size_mb=25)
        stats = custom.get_stats()
        assert stats["max_file_size_mb"] == 25
        custom.close()

    def test_write_count_after_batch(self, logger: CSVAlertLogger) -> None:
        alerts = [_make_alert(alert_id=f"s-{i}") for i in range(5)]
        logger.log_batch(alerts)
        stats = logger.get_stats()
        assert stats["write_count"] == 5


# ===================================================================
# Tests — file rotation
# ===================================================================

class TestRotation:
    """Tests for file rotation behavior."""

    def test_rotation_creates_new_file(self, logger: CSVAlertLogger) -> None:
        logger.log(_make_alert())
        old_file = logger._current_file
        logger._rotate()
        # Old file is renamed with timestamp suffix
        rotated = list(Path(logger.log_dir).glob("alerts_*.csv.*"))
        assert len(rotated) >= 1
        # New file exists with same date-based name
        assert logger._current_file is not None
        assert os.path.exists(logger._current_file)

    def test_rotated_file_renamed(self, logger: CSVAlertLogger, log_dir: Path) -> None:
        logger.log(_make_alert())
        old_file = logger._current_file
        logger._rotate()
        rotated = list(log_dir.glob("alerts_*.csv.*"))
        assert len(rotated) >= 1

    def test_rotation_preserves_data(self, logger: CSVAlertLogger, log_dir: Path) -> None:
        logger.log(_make_alert(alert_id="pre-rotate"))
        logger._rotate()
        rotated_files = list(log_dir.glob("alerts_*.csv.*"))
        assert len(rotated_files) >= 1
        rows = _read_csv_rows(str(rotated_files[0]))
        assert rows[0]["alert_id"] == "pre-rotate"

    def test_new_file_has_headers(self, logger: CSVAlertLogger) -> None:
        logger.log(_make_alert())
        logger._rotate()
        # New file should have headers
        raw = _read_csv_raw(logger._current_file)
        assert raw[0] == CSV_COLUMNS

    def test_rotation_triggered_by_size(self, logger: CSVAlertLogger, log_dir: Path) -> None:
        """Setting max_file_size_mb to 0 should trigger rotation on next write."""
        logger.max_file_size_mb = 0
        logger.log(_make_alert())
        # Rotation should have been triggered
        rotated = list(log_dir.glob("alerts_*.csv.*"))
        assert len(rotated) >= 1


# ===================================================================
# Tests — close
# ===================================================================

class TestClose:
    """Tests for the close method."""

    def test_close_sets_handle_to_none(self, logger: CSVAlertLogger) -> None:
        logger.close()
        assert logger._file_handle is None

    def test_close_idempotent(self, logger: CSVAlertLogger) -> None:
        logger.close()
        logger.close()  # should not raise

    def test_close_after_writes(self, logger: CSVAlertLogger) -> None:
        logger.log(_make_alert())
        logger.close()
        assert logger._file_handle is None


# ===================================================================
# Tests — edge cases
# ===================================================================

class TestEdgeCases:
    """Edge-case and robustness tests."""

    def test_log_empty_dict(self, logger: CSVAlertLogger) -> None:
        logger.log({})
        alerts = logger.read_alerts()
        assert len(alerts) == 1
        assert "timestamp" in alerts[0]

    def test_log_dict_with_extra_fields_ignored(self, logger: CSVAlertLogger) -> None:
        """Extra keys not in CSV_COLUMNS should be silently ignored."""
        alert = _make_alert()
        alert["extra_key"] = "should_not_appear"
        logger.log(alert)
        alerts = logger.read_alerts()
        assert "extra_key" not in alerts[0]

    def test_log_dict_with_missing_fields(self, logger: CSVAlertLogger) -> None:
        """Missing keys should default to empty string."""
        alert = {"alert_id": "partial-001"}
        logger.log(alert)
        alerts = logger.read_alerts()
        assert alerts[0]["alert_id"] == "partial-001"
        assert alerts[0]["severity"] == ""

    def test_nonexistent_log_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "new" / "dir"
        log = CSVAlertLogger(log_dir=str(target))
        assert target.exists()
        log.close()

    def test_unicode_in_message(self, logger: CSVAlertLogger) -> None:
        logger.log(_make_alert(message="Port scan from 10.0.0.1 — SYN flood"))
        loaded = logger.read_alerts()
        assert "SYN flood" in loaded[0]["message"]

    def test_large_message(self, logger: CSVAlertLogger) -> None:
        long_msg = "A" * 10_000
        logger.log(_make_alert(message=long_msg))
        loaded = logger.read_alerts()
        assert loaded[0]["message"] == long_msg

    def test_bool_values_csv_roundtrip(self, logger: CSVAlertLogger) -> None:
        logger.log(_make_alert(is_zero_day=True))
        logger.log(_make_alert(is_zero_day=False))
        loaded = logger.read_alerts()
        # CSV reads everything as strings
        assert loaded[0]["is_zero_day"] == "True"
        assert loaded[1]["is_zero_day"] == "False"

    def test_float_values_csv_roundtrip(self, logger: CSVAlertLogger) -> None:
        logger.log(_make_alert(anomaly_score=0.123456))
        loaded = logger.read_alerts()
        # CSV preserves the string representation
        assert loaded[0]["anomaly_score"] == "0.123456"

    def test_sequential_writes_dont_duplicate_header(self, logger: CSVAlertLogger, log_dir: Path) -> None:
        """Multiple log() calls should only produce one header row."""
        for i in range(20):
            logger.log(_make_alert(alert_id=f"dup-{i}"))
        csv_files = list(log_dir.glob("alerts_*.csv"))
        raw = _read_csv_raw(csv_files[0])
        header_count = sum(1 for row in raw if row and row[0] == "timestamp")
        assert header_count == 1

    def test_read_alerts_ordering(self, logger: CSVAlertLogger) -> None:
        """Alerts should be returned in insertion order."""
        for i in range(10):
            logger.log(_make_alert(alert_id=f"ord-{i:03d}"))
        loaded = logger.read_alerts()
        ids = [a["alert_id"] for a in loaded]
        assert ids == [f"ord-{i:03d}" for i in range(10)]
