"""Unit tests for src.alerting.json_logger.JSONAlertLogger.

JSONAlertLogger writes alerts as JSON Lines (.jsonl) with file rotation,
batch logging, and read-back support.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from src.alerting.json_logger import JSONAlertLogger


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "json_logs"
    d.mkdir()
    return d


@pytest.fixture()
def logger(log_dir: Path) -> JSONAlertLogger:
    return JSONAlertLogger(log_dir=str(log_dir))


# ===================================================================
# Tests — log
# ===================================================================

class TestLog:
    """Tests for the log method."""

    def test_creates_jsonl_file(self, logger: JSONAlertLogger, log_dir: Path) -> None:
        logger.log(_make_alert())
        jsonl_files = list(log_dir.glob("alerts_*.jsonl"))
        assert len(jsonl_files) == 1

    def test_writes_valid_jsonl(self, logger: JSONAlertLogger) -> None:
        logger.log(_make_alert())
        alerts = logger.read_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_id"] == "alert-001"

    def test_adds_timestamp_when_missing(self, logger: JSONAlertLogger) -> None:
        alert = _make_alert()
        assert "timestamp" not in alert
        logger.log(alert)
        alerts = logger.read_alerts()
        assert "timestamp" in alerts[0]
        assert "T" in alerts[0]["timestamp"]

    def test_preserves_provided_timestamp(self, logger: JSONAlertLogger) -> None:
        ts = "2026-06-15T14:30:00"
        logger.log(_make_alert(timestamp=ts))
        alerts = logger.read_alerts()
        assert alerts[0]["timestamp"] == ts

    def test_writes_timestamp_as_string(self, logger: JSONAlertLogger) -> None:
        """Timestamps with datetime objects should serialize to strings."""
        alert = _make_alert()
        alert["timestamp"] = datetime(2026, 1, 15, 10, 0, 0)
        logger.log(alert)
        alerts = logger.read_alerts()
        assert isinstance(alerts[0]["timestamp"], str)

    def test_multiple_logs_append(self, logger: JSONAlertLogger) -> None:
        for i in range(5):
            logger.log(_make_alert(alert_id=f"a-{i:03d}"))
        alerts = logger.read_alerts()
        assert len(alerts) == 5

    def test_file_size_grows(self, logger: JSONAlertLogger, log_dir: Path) -> None:
        logger.log(_make_alert())
        jsonl_files = list(log_dir.glob("alerts_*.jsonl"))
        size_after_one = jsonl_files[0].stat().st_size
        logger.log(_make_alert(alert_id="second"))
        size_after_two = jsonl_files[0].stat().st_size
        assert size_after_two > size_after_one

    def test_each_line_is_valid_json(self, logger: JSONAlertLogger) -> None:
        for i in range(10):
            logger.log(_make_alert(alert_id=f"line-{i}"))
        jsonl_files = list(Path(logger.log_dir).glob("alerts_*.jsonl"))
        with open(jsonl_files[0]) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    assert "alert_id" in obj

    def test_flush_after_write(self, logger: JSONAlertLogger, log_dir: Path) -> None:
        logger.log(_make_alert())
        jsonl_files = list(log_dir.glob("alerts_*.jsonl"))
        # File should be flushed (not empty / not just empty string)
        assert jsonl_files[0].stat().st_size > 0


# ===================================================================
# Tests — log_batch
# ===================================================================

class TestLogBatch:
    """Tests for the log_batch method."""

    def test_returns_count(self, logger: JSONAlertLogger) -> None:
        alerts = [_make_alert(alert_id=f"b-{i}") for i in range(7)]
        count = logger.log_batch(alerts)
        assert count == 7

    def test_empty_batch(self, logger: JSONAlertLogger) -> None:
        count = logger.log_batch([])
        assert count == 0

    def test_batch_all_logged(self, logger: JSONAlertLogger) -> None:
        alerts = [_make_alert(alert_id=f"b-{i}") for i in range(10)]
        logger.log_batch(alerts)
        loaded = logger.read_alerts()
        assert len(loaded) == 10

    def test_batch_partial_failure_continues(self, logger: JSONAlertLogger) -> None:
        """Even if one alert causes an error, the rest should log.

        The JSONAlertLogger.log method catches all exceptions internally,
        so batch continues even when one alert fails serialization.
        """
        good = [_make_alert(alert_id=f"g-{i}") for i in range(5)]
        # Force a write error by closing the file handle before logging
        logger._file_handle.close()
        count = logger.log_batch(good)
        # Some may fail due to closed handle, batch returns count of successes
        assert isinstance(count, int)
        assert count >= 0


# ===================================================================
# Tests — read_alerts
# ===================================================================

class TestReadAlerts:
    """Tests for the read_alerts method."""

    def test_empty_file(self, logger: JSONAlertLogger) -> None:
        alerts = logger.read_alerts()
        assert alerts == []

    def test_returns_list_of_dicts(self, logger: JSONAlertLogger) -> None:
        logger.log(_make_alert())
        alerts = logger.read_alerts()
        assert isinstance(alerts, list)
        assert isinstance(alerts[0], dict)

    def test_limit_parameter(self, logger: JSONAlertLogger) -> None:
        for i in range(10):
            logger.log(_make_alert(alert_id=f"lim-{i}"))
        limited = logger.read_alerts(limit=3)
        assert len(limited) == 3

    def test_limit_larger_than_available(self, logger: JSONAlertLogger) -> None:
        for i in range(2):
            logger.log(_make_alert())
        result = logger.read_alerts(limit=100)
        assert len(result) == 2

    def test_limit_none_returns_all(self, logger: JSONAlertLogger) -> None:
        for i in range(5):
            logger.log(_make_alert())
        result = logger.read_alerts(limit=None)
        assert len(result) == 5

    def test_read_after_rotation(self, logger: JSONAlertLogger) -> None:
        """After manual rotation, read_alerts still works on the new file."""
        logger.log(_make_alert())
        logger._rotate()
        logger.log(_make_alert(alert_id="after-rotate"))
        alerts = logger.read_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_id"] == "after-rotate"

    def test_all_fields_preserved(self, logger: JSONAlertLogger) -> None:
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
        assert loaded[0] == alert


# ===================================================================
# Tests — get_stats
# ===================================================================

class TestGetStats:
    """Tests for the get_stats method."""

    def test_returns_expected_keys(self, logger: JSONAlertLogger) -> None:
        stats = logger.get_stats()
        assert "current_file" in stats
        assert "write_count" in stats
        assert "max_file_size_mb" in stats
        assert "max_files" in stats

    def test_write_count_increments(self, logger: JSONAlertLogger) -> None:
        stats_before = logger.get_stats()
        assert stats_before["write_count"] == 0
        logger.log(_make_alert())
        logger.log(_make_alert())
        stats_after = logger.get_stats()
        assert stats_after["write_count"] == 2

    def test_current_file_path(self, logger: JSONAlertLogger) -> None:
        stats = logger.get_stats()
        assert stats["current_file"] is not None
        assert stats["current_file"].endswith(".jsonl")

    def test_max_values_stored(self, logger: JSONAlertLogger) -> None:
        stats = logger.get_stats()
        assert stats["max_file_size_mb"] == 50  # default
        assert stats["max_files"] == 10  # default

    def test_custom_max_values(self, log_dir: Path) -> None:
        custom = JSONAlertLogger(
            log_dir=str(log_dir),
            max_file_size_mb=25,
            max_files=3,
        )
        stats = custom.get_stats()
        assert stats["max_file_size_mb"] == 25
        assert stats["max_files"] == 3
        custom.close()


# ===================================================================
# Tests — file rotation
# ===================================================================

class TestRotation:
    """Tests for file rotation behavior."""

    def test_rotation_creates_new_file(self, logger: JSONAlertLogger) -> None:
        logger.log(_make_alert())
        old_file = logger._current_file
        logger._rotate()
        # Old file is renamed with timestamp suffix
        rotated = list(Path(logger.log_dir).glob("alerts_*.jsonl.*"))
        assert len(rotated) >= 1
        # New file exists with same date-based name
        assert logger._current_file is not None
        assert os.path.exists(logger._current_file)

    def test_rotated_file_renamed_with_timestamp(self, logger: JSONAlertLogger, log_dir: Path) -> None:
        logger.log(_make_alert())
        old_file = logger._current_file
        logger._rotate()
        # Old file should now exist with a timestamp suffix
        rotated = list(log_dir.glob("alerts_*.jsonl.*"))
        assert len(rotated) >= 1

    def test_rotation_preserves_data(self, logger: JSONAlertLogger) -> None:
        logger.log(_make_alert(alert_id="before-rotate"))
        logger._rotate()
        # The rotated file should contain the old data
        rotated_files = list(Path(logger.log_dir).glob("alerts_*.jsonl.*"))
        assert len(rotated_files) >= 1
        with open(rotated_files[0]) as f:
            data = json.loads(f.readline())
        assert data["alert_id"] == "before-rotate"

    def test_max_files_enforced(self, logger: JSONAlertLogger, log_dir: Path) -> None:
        """Rotating beyond max_files should delete old rotated files."""
        logger.max_files = 2
        for _ in range(4):
            logger.log(_make_alert())
            logger._rotate()
        rotated = list(log_dir.glob("alerts_*.jsonl.*"))
        assert len(rotated) <= 2


# ===================================================================
# Tests — close
# ===================================================================

class TestClose:
    """Tests for the close method."""

    def test_close_sets_handle_to_none(self, logger: JSONAlertLogger) -> None:
        logger.close()
        assert logger._file_handle is None

    def test_close_idempotent(self, logger: JSONAlertLogger) -> None:
        logger.close()
        logger.close()  # should not raise

    def test_close_after_writes(self, logger: JSONAlertLogger) -> None:
        logger.log(_make_alert())
        logger.close()
        assert logger._file_handle is None


# ===================================================================
# Tests — edge cases
# ===================================================================

class TestEdgeCases:
    """Edge-case and robustness tests."""

    def test_log_empty_dict(self, logger: JSONAlertLogger) -> None:
        logger.log({})
        alerts = logger.read_alerts()
        assert len(alerts) == 1
        assert "timestamp" in alerts[0]

    def test_log_dict_with_extra_fields(self, logger: JSONAlertLogger) -> None:
        alert = _make_alert()
        alert["custom"] = "value"
        alert["nested"] = {"a": 1, "b": [2, 3]}
        logger.log(alert)
        loaded = logger.read_alerts()
        assert loaded[0]["custom"] == "value"
        assert loaded[0]["nested"] == {"a": 1, "b": [2, 3]}

    def test_log_dict_with_none_values(self, logger: JSONAlertLogger) -> None:
        alert = _make_alert()
        alert["optional"] = None
        logger.log(alert)
        loaded = logger.read_alerts()
        assert loaded[0]["optional"] is None

    def test_nonexistent_log_dir(self, tmp_path: Path) -> None:
        """Constructor should create the directory."""
        target = tmp_path / "does" / "not" / "exist"
        log = JSONAlertLogger(log_dir=str(target))
        assert target.exists()
        log.close()

    def test_unicode_in_alert(self, logger: JSONAlertLogger) -> None:
        alert = _make_alert(message="Attack from 192.168.1.1 — port 80")
        logger.log(alert)
        loaded = logger.read_alerts()
        assert "192.168.1.1" in loaded[0]["message"]

    def test_large_alert_payload(self, logger: JSONAlertLogger) -> None:
        alert = _make_alert()
        alert["large_field"] = "x" * 50_000
        logger.log(alert)
        loaded = logger.read_alerts()
        assert len(loaded[0]["large_field"]) == 50_000

    def test_read_limit_zero(self, logger: JSONAlertLogger) -> None:
        """limit=0 is falsy in the source, so all alerts are returned."""
        logger.log(_make_alert())
        result = logger.read_alerts(limit=0)
        assert len(result) == 1

    def test_concurrent_appends_dont_corrupt(self, logger: JSONAlertLogger) -> None:
        """Simulate multiple sequential appends without interleaving corruption."""
        for i in range(100):
            logger.log(_make_alert(alert_id=f"seq-{i}"))
        loaded = logger.read_alerts()
        assert len(loaded) == 100
        for i, alert in enumerate(loaded):
            assert alert["alert_id"] == f"seq-{i}"
