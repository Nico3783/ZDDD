"""Unit tests for src.alerting.logger.AlertLogger.

AlertLogger writes alerts to JSON and/or CSV files, provides load/summary
utilities, and supports archival of old log files.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.alerting.logger import AlertLogger


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
) -> Dict[str, Any]:
    """Build a minimal alert dictionary."""
    alert: Dict[str, Any] = {
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


class _FakeAlert:
    """Mimics an Alert dataclass / object with attribute access."""

    def __init__(self) -> None:
        self.alert_id = "alert-obj-001"
        self.severity = "critical"
        self.predicted_class = "ZERO_DAY"
        self.anomaly_score = 0.97
        self.classification_confidence = 0.35
        self.is_zero_day = True
        self.sample_id = "sample-obj-001"
        self.message = "Possible zero-day exploit"
        self.details = {"src_ip": "10.0.0.1", "dst_port": 443}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def alert_dir(tmp_path: Path) -> Path:
    """Isolated directory for alert output."""
    d = tmp_path / "alerts"
    d.mkdir()
    return d


@pytest.fixture()
def json_logger(alert_dir: Path) -> AlertLogger:
    """AlertLogger in JSON-only mode."""
    return AlertLogger(log_dir=str(alert_dir), log_format="json")


@pytest.fixture()
def csv_logger(alert_dir: Path) -> AlertLogger:
    """AlertLogger in CSV-only mode."""
    return AlertLogger(log_dir=str(alert_dir), log_format="csv")


@pytest.fixture()
def dual_logger(alert_dir: Path) -> AlertLogger:
    """AlertLogger writing both JSON and CSV."""
    return AlertLogger(log_dir=str(alert_dir), log_format="both")


# ===================================================================
# Tests — log_alert
# ===================================================================

class TestLogAlert:
    """Tests for the log_alert method."""

    def test_json_log_creates_file(self, json_logger: AlertLogger, alert_dir: Path) -> None:
        json_logger.log_alert(_make_alert())
        json_files = list(alert_dir.glob("alerts_*.json"))
        assert len(json_files) == 1

    def test_json_log_writes_valid_jsonl(self, json_logger: AlertLogger) -> None:
        json_logger.log_alert(_make_alert())
        alerts = json_logger.load_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_id"] == "alert-001"
        assert alerts[0]["severity"] == "high"

    def test_json_log_adds_timestamp_when_missing(self, json_logger: AlertLogger) -> None:
        alert = _make_alert()
        assert "timestamp" not in alert
        json_logger.log_alert(alert)
        alerts = json_logger.load_alerts()
        assert "timestamp" in alerts[0]
        assert "T" in alerts[0]["timestamp"]

    def test_json_log_preserves_provided_timestamp(self, json_logger: AlertLogger) -> None:
        ts = "2026-06-01T12:00:00"
        json_logger.log_alert(_make_alert(timestamp=ts))
        alerts = json_logger.load_alerts()
        assert alerts[0]["timestamp"] == ts

    def test_csv_log_creates_file(self, csv_logger: AlertLogger, alert_dir: Path) -> None:
        csv_logger.log_alert(_make_alert())
        csv_files = list(alert_dir.glob("alerts_*.csv"))
        assert len(csv_files) == 1

    def test_csv_log_writes_row_with_headers(self, csv_logger: AlertLogger, alert_dir: Path) -> None:
        csv_logger.log_alert(_make_alert())
        csv_files = list(alert_dir.glob("alerts_*.csv"))
        with open(csv_files[0]) as f:
            reader = csv.reader(f)
            rows = list(reader)
        # header + 1 data row
        assert len(rows) == 2
        assert rows[0][0] == "timestamp"
        assert rows[1][1] == "alert-001"  # alert_id column

    def test_dual_log_creates_both_files(self, dual_logger: AlertLogger, alert_dir: Path) -> None:
        dual_logger.log_alert(_make_alert())
        json_files = list(alert_dir.glob("alerts_*.json"))
        csv_files = list(alert_dir.glob("alerts_*.csv"))
        assert len(json_files) == 1
        assert len(csv_files) == 1

    def test_log_object_with_attributes(self, json_logger: AlertLogger) -> None:
        json_logger.log_alert(_FakeAlert())
        alerts = json_logger.load_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_id"] == "alert-obj-001"
        assert alerts[0]["is_zero_day"] is True
        assert alerts[0]["details"] == {"src_ip": "10.0.0.1", "dst_port": 443}

    def test_log_object_adds_timestamp(self, json_logger: AlertLogger) -> None:
        json_logger.log_alert(_FakeAlert())
        alerts = json_logger.load_alerts()
        assert "timestamp" in alerts[0]

    def test_multiple_alerts_accumulate(self, json_logger: AlertLogger) -> None:
        for i in range(5):
            json_logger.log_alert(_make_alert(alert_id=f"alert-{i:03d}"))
        alerts = json_logger.load_alerts()
        assert len(alerts) == 5
        assert [a["alert_id"] for a in alerts] == [f"alert-{i:03d}" for i in range(5)]

    def test_csv_multiple_alerts_accumulate(self, csv_logger: AlertLogger, alert_dir: Path) -> None:
        for i in range(5):
            csv_logger.log_alert(_make_alert(alert_id=f"alert-{i:03d}"))
        csv_files = list(alert_dir.glob("alerts_*.csv"))
        with open(csv_files[0]) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 5

    def test_log_returns_none(self, json_logger: AlertLogger) -> None:
        result = json_logger.log_alert(_make_alert())
        assert result is None


# ===================================================================
# Tests — log_alerts (batch)
# ===================================================================

class TestLogAlerts:
    """Tests for the log_alerts batch method."""

    def test_batch_logs_all_alerts(self, json_logger: AlertLogger) -> None:
        alerts = [_make_alert(alert_id=f"batch-{i:03d}") for i in range(10)]
        json_logger.log_alerts(alerts)
        loaded = json_logger.load_alerts()
        assert len(loaded) == 10

    def test_batch_empty_list(self, json_logger: AlertLogger) -> None:
        json_logger.log_alerts([])
        loaded = json_logger.load_alerts()
        assert len(loaded) == 0

    def test_batch_mixed_objects_and_dicts(self, json_logger: AlertLogger) -> None:
        items: List[Any] = [_make_alert(alert_id="d1"), _FakeAlert()]
        json_logger.log_alerts(items)
        loaded = json_logger.load_alerts()
        assert len(loaded) == 2


# ===================================================================
# Tests — load_alerts
# ===================================================================

class TestLoadAlerts:
    """Tests for the load_alerts method."""

    def test_load_empty_file(self, json_logger: AlertLogger) -> None:
        alerts = json_logger.load_alerts()
        assert alerts == []

    def test_load_after_multiple_writes(self, json_logger: AlertLogger) -> None:
        for i in range(3):
            json_logger.log_alert(_make_alert(alert_id=f"id-{i}"))
        loaded = json_logger.load_alerts()
        assert len(loaded) == 3
        assert loaded[0]["alert_id"] == "id-0"
        assert loaded[2]["alert_id"] == "id-2"

    def test_load_preserves_all_fields(self, json_logger: AlertLogger) -> None:
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
        json_logger.log_alert(alert)
        loaded = json_logger.load_alerts()
        assert loaded[0] == alert

    def test_load_handles_blank_lines(self, json_logger: AlertLogger) -> None:
        """Manually inject blank lines and ensure they're skipped."""
        json_logger.log_alert(_make_alert())
        # Manually append junk
        date_str = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
        fpath = os.path.join(json_logger.log_dir, f"alerts_{date_str}.json")
        with open(fpath, "a") as f:
            f.write("\n\n\n")
        loaded = json_logger.load_alerts()
        assert len(loaded) == 1


# ===================================================================
# Tests — get_alert_summary
# ===================================================================

class TestGetAlertSummary:
    """Tests for the get_alert_summary method."""

    def test_empty_summary(self, json_logger: AlertLogger) -> None:
        summary = json_logger.get_alert_summary()
        assert summary["total"] == 0
        assert summary["by_severity"] == {}
        assert summary["by_class"] == {}

    def test_summary_total(self, json_logger: AlertLogger) -> None:
        for i in range(7):
            json_logger.log_alert(_make_alert(alert_id=f"s-{i}"))
        summary = json_logger.get_alert_summary()
        assert summary["total"] == 7

    def test_summary_by_severity(self, json_logger: AlertLogger) -> None:
        json_logger.log_alert(_make_alert(severity="critical"))
        json_logger.log_alert(_make_alert(severity="high"))
        json_logger.log_alert(_make_alert(severity="high"))
        json_logger.log_alert(_make_alert(severity="low"))
        summary = json_logger.get_alert_summary()
        assert summary["by_severity"]["critical"] == 1
        assert summary["by_severity"]["high"] == 2
        assert summary["by_severity"]["low"] == 1

    def test_summary_by_class(self, json_logger: AlertLogger) -> None:
        json_logger.log_alert(_make_alert(predicted_class="DoS Hulk"))
        json_logger.log_alert(_make_alert(predicted_class="DoS Hulk"))
        json_logger.log_alert(_make_alert(predicted_class="BENIGN"))
        summary = json_logger.get_alert_summary()
        assert summary["by_class"]["DoS Hulk"] == 2
        assert summary["by_class"]["BENIGN"] == 1

    def test_summary_zero_day_count(self, json_logger: AlertLogger) -> None:
        json_logger.log_alert(_make_alert(is_zero_day=True))
        json_logger.log_alert(_make_alert(is_zero_day=True))
        json_logger.log_alert(_make_alert(is_zero_day=False))
        summary = json_logger.get_alert_summary()
        assert summary["zero_day_count"] == 2


# ===================================================================
# Tests — archive_alerts
# ===================================================================

class TestArchiveAlerts:
    """Tests for the archive_alerts method."""

    def test_archive_json_moves_file(self, json_logger: AlertLogger, alert_dir: Path) -> None:
        json_logger.log_alert(_make_alert())
        archive_dir = json_logger.archive_alerts()
        # Original file should be gone (reinitialized)
        archived = list(Path(archive_dir).glob("alerts_*.json"))
        assert len(archived) == 1

    def test_archive_returns_directory_path(self, json_logger: AlertLogger) -> None:
        result = json_logger.archive_alerts()
        assert isinstance(result, str)
        assert os.path.isdir(result)

    def test_archive_creates_archived_subdir(self, json_logger: AlertLogger, alert_dir: Path) -> None:
        json_logger.archive_alerts()
        archived_dir = alert_dir / "archived"
        assert archived_dir.exists()

    def test_archive_custom_directory(self, json_logger: AlertLogger, tmp_path: Path) -> None:
        json_logger.log_alert(_make_alert())
        custom = str(tmp_path / "my_archive")
        result = json_logger.archive_alerts(archive_dir=custom)
        assert result == custom
        assert os.path.isdir(custom)

    def test_archive_csv_file(self, csv_logger: AlertLogger, alert_dir: Path) -> None:
        csv_logger.log_alert(_make_alert())
        csv_logger.close()
        archive_dir = csv_logger.archive_alerts()
        archived = list(Path(archive_dir).glob("alerts_*.csv"))
        assert len(archived) == 1

    def test_archive_both_formats(self, dual_logger: AlertLogger, alert_dir: Path) -> None:
        dual_logger.log_alert(_make_alert())
        dual_logger.close()
        archive_dir = dual_logger.archive_alerts()
        archived_json = list(Path(archive_dir).glob("alerts_*.json"))
        archived_csv = list(Path(archive_dir).glob("alerts_*.csv"))
        assert len(archived_json) == 1
        assert len(archived_csv) == 1

    def test_archive_empty_log(self, json_logger: AlertLogger) -> None:
        """Archiving with no data should still work and return path."""
        result = json_logger.archive_alerts()
        assert os.path.isdir(result)


# ===================================================================
# Tests — close
# ===================================================================

class TestClose:
    """Tests for the close method."""

    def test_close_csv_handle(self, csv_logger: AlertLogger) -> None:
        csv_logger.log_alert(_make_alert())
        csv_logger.close()
        assert csv_logger._csv_file_handle is None

    def test_close_idempotent(self, csv_logger: AlertLogger) -> None:
        csv_logger.close()
        csv_logger.close()  # should not raise
        assert csv_logger._csv_file_handle is None

    def test_close_json_logger(self, json_logger: AlertLogger) -> None:
        json_logger.close()  # no CSV handle, should be a no-op
        assert json_logger._csv_file_handle is None


# ===================================================================
# Tests — date-based file naming
# ===================================================================

class TestFileNaming:
    """Tests for date-based log file naming."""

    def test_json_file_name_contains_date(self, json_logger: AlertLogger) -> None:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        assert date_str in json_logger._json_file

    def test_csv_file_name_contains_date(self, csv_logger: AlertLogger) -> None:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        assert date_str in csv_logger._csv_file

    def test_log_dir_created(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "alerts"
        logger = AlertLogger(log_dir=str(target), log_format="json")
        assert target.exists()
        logger.close()


# ===================================================================
# Tests — edge cases
# ===================================================================

class TestEdgeCases:
    """Edge-case and error-path tests."""

    def test_alert_with_extra_fields(self, json_logger: AlertLogger) -> None:
        alert = _make_alert()
        alert["custom_field"] = "extra_value"
        alert["nested"] = {"key": [1, 2, 3]}
        json_logger.log_alert(alert)
        loaded = json_logger.load_alerts()
        assert loaded[0]["custom_field"] == "extra_value"
        assert loaded[0]["nested"] == {"key": [1, 2, 3]}

    def test_alert_with_none_values(self, json_logger: AlertLogger) -> None:
        alert = _make_alert()
        alert["optional_field"] = None
        json_logger.log_alert(alert)
        loaded = json_logger.load_alerts()
        assert loaded[0]["optional_field"] is None

    def test_json_file_is_jsonl_format(self, json_logger: AlertLogger) -> None:
        """Each line should be independently parseable JSON."""
        for i in range(3):
            json_logger.log_alert(_make_alert(alert_id=f"line-{i}"))
        date_str = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
        fpath = os.path.join(json_logger.log_dir, f"alerts_{date_str}.json")
        with open(fpath) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "alert_id" in obj

    def test_csv_headers_written_once(self, csv_logger: AlertLogger, alert_dir: Path) -> None:
        for i in range(3):
            csv_logger.log_alert(_make_alert())
        csv_files = list(alert_dir.glob("alerts_*.csv"))
        with open(csv_files[0]) as f:
            reader = csv.reader(f)
            header_count = sum(1 for row in reader if row[0] == "timestamp")
        assert header_count == 1
