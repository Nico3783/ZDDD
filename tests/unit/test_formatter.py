"""Unit tests for src/alerting/formatter.py.

Covers: format_alert_text, format_alert_json, format_alert_csv_row,
        format_alert_summary, format_alert_batch.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from src.alerting.formatter import (
    SEVERITY_ICONS,
    format_alert_batch,
    format_alert_csv_row,
    format_alert_json,
    format_alert_summary,
    format_alert_text,
)


# ---------------------------------------------------------------------------
# Shared alert dicts
# ---------------------------------------------------------------------------

def _make_alert(**overrides: Any) -> Dict[str, Any]:
    base = {
        "alert_id": "alert-001",
        "sample_id": "sample-001",
        "severity": "high",
        "predicted_class": "DoS Hulk",
        "anomaly_score": 0.85,
        "classification_confidence": 0.92,
        "is_zero_day": False,
        "message": "Known attack detected",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# format_alert_text
# ---------------------------------------------------------------------------

class TestFormatAlertText:
    """Tests for the format_alert_text function."""

    def test_returns_string(self):
        result = format_alert_text(_make_alert())
        assert isinstance(result, str)

    def test_contains_alert_id(self):
        alert = _make_alert(alert_id="alert-42")
        result = format_alert_text(alert)
        assert "alert-42" in result

    def test_contains_severity_uppercase(self):
        alert = _make_alert(severity="critical")
        result = format_alert_text(alert)
        assert "CRITICAL" in result

    def test_contains_predicted_class(self):
        alert = _make_alert(predicted_class="DoS GoldenEye")
        result = format_alert_text(alert)
        assert "DoS GoldenEye" in result

    def test_contains_anomaly_score(self):
        alert = _make_alert(anomaly_score=0.75)
        result = format_alert_text(alert)
        assert "0.7500" in result

    def test_contains_confidence(self):
        alert = _make_alert(classification_confidence=0.88)
        result = format_alert_text(alert)
        assert "0.8800" in result

    def test_zero_day_yes(self):
        alert = _make_alert(is_zero_day=True)
        result = format_alert_text(alert)
        assert "YES" in result

    def test_zero_day_no(self):
        alert = _make_alert(is_zero_day=False)
        result = format_alert_text(alert)
        assert "no" in result

    def test_contains_message(self):
        alert = _make_alert(message="Suspicious traffic spike")
        result = format_alert_text(alert)
        assert "Suspicious traffic spike" in result

    def test_severity_icon_present(self):
        from types import SimpleNamespace
        alert = SimpleNamespace(**_make_alert(severity="high"))
        result = format_alert_text(alert)
        assert SEVERITY_ICONS["high"] in result


# ---------------------------------------------------------------------------
# format_alert_json
# ---------------------------------------------------------------------------

class TestFormatAlertJson:
    """Tests for the format_alert_json function."""

    def test_returns_string(self):
        result = format_alert_json(_make_alert())
        assert isinstance(result, str)

    def test_is_valid_json(self):
        result = format_alert_json(_make_alert())
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_contains_expected_keys(self):
        alert = _make_alert()
        result = format_alert_json(alert)
        parsed = json.loads(result)
        assert "alert_id" in parsed
        assert "severity" in parsed
        assert "predicted_class" in parsed
        assert "anomaly_score" in parsed
        assert "classification_confidence" in parsed
        assert "is_zero_day" in parsed
        assert "message" in parsed

    def test_includes_formatted_at_timestamp(self):
        result = format_alert_json(_make_alert())
        parsed = json.loads(result)
        assert "formatted_at" in parsed

    def test_preserves_values(self):
        alert = _make_alert(alert_id="alert-99", severity="critical")
        parsed = json.loads(format_alert_json(alert))
        assert parsed["alert_id"] == "alert-99"
        assert parsed["severity"] == "critical"

    def test_anomaly_score_is_numeric(self):
        parsed = json.loads(format_alert_json(_make_alert(anomaly_score=0.73)))
        assert isinstance(parsed["anomaly_score"], float)
        assert abs(parsed["anomaly_score"] - 0.73) < 1e-10


# ---------------------------------------------------------------------------
# format_alert_csv_row
# ---------------------------------------------------------------------------

class TestFormatAlertCsvRow:
    """Tests for the format_alert_csv_row function."""

    def test_returns_list(self):
        result = format_alert_csv_row(_make_alert())
        assert isinstance(result, list)

    def test_correct_number_of_columns(self):
        result = format_alert_csv_row(_make_alert())
        # alert_id, sample_id, severity, predicted_class,
        # anomaly_score, classification_confidence, is_zero_day, message
        assert len(result) == 8

    def test_values_are_strings(self):
        result = format_alert_csv_row(_make_alert())
        assert all(isinstance(v, str) for v in result)

    def test_first_element_is_alert_id(self):
        alert = _make_alert(alert_id="alert-77")
        result = format_alert_csv_row(alert)
        assert result[0] == "alert-77"

    def test_second_element_is_sample_id(self):
        alert = _make_alert(sample_id="sample-12")
        result = format_alert_csv_row(alert)
        assert result[1] == "sample-12"

    def test_third_element_is_severity(self):
        alert = _make_alert(severity="low")
        result = format_alert_csv_row(alert)
        assert result[2] == "low"

    def test_csv_row_can_be_joined(self):
        result = format_alert_csv_row(_make_alert())
        csv_line = ",".join(result)
        assert isinstance(csv_line, str)
        parts = csv_line.split(",")
        assert len(parts) == 8


# ---------------------------------------------------------------------------
# format_alert_summary
# ---------------------------------------------------------------------------

class TestFormatAlertSummary:
    """Tests for the format_alert_summary function."""

    def test_empty_list_returns_no_alerts_message(self):
        result = format_alert_summary([])
        assert "no alerts to report" in result.lower()

    def test_returns_string(self):
        alerts = [_make_alert(severity="high")]
        result = format_alert_summary(alerts)
        assert isinstance(result, str)

    def test_contains_total_alerts(self):
        alerts = [_make_alert(severity="high"), _make_alert(severity="low")]
        result = format_alert_summary(alerts)
        assert "2" in result

    def test_contains_header(self):
        result = format_alert_summary([_make_alert()])
        assert "ALERT SUMMARY" in result.upper() or "SUMMARY" in result.upper()

    def test_contains_zero_day_count(self):
        alerts = [
            _make_alert(is_zero_day=True),
            _make_alert(is_zero_day=False),
        ]
        result = format_alert_summary(alerts)
        assert "1" in result  # 1 zero-day

    def test_severity_breakdown_present(self):
        alerts = [
            _make_alert(severity="critical"),
            _make_alert(severity="high"),
            _make_alert(severity="low"),
        ]
        result = format_alert_summary(alerts)
        assert "CRITICAL" in result.upper()
        assert "HIGH" in result.upper()
        assert "LOW" in result.upper()

    def test_class_distribution_present(self):
        alerts = [
            _make_alert(predicted_class="DoS Hulk"),
            _make_alert(predicted_class="DoS Hulk"),
            _make_alert(predicted_class="BENIGN"),
        ]
        result = format_alert_summary(alerts)
        assert "DoS Hulk" in result
        assert "BENIGN" in result


# ---------------------------------------------------------------------------
# format_alert_batch
# ---------------------------------------------------------------------------

class TestFormatAlertBatch:
    """Tests for the format_alert_batch function."""

    def test_text_output(self):
        alerts = [_make_alert(), _make_alert(alert_id="alert-002")]
        result = format_alert_batch(alerts, output_format="text")
        assert isinstance(result, list)
        assert len(result) == 2
        assert "alert-001" in result[0]

    def test_json_output(self):
        alerts = [_make_alert()]
        result = format_alert_batch(alerts, output_format="json")
        assert len(result) == 1
        parsed = json.loads(result[0])
        assert "alert_id" in parsed

    def test_csv_output(self):
        alerts = [_make_alert()]
        result = format_alert_batch(alerts, output_format="csv")
        assert len(result) == 1
        assert "alert-001" in result[0]

    def test_unknown_format_defaults_to_text(self):
        alerts = [_make_alert()]
        result = format_alert_batch(alerts, output_format="xml")
        assert len(result) == 1
        assert "alert-001" in result[0]

    def test_empty_list(self):
        result = format_alert_batch([], output_format="text")
        assert result == []

    def test_batch_preserves_order(self):
        alerts = [_make_alert(alert_id=f"alert-{i:03d}") for i in range(5)]
        result = format_alert_batch(alerts, output_format="text")
        for i, text in enumerate(result):
            assert f"alert-{i:03d}" in text


# ---------------------------------------------------------------------------
# SEVERITY_ICONS mapping
# ---------------------------------------------------------------------------

class TestSeverityIcons:
    """Tests for the SEVERITY_ICONS constant."""

    def test_critical_has_icon(self):
        assert "critical" in SEVERITY_ICONS
        assert len(SEVERITY_ICONS["critical"]) > 0

    def test_high_has_icon(self):
        assert "high" in SEVERITY_ICONS

    def test_medium_has_icon(self):
        assert "medium" in SEVERITY_ICONS

    def test_low_has_icon(self):
        assert "low" in SEVERITY_ICONS

    def test_all_icons_are_bracketed(self):
        for sev, icon in SEVERITY_ICONS.items():
            assert "[" in icon and "]" in icon, f"{sev} icon not bracketed: {icon}"
