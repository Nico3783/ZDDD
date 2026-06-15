from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.config import get_config

logger = logging.getLogger(__name__)

# Severity emoji mapping for human-readable output
SEVERITY_ICONS: Dict[str, str] = {
    "critical": "[!!!]",
    "high": "[!! ]",
    "medium": "[!  ]",
    "low": "[   ]",
}


def format_alert_text(alert: Any) -> str:
    """Format an alert as human-readable text.

    Args:
        alert: Alert object or dictionary.

    Returns:
        Formatted text string.
    """
    icon = SEVERITY_ICONS.get(getattr(alert, "severity", "unknown"), "[?] ")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if hasattr(alert, "alert_id"):
        alert_id = alert.alert_id
        severity = alert.severity.upper()
        predicted_class = alert.predicted_class
        anomaly_score = alert.anomaly_score
        confidence = alert.classification_confidence
        is_zero_day = alert.is_zero_day
        message = alert.message
    else:
        alert_id = alert.get("alert_id", "N/A")
        severity = alert.get("severity", "unknown").upper()
        predicted_class = alert.get("predicted_class", "unknown")
        anomaly_score = alert.get("anomaly_score", 0.0)
        confidence = alert.get("classification_confidence", 0.0)
        is_zero_day = alert.get("is_zero_day", False)
        message = alert.get("message", "No message")

    lines = [
        f"{icon} [{severity}] Alert: {alert_id}",
        f"  Time:      {timestamp}",
        f"  Class:     {predicted_class}",
        f"  Score:     {anomaly_score:.4f}",
        f"  Confidence:{confidence:.4f}",
        f"  Zero-Day:  {'YES' if is_zero_day else 'no'}",
        f"  Message:   {message}",
    ]

    return "\n".join(lines)


def format_alert_json(alert: Any) -> str:
    """Format an alert as a JSON string.

    Args:
        alert: Alert object or dictionary.

    Returns:
        JSON string representation of the alert.
    """
    if hasattr(alert, "__dict__"):
        alert_dict = {
            "alert_id": alert.alert_id,
            "sample_id": alert.sample_id,
            "severity": alert.severity,
            "predicted_class": alert.predicted_class,
            "anomaly_score": alert.anomaly_score,
            "classification_confidence": alert.classification_confidence,
            "is_zero_day": alert.is_zero_day,
            "message": alert.message,
            "details": alert.details if hasattr(alert, "details") else {},
            "metadata": alert.metadata if hasattr(alert, "metadata") else {},
            "formatted_at": datetime.now().isoformat(),
        }
    else:
        alert_dict = dict(alert)
        alert_dict["formatted_at"] = datetime.now().isoformat()

    return json.dumps(alert_dict, indent=2)


def format_alert_csv_row(alert: Any) -> List[str]:
    """Format an alert as a CSV row (list of string values).

    Args:
        alert: Alert object or dictionary.

    Returns:
        List of strings for CSV row.
    """
    if hasattr(alert, "alert_id"):
        return [
            getattr(alert, "alert_id", ""),
            getattr(alert, "sample_id", ""),
            getattr(alert, "severity", ""),
            getattr(alert, "predicted_class", ""),
            str(getattr(alert, "anomaly_score", 0.0)),
            str(getattr(alert, "classification_confidence", 0.0)),
            str(getattr(alert, "is_zero_day", False)),
            getattr(alert, "message", ""),
        ]

    return [
        alert.get("alert_id", ""),
        alert.get("sample_id", ""),
        alert.get("severity", ""),
        alert.get("predicted_class", ""),
        str(alert.get("anomaly_score", 0.0)),
        str(alert.get("classification_confidence", 0.0)),
        str(alert.get("is_zero_day", False)),
        alert.get("message", ""),
    ]


def format_alert_summary(alerts: List[Any]) -> str:
    """Format a batch of alerts as a summary report.

    Args:
        alerts: List of Alert objects or dictionaries.

    Returns:
        Summary report string.
    """
    if not alerts:
        return "No alerts to report."

    severity_counts: Dict[str, int] = {}
    class_counts: Dict[str, int] = {}
    zero_day_count = 0

    for alert in alerts:
        sev = getattr(alert, "severity", None) or alert.get("severity", "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

        cls = getattr(alert, "predicted_class", None) or alert.get("predicted_class", "unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1

        zd = getattr(alert, "is_zero_day", None)
        if zd is None:
            zd = alert.get("is_zero_day", False)
        if zd:
            zero_day_count += 1

    lines = [
        "=" * 60,
        f"  ALERT SUMMARY REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"  Total Alerts:       {len(alerts)}",
        f"  Zero-Day Threats:   {zero_day_count}",
        "",
        "  Severity Breakdown:",
    ]

    for sev in ["critical", "high", "medium", "low"]:
        if sev in severity_counts:
            icon = SEVERITY_ICONS.get(sev, "[?] ")
            lines.append(f"    {icon} {sev.upper():>10}: {severity_counts[sev]}")

    lines.append("")
    lines.append("  Class Distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {cls:>25}: {count}")

    lines.append("=" * 60)

    return "\n".join(lines)


def format_alert_batch(
    alerts: List[Any],
    output_format: str = "text",
) -> List[str]:
    """Format a batch of alerts in the specified format.

    Args:
        alerts: List of Alert objects or dictionaries.
        output_format: One of 'text', 'json', 'csv'.

    Returns:
        List of formatted strings.
    """
    if output_format == "text":
        return [format_alert_text(a) for a in alerts]
    elif output_format == "json":
        return [format_alert_json(a) for a in alerts]
    elif output_format == "csv":
        return [",".join(format_alert_csv_row(a)) for a in alerts]
    else:
        logger.warning("Unknown format '%s', defaulting to text", output_format)
        return [format_alert_text(a) for a in alerts]
