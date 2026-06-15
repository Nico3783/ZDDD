"""Alerts page — alert history and zero-day detection events."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.dashboard.components.charts import create_severity_chart, create_zero_day_timeline
from src.dashboard.components.tables import alerts_table

logger = logging.getLogger(__name__)


def render_alerts(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Render the alerts dashboard page.

    Args:
        alerts: List of alert dictionaries.

    Returns:
        Dictionary with page content for rendering.
    """
    severity_counts: Dict[str, int] = {}
    zero_day_alerts: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []

    for alert in alerts:
        sev = alert.get("severity", "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

        if alert.get("is_zero_day", False):
            zero_day_alerts.append(alert)

        timeline.append({
            "timestamp": alert.get("timestamp"),
            "anomaly_score": alert.get("anomaly_score", 0.0),
            "severity": sev,
            "is_zero_day": alert.get("is_zero_day", False),
        })

    return {
        "title": "Alerts",
        "total_alerts": len(alerts),
        "zero_day_alerts": len(zero_day_alerts),
        "severity_chart": create_severity_chart(severity_counts),
        "timeline_chart": create_zero_day_timeline(timeline),
        "alerts_table": alerts_table(alerts),
    }
