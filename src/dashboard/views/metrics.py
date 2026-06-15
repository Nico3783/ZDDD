"""Metrics page — detection performance charts and statistics."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.dashboard.components.charts import (
    create_anomaly_score_distribution,
    create_class_distribution_chart,
    create_latency_chart,
    create_severity_chart,
)
from src.dashboard.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


def render_metrics(metrics: MetricsCalculator) -> Dict[str, Any]:
    """Render the detailed metrics page.

    Args:
        metrics: MetricsCalculator instance.

    Returns:
        Dictionary with page content for rendering.
    """
    detection_metrics = metrics.get_detection_metrics()
    alert_metrics = metrics.get_alert_metrics()
    timeseries = metrics.get_timeseries()

    charts: List[Dict[str, Any]] = []
    if not timeseries.empty and "anomaly_score" in timeseries.columns:
        scores = timeseries["anomaly_score"].tolist()
        charts.append(create_anomaly_score_distribution(scores))

    if detection_metrics.get("class_distribution"):
        charts.append(create_class_distribution_chart(detection_metrics["class_distribution"]))

    if detection_metrics.get("severity_distribution"):
        charts.append(create_severity_chart(detection_metrics["severity_distribution"]))

    if not timeseries.empty and "latency_ms" in timeseries.columns:
        latencies = timeseries["latency_ms"].tolist()
        charts.append(create_latency_chart(latencies))

    return {
        "title": "Detection Metrics",
        "detection_metrics": detection_metrics,
        "alert_metrics": alert_metrics,
        "charts": charts,
        "timeseries_available": not timeseries.empty,
    }
