"""Overview page — threat summary and system health."""
from __future__ import annotations

import logging
from typing import Any, Dict

from src.dashboard.components.cards import metric_card, status_card
from src.dashboard.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


def render_overview(metrics: MetricsCalculator, model_status: Dict[str, Any]) -> Dict[str, Any]:
    """Render the overview dashboard page.

    Args:
        metrics: MetricsCalculator instance.
        model_status: Dictionary with model status information.

    Returns:
        Dictionary with page content for rendering.
    """
    detection_metrics = metrics.get_detection_metrics()
    alert_metrics = metrics.get_alert_metrics()

    cards = [
        metric_card("Total Detections", detection_metrics["total_detections"]),
        metric_card("Anomaly Rate", f"{detection_metrics['anomaly_rate']:.2%}"),
        metric_card("Zero-Day Rate", f"{detection_metrics['zero_day_rate']:.2%}"),
        metric_card("Avg Latency", f"{detection_metrics['avg_latency_ms']:.2f} ms"),
        metric_card("Total Alerts", alert_metrics["total_alerts"]),
        metric_card("Critical Alerts", alert_metrics.get("by_severity", {}).get("critical", 0)),
    ]

    system_status = status_card(
        "System Status",
        "healthy" if model_status.get("loaded", False) else "warning",
        f"Anomaly detector: {'loaded' if model_status.get('anomaly_loaded', False) else 'not loaded'} | "
        f"Classifier: {'loaded' if model_status.get('classifier_loaded', False) else 'not loaded'}",
    )

    return {
        "title": "Overview",
        "cards": cards,
        "system_status": system_status,
        "class_distribution": detection_metrics.get("class_distribution", {}),
        "severity_distribution": alert_metrics.get("by_severity", {}),
    }
