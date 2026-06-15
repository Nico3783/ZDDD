from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def create_anomaly_score_distribution(scores: List[float], thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Create data for anomaly score distribution chart.

    Args:
        scores: List of anomaly scores.
        thresholds: Optional threshold dict (e.g., {'critical': 0.9, 'high': 0.7}).

    Returns:
        Dictionary with chart configuration.
    """
    return {
        "type": "histogram",
        "data": scores,
        "title": "Anomaly Score Distribution",
        "x_label": "Anomaly Score",
        "y_label": "Frequency",
        "thresholds": thresholds or {},
    }


def create_timeseries_chart(df: pd.DataFrame, value_column: str, title: str = "Time Series") -> Dict[str, Any]:
    """Create data for a timeseries line chart.

    Args:
        df: DataFrame with timestamp and value columns.
        value_column: Column name for y-axis values.
        title: Chart title.

    Returns:
        Dictionary with chart configuration.
    """
    if df.empty or value_column not in df.columns:
        return {
            "type": "line",
            "title": title,
            "data": [],
            "x_label": "Time",
            "y_label": value_column,
        }

    return {
        "type": "line",
        "title": title,
        "data": df[value_column].tolist(),
        "timestamps": df["timestamp"].tolist() if "timestamp" in df.columns else [],
        "x_label": "Time",
        "y_label": value_column,
    }


def create_class_distribution_chart(distribution: Dict[str, int]) -> Dict[str, Any]:
    """Create data for class distribution pie/bar chart.

    Args:
        distribution: Dictionary mapping class names to counts.

    Returns:
        Dictionary with chart configuration.
    """
    labels = list(distribution.keys())
    values = list(distribution.values())

    return {
        "type": "pie",
        "title": "Traffic Class Distribution",
        "labels": labels,
        "values": values,
    }


def create_severity_chart(severity_dist: Dict[str, int]) -> Dict[str, Any]:
    """Create data for alert severity bar chart.

    Args:
        severity_dist: Dictionary mapping severity levels to counts.

    Returns:
        Dictionary with chart configuration.
    """
    severity_order = ["critical", "high", "medium", "low"]
    labels = [s for s in severity_order if s in severity_dist]
    values = [severity_dist[s] for s in labels]

    return {
        "type": "bar",
        "title": "Alert Severity Distribution",
        "labels": labels,
        "values": values,
        "colors": ["#dc2626", "#f97316", "#eab308", "#22c55e"],
    }


def create_latency_chart(latencies: List[float], title: str = "Detection Latency") -> Dict[str, Any]:
    """Create data for latency box/violin chart.

    Args:
        latencies: List of latency values in milliseconds.
        title: Chart title.

    Returns:
        Dictionary with chart configuration.
    """
    if not latencies:
        return {
            "type": "box",
            "title": title,
            "data": [],
        }

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    return {
        "type": "box",
        "title": title,
        "data": sorted_lat,
        "stats": {
            "min": sorted_lat[0],
            "q1": sorted_lat[n // 4],
            "median": sorted_lat[n // 2],
            "q3": sorted_lat[3 * n // 4],
            "max": sorted_lat[-1],
            "mean": sum(sorted_lat) / n,
        },
        "x_label": "Latency (ms)",
    }


def create_zero_day_timeline(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create data for zero-day detection timeline.

    Args:
        timeline: List of dicts with 'timestamp', 'detected', 'severity'.

    Returns:
        Dictionary with chart configuration.
    """
    return {
        "type": "scatter",
        "title": "Zero-Day Detection Timeline",
        "data": timeline,
        "x_label": "Time",
        "y_label": "Anomaly Score",
    }
