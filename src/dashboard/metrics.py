from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates real-time metrics for the detection dashboard.

    Provides aggregated statistics for streaming and batch analysis.
    """

    def __init__(self) -> None:
        """Initialize the metrics calculator."""
        self._detection_history: List[Dict[str, Any]] = []
        self._alert_history: List[Dict[str, Any]] = []
        logger.info("MetricsCalculator initialized")

    def record_detection(self, result: Any) -> None:
        """Record a detection result for metric calculation.

        Args:
            result: DetectionResult object or dictionary.
        """
        if hasattr(result, "__dict__"):
            record = {
                "timestamp": getattr(result, "timestamp", None),
                "is_anomaly": getattr(result, "is_anomaly", False),
                "anomaly_score": getattr(result, "anomaly_score", 0.0),
                "predicted_class": getattr(result, "predicted_class", "UNKNOWN"),
                "classification_confidence": getattr(result, "classification_confidence", 0.0),
                "is_zero_day": getattr(result, "is_zero_day", False),
                "severity": getattr(result, "severity", "low"),
                "latency_ms": getattr(result, "latency_ms", 0.0),
            }
        else:
            record = result

        self._detection_history.append(record)

    def record_alert(self, alert: Any) -> None:
        """Record an alert for metric calculation.

        Args:
            alert: Alert object or dictionary.
        """
        if hasattr(alert, "__dict__"):
            record = {
                "timestamp": getattr(alert, "timestamp", None),
                "severity": getattr(alert, "severity", "low"),
                "is_zero_day": getattr(alert, "is_zero_day", False),
                "predicted_class": getattr(alert, "predicted_class", "UNKNOWN"),
            }
        else:
            record = alert

        self._alert_history.append(record)

    def get_detection_metrics(self) -> Dict[str, Any]:
        """Calculate detection metrics from recorded history.

        Returns:
            Dictionary with detection metrics.
        """
        if not self._detection_history:
            return {
                "total_detections": 0,
                "anomaly_rate": 0.0,
                "zero_day_rate": 0.0,
                "avg_anomaly_score": 0.0,
                "avg_latency_ms": 0.0,
                "class_distribution": {},
                "severity_distribution": {},
            }

        total = len(self._detection_history)
        anomalies = sum(1 for d in self._detection_history if d.get("is_anomaly", False))
        zero_days = sum(1 for d in self._detection_history if d.get("is_zero_day", False))

        scores = [d.get("anomaly_score", 0.0) for d in self._detection_history]
        latencies = [d.get("latency_ms", 0.0) for d in self._detection_history]

        class_dist: Dict[str, int] = {}
        severity_dist: Dict[str, int] = {}
        for d in self._detection_history:
            cls = d.get("predicted_class", "UNKNOWN")
            class_dist[cls] = class_dist.get(cls, 0) + 1
            sev = d.get("severity", "low")
            severity_dist[sev] = severity_dist.get(sev, 0) + 1

        return {
            "total_detections": total,
            "anomaly_rate": anomalies / total if total > 0 else 0.0,
            "zero_day_rate": zero_days / total if total > 0 else 0.0,
            "avg_anomaly_score": sum(scores) / total if total > 0 else 0.0,
            "avg_latency_ms": sum(latencies) / total if total > 0 else 0.0,
            "class_distribution": class_dist,
            "severity_distribution": severity_dist,
        }

    def get_alert_metrics(self) -> Dict[str, Any]:
        """Calculate alert metrics from recorded history.

        Returns:
            Dictionary with alert metrics.
        """
        if not self._alert_history:
            return {
                "total_alerts": 0,
                "by_severity": {},
                "by_class": {},
                "zero_day_alerts": 0,
            }

        total = len(self._alert_history)
        by_severity: Dict[str, int] = {}
        by_class: Dict[str, int] = {}
        zero_day_count = 0

        for a in self._alert_history:
            sev = a.get("severity", "low")
            by_severity[sev] = by_severity.get(sev, 0) + 1

            cls = a.get("predicted_class", "UNKNOWN")
            by_class[cls] = by_class.get(cls, 0) + 1

            if a.get("is_zero_day", False):
                zero_day_count += 1

        return {
            "total_alerts": total,
            "by_severity": by_severity,
            "by_class": by_class,
            "zero_day_alerts": zero_day_count,
        }

    def get_timeseries(self, window_seconds: int = 60) -> pd.DataFrame:
        """Get detection timeseries data for plotting.

        Args:
            window_seconds: Rolling window size in seconds.

        Returns:
            DataFrame with timeseries data.
        """
        if not self._detection_history:
            return pd.DataFrame()

        df = pd.DataFrame(self._detection_history)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

        return df

    def reset(self) -> None:
        """Reset all recorded history."""
        self._detection_history.clear()
        self._alert_history.clear()
        logger.info("MetricsCalculator history reset")
