from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.detection_engine.engine import DetectionEngine, DetectionResult, DetectionStats

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Structured alert generated from a detection result."""

    alert_id: str
    sample_id: str
    severity: str
    predicted_class: str
    anomaly_score: float
    classification_confidence: float
    is_zero_day: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Manages alert generation, deduplication, and rate limiting.

    Converts DetectionResult objects into structured Alert objects
    with rate limiting and cooldown logic.
    """

    def __init__(self) -> None:
        """Initialize the AlertManager with configuration."""
        from src.core.config import get_config
        threshold_cfg = get_config().load("thresholds")
        alerting_cfg = threshold_cfg.get("alerting", {})

        self.cooldown_seconds = alerting_cfg.get("cooldown_seconds", 10)
        self.max_alerts_per_minute = alerting_cfg.get("max_alerts_per_minute", 60)
        self.batch_alerts = alerting_cfg.get("batch_alerts", True)
        self.minimum_confidence = alerting_cfg.get("minimum_confidence", 0.5)

        self._last_alert_times: Dict[str, float] = {}
        self._alert_count_minute = 0
        self._minute_start = 0.0

        logger.info(
            "AlertManager initialized: cooldown=%ds, max_per_minute=%d",
            self.cooldown_seconds,
            self.max_alerts_per_minute,
        )

    def process_detection_results(
        self,
        results: List[DetectionResult],
    ) -> List[Alert]:
        """Process detection results and generate alerts.

        Only anomalies generate alerts. Zero-day anomalies always alert.
        Known attack types alert if confidence exceeds minimum.

        Args:
            results: List of DetectionResult objects.

        Returns:
            List of Alert objects (filtered by rate limiting).
        """
        import time

        current_time = time.time()
        alerts: List[Alert] = []

        # Reset minute counter if needed
        if current_time - self._minute_start > 60:
            self._alert_count_minute = 0
            self._minute_start = current_time

        for result in results:
            if not result.is_anomaly:
                continue

            # Check rate limit
            if self._alert_count_minute >= self.max_alerts_per_minute:
                logger.warning("Alert rate limit reached (%d/min)", self.max_alerts_per_minute)
                break

            # Check cooldown
            last_time = self._last_alert_times.get(result.sample_id, 0)
            if current_time - last_time < self.cooldown_seconds:
                continue

            # Generate alert
            alert = self._create_alert(result)
            alerts.append(alert)

            self._last_alert_times[result.sample_id] = current_time
            self._alert_count_minute += 1

        logger.info(
            "Alert processing: %d results → %d alerts (rate_limited=%d)",
            len(results),
            len(alerts),
            max(0, len([r for r in results if r.is_anomaly]) - len(alerts)),
        )

        return alerts

    def _create_alert(self, result: DetectionResult) -> Alert:
        """Create an Alert from a DetectionResult.

        Args:
            result: DetectionResult to convert.

        Returns:
            Structured Alert object.
        """
        if result.is_zero_day:
            message = (
                f"ZERO-DAY THREAT DETECTED: Anomalous traffic (score={result.anomaly_score:.4f}) "
                f"not matching any known attack class. Severity: {result.severity.upper()}"
            )
        else:
            message = (
                f"DoS ATTACK DETECTED: {result.predicted_class} "
                f"(anomaly_score={result.anomaly_score:.4f}, "
                f"confidence={result.classification_confidence:.4f}). "
                f"Severity: {result.severity.upper()}"
            )

        alert_id = f"alert_{result.sample_id}_{int(result.anomaly_score * 10000)}"

        return Alert(
            alert_id=alert_id,
            sample_id=result.sample_id,
            severity=result.severity,
            predicted_class=result.predicted_class,
            anomaly_score=result.anomaly_score,
            classification_confidence=result.classification_confidence,
            is_zero_day=result.is_zero_day,
            message=message,
            details={
                "anomaly_threshold": result.anomaly_threshold,
                "class_probabilities": result.class_probabilities,
            },
            metadata=result.metadata,
        )

    def get_alert_summary(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Get summary statistics for a batch of alerts.

        Args:
            alerts: List of Alert objects.

        Returns:
            Dictionary with alert summary.
        """
        if not alerts:
            return {"total": 0, "by_severity": {}, "by_class": {}, "zero_day_count": 0}

        by_severity = {}
        by_class = {}
        zero_day_count = 0

        for alert in alerts:
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
            by_class[alert.predicted_class] = by_class.get(alert.predicted_class, 0) + 1
            if alert.is_zero_day:
                zero_day_count += 1

        return {
            "total": len(alerts),
            "by_severity": by_severity,
            "by_class": by_class,
            "zero_day_count": zero_day_count,
        }

    def alerts_to_dataframe(self, alerts: List[Alert]) -> pd.DataFrame:
        """Convert alerts to a DataFrame.

        Args:
            alerts: List of Alert objects.

        Returns:
            DataFrame with alert data.
        """
        records = []
        for a in alerts:
            records.append({
                "alert_id": a.alert_id,
                "sample_id": a.sample_id,
                "severity": a.severity,
                "predicted_class": a.predicted_class,
                "anomaly_score": a.anomaly_score,
                "classification_confidence": a.classification_confidence,
                "is_zero_day": a.is_zero_day,
                "message": a.message,
            })

        return pd.DataFrame(records)
