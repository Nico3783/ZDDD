from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.config import get_config
from src.detection_engine.engine import DetectionResult

logger = logging.getLogger(__name__)


class AlertGenerator:
    """Generates structured alerts from detection results.

    Applies alert policies (minimum confidence, severity thresholds, zero-day boost)
    to produce actionable security alerts.
    """

    def __init__(self) -> None:
        """Initialize the AlertGenerator from configuration."""
        threshold_cfg = get_config().load("thresholds")
        alerting_cfg = threshold_cfg.get("alerting", {})

        self.minimum_confidence: float = alerting_cfg.get("minimum_confidence", 0.5)
        self.severity_boost_zero_day: float = alerting_cfg.get("severity_boost_zero_day", 0.2)
        self.alert_enabled: bool = alerting_cfg.get("enabled", True)

        self._generation_count = 0
        logger.info(
            "AlertGenerator initialized: min_confidence=%.2f, zero_day_boost=%.2f",
            self.minimum_confidence,
            self.severity_boost_zero_day,
        )

    def generate_alerts(
        self,
        results: List[DetectionResult],
        filter_normal: bool = True,
        filter_below_confidence: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate alerts from detection results.

        Args:
            results: List of DetectionResult objects.
            filter_normal: If True, exclude non-anomalous results.
            filter_below_confidence: If True, exclude results below minimum confidence.

        Returns:
            List of alert dictionaries ready for dispatch.
        """
        if not self.alert_enabled:
            logger.info("Alert generation disabled, returning empty list")
            return []

        alerts: List[Dict[str, Any]] = []

        for result in results:
            # Filter non-anomalies
            if filter_normal and not result.is_anomaly:
                continue

            # Filter below minimum confidence for known attacks
            if (
                filter_below_confidence
                and not result.is_zero_day
                and result.classification_confidence < self.minimum_confidence
            ):
                continue

            alert = self._result_to_alert(result)
            alerts.append(alert)

        self._generation_count += len(alerts)

        logger.info(
            "Generated %d alerts from %d results (filtered=%d)",
            len(alerts),
            len(results),
            len(results) - len(alerts),
        )

        return alerts

    def _result_to_alert(self, result: DetectionResult) -> Dict[str, Any]:
        """Convert a DetectionResult to an alert dictionary.

        Args:
            result: DetectionResult to convert.

        Returns:
            Alert dictionary with all required fields.
        """
        alert_id = f"alert_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        message = self._build_message(result)
        severity = self._adjust_severity(result)

        return {
            "alert_id": alert_id,
            "sample_id": result.sample_id,
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "predicted_class": result.predicted_class,
            "anomaly_score": result.anomaly_score,
            "classification_confidence": result.classification_confidence,
            "is_zero_day": result.is_zero_day,
            "message": message,
            "details": {
                "anomaly_threshold": result.anomaly_threshold,
                "class_probabilities": result.class_probabilities,
                "detection_latency_ms": result.detection_latency_ms,
            },
            "metadata": result.metadata,
        }

    def _build_message(self, result: DetectionResult) -> str:
        """Build a human-readable alert message.

        Args:
            result: DetectionResult.

        Returns:
            Alert message string.
        """
        if result.is_zero_day:
            return (
                f"ZERO-DAY THREAT DETECTED: Anomalous traffic "
                f"(score={result.anomaly_score:.4f}) not matching any known attack class. "
                f"Severity: {result.severity.upper()}"
            )

        return (
            f"DoS ATTACK DETECTED: {result.predicted_class} "
            f"(anomaly_score={result.anomaly_score:.4f}, "
            f"confidence={result.classification_confidence:.4f}). "
            f"Severity: {result.severity.upper()}"
        )

    def _adjust_severity(self, result: DetectionResult) -> str:
        """Adjust severity based on zero-day boost.

        Args:
            result: DetectionResult.

        Returns:
            Adjusted severity string.
        """
        if not result.is_zero_day:
            return result.severity

        # Boost severity for zero-day threats
        severity_order = ["low", "medium", "high", "critical"]
        current_idx = severity_order.index(result.severity) if result.severity in severity_order else 0

        if self.severity_boost_zero_day > 0 and current_idx < len(severity_order) - 1:
            # Boost by one level for zero-day
            boosted_idx = min(current_idx + 1, len(severity_order) - 1)
            boosted = severity_order[boosted_idx]
            logger.debug(
                "Zero-day severity boost: %s → %s (score=%.4f)",
                result.severity, boosted, result.anomaly_score,
            )
            return boosted

        return result.severity

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get alert generation statistics.

        Returns:
            Dictionary with generation stats.
        """
        return {
            "total_generated": self._generation_count,
            "alert_enabled": self.alert_enabled,
            "minimum_confidence": self.minimum_confidence,
            "severity_boost_zero_day": self.severity_boost_zero_day,
        }


def generate_alerts_from_results(
    results: List[DetectionResult],
    minimum_confidence: Optional[float] = None,
    filter_normal: bool = True,
) -> List[Dict[str, Any]]:
    """Convenience function to generate alerts from detection results.

    Args:
        results: List of DetectionResult objects.
        minimum_confidence: Override for minimum confidence threshold.
        filter_normal: Whether to exclude non-anomalous results.

    Returns:
        List of alert dictionaries.
    """
    generator = AlertGenerator()

    if minimum_confidence is not None:
        generator.minimum_confidence = minimum_confidence

    return generator.generate_alerts(results, filter_normal=filter_normal)
