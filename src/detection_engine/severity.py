from __future__ import annotations

import logging
from typing import Dict, Optional

from src.core.config import get_config

logger = logging.getLogger(__name__)


class SeverityCalculator:
    """Computes alert severity from anomaly scores and classification confidence.

    Severity levels (applied to effective_score after zero-day boost):
        critical: effective_score >= 0.9
        high: effective_score >= 0.7
        medium: effective_score >= 0.5
        low: everything else

    Zero-day boost: if is_zero_day, anomaly_score is increased by zero_day_boost
    (default 0.2), capped at 1.0, then compared against thresholds.

    Supports configurable thresholds loaded from config/thresholds.yaml.
    """

    def __init__(self) -> None:
        """Initialize SeverityCalculator from configuration."""
        threshold_cfg = get_config().load("thresholds")
        alerting_cfg = threshold_cfg.get("alerting", {})

        self.severity_levels: Dict[str, float] = alerting_cfg.get("severity_levels", {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.5,
            "low": 0.0,
        })

        # Flatten nested severity_levels: extract min_score from dicts
        self.severity_levels = {
            k: v["min_score"] if isinstance(v, dict) and "min_score" in v else v
            for k, v in self.severity_levels.items()
        }

        self.zero_day_boost: float = alerting_cfg.get("severity_boost_zero_day", 0.2)
        self.zero_day_high_confidence: float = alerting_cfg.get("zero_day_high_confidence", 0.7)

        logger.info(
            "SeverityCalculator initialized: levels=%s, zero_day_boost=%.2f",
            self.severity_levels, self.zero_day_boost,
        )

    def compute(
        self,
        anomaly_score: float,
        classification_confidence: float = 1.0,
        is_zero_day: bool = False,
    ) -> str:
        """Compute severity from anomaly score, confidence, and zero-day flag.

        If is_zero_day is True, the anomaly_score is boosted by zero_day_boost
        (capped at 1.0) before comparing against severity thresholds.

        Args:
            anomaly_score: Normalized anomaly score (0-1).
            classification_confidence: Classification confidence (0-1).
            is_zero_day: Whether the sample is a zero-day anomaly.

        Returns:
            Severity string: 'critical', 'high', 'medium', or 'low'.
        """
        effective_score = anomaly_score
        if is_zero_day:
            effective_score = round(min(1.0, anomaly_score + self.zero_day_boost), 10)

        if effective_score >= self.severity_levels.get("critical", 0.9):
            return "critical"
        if effective_score >= self.severity_levels.get("high", 0.7):
            return "high"
        if effective_score >= self.severity_levels.get("medium", 0.5):
            return "medium"
        return "low"

    def compute_batch(
        self,
        anomaly_scores: list[float],
        classification_confidences: Optional[list[float]] = None,
        zero_day_flags: Optional[list[bool]] = None,
    ) -> list[str]:
        """Compute severity for a batch of results.

        Args:
            anomaly_scores: List of anomaly scores.
            classification_confidences: List of classification confidences.
            zero_day_flags: List of zero-day flags.

        Returns:
            List of severity strings.
        """
        n = len(anomaly_scores)
        if classification_confidences is None:
            classification_confidences = [1.0] * n
        if zero_day_flags is None:
            zero_day_flags = [False] * n

        return [
            self.compute(score, conf, zd)
            for score, conf, zd in zip(anomaly_scores, classification_confidences, zero_day_flags)
        ]

    def get_config(self) -> Dict[str, float]:
        """Get the current severity threshold configuration.

        Returns:
            Dictionary of severity level thresholds.
        """
        return dict(self.severity_levels)


def compute_severity(
    anomaly_score: float,
    classification_confidence: float = 1.0,
    is_zero_day: bool = False,
) -> str:
    """Convenience function to compute severity without instantiating the class.

    Args:
        anomaly_score: Normalized anomaly score (0-1).
        classification_confidence: Classification confidence.
        is_zero_day: Whether the sample is a zero-day anomaly.

    Returns:
        Severity string.
    """
    calc = SeverityCalculator()
    return calc.compute(anomaly_score, classification_confidence, is_zero_day)
