from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class DetectionDecision:
    """Result of the decision logic applied to a single sample."""

    predicted_class: str
    classification_confidence: float
    class_probabilities: Dict[str, float]
    is_zero_day: bool
    decision_reason: str


class DetectionDecisionMaker:
    """Applies zero-day detection rules and classification decisions.

    Decision logic:
        1. Anomaly detected by Isolation Forest
        2. Random Forest classifies the sample
        3. If RF predicts BENIGN or confidence < minimum → zero-day
        4. Severity computed from anomaly score + zero-day flag
    """

    def __init__(self) -> None:
        """Initialize the DecisionMaker from configuration."""
        threshold_cfg = get_config().load("thresholds")
        alerting_cfg = threshold_cfg.get("alerting", {})
        classification_cfg = threshold_cfg.get("classification", {})

        self.minimum_confidence: float = alerting_cfg.get("minimum_confidence", 0.5)
        self.classification_threshold: float = classification_cfg.get("decision_threshold", 0.5)

        logger.info(
            "DetectionDecisionMaker initialized: min_confidence=%.2f, class_threshold=%.2f",
            self.minimum_confidence, self.classification_threshold,
        )

    def decide(
        self,
        is_anomaly: bool,
        rf_class: Optional[str] = None,
        rf_confidence: Optional[float] = None,
        rf_probabilities: Optional[Dict[str, float]] = None,
        anomaly_class_label: str = "BENIGN",
    ) -> DetectionDecision:
        """Apply decision logic to determine classification and zero-day status.

        Args:
            is_anomaly: Whether the sample is anomalous (from Isolation Forest).
            rf_class: Random Forest predicted class name.
            rf_confidence: Random Forest prediction confidence.
            rf_probabilities: Random Forest class probabilities.
            anomaly_class_label: The label RF uses for normal/benign traffic.

        Returns:
            DetectionDecision with final classification and zero-day flag.
        """
        if not is_anomaly:
            return DetectionDecision(
                predicted_class=anomaly_class_label,
                classification_confidence=1.0,
                class_probabilities={anomaly_class_label: 1.0},
                is_zero_day=False,
                decision_reason="not_anomalous",
            )

        # Anomaly detected — apply RF classification
        if rf_class is None or rf_confidence is None:
            return DetectionDecision(
                predicted_class="UNKNOWN",
                classification_confidence=0.0,
                class_probabilities={},
                is_zero_day=True,
                decision_reason="anomaly_no_classifier",
            )

        # Check if RF classifies as BENIGN or low confidence
        if rf_class == anomaly_class_label:
            return DetectionDecision(
                predicted_class="ZERO_DAY",
                classification_confidence=rf_confidence,
                class_probabilities=rf_probabilities or {},
                is_zero_day=True,
                decision_reason="anomaly_classified_as_benign",
            )

        if rf_confidence < self.minimum_confidence:
            return DetectionDecision(
                predicted_class="ZERO_DAY",
                classification_confidence=rf_confidence,
                class_probabilities=rf_probabilities or {},
                is_zero_day=True,
                decision_reason="low_classification_confidence",
            )

        # Known attack with sufficient confidence
        return DetectionDecision(
            predicted_class=rf_class,
            classification_confidence=rf_confidence,
            class_probabilities=rf_probabilities or {},
            is_zero_day=False,
            decision_reason="known_attack",
        )

    def decide_batch(
        self,
        anomaly_flags: list[bool],
        rf_classes: Optional[list[str]] = None,
        rf_confidences: Optional[list[float]] = None,
        rf_probabilities: Optional[list[Dict[str, float]]] = None,
    ) -> list[DetectionDecision]:
        """Apply decision logic to a batch of samples.

        Args:
            anomaly_flags: List of anomaly flags from Isolation Forest.
            rf_classes: List of RF predicted classes.
            rf_confidences: List of RF confidences.
            rf_probabilities: List of RF probability dictionaries.

        Returns:
            List of DetectionDecision objects.
        """
        n = len(anomaly_flags)
        if rf_classes is None:
            rf_classes = [None] * n
        if rf_confidences is None:
            rf_confidences = [None] * n
        if rf_probabilities is None:
            rf_probabilities = [None] * n

        return [
            self.decide(is_anomaly, cls, conf, probs)
            for is_anomaly, cls, conf, probs in zip(
                anomaly_flags, rf_classes, rf_confidences, rf_probabilities,
            )
        ]

    def get_config(self) -> Dict[str, float]:
        """Get the current decision configuration.

        Returns:
            Dictionary with decision thresholds.
        """
        return {
            "minimum_confidence": self.minimum_confidence,
            "classification_threshold": self.classification_threshold,
        }
