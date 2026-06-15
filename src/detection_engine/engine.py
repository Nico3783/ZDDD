from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.anomaly_detection.isolation_forest import IsolationForestModel
from src.classification.random_forest import RandomForestClassifierModel
from src.core.config import get_config
from src.core.exceptions import DetectionError

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Structured result of a complete detection pipeline run."""

    sample_id: str
    is_anomaly: bool
    anomaly_score: float
    anomaly_threshold: float
    predicted_class: str
    classification_confidence: float
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    severity: str = "low"
    is_zero_day: bool = False
    detection_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionStats:
    """Aggregated detection statistics for a batch."""

    total_samples: int = 0
    anomalies_detected: int = 0
    normal_samples: int = 0
    zero_day_detected: int = 0
    severity_counts: Dict[str, int] = field(default_factory=dict)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    mean_anomaly_score: float = 0.0
    mean_classification_confidence: float = 0.0
    mean_latency_ms: float = 0.0

    @property
    def anomaly_rate(self) -> float:
        """Proportion of samples flagged as anomalies."""
        if self.total_samples == 0:
            return 0.0
        return self.anomalies_detected / self.total_samples


class DetectionEngine:
    """Orchestrates the full detection pipeline: anomaly detection → classification → alerting.

    Pipeline:
        1. Isolation Forest detects anomalies (zero-day + known attacks)
        2. Random Forest classifies known attack types
        3. Anomalies not recognized by RF are flagged as zero-day
        4. Severity is computed from anomaly score + classification confidence
    """

    def __init__(
        self,
        anomaly_model: Optional[IsolationForestModel] = None,
        classifier_model: Optional[RandomForestClassifierModel] = None,
    ) -> None:
        """Initialize the detection engine.

        Args:
            anomaly_model: Trained Isolation Forest model.
            classifier_model: Trained Random Forest classifier.
        """
        self.anomaly_model = anomaly_model
        self.classifier_model = classifier_model
        self._load_config()
        logger.info("DetectionEngine initialized")

    def _load_config(self) -> None:
        """Load detection configuration from YAML files."""
        threshold_cfg = get_config().load("thresholds")
        self.anomaly_threshold = threshold_cfg.get("anomaly_detection", {}).get(
            "score_threshold_percentile", 95.0
        )
        self.classification_threshold = threshold_cfg.get("classification", {}).get(
            "decision_threshold", 0.5
        )
        self.minimum_confidence = threshold_cfg.get("alerting", {}).get(
            "minimum_confidence", 0.5
        )
        self.severity_levels = threshold_cfg.get("alerting", {}).get("severity_levels", {})

        settings_cfg = get_config().load("settings")
        self.pipeline_cfg = settings_cfg.get("pipeline", {})

    def detect_batch(
        self,
        features: pd.DataFrame,
        sample_ids: Optional[List[str]] = None,
    ) -> List[DetectionResult]:
        """Run the full detection pipeline on a batch of samples.

        Args:
            features: DataFrame of preprocessed features.
            sample_ids: Optional list of sample identifiers.

        Returns:
            List of DetectionResult objects.

        Raises:
            DetectionError: If detection fails.
        """
        import time

        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(features))]

        results = []

        for idx in range(len(features)):
            start_time = time.time()
            sample = features.iloc[idx:idx + 1]
            sample_id = sample_ids[idx]

            try:
                result = self._detect_single(sample, sample_id)
                result.detection_latency_ms = (time.time() - start_time) * 1000
                results.append(result)
            except Exception as e:
                logger.error("Detection failed for sample %s: %s", sample_id, e)
                results.append(DetectionResult(
                    sample_id=sample_id,
                    is_anomaly=False,
                    anomaly_score=0.0,
                    anomaly_threshold=self.anomaly_threshold,
                    predicted_class="ERROR",
                    classification_confidence=0.0,
                    severity="low",
                    metadata={"error": str(e)},
                ))

        stats = self.compute_stats(results)
        logger.info(
            "Batch detection complete: %d samples, %d anomalies, %d zero-day, mean_latency=%.2fms",
            stats.total_samples,
            stats.anomalies_detected,
            stats.zero_day_detected,
            stats.mean_latency_ms,
        )

        return results

    def _detect_single(
        self,
        features: pd.DataFrame,
        sample_id: str,
    ) -> DetectionResult:
        """Run detection on a single sample.

        Args:
            features: Single-sample DataFrame.
            sample_id: Sample identifier.

        Returns:
            DetectionResult for the sample.
        """
        # Step 1: Anomaly detection
        if self.anomaly_model is not None and self.anomaly_model.model is not None:
            decision_scores = self.anomaly_model.decision_function(features)
            raw_decision = float(decision_scores[0])
            is_anomaly = raw_decision < 0
            # Map decision score to 0-1 range (sigmoid) where higher = more anomalous
            # Clip to prevent overflow in exp
            clipped = max(-500.0, min(500.0, raw_decision))
            anomaly_score = float(1.0 / (1.0 + np.exp(clipped)))
        else:
            anomaly_score = 0.0
            is_anomaly = False

        # Step 2: Classification (if anomaly detected and classifier available)
        predicted_class = "BENIGN"
        classification_confidence = 1.0
        class_probabilities: Dict[str, float] = {}

        if is_anomaly and self.classifier_model is not None and self.classifier_model.model is not None:
            try:
                probs = self.classifier_model.predict_proba(features)
                prob_row = probs[0]
                for cls_idx, cls_name in enumerate(self.classifier_model.class_names):
                    class_probabilities[cls_name] = float(prob_row[cls_idx])

                pred_idx = prob_row.argmax()
                predicted_class = self.classifier_model.class_names[pred_idx]
                classification_confidence = float(prob_row[pred_idx])
            except Exception as e:
                logger.warning("Classification failed for %s: %s", sample_id, e)
                predicted_class = "UNKNOWN"
                classification_confidence = 0.0

        # Step 3: Zero-day detection
        is_zero_day = False
        if is_anomaly:
            if predicted_class == "BENIGN" or classification_confidence < self.minimum_confidence:
                is_zero_day = True
                predicted_class = "ZERO_DAY"

        # Step 4: Severity calculation
        severity = self._compute_severity(anomaly_score, classification_confidence, is_zero_day)

        return DetectionResult(
            sample_id=sample_id,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_threshold=self.anomaly_model.threshold if self.anomaly_model else 0.5,
            predicted_class=predicted_class,
            classification_confidence=classification_confidence,
            class_probabilities=class_probabilities,
            severity=severity,
            is_zero_day=is_zero_day,
        )

    def _compute_severity(
        self,
        anomaly_score: float,
        confidence: float,
        is_zero_day: bool,
    ) -> str:
        """Compute alert severity based on anomaly score and confidence.

        Severity levels:
            critical: anomaly_score >= 0.9 OR (zero-day AND high confidence)
            high: anomaly_score >= 0.7
            medium: anomaly_score >= 0.5
            low: everything else

        Args:
            anomaly_score: Normalized anomaly score (0-1).
            confidence: Classification confidence.
            is_zero_day: Whether the sample is a zero-day anomaly.

        Returns:
            Severity string.
        """
        if anomaly_score >= 0.9:
            return "critical"
        if is_zero_day and confidence >= 0.7:
            return "critical"
        if anomaly_score >= 0.7:
            return "high"
        if anomaly_score >= 0.5:
            return "medium"
        return "low"

    def compute_stats(self, results: List[DetectionResult]) -> DetectionStats:
        """Compute aggregated detection statistics.

        Args:
            results: List of DetectionResult objects.

        Returns:
            DetectionStats with aggregated metrics.
        """
        if not results:
            return DetectionStats()

        stats = DetectionStats(
            total_samples=len(results),
            anomalies_detected=sum(1 for r in results if r.is_anomaly),
            normal_samples=sum(1 for r in results if not r.is_anomaly),
            zero_day_detected=sum(1 for r in results if r.is_zero_day),
            mean_anomaly_score=sum(r.anomaly_score for r in results) / len(results),
            mean_classification_confidence=sum(r.classification_confidence for r in results) / len(results),
            mean_latency_ms=sum(r.detection_latency_ms for r in results) / len(results),
        )

        # Severity counts
        for r in results:
            stats.severity_counts[r.severity] = stats.severity_counts.get(r.severity, 0) + 1

        # Class distribution
        for r in results:
            stats.class_distribution[r.predicted_class] = stats.class_distribution.get(r.predicted_class, 0) + 1

        return stats

    def update_models(
        self,
        anomaly_model: Optional[IsolationForestModel] = None,
        classifier_model: Optional[RandomForestClassifierModel] = None,
    ) -> None:
        """Update the detection engine with new models.

        Args:
            anomaly_model: New or updated Isolation Forest model.
            classifier_model: New or updated Random Forest classifier.
        """
        if anomaly_model is not None:
            self.anomaly_model = anomaly_model
            logger.info("Anomaly detection model updated")
        if classifier_model is not None:
            self.classifier_model = classifier_model
            logger.info("Classifier model updated")

    def results_to_dataframe(self, results: List[DetectionResult]) -> pd.DataFrame:
        """Convert detection results to a DataFrame.

        Args:
            results: List of DetectionResult objects.

        Returns:
            DataFrame with detection results.
        """
        records = []
        for r in results:
            records.append({
                "sample_id": r.sample_id,
                "is_anomaly": r.is_anomaly,
                "anomaly_score": r.anomaly_score,
                "predicted_class": r.predicted_class,
                "classification_confidence": r.classification_confidence,
                "severity": r.severity,
                "is_zero_day": r.is_zero_day,
                "detection_latency_ms": r.detection_latency_ms,
            })

        return pd.DataFrame(records)
