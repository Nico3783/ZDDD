from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.anomaly_detection.isolation_forest import IsolationForestModel
from src.core.config import get_config
from src.core.exceptions import ModelError

logger = logging.getLogger(__name__)


@dataclass
class AnomalyPrediction:
    """Structured result of an anomaly detection prediction."""

    is_anomaly: bool
    anomaly_score: float
    raw_score: float
    confidence: float
    severity: str
    features_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def classify_severity(score: float) -> str:
    """Classify an anomaly score into a severity level.

    Severity levels are defined in config/thresholds.yaml.

    Args:
        score: Normalized anomaly score (0-1, higher = more anomalous).

    Returns:
        Severity string: 'critical', 'high', 'medium', or 'low'.
    """
    threshold_cfg = get_config().load("thresholds")
    severity_levels = threshold_cfg.get("alerting", {}).get("severity_levels", {})

    # Sort by min_score descending to find the matching level
    sorted_levels = sorted(
        severity_levels.items(),
        key=lambda x: x[1].get("min_score", 0),
        reverse=True,
    )

    for level_name, level_cfg in sorted_levels:
        min_score = level_cfg.get("min_score", 0)
        if score >= min_score:
            return level_name

    return "low"


def detect_anomaly(
    model: IsolationForestModel,
    features: pd.DataFrame,
    threshold: Optional[float] = None,
) -> List[AnomalyPrediction]:
    """Run anomaly detection on a batch of samples.

    Args:
        model: Trained IsolationForestModel.
        features: DataFrame of features to score.
        threshold: Optional override for the anomaly threshold.

    Returns:
        List of AnomalyPrediction objects, one per sample.

    Raises:
        ModelError: If the model has not been trained.
    """
    if model.model is None:
        raise ModelError("Model not trained")

    scores = model.score_samples(features)
    normalized_scores = model.compute_anomaly_scores(features)
    threshold = threshold or model.threshold

    predictions = []
    for idx in range(len(features)):
        raw = float(scores[idx])
        norm = float(normalized_scores.iloc[idx])
        is_anomaly = norm > threshold
        severity = classify_severity(norm) if is_anomaly else "low"
        confidence = min(abs(norm - threshold) / threshold, 1.0) if threshold > 0 else 0.0

        predictions.append(AnomalyPrediction(
            is_anomaly=is_anomaly,
            anomaly_score=norm,
            raw_score=raw,
            confidence=confidence,
            severity=severity,
            features_used=model.feature_names,
            metadata={"index": int(features.index[idx]) if hasattr(features.index[idx], '__int__') else idx},
        ))

    n_anomalies = sum(1 for p in predictions if p.is_anomaly)
    logger.info(
        "Anomaly detection: %d/%d samples flagged as anomalies (%.1f%%)",
        n_anomalies,
        len(predictions),
        n_anomalies / len(predictions) * 100 if predictions else 0,
    )

    return predictions


def detect_single(
    model: IsolationForestModel,
    features: pd.Series,
    threshold: Optional[float] = None,
) -> AnomalyPrediction:
    """Run anomaly detection on a single sample.

    Args:
        model: Trained IsolationForestModel.
        features: Series of feature values for one sample.
        threshold: Optional override for the anomaly threshold.

    Returns:
        AnomalyPrediction for the single sample.
    """
    # Convert Series to DataFrame for consistent API
    df = features.to_frame().T
    # Ensure column order matches training
    if model.feature_names:
        df = df.reindex(columns=model.feature_names, fill_value=0.0)

    predictions = detect_anomaly(model, df, threshold)
    return predictions[0]


def batch_detect(
    model: IsolationForestModel,
    features: pd.DataFrame,
    batch_size: int = 1000,
    threshold: Optional[float] = None,
) -> List[AnomalyPrediction]:
    """Run anomaly detection in batches for large datasets.

    Args:
        model: Trained IsolationForestModel.
        features: DataFrame of features to score.
        batch_size: Number of samples per batch.
        threshold: Optional override for the anomaly threshold.

    Returns:
        List of AnomalyPrediction objects for all samples.
    """
    all_predictions: List[AnomalyPrediction] = []
    n_batches = (len(features) + batch_size - 1) // batch_size

    for i in range(0, len(features), batch_size):
        batch = features.iloc[i:i + batch_size]
        batch_preds = detect_anomaly(model, batch, threshold)
        all_predictions.extend(batch_preds)

    logger.info("Batch detection complete: %d samples in %d batches", len(features), n_batches)
    return all_predictions


def predictions_to_dataframe(predictions: List[AnomalyPrediction]) -> pd.DataFrame:
    """Convert a list of AnomalyPrediction objects to a DataFrame.

    Args:
        predictions: List of AnomalyPrediction objects.

    Returns:
        DataFrame with prediction results.
    """
    records = []
    for pred in predictions:
        records.append({
            "is_anomaly": pred.is_anomaly,
            "anomaly_score": pred.anomaly_score,
            "raw_score": pred.raw_score,
            "confidence": pred.confidence,
            "severity": pred.severity,
        })

    return pd.DataFrame(records)
