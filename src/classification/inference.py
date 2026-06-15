from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.classification.random_forest import RandomForestClassifierModel
from src.core.exceptions import ModelError

logger = logging.getLogger(__name__)


@dataclass
class ClassificationPrediction:
    """Structured result of a classification prediction."""

    predicted_label: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    is_anomalous: bool = False
    anomaly_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def classify_traffic(
    model: RandomForestClassifierModel,
    features: pd.DataFrame,
    top_k: int = 3,
) -> List[ClassificationPrediction]:
    """Classify a batch of network traffic samples.

    Args:
        model: Trained RandomForestClassifierModel.
        features: DataFrame of features to classify.
        top_k: Number of top predictions to include.

    Returns:
        List of ClassificationPrediction objects.

    Raises:
        ModelError: If the model has not been trained.
    """
    if model.model is None:
        raise ModelError("Model not trained")

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    top_preds = model.get_top_predictions(features, top_k=top_k)

    results = []
    for idx in range(len(features)):
        pred_label = predictions[idx]
        prob_row = probabilities[idx]
        max_prob = float(prob_row.max())

        # Build probability dict
        prob_dict = {}
        for cls_idx, cls_name in enumerate(model.class_names):
            prob_dict[cls_name] = float(prob_row[cls_idx])

        results.append(ClassificationPrediction(
            predicted_label=pred_label,
            confidence=max_prob,
            probabilities=prob_dict,
            is_anomalous=pred_label.strip() != "BENIGN",
            metadata={"index": idx},
        ))

    n_anomalous = sum(1 for r in results if r.is_anomalous)
    logger.info(
        "Classification: %d/%d samples classified as anomalous (%.1f%%)",
        n_anomalous,
        len(results),
        n_anomalous / len(results) * 100 if results else 0,
    )

    return results


def classify_single(
    model: RandomForestClassifierModel,
    features: pd.Series,
    top_k: int = 3,
) -> ClassificationPrediction:
    """Classify a single network traffic sample.

    Args:
        model: Trained RandomForestClassifierModel.
        features: Series of feature values.
        top_k: Number of top predictions to include.

    Returns:
        ClassificationPrediction for the single sample.
    """
    df = features.to_frame().T
    if model.feature_names:
        df = df.reindex(columns=model.feature_names, fill_value=0.0)

    predictions = classify_traffic(model, df, top_k=top_k)
    return predictions[0]


def batch_classify(
    model: RandomForestClassifierModel,
    features: pd.DataFrame,
    batch_size: int = 1000,
    top_k: int = 3,
) -> List[ClassificationPrediction]:
    """Classify traffic in batches for large datasets.

    Args:
        model: Trained RandomForestClassifierModel.
        features: DataFrame of features.
        batch_size: Number of samples per batch.
        top_k: Number of top predictions per sample.

    Returns:
        List of ClassificationPrediction objects.
    """
    all_predictions: List[ClassificationPrediction] = []
    n_batches = (len(features) + batch_size - 1) // batch_size

    for i in range(0, len(features), batch_size):
        batch = features.iloc[i:i + batch_size]
        batch_preds = classify_traffic(model, batch, top_k=top_k)
        all_predictions.extend(batch_preds)

    logger.info("Batch classification complete: %d samples in %d batches", len(features), n_batches)
    return all_predictions


def classify_with_confidence_filter(
    model: RandomForestClassifierModel,
    features: pd.DataFrame,
    min_confidence: float = 0.5,
    top_k: int = 3,
) -> List[ClassificationPrediction]:
    """Classify traffic with a minimum confidence threshold.

    Samples below the confidence threshold are marked as 'uncertain'.

    Args:
        model: Trained RandomForestClassifierModel.
        features: DataFrame of features.
        min_confidence: Minimum confidence to accept prediction.
        top_k: Number of top predictions.

    Returns:
        List of ClassificationPrediction objects.
    """
    predictions = classify_traffic(model, features, top_k=top_k)

    for pred in predictions:
        if pred.confidence < min_confidence:
            pred.metadata["low_confidence"] = True
            pred.metadata["original_label"] = pred.predicted_label
        else:
            pred.metadata["low_confidence"] = False

    n_uncertain = sum(1 for p in predictions if p.metadata.get("low_confidence"))
    logger.info(
        "Confidence filter: %d/%d samples below threshold %.2f",
        n_uncertain,
        len(predictions),
        min_confidence,
    )

    return predictions


def predictions_to_dataframe(predictions: List[ClassificationPrediction]) -> pd.DataFrame:
    """Convert predictions to a DataFrame.

    Args:
        predictions: List of ClassificationPrediction objects.

    Returns:
        DataFrame with prediction results.
    """
    records = []
    for pred in predictions:
        record = {
            "predicted_label": pred.predicted_label,
            "confidence": pred.confidence,
            "is_anomalous": pred.is_anomalous,
        }
        # Add per-class probabilities as columns
        for cls_name, prob in pred.probabilities.items():
            record[f"prob_{cls_name}"] = prob
        records.append(record)

    return pd.DataFrame(records)
