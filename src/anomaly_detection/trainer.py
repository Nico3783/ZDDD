from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.anomaly_detection.isolation_forest import IsolationForestModel
from src.core.config import get_config
from src.core.exceptions import ModelError
from src.features.selector import select_features_by_config

logger = logging.getLogger(__name__)


def train_anomaly_detector(
    features: pd.DataFrame,
    contamination: Optional[float] = None,
    auto_select_features: bool = True,
) -> IsolationForestModel:
    """Train the Isolation Forest anomaly detection model.

    This is the primary training entry point. It handles:
        1. Feature selection from config
        2. Model training with configured hyperparameters
        3. Threshold initialization

    Args:
        features: Training features (typically benign-only for unsupervised learning).
        contamination: Expected anomaly proportion. If None, reads from config.
        auto_select_features: Whether to auto-select features from config.

    Returns:
        Trained IsolationForestModel instance.
    """
    if auto_select_features:
        features = select_features_by_config(features)

    # Remove any non-numeric columns that might have slipped through
    numeric_cols = features.select_dtypes(include=["number"]).columns
    features = features[numeric_cols]

    logger.info(
        "Training Isolation Forest on %d samples, %d features",
        len(features),
        len(features.columns),
    )

    model = IsolationForestModel()
    model.train(features, contamination=contamination)

    # Set initial threshold from config using normalized scores (0-1 range)
    threshold_cfg = get_config().load("thresholds")
    initial_percentile = threshold_cfg.get("anomaly_detection", {}).get(
        "score_threshold_percentile", 95.0
    )
    normalized_scores = model.compute_anomaly_scores(features)
    initial_threshold = float(normalized_scores.quantile(initial_percentile / 100))
    model.set_threshold(initial_threshold)

    logger.info(
        "Anomaly detector trained. Threshold=%.4f (percentile=%.1f%%)",
        initial_threshold,
        initial_percentile,
    )

    return model


def train_anomaly_detector_on_full_dataset(
    df: pd.DataFrame,
    label_column: Optional[str] = None,
    contamination: Optional[float] = None,
) -> IsolationForestModel:
    """Train anomaly detector using all data (unsupervised).

    The model learns the normal data distribution and flags deviations.
    For best results, pass only benign traffic.

    Args:
        df: Full dataset with labels.
        label_column: Label column name. If None, reads from config.
        contamination: Expected anomaly proportion. If None, reads from config.

    Returns:
        Trained IsolationForestModel instance.
    """
    if label_column is None:
        cfg = get_config().load("features")
        label_column = cfg.get("categorical", {}).get("label_column", " Label").strip()

    clean_label = label_column.strip()
    if clean_label in df.columns:
        features = df.drop(columns=[clean_label])
    else:
        features = df.copy()

    return train_anomaly_detector(features, contamination=contamination)


def evaluate_anomaly_detector(
    model: IsolationForestModel,
    features: pd.DataFrame,
    true_labels: Optional[pd.Series] = None,
) -> dict:
    """Evaluate the anomaly detector's performance.

    Args:
        model: Trained IsolationForestModel.
        features: Evaluation features.
        true_labels: Ground truth labels (optional, for supervised evaluation).

    Returns:
        Dictionary with evaluation metrics.
    """
    scores = model.score_samples(features)
    predictions = model.predict(features)
    binary_predictions = model.predict_binary(features)

    results = {
        "n_samples": len(features),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "threshold": model.threshold,
        "n_predicted_anomalies": int((predictions == -1).sum()),
        "n_predicted_normal": int((predictions == 1).sum()),
        "anomaly_rate": float((predictions == -1).mean()),
    }

    if true_labels is not None:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
        )

        # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
        pred_binary = (predictions == -1).astype(int)
        true_binary = (true_labels.str.strip() != "BENIGN").astype(int)

        results.update({
            "accuracy": float(accuracy_score(true_binary, pred_binary)),
            "precision": float(precision_score(true_binary, pred_binary, zero_division=0)),
            "recall": float(recall_score(true_binary, pred_binary, zero_division=0)),
            "f1": float(f1_score(true_binary, pred_binary, zero_division=0)),
            "confusion_matrix": confusion_matrix(true_binary, pred_binary).tolist(),
        })

    logger.info(
        "Anomaly detector evaluation: anomaly_rate=%.2f%%, n_anomalies=%d",
        results["anomaly_rate"] * 100,
        results["n_predicted_anomalies"],
    )

    return results
