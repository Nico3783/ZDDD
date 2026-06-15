from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.classification.random_forest import RandomForestClassifierModel
from src.core.config import get_config
from src.features.selector import select_features_by_config

logger = logging.getLogger(__name__)


def train_classifier(
    features: pd.DataFrame,
    labels: pd.Series,
    auto_select_features: bool = True,
) -> RandomForestClassifierModel:
    """Train the Random Forest classifier.

    This is the primary training entry point. It handles:
        1. Feature selection from config
        2. Model training with configured hyperparameters

    Args:
        features: Training features.
        labels: Training labels.
        auto_select_features: Whether to auto-select features from config.

    Returns:
        Trained RandomForestClassifierModel instance.
    """
    if auto_select_features:
        features = select_features_by_config(features)

    # Remove any non-numeric columns that might have slipped through
    numeric_cols = features.select_dtypes(include=["number"]).columns
    features = features[numeric_cols]

    logger.info(
        "Training Random Forest on %d samples, %d features, %d classes",
        len(features),
        len(features.columns),
        labels.nunique(),
    )

    model = RandomForestClassifierModel()
    model.train(features, labels)

    logger.info(
        "Classifier trained: n_estimators=%d, max_depth=%s, oob_score=%s",
        model.training_stats.get("n_estimators", 0),
        model.training_stats.get("max_depth", "None"),
        model.training_stats.get("oob_score", "N/A"),
    )

    return model


def train_classifier_with_validation(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features: Optional[pd.DataFrame] = None,
    val_labels: Optional[pd.Series] = None,
    auto_select_features: bool = True,
) -> tuple[RandomForestClassifierModel, Optional[dict]]:
    """Train classifier with optional validation evaluation.

    Args:
        train_features: Training features.
        train_labels: Training labels.
        val_features: Validation features (optional).
        val_labels: Validation labels (optional).
        auto_select_features: Whether to auto-select features.

    Returns:
        Tuple of (trained model, validation metrics or None).
    """
    model = train_classifier(train_features, train_labels, auto_select_features)

    val_metrics = None
    if val_features is not None and val_labels is not None:
        val_metrics = evaluate_classifier(model, val_features, val_labels)
        logger.info(
            "Validation metrics: accuracy=%.4f, f1=%.4f",
            val_metrics["accuracy"],
            val_metrics["f1_weighted"],
        )

    return model, val_metrics


def evaluate_classifier(
    model: RandomForestClassifierModel,
    features: pd.DataFrame,
    labels: pd.Series,
) -> dict:
    """Evaluate the classifier's performance.

    Args:
        model: Trained RandomForestClassifierModel.
        features: Test features.
        labels: True labels.

    Returns:
        Dictionary with evaluation metrics.
    """
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    results = {
        "n_samples": len(features),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_weighted": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(labels, predictions, zero_division=0),
        "class_names": list(model.class_names),
    }

    # Per-class metrics
    per_class = {}
    for cls_name in model.class_names:
        cls_mask = labels == cls_name
        if cls_mask.sum() > 0:
            cls_pred = predictions[cls_mask]
            cls_true = labels[cls_mask]
            per_class[cls_name] = {
                "support": int(cls_mask.sum()),
                "accuracy": float((cls_pred == cls_true).mean()),
                "precision": float(precision_score(labels, predictions, labels=[cls_name], average=None, zero_division=0)[0]),
                "recall": float(recall_score(labels, predictions, labels=[cls_name], average=None, zero_division=0)[0]),
                "f1": float(f1_score(labels, predictions, labels=[cls_name], average=None, zero_division=0)[0]),
            }
    results["per_class"] = per_class

    logger.info(
        "Classifier evaluation: accuracy=%.4f, f1_weighted=%.4f, f1_macro=%.4f",
        results["accuracy"],
        results["f1_weighted"],
        results["f1_macro"],
    )

    return results
