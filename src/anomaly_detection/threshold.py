from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.anomaly_detection.isolation_forest import IsolationForestModel
from src.core.config import get_config

logger = logging.getLogger(__name__)


def optimize_threshold(
    model: IsolationForestModel,
    features: pd.DataFrame,
    true_labels: pd.Series,
    metric: str = "f1",
    step: float = 0.001,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
) -> tuple[float, dict]:
    """Find the optimal anomaly score threshold using grid search.

    Args:
        model: Trained IsolationForestModel.
        features: Validation features.
        true_labels: Ground truth labels (BENIGN vs attack).
        metric: Optimization metric ('f1', 'precision', 'recall').
        step: Step size for threshold search.
        min_threshold: Minimum threshold to search.
        max_threshold: Maximum threshold to search.

    Returns:
        Tuple of (best_threshold, results_dict_with_all_metrics).
    """
    # Get normalized anomaly scores
    normalized_scores = model.compute_anomaly_scores(features)

    # Convert labels to binary (0=benign, 1=attack)
    binary_labels = (true_labels.str.strip() != "BENIGN").astype(int)

    best_score = -1.0
    best_threshold = 0.5
    results = {"thresholds": [], "scores": []}

    thresholds = np.arange(min_threshold, max_threshold + step, step)

    for t in thresholds:
        preds = (normalized_scores > t).astype(int)

        if metric == "f1":
            score = f1_score(binary_labels, preds, zero_division=0)
        elif metric == "precision":
            score = precision_score(binary_labels, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(binary_labels, preds, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        results["thresholds"].append(float(t))
        results["scores"].append(float(score))

        if score > best_score:
            best_score = score
            best_threshold = float(t)

    # Compute full metrics at best threshold
    best_preds = (normalized_scores > best_threshold).astype(int)
    final_metrics = {
        "threshold": best_threshold,
        "f1": float(f1_score(binary_labels, best_preds, zero_division=0)),
        "precision": float(precision_score(binary_labels, best_preds, zero_division=0)),
        "recall": float(recall_score(binary_labels, best_preds, zero_division=0)),
        "accuracy": float((binary_labels == best_preds).mean()),
        "optimization_metric": metric,
        "optimization_score": best_score,
    }

    # Additional stats
    final_metrics["n_samples"] = len(features)
    final_metrics["n_anomalies_detected"] = int(best_preds.sum())
    final_metrics["anomaly_rate"] = float(best_preds.mean())

    logger.info(
        "Threshold optimization (%s): best=%.4f, score=%.4f, "
        "precision=%.4f, recall=%.4f, f1=%.4f",
        metric,
        best_threshold,
        best_score,
        final_metrics["precision"],
        final_metrics["recall"],
        final_metrics["f1"],
    )

    return best_threshold, final_metrics


def auto_optimize_threshold(
    model: IsolationForestModel,
    features: pd.DataFrame,
    true_labels: pd.Series,
) -> float:
    """Automatically optimize the threshold and update the model.

    Reads optimization settings from config/models.yaml and applies
    the best threshold to the model.

    Args:
        model: Trained IsolationForestModel (will be updated in-place).
        features: Validation features.
        true_labels: Ground truth labels.

    Returns:
        The optimized threshold value.
    """
    cfg = get_config().load("models")
    opt_cfg = cfg.get("threshold_optimization", {})

    metric = opt_cfg.get("metric", "f1")
    step = opt_cfg.get("step", 0.001)
    min_t = opt_cfg.get("min_threshold", 0.0)
    max_t = opt_cfg.get("max_threshold", 1.0)

    best_threshold, metrics = optimize_threshold(
        model=model,
        features=features,
        true_labels=true_labels,
        metric=metric,
        step=step,
        min_threshold=min_t,
        max_threshold=max_t,
    )

    model.set_threshold(best_threshold)
    return best_threshold


def sensitivity_analysis(
    model: IsolationForestModel,
    features: pd.DataFrame,
    true_labels: pd.Series,
    thresholds: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Analyze detector sensitivity across different thresholds.

    Args:
        model: Trained IsolationForestModel.
        features: Validation features.
        true_labels: Ground truth labels.
        thresholds: List of thresholds to test. If None, uses a default range.

    Returns:
        DataFrame with metrics for each threshold.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    normalized_scores = model.compute_anomaly_scores(features)
    binary_labels = (true_labels.str.strip() != "BENIGN").astype(int)

    rows = []
    for t in thresholds:
        preds = (normalized_scores > t).astype(int)
        rows.append({
            "threshold": t,
            "f1": float(f1_score(binary_labels, preds, zero_division=0)),
            "precision": float(precision_score(binary_labels, preds, zero_division=0)),
            "recall": float(recall_score(binary_labels, preds, zero_division=0)),
            "accuracy": float((binary_labels == preds).mean()),
            "n_anomalies": int(preds.sum()),
            "anomaly_rate": float(preds.mean()),
        })

    df = pd.DataFrame(rows)
    logger.info("Sensitivity analysis complete for %d thresholds", len(thresholds))
    return df
