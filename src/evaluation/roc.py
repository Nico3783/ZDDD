from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve for binary classification.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_scores: Predicted scores/probabilities.

    Returns:
        Tuple of (fpr, tpr, thresholds).
    """
    sorted_indices = np.argsort(-y_scores)
    sorted_true = y_true[sorted_indices]

    tp = np.cumsum(sorted_true)
    fp = np.cumsum(1 - sorted_true)

    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    tpr = tp / total_pos if total_pos > 0 else np.zeros_like(tp, dtype=float)
    fpr = fp / total_neg if total_neg > 0 else np.zeros_like(fp, dtype=float)

    # Add origin point
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[1.0], y_scores[sorted_indices]])

    return fpr, tpr, thresholds


def compute_roc_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """Compute ROC-AUC score using trapezoidal rule.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted scores.

    Returns:
        ROC-AUC score between 0.0 and 1.0.
    """
    classes = np.unique(y_true)
    if len(classes) < 2:
        return 0.5

    if len(classes) == 2:
        fpr, tpr, _ = compute_roc_curve(y_true, y_scores)
        return float(np.trapezoid(tpr, fpr))

    # Multi-class: one-vs-rest
    aucs: List[float] = []
    supports: List[int] = []

    for cls in classes:
        binary_true = (y_true == cls).astype(int)
        class_idx = list(classes).index(cls)
        class_scores = y_scores if y_scores.ndim == 1 else y_scores[:, class_idx]

        fpr, tpr, _ = compute_roc_curve(binary_true, class_scores)
        auc = float(np.trapezoid(tpr, fpr))
        aucs.append(auc)
        supports.append(int(np.sum(binary_true)))

    total = sum(supports)
    weights = [s / total if total > 0 else 0 for s in supports]
    return float(sum(a * w for a, w in zip(aucs, weights)))


def compute_multiclass_roc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute per-class ROC curves and AUC for multi-class classification.

    Args:
        y_true: Ground truth labels.
        y_scores: Prediction scores/probabilities (n_samples x n_classes).
        labels: Optional class labels.

    Returns:
        Dictionary with per-class ROC data and AUC scores.
    """
    classes = np.unique(y_true)
    if labels is None:
        labels = [str(c) for c in classes]

    per_class_roc: Dict[str, Dict[str, Any]] = {}

    for i, cls in enumerate(classes):
        binary_true = (y_true == cls).astype(int)
        class_scores = y_scores if y_scores.ndim == 1 else y_scores[:, i]

        fpr, tpr, thresholds = compute_roc_curve(binary_true, class_scores)
        auc = float(np.trapezoid(tpr, fpr))

        per_class_roc[str(cls)] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": auc,
            "support": int(np.sum(binary_true)),
        }

    # Weighted average AUC
    supports = [per_class_roc[l]["support"] for l in labels if l in per_class_roc]
    aucs = [per_class_roc[l]["auc"] for l in labels if l in per_class_roc]
    total = sum(supports)
    weights = [s / total if total > 0 else 0 for s in supports]
    weighted_auc = float(sum(a * w for a, w in zip(aucs, weights)))

    return {
        "per_class": per_class_roc,
        "weighted_auc": weighted_auc,
        "labels": labels,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Find the optimal classification threshold.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted scores.
        metric: Metric to optimize ('f1', 'accuracy', 'youden').

    Returns:
        Tuple of (optimal_threshold, best_score).
    """
    sorted_indices = np.argsort(-y_scores)
    sorted_true = y_true[sorted_indices]

    best_threshold = 0.5
    best_score = 0.0

    thresholds = np.unique(y_scores)

    for threshold in thresholds:
        binary_pred = (y_scores >= threshold).astype(int)

        tp = int(np.sum((binary_pred == 1) & (y_true == 1)))
        fp = int(np.sum((binary_pred == 1) & (y_true == 0)))
        fn = int(np.sum((binary_pred == 0) & (y_true == 1)))
        tn = int(np.sum((binary_pred == 0) & (y_true == 0)))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "accuracy":
            score = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        elif metric == "youden":
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            score = tpr - fpr
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    logger.info("Optimal threshold (metric=%s): %.4f (score=%.4f)", metric, best_threshold, best_score)
    return best_threshold, best_score
