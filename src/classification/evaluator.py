from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate Random Forest classification model performance.

    Args:
        y_true: Ground truth class labels.
        y_pred: Predicted class labels.
        class_names: Optional list of class names for reporting.

    Returns:
        Dictionary with classification metrics.
    """
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    total = len(y_true)

    correct = int(np.sum(y_true == y_pred))
    accuracy = correct / total if total > 0 else 0.0

    per_class: Dict[str, Dict[str, Any]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    supports: List[int] = []

    for cls in classes:
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))
        support = int(np.sum(y_true == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        per_class[str(cls)] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    # Weighted averages
    total_support = sum(supports)
    weighted_precision = sum(p * s for p, s in zip(precisions, supports)) / total_support if total_support > 0 else 0.0
    weighted_recall = sum(r * s for r, s in zip(recalls, supports)) / total_support if total_support > 0 else 0.0
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support if total_support > 0 else 0.0

    # Macro averages
    n_classes = len(classes)
    macro_precision = sum(precisions) / n_classes if n_classes > 0 else 0.0
    macro_recall = sum(recalls) / n_classes if n_classes > 0 else 0.0
    macro_f1 = sum(f1s) / n_classes if n_classes > 0 else 0.0

    results: Dict[str, Any] = {
        "accuracy": accuracy,
        "total_samples": total,
        "n_classes": n_classes,
        "weighted": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1_score": weighted_f1,
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
        },
        "per_class": per_class,
    }

    logger.info(
        "Classification evaluation: accuracy=%.4f, weighted_f1=%.4f, classes=%d",
        accuracy,
        weighted_f1,
        n_classes,
    )

    return results


def evaluate_confidence_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Evaluate confidence calibration of the classifier.

    Measures how well predicted confidence aligns with actual accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        confidences: Prediction confidence scores.
        n_bins: Number of calibration bins.

    Returns:
        Dictionary with calibration metrics.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies: List[float] = []
    bin_confidences: List[float] = []
    bin_counts: List[int] = []

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if not np.any(mask):
            continue

        bin_correct = float(np.mean(y_pred[mask] == y_true[mask]))
        bin_conf = float(np.mean(confidences[mask]))

        bin_accuracies.append(bin_correct)
        bin_confidences.append(bin_conf)
        bin_counts.append(int(np.sum(mask)))

    # Expected Calibration Error (ECE)
    total = sum(bin_counts)
    ece = 0.0
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        weight = count / total if total > 0 else 0.0
        ece += weight * abs(acc - conf)

    return {
        "ece": ece,
        "n_bins": n_bins,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }
