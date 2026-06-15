from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute confusion matrix with per-class breakdown.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Optional list of class labels.

    Returns:
        Dictionary with confusion matrix, labels, and per-class stats.
    """
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

    n_classes = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            matrix[label_to_idx[true_label]][label_to_idx[pred_label]] += 1

    # Per-class breakdown
    per_class: Dict[str, Dict[str, int]] = {}
    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        fp = int(np.sum(matrix[:, i])) - tp
        fn = int(np.sum(matrix[i, :])) - tp
        tn = int(np.sum(matrix)) - tp - fp - fn

        per_class[label] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "support": int(np.sum(matrix[i, :])),
        }

    return {
        "matrix": matrix.tolist(),
        "labels": labels,
        "per_class": per_class,
    }


def confusion_matrix_to_dataframe(
    confusion: Dict[str, Any],
) -> Any:
    """Convert confusion matrix dict to a pandas DataFrame.

    Args:
        confusion: Dictionary from compute_confusion_matrix.

    Returns:
        DataFrame with confusion matrix.
    """
    import pandas as pd

    labels = confusion.get("labels", [])
    matrix = confusion.get("matrix", [])

    return pd.DataFrame(matrix, index=labels, columns=labels)


def format_confusion_matrix(confusion: Dict[str, Any]) -> str:
    """Format confusion matrix as human-readable text.

    Args:
        confusion: Dictionary from compute_confusion_matrix.

    Returns:
        Formatted text string.
    """
    labels = confusion.get("labels", [])
    matrix = confusion.get("matrix", [])

    if not labels or not matrix:
        return "Empty confusion matrix"

    # Find max label width
    max_width = max(len(str(l)) for l in labels)
    max_width = max(max_width, 8)

    lines = ["Confusion Matrix:", ""]
    header = " " * (max_width + 2) + " ".join(f"{str(l):>8}" for l in labels)
    lines.append(header)
    lines.append(" " * (max_width + 2) + "-" * (9 * len(labels)))

    for i, label in enumerate(labels):
        row = f"{str(label):>{max_width}} | "
        row += " ".join(f"{matrix[i][j]:>8d}" for j in range(len(labels)))
        lines.append(row)

    return "\n".join(lines)


def compute_false_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_class: str,
) -> float:
    """Compute false positive rate for a specific class.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        target_class: Class to compute FPR for.

    Returns:
        False positive rate (0.0 to 1.0).
    """
    binary_true = (y_true == target_class).astype(int)
    binary_pred = (y_pred == target_class).astype(int)

    fp = int(np.sum((binary_pred == 1) & (binary_true == 0)))
    tn = int(np.sum((binary_pred == 0) & (binary_true == 0)))

    total_negatives = fp + tn
    return fp / total_negatives if total_negatives > 0 else 0.0
