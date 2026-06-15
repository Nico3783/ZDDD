from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """Evaluates detection engine performance metrics.

    Computes accuracy, precision, recall, F1-score, ROC-AUC,
    confusion matrix, and classification reports.
    """

    def __init__(self) -> None:
        """Initialize the performance evaluator."""
        logger.info("PerformanceEvaluator initialized")

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute overall accuracy.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Accuracy score between 0.0 and 1.0.
        """
        return float(np.mean(y_true == y_pred))

    def compute_precision_recall_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "weighted",
    ) -> Dict[str, float]:
        """Compute precision, recall, and F1-score.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            average: Averaging method ('weighted', 'macro', 'micro').

        Returns:
            Dictionary with precision, recall, f1_score.
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
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

        total = sum(supports)

        if average == "weighted":
            weights = [s / total if total > 0 else 0 for s in supports]
            return {
                "precision": float(sum(p * w for p, w in zip(precisions, weights))),
                "recall": float(sum(r * w for r, w in zip(recalls, weights))),
                "f1_score": float(sum(f * w for f, w in zip(f1s, weights))),
            }
        elif average == "macro":
            n = len(classes)
            return {
                "precision": sum(precisions) / n if n > 0 else 0.0,
                "recall": sum(recalls) / n if n > 0 else 0.0,
                "f1_score": sum(f1s) / n if n > 0 else 0.0,
            }
        else:
            return {
                "precision": float(np.mean(precisions)),
                "recall": float(np.mean(recalls)),
                "f1_score": float(np.mean(f1s)),
            }

    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute confusion matrix.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            labels: Optional list of class labels.

        Returns:
            Dictionary with confusion matrix and labels.
        """
        if labels is None:
            labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

        n_classes = len(labels)
        label_to_idx = {label: i for i, label in enumerate(labels)}
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label in label_to_idx and pred_label in label_to_idx:
                matrix[label_to_idx[true_label]][label_to_idx[pred_label]] += 1

        return {
            "matrix": matrix.tolist(),
            "labels": labels,
            "true_labels": y_true.tolist(),
            "pred_labels": y_pred.tolist(),
        }

    def compute_roc_auc(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        average: str = "weighted",
    ) -> float:
        """Compute ROC-AUC score.

        Uses trapezoidal approximation for multi-class ROC-AUC.

        Args:
            y_true: Ground truth binary labels.
            y_scores: Predicted scores/probabilities.
            average: Averaging method.

        Returns:
            ROC-AUC score between 0.0 and 1.0.
        """
        classes = np.unique(y_true)
        if len(classes) < 2:
            return 0.5

        aucs: List[float] = []
        supports: List[int] = []

        for cls in classes:
            binary_true = (y_true == cls).astype(int)
            class_scores = y_scores if y_scores.ndim == 1 else y_scores[:, list(classes).index(cls)]

            sorted_indices = np.argsort(-class_scores)
            sorted_true = binary_true[sorted_indices]

            tp = np.cumsum(sorted_true)
            fp = np.cumsum(1 - sorted_true)

            tpr = tp / np.sum(sorted_true) if np.sum(sorted_true) > 0 else np.zeros_like(tp, dtype=float)
            fpr = fp / np.sum(1 - sorted_true) if np.sum(1 - sorted_true) > 0 else np.zeros_like(fp, dtype=float)

            # Trapezoidal AUC
            auc = float(np.trapezoid(tpr, fpr))
            aucs.append(auc)
            supports.append(int(np.sum(binary_true)))

        if average == "weighted":
            total = sum(supports)
            weights = [s / total if total > 0 else 0 for s in supports]
            return float(sum(a * w for a, w in zip(aucs, weights)))
        else:
            return float(np.mean(aucs))

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Full evaluation of detection performance.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_scores: Optional prediction scores for ROC-AUC.
            labels: Optional class labels.

        Returns:
            Dictionary with all evaluation metrics.
        """
        accuracy = self.compute_accuracy(y_true, y_pred)
        prf = self.compute_precision_recall_f1(y_true, y_pred)
        confusion = self.compute_confusion_matrix(y_true, y_pred, labels)

        result: Dict[str, Any] = {
            "accuracy": accuracy,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1_score": prf["f1_score"],
            "confusion_matrix": confusion,
            "total_samples": len(y_true),
        }

        if y_scores is not None:
            result["roc_auc"] = self.compute_roc_auc(y_true, y_scores)

        # Per-class metrics
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        per_class: Dict[str, Dict[str, float]] = {}
        for cls in classes:
            binary_true = (y_true == cls).astype(int)
            binary_pred = (y_pred == cls).astype(int)
            tp = int(np.sum((binary_pred == 1) & (binary_true == 1)))
            fp = int(np.sum((binary_pred == 1) & (binary_true == 0)))
            fn = int(np.sum((binary_pred == 0) & (binary_true == 1)))
            tn = int(np.sum((binary_pred == 0) & (binary_true == 0)))

            per_class[cls] = {
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
                "support": int(np.sum(binary_true == 1)),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
            }

        result["per_class"] = per_class

        logger.info(
            "Evaluation complete: accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f",
            accuracy,
            prf["precision"],
            prf["recall"],
            prf["f1_score"],
        )

        return result

    def to_dataframe(self, evaluation: Dict[str, Any]) -> pd.DataFrame:
        """Convert evaluation results to a DataFrame.

        Args:
            evaluation: Evaluation dictionary from evaluate().

        Returns:
            DataFrame with per-class metrics.
        """
        if "per_class" not in evaluation:
            return pd.DataFrame()

        records = []
        for cls, metrics in evaluation["per_class"].items():
            records.append({"class": cls, **metrics})

        return pd.DataFrame(records)
