from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_anomaly_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate anomaly detection model performance.

    Computes detection rate, false alarm rate, and anomaly-score-based
    metrics for the Isolation Forest detector.

    Args:
        y_true: Ground truth labels (0=normal, 1=anomaly).
        y_pred: Predicted labels (0=normal, 1=anomaly).
        anomaly_scores: Optional anomaly scores for score-based analysis.

    Returns:
        Dictionary with evaluation metrics.
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * detection_rate / (precision + detection_rate) if (precision + detection_rate) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    results: Dict[str, Any] = {
        "detection_rate": detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "total_samples": len(y_true),
        "anomaly_ratio": float(np.mean(y_true)) if len(y_true) > 0 else 0.0,
    }

    if anomaly_scores is not None:
        results["score_stats"] = {
            "mean": float(np.mean(anomaly_scores)),
            "std": float(np.std(anomaly_scores)),
            "min": float(np.min(anomaly_scores)),
            "max": float(np.max(anomaly_scores)),
            "median": float(np.median(anomaly_scores)),
        }

    logger.info(
        "Anomaly model evaluation: detection_rate=%.4f, far=%.4f, f1=%.4f",
        detection_rate,
        false_alarm_rate,
        f1,
    )

    return results


def evaluate_by_attack_type(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Evaluate anomaly detection performance broken down by attack type.

    Args:
        y_true: Ground truth labels (0=normal, 1=anomaly).
        y_pred: Predicted labels (0=normal, 1=anomaly).
        attack_labels: Attack type for each sample.

    Returns:
        Dictionary mapping attack type to metrics.
    """
    unique_attacks = np.unique(attack_labels)
    results: Dict[str, Dict[str, float]] = {}

    for attack in unique_attacks:
        mask = attack_labels == attack
        if not np.any(mask):
            continue

        y_true_sub = y_true[mask]
        y_pred_sub = y_pred[mask]

        tp = int(np.sum((y_pred_sub == 1) & (y_true_sub == 1)))
        fp = int(np.sum((y_pred_sub == 1) & (y_true_sub == 0)))
        fn = int(np.sum((y_pred_sub == 0) & (y_true_sub == 1)))
        tn = int(np.sum((y_pred_sub == 0) & (y_true_sub == 0)))

        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results[str(attack)] = {
            "detection_rate": detection_rate,
            "false_alarm_rate": false_alarm_rate,
            "support": int(np.sum(mask)),
        }

    return results
