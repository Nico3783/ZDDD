"""Isolation Forest-based anomaly detection.

Provides the core anomaly detection model, training pipeline, inference
pipeline, threshold optimization, and evaluation for zero-day DoS detection.
"""

from src.anomaly_detection.evaluator import (
    evaluate_anomaly_model,
    evaluate_by_attack_type,
)
from src.anomaly_detection.inference import (
    AnomalyPrediction,
    batch_detect,
    classify_severity,
    detect_anomaly,
    detect_single,
    predictions_to_dataframe,
)
from src.anomaly_detection.isolation_forest import IsolationForestModel
from src.anomaly_detection.threshold import (
    auto_optimize_threshold,
    optimize_threshold,
    sensitivity_analysis,
)
from src.anomaly_detection.trainer import (
    evaluate_anomaly_detector,
    train_anomaly_detector,
    train_anomaly_detector_on_full_dataset,
)

__all__ = [
    # Model
    "IsolationForestModel",
    # Trainer
    "train_anomaly_detector",
    "train_anomaly_detector_on_full_dataset",
    "evaluate_anomaly_detector",
    # Evaluator
    "evaluate_anomaly_model",
    "evaluate_by_attack_type",
    # Inference
    "AnomalyPrediction",
    "batch_detect",
    "classify_severity",
    "detect_anomaly",
    "detect_single",
    "predictions_to_dataframe",
    # Threshold
    "auto_optimize_threshold",
    "optimize_threshold",
    "sensitivity_analysis",
]
