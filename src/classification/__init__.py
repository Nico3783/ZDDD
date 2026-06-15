"""Random Forest classification system.

Provides the core classification model, training pipeline, inference
pipeline, feature importance analysis, and evaluation for DoS attack classification.
"""

from src.classification.evaluator import (
    evaluate_classification_model,
    evaluate_confidence_calibration,
)
from src.classification.importance import (
    analyze_feature_redundancy,
    get_feature_importance,
    get_feature_importance_by_class,
    get_top_features,
    plot_feature_importance,
    suggest_feature_selection,
)
from src.classification.inference import (
    ClassificationPrediction,
    batch_classify,
    classify_single,
    classify_traffic,
    classify_with_confidence_filter,
    predictions_to_dataframe,
)
from src.classification.random_forest import RandomForestClassifierModel
from src.classification.trainer import (
    evaluate_classifier,
    train_classifier,
    train_classifier_with_validation,
)

__all__ = [
    # Model
    "RandomForestClassifierModel",
    # Trainer
    "train_classifier",
    "train_classifier_with_validation",
    "evaluate_classifier",
    # Evaluator
    "evaluate_classification_model",
    "evaluate_confidence_calibration",
    # Inference
    "ClassificationPrediction",
    "batch_classify",
    "classify_single",
    "classify_traffic",
    "classify_with_confidence_filter",
    "predictions_to_dataframe",
    # Importance
    "analyze_feature_redundancy",
    "get_feature_importance",
    "get_feature_importance_by_class",
    "get_top_features",
    "plot_feature_importance",
    "suggest_feature_selection",
]
