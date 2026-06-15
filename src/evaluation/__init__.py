"""Evaluation module.

Computes accuracy, precision, recall, F1-score, ROC-AUC,
confusion matrix, per-class classification reports, latency,
and throughput metrics.
"""

from src.evaluation.confusion import (
    compute_confusion_matrix,
    compute_false_positive_rate,
    confusion_matrix_to_dataframe,
    format_confusion_matrix,
)
from src.evaluation.latency import LatencyMeasurement, LatencyTracker
from src.evaluation.metrics import PerformanceEvaluator
from src.evaluation.reports import (
    format_experiment_report,
    format_report_summary,
    generate_evaluation_report,
    generate_experiment_report,
    save_evaluation_report,
)
from src.evaluation.roc import (
    compute_multiclass_roc,
    compute_roc_auc,
    compute_roc_curve,
    find_optimal_threshold,
)
from src.evaluation.throughput import ThroughputMeasurement, ThroughputTracker

__all__ = [
    "LatencyMeasurement",
    "LatencyTracker",
    "PerformanceEvaluator",
    "ThroughputMeasurement",
    "ThroughputTracker",
    "compute_confusion_matrix",
    "compute_false_positive_rate",
    "compute_multiclass_roc",
    "compute_roc_auc",
    "compute_roc_curve",
    "confusion_matrix_to_dataframe",
    "find_optimal_threshold",
    "format_confusion_matrix",
    "format_experiment_report",
    "format_report_summary",
    "generate_evaluation_report",
    "generate_experiment_report",
    "save_evaluation_report",
]
