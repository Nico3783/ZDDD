from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.config import get_config
from src.core.constants import REPORTS_DIR

logger = logging.getLogger(__name__)


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    report_name: str = "evaluation_report",
    latency_stats: Optional[Dict[str, Any]] = None,
    throughput_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive evaluation report.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_scores: Optional prediction scores for ROC-AUC.
        labels: Optional class labels.
        report_name: Name for the report.
        latency_stats: Optional latency statistics from LatencyTracker.get_statistics().
        throughput_stats: Optional throughput statistics from ThroughputTracker.get_statistics().

    Returns:
        Dictionary with full evaluation report.
    """
    from src.evaluation.metrics import PerformanceEvaluator

    evaluator = PerformanceEvaluator()
    evaluation = evaluator.evaluate(y_true, y_pred, y_scores, labels)

    report = {
        "report_name": report_name,
        "summary": {
            "accuracy": evaluation["accuracy"],
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1_score": evaluation["f1_score"],
            "total_samples": evaluation["total_samples"],
        },
        "per_class": evaluation["per_class"],
        "confusion_matrix": evaluation["confusion_matrix"],
    }

    if "roc_auc" in evaluation:
        report["summary"]["roc_auc"] = evaluation["roc_auc"]

    if latency_stats:
        report["latency"] = latency_stats

    if throughput_stats:
        report["throughput"] = throughput_stats

    logger.info("Evaluation report generated: %s", report_name)
    return report


def generate_experiment_report(
    experiment_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    latency_tracker: Optional[Any] = None,
    throughput_tracker: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a full experiment report combining all evaluation dimensions.

    Args:
        experiment_name: Human-readable experiment identifier.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_scores: Optional prediction scores for ROC-AUC.
        labels: Optional class labels.
        latency_tracker: Optional LatencyTracker instance for latency data.
        throughput_tracker: Optional ThroughputTracker instance for throughput data.
        metadata: Optional experiment metadata (dataset, model params, etc.).

    Returns:
        Dictionary with complete experiment report.
    """
    from datetime import datetime, timezone

    from src.evaluation.metrics import PerformanceEvaluator

    evaluator = PerformanceEvaluator()
    evaluation = evaluator.evaluate(y_true, y_pred, y_scores, labels)

    report: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "summary": {
            "accuracy": evaluation["accuracy"],
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1_score": evaluation["f1_score"],
            "total_samples": evaluation["total_samples"],
        },
        "per_class": evaluation["per_class"],
        "confusion_matrix": evaluation["confusion_matrix"],
    }

    if "roc_auc" in evaluation:
        report["summary"]["roc_auc"] = evaluation["roc_auc"]

    if latency_tracker is not None:
        report["latency"] = latency_tracker.get_statistics()
        report["summary"]["overall_latency"] = latency_tracker.get_overall_latency()

    if throughput_tracker is not None:
        report["throughput"] = throughput_tracker.get_statistics()

    logger.info("Experiment report generated: %s", experiment_name)
    return report


def format_experiment_report(report: Dict[str, Any]) -> str:
    """Format an experiment report as human-readable text.

    Args:
        report: Experiment report dictionary from generate_experiment_report.

    Returns:
        Formatted text report.
    """
    summary = report.get("summary", {})
    per_class = report.get("per_class", {})
    latency = report.get("latency", {})
    throughput = report.get("throughput", {})
    metadata = report.get("metadata", {})

    lines = [
        "=" * 70,
        f"  EXPERIMENT REPORT: {report.get('experiment_name', 'N/A')}",
        f"  Timestamp: {report.get('timestamp', 'N/A')}",
        "=" * 70,
    ]

    if metadata:
        lines.append("")
        lines.append("  Metadata:")
        for key, value in metadata.items():
            lines.append(f"    {key}: {value}")

    lines.append("")
    lines.append("  Classification Metrics:")
    lines.append(f"    Total Samples: {summary.get('total_samples', 0)}")
    lines.append(f"    Accuracy:      {summary.get('accuracy', 0):.4f}")
    lines.append(f"    Precision:     {summary.get('precision', 0):.4f}")
    lines.append(f"    Recall:        {summary.get('recall', 0):.4f}")
    lines.append(f"    F1-Score:      {summary.get('f1_score', 0):.4f}")

    if "roc_auc" in summary:
        lines.append(f"    ROC-AUC:       {summary['roc_auc']:.4f}")

    if per_class:
        lines.append("")
        lines.append("  Per-Class Metrics:")
        lines.append(
            f"  {'Class':>25} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}"
        )
        lines.append("  " + "-" * 57)

        for cls, metrics in sorted(per_class.items()):
            lines.append(
                f"  {cls:>25} "
                f"{metrics.get('precision', 0):>8.4f} "
                f"{metrics.get('recall', 0):>8.4f} "
                f"{metrics.get('f1_score', 0):>8.4f} "
                f"{metrics.get('support', 0):>8d}"
            )

    if latency:
        lines.append("")
        lines.append("  Latency Analysis:")
        if "overall" in latency:
            overall = latency["overall"]
            lines.append(f"    Overall Mean Latency:  {overall.get('mean_ms', 0):.3f} ms")
            lines.append(f"    Overall P95 Latency:   {overall.get('p95_ms', 0):.3f} ms")
            lines.append(f"    Overall P99 Latency:   {overall.get('p99_ms', 0):.3f} ms")
            lines.append(f"    Overall Total Time:    {overall.get('total_ms', 0):.1f} ms")
        for op_name in sorted(latency.keys()):
            if op_name == "overall":
                continue
            op = latency[op_name]
            lines.append(
                f"    {op_name:>20}: "
                f"mean={op.get('mean_ms', 0):.3f}ms "
                f"p95={op.get('p95_ms', 0):.3f}ms "
                f"n={op.get('count', 0)}"
            )

    if throughput:
        lines.append("")
        lines.append("  Throughput Analysis:")
        for op_name in sorted(throughput.keys()):
            op = throughput[op_name]
            lines.append(
                f"    {op_name:>20}: "
                f"avg={op.get('avg_rate', 0):.1f}/s "
                f"peak={op.get('max_rate', 0):.1f}/s "
                f"total={op.get('total_samples', 0)}"
            )

    lines.append("=" * 70)
    return "\n".join(lines)


def save_evaluation_report(
    report: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> str:
    """Save evaluation report to JSON file.

    Args:
        report: Report dictionary from generate_evaluation_report.
        output_dir: Output directory. If None, uses config default.

    Returns:
        Path to saved report file.
    """
    import json
    import os
    from datetime import datetime

    if output_dir is None:
        output_dir = str(REPORTS_DIR)

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = report.get("report_name", "evaluation")
    filepath = os.path.join(output_dir, f"{report_name}_{timestamp}.json")

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Evaluation report saved: %s", filepath)
    return filepath


def format_report_summary(report: Dict[str, Any]) -> str:
    """Format evaluation report as human-readable text.

    Args:
        report: Report dictionary.

    Returns:
        Formatted text summary.
    """
    summary = report.get("summary", {})
    per_class = report.get("per_class", {})

    lines = [
        "=" * 60,
        f"  EVALUATION REPORT: {report.get('report_name', 'N/A')}",
        "=" * 60,
        f"  Total Samples: {summary.get('total_samples', 0)}",
        f"  Accuracy:      {summary.get('accuracy', 0):.4f}",
        f"  Precision:     {summary.get('precision', 0):.4f}",
        f"  Recall:        {summary.get('recall', 0):.4f}",
        f"  F1-Score:      {summary.get('f1_score', 0):.4f}",
    ]

    if "roc_auc" in summary:
        lines.append(f"  ROC-AUC:       {summary['roc_auc']:.4f}")

    lines.append("")
    lines.append("  Per-Class Metrics:")
    lines.append(f"  {'Class':>25} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}")
    lines.append("  " + "-" * 57)

    for cls, metrics in sorted(per_class.items()):
        lines.append(
            f"  {cls:>25} "
            f"{metrics.get('precision', 0):>8.4f} "
            f"{metrics.get('recall', 0):>8.4f} "
            f"{metrics.get('f1_score', 0):>8.4f} "
            f"{metrics.get('support', 0):>8d}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)
