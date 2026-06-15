#!/usr/bin/env python3
"""Evaluate trained models and generate dissertation-grade visual reports.

Usage:
    python scripts/evaluate_models.py [--data data/processed/processed_data.csv] [--models models/trained]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_model(models_dir: Path, name: str) -> Path | None:
    """Find a model file by name, checking .joblib and .pkl extensions."""
    for ext in (".joblib", ".pkl"):
        path = models_dir / f"{name}{ext}"
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def evaluate(data_path: Path, models_dir: Path, output_dir: str = "reports/figures") -> dict:
    """Evaluate trained models and generate all visual reports.

    Args:
        data_path: Path to processed data CSV.
        models_dir: Directory containing trained models.
        output_dir: Directory for figure output.

    Returns:
        Evaluation results dict.
    """
    import numpy as np
    import pandas as pd

    from src.anomaly_detection.isolation_forest import IsolationForestModel
    from src.classification.random_forest import RandomForestClassifierModel
    from src.evaluation.metrics import PerformanceEvaluator
    from src.evaluation.reports import generate_experiment_report, save_evaluation_report
    from src.evaluation.visualizations import generate_all_visualizations
    from src.classification.importance import get_feature_importance
    from src.core.constants import REPORTS_DIR

    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return {}

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    label_col = "Label" if "Label" in df.columns else "label"
    labels = df[label_col].values

    feature_cols = [
        c for c in df.columns
        if c not in [label_col, "Flow ID", "Source IP", "Destination IP",
                      "Source Port", "Destination Port", "Timestamp"]
    ]
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Binary labels: BENIGN=0, attack=1
    y_true_binary = (labels != "BENIGN").astype(int)

    # Load trained models
    iforest_path = _find_model(models_dir, "isolation_forest")
    rf_path = _find_model(models_dir, "random_forest")

    if iforest_path is None and rf_path is None:
        logger.error(
            "No trained models found in %s. Run training first:\n"
            "  python scripts/train_iforest.py\n"
            "  python scripts/train_random_forest.py",
            models_dir,
        )
        return {}

    results = {}

    # Data containers for visualizations
    if_metrics: dict = {}
    rf_metrics: dict = {}
    if_per_class: dict = {}
    rf_per_class: dict = {}
    y_pred_binary = None
    if_scores = None
    if_threshold = 0.5
    y_pred_int = None
    y_true_int = None
    rf_probabilities = None
    class_names = []
    feature_names = []
    feature_importances = np.array([])

    # --- Isolation Forest Evaluation ---
    if iforest_path is not None:
        logger.info("Loading Isolation Forest model from %s", iforest_path)
        iforest = IsolationForestModel.load(str(iforest_path))

        # Get predictions (0=normal, 1=anomaly)
        y_pred_binary_raw = iforest.predict(features)
        y_pred_binary = (np.array(y_pred_binary_raw) == -1).astype(int)

        # Get anomaly scores (normalized 0-1)
        if_scores_raw = iforest.compute_anomaly_scores(features)
        if_scores = np.array(if_scores_raw).flatten()
        if_threshold = getattr(iforest, "threshold", 0.5)

        evaluator = PerformanceEvaluator()
        if_results = evaluator.evaluate(y_true_binary, y_pred_binary, y_scores=if_scores)
        results["anomaly_detection"] = if_results
        if_metrics = {
            "accuracy": if_results["accuracy"],
            "precision": if_results["precision"],
            "recall": if_results["recall"],
            "f1_score": if_results["f1_score"],
        }
        if_per_class = if_results.get("per_class", {})
        # Map numeric class keys to readable names for binary
        if_per_class_named = {}
        for cls_key, cls_m in if_per_class.items():
            name = "Benign" if cls_key == 0 else "Attack"
            if_per_class_named[name] = cls_m
        if_per_class = if_per_class_named

        logger.info("Isolation Forest Results:")
        logger.info("  Accuracy:  %.4f", if_results["accuracy"])
        logger.info("  Precision: %.4f", if_results["precision"])
        logger.info("  Recall:    %.4f", if_results["recall"])
        logger.info("  F1-Score:  %.4f", if_results["f1_score"])
    else:
        logger.warning("Isolation Forest model not found in %s", models_dir)

    # --- Random Forest Evaluation ---
    if rf_path is not None:
        logger.info("Loading Random Forest model from %s", rf_path)
        rf = RandomForestClassifierModel.load(str(rf_path))
        y_pred_labels = rf.predict(features)

        # Get RF feature names and importances for visualization
        feature_names = list(rf.feature_names) if rf.feature_names else feature_cols
        importance_df = get_feature_importance(rf)
        feature_importances = importance_df["importance"].values

        # Get probabilities for ROC curve
        try:
            rf_probabilities = rf.predict_proba(features)
        except Exception as e:
            logger.warning("Could not get RF probabilities: %s", e)
            rf_probabilities = None

        # Map string labels to integers, preserving class order
        unique_true = sorted(set(labels))
        unique_pred = sorted(set(y_pred_labels))
        all_classes = sorted(set(unique_true) | set(unique_pred))
        label_to_int = {l: i for i, l in enumerate(all_classes)}
        class_names = all_classes  # Keep string class names for plot labels

        y_true_int = np.array([label_to_int[l] for l in labels])
        y_pred_int = np.array([label_to_int[l] for l in y_pred_labels])

        evaluator = PerformanceEvaluator()
        cls_results = evaluator.evaluate(y_true_int, y_pred_int, y_scores=rf_probabilities)
        results["classification"] = cls_results
        rf_metrics = {
            "accuracy": cls_results["accuracy"],
            "precision": cls_results["precision"],
            "recall": cls_results["recall"],
            "f1_score": cls_results["f1_score"],
        }
        # Map numeric class keys back to class names
        rf_per_class = {}
        int_to_label = {v: k for k, v in label_to_int.items()}
        for cls_key, cls_m in cls_results.get("per_class", {}).items():
            rf_per_class[int_to_label[cls_key]] = cls_m

        logger.info("Random Forest Results:")
        logger.info("  Accuracy:  %.4f", cls_results["accuracy"])
        logger.info("  Precision: %.4f", cls_results["precision"])
        logger.info("  Recall:    %.4f", cls_results["recall"])
        logger.info("  F1-Score:  %.4f", cls_results["f1_score"])
    else:
        logger.warning("Random Forest model not found in %s", models_dir)

    # --- Generate Visualizations ---
    figures_dir = REPORTS_DIR / "evaluation" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if (y_pred_binary is not None and if_scores is not None
            and if_metrics and y_pred_int is not None and y_true_int is not None
            and rf_metrics and rf_probabilities is not None and feature_names):

        logger.info("Generating dissertation-grade visual reports...")
        saved_plots = generate_all_visualizations(
            y_true_binary=y_true_binary,
            y_pred_binary=y_pred_binary,
            if_scores=if_scores,
            if_threshold=if_threshold,
            y_true_multiclass=y_true_int,
            y_pred_multiclass=y_pred_int,
            rf_probabilities=rf_probabilities,
            class_names=class_names,
            feature_names=feature_names,
            feature_importances=feature_importances,
            if_metrics=if_metrics,
            rf_metrics=rf_metrics,
            if_per_class=if_per_class,
            rf_per_class=rf_per_class,
            output_dir=str(figures_dir),
        )
        logger.info("Generated %d visual plots:", len(saved_plots))
        for p in saved_plots:
            logger.info("  -> %s", p)
        results["figures"] = saved_plots
    else:
        logger.warning("Insufficient data to generate all visualizations")
        if not feature_names:
            logger.warning("  RF feature_names empty")
        if rf_probabilities is None:
            logger.warning("  RF probabilities unavailable")

    # --- Save evaluation report (JSON) ---
    if results:
        output_json_dir = str(REPORTS_DIR / "evaluation")
        report = generate_experiment_report(
            experiment_name="model_evaluation",
            y_true=y_true_binary,
            y_pred=y_pred_binary if y_pred_binary is not None else y_pred_int,
            metadata={
                "data_path": str(data_path),
                "models_dir": str(models_dir),
                "if_threshold": float(if_threshold),
                "n_features": len(feature_cols),
                "figures_dir": str(figures_dir),
            },
        )
        filepath = save_evaluation_report(report, output_dir=output_json_dir)
        logger.info("Evaluation report saved to %s", filepath)
        results["report_path"] = filepath

    return results


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--data", type=Path, default=Path("data/processed/processed_data.csv"))
    parser.add_argument("--models", type=Path, default=Path("models/trained"))
    parser.add_argument("--output", type=str, default="reports/figures", help="Output dir for figures")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")

    evaluate(args.data, args.models, output_dir=args.output)


if __name__ == "__main__":
    main()
