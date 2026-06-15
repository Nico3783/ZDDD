#!/usr/bin/env python3
"""Evaluate trained models against test data.

Usage:
    python scripts/evaluate_models.py [--data data/processed/processed_data.csv] [--models models/trained]
"""
from __future__ import annotations

import argparse
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


def evaluate(data_path: Path, models_dir: Path) -> dict:
    """Evaluate trained models on test data.

    Args:
        data_path: Path to processed data CSV.
        models_dir: Directory containing trained models.

    Returns:
        Evaluation results dict.
    """
    import numpy as np
    import pandas as pd

    from src.anomaly_detection.isolation_forest import IsolationForestModel
    from src.classification.random_forest import RandomForestClassifierModel
    from src.evaluation.metrics import PerformanceEvaluator
    from src.evaluation.reports import generate_experiment_report, save_evaluation_report
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

    # --- Isolation Forest Evaluation ---
    if iforest_path is not None:
        logger.info("Loading Isolation Forest model from %s", iforest_path)
        iforest = IsolationForestModel.load(str(iforest_path))
        y_pred_anomaly = iforest.predict(features)
        # IsolationForest returns -1 (anomaly) / 1 (normal); convert to 0/1
        y_pred_binary = (np.array(y_pred_anomaly) == -1).astype(int)

        evaluator = PerformanceEvaluator()
        anomaly_results = evaluator.evaluate(y_true_binary, y_pred_binary)
        results["anomaly_detection"] = anomaly_results

        logger.info("Isolation Forest Results:")
        logger.info("  Accuracy:  %.4f", anomaly_results["accuracy"])
        logger.info("  Precision: %.4f", anomaly_results["precision"])
        logger.info("  Recall:    %.4f", anomaly_results["recall"])
        logger.info("  F1-Score:  %.4f", anomaly_results["f1_score"])
    else:
        logger.warning("Isolation Forest model not found in %s", models_dir)

    # --- Random Forest Evaluation ---
    if rf_path is not None:
        logger.info("Loading Random Forest model from %s", rf_path)
        rf = RandomForestClassifierModel.load(str(rf_path))
        y_pred_labels = rf.predict(features)
        # Convert string labels to integers for evaluation
        unique_labels = sorted(set(labels) | set(y_pred_labels))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        y_true_int = np.array([label_to_int[l] for l in labels])
        y_pred_int = np.array([label_to_int[l] for l in y_pred_labels])

        evaluator = PerformanceEvaluator()
        cls_results = evaluator.evaluate(y_true_int, y_pred_int)
        results["classification"] = cls_results

        logger.info("Random Forest Results:")
        logger.info("  Accuracy:  %.4f", cls_results["accuracy"])
        logger.info("  Precision: %.4f", cls_results["precision"])
        logger.info("  Recall:    %.4f", cls_results["recall"])
        logger.info("  F1-Score:  %.4f", cls_results["f1_score"])
    else:
        logger.warning("Random Forest model not found in %s", models_dir)

    # Save results
    if results:
        output_dir = str(REPORTS_DIR / "evaluation")
        report = generate_experiment_report(
            experiment_name="model_evaluation",
            y_true=y_true_binary,
            y_pred=y_pred_binary if "anomaly_detection" in results else y_pred_int,
            metadata={"data_path": str(data_path), "models_dir": str(models_dir)},
        )
        filepath = save_evaluation_report(report, output_dir=output_dir)
        logger.info("Results saved to %s", filepath)

    return results


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--data", type=Path, default=Path("data/processed/processed_data.csv"))
    parser.add_argument("--models", type=Path, default=Path("models/trained"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")

    evaluate(args.data, args.models)


if __name__ == "__main__":
    main()
