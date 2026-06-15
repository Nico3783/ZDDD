#!/usr/bin/env python3
"""Generate evaluation reports from trained model results.

Usage:
    python scripts/generate_report.py [--data data/processed/processed_data.csv] [--models models/trained] [--output reports]
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


def generate(data_path: Path, models_dir: Path, output_dir: Path) -> None:
    """Generate a comprehensive evaluation report.

    Args:
        data_path: Path to processed data CSV.
        models_dir: Directory containing trained models.
        output_dir: Directory to save the report.
    """
    import numpy as np
    import pandas as pd

    from src.anomaly_detection.isolation_forest import IsolationForestModel
    from src.classification.random_forest import RandomForestClassifierModel
    from src.evaluation.reports import (
        generate_experiment_report,
        format_experiment_report,
        save_evaluation_report,
    )

    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

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

    y_true_binary = (labels != "BENIGN").astype(int).values

    # Load models and generate predictions
    iforest_path = _find_model(models_dir, "isolation_forest")
    rf_path = _find_model(models_dir, "random_forest")

    if iforest_path is None and rf_path is None:
        logger.error(
            "No trained models found in %s. Run training first:\n"
            "  python scripts/train_iforest.py\n"
            "  python scripts/train_random_forest.py",
            models_dir,
        )
        return

    if iforest_path is not None:
        iforest = IsolationForestModel.load(str(iforest_path))
        y_pred_anomaly = iforest.predict(features)
        y_pred_binary = (np.array(y_pred_anomaly) == -1).astype(int)

        report = generate_experiment_report(
            experiment_name="anomaly_detection_evaluation",
            y_true=y_true_binary,
            y_pred=y_pred_binary,
            metadata={"data_path": str(data_path), "model": "IsolationForest"},
        )
        filepath = save_evaluation_report(report, output_dir=str(output_dir))
        logger.info("Anomaly detection report saved: %s", filepath)
        print(format_experiment_report(report))
    else:
        logger.warning("Isolation Forest model not found in %s", models_dir)

    if rf_path is not None:
        rf = RandomForestClassifierModel.load(str(rf_path))
        y_pred_labels = rf.predict(features)
        unique_labels = sorted(set(labels) | set(y_pred_labels))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        y_true_int = np.array([label_to_int[l] for l in labels])
        y_pred_int = np.array([label_to_int[l] for l in y_pred_labels])

        report = generate_experiment_report(
            experiment_name="classification_evaluation",
            y_true=y_true_int,
            y_pred=y_pred_int,
            metadata={"data_path": str(data_path), "model": "RandomForest",
                       "class_names": unique_labels},
        )
        filepath = save_evaluation_report(report, output_dir=str(output_dir))
        logger.info("Classification report saved: %s", filepath)
        print(format_experiment_report(report))
    else:
        logger.warning("Random Forest model not found in %s", models_dir)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--data", type=Path, default=Path("data/processed/processed_data.csv"))
    parser.add_argument("--models", type=Path, default=Path("models/trained"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")

    generate(args.data, args.models, args.output)


if __name__ == "__main__":
    main()
