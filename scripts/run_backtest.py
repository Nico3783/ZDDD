#!/usr/bin/env python3
"""Run backtesting evaluation against historical data.

Usage:
    python scripts/run_backtest.py [--data data/processed/processed_data.csv] [--models models/trained] [--window-size 100]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_model(models_dir: Path, name: str) -> Path | None:
    """Find a model file by name, checking .joblib and .pkl extensions."""
    for ext in (".joblib", ".pkl"):
        path = models_dir / f"{name}{ext}"
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def backtest(data_path: Path, models_dir: Path, window_size: int) -> dict:
    """Run backtesting evaluation.

    Loads trained models and evaluates them using a sliding window approach
    on historical data to simulate real-time detection performance.

    Args:
        data_path: Path to processed data CSV.
        models_dir: Directory containing trained models.
        window_size: Number of samples per sliding window.

    Returns:
        Backtest results dict.
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

    y_true_binary = (labels != "BENIGN").astype(int).values

    # Load models
    iforest_path = _find_model(models_dir, "isolation_forest")
    rf_path = _find_model(models_dir, "random_forest")

    anomaly_model = None
    classifier_model = None

    if iforest_path is not None:
        anomaly_model = IsolationForestModel.load(str(iforest_path))
        logger.info("Loaded Isolation Forest model from %s", iforest_path)
    if rf_path is not None:
        classifier_model = RandomForestClassifierModel.load(str(rf_path))
        logger.info("Loaded Random Forest model from %s", rf_path)

    if anomaly_model is None and classifier_model is None:
        logger.error(
            "No trained models found in %s. Run training first:\n"
            "  python scripts/train_iforest.py\n"
            "  python scripts/train_random_forest.py",
            models_dir,
        )
        return {}

    # Sliding window backtest
    n_samples = len(features)
    all_preds = []
    all_true = []
    window_latencies = []

    logger.info("Running backtest: %d samples, window_size=%d", n_samples, window_size)

    start_time = time.time()

    for i in range(0, n_samples, window_size):
        window_end = min(i + window_size, n_samples)
        window_features = features.iloc[i:window_end]
        window_true = y_true_binary[i:window_end]

        t0 = time.time()
        if anomaly_model is not None:
            preds = anomaly_model.predict(window_features)
            preds_binary = (np.array(preds) == -1).astype(int)
        else:
            preds_binary = np.zeros(len(window_features), dtype=int)
        latency = time.time() - t0
        window_latencies.append(latency)

        all_preds.extend(preds_binary.tolist())
        all_true.extend(window_true.tolist())

        if (i // window_size) % 10 == 0:
            logger.info(
                "Window %d/%d | samples=%d | latency=%.3fs",
                i // window_size + 1, (n_samples + window_size - 1) // window_size,
                window_end - i, latency,
            )

    total_time = time.time() - start_time

    # Evaluate
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    evaluator = PerformanceEvaluator()
    results = evaluator.evaluate(all_true, all_preds)

    results["backtest"] = {
        "total_samples": n_samples,
        "window_size": window_size,
        "total_time_seconds": total_time,
        "avg_window_latency_ms": np.mean(window_latencies) * 1000,
        "total_windows": len(window_latencies),
    }

    logger.info("Backtest complete:")
    logger.info("  Accuracy:  %.4f", results["accuracy"])
    logger.info("  Precision: %.4f", results["precision"])
    logger.info("  Recall:    %.4f", results["recall"])
    logger.info("  F1-Score:  %.4f", results["f1_score"])
    logger.info("  Total time: %.2f seconds", total_time)

    # Save report
    report = generate_experiment_report(
        experiment_name="backtest_evaluation",
        y_true=all_true,
        y_pred=all_preds,
        metadata={
            "data_path": str(data_path),
            "window_size": window_size,
            "total_time_seconds": total_time,
        },
    )
    output_dir = str(REPORTS_DIR / "backtest")
    filepath = save_evaluation_report(report, output_dir=output_dir)
    logger.info("Backtest report saved: %s", filepath)

    return results


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run backtesting evaluation")
    parser.add_argument("--data", type=Path, default=Path("data/processed/processed_data.csv"))
    parser.add_argument("--models", type=Path, default=Path("models/trained"))
    parser.add_argument("--window-size", type=int, default=100, help="Sliding window size")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")

    backtest(args.data, args.models, args.window_size)


if __name__ == "__main__":
    main()
