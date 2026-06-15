#!/usr/bin/env python3
"""End-to-end demonstration of the Zero-Day DoS Detection Engine.

Runs the full pipeline: load data → preprocess → train models → detect → evaluate → report.

Usage:
    python scripts/demo.py [--samples 5000] [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS_TO_DROP = [
    "Flow ID", "Source IP", "Destination IP",
    "Source Port", "Destination Port", "Protocol", "Timestamp",
]


def load_and_preprocess(data_path: Path, n_samples: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load CICIDS2017 CSV and prepare features + binary labels.

    Returns:
        (X_train_df, y_binary_series, X_test_df) — all as DataFrames/Series
        so the model wrappers receive the types they expect.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    logger.info("Loading dataset from %s ...", data_path)
    df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.strip()

    label_col = "Label"
    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_TO_DROP + [label_col]]

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42)

    X = df[feature_cols].copy()
    y_binary = (df[label_col] != "BENIGN").astype(int)

    logger.info("Loaded %d samples, %d features", X.shape[0], X.shape[1])
    logger.info("Attack ratio: %.2f%%", y_binary.mean() * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    scaler = StandardScaler()
    feature_names = list(X.columns)
    X_train_arr = scaler.fit_transform(X_train.values)
    X_test_arr = scaler.transform(X_test.values)

    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_arr, columns=feature_names)

    logger.info("Train: %d | Test: %d", len(X_train_df), len(X_test_df))
    return X_train_df, y_train.reset_index(drop=True), X_test_df, y_test.reset_index(drop=True), scaler


def run_pipeline(n_samples: int) -> dict:
    """Execute the full detection pipeline and return results."""
    from src.anomaly_detection.isolation_forest import IsolationForestModel
    from src.classification.random_forest import RandomForestClassifierModel
    from src.evaluation.latency import LatencyTracker
    from src.evaluation.metrics import PerformanceEvaluator
    from src.evaluation.reports import (
        format_experiment_report,
        generate_experiment_report,
        save_evaluation_report,
    )
    from src.evaluation.throughput import ThroughputTracker

    data_path = Path("data/raw/cicids2017/Wednesday-workingHours.pcap_ISCX.csv")
    if not data_path.exists():
        logger.error("Dataset not found: %s", data_path)
        return {}

    X_train, y_train, X_test, y_test, scaler = load_and_preprocess(data_path, n_samples)

    latency = LatencyTracker()
    throughput = ThroughputTracker()
    results: dict = {}

    # Isolation Forest 
    logger.info("Training Isolation Forest ...")
    t0 = time.perf_counter()
    iforest = IsolationForestModel()
    iforest.train(X_train, contamination=0.1, n_estimators=100, random_state=42)
    if_ms = (time.perf_counter() - t0) * 1000
    latency.record("iforest_train", if_ms)
    logger.info("  trained in %.3fs", if_ms / 1000)

    t0 = time.perf_counter()
    raw_preds = iforest.predict(X_test)
    anomaly_preds = np.where(raw_preds == -1, 1, 0)
    if_inf_ms = (time.perf_counter() - t0) * 1000
    latency.record("iforest_inference", if_inf_ms)
    throughput.record_rate("iforest_inference", samples=len(X_test), elapsed_seconds=if_inf_ms / 1000)
    logger.info("  anomaly rate: %.2f%%", anomaly_preds.mean() * 100)

    # Random Forest 
    logger.info("Training Random Forest ...")
    t0 = time.perf_counter()
    rforest = RandomForestClassifierModel()
    rforest.train(X_train, y_train, n_estimators=100, random_state=42)
    rf_ms = (time.perf_counter() - t0) * 1000
    latency.record("rf_train", rf_ms)
    logger.info("  trained in %.3fs", rf_ms / 1000)

    t0 = time.perf_counter()
    rf_raw = rforest.predict(X_test)
    rf_preds = np.where(rf_raw == "BENIGN", 0, 1)
    rf_inf_ms = (time.perf_counter() - t0) * 1000
    latency.record("rf_inference", rf_inf_ms)
    throughput.record_rate("rf_inference", samples=len(X_test), elapsed_seconds=rf_inf_ms / 1000)

    # Combined pipeline: anomaly flagged → classify; else benign 
    combined_preds = np.where(anomaly_preds == 1, rf_preds, 0)

    # Evaluate 
    evaluator = PerformanceEvaluator()
    metrics = evaluator.evaluate(y_test.values, combined_preds)
    results["metrics"] = metrics

    logger.info(" Results ")
    logger.info("  Accuracy:  %.4f", metrics["accuracy"])
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall:    %.4f", metrics["recall"])
    logger.info("  F1-Score:  %.4f", metrics["f1_score"])

    # Experiment Report 
    report = generate_experiment_report(
        experiment_name="demo_run",
        y_true=y_test.values,
        y_pred=combined_preds,
        latency_tracker=latency,
        throughput_tracker=throughput,
        metadata={
            "dataset": "CICIDS2017 (Wednesday-workingHours)",
            "n_samples": int(n_samples),
            "n_features": X_train.shape[1],
            "anomaly_model": "IsolationForest",
            "classifier": "RandomForest",
            "contamination": 0.1,
        },
    )
    results["report"] = report

    report_text = format_experiment_report(report)
    print("\n" + report_text)

    report_path = save_evaluation_report(report, output_dir="reports")
    logger.info("Report saved: %s", report_path)

    return results


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Zero-Day DoS Detection Engine — Demo")
    parser.add_argument("--samples", type=int, default=5000, help="Max samples to use")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(args.samples)
    logger.info("Demo complete.")


if __name__ == "__main__":
    main()
