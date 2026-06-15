#!/usr/bin/env python3
"""Train the Isolation Forest anomaly detection model.

Usage:
    python scripts/train_iforest.py [--data data/processed/processed_data.csv] [--output models/trained]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def train(data_path: Path, output_path: Path) -> None:
    """Train the Isolation Forest model.

    Args:
        data_path: Path to processed training data CSV.
        output_path: Directory to save the trained model.
    """
    import pandas as pd

    from src.anomaly_detection.trainer import train_anomaly_detector

    output_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    label_col = "Label" if "Label" in df.columns else "label"
    if label_col not in df.columns:
        logger.error("Label column '%s' not found in data", label_col)
        return

    feature_cols = [
        c for c in df.columns
        if c not in [label_col, "Flow ID", "Source IP", "Destination IP",
                      "Source Port", "Destination Port", "Timestamp"]
    ]
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    logger.info(
        "Training Isolation Forest on %d samples, %d features",
        len(features), len(features.columns),
    )

    result = train_anomaly_detector(
        features=features,
        contamination=0.1,
    )

    model_path = output_path / "isolation_forest.pkl"
    result.save(str(model_path))
    logger.info("Model saved to %s", model_path)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Train Isolation Forest model")
    parser.add_argument("--data", type=Path, default=Path("data/processed/processed_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/trained"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")

    train(args.data, args.output)


if __name__ == "__main__":
    main()
