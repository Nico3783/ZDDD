#!/usr/bin/env python3
"""Optimize anomaly detection thresholds.

Usage:
    python scripts/optimize_thresholds.py [--data data/processed] [--output config/thresholds.yaml]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def optimize(data_path: Path, output_path: Path) -> None:
    """Run threshold optimization.

    Args:
        data_path: Path to processed data CSV.
        output_path: Path to save optimized thresholds YAML.
    """
    import numpy as np
    import pandas as pd
    import yaml

    from src.anomaly_detection.threshold import optimize_threshold

    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    label_col = "Label" if "Label" in df.columns else "label"
    y = (df[label_col] != "BENIGN").astype(int).values

    feature_cols = [c for c in df.columns if c not in [label_col, "Flow ID", "Source IP", "Destination IP"]]
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info("Running threshold optimization on %d samples", len(y))

    # Placeholder: generate sample scores for optimization
    scores = np.random.uniform(0.3, 0.95, size=len(y))

    result = optimize_threshold(
        y_true=y,
        anomaly_scores=scores,
        optimization_target="f1",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump({"thresholds": result}, f, default_flow_style=False)

    logger.info("Optimized thresholds saved to %s", output_path)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Optimize detection thresholds")
    parser.add_argument("--data", type=Path, default=Path("data/processed/processed_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("config/thresholds.yaml"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")

    optimize(args.data, args.output)


if __name__ == "__main__":
    main()
