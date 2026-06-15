#!/usr/bin/env python3
"""Preprocess raw dataset into clean, encoded features for training.

Usage:
    python scripts/preprocess_data.py [--input data/raw] [--output data/processed]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def preprocess(input_dir: Path, output_dir: Path) -> None:
    """Run the preprocessing pipeline on raw data.

    Args:
        input_dir: Directory containing raw CSV files.
        output_dir: Directory to write processed data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list(input_dir.glob("*.csv"))

    if not csv_files:
        logger.error("No CSV files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d CSV file(s) in %s", len(csv_files), input_dir)

    dfs = []
    for f in csv_files:
        logger.info("Loading %s", f.name)
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)
    logger.info("Combined dataset: %d rows, %d columns", len(raw), len(raw.columns))

    # Basic cleanup
    raw.columns = raw.columns.str.strip()
    raw = raw.dropna(how="all", axis=0)
    raw = raw.dropna(how="all", axis=1)
    raw = raw.replace([float("inf"), float("-inf")], float("nan"))
    raw = raw.dropna()

    logger.info("After cleanup: %d rows, %d columns", len(raw), len(raw.columns))

    output_path = output_dir / "processed_data.csv"
    raw.to_csv(output_path, index=False)
    logger.info("Saved processed data to %s", output_path)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Preprocess CICIDS2017 dataset")
    parser.add_argument("--input", type=Path, default=Path("data/raw/cicids2017"), help="Input directory")
    parser.add_argument("--output", type=Path, default=Path("data/processed"), help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")

    preprocess(args.input, args.output)


if __name__ == "__main__":
    main()
