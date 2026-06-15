#!/usr/bin/env python3
"""Download and prepare the CICIDS2017 dataset for training.

Usage:
    python scripts/download_dataset.py [--output-dir data/raw]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def download_dataset(output_dir: Path) -> None:
    """Download the CICIDS2017 dataset.

    Args:
        output_dir: Directory to save downloaded files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Dataset download target: %s", output_dir)

    # Placeholder: implement actual download logic
    logger.warning(
        "Automatic download not implemented. Please download CICIDS2017 manually "
        "from https://www.unb.ca/cic/datasets/ids-2017.html and place CSV files in %s",
        output_dir,
    )


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Download CICIDS2017 dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for dataset files",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")

    download_dataset(args.output_dir)


if __name__ == "__main__":
    main()
