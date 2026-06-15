#!/usr/bin/env python3
"""Run the test suite.

Usage:
    python scripts/run_tests.py [--verbose] [--unit-only] [--integration-only]
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run the test suite")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--module", type=str, help="Run tests for a specific module (e.g. 'anomaly_detection')")
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "pytest", "tests/", "-o", "addopts="]

    if args.verbose:
        cmd.append("-v")
    if args.unit_only:
        cmd.extend(["-m", "not slow"])
    if args.integration_only:
        cmd.extend(["-m", "slow"])
    if args.module:
        cmd.extend(["-k", args.module])

    cmd.append("--tb=short")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
