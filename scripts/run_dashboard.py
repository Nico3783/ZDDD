#!/usr/bin/env python3
"""Launch the Streamlit dashboard.

Usage:
    python scripts/run_dashboard.py [--port 8501] [--host localhost]
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Launch the detection dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Server port")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")

    app_path = Path(__file__).resolve().parent.parent / "src" / "dashboard" / "app.py"
    if not app_path.exists():
        logging.error("Dashboard app not found at %s", app_path)
        sys.exit(1)

    logging.info("Launching Streamlit dashboard on %s:%d", args.host, args.port)
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", str(args.port),
            "--server.address", args.host,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
