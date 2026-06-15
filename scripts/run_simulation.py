#!/usr/bin/env python3
"""Run the real-time streaming detection simulation.

Usage:
    python scripts/run_simulation.py [--data data/processed/processed_data.csv] [--duration 60] [--batch-size 100]
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


def simulate(data_path: Path, duration: int, batch_size: int) -> None:
    """Run the streaming simulation for a given duration.

    Args:
        data_path: Path to the processed data CSV to stream.
        duration: Simulation duration in seconds.
        batch_size: Number of packets per batch.
    """
    import pandas as pd

    from src.streaming.simulator import StreamSimulator
    from src.detection_engine.engine import DetectionEngine
    from src.detection_engine.orchestrator import DetectionOrchestrator
    from src.anomaly_detection.isolation_forest import IsolationForestModel
    from src.classification.random_forest import RandomForestClassifierModel
    from src.alerting.generator import AlertGenerator
    from src.alerting.logger import AlertLogger
    from src.core.constants import MODELS_DIR, LOGS_DIR

    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return

    logger.info("Starting simulation for %d seconds", duration)
    logger.info("Data source: %s", data_path)

    df = pd.read_csv(data_path)
    label_col = "Label" if "Label" in df.columns else "label"
    feature_cols = [
        c for c in df.columns
        if c not in [label_col, "Flow ID", "Source IP", "Destination IP",
                      "Source Port", "Destination Port", "Timestamp"]
    ]
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Load models
    iforest_path = _find_model(MODELS_DIR / "trained", "isolation_forest")
    rf_path = _find_model(MODELS_DIR / "trained", "random_forest")

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
            "No trained models found. Run training first:\n"
            "  python scripts/train_iforest.py\n"
            "  python scripts/train_random_forest.py"
        )
        return

    # Set up detection engine
    engine = DetectionEngine(anomaly_model=anomaly_model, classifier_model=classifier_model)
    alert_logger = AlertLogger(log_dir=str(LOGS_DIR / "alerts"))
    orchestrator = DetectionOrchestrator(engine=engine, alert_logger=alert_logger)

    # Set up streaming simulator
    simulator = StreamSimulator(random_state=42)

    logger.info("Streaming %d rows in batches of %d", len(features), batch_size)

    start = time.time()
    total_packets = 0
    total_alerts = 0

    try:
        for batch_df in simulator.stream_from_dataframe(
            df=features, batch_size=batch_size, delay_seconds=0.01, jitter=0.005,
        ):
            elapsed = time.time() - start
            if elapsed >= duration:
                break

            result = orchestrator.process_dataframe(batch_df, return_details=True)
            batch_count = len(batch_df)
            total_packets += batch_count

            if "total_alerts" in result:
                alert_count = result["total_alerts"]
                total_alerts += alert_count
                if alert_count > 0:
                    logger.info(
                        "t=%.1fs | batch=%d | alerts=%d | total_packets=%d",
                        elapsed, batch_count, alert_count, total_packets,
                    )

            if total_packets % (batch_size * 10) == 0:
                logger.info(
                    "t=%.1fs | processed=%d packets | alerts=%d",
                    elapsed, total_packets, total_alerts,
                )

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        elapsed = time.time() - start
        stats = orchestrator.get_stats()
        logger.info("Simulation complete:")
        logger.info("  Duration:       %.1f seconds", elapsed)
        logger.info("  Total packets:  %d", total_packets)
        logger.info("  Total alerts:   %d", total_alerts)
        logger.info("  Detection stats: %s", stats)
        alert_logger.close()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run streaming simulation")
    parser.add_argument("--data", type=Path, default=Path("data/processed/processed_data.csv"))
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--batch-size", type=int, default=100, help="Packets per batch")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")

    simulate(args.data, args.duration, args.batch_size)


if __name__ == "__main__":
    main()
