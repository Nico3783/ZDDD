from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.alerting.logger import AlertLogger
from src.core.exceptions import DetectionError
from src.detection_engine.decision_logic import DetectionDecisionMaker
from src.detection_engine.engine import DetectionEngine
from src.detection_engine.severity import SeverityCalculator

logger = logging.getLogger(__name__)


class DetectionOrchestrator:
    """High-level orchestrator for the full detection pipeline.

    Coordinates preprocessing, feature extraction, anomaly detection,
    classification, alert generation, and response handling.
    """

    def __init__(
        self,
        engine: Optional[DetectionEngine] = None,
        alert_logger: Optional[AlertLogger] = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            engine: DetectionEngine instance. Created from config if None.
            alert_logger: AlertLogger instance. Created from config if None.
        """
        self.engine = engine or DetectionEngine()
        self.alert_logger = alert_logger or AlertLogger()
        self.decision_logic = DetectionDecisionMaker()
        self.severity_calc = SeverityCalculator()
        self._response_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self._processed_count = 0
        self._alert_count = 0

        logger.info("DetectionOrchestrator initialized")

    def process_dataframe(
        self,
        df: pd.DataFrame,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """Process a full DataFrame through the detection pipeline.

        Args:
            df: Input DataFrame with raw network traffic features.
            return_details: If True, include per-row predictions in result.

        Returns:
            Dictionary with summary statistics and optionally per-row details.
        """
        start_time = time.time()

        try:
            results = self.engine.detect_batch(df)
        except Exception as e:
            raise DetectionError(f"Orchestration failed: {e}") from e

        predictions = self.engine.results_to_dataframe(results)
        alerts = predictions[predictions["is_anomaly"] | predictions["is_zero_day"]].to_dict("records")
        self._processed_count += len(df)
        self._alert_count += len(alerts)

        for alert in alerts:
            self.alert_logger.log_alert(alert)
            self._notify_response_handlers(alert)

        elapsed = time.time() - start_time
        result: Dict[str, Any] = {
            "total_samples": len(df),
            "total_alerts": len(alerts),
            "anomaly_count": int(predictions["is_anomaly"].sum()),
            "zero_day_count": int(predictions["is_zero_day"].sum()),
            "processing_time_seconds": round(elapsed, 4),
            "throughput_samples_per_sec": round(len(df) / max(elapsed, 1e-9), 2),
        }

        if return_details:
            result["predictions"] = predictions.to_dict("records")

        return result

    def process_batch(
        self,
        records: List[Dict[str, Any]],
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """Process a batch of records (list of dicts).

        Args:
            records: List of record dictionaries.
            return_details: If True, include per-row predictions in result.

        Returns:
            Dictionary with summary statistics and optionally per-row details.
        """
        if not records:
            return {
                "total_samples": 0,
                "total_alerts": 0,
                "anomaly_count": 0,
                "zero_day_count": 0,
                "processing_time_seconds": 0.0,
                "throughput_samples_per_sec": 0.0,
            }

        df = pd.DataFrame(records)
        return self.process_dataframe(df, return_details=return_details)

    def process_single(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single record through the pipeline.

        Args:
            record: Single record dictionary.

        Returns:
            Detection result dictionary.
        """
        df = pd.DataFrame([record])
        result = self.process_dataframe(df, return_details=True)
        predictions = result.get("predictions", [])
        return predictions[0] if predictions else {}

    def register_response_handler(
        self,
        handler: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Register a callback for handling alerts.

        Args:
            handler: Function to call with each alert dictionary.
        """
        self._response_handlers.append(handler)
        logger.info("Registered response handler: %s", handler.__name__)

    def _notify_response_handlers(self, alert: Dict[str, Any]) -> None:
        """Notify all registered response handlers of a new alert."""
        for handler in self._response_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error("Response handler %s failed: %s", handler.__name__, e)

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics.

        Returns:
            Dictionary with processing stats.
        """
        return {
            "processed_count": self._processed_count,
            "alert_count": self._alert_count,
            "alert_rate": round(
                self._alert_count / max(self._processed_count, 1), 4
            ),
            "response_handler_count": len(self._response_handlers),
        }

    def reset_stats(self) -> None:
        """Reset processing counters."""
        self._processed_count = 0
        self._alert_count = 0
