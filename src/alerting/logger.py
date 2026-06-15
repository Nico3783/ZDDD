from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.config import get_config
from src.core.constants import LOGS_DIR
from src.core.exceptions import AlertError

logger = logging.getLogger(__name__)


class AlertLogger:
    """Logs security alerts to JSON and CSV files.

    Provides structured alert logging with automatic file rotation
    and archival support.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_format: str = "json",
    ) -> None:
        """Initialize the alert logger.

        Args:
            log_dir: Directory for alert logs. If None, uses config default.
            log_format: Output format ('json' or 'csv').
        """
        if log_dir is None:
            cfg = get_config().load("paths")
            log_dir = cfg.get("alerts_dir", str(LOGS_DIR / "alerts"))

        self.log_dir = log_dir
        self.log_format = log_format
        self._ensure_log_dir()

        self._json_file: Optional[str] = None
        self._csv_file: Optional[str] = None
        self._csv_writer = None
        self._csv_file_handle = None

        self._init_files()
        logger.info("AlertLogger initialized: dir=%s, format=%s", log_dir, log_format)

    def _ensure_log_dir(self) -> None:
        """Create the log directory if it doesn't exist."""
        os.makedirs(self.log_dir, exist_ok=True)

    def _init_files(self) -> None:
        """Initialize log files with date-based naming."""
        date_str = datetime.now().strftime("%Y-%m-%d")

        if self.log_format == "json":
            self._json_file = os.path.join(self.log_dir, f"alerts_{date_str}.json")
        elif self.log_format == "csv":
            self._csv_file = os.path.join(self.log_dir, f"alerts_{date_str}.csv")
            self._init_csv()
        else:
            # Both formats
            self._json_file = os.path.join(self.log_dir, f"alerts_{date_str}.json")
            self._csv_file = os.path.join(self.log_dir, f"alerts_{date_str}.csv")
            self._init_csv()

    def _init_csv(self) -> None:
        """Initialize CSV file with headers."""
        if self._csv_file:
            file_exists = os.path.exists(self._csv_file) and os.path.getsize(self._csv_file) > 0
            self._csv_file_handle = open(self._csv_file, "a", newline="")
            self._csv_writer = csv.writer(self._csv_file_handle)

            if not file_exists:
                self._csv_writer.writerow([
                    "timestamp",
                    "alert_id",
                    "severity",
                    "predicted_class",
                    "anomaly_score",
                    "classification_confidence",
                    "is_zero_day",
                    "sample_id",
                    "message",
                ])

    def log_alert(self, alert: Any) -> None:
        """Log a single alert.

        Args:
            alert: Alert object or dictionary with alert data.
        """
        if hasattr(alert, "__dict__"):
            alert_dict = {
                "timestamp": datetime.now().isoformat(),
                "alert_id": alert.alert_id,
                "severity": alert.severity,
                "predicted_class": alert.predicted_class,
                "anomaly_score": alert.anomaly_score,
                "classification_confidence": alert.classification_confidence,
                "is_zero_day": alert.is_zero_day,
                "sample_id": alert.sample_id,
                "message": alert.message,
                "details": alert.details if hasattr(alert, "details") else {},
            }
        else:
            alert_dict = alert
            if "timestamp" not in alert_dict:
                alert_dict["timestamp"] = datetime.now().isoformat()

        # Write JSON
        if self._json_file:
            self._write_json(alert_dict)

        # Write CSV
        if self._csv_writer:
            self._write_csv(alert_dict)

        logger.info(
            "Alert logged: id=%s, severity=%s, class=%s",
            alert_dict.get("alert_id", "N/A"),
            alert_dict.get("severity", "N/A"),
            alert_dict.get("predicted_class", "N/A"),
        )

    def log_alerts(self, alerts: List[Any]) -> None:
        """Log multiple alerts.

        Args:
            alerts: List of Alert objects or dictionaries.
        """
        for alert in alerts:
            self.log_alert(alert)

        logger.info("Logged %d alerts", len(alerts))

    def _write_json(self, alert_dict: Dict[str, Any]) -> None:
        """Append an alert to the JSON log file.

        Args:
            alert_dict: Alert data dictionary.
        """
        if not self._json_file:
            return

        try:
            # Append as JSON lines (one JSON object per line)
            with open(self._json_file, "a") as f:
                f.write(json.dumps(alert_dict) + "\n")
        except Exception as e:
            logger.error("Failed to write JSON alert: %s", e)

    def _write_csv(self, alert_dict: Dict[str, Any]) -> None:
        """Append an alert to the CSV log file.

        Args:
            alert_dict: Alert data dictionary.
        """
        if not self._csv_writer:
            return

        try:
            self._csv_writer.writerow([
                alert_dict.get("timestamp", ""),
                alert_dict.get("alert_id", ""),
                alert_dict.get("severity", ""),
                alert_dict.get("predicted_class", ""),
                alert_dict.get("anomaly_score", 0.0),
                alert_dict.get("classification_confidence", 0.0),
                alert_dict.get("is_zero_day", False),
                alert_dict.get("sample_id", ""),
                alert_dict.get("message", ""),
            ])
            self._csv_file_handle.flush()
        except Exception as e:
            logger.error("Failed to write CSV alert: %s", e)

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get a summary of logged alerts.

        Returns:
            Dictionary with alert summary statistics.
        """
        alerts = self.load_alerts()

        if not alerts:
            return {"total": 0, "by_severity": {}, "by_class": {}}

        by_severity: Dict[str, int] = {}
        by_class: Dict[str, int] = {}
        zero_day_count = 0

        for alert in alerts:
            severity = alert.get("severity", "unknown")
            by_severity[severity] = by_severity.get(severity, 0) + 1

            cls = alert.get("predicted_class", "unknown")
            by_class[cls] = by_class.get(cls, 0) + 1

            if alert.get("is_zero_day", False):
                zero_day_count += 1

        return {
            "total": len(alerts),
            "by_severity": by_severity,
            "by_class": by_class,
            "zero_day_count": zero_day_count,
        }

    def load_alerts(self) -> List[Dict[str, Any]]:
        """Load all alerts from the current log file.

        Returns:
            List of alert dictionaries.
        """
        alerts = []

        if self._json_file and os.path.exists(self._json_file):
            try:
                with open(self._json_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            alerts.append(json.loads(line))
            except Exception as e:
                logger.error("Failed to load JSON alerts: %s", e)

        return alerts

    def archive_alerts(self, archive_dir: Optional[str] = None) -> str:
        """Archive current alert logs and start fresh.

        Args:
            archive_dir: Directory for archived logs. If None, uses 'archived' subdir.

        Returns:
            Path to the archive directory.
        """
        if archive_dir is None:
            archive_dir = os.path.join(self.log_dir, "archived")

        os.makedirs(archive_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        archived_count = 0

        # Archive JSON
        if self._json_file and os.path.exists(self._json_file):
            archive_path = os.path.join(archive_dir, f"alerts_{date_str}.json")
            os.rename(self._json_file, archive_path)
            archived_count += 1

        # Archive CSV
        if self._csv_file and os.path.exists(self._csv_file):
            if self._csv_file_handle:
                self._csv_file_handle.close()
            archive_path = os.path.join(archive_dir, f"alerts_{date_str}.csv")
            os.rename(self._csv_file, archive_path)
            archived_count += 1

        # Reinitialize files
        self._init_files()

        logger.info("Archived %d alert files to %s", archived_count, archive_dir)
        return archive_dir

    def close(self) -> None:
        """Close open file handles."""
        if self._csv_file_handle:
            self._csv_file_handle.close()
            self._csv_file_handle = None
