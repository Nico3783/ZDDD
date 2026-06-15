from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.constants import LOGS_DIR

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "timestamp",
    "alert_id",
    "severity",
    "predicted_class",
    "anomaly_score",
    "classification_confidence",
    "is_zero_day",
    "sample_id",
    "message",
]


class CSVAlertLogger:
    """CSV alert logger for tabular alert export.

    Writes alerts as CSV rows for easy import into spreadsheets,
    databases, and analysis tools.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        max_file_size_mb: int = 50,
    ) -> None:
        """Initialize the CSV alert logger.

        Args:
            log_dir: Directory for log files. If None, uses config default.
            max_file_size_mb: Maximum file size before rotation.
        """
        self.log_dir = log_dir or str(LOGS_DIR / "alerts")
        self.max_file_size_mb = max_file_size_mb

        os.makedirs(self.log_dir, exist_ok=True)
        self._current_file: Optional[str] = None
        self._file_handle = None
        self._writer = None
        self._write_count = 0

        self._init_file()
        logger.info("CSVAlertLogger initialized: dir=%s", self.log_dir)

    def _init_file(self) -> None:
        """Initialize the current CSV file with headers."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        self._current_file = os.path.join(self.log_dir, f"alerts_{date_str}.csv")
        file_exists = os.path.exists(self._current_file) and os.path.getsize(self._current_file) > 0

        self._file_handle = open(self._current_file, "a", newline="")
        self._writer = csv.writer(self._file_handle)

        if not file_exists:
            self._writer.writerow(CSV_COLUMNS)
            self._file_handle.flush()

    def log(self, alert: Dict[str, Any]) -> None:
        """Log a single alert as a CSV row.

        Args:
            alert: Alert data dictionary.
        """
        if not self._writer:
            return

        if "timestamp" not in alert:
            alert["timestamp"] = datetime.now().isoformat()

        row = [alert.get(col, "") for col in CSV_COLUMNS]

        try:
            self._writer.writerow(row)
            self._file_handle.flush()
            self._write_count += 1
            self._check_rotation()
        except Exception as e:
            logger.error("Failed to write CSV alert: %s", e)
            self._reinit_handle()

    def log_batch(self, alerts: List[Dict[str, Any]]) -> int:
        """Log multiple alerts efficiently.

        Args:
            alerts: List of alert dictionaries.

        Returns:
            Number of alerts successfully logged.
        """
        written = 0
        for alert in alerts:
            try:
                self.log(alert)
                written += 1
            except Exception as e:
                logger.error("Failed to log alert: %s", e)
        return written

    def _check_rotation(self) -> None:
        """Check if the current file needs rotation."""
        if not self._current_file:
            return

        try:
            size_mb = os.path.getsize(self._current_file) / (1024 * 1024)
            if size_mb >= self.max_file_size_mb:
                self._rotate()
        except OSError:
            pass

    def _rotate(self) -> None:
        """Rotate the current log file."""
        if self._file_handle:
            self._file_handle.close()

        if self._current_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_name = f"{self._current_file}.{timestamp}"
            os.rename(self._current_file, rotated_name)
            logger.info("Rotated CSV log file to %s", rotated_name)

        self._init_file()

    def _reinit_handle(self) -> None:
        """Reinitialize the file handle after an error."""
        try:
            if self._file_handle:
                self._file_handle.close()
        except Exception:
            pass
        self._init_file()

    def read_alerts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read alerts from the current CSV log file.

        Args:
            limit: Maximum number of alerts to read. None for all.

        Returns:
            List of alert dictionaries.
        """
        if not self._current_file or not os.path.exists(self._current_file):
            return []

        alerts: List[Dict[str, Any]] = []
        try:
            with open(self._current_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    alerts.append(dict(row))
                    if limit and len(alerts) >= limit:
                        break
        except Exception as e:
            logger.error("Failed to read CSV alerts: %s", e)

        return alerts

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "current_file": self._current_file,
            "write_count": self._write_count,
            "max_file_size_mb": self.max_file_size_mb,
        }

    def close(self) -> None:
        """Close the current file handle."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
