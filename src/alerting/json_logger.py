from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.constants import LOGS_DIR

logger = logging.getLogger(__name__)


class JSONAlertLogger:
    """Structured JSON alert logger with rotation and archival.

    Writes alerts as JSON Lines (one JSON object per line) for efficient
    append and streaming reads.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        max_file_size_mb: int = 50,
        max_files: int = 10,
    ) -> None:
        """Initialize the JSON alert logger.

        Args:
            log_dir: Directory for log files. If None, uses config default.
            max_file_size_mb: Maximum file size before rotation.
            max_files: Maximum number of rotated files to keep.
        """
        self.log_dir = log_dir or str(LOGS_DIR / "alerts")
        self.max_file_size_mb = max_file_size_mb
        self.max_files = max_files

        os.makedirs(self.log_dir, exist_ok=True)
        self._current_file: Optional[str] = None
        self._file_handle = None
        self._write_count = 0

        self._init_file()
        logger.info("JSONAlertLogger initialized: dir=%s", self.log_dir)

    def _init_file(self) -> None:
        """Initialize the current log file with date-based naming."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        self._current_file = os.path.join(self.log_dir, f"alerts_{date_str}.jsonl")
        self._file_handle = open(self._current_file, "a")
        self._write_count = 0

    def log(self, alert: Dict[str, Any]) -> None:
        """Log a single alert as a JSON line.

        Args:
            alert: Alert data dictionary.
        """
        if "timestamp" not in alert:
            alert["timestamp"] = datetime.now().isoformat()

        line = json.dumps(alert, default=str) + "\n"

        try:
            self._file_handle.write(line)
            self._file_handle.flush()
            self._write_count += 1
            self._check_rotation()
        except Exception as e:
            logger.error("Failed to write JSON alert: %s", e)
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
            logger.info("Rotated log file to %s", rotated_name)

        self._cleanup_old_files()
        self._init_file()

    def _cleanup_old_files(self) -> None:
        """Remove old rotated files beyond the retention limit."""
        import glob

        pattern = os.path.join(self.log_dir, "alerts_*.jsonl.*")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        for old_file in files[self.max_files - 1:]:
            try:
                os.remove(old_file)
                logger.debug("Removed old log file: %s", old_file)
            except OSError as e:
                logger.warning("Failed to remove old log file %s: %s", old_file, e)

    def _reinit_handle(self) -> None:
        """Reinitialize the file handle after an error."""
        try:
            if self._file_handle:
                self._file_handle.close()
        except Exception:
            pass
        self._init_file()

    def read_alerts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read alerts from the current log file.

        Args:
            limit: Maximum number of alerts to read. None for all.

        Returns:
            List of alert dictionaries.
        """
        alerts: List[Dict[str, Any]] = []

        if not self._current_file or not os.path.exists(self._current_file):
            return alerts

        try:
            with open(self._current_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        alerts.append(json.loads(line))
                        if limit and len(alerts) >= limit:
                            break
        except Exception as e:
            logger.error("Failed to read alerts: %s", e)

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
            "max_files": self.max_files,
        }

    def close(self) -> None:
        """Close the current file handle."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
