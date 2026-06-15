from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.config import get_config
from src.core.constants import LOGS_DIR
from src.core.exceptions import AlertError

logger = logging.getLogger(__name__)


class AlertNotifier:
    """Dispatches alerts to configured output channels.

    Supports console, file (JSON), and optional external webhook outputs.
    Manages output file rotation and provides alert dispatch history.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        enable_console: bool = True,
        enable_file: bool = True,
    ) -> None:
        """Initialize the AlertNotifier.

        Args:
            output_dir: Directory for alert output files. If None, uses config.
            enable_console: Whether to print alerts to console.
            enable_file: Whether to write alerts to files.
        """
        if output_dir is None:
            cfg = get_config().load("alerts")
            output_dir = cfg.get("output_dir", str(LOGS_DIR / "alerts"))

        self.output_dir = output_dir
        self.enable_console = enable_console
        self.enable_file = enable_file

        self._dispatch_count = 0
        self._dispatch_history: List[Dict[str, Any]] = []

        if self.enable_file:
            os.makedirs(self.output_dir, exist_ok=True)

        logger.info(
            "AlertNotifier initialized: console=%s, file=%s, dir=%s",
            enable_console, enable_file, output_dir,
        )

    def notify(self, alerts: List[Dict[str, Any]]) -> int:
        """Dispatch alerts to all configured channels.

        Args:
            alerts: List of alert dictionaries.

        Returns:
            Number of alerts successfully dispatched.
        """
        if not alerts:
            return 0

        dispatched = 0

        for alert in alerts:
            success = self._dispatch_single(alert)
            if success:
                dispatched += 1
                self._dispatch_count += 1

                self._dispatch_history.append({
                    "alert_id": alert.get("alert_id", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                    "severity": alert.get("severity", "unknown"),
                    "channel": "console" if self.enable_console else "file",
                })

                # Keep history bounded
                if len(self._dispatch_history) > 1000:
                    self._dispatch_history = self._dispatch_history[-500:]

        logger.info(
            "Dispatched %d/%d alerts (total=%d)",
            dispatched, len(alerts), self._dispatch_count,
        )

        return dispatched

    def _dispatch_single(self, alert: Dict[str, Any]) -> bool:
        """Dispatch a single alert to all channels.

        Args:
            alert: Alert dictionary.

        Returns:
            True if at least one channel succeeded.
        """
        success = False

        if self.enable_console:
            try:
                self._dispatch_console(alert)
                success = True
            except Exception as e:
                logger.error("Console dispatch failed: %s", e)

        if self.enable_file:
            try:
                self._dispatch_file(alert)
                success = True
            except Exception as e:
                logger.error("File dispatch failed: %s", e)

        return success

    def _dispatch_console(self, alert: Dict[str, Any]) -> None:
        """Print alert to console.

        Args:
            alert: Alert dictionary.
        """
        severity = alert.get("severity", "unknown").upper()
        alert_id = alert.get("alert_id", "N/A")
        message = alert.get("message", "No message")
        is_zero_day = alert.get("is_zero_day", False)

        prefix = "!!! " if is_zero_day else "    "
        icon = "[ZERO-DAY]" if is_zero_day else "[ATTACK]"

        print(
            f"{prefix}{icon} [{severity}] {alert_id}\n"
            f"         {message}\n"
        )

    def _dispatch_file(self, alert: Dict[str, Any]) -> None:
        """Write alert to JSON file.

        Args:
            alert: Alert dictionary.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(self.output_dir, f"alerts_{date_str}.jsonl")

        alert_copy = dict(alert)
        alert_copy["dispatched_at"] = datetime.now().isoformat()

        with open(filepath, "a") as f:
            f.write(json.dumps(alert_copy) + "\n")

    def get_dispatch_stats(self) -> Dict[str, Any]:
        """Get dispatch statistics.

        Returns:
            Dictionary with dispatch stats.
        """
        return {
            "total_dispatched": self._dispatch_count,
            "console_enabled": self.enable_console,
            "file_enabled": self.enable_file,
            "output_dir": self.output_dir,
            "recent_history": self._dispatch_history[-10:],
        }

    def clear_history(self) -> None:
        """Clear the dispatch history."""
        self._dispatch_history.clear()
        logger.info("Dispatch history cleared")


def dispatch_alerts(
    alerts: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
) -> int:
    """Convenience function to dispatch alerts to configured channels.

    Args:
        alerts: List of alert dictionaries.
        output_dir: Output directory override.
        enable_console: Whether to print to console.
        enable_file: Whether to write to files.

    Returns:
        Number of alerts dispatched.
    """
    notifier = AlertNotifier(
        output_dir=output_dir,
        enable_console=enable_console,
        enable_file=enable_file,
    )
    return notifier.notify(alerts)
