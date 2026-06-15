from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.core.config import get_config

logger = logging.getLogger(__name__)


class ResponseAction:
    """Represents a single response action to a detection event."""

    def __init__(
        self,
        action_type: str,
        target: str,
        parameters: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> None:
        """Initialize a response action.

        Args:
            action_type: Type of action (e.g., "log", "alert", "block", "quarantine").
            target: Target identifier (e.g., IP address, flow ID).
            parameters: Additional parameters for the action.
            description: Human-readable description.
        """
        self.action_type = action_type
        self.target = target
        self.parameters = parameters or {}
        self.description = description
        self.timestamp = datetime.now().isoformat()
        self.executed = False
        self.success = False
        self.error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type,
            "target": self.target,
            "parameters": self.parameters,
            "description": self.description,
            "timestamp": self.timestamp,
            "executed": self.executed,
            "success": self.success,
            "error_message": self.error_message,
        }


class ResponseHandler:
    """Handles responses to detection events.

    Provides configurable response actions for different alert severities
    and types, including logging, notification, blocking, and quarantine.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the response handler.

        Args:
            config: Response handler configuration dict. Loaded from config if None.
        """
        if config is None:
            try:
                self.config = get_config("response_handler")
            except Exception:
                self.config = {
                    "enabled": True,
                    "log_alerts": True,
                    "block_critical": False,
                    "quarantine_zero_day": False,
                    "cooldown_seconds": 10,
                    "max_actions_per_minute": 60,
                }
        else:
            self.config = config

        self._action_handlers: Dict[str, Callable[..., Any]] = {
            "log": self._handle_log,
            "alert": self._handle_alert,
            "block": self._handle_block,
            "quarantine": self._handle_quarantine,
            "notify": self._handle_notify,
        }
        self._action_history: List[ResponseAction] = []
        self._action_timestamps: List[float] = []
        self._cooldown_seconds: float = float(self.config.get("cooldown_seconds", 10))
        self._max_actions_per_minute: int = int(self.config.get("max_actions_per_minute", 60))

        logger.info("ResponseHandler initialized")

    def handle_detection(self, detection_result: Dict[str, Any]) -> List[ResponseAction]:
        """Handle a detection result by executing appropriate response actions.

        Args:
            detection_result: Detection result dictionary.

        Returns:
            List of ResponseAction objects that were executed.
        """
        actions: List[ResponseAction] = []

        if not self.config.get("enabled", True):
            return actions

        severity = detection_result.get("severity", "low")
        is_zero_day = detection_result.get("is_zero_day", False)
        is_anomaly = detection_result.get("is_anomaly", False)

        if not is_anomaly and not is_zero_day:
            return actions

        target = detection_result.get("source_ip", detection_result.get("sample_id", "unknown"))

        if self.config.get("log_alerts", True):
            actions.append(
                ResponseAction(
                    action_type="log",
                    target=target,
                    description=f"Log {severity} severity detection",
                )
            )

        if severity in ("critical", "high"):
            actions.append(
                ResponseAction(
                    action_type="alert",
                    target=target,
                    parameters={"severity": severity},
                    description=f"Alert for {severity} severity detection",
                )
            )

        if is_zero_day and self.config.get("quarantine_zero_day", False):
            actions.append(
                ResponseAction(
                    action_type="quarantine",
                    target=target,
                    parameters={"reason": "zero_day_detection"},
                    description="Quarantine zero-day detection",
                )
            )

        if severity == "critical" and self.config.get("block_critical", False):
            actions.append(
                ResponseAction(
                    action_type="block",
                    target=target,
                    parameters={"severity": severity},
                    description="Block critical severity source",
                )
            )

        for action in actions:
            self._execute_action(action)

        return actions

    def _execute_action(self, action: ResponseAction) -> None:
        """Execute a single response action."""
        if not self._rate_limit_check():
            logger.warning("Rate limit reached, skipping action: %s", action.action_type)
            return

        handler = self._action_handlers.get(action.action_type)
        if not handler:
            logger.warning("Unknown action type: %s", action.action_type)
            return

        try:
            handler(action)
            action.executed = True
            action.success = True
            self._action_history.append(action)
            self._action_timestamps.append(time.time())
            logger.info(
                "Action executed: type=%s target=%s",
                action.action_type,
                action.target,
            )
        except Exception as e:
            action.executed = True
            action.success = False
            action.error_message = str(e)
            self._action_history.append(action)
            logger.error("Action failed: type=%s error=%s", action.action_type, e)

    def _rate_limit_check(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        cutoff = now - 60.0
        self._action_timestamps = [t for t in self._action_timestamps if t > cutoff]
        return len(self._action_timestamps) < self._max_actions_per_minute

    def _handle_log(self, action: ResponseAction) -> None:
        """Handle logging action."""
        logger.info(
            "Detection log: target=%s description=%s",
            action.target,
            action.description,
        )

    def _handle_alert(self, action: ResponseAction) -> None:
        """Handle alert notification action."""
        logger.warning(
            "ALERT: target=%s severity=%s",
            action.target,
            action.parameters.get("severity", "unknown"),
        )

    def _handle_block(self, action: ResponseAction) -> None:
        """Handle blocking action."""
        logger.warning(
            "BLOCK: target=%s (simulated)",
            action.target,
        )

    def _handle_quarantine(self, action: ResponseAction) -> None:
        """Handle quarantine action."""
        logger.warning(
            "QUARANTINE: target=%s reason=%s (simulated)",
            action.target,
            action.parameters.get("reason", "unknown"),
        )

    def _handle_notify(self, action: ResponseAction) -> None:
        """Handle notification action."""
        logger.info(
            "NOTIFY: target=%s description=%s",
            action.target,
            action.description,
        )

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get action history.

        Args:
            limit: Maximum number of entries. None for all.

        Returns:
            List of action dictionaries.
        """
        history = [a.to_dict() for a in self._action_history]
        if limit:
            history = history[-limit:]
        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get response handler statistics.

        Returns:
            Dictionary with stats.
        """
        total = len(self._action_history)
        success = sum(1 for a in self._action_history if a.success)
        failed = total - success

        action_counts: Dict[str, int] = {}
        for a in self._action_history:
            action_counts[a.action_type] = action_counts.get(a.action_type, 0) + 1

        return {
            "total_actions": total,
            "successful_actions": success,
            "failed_actions": failed,
            "action_counts": action_counts,
            "rate_limit_remaining": self._max_actions_per_minute - len(self._action_timestamps),
        }
