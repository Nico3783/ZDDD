from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def metric_card(title: str, value: Any, delta: Any = None, delta_suffix: str = "%") -> Dict[str, Any]:
    """Create a metric card for display.

    Args:
        title: Card title.
        value: Primary value to display.
        delta: Optional delta value for comparison.
        delta_suffix: Suffix for delta display.

    Returns:
        Dictionary with card data.
    """
    card: Dict[str, Any] = {"title": title, "value": value}
    if delta is not None:
        card["delta"] = delta
        card["delta_suffix"] = delta_suffix
        card["delta_positive"] = float(delta) >= 0 if isinstance(delta, (int, float)) else False
    return card


def status_card(title: str, status: str, details: str = "") -> Dict[str, Any]:
    """Create a status indicator card.

    Args:
        title: Card title.
        status: Status string (e.g., 'healthy', 'warning', 'critical').
        details: Additional details.

    Returns:
        Dictionary with status card data.
    """
    return {
        "title": title,
        "status": status.lower(),
        "details": details,
        "icon": _status_icon(status),
    }


def alert_summary_card(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a summary card from a list of alerts.

    Args:
        alerts: List of alert dictionaries.

    Returns:
        Dictionary with alert summary.
    """
    total = len(alerts)
    critical = sum(1 for a in alerts if a.get("severity") == "critical")
    high = sum(1 for a in alerts if a.get("severity") == "high")
    zero_days = sum(1 for a in alerts if a.get("is_zero_day", False))

    return {
        "title": "Alert Summary",
        "total": total,
        "critical": critical,
        "high": high,
        "zero_days": zero_days,
        "zero_day_rate": zero_days / total if total > 0 else 0.0,
    }


def model_status_card(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create a model status card.

    Args:
        model_info: Dictionary with model information.

    Returns:
        Dictionary with model status card data.
    """
    return {
        "title": model_info.get("name", "Unknown Model"),
        "loaded": model_info.get("loaded", False),
        "version": model_info.get("version", "N/A"),
        "last_trained": model_info.get("last_trained", "N/A"),
        "accuracy": model_info.get("accuracy", None),
    }


def _status_icon(status: str) -> str:
    """Map status to icon string.

    Args:
        status: Status string.

    Returns:
        Icon string.
    """
    icons = {
        "healthy": "\u2705",
        "warning": "\u26a0\ufe0f",
        "critical": "\u274c",
        "unknown": "\u2753",
    }
    return icons.get(status.lower(), icons["unknown"])
