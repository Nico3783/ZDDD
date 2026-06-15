"""Alerting and security event management.

Provides JSON/CSV alert logging, alert generation, formatting, dispatch, and archival.
"""

from src.alerting.csv_logger import CSVAlertLogger
from src.alerting.formatter import (
    SEVERITY_ICONS,
    format_alert_batch,
    format_alert_csv_row,
    format_alert_json,
    format_alert_summary,
    format_alert_text,
)
from src.alerting.generator import (
    AlertGenerator,
    generate_alerts_from_results,
)
from src.alerting.json_logger import JSONAlertLogger
from src.alerting.logger import AlertLogger
from src.alerting.notifier import (
    AlertNotifier,
    dispatch_alerts,
)

__all__ = [
    # Formatter
    "SEVERITY_ICONS",
    "format_alert_batch",
    "format_alert_csv_row",
    "format_alert_json",
    "format_alert_summary",
    "format_alert_text",
    # Generator
    "AlertGenerator",
    "generate_alerts_from_results",
    # Loggers
    "AlertLogger",
    "JSONAlertLogger",
    "CSVAlertLogger",
    # Notifier
    "AlertNotifier",
    "dispatch_alerts",
]
