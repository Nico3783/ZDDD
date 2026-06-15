"""Dashboard UI component modules."""

from src.dashboard.components.cards import alert_summary_card, metric_card, model_status_card, status_card
from src.dashboard.components.charts import (
    create_anomaly_score_distribution,
    create_class_distribution_chart,
    create_latency_chart,
    create_severity_chart,
    create_timeseries_chart,
    create_zero_day_timeline,
)
from src.dashboard.components.tables import (
    alerts_table,
    confusion_matrix_table,
    feature_importance_table,
    model_comparison_table,
)

__all__ = [
    # Cards
    "alert_summary_card",
    "metric_card",
    "model_status_card",
    "status_card",
    # Charts
    "create_anomaly_score_distribution",
    "create_class_distribution_chart",
    "create_latency_chart",
    "create_severity_chart",
    "create_timeseries_chart",
    "create_zero_day_timeline",
    # Tables
    "alerts_table",
    "confusion_matrix_table",
    "feature_importance_table",
    "model_comparison_table",
]
