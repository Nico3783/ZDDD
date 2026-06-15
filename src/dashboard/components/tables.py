from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def alerts_table(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format alerts for table display.

    Args:
        alerts: List of alert dictionaries.

    Returns:
        Dictionary with table configuration.
    """
    columns = [
        {"key": "timestamp", "label": "Time", "type": "datetime"},
        {"key": "severity", "label": "Severity", "type": "badge"},
        {"key": "predicted_class", "label": "Class", "type": "text"},
        {"key": "anomaly_score", "label": "Score", "type": "number"},
        {"key": "is_zero_day", "label": "Zero-Day", "type": "boolean"},
    ]

    rows = []
    for alert in alerts:
        rows.append({
            "timestamp": alert.get("timestamp", "N/A"),
            "severity": alert.get("severity", "unknown"),
            "predicted_class": alert.get("predicted_class", "UNKNOWN"),
            "anomaly_score": round(alert.get("anomaly_score", 0.0), 4),
            "is_zero_day": alert.get("is_zero_day", False),
        })

    return {
        "columns": columns,
        "rows": rows,
        "total": len(rows),
        "sortable": True,
        "filterable": True,
    }


def model_comparison_table(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format model comparison data for table display.

    Args:
        models: List of model info dictionaries.

    Returns:
        Dictionary with table configuration.
    """
    columns = [
        {"key": "name", "label": "Model", "type": "text"},
        {"key": "accuracy", "label": "Accuracy", "type": "number"},
        {"key": "f1_score", "label": "F1 Score", "type": "number"},
        {"key": "detection_rate", "label": "Detection Rate", "type": "number"},
        {"key": "false_alarm_rate", "label": "FAR", "type": "number"},
        {"key": "trained_at", "label": "Last Trained", "type": "datetime"},
    ]

    rows = []
    for model in models:
        rows.append({
            "name": model.get("name", "Unknown"),
            "accuracy": round(model.get("accuracy", 0.0), 4),
            "f1_score": round(model.get("f1_score", 0.0), 4),
            "detection_rate": round(model.get("detection_rate", 0.0), 4),
            "false_alarm_rate": round(model.get("false_alarm_rate", 0.0), 4),
            "trained_at": model.get("trained_at", "N/A"),
        })

    return {
        "columns": columns,
        "rows": rows,
        "total": len(rows),
        "sortable": True,
    }


def feature_importance_table(importances: List[Dict[str, Any]], top_n: int = 10) -> Dict[str, Any]:
    """Format feature importance data for table display.

    Args:
        importances: List of dicts with 'feature' and 'importance'.
        top_n: Number of top features to show.

    Returns:
        Dictionary with table configuration.
    """
    sorted_imp = sorted(importances, key=lambda x: x.get("importance", 0), reverse=True)[:top_n]

    columns = [
        {"key": "rank", "label": "#", "type": "number"},
        {"key": "feature", "label": "Feature", "type": "text"},
        {"key": "importance", "label": "Importance", "type": "number"},
        {"key": "bar", "label": "", "type": "bar"},
    ]

    max_imp = sorted_imp[0].get("importance", 1.0) if sorted_imp else 1.0
    rows = []
    for i, item in enumerate(sorted_imp, 1):
        imp = item.get("importance", 0.0)
        rows.append({
            "rank": i,
            "feature": item.get("feature", "unknown"),
            "importance": round(imp, 4),
            "bar": round(imp / max_imp, 2) if max_imp > 0 else 0,
        })

    return {
        "columns": columns,
        "rows": rows,
        "total": len(rows),
    }


def confusion_matrix_table(matrix: List[List[int]], labels: List[str]) -> Dict[str, Any]:
    """Format confusion matrix for table display.

    Args:
        matrix: 2D confusion matrix.
        labels: Class labels.

    Returns:
        Dictionary with table configuration.
    """
    columns = [{"key": "actual", "label": "Actual \\ Predicted", "type": "text"}]
    for label in labels:
        columns.append({"key": f"pred_{label}", "label": label, "type": "number"})

    rows = []
    for i, label in enumerate(labels):
        row: Dict[str, Any] = {"actual": label}
        for j, pred_label in enumerate(labels):
            row[f"pred_{pred_label}"] = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else 0
        rows.append(row)

    return {
        "columns": columns,
        "rows": rows,
        "total": len(rows),
    }
