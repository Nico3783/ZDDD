"""Models page — trained model status and performance comparison."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.dashboard.components.cards import model_status_card
from src.dashboard.components.tables import (
    confusion_matrix_table,
    feature_importance_table,
    model_comparison_table,
)

logger = logging.getLogger(__name__)


def render_models(
    models: List[Dict[str, Any]],
    feature_importances: Dict[str, List[Dict[str, Any]]],
    confusion_matrices: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Render the models dashboard page.

    Args:
        models: List of model info dictionaries.
        feature_importances: Dict mapping model names to importance lists.
        confusion_matrices: Dict mapping model names to confusion matrix data.

    Returns:
        Dictionary with page content for rendering.
    """
    model_cards = [model_status_card(m) for m in models]

    comparison_table = model_comparison_table(models)

    importance_tables: Dict[str, Dict[str, Any]] = {}
    for model in models:
        model_name = model.get("name", "unknown")
        if model_name in feature_importances:
            importance_tables[model_name] = feature_importance_table(
                feature_importances[model_name]
            )

    confusion_tables: Dict[str, Dict[str, Any]] = {}
    for model_name, cm_data in confusion_matrices.items():
        matrix = cm_data.get("matrix", [])
        labels = cm_data.get("labels", [])
        if matrix and labels:
            confusion_tables[model_name] = confusion_matrix_table(matrix, labels)

    return {
        "title": "Models",
        "model_cards": model_cards,
        "comparison_table": comparison_table,
        "importance_tables": importance_tables,
        "confusion_tables": confusion_tables,
    }
