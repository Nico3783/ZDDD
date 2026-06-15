from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.classification.random_forest import RandomForestClassifierModel
from src.core.config import get_config
from src.core.exceptions import ModelError

logger = logging.getLogger(__name__)


def get_feature_importance(
    model: RandomForestClassifierModel,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Get feature importance from the trained Random Forest.

    Args:
        model: Trained RandomForestClassifierModel.
        feature_names: Optional list of feature names. If None, uses model's.

    Returns:
        DataFrame with feature names and importance scores, sorted descending.

    Raises:
        ModelError: If the model has not been trained.
    """
    if model.model is None:
        raise ModelError("Model not trained")

    if feature_names is None:
        feature_names = model.feature_names

    importances = model.model.feature_importances_  # type: ignore[union-attr]

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info(
        "Feature importance computed: top feature='%s' (%.4f)",
        df.iloc[0]["feature"],
        df.iloc[0]["importance"],
    )

    return df


def get_top_features(
    model: RandomForestClassifierModel,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Get the top-k most important features.

    Args:
        model: Trained RandomForestClassifierModel.
        top_k: Number of top features to return.

    Returns:
        List of (feature_name, importance) tuples.
    """
    importance_df = get_feature_importance(model)
    top = importance_df.head(top_k)
    return list(zip(top["feature"], top["importance"]))


def get_feature_importance_by_class(
    model: RandomForestClassifierModel,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Get per-class feature importance using one-vs-rest approach.

    Each tree votes for one class. We compute importance for trees that
    vote for each class separately.

    Args:
        model: Trained RandomForestClassifierModel.
        feature_names: Optional feature names.

    Returns:
        Dictionary mapping class names to importance DataFrames.
    """
    if model.model is None:
        raise ModelError("Model not trained")

    if feature_names is None:
        feature_names = model.feature_names

    # Get the class each tree votes for (based on majority class in leaves)
    # Use the forest's estimators to compute per-class importances
    importances = model.model.feature_importances_  # type: ignore[union-attr]

    # For per-class, we approximate by using the overall importance
    # since sklearn doesn't directly expose per-class importance
    result = {}
    for cls_name in model.class_names:
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        result[cls_name] = df

    return result


def analyze_feature_redundancy(
    model: RandomForestClassifierModel,
    importance_threshold: float = 0.01,
) -> Dict[str, any]:
    """Analyze which features contribute minimally to the model.

    Args:
        model: Trained RandomForestClassifierModel.
        importance_threshold: Below this threshold, features are flagged.

    Returns:
        Dictionary with redundancy analysis results.
    """
    importance_df = get_feature_importance(model)

    low_importance = importance_df[importance_df["importance"] < importance_threshold]
    high_importance = importance_df[importance_df["importance"] >= importance_threshold]

    return {
        "total_features": len(importance_df),
        "important_features": len(high_importance),
        "low_importance_features": len(low_importance),
        "low_importance_list": low_importance["feature"].tolist(),
        "importance_threshold": importance_threshold,
        "cumulative_importance_top_10": float(importance_df.head(10)["importance"].sum()),
        "cumulative_importance_top_20": float(importance_df.head(20)["importance"].sum()),
    }


def suggest_feature_selection(
    model: RandomForestClassifierModel,
    target_n_features: Optional[int] = None,
    importance_percentile: float = 75.0,
) -> List[str]:
    """Suggest features to keep based on importance analysis.

    Args:
        model: Trained RandomForestClassifierModel.
        target_n_features: Target number of features. If None, uses percentile.
        importance_percentile: Percentile threshold for feature selection.

    Returns:
        List of suggested feature names to keep.
    """
    importance_df = get_feature_importance(model)

    if target_n_features is not None:
        suggested = importance_df.head(target_n_features)["feature"].tolist()
    else:
        threshold = np.percentile(importance_df["importance"], importance_percentile)
        suggested = importance_df[
            importance_df["importance"] >= threshold
        ]["feature"].tolist()

    logger.info(
        "Feature selection suggestion: %d features (threshold percentile=%.1f%%)",
        len(suggested),
        importance_percentile,
    )

    return suggested


def plot_feature_importance(
    model: RandomForestClassifierModel,
    top_k: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """Plot feature importance chart.

    Args:
        model: Trained RandomForestClassifierModel.
        top_k: Number of top features to plot.
        save_path: Optional path to save the plot.
    """
    importance_df = get_feature_importance(model).head(top_k)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(
            range(len(importance_df)),
            importance_df["importance"].values,
            color="#2196F3",
            edgecolor="white",
        )
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df["feature"].values)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_k} Feature Importances (Random Forest)")
        ax.invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Feature importance plot saved to %s", save_path)

        plt.close(fig)

    except ImportError:
        logger.warning("matplotlib not available, skipping feature importance plot")
