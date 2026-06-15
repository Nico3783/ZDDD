from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.config import get_config

logger = logging.getLogger(__name__)


def select_features_by_config(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Select only the feature columns defined in config.

    Args:
        df: DataFrame containing all columns.
        feature_columns: List of feature column names. If None, loads from config.

    Returns:
        DataFrame with only the selected feature columns.
    """
    if feature_columns is None:
        cfg = get_config().load("features")
        feature_columns = [f.strip() for f in cfg.get("flow_features", [])]

    # Filter to columns that exist in the DataFrame
    available = [f for f in feature_columns if f in df.columns]
    missing = [f for f in feature_columns if f not in df.columns]

    if missing:
        logger.warning("Config features not found in DataFrame: %s", missing)

    selected = df[available].copy()
    logger.info("Selected %d features from %d available columns", len(available), len(df.columns))
    return selected


def select_features_by_variance(
    features: pd.DataFrame,
    threshold: float = 0.0,
) -> List[str]:
    """Select features with variance above a threshold.

    Args:
        features: DataFrame of numeric features.
        threshold: Minimum variance threshold.

    Returns:
        List of feature names with variance > threshold.
    """
    variances = features.var()
    selected = variances[variances > threshold].index.tolist()
    logger.info(
        "Selected %d/%d features with variance > %.6f",
        len(selected),
        len(features.columns),
        threshold,
    )
    return selected


def select_features_by_correlation(
    features: pd.DataFrame,
    target: Optional[pd.Series] = None,
    threshold: float = 0.01,
) -> List[str]:
    """Select features based on correlation with target or between features.

    If target is provided, selects features with absolute correlation > threshold
    with the target. Otherwise, selects features that have at least one
    inter-feature correlation > threshold.

    Args:
        features: DataFrame of numeric features.
        target: Target series for correlation (optional).
        threshold: Minimum absolute correlation threshold.

    Returns:
        List of selected feature names.
    """
    if target is not None:
        # Select by correlation with target
        correlations = features.corrwith(target).abs()
        selected = correlations[correlations > threshold].index.tolist()
        logger.info(
            "Selected %d features correlated with target (threshold=%.3f)",
            len(selected),
            threshold,
        )
    else:
        # Select features with at least one meaningful inter-feature correlation
        corr_matrix = features.corr().abs()
        # For each feature, check max correlation with any other feature
        selected = []
        for col in corr_matrix.columns:
            max_corr = corr_matrix[col].drop(labels=[col]).max()
            if max_corr > threshold:
                selected.append(col)
        logger.info(
            "Selected %d features with inter-feature correlation > %.3f",
            len(selected),
            threshold,
        )

    return selected


def remove_redundant_features(
    features: pd.DataFrame,
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, List[str]]:
    """Remove one of each pair of highly correlated features.

    Keeps the feature that appears first in the DataFrame.

    Args:
        features: DataFrame of numeric features.
        threshold: Absolute correlation threshold above which features
            are considered redundant.

    Returns:
        Tuple of (reduced DataFrame, list of dropped feature names).
    """
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop: List[str] = []
    for col in upper.columns:
        for idx in upper.index:
            if upper.loc[idx, col] > threshold:
                to_drop.append(col)
                break

    reduced = features.drop(columns=to_drop)
    logger.info(
        "Removed %d redundant features (threshold=%.2f): %s",
        len(to_drop),
        threshold,
        to_drop,
    )
    return reduced, to_drop


def get_feature_names(config_path: Optional[str] = None) -> List[str]:
    """Get the list of selected feature names from config.

    Args:
        config_path: Optional path to a custom config. Not yet used.

    Returns:
        List of feature column names.
    """
    cfg = get_config().load("features")
    return [f.strip() for f in cfg.get("flow_features", [])]
