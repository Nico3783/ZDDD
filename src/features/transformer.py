from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.core.config import get_config

logger = logging.getLogger(__name__)


def add_interaction_features(
    features: pd.DataFrame,
    feature_pairs: Optional[list[tuple[str, str]]] = None,
    operations: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Create interaction features from pairs of existing features.

    Args:
        features: DataFrame of numeric features.
        feature_pairs: List of (col_a, col_b) tuples. If None, uses top
            correlated pairs.
        operations: List of operations to apply ('multiply', 'divide', 'add',
            'subtract'). Defaults to ['multiply'].

    Returns:
        DataFrame with original and interaction features added.
    """
    if operations is None:
        operations = ["multiply"]

    if feature_pairs is None:
        # Auto-select pairs from top variance features
        variances = features.var().sort_values(ascending=False)
        top_cols = variances.head(10).index.tolist()
        feature_pairs = [
            (top_cols[i], top_cols[j])
            for i in range(len(top_cols))
            for j in range(i + 1, min(i + 3, len(top_cols)))
        ]

    result = features.copy()
    n_created = 0

    for col_a, col_b in feature_pairs:
        if col_a not in features.columns or col_b not in features.columns:
            continue

        for op in operations:
            feature_name = f"{col_a.strip()}__{op}__{col_b.strip()}"

            if op == "multiply":
                result[feature_name] = features[col_a] * features[col_b]
            elif op == "divide":
                # Safe division with zero handling
                denominator = features[col_b].replace(0, np.nan)
                result[feature_name] = features[col_a] / denominator
                result[feature_name] = result[feature_name].fillna(0.0)
            elif op == "add":
                result[feature_name] = features[col_a] + features[col_b]
            elif op == "subtract":
                result[feature_name] = features[col_a] - features[col_b]

            n_created += 1

    logger.info("Created %d interaction features from %d pairs", n_created, len(feature_pairs))
    return result


def add_statistical_features(
    features: pd.DataFrame,
    row_wise: bool = True,
) -> pd.DataFrame:
    """Add row-wise statistical summary features.

    Args:
        features: DataFrame of numeric features.
        row_wise: Whether to compute row-wise statistics.

    Returns:
        DataFrame with statistical features added.
    """
    result = features.copy()

    if row_wise:
        numeric = features.select_dtypes(include=[np.number])
        result["row_mean"] = numeric.mean(axis=1)
        result["row_std"] = numeric.std(axis=1)
        result["row_max"] = numeric.max(axis=1)
        result["row_min"] = numeric.min(axis=1)
        result["row_median"] = numeric.median(axis=1)

        logger.info("Added 5 row-wise statistical features")

    return result


def add_ratio_features(
    features: pd.DataFrame,
    pairs: Optional[list[tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Add ratio features for specified pairs.

    Args:
        features: DataFrame of numeric features.
        pairs: List of (numerator, denominator) column name pairs.

    Returns:
        DataFrame with ratio features added.
    """
    result = features.copy()

    if pairs is None:
        # Default: forward/backward packet ratios
        default_pairs = [
            (" Fwd Packet Length Mean", " Bwd Packet Length Mean"),
            (" Total Fwd Packets", " Total Backward Packets"),
            (" Fwd Packets/s", " Bwd Packets/s"),
            (" Subflow Fwd Bytes", " Subflow Bwd Bytes"),
        ]
        pairs = [(a.strip(), b.strip()) for a, b in default_pairs]

    n_created = 0
    for num_col, den_col in pairs:
        if num_col in features.columns and den_col in features.columns:
            ratio_name = f"{num_col}__ratio__{den_col}"
            denominator = features[den_col].replace(0, np.nan)
            result[ratio_name] = features[num_col] / denominator
            result[ratio_name] = result[ratio_name].fillna(0.0)
            n_created += 1

    logger.info("Created %d ratio features", n_created)
    return result


def log_transform_features(
    features: pd.DataFrame,
    columns: Optional[list[str]] = None,
    offset: float = 1.0,
) -> pd.DataFrame:
    """Apply log(1 + x) transformation to selected features.

    Useful for features with heavy right skew.

    Args:
        features: DataFrame of numeric features.
        columns: Columns to transform. If None, transforms all numeric.
        offset: Offset added before log to avoid log(0).

    Returns:
        DataFrame with log-transformed features replacing originals.
    """
    result = features.copy()

    if columns is None:
        columns = list(features.select_dtypes(include=[np.number]).columns)

    for col in columns:
        if col in result.columns:
            # Shift to ensure all values are positive
            min_val = result[col].min()
            shift = abs(min_val) + offset if min_val < 0 else offset
            result[col] = np.log1p(result[col] + shift - offset)

    logger.info("Applied log transform to %d features", len(columns))
    return result


def apply_feature_transformations(
    features: pd.DataFrame,
    interaction: bool = False,
    statistical: bool = False,
    ratios: bool = False,
    log_transform: bool = False,
) -> pd.DataFrame:
    """Apply the full feature transformation pipeline.

    Args:
        features: DataFrame of numeric features.
        interaction: Whether to add interaction features.
        statistical: Whether to add row-wise statistical features.
        ratios: Whether to add ratio features.
        log_transform: Whether to apply log transformation.

    Returns:
        DataFrame with transformations applied.
    """
    result = features.copy()

    if interaction:
        result = add_interaction_features(result)

    if statistical:
        result = add_statistical_features(result)

    if ratios:
        result = add_ratio_features(result)

    if log_transform:
        result = log_transform_features(result)

    logger.info(
        "Feature transformations complete: %d -> %d columns",
        len(features.columns),
        len(result.columns),
    )
    return result
