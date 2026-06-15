from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.core.config import get_config
from src.core.exceptions import DataError

logger = logging.getLogger(__name__)


@dataclass
class SchemaValidationResult:
    """Result of feature schema validation."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def validate_feature_schema(
    features: pd.DataFrame,
    expected_features: Optional[List[str]] = None,
) -> SchemaValidationResult:
    """Validate that the feature DataFrame conforms to the expected schema.

    Checks:
        - All expected features are present
        - No unexpected extra features
        - All features are numeric
        - Feature count matches expected

    Args:
        features: DataFrame to validate.
        expected_features: List of expected feature column names.
            If None, loads from config.

    Returns:
        SchemaValidationResult with errors/warnings.
    """
    result = SchemaValidationResult()

    if expected_features is None:
        cfg = get_config().load("features")
        expected_features = [f.strip() for f in cfg.get("flow_features", [])]

    actual_features = set(features.columns)
    expected_set = set(expected_features)

    # Missing features
    missing = expected_set - actual_features
    if missing:
        result.errors.append(f"Missing features: {sorted(missing)}")
        result.is_valid = False

    # Extra features
    extra = actual_features - expected_set
    if extra:
        result.warnings.append(f"Extra features not in schema: {sorted(extra)}")

    # Non-numeric columns
    non_numeric = [col for col in features.columns if not pd.api.types.is_numeric_dtype(features[col])]
    if non_numeric:
        result.errors.append(f"Non-numeric features found: {non_numeric}")
        result.is_valid = False

    # Feature count
    result.stats["expected_count"] = len(expected_features)
    result.stats["actual_count"] = len(features.columns)
    result.stats["missing_count"] = len(missing)
    result.stats["extra_count"] = len(extra)

    return result


def validate_feature_values(
    features: pd.DataFrame,
    check_finite: bool = True,
    check_range: bool = True,
    value_range: tuple[float, float] = (-1e10, 1e10),
) -> SchemaValidationResult:
    """Validate feature values are within acceptable ranges.

    Args:
        features: DataFrame to validate.
        check_finite: Whether to check for non-finite values.
        check_range: Whether to check value ranges.
        value_range: (min, max) acceptable range for feature values.

    Returns:
        SchemaValidationResult with errors/warnings.
    """
    result = SchemaValidationResult()

    numeric_cols = features.select_dtypes(include=[np.number]).columns

    if check_finite:
        for col in numeric_cols:
            n_nan = int(features[col].isnull().sum())
            n_inf = int(np.isinf(features[col]).sum())
            if n_nan > 0:
                result.warnings.append(f"Column '{col}' has {n_nan} NaN values")
            if n_inf > 0:
                result.errors.append(f"Column '{col}' has {n_inf} infinite values")
                result.is_valid = False

    if check_range:
        for col in numeric_cols:
            min_val = float(features[col].min())
            max_val = float(features[col].max())
            if min_val < value_range[0] or max_val > value_range[1]:
                result.warnings.append(
                    f"Column '{col}' range [{min_val:.2e}, {max_val:.2e}] "
                    f"exceeds expected [{value_range[0]:.2e}, {value_range[1]:.2e}]"
                )

    return result


def get_feature_set_difference(
    features_a: pd.DataFrame,
    features_b: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Compare feature sets between two DataFrames.

    Args:
        features_a: First DataFrame.
        features_b: Second DataFrame.

    Returns:
        Dictionary with 'only_in_a', 'only_in_b', and 'common' lists.
    """
    cols_a = set(features_a.columns)
    cols_b = set(features_b.columns)

    return {
        "only_in_a": sorted(cols_a - cols_b),
        "only_in_b": sorted(cols_b - cols_a),
        "common": sorted(cols_a & cols_b),
    }


def ensure_feature_consistency(
    train_features: pd.DataFrame,
    *other_features: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Ensure all DataFrames have the same feature columns as training data.

    Drops extra columns and adds missing columns (filled with 0) in
    non-training DataFrames to match the training feature set.

    Args:
        train_features: Training DataFrame (reference).
        *other_features: Other DataFrames to align.

    Returns:
        Tuple of (train, aligned_other_1, aligned_other_2, ...).
    """
    train_cols = set(train_features.columns)
    aligned = [train_features]

    for df in other_features:
        # Drop columns not in training set
        extra = set(df.columns) - train_cols
        if extra:
            logger.warning("Dropping %d columns not in training set: %s", len(extra), extra)

        # Add missing columns filled with 0
        missing = train_cols - set(df.columns)
        if missing:
            logger.warning("Adding %d missing columns filled with 0: %s", len(missing), missing)

        df_aligned = df.reindex(columns=sorted(train_cols), fill_value=0.0)
        aligned.append(df_aligned)

    return tuple(aligned)
