from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.config import get_config
from src.core.exceptions import DataError
from src.features.extractor import (
    ExtractionResult,
    extract_aggregate_features,
    extract_difference_features,
    extract_ratio_features,
    extract_statistical_features,
    select_top_features,
)
from src.features.selector import (
    get_feature_names,
    remove_redundant_features,
    select_features_by_config,
    select_features_by_correlation,
    select_features_by_variance,
)
from src.features.transformer import (
    add_interaction_features,
    add_ratio_features,
    add_statistical_features,
    apply_feature_transformations,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringResult:
    """Result of the feature engineering pipeline."""

    original_features: int = 0
    final_features: int = 0
    features_added: int = 0
    features_removed: int = 0
    methods_applied: List[str] = field(default_factory=list)
    selected_features: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def run_feature_engineering(
    df: pd.DataFrame,
    label_column: str = "Label",
    extract: bool = True,
    transform: bool = True,
    select: bool = True,
    n_features: Optional[int] = None,
    correlation_threshold: Optional[float] = None,
    variance_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, FeatureEngineeringResult, List[str]]:
    """Run the full feature engineering pipeline: extract → transform → select.

    This is the primary entry point for feature engineering on preprocessed DataFrames.

    Args:
        df: Preprocessed input DataFrame.
        label_column: Name of the label column.
        extract: Whether to run feature extraction.
        transform: Whether to run feature transformations.
        select: Whether to run feature selection.
        n_features: Maximum number of features to keep. If None, uses config.
        correlation_threshold: Max correlation for redundancy removal. If None, uses config.
        variance_threshold: Min variance for variance-based selection. If None, uses config.

    Returns:
        Tuple of (engineered DataFrame, FeatureEngineeringResult, selected feature names).

    Raises:
        DataError: If the input DataFrame is empty.
    """
    if df.empty:
        raise DataError("Cannot engineer features on an empty DataFrame")

    # Separate features and labels
    has_label = label_column in df.columns
    labels = df[label_column] if has_label else None
    features = df.drop(columns=[label_column]) if has_label else df.copy()

    result = FeatureEngineeringResult(
        original_features=len(features.columns),
    )

    logger.info("Starting feature engineering: %d features", result.original_features)

    # Step 1: Extraction
    if extract:
        features = extract_statistical_features(features)
        features = extract_ratio_features(features)
        features = extract_difference_features(features)
        result.methods_applied.extend([
            "statistical_features", "ratio_features", "difference_features",
        ])
        logger.info("After extraction: %d features", len(features.columns))

    # Step 2: Transformation
    if transform:
        features = add_statistical_features(features)
        features = add_interaction_features(features)
        features = add_ratio_features(features)
        result.methods_applied.extend([
            "add_statistical", "add_interaction", "add_ratio",
        ])
        logger.info("After transformation: %d features", len(features.columns))

    # Drop any NaN columns created during extraction/transformation
    nan_cols = features.columns[features.isna().any()].tolist()
    if nan_cols:
        features = features.drop(columns=nan_cols)
        logger.info("Dropped %d columns with NaN from extraction", len(nan_cols))

    result.features_added = len(features.columns) - result.original_features

    # Step 3: Selection
    if select:
        if n_features is None:
            cfg = get_config().load("features")
            n_features = cfg.get("selection", {}).get("n_features", 30)

        if correlation_threshold is None:
            cfg = get_config().load("features")
            correlation_threshold = cfg.get("selection", {}).get("correlation_threshold", 0.95)

        if variance_threshold is None:
            cfg = get_config().load("features")
            variance_threshold = cfg.get("selection", {}).get("variance_threshold", 0.01)

        # Remove highly correlated features
        features = remove_redundant_features(features, threshold=correlation_threshold)
        result.methods_applied.append("correlation_removal")

        # Remove low-variance features
        low_var = select_features_by_variance(features, threshold=variance_threshold)
        if low_var:
            features = features[low_var]
            result.methods_applied.append("variance_selection")

        # Keep top N by variance if still too many
        if len(features.columns) > n_features:
            top_cols = select_top_features(features, labels, n_features=n_features)
            features = features[top_cols]
            result.methods_applied.append("top_n_selection")

        logger.info("After selection: %d features", len(features.columns))

    result.final_features = len(features.columns)
    result.selected_features = features.columns.tolist()
    result.features_removed = result.original_features - result.final_features + result.features_added

    # Reattach labels
    if has_label:
        features[label_column] = labels.values

    logger.info(
        "Feature engineering complete: %d → %d features, methods=%s",
        result.original_features, result.final_features, result.methods_applied,
    )

    return features, result, result.selected_features


def run_feature_engineering_on_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str = "Label",
    selected_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureEngineeringResult]:
    """Apply consistent feature engineering across train/val/test splits.

    Fits feature selection on training data, then applies the same feature
    set to validation and test sets.

    Args:
        train_df: Training DataFrame (preprocessed).
        val_df: Validation DataFrame (preprocessed).
        test_df: Test DataFrame (preprocessed).
        label_column: Name of the label column.
        selected_features: Pre-selected feature list. If None, runs full pipeline on train.

    Returns:
        Tuple of (train, val, test DataFrames, FeatureEngineeringResult).
    """
    if train_df.empty:
        raise DataError("Training DataFrame is empty")

    has_label = label_column in train_df.columns

    if selected_features is None:
        _, result, selected_features = run_feature_engineering(
            train_df, label_column=label_column,
        )
    else:
        result = FeatureEngineeringResult(
            original_features=len(train_df.columns) - (1 if has_label else 0),
            final_features=len(selected_features),
            selected_features=selected_features,
        )

    # Apply same features to all splits
    cols_to_keep = [c for c in selected_features if c in train_df.columns]
    if has_label:
        cols_to_keep.append(label_column)

    train_out = train_df[cols_to_keep].copy()
    val_out = val_df[cols_to_keep].copy() if set(cols_to_keep).issubset(val_df.columns) else val_df[[c for c in cols_to_keep if c in val_df.columns]].copy()
    test_out = test_df[cols_to_keep].copy() if set(cols_to_keep).issubset(test_df.columns) else test_df[[c for c in cols_to_keep if c in test_df.columns]].copy()

    logger.info(
        "Feature engineering applied to splits: train=%s, val=%s, test=%s",
        train_out.shape, val_out.shape, test_out.shape,
    )

    return train_out, val_out, test_out, result
