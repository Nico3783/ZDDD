from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.config import get_config
from src.core.exceptions import DataError

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of a feature extraction operation."""

    original_columns: List[str] = field(default_factory=list)
    extracted_columns: List[str] = field(default_factory=list)
    total_features: int = 0
    extraction_methods: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def extract_statistical_features(
    df: pd.DataFrame,
    window_cols: Optional[List[str]] = None,
    window_size: int = 5,
) -> pd.DataFrame:
    """Extract rolling statistical features (mean, std, min, max) from numeric columns.

    Args:
        df: Input DataFrame with time-series-like numeric features.
        window_cols: Columns to compute rolling stats on. If None, uses all numeric.
        window_size: Rolling window size.

    Returns:
        DataFrame with original and rolling statistical features added.
    """
    if window_cols is None:
        window_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    result = df.copy()
    n_created = 0

    for col in window_cols:
        if col not in result.columns:
            continue

        for stat_name, stat_func in [
            ("rmean", lambda s: s.rolling(window_size, min_periods=1).mean()),
            ("rstd", lambda s: s.rolling(window_size, min_periods=1).std().fillna(0.0)),
            ("rmin", lambda s: s.rolling(window_size, min_periods=1).min()),
            ("rmax", lambda s: s.rolling(window_size, min_periods=1).max()),
        ]:
            feature_name = f"{col.strip()}__{stat_name}"
            result[feature_name] = stat_func(result[col])
            n_created += 1

    logger.info(
        "Extracted %d rolling statistical features from %d columns",
        n_created, len(window_cols),
    )
    return result


def extract_ratio_features(
    df: pd.DataFrame,
    numerator_cols: Optional[List[str]] = None,
    denominator_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract ratio features between pairs of numeric columns.

    Args:
        df: Input DataFrame.
        numerator_cols: Columns to use as numerators. If None, uses first 5 numeric.
        denominator_cols: Columns to use as denominators. If None, uses last 5 numeric.

    Returns:
        DataFrame with ratio features added.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numerator_cols is None:
        numerator_cols = numeric_cols[:5]
    if denominator_cols is None:
        denominator_cols = numeric_cols[-5:]

    result = df.copy()
    n_created = 0

    for num_col in numerator_cols:
        if num_col not in result.columns:
            continue
        for den_col in denominator_cols:
            if den_col not in result.columns or num_col == den_col:
                continue

            feature_name = f"{num_col.strip()}__ratio__{den_col.strip()}"
            denominator = result[den_col].replace(0, np.nan)
            result[feature_name] = (result[num_col] / denominator).fillna(0.0)
            n_created += 1

    logger.info("Extracted %d ratio features", n_created)
    return result


def extract_difference_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lag: int = 1,
) -> pd.DataFrame:
    """Extract lag-difference features for consecutive column pairs.

    Args:
        df: Input DataFrame.
        columns: Columns to compute differences for. If None, uses all numeric.
        lag: Number of periods for diff.

    Returns:
        DataFrame with difference features added.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = df.copy()
    n_created = 0

    for i in range(len(columns)):
        for j in range(i + 1, min(i + 3, len(columns))):
            col_a, col_b = columns[i], columns[j]
            if col_a not in result.columns or col_b not in result.columns:
                continue

            feature_name = f"{col_a.strip()}__diff__{col_b.strip()}"
            result[feature_name] = result[col_a] - result[col_b]
            n_created += 1

    logger.info("Extracted %d difference features", n_created)
    return result


def extract_aggregate_features(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    agg_cols: Optional[List[str]] = None,
    agg_funcs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract group-by aggregate features.

    Args:
        df: Input DataFrame.
        group_cols: Columns to group by. If None, skips aggregation.
        agg_cols: Columns to aggregate. If None, uses all numeric.
        agg_funcs: Aggregation functions ('mean', 'std', 'min', 'max', 'sum').

    Returns:
        DataFrame with aggregate features merged back.
    """
    if group_cols is None or not any(c in df.columns for c in group_cols):
        return df.copy()

    if agg_cols is None:
        agg_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if agg_funcs is None:
        agg_funcs = ["mean", "std"]

    valid_group = [c for c in group_cols if c in df.columns]
    valid_agg = [c for c in agg_cols if c in df.columns]

    if not valid_group or not valid_agg:
        return df.copy()

    result = df.copy()
    n_created = 0

    for func in agg_funcs:
        agg_df = result.groupby(valid_group)[valid_agg].agg(func)
        agg_df.columns = [
            f"{col.strip()}__{func}__{'_'.join(valid_group)}"
            for col in valid_agg
        ]
        result = result.merge(agg_df, on=valid_group, how="left")
        n_created += len(valid_agg)

    logger.info(
        "Extracted %d aggregate features (grouped by %s)",
        n_created, valid_group,
    )
    return result


def select_top_features(
    df: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    n_features: int = 30,
) -> List[str]:
    """Select top N features by variance or mutual information with labels.

    Args:
        df: DataFrame of numeric features.
        labels: Optional target labels for mutual-information ranking.
        n_features: Number of top features to select.

    Returns:
        List of selected feature column names.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if labels is not None:
        from sklearn.feature_selection import mutual_info_classif

        mi_scores = mutual_info_classif(df[numeric_cols], labels, random_state=42)
        ranking = pd.Series(mi_scores, index=numeric_cols).sort_values(ascending=False)
    else:
        ranking = df[numeric_cols].var().sort_values(ascending=False)

    selected = ranking.head(n_features).index.tolist()
    logger.info("Selected top %d features from %d candidates", len(selected), len(numeric_cols))
    return selected
