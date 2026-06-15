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
class CleaningResult:
    """Result of a data cleaning operation."""

    original_rows: int = 0
    rows_after: int = 0
    rows_dropped: int = 0
    columns_cleaned: List[str] = field(default_factory=list)
    operations_applied: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def drop_duplicates(df: pd.DataFrame, reset_index: bool = True) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows from the DataFrame.

    Args:
        df: DataFrame to clean.
        reset_index: Whether to reset the index after dropping.

    Returns:
        Tuple of (cleaned DataFrame, number of rows dropped).
    """
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=reset_index if reset_index else False)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info("Dropped %d duplicate rows (from %d to %d)", n_dropped, n_before, len(df))
    return df, n_dropped


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
    threshold: float = 0.5,
    fill_value: float = 0.0,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Handle missing values in the DataFrame.

    Args:
        df: DataFrame to clean.
        strategy: One of 'drop', 'fill', or 'drop_columns'.
            - drop: Drop rows with any null values.
            - fill: Fill null values with fill_value.
            - drop_columns: Drop columns with >threshold fraction of nulls,
              then drop remaining rows with nulls.
        threshold: Fraction threshold for drop_columns strategy (0.0-1.0).
        fill_value: Value to use for fill strategy.

    Returns:
        Tuple of (cleaned DataFrame, dict of column -> nulls dropped).
    """
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())

    if total_nulls == 0:
        logger.info("No missing values found")
        return df, {}

    dropped_cols: Dict[str, int] = {}

    if strategy == "drop_columns":
        # Drop columns with too many nulls
        cols_to_drop = [
            col for col in df.columns
            if df[col].isnull().mean() > threshold
        ]
        if cols_to_drop:
            for col in cols_to_drop:
                dropped_cols[col] = int(df[col].isnull().sum())
            df = df.drop(columns=cols_to_drop)
            logger.info("Dropped %d columns with >%.0f%% nulls: %s", len(cols_to_drop), threshold * 100, cols_to_drop)

        # Drop remaining rows with nulls
        n_before = len(df)
        df = df.dropna().reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info("Dropped %d rows with remaining null values", n_dropped)

    elif strategy == "fill":
        df = df.fillna(fill_value)
        logger.info("Filled %d null values with %s", total_nulls, fill_value)

    else:  # strategy == "drop"
        n_before = len(df)
        df = df.dropna().reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info("Dropped %d rows with null values", n_dropped)

    return df, dropped_cols


def handle_infinite_values(
    df: pd.DataFrame,
    replace_with: float = 0.0,
) -> tuple[pd.DataFrame, List[str]]:
    """Replace infinite values with a finite replacement.

    Args:
        df: DataFrame to clean.
        replace_with: Value to replace inf/-inf with.

    Returns:
        Tuple of (cleaned DataFrame, list of columns that had infinities).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_with_inf: List[str] = []

    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_count = int(np.isinf(df[col]).sum())
            df[col] = df[col].replace([np.inf, -np.inf], replace_with)
            cols_with_inf.append(col)
            logger.debug("Replaced %d infinite values in '%s' with %s", inf_count, col, replace_with)

    if cols_with_inf:
        logger.info("Replaced infinite values in %d columns: %s", len(cols_with_inf), cols_with_inf)

    return df, cols_with_inf


def clip_outliers(
    df: pd.DataFrame,
    percentile: float = 99.0,
    columns: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, Dict[str, tuple[float, float]]]:
    """Clip outlier values at the given percentile.

    Values below the (100-percentile)/2 percentile or above the
    (100+percentile)/2 percentile are clipped.

    Args:
        df: DataFrame to clean.
        percentile: Percentile threshold for clipping (e.g., 99 clips at 0.5th and 99.5th).
        columns: Specific columns to clip. If None, clips all numeric columns.

    Returns:
        Tuple of (cleaned DataFrame, dict of column -> (lower, upper) clip bounds).
    """
    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)

    clip_bounds: Dict[str, tuple[float, float]] = {}
    lower_pct = (100 - percentile) / 2
    upper_pct = 100 - lower_pct

    for col in columns:
        if col in df.columns:
            lower = float(df[col].quantile(lower_pct / 100))
            upper = float(df[col].quantile(upper_pct / 100))
            n_clipped = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_clipped > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                clip_bounds[col] = (lower, upper)
                logger.debug("Clipped %d outliers in '%s' to [%.4f, %.4f]", n_clipped, col, lower, upper)

    if clip_bounds:
        logger.info("Clipped outliers in %d columns", len(clip_bounds))

    return df, clip_bounds


def clean_dataset(
    df: pd.DataFrame,
    drop_dupes: bool = True,
    handle_missing: str = "drop",
    handle_inf: bool = True,
    clip: bool = False,
    clip_percentile: float = 99.0,
) -> tuple[pd.DataFrame, CleaningResult]:
    """Apply the full cleaning pipeline to a DataFrame.

    Pipeline order:
        1. Drop duplicates
        2. Handle infinite values
        3. Handle missing values
        4. Clip outliers (optional)

    Args:
        df: DataFrame to clean.
        drop_dupes: Whether to drop duplicate rows.
        handle_missing: Missing value strategy ('drop', 'fill', 'drop_columns').
        handle_inf: Whether to replace infinite values.
        clip: Whether to clip outliers.
        clip_percentile: Percentile for outlier clipping.

    Returns:
        Tuple of (cleaned DataFrame, CleaningResult with stats).
    """
    result = CleaningResult(original_rows=len(df))
    operations: List[str] = []

    # Step 1: Duplicates
    if drop_dupes:
        df, n_dupes = drop_duplicates(df)
        result.stats["duplicates_dropped"] = n_dupes
        if n_dupes > 0:
            operations.append(f"duplicates({n_dupes})")

    # Step 2: Infinite values
    if handle_inf:
        df, inf_cols = handle_infinite_values(df)
        result.stats["infinite_columns"] = inf_cols
        if inf_cols:
            operations.append(f"infinite({len(inf_cols)} cols)")

    # Step 3: Missing values
    null_before = int(df.isnull().sum().sum())
    df, dropped_cols = handle_missing_values(df, strategy=handle_missing)
    result.stats["nulls_before"] = null_before
    result.stats["columns_dropped_for_nulls"] = dropped_cols
    if null_before > 0:
        operations.append(f"nulls({null_before})")

    # Step 4: Outlier clipping
    if clip:
        df, clip_bounds = clip_outliers(df, percentile=clip_percentile)
        result.stats["clip_bounds"] = {k: list(v) for k, v in clip_bounds.items()}
        if clip_bounds:
            operations.append(f"clip({len(clip_bounds)} cols)")

    result.rows_after = len(df)
    result.rows_dropped = result.original_rows - result.rows_after
    result.operations_applied = operations
    result.columns_cleaned = list(df.columns)

    logger.info(
        "Cleaning complete: %d -> %d rows (dropped %d). Operations: %s",
        result.original_rows,
        result.rows_after,
        result.rows_dropped,
        operations or ["none"],
    )

    return df, result
