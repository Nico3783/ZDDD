from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.core.config import get_config
from src.core.exceptions import DataError

logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "mean": "mean",
    "median": "median",
    "most_frequent": "most_frequent",
    "constant": "constant",
    "drop": "drop",
}


@dataclass
class ImputationResult:
    """Result of an imputation operation."""

    original_rows: int = 0
    rows_after: int = 0
    columns_imputed: List[str] = field(default_factory=list)
    columns_with_missing: List[str] = field(default_factory=list)
    missing_before: Dict[str, int] = field(default_factory=dict)
    missing_after: Dict[str, int] = field(default_factory=dict)
    strategy_used: str = ""
    fill_value: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)


def get_missing_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Return a mapping of column name to count of missing values.

    Args:
        df: DataFrame to inspect.

    Returns:
        Dictionary mapping column names to their missing value counts.
    """
    missing = df.isnull().sum()
    return {col: int(count) for col, count in missing.items() if count > 0}


def impute_missing(
    df: pd.DataFrame,
    strategy: Optional[str] = None,
    fill_value: float = 0.0,
    columns: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, ImputationResult]:
    """Impute missing values in a DataFrame.

    Strategies:
        - ``mean`` / ``median`` / ``most_frequent`` / ``constant``: use
          sklearn SimpleImputer on the specified columns.
        - ``drop``: drop rows containing any missing values.

    Columns whose missing fraction exceeds ``threshold`` are dropped entirely.

    Args:
        df: Input DataFrame.
        strategy: Imputation strategy.  If None, reads from config.
        fill_value: Value to use when strategy is ``constant``.
        columns: Specific columns to impute.  If None, imputes all numeric columns.
        threshold: Fraction of missing values above which a column is dropped.

    Returns:
        Tuple of (imputed DataFrame, ImputationResult).

    Raises:
        DataError: If the strategy is unknown or the DataFrame is empty.
    """
    if df.empty:
        raise DataError("Cannot impute missing values in an empty DataFrame")

    if strategy is None:
        cfg = get_config().load("features")
        strategy = cfg.get("imputation", {}).get("strategy", "mean")

    if strategy not in STRATEGY_MAP:
        raise DataError(
            f"Unknown imputation strategy '{strategy}'. Choose from: {list(STRATEGY_MAP.keys())}"
        )

    result = ImputationResult(
        original_rows=len(df),
        strategy_used=strategy,
        fill_value=fill_value,
    )

    missing_summary = get_missing_summary(df)
    result.missing_before = missing_summary
    result.columns_with_missing = list(missing_summary.keys())

    if not missing_summary:
        logger.info("No missing values found — skipping imputation")
        result.rows_after = len(df)
        return df.copy(), result

    # Drop columns above threshold
    n_rows = len(df)
    cols_to_drop = [
        col for col, count in missing_summary.items()
        if count / n_rows > threshold
    ]
    if cols_to_drop:
        logger.info(
            "Dropping %d columns with >%.0f%% missing: %s",
            len(cols_to_drop), threshold * 100, cols_to_drop,
        )
        df = df.drop(columns=cols_to_drop)

    if strategy == "drop":
        n_before = len(df)
        df = df.dropna().reset_index(drop=True)
        n_dropped = n_before - len(df)
        result.rows_after = len(df)
        result.stats["rows_dropped"] = n_dropped
        logger.info("Dropped %d rows with missing values", n_dropped)
        return df, result

    # Determine columns to impute
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    numeric_cols_with_missing = [
        col for col in columns
        if col in df.columns and df[col].isnull().any()
    ]

    if not numeric_cols_with_missing:
        result.rows_after = len(df)
        return df.copy(), result

    result.columns_imputed = numeric_cols_with_missing

    sklearn_strategy = STRATEGY_MAP[strategy]
    imputer = SimpleImputer(strategy=sklearn_strategy, fill_value=fill_value)

    df_imputed = df.copy()
    df_imputed[numeric_cols_with_missing] = imputer.fit_transform(
        df[numeric_cols_with_missing]
    )

    result.missing_after = get_missing_summary(df_imputed)
    result.rows_after = len(df_imputed)
    result.stats["imputer_classes"] = {
        col: float(imputer.statistics_[i])
        for i, col in enumerate(numeric_cols_with_missing)
    }

    logger.info(
        "Imputed %d columns using '%s' strategy",
        len(numeric_cols_with_missing), strategy,
    )

    return df_imputed, result


def create_imputer(
    strategy: Optional[str] = None,
    fill_value: float = 0.0,
) -> SimpleImputer:
    """Create and return a fitted-ready SimpleImputer instance.

    Args:
        strategy: Imputation strategy.  If None, reads from config.
        fill_value: Value for constant strategy.

    Returns:
        An unfitted SimpleImputer configured with the requested strategy.
    """
    if strategy is None:
        cfg = get_config().load("features")
        strategy = cfg.get("imputation", {}).get("strategy", "mean")

    sklearn_strategy = STRATEGY_MAP.get(strategy, "mean")
    return SimpleImputer(strategy=sklearn_strategy, fill_value=fill_value)


def fit_imputer(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: Optional[str] = None,
    fill_value: float = 0.0,
) -> tuple[SimpleImputer, List[str]]:
    """Fit an imputer on the given columns and return it.

    Args:
        df: DataFrame to fit on.
        columns: Columns to fit.  If None, all numeric columns.
        strategy: Imputation strategy.
        fill_value: Value for constant strategy.

    Returns:
        Tuple of (fitted imputer, list of column names).
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    imputer = create_imputer(strategy=strategy, fill_value=fill_value)
    imputer.fit(df[columns])

    logger.info("Fitted imputer on %d columns with strategy '%s'", len(columns), strategy)
    return imputer, columns


def transform_with_imputer(
    df: pd.DataFrame,
    imputer: SimpleImputer,
    columns: List[str],
) -> pd.DataFrame:
    """Transform a DataFrame using a fitted imputer.

    Args:
        df: DataFrame to transform.
        imputer: Fitted SimpleImputer.
        columns: Columns the imputer was fitted on.

    Returns:
        DataFrame with imputed values in the specified columns.
    """
    df_result = df.copy()
    existing = [col for col in columns if col in df_result.columns]
    df_result[existing] = imputer.transform(df_result[existing])
    return df_result
