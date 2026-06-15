from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

import numpy as np
import pandas as pd

from src.core.exceptions import ConfigError, DataError


def check_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")
    return value


def check_non_negative_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def check_float(name: str, value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    val = float(value)
    if not (lo <= val <= hi):
        raise ConfigError(f"{name} must be in [{lo}, {hi}], got {val}")
    return val


def check_range(name: str, value: float, lo: float, hi: float) -> float:
    val = float(value)
    if val < lo or val > hi:
        raise ConfigError(f"{name} must be in [{lo}, {hi}], got {val}")
    return val


def check_required_keys(name: str, keys: Sequence[str], d: Dict[str, Any]) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ConfigError(f"{name} missing required keys: {missing}")


def check_not_empty(name: str, value: Any) -> Any:
    if value is None:
        raise ConfigError(f"{name} must not be None")
    if isinstance(value, (str, list, tuple, dict, set)) and len(value) == 0:
        raise ConfigError(f"{name} must not be empty")
    return value


def check_type(name: str, value: Any, expected: type) -> Any:
    if not isinstance(value, expected):
        raise ConfigError(
            f"{name} must be of type {expected.__name__}, got {type(value).__name__}"
        )
    return value


def check_dataframe_has_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame",
) -> None:
    """Validate that a DataFrame contains all required columns."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise DataError(f"{name} is missing required columns: {missing}")


def check_no_nan_inf(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Validate that a DataFrame contains no NaN or Inf values."""
    nan_count = int(df.isna().sum().sum())
    inf_count = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
    if nan_count > 0:
        raise DataError(f"{name} contains {nan_count} NaN values")
    if inf_count > 0:
        raise DataError(f"{name} contains {inf_count} infinite values")


def check_nan_inf(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Alias for check_no_nan_inf for backward compatibility."""
    check_no_nan_inf(df, name)


def validate_columns(
    df: pd.DataFrame,
    required: List[str],
    name: str = "DataFrame",
    optional: List[str] | None = None,
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{name} missing required columns: {missing}")
    if optional:
        known: Set[str] = set(required) | set(optional)
        unexpected = [c for c in df.columns if c not in known]
        if unexpected:
            raise DataError(f"{name} has unexpected columns: {unexpected}")
