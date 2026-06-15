"""Unit tests for src.preprocessing.cleaner module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaner import (
    CleaningResult,
    clip_outliers,
    clean_dataset,
    drop_duplicates,
    handle_infinite_values,
    handle_missing_values,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df_with_nans() -> pd.DataFrame:
    """Return a small DataFrame with known NaN positions."""
    return pd.DataFrame({
        "col_a": [1.0, np.nan, 3.0, np.nan, 5.0],
        "col_b": [10.0, 20.0, np.nan, 40.0, 50.0],
        "col_c": [100.0, 200.0, 300.0, 400.0, np.nan],
    })


def _df_no_nans() -> pd.DataFrame:
    """Return a small DataFrame with no missing values."""
    return pd.DataFrame({
        "col_a": [1.0, 2.0, 3.0],
        "col_b": [10.0, 20.0, 30.0],
    })


def _df_with_duplicates() -> pd.DataFrame:
    """Return a DataFrame with duplicate rows."""
    return pd.DataFrame({
        "col_a": [1, 2, 1, 3, 2],
        "col_b": [10, 20, 10, 30, 20],
    })


# ---------------------------------------------------------------------------
# drop_duplicates
# ---------------------------------------------------------------------------


class TestDropDuplicates:
    """Tests for drop_duplicates() standalone function."""

    def test_removes_duplicate_rows(self):
        df = _df_with_duplicates()
        result, n_dropped = drop_duplicates(df)
        assert n_dropped == 2
        assert len(result) == 3

    def test_no_duplicates_returns_zero_dropped(self):
        df = _df_no_nans()
        result, n_dropped = drop_duplicates(df)
        assert n_dropped == 0
        assert len(result) == 3

    def test_index_is_reset_after_drop(self):
        df = _df_with_duplicates()
        result, _ = drop_duplicates(df, reset_index=True)
        assert list(result.index) == [0, 1, 2]

    def test_index_not_reset_when_disabled(self):
        df = _df_with_duplicates()
        result, _ = drop_duplicates(df, reset_index=False)
        # Duplicates removed; old index is reset to a fresh RangeIndex
        assert isinstance(result.index, pd.RangeIndex)
        assert len(result) == 3

    def test_all_duplicates_keeps_first(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
        result, n_dropped = drop_duplicates(df)
        assert len(result) == 1
        assert n_dropped == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        result, n_dropped = drop_duplicates(df)
        assert len(result) == 0
        assert n_dropped == 0


# ---------------------------------------------------------------------------
# handle_missing_values
# ---------------------------------------------------------------------------


class TestHandleMissingValues:
    """Tests for handle_missing_values() with all strategies."""

    def test_drop_removes_rows_with_nan(self):
        df = _df_with_nans()
        result, dropped_cols = handle_missing_values(df, strategy="drop")
        assert result.isnull().sum().sum() == 0
        assert len(result) == 1  # Only row 0 has no NaN
        assert dropped_cols == {}

    def test_drop_keeps_complete_rows(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        result, _ = handle_missing_values(df, strategy="drop")
        assert len(result) == 2
        assert list(result["a"]) == [1.0, 3.0]

    def test_fill_replaces_nan_with_default(self):
        df = _df_with_nans()
        result, dropped_cols = handle_missing_values(df, strategy="fill", fill_value=0.0)
        assert result.isnull().sum().sum() == 0
        assert len(result) == 5
        assert dropped_cols == {}

    def test_fill_replaces_nan_with_custom_value(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result, _ = handle_missing_values(df, strategy="fill", fill_value=-999.0)
        assert result["a"].iloc[1] == -999.0

    def test_drop_columns_removes_high_null_columns(self):
        # col_a has 60% nulls, which exceeds default threshold of 0.5
        df = pd.DataFrame({
            "col_a": [np.nan, np.nan, np.nan, 4.0, 5.0],
            "col_b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result, dropped_cols = handle_missing_values(df, strategy="drop_columns")
        assert "col_a" in dropped_cols
        assert "col_a" not in result.columns
        assert "col_b" in result.columns

    def test_drop_columns_below_threshold_keeps_column(self):
        # col_a has 20% nulls, below 0.5 threshold
        df = pd.DataFrame({
            "col_a": [1.0, np.nan, 3.0, 4.0, 5.0],
            "col_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result, dropped_cols = handle_missing_values(df, strategy="drop_columns")
        assert "col_a" not in dropped_cols
        assert "col_a" in result.columns

    def test_no_missing_values_returns_unchanged(self):
        df = _df_no_nans()
        result, dropped_cols = handle_missing_values(df, strategy="drop")
        assert result.equals(df)
        assert dropped_cols == {}

    def test_drop_returns_tuple(self):
        df = _df_with_nans()
        result = handle_missing_values(df, strategy="drop")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fill_returns_tuple(self):
        df = _df_with_nans()
        result = handle_missing_values(df, strategy="fill")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_drop_columns_returns_dropped_cols_dict(self):
        df = pd.DataFrame({
            "a": [np.nan, np.nan, np.nan, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        _, dropped_cols = handle_missing_values(df, strategy="drop_columns")
        assert isinstance(dropped_cols, dict)

    def test_drop_preserves_remaining_data(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        })
        result, _ = handle_missing_values(df, strategy="drop")
        assert list(result["b"]) == [10.0, 30.0, 40.0]

    def test_fill_does_not_modify_other_columns(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [10.0, 20.0, 30.0],
        })
        result, _ = handle_missing_values(df, strategy="fill", fill_value=0.0)
        assert list(result["b"]) == [10.0, 20.0, 30.0]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="float64")})
        result, dropped_cols = handle_missing_values(df, strategy="drop")
        assert len(result) == 0
        assert dropped_cols == {}


# ---------------------------------------------------------------------------
# handle_infinite_values
# ---------------------------------------------------------------------------


class TestHandleInfiniteValues:
    """Tests for handle_infinite_values()."""

    def test_replaces_positive_inf(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        result, cols = handle_infinite_values(df)
        assert np.isinf(result["a"]).sum() == 0
        assert "a" in cols

    def test_replaces_negative_inf(self):
        df = pd.DataFrame({"a": [1.0, -np.inf, 3.0]})
        result, cols = handle_infinite_values(df)
        assert np.isinf(result["a"]).sum() == 0
        assert "a" in cols

    def test_no_inf_returns_empty_list(self):
        df = _df_no_nans()
        result, cols = handle_infinite_values(df)
        assert cols == []

    def test_custom_replacement_value(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        result, _ = handle_infinite_values(df, replace_with=-1.0)
        assert result["a"].iloc[1] == -1.0

    def test_non_numeric_columns_ignored(self):
        df = pd.DataFrame({"a": ["foo", "bar"], "b": [np.inf, 1.0]})
        _, cols = handle_infinite_values(df)
        assert "a" not in cols
        assert "b" in cols


# ---------------------------------------------------------------------------
# clip_outliers
# ---------------------------------------------------------------------------


class TestClipOutliers:
    """Tests for clip_outliers()."""

    def test_extreme_values_are_clipped(self):
        df = pd.DataFrame({"a": list(range(100)) + [10000]})
        result, bounds = clip_outliers(df, percentile=99.0)
        assert 10000 not in result["a"].values
        assert "a" in bounds

    def test_no_outliers_no_clipping(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result, bounds = clip_outliers(df, percentile=100.0)
        assert bounds == {}
        assert list(result["a"]) == [1.0, 2.0, 3.0]

    def test_specific_columns_only(self):
        df = pd.DataFrame({
            "a": list(range(100)) + [10000],
            "b": list(range(100)) + [10000],
        })
        result, bounds = clip_outliers(df, percentile=99.0, columns=["a"])
        assert "a" in bounds
        assert "b" not in bounds

    def test_bounds_are_tuple_of_floats(self):
        df = pd.DataFrame({"a": list(range(100)) + [10000]})
        _, bounds = clip_outliers(df, percentile=99.0)
        if bounds:
            lower, upper = bounds["a"]
            assert isinstance(lower, float)
            assert isinstance(upper, float)


# ---------------------------------------------------------------------------
# clean_dataset (full pipeline)
# ---------------------------------------------------------------------------


class TestCleanDataset:
    """Tests for the clean_dataset() full-pipeline function."""

    def test_returns_tuple_of_dataframe_and_result(self):
        df = _df_no_nans()
        result_df, result = clean_dataset(df)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(result, CleaningResult)

    def test_cleaning_result_tracks_original_rows(self):
        df = _df_no_nans()
        _, result = clean_dataset(df)
        assert result.original_rows == 3

    def test_cleaning_result_tracks_rows_after(self):
        df = _df_with_nans()
        _, result = clean_dataset(df, handle_missing="drop")
        assert result.rows_after == len(df.dropna())

    def test_cleaning_result_tracks_duplicates_dropped(self):
        df = _df_with_duplicates()
        _, result = clean_dataset(df, drop_dupes=True)
        assert result.stats["duplicates_dropped"] == 2

    def test_no_duplicates_no_drop(self):
        df = _df_no_nans()
        _, result = clean_dataset(df, drop_dupes=True)
        assert result.stats["duplicates_dropped"] == 0

    def test_infinite_values_handled(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, 6.0]})
        result_df, result = clean_dataset(df, handle_inf=True)
        assert np.isinf(result_df["a"]).sum() == 0
        assert "a" in result.stats["infinite_columns"]

    def test_infinite_values_not_handled_when_disabled(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        result_df, _ = clean_dataset(df, handle_inf=False)
        assert np.isinf(result_df["a"]).sum() == 1

    def test_missing_values_fill_strategy(self):
        df = _df_with_nans()
        result_df, result = clean_dataset(df, handle_missing="fill")
        assert result_df.isnull().sum().sum() == 0
        assert len(result_df) == 5

    def test_missing_values_drop_strategy(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        result_df, _ = clean_dataset(df, handle_missing="drop")
        assert len(result_df) == 2

    def test_clip_enabled(self):
        df = pd.DataFrame({"a": list(range(100)) + [10000]})
        result_df, result = clean_dataset(df, clip=True, clip_percentile=99.0)
        assert "clip_bounds" in result.stats

    def test_clip_disabled(self):
        df = pd.DataFrame({"a": list(range(100)) + [10000]})
        result_df, result = clean_dataset(df, clip=False)
        assert "clip_bounds" not in result.stats

    def test_operations_applied_list(self):
        df = _df_with_duplicates()
        _, result = clean_dataset(df, drop_dupes=True, handle_inf=True)
        assert isinstance(result.operations_applied, list)

    def test_rows_dropped_matches_difference(self):
        df = _df_with_duplicates()
        _, result = clean_dataset(df, drop_dupes=True, handle_missing="drop")
        assert result.rows_dropped == result.original_rows - result.rows_after

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series(dtype="float64")})
        result_df, result = clean_dataset(df)
        assert len(result_df) == 0
        assert result.original_rows == 0
        assert result.rows_after == 0


# ---------------------------------------------------------------------------
# CleaningResult
# ---------------------------------------------------------------------------


class TestCleaningResult:
    """Tests for the CleaningResult dataclass."""

    def test_default_values(self):
        result = CleaningResult()
        assert result.original_rows == 0
        assert result.rows_after == 0
        assert result.rows_dropped == 0
        assert result.columns_cleaned == []
        assert result.operations_applied == []
        assert result.stats == {}

    def test_can_set_fields(self):
        result = CleaningResult(
            original_rows=100,
            rows_after=80,
            rows_dropped=20,
            columns_cleaned=["a", "b"],
            operations_applied=["duplicates(10)", "nulls(10)"],
            stats={"duplicates_dropped": 10},
        )
        assert result.original_rows == 100
        assert result.rows_after == 80
        assert result.rows_dropped == 20
        assert len(result.columns_cleaned) == 2
        assert len(result.operations_applied) == 2
        assert result.stats["duplicates_dropped"] == 10
