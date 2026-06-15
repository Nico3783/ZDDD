"""Unit tests for src/preprocessing/scaler.py.

Covers: fit_scaler, transform_features, fit_transform_features,
        inverse_transform_features, NaN/Inf handling, non-numeric preservation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.core.exceptions import DataError
from src.preprocessing.scaler import (
    fit_scaler,
    fit_transform_features,
    inverse_transform_features,
    transform_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_numeric_df(rows: int = 20, cols: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        rng.rand(rows, cols) * 100,
        columns=[f"f{i}" for i in range(cols)],
    )


# ---------------------------------------------------------------------------
# fit_scaler
# ---------------------------------------------------------------------------

class TestFitScaler:
    """Tests for the fit_scaler function."""

    def test_standard_returns_scaler_and_names(self):
        df = _make_numeric_df()
        scaler, names = fit_scaler(df, method="standard")
        assert isinstance(scaler, StandardScaler)
        assert names == list(df.columns)

    def test_minmax_returns_minmax_scaler(self):
        df = _make_numeric_df()
        scaler, _ = fit_scaler(df, method="minmax")
        assert isinstance(scaler, MinMaxScaler)

    def test_robust_returns_robust_scaler(self):
        df = _make_numeric_df()
        scaler, _ = fit_scaler(df, method="robust")
        assert isinstance(scaler, RobustScaler)

    def test_raises_on_unknown_method(self):
        df = _make_numeric_df()
        with pytest.raises(DataError, match="Unknown scaling method"):
            fit_scaler(df, method="unknown_method")

    def test_feature_names_preserve_column_order(self):
        df = _make_numeric_df()
        df = df[["f3", "f0", "f1", "f2"]]  # reorder
        _, names = fit_scaler(df, method="standard")
        assert names == ["f3", "f0", "f1", "f2"]

    def test_handles_inf_values_during_fit(self):
        df = _make_numeric_df()
        df.iloc[0, 0] = np.inf
        df.iloc[1, 1] = -np.inf
        scaler, _ = fit_scaler(df, method="standard")
        assert hasattr(scaler, "mean_")

    def test_handles_nan_values_during_fit(self):
        df = _make_numeric_df()
        df.iloc[0, 0] = np.nan
        scaler, _ = fit_scaler(df, method="standard")
        assert hasattr(scaler, "mean_")


# ---------------------------------------------------------------------------
# transform_features
# ---------------------------------------------------------------------------

class TestTransformFeatures:
    """Tests for the transform_features function."""

    def test_transform_after_fit(self):
        df = _make_numeric_df()
        scaler, _ = fit_scaler(df, method="standard")
        scaled = transform_features(df, scaler)
        assert scaled.shape == df.shape
        assert list(scaled.columns) == list(df.columns)

    def test_standardized_has_zero_mean(self):
        df = _make_numeric_df(rows=100)
        scaler, _ = fit_scaler(df, method="standard")
        scaled = transform_features(df, scaler)
        means = scaled.mean()
        np.testing.assert_allclose(means, 0.0, atol=1e-10)

    def test_standardized_has_unit_variance(self):
        df = _make_numeric_df(rows=100)
        scaler, _ = fit_scaler(df, method="standard")
        scaled = transform_features(df, scaler)
        stds = scaled.std(ddof=0)
        np.testing.assert_allclose(stds, 1.0, atol=1e-10)

    def test_minmax_within_range(self):
        df = _make_numeric_df(rows=100)
        scaler, _ = fit_scaler(df, method="minmax")
        scaled = transform_features(df, scaler)
        assert scaled.min().min() >= 0.0 - 1e-10
        assert scaled.max().max() <= 1.0 + 1e-10

    def test_index_preserved(self):
        df = _make_numeric_df()
        df.index = [f"row_{i}" for i in range(len(df))]
        scaler, _ = fit_scaler(df, method="standard")
        scaled = transform_features(df, scaler)
        assert list(scaled.index) == list(df.index)

    def test_inf_values_replaced_before_transform(self):
        df = _make_numeric_df()
        df.iloc[0, 0] = np.inf
        scaler, _ = fit_scaler(df, method="standard")
        scaled = transform_features(df, scaler)
        assert not np.isinf(scaled.values).any()

    def test_nan_values_replaced_before_transform(self):
        df = _make_numeric_df()
        df.iloc[0, 0] = np.nan
        scaler, _ = fit_scaler(df, method="standard")
        scaled = transform_features(df, scaler)
        assert not np.isnan(scaled.values).any()


# ---------------------------------------------------------------------------
# fit_transform_features
# ---------------------------------------------------------------------------

class TestFitTransformFeatures:
    """Tests for the fit_transform_features function."""

    def test_returns_triple(self):
        df = _make_numeric_df()
        scaled, scaler, names = fit_transform_features(df, method="standard")
        assert isinstance(scaled, pd.DataFrame)
        assert isinstance(scaler, StandardScaler)
        assert names == list(df.columns)

    def test_matches_separate_fit_then_transform(self):
        df = _make_numeric_df()
        scaled_combined, _, _ = fit_transform_features(df, method="standard")
        scaler, _ = fit_scaler(df, method="standard")
        scaled_separate = transform_features(df, scaler)
        pd.testing.assert_frame_equal(scaled_combined, scaled_separate)

    def test_minmax_fit_transform(self):
        df = _make_numeric_df()
        scaled, scaler, _ = fit_transform_features(df, method="minmax")
        assert isinstance(scaler, MinMaxScaler)
        assert scaled.min().min() >= 0.0 - 1e-10
        assert scaled.max().max() <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# inverse_transform_features
# ---------------------------------------------------------------------------

class TestInverseTransformFeatures:
    """Tests for the inverse_transform_features function."""

    def test_inverse_reverses_standard(self):
        df = _make_numeric_df()
        scaled, scaler, _ = fit_transform_features(df, method="standard")
        recovered = inverse_transform_features(scaled, scaler)
        np.testing.assert_allclose(recovered.values, df.values, atol=1e-10)

    def test_inverse_reverses_minmax(self):
        df = _make_numeric_df()
        scaled, scaler, _ = fit_transform_features(df, method="minmax")
        recovered = inverse_transform_features(scaled, scaler)
        np.testing.assert_allclose(recovered.values, df.values, atol=1e-10)

    def test_inverse_preserves_columns(self):
        df = _make_numeric_df()
        scaled, scaler, _ = fit_transform_features(df, method="standard")
        recovered = inverse_transform_features(scaled, scaler)
        assert list(recovered.columns) == list(df.columns)

    def test_inverse_preserves_index(self):
        df = _make_numeric_df()
        df.index = [f"row_{i}" for i in range(len(df))]
        scaled, scaler, _ = fit_transform_features(df, method="standard")
        recovered = inverse_transform_features(scaled, scaler)
        assert list(recovered.index) == list(df.index)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestScalerEdgeCases:
    """Edge-case tests for the scaler module."""

    def test_single_column(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        scaled, scaler, _ = fit_transform_features(df, method="standard")
        recovered = inverse_transform_features(scaled, scaler)
        np.testing.assert_allclose(recovered.values, df.values, atol=1e-10)

    def test_single_row(self):
        df = pd.DataFrame({"a": [10.0], "b": [20.0]})
        scaled, scaler, _ = fit_transform_features(df, method="standard")
        assert scaled.shape == df.shape

    def test_identical_values(self):
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [3.0, 3.0, 3.0]})
        # StandardScaler with identical values produces 0 std; NaN after fill.
        # MinMaxScaler handles identical values by producing all zeros.
        scaled, scaler, _ = fit_transform_features(df, method="minmax")
        assert scaled.min().min() >= 0.0 - 1e-10
        assert scaled.max().max() <= 1.0 + 1e-10

    def test_large_values(self):
        df = pd.DataFrame({"x": [1e9, 2e9, 3e9]})
        scaled, scaler, _ = fit_transform_features(df, method="standard")
        recovered = inverse_transform_features(scaled, scaler)
        np.testing.assert_allclose(recovered.values, df.values, atol=1.0)
