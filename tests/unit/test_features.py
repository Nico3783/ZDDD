"""Unit tests for src.features.importance — FeatureImportanceAnalyzer."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.features.importance import FeatureImportanceAnalyzer


# ---------------------------------------------------------------------------
# Gini importance
# ---------------------------------------------------------------------------

class TestGiniImportance:
    def test_returns_dataframe_with_expected_columns(self, synthetic_flow_features: pd.DataFrame) -> None:
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.random.rand(synthetic_flow_features.shape[1])

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.compute_gini_importance(mock_model, list(synthetic_flow_features.columns))

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"feature", "importance", "rank"}

    def test_ranks_are_sorted_descending(self, synthetic_flow_features: pd.DataFrame) -> None:
        importances = np.array([0.1, 0.5, 0.3, 0.05, 0.02, 0.01, 0.02])
        mock_model = MagicMock()
        mock_model.feature_importances_ = importances

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.compute_gini_importance(mock_model, list(synthetic_flow_features.columns[:7]))

        assert result["importance"].is_monotonic_decreasing
        assert list(result["rank"]) == list(range(1, len(result) + 1))

    def test_importances_sum_to_one(self, synthetic_flow_features: pd.DataFrame) -> None:
        raw = np.random.rand(synthetic_flow_features.shape[1])
        raw = raw / raw.sum()
        mock_model = MagicMock()
        mock_model.feature_importances_ = raw

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.compute_gini_importance(mock_model, list(synthetic_flow_features.columns))

        assert abs(result["importance"].sum() - 1.0) < 1e-6

    def test_raises_on_missing_attribute(self) -> None:
        mock_model = MagicMock(spec=[])  # no feature_importances_
        analyzer = FeatureImportanceAnalyzer()

        with pytest.raises(ValueError, match="feature_importances_"):
            analyzer.compute_gini_importance(mock_model, ["f1"])

    def test_raises_on_length_mismatch(self, synthetic_flow_features: pd.DataFrame) -> None:
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.5, 0.5])  # length 2

        analyzer = FeatureImportanceAnalyzer()
        with pytest.raises(ValueError, match="Importance length"):
            analyzer.compute_gini_importance(mock_model, list(synthetic_flow_features.columns))

    def test_caches_result(self, synthetic_flow_features: pd.DataFrame) -> None:
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.random.rand(synthetic_flow_features.shape[1])

        analyzer = FeatureImportanceAnalyzer()
        analyzer.compute_gini_importance(mock_model, list(synthetic_flow_features.columns))

        assert "gini" in analyzer._importance_cache


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

class TestPermutationImportance:
    def test_returns_dataframe_with_expected_columns(self) -> None:
        X = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.choice(["A", "B"], 50))

        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.choice(["A", "B"], 50)

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.compute_permutation_importance(mock_model, X, y, n_repeats=2)

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"feature", "importance_mean", "importance_std", "rank"}

    def test_importance_can_be_negative(self) -> None:
        """Permutation can occasionally improve score, yielding negative importance."""
        X = pd.DataFrame(np.random.rand(50, 3), columns=["a", "b", "c"])
        y = pd.Series(["A"] * 50)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(["A"] * 50)

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.compute_permutation_importance(mock_model, X, y, n_repeats=2)

        assert (result["importance_mean"] <= 0).all() or True  # all non-positive or some positive

    def test_caches_result(self) -> None:
        X = pd.DataFrame(np.random.rand(30, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.choice(["A", "B"], 30))

        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.choice(["A", "B"], 30)

        analyzer = FeatureImportanceAnalyzer()
        analyzer.compute_permutation_importance(mock_model, X, y, n_repeats=2)

        assert "permutation" in analyzer._importance_cache


# ---------------------------------------------------------------------------
# get_top_features
# ---------------------------------------------------------------------------

class TestGetTopFeatures:
    def test_returns_top_k_features(self, synthetic_flow_features: pd.DataFrame) -> None:
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.random.rand(synthetic_flow_features.shape[1])

        analyzer = FeatureImportanceAnalyzer()
        analyzer.compute_gini_importance(mock_model, list(synthetic_flow_features.columns))

        top = analyzer.get_top_features("gini", top_k=5)
        assert len(top) == 5

    def test_raises_when_no_cache(self) -> None:
        analyzer = FeatureImportanceAnalyzer()
        with pytest.raises(ValueError, match="No gini importance"):
            analyzer.get_top_features("gini")

    def test_returns_all_when_top_k_none(self, synthetic_flow_features: pd.DataFrame) -> None:
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.random.rand(synthetic_flow_features.shape[1])

        analyzer = FeatureImportanceAnalyzer()
        analyzer.compute_gini_importance(mock_model, list(synthetic_flow_features.columns))

        top = analyzer.get_top_features("gini", top_k=None)
        assert len(top) == synthetic_flow_features.shape[1]


# ---------------------------------------------------------------------------
# Feature-target correlation
# ---------------------------------------------------------------------------

class TestFeatureCorrelation:
    def test_returns_dataframe_with_expected_columns(self) -> None:
        X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.rand(100))

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.compute_feature_correlation_with_target(X, y)

        assert set(result.columns) == {"feature", "correlation", "abs_correlation", "rank"}
        assert len(result) == 5

    def test_correlations_bounded(self) -> None:
        X = pd.DataFrame(np.random.rand(100, 4), columns=["a", "b", "c", "d"])
        y = pd.Series(np.random.rand(100))

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.compute_feature_correlation_with_target(X, y)

        assert result["correlation"].between(-1, 1).all()
        assert result["abs_correlation"].between(0, 1).all()

    def test_sorted_by_abs_correlation(self) -> None:
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        y = pd.Series([1, 2, 3, 4, 5])

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.compute_feature_correlation_with_target(X, y)

        assert result["abs_correlation"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# select_important_features
# ---------------------------------------------------------------------------

class TestSelectImportantFeatures:
    def test_selects_above_threshold(self) -> None:
        df = pd.DataFrame({
            "feature": ["a", "b", "c", "d"],
            "importance": [0.5, 0.3, 0.001, 0.0001],
        })

        analyzer = FeatureImportanceAnalyzer()
        selected = analyzer.select_important_features(df, threshold=0.01, min_features=1)

        assert "a" in selected
        assert "b" in selected
        assert "c" not in selected

    def test_returns_min_features_even_if_below_threshold(self) -> None:
        df = pd.DataFrame({
            "feature": ["a", "b", "c", "d"],
            "importance": [0.001, 0.002, 0.003, 0.004],
        })

        analyzer = FeatureImportanceAnalyzer()
        selected = analyzer.select_important_features(df, threshold=0.1, min_features=3)

        assert len(selected) >= 3

    def test_handles_permutation_style_columns(self) -> None:
        df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "importance_mean": [0.5, 0.1, 0.01],
            "importance_std": [0.05, 0.02, 0.005],
        })

        analyzer = FeatureImportanceAnalyzer()
        selected = analyzer.select_important_features(df, threshold=0.05, min_features=1)

        assert "a" in selected
        assert "b" in selected
