"""Unit tests for src.anomaly_detection.isolation_forest.IsolationForestModel."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from src.core.exceptions import ModelError


# ---------------------------------------------------------------------------
# Import the class under test
# ---------------------------------------------------------------------------
from src.anomaly_detection.isolation_forest import IsolationForestModel


# Construction
class TestConstruction:
    """Verify the constructor and default state."""

    def test_default_state(self):
        m = IsolationForestModel()
        assert m.model is None
        assert m.threshold == 0.5
        assert m.feature_names == []
        assert m.training_stats == {}


# Training
class TestTrain:
    """Verify train() sets internal state correctly."""

    def test_train_sets_model(
        self,
        synthetic_benign_features: pd.DataFrame,
        mock_config_loader,
    ):
        m = IsolationForestModel()
        m.train(synthetic_benign_features, contamination=0.1, n_estimators=50, n_jobs=1, random_state=42)
        assert m.model is not None
        assert isinstance(m.model, IsolationForest)

    def test_train_sets_feature_names(
        self,
        synthetic_benign_features: pd.DataFrame,
        mock_config_loader,
    ):
        m = IsolationForestModel()
        m.train(synthetic_benign_features, contamination=0.1, n_estimators=50, n_jobs=1, random_state=42)
        assert m.feature_names == list(synthetic_benign_features.columns)

    def test_train_sets_training_stats(
        self,
        synthetic_benign_features: pd.DataFrame,
        mock_config_loader,
    ):
        m = IsolationForestModel()
        m.train(synthetic_benign_features, contamination=0.1, n_estimators=50, n_jobs=1, random_state=42)
        assert m.training_stats["n_samples"] == len(synthetic_benign_features)
        assert m.training_stats["n_features"] == len(synthetic_benign_features.columns)
        assert m.training_stats["contamination"] == 0.1
        assert m.training_stats["n_estimators"] == 50

    def test_train_default_contamination_from_config(
        self,
        synthetic_benign_features: pd.DataFrame,
        mock_config_loader,
    ):
        m = IsolationForestModel()
        m.train(synthetic_benign_features, n_estimators=50, n_jobs=1, random_state=42)
        assert m.training_stats["contamination"] == 0.1  # from sample_models_yaml

    def test_train_raises_on_empty_dataframe(
        self,
        mock_config_loader,
    ):
        m = IsolationForestModel()
        empty_df = pd.DataFrame()
        with pytest.raises(ModelError, match="training failed"):
            m.train(empty_df, contamination=0.1, n_estimators=10, n_jobs=1, random_state=42)


# Predict
class TestPredict:
    """Verify predict() returns correct shapes and values."""

    def test_predict_output_length(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        preds = trained_isolation_forest.predict(synthetic_attack_features)
        assert len(preds) == len(synthetic_attack_features)

    def test_predict_values_are_minus1_or_1(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        preds = trained_isolation_forest.predict(synthetic_attack_features)
        unique = set(preds)
        assert unique <= {-1, 1}

    def test_predict_is_ndarray(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        preds = trained_isolation_forest.predict(synthetic_attack_features)
        assert isinstance(preds, np.ndarray)

    def test_predict_raises_when_untrained(self):
        m = IsolationForestModel()
        fake_df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ModelError, match="not been trained"):
            m.predict(fake_df)


# Score Samples
class TestScoreSamples:
    """Verify score_samples() returns float arrays."""

    def test_score_samples_output_length(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        scores = trained_isolation_forest.score_samples(synthetic_attack_features)
        assert len(scores) == len(synthetic_attack_features)

    def test_score_samples_are_floats(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        scores = trained_isolation_forest.score_samples(synthetic_attack_features)
        assert scores.dtype.kind == "f"

    def test_score_samples_is_ndarray(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        scores = trained_isolation_forest.score_samples(synthetic_attack_features)
        assert isinstance(scores, np.ndarray)

    def test_score_samples_raises_when_untrained(self):
        m = IsolationForestModel()
        fake_df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ModelError, match="not been trained"):
            m.score_samples(fake_df)


# Decision Function
class TestDecisionFunction:
    """Verify decision_function() returns shifted scores."""

    def test_decision_function_output_length(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        result = trained_isolation_forest.decision_function(synthetic_attack_features)
        assert len(result) == len(synthetic_attack_features)

    def test_decision_function_is_float_array(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        result = trained_isolation_forest.decision_function(synthetic_attack_features)
        assert result.dtype.kind == "f"


# Predict Binary
class TestPredictBinary:
    """Verify predict_binary() returns 0/1 predictions."""

    def test_predict_binary_output_length(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        result = trained_isolation_forest.predict_binary(synthetic_attack_features)
        assert len(result) == len(synthetic_attack_features)

    def test_predict_binary_values_are_0_or_1(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        result = trained_isolation_forest.predict_binary(synthetic_attack_features)
        unique = set(result)
        assert unique <= {0, 1}

    def test_predict_binary_custom_threshold(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        # Very low threshold means most samples classified as normal (0)
        result = trained_isolation_forest.predict_binary(synthetic_attack_features, threshold=0.01)
        assert len(result) == len(synthetic_attack_features)


# Compute Anomaly Scores
class TestComputeAnomalyScores:
    """Verify compute_anomaly_scores() returns normalized 0-1 scores."""

    def test_returns_series(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        result = trained_isolation_forest.compute_anomaly_scores(synthetic_attack_features)
        assert isinstance(result, pd.Series)

    def test_values_in_zero_one_range(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        result = trained_isolation_forest.compute_anomaly_scores(synthetic_attack_features)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_index_matches_input(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
    ):
        result = trained_isolation_forest.compute_anomaly_scores(synthetic_attack_features)
        pd.testing.assert_index_equal(result.index, synthetic_attack_features.index)


# Threshold
class TestThreshold:
    """Verify threshold property and set_threshold()."""

    def test_default_threshold(self):
        m = IsolationForestModel()
        assert m.threshold == 0.5

    def test_set_threshold(self, mock_config_loader):
        m = IsolationForestModel()
        m.set_threshold(0.75)
        assert m.threshold == 0.75

    def test_set_threshold_validates_range(self, mock_config_loader):
        m = IsolationForestModel()
        with pytest.raises(ModelError, match="outside valid range"):
            m.set_threshold(2.0)

    def test_set_threshold_lower_bound(self, mock_config_loader):
        m = IsolationForestModel()
        m.set_threshold(0.0)
        assert m.threshold == 0.0


# Feature Names
class TestFeatureNames:
    """Verify feature_names attribute after training."""

    def test_feature_names_matches_columns(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_benign_features: pd.DataFrame,
    ):
        assert trained_isolation_forest.feature_names == list(synthetic_benign_features.columns)

    def test_feature_names_is_list(
        self,
        trained_isolation_forest: IsolationForestModel,
    ):
        assert isinstance(trained_isolation_forest.feature_names, list)

    def test_feature_names_count_matches_columns(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_benign_features: pd.DataFrame,
    ):
        assert len(trained_isolation_forest.feature_names) == synthetic_benign_features.shape[1]


# Save / Load
class TestSaveLoad:
    """Verify save persists and load restores the model."""

    def test_save_creates_model_file(
        self,
        trained_isolation_forest: IsolationForestModel,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_iforest.joblib")
        metadata_path = str(tmp_model_dir / "test_iforest_meta.joblib")
        trained_isolation_forest.save(model_path=model_path, metadata_path=metadata_path)
        assert Path(model_path).exists()
        assert Path(metadata_path).exists()

    def test_save_then_load_restores_model(
        self,
        trained_isolation_forest: IsolationForestModel,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_iforest.joblib")
        metadata_path = str(tmp_model_dir / "test_iforest_meta.joblib")
        trained_isolation_forest.save(model_path=model_path, metadata_path=metadata_path)

        loaded = IsolationForestModel.load(model_path=model_path, metadata_path=metadata_path)
        assert loaded.model is not None
        assert isinstance(loaded.model, IsolationForest)

    def test_loaded_threshold_matches(
        self,
        trained_isolation_forest: IsolationForestModel,
        tmp_model_dir: Path,
    ):
        trained_isolation_forest.threshold = 0.65
        model_path = str(tmp_model_dir / "test_iforest.joblib")
        metadata_path = str(tmp_model_dir / "test_iforest_meta.joblib")
        trained_isolation_forest.save(model_path=model_path, metadata_path=metadata_path)

        loaded = IsolationForestModel.load(model_path=model_path, metadata_path=metadata_path)
        assert loaded.threshold == 0.65

    def test_loaded_feature_names_match(
        self,
        trained_isolation_forest: IsolationForestModel,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_iforest.joblib")
        metadata_path = str(tmp_model_dir / "test_iforest_meta.joblib")
        trained_isolation_forest.save(model_path=model_path, metadata_path=metadata_path)

        loaded = IsolationForestModel.load(model_path=model_path, metadata_path=metadata_path)
        assert loaded.feature_names == trained_isolation_forest.feature_names

    def test_loaded_training_stats_match(
        self,
        trained_isolation_forest: IsolationForestModel,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_iforest.joblib")
        metadata_path = str(tmp_model_dir / "test_iforest_meta.joblib")
        trained_isolation_forest.save(model_path=model_path, metadata_path=metadata_path)

        loaded = IsolationForestModel.load(model_path=model_path, metadata_path=metadata_path)
        assert loaded.training_stats == trained_isolation_forest.training_stats

    def test_loaded_model_predicts(
        self,
        trained_isolation_forest: IsolationForestModel,
        synthetic_attack_features: pd.DataFrame,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_iforest.joblib")
        metadata_path = str(tmp_model_dir / "test_iforest_meta.joblib")
        trained_isolation_forest.save(model_path=model_path, metadata_path=metadata_path)

        loaded = IsolationForestModel.load(model_path=model_path, metadata_path=metadata_path)
        preds = loaded.predict(synthetic_attack_features)
        assert len(preds) == len(synthetic_attack_features)
        assert set(preds) <= {-1, 1}

    def test_load_nonexistent_model_raises(self):
        with pytest.raises(ModelError, match="not found"):
            IsolationForestModel.load(
                model_path="/tmp/nonexistent_iforest.joblib",
                metadata_path="/tmp/nonexistent_iforest_meta.joblib",
            )

    def test_load_nonexistent_metadata_raises(
        self,
        trained_isolation_forest: IsolationForestModel,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_iforest.joblib")
        metadata_path = str(tmp_model_dir / "test_iforest_meta.joblib")
        trained_isolation_forest.save(model_path=model_path, metadata_path=metadata_path)

        with pytest.raises(ModelError, match="not found"):
            IsolationForestModel.load(
                model_path=model_path,
                metadata_path=str(tmp_model_dir / "does_not_exist_meta.joblib"),
            )
