"""Unit tests for src.classification.random_forest.RandomForestClassifierModel."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.core.exceptions import ModelError


# ---------------------------------------------------------------------------
# Import the class under test
# ---------------------------------------------------------------------------
from src.classification.random_forest import RandomForestClassifierModel


# Construction
class TestConstruction:
    """Verify the constructor and default state."""

    def test_default_state(self):
        m = RandomForestClassifierModel()
        assert m.model is None
        assert m.label_encoder is None
        assert m.feature_names == []
        assert m.class_names == []
        assert m.training_stats == {}


# Training
class TestTrain:
    """Verify train() sets internal state correctly."""

    def test_train_sets_model(
        self,
        synthetic_flow_features: pd.DataFrame,
        synthetic_labels: pd.Series,
        mock_config_loader,
    ):
        m = RandomForestClassifierModel()
        m.train(synthetic_flow_features, synthetic_labels, n_estimators=50, n_jobs=1, random_state=42)
        assert m.model is not None
        assert isinstance(m.model, RandomForestClassifier)

    def test_train_sets_label_encoder(
        self,
        synthetic_flow_features: pd.DataFrame,
        synthetic_labels: pd.Series,
        mock_config_loader,
    ):
        m = RandomForestClassifierModel()
        m.train(synthetic_flow_features, synthetic_labels, n_estimators=50, n_jobs=1, random_state=42)
        assert m.label_encoder is not None

    def test_train_sets_feature_names(
        self,
        synthetic_flow_features: pd.DataFrame,
        synthetic_labels: pd.Series,
        mock_config_loader,
    ):
        m = RandomForestClassifierModel()
        m.train(synthetic_flow_features, synthetic_labels, n_estimators=50, n_jobs=1, random_state=42)
        assert m.feature_names == list(synthetic_flow_features.columns)

    def test_train_sets_class_names(
        self,
        synthetic_flow_features: pd.DataFrame,
        synthetic_labels: pd.Series,
        mock_config_loader,
    ):
        m = RandomForestClassifierModel()
        m.train(synthetic_flow_features, synthetic_labels, n_estimators=50, n_jobs=1, random_state=42)
        expected_classes = sorted(synthetic_labels.unique())
        assert sorted(m.class_names) == sorted(expected_classes)

    def test_train_sets_training_stats(
        self,
        synthetic_flow_features: pd.DataFrame,
        synthetic_labels: pd.Series,
        mock_config_loader,
    ):
        m = RandomForestClassifierModel()
        m.train(synthetic_flow_features, synthetic_labels, n_estimators=50, n_jobs=1, random_state=42)
        assert m.training_stats["n_samples"] == len(synthetic_flow_features)
        assert m.training_stats["n_features"] == synthetic_flow_features.shape[1]
        assert m.training_stats["n_classes"] == len(synthetic_labels.unique())
        assert m.training_stats["n_estimators"] == 50

    def test_train_default_params_from_config(
        self,
        synthetic_flow_features: pd.DataFrame,
        synthetic_labels: pd.Series,
        mock_config_loader,
    ):
        m = RandomForestClassifierModel()
        m.train(synthetic_flow_features, synthetic_labels, n_jobs=1)
        assert m.training_stats["n_estimators"] == 100  # from sample_models_yaml

    def test_train_raises_on_empty_dataframe(
        self,
        mock_config_loader,
    ):
        m = RandomForestClassifierModel()
        with pytest.raises(ModelError, match="training failed"):
            m.train(pd.DataFrame(), pd.Series(dtype=str), n_estimators=10, n_jobs=1, random_state=42)


# Predict
class TestPredict:
    """Verify predict() returns correct shapes and values."""

    def test_predict_output_length(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        preds = trained_random_forest.predict(synthetic_flow_features)
        assert len(preds) == len(synthetic_flow_features)

    def test_predict_returns_string_labels(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        preds = trained_random_forest.predict(synthetic_flow_features)
        assert preds.dtype.kind == "U" or preds.dtype == object

    def test_predict_values_are_from_training_classes(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        preds = trained_random_forest.predict(synthetic_flow_features)
        unique_preds = set(preds)
        assert unique_preds <= set(trained_random_forest.class_names)

    def test_predict_is_ndarray(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        preds = trained_random_forest.predict(synthetic_flow_features)
        assert isinstance(preds, np.ndarray)

    def test_predict_raises_when_untrained(self):
        m = RandomForestClassifierModel()
        fake_df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ModelError, match="not been trained"):
            m.predict(fake_df)

    def test_predict_single_sample(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        single = synthetic_flow_features.iloc[:1]
        preds = trained_random_forest.predict(single)
        assert len(preds) == 1
        assert preds[0] in trained_random_forest.class_names


# Predict Proba
class TestPredictProba:
    """Verify predict_proba() returns valid probability arrays."""

    def test_predict_proba_shape(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        proba = trained_random_forest.predict_proba(synthetic_flow_features)
        n_classes = len(trained_random_forest.class_names)
        assert proba.shape == (len(synthetic_flow_features), n_classes)

    def test_predict_proba_is_2d(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        proba = trained_random_forest.predict_proba(synthetic_flow_features)
        assert proba.ndim == 2

    def test_predict_proba_rows_sum_to_one(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        proba = trained_random_forest.predict_proba(synthetic_flow_features)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_values_between_0_and_1(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        proba = trained_random_forest.predict_proba(synthetic_flow_features)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predict_proba_is_ndarray(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        proba = trained_random_forest.predict_proba(synthetic_flow_features)
        assert isinstance(proba, np.ndarray)

    def test_predict_proba_raises_when_untrained(self):
        m = RandomForestClassifierModel()
        fake_df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ModelError, match="not been trained"):
            m.predict_proba(fake_df)

    def test_predict_proba_single_sample(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        single = synthetic_flow_features.iloc[:1]
        proba = trained_random_forest.predict_proba(single)
        n_classes = len(trained_random_forest.class_names)
        assert proba.shape == (1, n_classes)


# Predict Encoded
class TestPredictEncoded:
    """Verify predict_encoded() returns integer class indices."""

    def test_predict_encoded_returns_integers(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        encoded = trained_random_forest.predict_encoded(synthetic_flow_features)
        assert np.issubdtype(encoded.dtype, np.integer)

    def test_predict_encoded_range(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        encoded = trained_random_forest.predict_encoded(synthetic_flow_features)
        n_classes = len(trained_random_forest.class_names)
        assert encoded.min() >= 0
        assert encoded.max() < n_classes


# Score
class TestScore:
    """Verify score() returns accuracy between 0 and 1."""

    def test_score_returns_float(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
        synthetic_labels: pd.Series,
    ):
        result = trained_random_forest.score(synthetic_flow_features, synthetic_labels)
        assert isinstance(result, float)

    def test_score_in_valid_range(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
        synthetic_labels: pd.Series,
    ):
        result = trained_random_forest.score(synthetic_flow_features, synthetic_labels)
        assert 0.0 <= result <= 1.0


# Top Predictions
class TestGetTopPredictions:
    """Verify get_top_predictions() returns structured results."""

    def test_returns_list_of_lists(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        results = trained_random_forest.get_top_predictions(synthetic_flow_features.iloc[:5])
        assert isinstance(results, list)
        assert len(results) == 5
        for sample in results:
            assert isinstance(sample, list)

    def test_top_k_entries_per_sample(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        results = trained_random_forest.get_top_predictions(synthetic_flow_features.iloc[:3], top_k=2)
        for sample in results:
            assert len(sample) == 2

    def test_prediction_dicts_have_label_and_probability(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        results = trained_random_forest.get_top_predictions(synthetic_flow_features.iloc[:2], top_k=1)
        for sample in results:
            assert len(sample) == 1
            entry = sample[0]
            assert "label" in entry
            assert "probability" in entry
            assert entry["label"] in trained_random_forest.class_names
            assert 0.0 <= entry["probability"] <= 1.0


# Classes / Class Names
class TestClassNames:
    """Verify class_names attribute after training."""

    def test_class_names_are_set(
        self,
        trained_random_forest: RandomForestClassifierModel,
    ):
        assert len(trained_random_forest.class_names) > 0

    def test_class_names_match_unique_labels(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_labels: pd.Series,
    ):
        assert sorted(trained_random_forest.class_names) == sorted(synthetic_labels.unique())

    def test_class_names_is_list(
        self,
        trained_random_forest: RandomForestClassifierModel,
    ):
        assert isinstance(trained_random_forest.class_names, list)


# Feature Names
class TestFeatureNames:
    """Verify feature_names attribute after training."""

    def test_feature_names_matches_columns(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        assert trained_random_forest.feature_names == list(synthetic_flow_features.columns)

    def test_feature_names_is_list(
        self,
        trained_random_forest: RandomForestClassifierModel,
    ):
        assert isinstance(trained_random_forest.feature_names, list)

    def test_feature_names_count_matches_columns(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
    ):
        assert len(trained_random_forest.feature_names) == synthetic_flow_features.shape[1]


# Save / Load
class TestSaveLoad:
    """Verify save persists and load restores the model."""

    def test_save_creates_files(
        self,
        trained_random_forest: RandomForestClassifierModel,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_rf.joblib")
        metadata_path = str(tmp_model_dir / "test_rf_meta.joblib")
        trained_random_forest.save(model_path=model_path, metadata_path=metadata_path)
        assert Path(model_path).exists()
        assert Path(metadata_path).exists()

    def test_save_then_load_restores_model(
        self,
        trained_random_forest: RandomForestClassifierModel,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_rf.joblib")
        metadata_path = str(tmp_model_dir / "test_rf_meta.joblib")
        trained_random_forest.save(model_path=model_path, metadata_path=metadata_path)

        loaded = RandomForestClassifierModel.load(model_path=model_path, metadata_path=metadata_path)
        assert loaded.model is not None
        assert isinstance(loaded.model, RandomForestClassifier)

    def test_loaded_class_names_match(
        self,
        trained_random_forest: RandomForestClassifierModel,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_rf.joblib")
        metadata_path = str(tmp_model_dir / "test_rf_meta.joblib")
        trained_random_forest.save(model_path=model_path, metadata_path=metadata_path)

        loaded = RandomForestClassifierModel.load(model_path=model_path, metadata_path=metadata_path)
        assert loaded.class_names == trained_random_forest.class_names

    def test_loaded_feature_names_match(
        self,
        trained_random_forest: RandomForestClassifierModel,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_rf.joblib")
        metadata_path = str(tmp_model_dir / "test_rf_meta.joblib")
        trained_random_forest.save(model_path=model_path, metadata_path=metadata_path)

        loaded = RandomForestClassifierModel.load(model_path=model_path, metadata_path=metadata_path)
        assert loaded.feature_names == trained_random_forest.feature_names

    def test_loaded_label_encoder_match(
        self,
        trained_random_forest: RandomForestClassifierModel,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_rf.joblib")
        metadata_path = str(tmp_model_dir / "test_rf_meta.joblib")
        trained_random_forest.save(model_path=model_path, metadata_path=metadata_path)

        loaded = RandomForestClassifierModel.load(model_path=model_path, metadata_path=metadata_path)
        assert loaded.label_encoder is not None
        assert list(loaded.label_encoder.classes_) == list(trained_random_forest.label_encoder.classes_)

    def test_loaded_model_predicts(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_rf.joblib")
        metadata_path = str(tmp_model_dir / "test_rf_meta.joblib")
        trained_random_forest.save(model_path=model_path, metadata_path=metadata_path)

        loaded = RandomForestClassifierModel.load(model_path=model_path, metadata_path=metadata_path)
        preds = loaded.predict(synthetic_flow_features)
        assert len(preds) == len(synthetic_flow_features)
        assert set(preds) <= set(trained_random_forest.class_names)

    def test_loaded_model_predict_proba(
        self,
        trained_random_forest: RandomForestClassifierModel,
        synthetic_flow_features: pd.DataFrame,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_rf.joblib")
        metadata_path = str(tmp_model_dir / "test_rf_meta.joblib")
        trained_random_forest.save(model_path=model_path, metadata_path=metadata_path)

        loaded = RandomForestClassifierModel.load(model_path=model_path, metadata_path=metadata_path)
        proba = loaded.predict_proba(synthetic_flow_features)
        n_classes = len(loaded.class_names)
        assert proba.shape == (len(synthetic_flow_features), n_classes)

    def test_load_nonexistent_model_raises(self):
        with pytest.raises(ModelError, match="not found"):
            RandomForestClassifierModel.load(
                model_path="/tmp/nonexistent_rf.joblib",
                metadata_path="/tmp/nonexistent_rf_meta.joblib",
            )

    def test_load_nonexistent_metadata_raises(
        self,
        trained_random_forest: RandomForestClassifierModel,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_rf.joblib")
        metadata_path = str(tmp_model_dir / "test_rf_meta.joblib")
        trained_random_forest.save(model_path=model_path, metadata_path=metadata_path)

        with pytest.raises(ModelError, match="not found"):
            RandomForestClassifierModel.load(
                model_path=model_path,
                metadata_path=str(tmp_model_dir / "does_not_exist_meta.joblib"),
            )

    def test_loaded_training_stats_match(
        self,
        trained_random_forest: RandomForestClassifierModel,
        tmp_model_dir: Path,
    ):
        model_path = str(tmp_model_dir / "test_rf.joblib")
        metadata_path = str(tmp_model_dir / "test_rf_meta.joblib")
        trained_random_forest.save(model_path=model_path, metadata_path=metadata_path)

        loaded = RandomForestClassifierModel.load(model_path=model_path, metadata_path=metadata_path)
        assert loaded.training_stats["n_samples"] == trained_random_forest.training_stats["n_samples"]
        assert loaded.training_stats["n_features"] == trained_random_forest.training_stats["n_features"]
        assert loaded.training_stats["n_classes"] == trained_random_forest.training_stats["n_classes"]
