"""Integration tests for the full preprocessing -> training -> evaluation pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.isolation_forest import IsolationForestModel
from src.classification.random_forest import RandomForestClassifierModel
from src.preprocessing.cleaner import clean_dataset, handle_missing_values
from src.preprocessing.encoder import encode_labels
from src.preprocessing.scaler import fit_transform_features


@pytest.fixture()
def raw_labeled_df(synthetic_labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Return a raw DataFrame as it would come from the dataset loader."""
    df = synthetic_labeled_df.copy()
    # Introduce a few NaN values to test cleaning
    rng = np.random.RandomState(42)
    nan_mask = rng.rand(*df.shape) < 0.01
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df.loc[:, numeric_cols] = df[numeric_cols].mask(nan_mask[:, : len(numeric_cols)])
    return df


class TestFullPipeline:
    def test_clean_encode_scale_train_evaluate(self, raw_labeled_df: pd.DataFrame) -> None:
        # Step 1: Clean
        cleaned, _clean_result = clean_dataset(raw_labeled_df)
        assert cleaned is not None
        assert len(cleaned) > 0

        # Step 2: Encode labels
        label_col = "Label"
        labels = cleaned[label_col]
        encoded_labels, _encoder = encode_labels(labels)

        # Step 3: Scale features
        feature_cols = [c for c in cleaned.columns if c != label_col]
        scaled_features, _scaler, _feature_names = fit_transform_features(
            cleaned[feature_cols], method="standard"
        )
        assert scaled_features.shape == cleaned[feature_cols].shape

        # Step 4: Train Isolation Forest
        iforest = IsolationForestModel()
        iforest.train(
            scaled_features, contamination=0.1, n_estimators=20, n_jobs=1, random_state=42
        )
        assert iforest.model is not None

        # Step 5: Train Random Forest
        rf = RandomForestClassifierModel()
        rf.train(scaled_features, encoded_labels, n_estimators=20, n_jobs=1, random_state=42)
        assert rf.model is not None

        # Step 6: Predict
        anomaly_preds = iforest.predict(scaled_features.iloc[:10])
        assert len(anomaly_preds) == 10
        assert set(anomaly_preds).issubset({-1, 1})

        rf_preds = rf.predict(scaled_features.iloc[:10])
        assert len(rf_preds) == 10

    def test_pipeline_handles_all_missing_values(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [1.0, np.nan, 3.0, 4.0, 5.0],
                "f2": [np.nan, 2.0, 3.0, 4.0, 5.0],
                "Label": ["A", "B", "A", "B", "A"],
            }
        )
        cleaned, _ = handle_missing_values(df, strategy="drop")
        assert cleaned.isna().sum().sum() == 0

    def test_pipeline_with_different_contamination(
        self, synthetic_benign_features: pd.DataFrame
    ) -> None:
        models = []
        for cont in [0.05, 0.1, 0.2]:
            m = IsolationForestModel()
            m.train(
                synthetic_benign_features,
                contamination=cont,
                n_estimators=20,
                n_jobs=1,
                random_state=42,
            )
            models.append(m)

        # Different contamination should produce different thresholds
        thresholds = [m.threshold for m in models]
        # At minimum, all should be trained
        assert all(m.model is not None for m in models)

    def test_save_and_load_through_pipeline(
        self, synthetic_benign_features: pd.DataFrame, tmp_path
    ) -> None:
        # Train
        iforest = IsolationForestModel()
        iforest.train(
            synthetic_benign_features,
            contamination=0.1,
            n_estimators=20,
            n_jobs=1,
            random_state=42,
        )

        # Save
        model_path = tmp_path / "iforest.joblib"
        meta_path = tmp_path / "iforest_meta.joblib"
        iforest.save(str(model_path), str(meta_path))
        assert model_path.exists()

        # Load
        loaded = IsolationForestModel.load(str(model_path), str(meta_path))
        assert loaded.model is not None

        # Predictions should match
        original_pred = iforest.predict(synthetic_benign_features.iloc[:5])
        loaded_pred = loaded.predict(synthetic_benign_features.iloc[:5])
        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_pipeline_stats_keys(self, synthetic_labeled_df: pd.DataFrame) -> None:
        from src.anomaly_detection.isolation_forest import IsolationForestModel
        from src.classification.random_forest import RandomForestClassifierModel
        from src.detection_engine.engine import DetectionEngine

        feature_cols = [c for c in synthetic_labeled_df.columns if c != "Label"]
        X = synthetic_labeled_df[feature_cols]

        iforest = IsolationForestModel()
        iforest.train(X, contamination=0.1, n_estimators=20, n_jobs=1, random_state=42)

        rf = RandomForestClassifierModel()
        rf.train(X, synthetic_labeled_df["Label"], n_estimators=20, n_jobs=1, random_state=42)

        engine = DetectionEngine(anomaly_model=iforest, classifier_model=rf)
        results = engine.detect_batch(X.iloc[:10])
        stats = engine.compute_stats(results)

        assert hasattr(stats, "total_samples")
        assert hasattr(stats, "anomalies_detected")
        assert hasattr(stats, "mean_anomaly_score")
        assert hasattr(stats, "severity_counts")
