from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.core.config import get_config
from src.core.constants import MODELS_DIR
from src.core.exceptions import ModelError
from src.utils.model_utils import load_model, save_model

logger = logging.getLogger(__name__)


@dataclass
class IsolationForestModel:
    """Wrapper around sklearn IsolationForest with project-specific defaults."""

    model: Optional[IsolationForest] = None
    threshold: float = 0.5
    feature_names: List[str] = field(default_factory=list)
    training_stats: Dict[str, Any] = field(default_factory=dict)

    def train(
        self,
        features: pd.DataFrame,
        contamination: Optional[float] = None,
        n_estimators: Optional[int] = None,
        max_samples: Optional[str | float] = None,
        max_features: Optional[float] = None,
        bootstrap: Optional[bool] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Train the Isolation Forest model.

        Args:
            features: Training features (typically benign-only).
            contamination: Expected proportion of anomalies.
            n_estimators: Number of base estimators.
            max_samples: Number of samples for each estimator.
            max_features: Fraction of features per estimator.
            bootstrap: Whether to use bootstrap samples.
            n_jobs: Number of parallel jobs.
            random_state: Random seed.

        Raises:
            ModelError: If training fails.
        """
        cfg = get_config().load("models")
        if_cfg = cfg.get("isolation_forest", {})

        contamination = contamination if contamination is not None else if_cfg.get("contamination", 0.1)
        n_estimators = n_estimators if n_estimators is not None else if_cfg.get("n_estimators", 200)
        max_samples = max_samples if max_samples is not None else if_cfg.get("max_samples", "auto")
        max_features = max_features if max_features is not None else if_cfg.get("max_features", 1.0)
        bootstrap = bootstrap if bootstrap is not None else if_cfg.get("bootstrap", False)
        n_jobs = n_jobs if n_jobs is not None else if_cfg.get("n_jobs", -1)
        random_state = random_state if random_state is not None else if_cfg.get("random_state", 42)

        try:
            self.model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                bootstrap=bootstrap,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            self.model.fit(features)
            self.feature_names = list(features.columns)

            # Store training statistics
            self.training_stats = {
                "n_samples": len(features),
                "n_features": len(self.feature_names),
                "contamination": contamination,
                "n_estimators": n_estimators,
                "max_samples": max_samples,
            }

            logger.info(
                "Isolation Forest trained: %d samples, %d features, contamination=%.3f",
                len(features),
                len(self.feature_names),
                contamination,
            )

        except Exception as e:
            raise ModelError(f"Isolation Forest training failed: {e}") from e

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict anomaly labels (-1 for anomaly, 1 for normal).

        Args:
            features: DataFrame of features to predict.

        Returns:
            Array of predictions (-1 or 1).

        Raises:
            ModelError: If the model has not been trained.
        """
        self._check_trained()
        if self.feature_names:
            features = features[self.feature_names]
        return self.model.predict(features)  # type: ignore[union-attr]

    def score_samples(self, features: pd.DataFrame) -> np.ndarray:
        """Compute anomaly scores for each sample.

        Lower scores indicate more anomalous.

        Args:
            features: DataFrame of features to score.

        Returns:
            Array of anomaly scores (lower = more anomalous).

        Raises:
            ModelError: If the model has not been trained.
        """
        self._check_trained()
        if self.feature_names:
            features = features[self.feature_names]
        return self.model.score_samples(features)  # type: ignore[union-attr]

    def decision_function(self, features: pd.DataFrame) -> np.ndarray:
        """Compute the decision function (shifted anomaly scores).

        The decision function shifts scores so that the threshold is at 0.
        Negative values indicate anomalies.

        Args:
            features: DataFrame of features.

        Returns:
            Array of shifted decision scores.

        Raises:
            ModelError: If the model has not been trained.
        """
        self._check_trained()
        if self.feature_names:
            features = features[self.feature_names]
        return self.model.decision_function(features)  # type: ignore[union-attr]

    def predict_binary(
        self,
        features: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict binary labels using the configured threshold.

        Converts the continuous anomaly score to a binary 0/1 prediction
        where 1 = anomaly.

        Args:
            features: DataFrame of features.
            threshold: Score threshold. If None, uses self.threshold.

        Returns:
            Array of binary predictions (0=normal, 1=anomaly).
        """
        if threshold is None:
            threshold = self.threshold

        scores = self.compute_anomaly_scores(features)
        # Higher normalized score = more anomalous
        # Convert to binary: 1 if score >= threshold, else 0
        predictions = (scores >= threshold).astype(int)
        return predictions

    def compute_anomaly_scores(self, features: pd.DataFrame) -> pd.Series:
        """Compute normalized anomaly scores (0-1, higher = more anomalous).

        Inverts and normalizes the raw scores so that higher values
        indicate greater anomaly likelihood.

        Args:
            features: DataFrame of features.

        Returns:
            Series of normalized anomaly scores between 0 and 1.
        """
        raw_scores = self.score_samples(features)
        # Invert: lower raw score = more anomalous -> higher normalized score
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        score_range = max_score - min_score

        if score_range == 0:
            normalized = np.zeros_like(raw_scores)
        else:
            normalized = 1.0 - (raw_scores - min_score) / score_range

        return pd.Series(normalized, index=features.index, name="anomaly_score")

    def set_threshold(self, threshold: float) -> None:
        """Set the anomaly score threshold.

        Args:
            threshold: New threshold value.

        Raises:
            ModelError: If threshold is outside valid range.
        """
        threshold_cfg = get_config().load("thresholds")
        min_t = threshold_cfg.get("anomaly_detection", {}).get("min_anomaly_score", 0.0)
        max_t = threshold_cfg.get("anomaly_detection", {}).get("max_anomaly_score", 1.0)

        if not (min_t <= threshold <= max_t):
            raise ModelError(
                f"Threshold {threshold} outside valid range [{min_t}, {max_t}]"
            )

        self.threshold = threshold
        logger.info("Anomaly threshold set to %.4f", threshold)

    def save(self, model_path: Optional[str] = None, metadata_path: Optional[str] = None) -> None:
        """Persist the trained model and metadata to disk.

        Args:
            model_path: Path for the model file. If None, uses config default.
            metadata_path: Path for metadata. If None, uses config default.
        """
        self._check_trained()

        if model_path is None:
            model_path = str(MODELS_DIR / "isolation_forest.joblib")
        if metadata_path is None:
            metadata_path = str(MODELS_DIR / "isolation_forest_metadata.joblib")

        save_model(self.model, model_path)
        metadata = {
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "training_stats": self.training_stats,
        }
        save_model(metadata, metadata_path)
        logger.info("Isolation Forest model saved to %s", model_path)

    @classmethod
    def load(cls, model_path: Optional[str] = None, metadata_path: Optional[str] = None) -> IsolationForestModel:
        """Load a persisted model from disk.

        Args:
            model_path: Path to the model file. If None, uses config default.
            metadata_path: Path to metadata. If None, uses config default.

        Returns:
            Loaded IsolationForestModel instance.
        """
        if model_path is None:
            model_path = str(MODELS_DIR / "isolation_forest.joblib")
        if metadata_path is None:
            metadata_path = str(MODELS_DIR / "isolation_forest_metadata.joblib")

        model = load_model(model_path)
        metadata = load_model(metadata_path)

        instance = cls(
            model=model,
            threshold=metadata.get("threshold", 0.5),
            feature_names=metadata.get("feature_names", []),
            training_stats=metadata.get("training_stats", {}),
        )

        logger.info("Isolation Forest model loaded from %s", model_path)
        return instance

    def _check_trained(self) -> None:
        """Raise ModelError if the model has not been trained."""
        if self.model is None:
            raise ModelError("Isolation Forest model has not been trained. Call train() first.")
