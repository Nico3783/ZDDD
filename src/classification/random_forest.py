from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.core.config import get_config
from src.core.constants import MODELS_DIR
from src.core.exceptions import ModelError
from src.utils.model_utils import load_model, save_model

logger = logging.getLogger(__name__)


class RandomForestClassifierModel:
    """Wrapper around sklearn RandomForestClassifier with project-specific defaults."""

    model: Optional[RandomForestClassifier] = None
    label_encoder: Optional[LabelEncoder] = None
    feature_names: List[str] = []
    class_names: List[str] = []
    training_stats: Dict[str, Any] = {}

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        n_estimators: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
        max_features: Optional[str | float] = None,
        bootstrap: Optional[bool] = None,
        oob_score: Optional[bool] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        class_weight: Optional[str] = None,
        criterion: Optional[str] = None,
    ) -> None:
        """Train the Random Forest classifier.

        Args:
            features: Training features.
            labels: Training labels (multi-class DoS attack labels).
            n_estimators: Number of trees.
            max_depth: Maximum tree depth.
            min_samples_split: Min samples to split a node.
            min_samples_leaf: Min samples in a leaf.
            max_features: Features per split.
            bootstrap: Whether to use bootstrap sampling.
            oob_score: Whether to compute out-of-bag score.
            n_jobs: Number of parallel jobs.
            random_state: Random seed.
            class_weight: Class weight strategy.
            criterion: Split criterion.

        Raises:
            ModelError: If training fails.
        """
        cfg = get_config().load("models")
        rf_cfg = cfg.get("random_forest", {})

        n_estimators = n_estimators if n_estimators is not None else rf_cfg.get("n_estimators", 150)
        max_depth = max_depth if max_depth is not None else rf_cfg.get("max_depth", 20)
        min_samples_split = min_samples_split if min_samples_split is not None else rf_cfg.get("min_samples_split", 5)
        min_samples_leaf = min_samples_leaf if min_samples_leaf is not None else rf_cfg.get("min_samples_leaf", 2)
        max_features = max_features if max_features is not None else rf_cfg.get("max_features", "sqrt")
        bootstrap = bootstrap if bootstrap is not None else rf_cfg.get("bootstrap", True)
        oob_score = oob_score if oob_score is not None else rf_cfg.get("oob_score", True)
        n_jobs = n_jobs if n_jobs is not None else rf_cfg.get("n_jobs", -1)
        random_state = random_state if random_state is not None else rf_cfg.get("random_state", 42)
        class_weight = class_weight if class_weight is not None else rf_cfg.get("class_weight", "balanced")
        criterion = criterion if criterion is not None else rf_cfg.get("criterion", "gini")

        try:
            # Encode labels
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            self.class_names = list(self.label_encoder.classes_)

            # Train
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                class_weight=class_weight,
                criterion=criterion,
            )
            self.model.fit(features, encoded_labels)
            self.feature_names = list(features.columns)

            self.training_stats = {
                "n_samples": len(features),
                "n_features": len(self.feature_names),
                "n_classes": len(self.class_names),
                "class_names": self.class_names,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "oob_score": float(self.model.oob_score_) if oob_score else None,
            }

            logger.info(
                "Random Forest trained: %d samples, %d features, %d classes, oob_score=%.4f",
                len(features),
                len(self.feature_names),
                len(self.class_names),
                self.model.oob_score_ if oob_score else 0.0,
            )

        except Exception as e:
            raise ModelError(f"Random Forest training failed: {e}") from e

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            features: DataFrame of features.

        Returns:
            Array of predicted class labels (original string labels).

        Raises:
            ModelError: If the model has not been trained.
        """
        self._check_trained()
        if self.feature_names:
            features = features[self.feature_names]
        encoded = self.model.predict(features)  # type: ignore[union-attr]
        return self.label_encoder.inverse_transform(encoded)  # type: ignore[union-attr]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            features: DataFrame of features.

        Returns:
            Array of shape (n_samples, n_classes) with probabilities.

        Raises:
            ModelError: If the model has not been trained.
        """
        self._check_trained()
        if self.feature_names:
            features = features[self.feature_names]
        return self.model.predict_proba(features)  # type: ignore[union-attr]

    def predict_encoded(self, features: pd.DataFrame) -> np.ndarray:
        """Predict encoded class indices.

        Args:
            features: DataFrame of features.

        Returns:
            Array of encoded class indices.

        Raises:
            ModelError: If the model has not been trained.
        """
        self._check_trained()
        return self.model.predict(features)  # type: ignore[union-attr]

    def get_top_predictions(
        self,
        features: pd.DataFrame,
        top_k: int = 3,
    ) -> List[List[Dict[str, Any]]]:
        """Get top-k predictions with probabilities for each sample.

        Args:
            features: DataFrame of features.
            top_k: Number of top predictions per sample.

        Returns:
            List of lists, each containing top-k prediction dicts.
        """
        probabilities = self.predict_proba(features)
        results = []

        for prob_row in probabilities:
            top_indices = np.argsort(prob_row)[::-1][:top_k]
            top_preds = []
            for idx in top_indices:
                top_preds.append({
                    "label": self.class_names[idx],
                    "probability": float(prob_row[idx]),
                })
            results.append(top_preds)

        return results

    def score(self, features: pd.DataFrame, labels: pd.Series) -> float:
        """Compute accuracy score.

        Args:
            features: Test features.
            labels: True labels.

        Returns:
            Accuracy score.
        """
        self._check_trained()
        encoded = self.label_encoder.transform(labels)  # type: ignore[union-attr]
        return float(self.model.score(features, encoded))  # type: ignore[union-attr]

    def set_class_weights(self, class_weights: Dict[str, float]) -> None:
        """Set custom class weights and retrain.

        Args:
            class_weights: Dictionary mapping class names to weights.
        """
        self._check_trained()

        # Create weight array in the order of class_names
        weights = np.ones(len(self.class_names))
        for cls_name, weight in class_weights.items():
            if cls_name in self.class_names:
                idx = self.class_names.index(cls_name)
                weights[idx] = weight

        self.model.class_weight = weights  # type: ignore[union-attr]
        logger.info("Class weights updated: %s", class_weights)

    def save(
        self,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> None:
        """Persist the trained model and metadata to disk.

        Args:
            model_path: Path for the model file. If None, uses config default.
            metadata_path: Path for metadata. If None, uses config default.
        """
        self._check_trained()

        if model_path is None:
            model_path = str(MODELS_DIR / "random_forest.joblib")
        if metadata_path is None:
            metadata_path = str(MODELS_DIR / "random_forest_metadata.joblib")

        save_model(self.model, model_path)
        metadata = {
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "training_stats": self.training_stats,
        }
        save_model(metadata, metadata_path)
        logger.info("Random Forest model saved to %s", model_path)

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> RandomForestClassifierModel:
        """Load a persisted model from disk.

        Args:
            model_path: Path to the model file. If None, uses config default.
            metadata_path: Path to metadata. If None, uses config default.

        Returns:
            Loaded RandomForestClassifierModel instance.
        """
        if model_path is None:
            model_path = str(MODELS_DIR / "random_forest.joblib")
        if metadata_path is None:
            metadata_path = str(MODELS_DIR / "random_forest_metadata.joblib")

        model = load_model(model_path)
        metadata = load_model(metadata_path)

        instance = cls()
        instance.model = model
        instance.label_encoder = metadata.get("label_encoder")
        instance.feature_names = metadata.get("feature_names", [])
        instance.class_names = metadata.get("class_names", [])
        instance.training_stats = metadata.get("training_stats", {})

        logger.info("Random Forest model loaded from %s", model_path)
        return instance

    def _check_trained(self) -> None:
        """Raise ModelError if the model has not been trained."""
        if self.model is None:
            raise ModelError("Random Forest model has not been trained. Call train() first.")
        if self.label_encoder is None:
            raise ModelError("Label encoder has not been initialized. Call train() first.")
