from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.constants import RANDOM_SEED

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """Tree-based feature importance analysis and ranking.

    Supports Gini importance, permutation importance, and SHAP-style
    contribution analysis for Random Forest and Isolation Forest models.
    """

    def __init__(self, random_state: int = RANDOM_SEED) -> None:
        """Initialize the feature importance analyzer.

        Args:
            random_state: Random state for reproducibility.
        """
        self.random_state = random_state
        self._importance_cache: Dict[str, np.ndarray] = {}

    def compute_gini_importance(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Compute Gini (MDI) importance from a tree-based model.

        Args:
            model: Fitted tree-based model with feature_importances_ attribute.
            feature_names: List of feature names.

        Returns:
            DataFrame with columns ['feature', 'importance', 'rank'].
        """
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not have feature_importances_ attribute")

        importances = np.array(model.feature_importances_)

        if len(importances) != len(feature_names):
            raise ValueError(
                f"Importance length ({len(importances)}) != "
                f"feature names length ({len(feature_names)})"
            )

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        self._importance_cache["gini"] = importances
        logger.info(
            "Gini importance computed for %d features, top: %s",
            len(feature_names),
            df.iloc[0]["feature"] if len(df) > 0 else "none",
        )
        return df

    def compute_permutation_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
        scoring: str = "accuracy",
    ) -> pd.DataFrame:
        """Compute permutation importance by feature shuffling.

        Args:
            model: Fitted model with predict() method.
            X: Feature DataFrame.
            y: Target Series.
            n_repeats: Number of permutation repeats.
            scoring: Scoring metric name.

        Returns:
            DataFrame with columns ['feature', 'importance_mean', 'importance_std', 'rank'].
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        scorers = {
            "accuracy": accuracy_score,
            "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        scorer = scorers.get(scoring, accuracy_score)

        baseline_pred = model.predict(X)
        baseline_score = scorer(y, baseline_pred)

        importances = np.zeros((n_repeats, X.shape[1]))

        for repeat in range(n_repeats):
            for col_idx in range(X.shape[1]):
                X_permuted = X.copy()
                X_permuted.iloc[:, col_idx] = np.random.RandomState(
                    self.random_state + repeat
                ).permutation(X_permuted.iloc[:, col_idx].values)

                perm_pred = model.predict(X_permuted)
                perm_score = scorer(y, perm_pred)
                importances[repeat, col_idx] = baseline_score - perm_score

        mean_importance = np.mean(importances, axis=0)
        std_importance = np.std(importances, axis=0)

        df = pd.DataFrame({
            "feature": list(X.columns),
            "importance_mean": mean_importance,
            "importance_std": std_importance,
        })
        df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        self._importance_cache["permutation"] = mean_importance
        logger.info("Permutation importance computed for %d features", X.shape[1])
        return df

    def get_top_features(
        self,
        importance_type: str = "gini",
        top_k: Optional[int] = None,
    ) -> List[str]:
        """Get top feature names by importance.

        Args:
            importance_type: Type of importance ('gini' or 'permutation').
            top_k: Number of top features to return. None for all.

        Returns:
            List of feature names sorted by importance descending.
        """
        if importance_type not in self._importance_cache:
            raise ValueError(f"No {importance_type} importance computed yet")

        importances = self._importance_cache[importance_type]

        indices = np.argsort(importances)[::-1]
        if top_k:
            indices = indices[:top_k]

        return [str(i) for i in indices]

    def compute_feature_correlation_with_target(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Compute correlation of each feature with the target variable.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            DataFrame with columns ['feature', 'correlation', 'abs_correlation', 'rank'].
        """
        correlations = {}
        for col in X.columns:
            try:
                corr = X[col].corr(y)
                correlations[col] = float(corr) if not np.isnan(corr) else 0.0
            except Exception:
                correlations[col] = 0.0

        df = pd.DataFrame({
            "feature": list(correlations.keys()),
            "correlation": list(correlations.values()),
        })
        df["abs_correlation"] = df["correlation"].abs()
        df = df.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        logger.info("Feature-target correlation computed for %d features", len(X.columns))
        return df

    def select_important_features(
        self,
        importance_df: pd.DataFrame,
        threshold: float = 0.01,
        min_features: int = 5,
    ) -> List[str]:
        """Select features above an importance threshold.

        Args:
            importance_df: DataFrame from compute_gini_importance or compute_permutation_importance.
            threshold: Minimum importance value.
            minimum_features: Minimum number of features to select.

        Returns:
            List of selected feature names.
        """
        importance_col = "importance" if "importance" in importance_df.columns else "importance_mean"
        selected = importance_df[importance_df[importance_col] >= threshold]

        if len(selected) < min_features:
            selected = importance_df.head(min_features)

        features = selected["feature"].tolist()
        logger.info(
            "Selected %d features (threshold=%.4f, min=%d)",
            len(features),
            threshold,
            min_features,
        )
        return features
