from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.core.config import get_config
from src.core.constants import MODELS_DIR
from src.core.exceptions import DataError
from src.utils.model_utils import save_model, load_model

logger = logging.getLogger(__name__)

SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def fit_scaler(
    features: pd.DataFrame,
    method: Optional[str] = None,
) -> tuple[StandardScaler | MinMaxScaler | RobustScaler, List[str]]:
    """Fit a scaler on the feature DataFrame.

    Args:
        features: DataFrame of numeric features to scale.
        method: Scaling method ('standard', 'minmax', 'robust').
            If None, reads from config.

    Returns:
        Tuple of (fitted scaler, list of feature column names).

    Raises:
        DataError: If the scaling method is unknown.
    """
    if method is None:
        cfg = get_config().load("features")
        method = cfg.get("transformation", {}).get("scale_method", "standard")

    if method not in SCALER_MAP:
        raise DataError(
            f"Unknown scaling method '{method}'. Choose from: {list(SCALER_MAP.keys())}"
        )

    scaler_class = SCALER_MAP[method]
    scaler = scaler_class()

    # Replace infinities before fitting
    features_clean = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler.fit(features_clean)
    feature_names = list(features.columns)

    logger.info("Fitted %s scaler on %d features", method, len(feature_names))
    return scaler, feature_names


def transform_features(
    features: pd.DataFrame,
    scaler: StandardScaler | MinMaxScaler | RobustScaler,
) -> pd.DataFrame:
    """Apply a fitted scaler to transform features.

    Args:
        features: DataFrame of numeric features.
        scaler: Fitted scaler to apply.

    Returns:
        Scaled DataFrame with the same column names.
    """
    feature_names = list(features.columns)

    # Replace infinities before scaling
    features_clean = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaled_array = scaler.transform(features_clean)
    scaled_df = pd.DataFrame(scaled_array, columns=feature_names, index=features.index)

    logger.debug("Scaled %d features using %s", len(feature_names), type(scaler).__name__)
    return scaled_df


def fit_transform_features(
    features: pd.DataFrame,
    method: Optional[str] = None,
) -> tuple[pd.DataFrame, StandardScaler | MinMaxScaler | RobustScaler, List[str]]:
    """Fit a scaler and transform features in one step.

    Args:
        features: DataFrame of numeric features.
        method: Scaling method. If None, reads from config.

    Returns:
        Tuple of (scaled DataFrame, fitted scaler, feature names).
    """
    scaler, feature_names = fit_scaler(features, method)
    scaled_df = transform_features(features, scaler)
    return scaled_df, scaler, feature_names


def inverse_transform_features(
    scaled_features: pd.DataFrame,
    scaler: StandardScaler | MinMaxScaler | RobustScaler,
) -> pd.DataFrame:
    """Inverse-transform scaled features back to original scale.

    Args:
        scaled_features: DataFrame of scaled features.
        scaler: Fitted scaler to inverse-apply.

    Returns:
        DataFrame with original-scale feature values.
    """
    feature_names = list(scaled_features.columns)
    original_array = scaler.inverse_transform(scaled_features.values)
    return pd.DataFrame(original_array, columns=feature_names, index=scaled_features.index)


def save_scaler(
    scaler: StandardScaler | MinMaxScaler | RobustScaler,
    filename: str = "scaler.joblib",
) -> None:
    """Persist a fitted scaler to disk.

    Args:
        scaler: Fitted scaler to save.
        filename: Filename (relative to models directory).
    """
    path = MODELS_DIR / filename
    save_model(scaler, path)
    logger.info("Scaler saved to %s", path)


def load_scaler(filename: str = "scaler.joblib") -> StandardScaler | MinMaxScaler | RobustScaler:
    """Load a persisted scaler from disk.

    Args:
        filename: Filename (relative to models directory).

    Returns:
        Loaded scaler.

    Raises:
        DataError: If the file does not exist.
    """
    path = MODELS_DIR / filename
    scaler = load_model(path)
    logger.info("Scaler loaded from %s", path)
    return scaler
