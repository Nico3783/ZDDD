from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.core.config import get_config
from src.core.constants import MODELS_DIR
from src.core.exceptions import DataError
from src.utils.model_utils import save_model

logger = logging.getLogger(__name__)


def encode_labels(
    labels: pd.Series,
    encoder: Optional[LabelEncoder] = None,
) -> tuple[np.ndarray, LabelEncoder]:
    """Encode string labels to integer values.

    Args:
        labels: Series of string labels.
        encoder: Pre-fitted LabelEncoder. If None, fits a new one.

    Returns:
        Tuple of (encoded integer array, fitted LabelEncoder).

    Raises:
        DataError: If labels contain null values.
    """
    if labels.isnull().any():
        raise DataError("Labels contain null values — clean before encoding")

    if encoder is None:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(labels.astype(str))
        logger.info("Fitted label encoder with classes: %s", list(encoder.classes_))
    else:
        encoded = encoder.transform(labels.astype(str))
        logger.info("Transformed labels using existing encoder")

    return encoded, encoder


def decode_labels(
    encoded: np.ndarray,
    encoder: LabelEncoder,
) -> np.ndarray:
    """Decode integer labels back to original string values.

    Args:
        encoded: Array of integer-encoded labels.
        encoder: Fitted LabelEncoder to use for decoding.

    Returns:
        Array of decoded string labels.
    """
    return encoder.inverse_transform(encoded)


def create_binary_labels(
    labels: pd.Series,
    benign_label: str = "BENIGN",
) -> pd.Series:
    """Convert multi-class labels to binary (0=benign, 1=attack).

    Args:
        labels: Series of string labels.
        benign_label: The label to treat as benign (class 0).

    Returns:
        Series with binary integer labels (0 or 1).
    """
    binary = (labels.str.strip() != benign_label).astype(int)
    n_attack = int(binary.sum())
    n_benign = int((binary == 0).sum())
    logger.info(
        "Binary labels created: %d benign (0), %d attack (1)",
        n_benign,
        n_attack,
    )
    return binary


def get_label_mapping(labels: pd.Series) -> Dict[str, int]:
    """Get a mapping of label strings to integer values.

    Args:
        labels: Series of string labels.

    Returns:
        Dictionary mapping label strings to their integer codes.
    """
    unique_sorted = sorted(labels.str.strip().unique())
    return {label: idx for idx, label in enumerate(unique_sorted)}


def save_label_encoder(
    encoder: LabelEncoder,
    filename: str = "label_encoder.joblib",
) -> None:
    """Persist a fitted LabelEncoder to disk.

    Args:
        encoder: Fitted LabelEncoder to save.
        filename: Filename (relative to models directory).
    """
    path = MODELS_DIR / filename
    save_model(encoder, path)
    logger.info("Label encoder saved to %s", path)


def load_label_encoder(filename: str = "label_encoder.joblib") -> LabelEncoder:
    """Load a persisted LabelEncoder from disk.

    Args:
        filename: Filename (relative to models directory).

    Returns:
        Loaded LabelEncoder.

    Raises:
        DataError: If the file does not exist.
    """
    from src.utils.model_utils import load_model

    path = MODELS_DIR / filename
    encoder = load_model(path)
    logger.info("Label encoder loaded from %s", path)
    return encoder
