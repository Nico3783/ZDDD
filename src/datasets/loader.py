from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.core.config import get_config
from src.core.constants import CONFIG_DIR, DATA_DIR
from src.core.exceptions import DataError
from src.utils.file_utils import read_csv

logger = logging.getLogger(__name__)

BENIGN_LABEL = "BENIGN"
DEFAULT_LABEL_COLUMN = " Label"

KNOWN_ATTACK_LABELS: List[str] = [
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS Slowloris",
    "DoS Slowhttptest",
    "Heartbleed",
]


def _get_feature_columns() -> List[str]:
    """Load the list of selected features from config."""
    cfg = get_config().load("features")
    return cfg.get("flow_features", [])


def _get_label_column() -> str:
    """Load the label column name from config."""
    cfg = get_config().load("features")
    return cfg.get("categorical", {}).get("label_column", DEFAULT_LABEL_COLUMN)


def load_raw_dataset(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the raw CICIDS2017 CSV file.

    Args:
        path: Path to the CSV file. If None, reads from config/paths.yaml.

    Returns:
        DataFrame with the raw dataset.

    Raises:
        DataError: If the file is missing or cannot be parsed.
    """
    if path is None:
        paths_cfg = get_config().load("paths")
        path = DATA_DIR.parent / paths_cfg["data"]["raw"].lstrip("./")

    path = Path(path)
    if not path.exists():
        raise DataError(f"Dataset file not found: {path}")

    logger.info("Loading dataset from %s", path)
    df = read_csv(path)
    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))
    return df


def strip_column_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading/trailing whitespace from all column names.

    The CICIDS2017 dataset is known to have spaces in column headers.

    Args:
        df: DataFrame to clean.

    Returns:
        DataFrame with cleaned column names.
    """
    df.columns = [col.strip() for col in df.columns]
    logger.debug("Stripped whitespace from %d column names", len(df.columns))
    return df


def standardize_labels(
    df: pd.DataFrame,
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    """Standardize label values by stripping whitespace.

    Args:
        df: DataFrame containing the label column.
        label_column: Name of the label column. If None, reads from config.

    Returns:
        DataFrame with standardized labels.

    Raises:
        DataError: If the label column is not found.
    """
    if label_column is None:
        label_column = _get_label_column()

    # After stripping column whitespace, the label column won't have leading space
    clean_label_col = label_column.strip()

    if clean_label_col not in df.columns:
        raise DataError(
            f"Label column '{clean_label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df[clean_label_col] = df[clean_label_col].str.strip()
    logger.info(
        "Label distribution:\n%s",
        df[clean_label_col].value_counts().to_string(),
    )
    return df


def filter_dos_traffic(df: pd.DataFrame, label_column: Optional[str] = None) -> pd.DataFrame:
    """Filter dataset to retain only BENIGN and known DoS attack labels.

    Args:
        df: DataFrame to filter.
        label_column: Name of the label column. If None, reads from config.

    Returns:
        Filtered DataFrame with only BENIGN and DoS-related labels.
    """
    if label_column is None:
        label_column = _get_label_column()

    clean_label_col = label_column.strip()
    valid_labels = [BENIGN_LABEL] + KNOWN_ATTACK_LABELS
    mask = df[clean_label_col].isin(valid_labels)
    filtered = df[mask].copy()
    logger.info(
        "Filtered to %d rows (from %d) retaining labels: %s",
        len(filtered),
        len(df),
        valid_labels,
    )
    return filtered


def separate_features_and_labels(
    df: pd.DataFrame,
    label_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature columns from the label column.

    Args:
        df: DataFrame to split.
        label_column: Name of the label column. If None, reads from config.

    Returns:
        Tuple of (features DataFrame, labels Series).

    Raises:
        DataError: If the label column is not found.
    """
    if label_column is None:
        label_column = _get_label_column()

    clean_label_col = label_column.strip()
    if clean_label_col not in df.columns:
        raise DataError(f"Label column '{clean_label_col}' not found")

    labels = df[clean_label_col].copy()
    features = df.drop(columns=[clean_label_col])
    return features, labels


def extract_benign_only(
    df: pd.DataFrame,
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    """Extract only BENIGN traffic rows for anomaly detection training.

    Args:
        df: DataFrame to filter.
        label_column: Name of the label column. If None, reads from config.

    Returns:
        DataFrame containing only BENIGN traffic (features only, no label column).
    """
    if label_column is None:
        label_column = _get_label_column()

    clean_label_col = label_column.strip()
    benign_mask = df[clean_label_col].str.strip() == BENIGN_LABEL
    benign_df = df[benign_mask].drop(columns=[clean_label_col]).copy()
    logger.info("Extracted %d benign rows for anomaly detection training", len(benign_df))
    return benign_df


def load_clean_dataset(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load, clean, and standardize the CICIDS2017 dataset end-to-end.

    Steps:
        1. Load raw CSV
        2. Strip column whitespace
        3. Standardize label values
        4. Filter to DoS-relevant traffic

    Args:
        path: Path to the CSV file. If None, reads from config.

    Returns:
        Cleaned DataFrame with BENIGN and DoS attack labels.
    """
    df = load_raw_dataset(path)
    df = strip_column_whitespace(df)
    df = standardize_labels(df)
    df = filter_dos_traffic(df)
    logger.info("Clean dataset ready: %d rows, %d columns", len(df), len(df.columns))
    return df
