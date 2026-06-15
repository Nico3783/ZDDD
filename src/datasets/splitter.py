from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.config import get_config
from src.core.constants import DATA_DIR
from src.core.exceptions import DataError
from src.utils.file_utils import write_csv

logger = logging.getLogger(__name__)


def split_dataset(
    df: pd.DataFrame,
    label_column: Optional[str] = None,
    test_size: Optional[float] = None,
    val_size: Optional[float] = None,
    random_state: Optional[int] = None,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train, validation, and test sets.

    Uses stratified splitting to preserve class distribution across splits.

    Args:
        df: DataFrame to split.
        label_column: Name of the label column. If None, reads from config.
        test_size: Fraction for test set. If None, reads from config.
        val_size: Fraction for validation set. If None, reads from config.
        random_state: Random seed. If None, reads from config.
        stratify: Whether to stratify by label.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        DataError: If split sizes are invalid.
    """
    if label_column is None:
        cfg = get_config().load("features")
        label_column = cfg.get("categorical", {}).get("label_column", " Label").strip()

    if test_size is None:
        settings = get_config().load("settings")
        test_size = settings.get("test_size", 0.2)

    if val_size is None:
        settings = get_config().load("settings")
        val_size = settings.get("val_size", 0.1)

    if random_state is None:
        settings = get_config().load("settings")
        random_state = settings.get("random_state", 42)

    if test_size + val_size >= 1.0:
        raise DataError(
            f"test_size ({test_size}) + val_size ({val_size}) must be < 1.0"
        )

    labels = df[label_column].str.strip() if stratify else None

    # First split: separate test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Second split: separate validation set from train_val
    adjusted_val_size = val_size / (1 - test_size)
    train_val_labels = (
        train_val[label_column].str.strip() if stratify else None
    )

    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val_labels,
    )

    logger.info(
        "Dataset split: train=%d, val=%d, test=%d",
        len(train),
        len(val),
        len(test),
    )

    return train, val, test


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
) -> tuple[Path, Path, Path]:
    """Save train/val/test splits to CSV files.

    Args:
        train: Training split DataFrame.
        val: Validation split DataFrame.
        test: Test split DataFrame.
        output_dir: Directory to save files. If None, uses config/paths.yaml.

    Returns:
        Tuple of (train_path, val_path, test_path).
    """
    if output_dir is None:
        paths_cfg = get_config().load("paths")
        output_dir = DATA_DIR.parent / paths_cfg["data"]["processed"].lstrip("./")

    output_dir = Path(output_dir)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    write_csv(train, train_path)
    write_csv(val, val_path)
    write_csv(test, test_path)

    logger.info("Splits saved to %s", output_dir)
    return train_path, val_path, test_path


def load_splits(
    data_dir: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved train/val/test splits.

    Args:
        data_dir: Directory containing the CSV files. If None, uses config.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        DataError: If any split file is missing.
    """
    if data_dir is None:
        paths_cfg = get_config().load("paths")
        data_dir = DATA_DIR.parent / paths_cfg["data"]["processed"].lstrip("./")

    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    test_path = data_dir / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise DataError(f"Split file not found: {p}")

    from src.utils.file_utils import read_csv

    train = read_csv(train_path)
    val = read_csv(val_path)
    test = read_csv(test_path)

    logger.info(
        "Loaded splits: train=%d, val=%d, test=%d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def validate_split_sizes(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    expected_test_ratio: float = 0.2,
    expected_val_ratio: float = 0.1,
    tolerance: float = 0.02,
) -> bool:
    """Validate that split sizes match expected ratios.

    Args:
        train: Training split.
        val: Validation split.
        test: Test split.
        expected_test_ratio: Expected fraction of total for test.
        expected_val_ratio: Expected fraction of total for validation.
        tolerance: Allowed deviation from expected ratios.

    Returns:
        True if splits are within tolerance.
    """
    total = len(train) + len(val) + len(test)
    test_ratio = len(test) / total
    val_ratio = len(val) / total

    ok = True
    if abs(test_ratio - expected_test_ratio) > tolerance:
        logger.warning(
            "Test ratio %.3f deviates from expected %.3f (tol=%.3f)",
            test_ratio,
            expected_test_ratio,
            tolerance,
        )
        ok = False

    if abs(val_ratio - expected_val_ratio) > tolerance:
        logger.warning(
            "Val ratio %.3f deviates from expected %.3f (tol=%.3f)",
            val_ratio,
            expected_val_ratio,
            tolerance,
        )
        ok = False

    if ok:
        logger.info(
            "Split sizes validated: train=%.1f%%, val=%.1f%%, test=%.1f%%",
            len(train) / total * 100,
            len(val) / total * 100,
            len(test) / total * 100,
        )

    return ok


def validate_split_class_distribution(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label_column: Optional[str] = None,
    tolerance: float = 0.05,
) -> bool:
    """Validate that class distributions are preserved across splits.

    Args:
        train: Training split.
        val: Validation split.
        test: Test split.
        label_column: Label column name. If None, reads from config.
        tolerance: Maximum allowed deviation in class proportions.

    Returns:
        True if distributions are within tolerance.
    """
    if label_column is None:
        cfg = get_config().load("features")
        label_column = cfg.get("categorical", {}).get("label_column", " Label").strip()

    clean_label = label_column.strip()
    train_dist = train[clean_label].str.strip().value_counts(normalize=True)
    val_dist = val[clean_label].str.strip().value_counts(normalize=True)
    test_dist = test[clean_label].str.strip().value_counts(normalize=True)

    ok = True
    for label in train_dist.index:
        train_prop = train_dist.get(label, 0)
        val_prop = val_dist.get(label, 0)
        test_prop = test_dist.get(label, 0)

        if abs(val_prop - train_prop) > tolerance:
            logger.warning(
                "Class '%s': train=%.3f, val=%.3f (diff=%.3f > tol=%.3f)",
                label,
                train_prop,
                val_prop,
                abs(val_prop - train_prop),
                tolerance,
            )
            ok = False

        if abs(test_prop - train_prop) > tolerance:
            logger.warning(
                "Class '%s': train=%.3f, test=%.3f (diff=%.3f > tol=%.3f)",
                label,
                train_prop,
                test_prop,
                abs(test_prop - train_prop),
                tolerance,
            )
            ok = False

    if ok:
        logger.info("Class distributions validated across all splits")

    return ok
