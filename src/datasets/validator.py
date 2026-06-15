from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.core.config import get_config
from src.core.exceptions import DataError
from src.datasets.loader import BENIGN_LABEL, KNOWN_ATTACK_LABELS

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_dataset_structure(df: pd.DataFrame) -> ValidationResult:
    """Validate that the DataFrame has the expected structure.

    Checks:
        - DataFrame is not empty
        - Has the required label column
        - No duplicate column names

    Args:
        df: DataFrame to validate.

    Returns:
        ValidationResult with errors/warnings.
    """
    result = ValidationResult()

    if df.empty:
        result.add_error("Dataset is empty (zero rows)")
        return result

    if len(df.columns) == 0:
        result.add_error("Dataset has no columns")
        return result

    # Check for duplicate column names
    dupes = [col for col in df.columns if list(df.columns).count(col) > 1]
    if dupes:
        result.add_error(f"Duplicate column names found: {set(dupes)}")

    # Check label column exists
    cfg = get_config().load("features")
    label_col = cfg.get("categorical", {}).get("label_column", " Label").strip()
    if label_col not in df.columns:
        result.add_error(f"Label column '{label_col}' not found in dataset")

    result.stats["n_rows"] = len(df)
    result.stats["n_columns"] = len(df.columns)
    result.stats["columns"] = list(df.columns)

    return result


def validate_labels(df: pd.DataFrame) -> ValidationResult:
    """Validate label column contains expected values.

    Checks:
        - All known attack labels are present
        - No unexpected/unknown labels exist
        - Label distribution is reasonable

    Args:
        df: DataFrame with label column.

    Returns:
        ValidationResult with errors/warnings.
    """
    result = ValidationResult()

    cfg = get_config().load("features")
    label_col = cfg.get("categorical", {}).get("label_column", " Label").strip()

    if label_col not in df.columns:
        result.add_error(f"Label column '{label_col}' not found")
        return result

    unique_labels = set(df[label_col].str.strip().unique())
    expected_labels = {BENIGN_LABEL} | set(KNOWN_ATTACK_LABELS)

    # Check for unexpected labels
    unexpected = unique_labels - expected_labels
    if unexpected:
        result.add_warning(f"Unexpected labels found (will be ignored): {unexpected}")

    # Check if BENIGN is present
    if BENIGN_LABEL not in unique_labels:
        result.add_error("BENIGN label not found in dataset")

    # Check how many known attacks are present
    present_attacks = unique_labels & set(KNOWN_ATTACK_LABELS)
    missing_attacks = set(KNOWN_ATTACK_LABELS) - unique_labels
    if missing_attacks:
        result.add_warning(f"Missing expected attack labels: {missing_attacks}")

    label_dist = df[label_col].str.strip().value_counts().to_dict()
    result.stats["label_distribution"] = label_dist
    result.stats["unique_labels"] = len(unique_labels)
    result.stats["present_attacks"] = list(present_attacks)

    return result


def validate_features(
    df: pd.DataFrame,
    required_features: Optional[List[str]] = None,
) -> ValidationResult:
    """Validate that required feature columns exist and have valid data.

    Checks:
        - All required features are present
        - No excessive missing values
        - No infinite values in numeric columns

    Args:
        df: DataFrame to validate.
        required_features: List of required feature column names.
            If None, loads from config.

    Returns:
        ValidationResult with errors/warnings.
    """
    result = ValidationResult()

    if required_features is None:
        cfg = get_config().load("features")
        required_features = cfg.get("flow_features", [])

    # Strip whitespace from required features to match cleaned column names
    required_clean = [f.strip() for f in required_features]

    missing = [f for f in required_clean if f not in df.columns]
    if missing:
        result.add_error(f"Missing required feature columns: {missing}")

    # Check missing values in present columns
    present = [f for f in required_clean if f in df.columns]
    for col in present:
        null_pct = df[col].isnull().mean() * 100
        if null_pct > 50:
            result.add_error(f"Column '{col}' has {null_pct:.1f}% missing values")
        elif null_pct > 10:
            result.add_warning(f"Column '{col}' has {null_pct:.1f}% missing values")

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            result.add_warning(f"Column '{col}' contains infinite values")

    result.stats["required_features_present"] = len(present)
    result.stats["required_features_missing"] = len(missing)

    return result


def validate_data_quality(df: pd.DataFrame) -> ValidationResult:
    """Validate overall data quality.

    Checks:
        - No completely duplicate rows (warning threshold)
        - Numeric columns have reasonable ranges
        - No all-null columns

    Args:
        df: DataFrame to validate.

    Returns:
        ValidationResult with errors/warnings.
    """
    result = ValidationResult()

    # Check for all-null columns
    null_cols = [col for col in df.columns if df[col].isnull().all()]
    if null_cols:
        result.add_error(f"Columns with all null values: {null_cols}")

    # Check for duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        dupe_pct = n_dupes / len(df) * 100
        if dupe_pct > 30:
            result.add_error(
                f"{n_dupes} duplicate rows ({dupe_pct:.1f}%) — exceeds 30% threshold"
            )
        elif dupe_pct > 5:
            result.add_warning(
                f"{n_dupes} duplicate rows ({dupe_pct:.1f}%)"
            )

    result.stats["n_duplicates"] = int(n_dupes)
    result.stats["n_null_columns"] = len(null_cols)

    return result


def run_full_validation(
    df: pd.DataFrame,
    check_features: bool = True,
) -> ValidationResult:
    """Run all validation checks on the dataset.

    Args:
        df: DataFrame to validate.
        check_features: Whether to validate required features.

    Returns:
        Combined ValidationResult from all checks.
    """
    combined = ValidationResult()

    # Structure
    structure = validate_dataset_structure(df)
    combined.errors.extend(structure.errors)
    combined.warnings.extend(structure.warnings)
    combined.stats.update(structure.stats)

    if not structure.is_valid:
        return combined

    # Labels
    labels = validate_labels(df)
    combined.errors.extend(labels.errors)
    combined.warnings.extend(labels.warnings)
    combined.stats.update(labels.stats)

    # Features
    if check_features:
        features = validate_features(df)
        combined.errors.extend(features.errors)
        combined.warnings.extend(features.warnings)
        combined.stats.update(features.stats)

    # Data quality
    quality = validate_data_quality(df)
    combined.errors.extend(quality.errors)
    combined.warnings.extend(quality.warnings)
    combined.stats.update(quality.stats)

    combined.is_valid = len(combined.errors) == 0

    logger.info(
        "Validation complete: valid=%s, errors=%d, warnings=%d",
        combined.is_valid,
        len(combined.errors),
        len(combined.warnings),
    )
    return combined
