from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.config import get_config
from src.datasets.loader import BENIGN_LABEL, KNOWN_ATTACK_LABELS

logger = logging.getLogger(__name__)


@dataclass
class DatasetProfile:
    """Comprehensive profile of the dataset."""

    n_rows: int = 0
    n_columns: int = 0
    memory_usage_mb: float = 0.0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    class_imbalance_ratio: float = 0.0
    numeric_feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    missing_values: Dict[str, int] = field(default_factory=dict)
    total_missing: int = 0
    duplicate_rows: int = 0
    unique_labels: List[str] = field(default_factory=list)
    n_features: int = 0
    feature_names: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def profile_dataset(
    df: pd.DataFrame,
    label_column: Optional[str] = None,
) -> DatasetProfile:
    """Generate a comprehensive profile of the dataset.

    Args:
        df: DataFrame to profile.
        label_column: Name of the label column. If None, reads from config.

    Returns:
        DatasetProfile with full statistics.
    """
    if label_column is None:
        cfg = get_config().load("features")
        label_column = cfg.get("categorical", {}).get("label_column", " Label").strip()

    profile = DatasetProfile()
    profile.n_rows = len(df)
    profile.n_columns = len(df.columns)
    profile.memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Label analysis
    if label_column in df.columns:
        labels = df[label_column].str.strip()
        profile.label_distribution = labels.value_counts().to_dict()
        profile.unique_labels = sorted(labels.unique().tolist())

        counts = np.array(list(profile.label_distribution.values()))
        if len(counts) > 0 and counts.min() > 0:
            profile.class_imbalance_ratio = float(counts.max() / counts.min())

    # Feature columns (exclude label)
    feature_cols = [c for c in df.columns if c.strip() != label_column.strip()]
    profile.n_features = len(feature_cols)
    profile.feature_names = feature_cols

    # Numeric statistics
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe().to_dict()
    profile.numeric_feature_stats = stats

    # Missing values
    missing = df.isnull().sum()
    profile.missing_values = {k: int(v) for k, v in missing.items() if v > 0}
    profile.total_missing = int(missing.sum())

    # Duplicates
    profile.duplicate_rows = int(df.duplicated().sum())

    # Summary
    profile.summary = {
        "rows": profile.n_rows,
        "columns": profile.n_columns,
        "features": profile.n_features,
        "memory_mb": round(profile.memory_usage_mb, 2),
        "total_missing": profile.total_missing,
        "duplicates": profile.duplicate_rows,
        "unique_labels": len(profile.unique_labels),
        "class_imbalance_ratio": round(profile.class_imbalance_ratio, 2),
    }

    logger.info("Dataset profile generated: %s", profile.summary)
    return profile


def profile_class_distribution(
    df: pd.DataFrame,
    label_column: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Profile class distribution with percentages and counts.

    Args:
        df: DataFrame to analyze.
        label_column: Name of the label column. If None, reads from config.

    Returns:
        Dictionary mapping class labels to their counts and percentages.
    """
    if label_column is None:
        cfg = get_config().load("features")
        label_column = cfg.get("categorical", {}).get("label_column", " Label").strip()

    labels = df[label_column].str.strip()
    total = len(labels)
    distribution = {}

    for label, count in labels.value_counts().items():
        distribution[label] = {
            "count": int(count),
            "percentage": round(count / total * 100, 2),
            "is_benign": label == BENIGN_LABEL,
            "is_known_attack": label in KNOWN_ATTACK_LABELS,
        }

    return distribution


def profile_feature_correlations(
    df: pd.DataFrame,
    max_features: int = 20,
) -> pd.DataFrame:
    """Compute pairwise Pearson correlations for numeric features.

    Args:
        df: DataFrame with numeric features.
        max_features: Maximum number of features to include.

    Returns:
        Correlation matrix DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) > max_features:
        # Keep the top features by variance
        variances = numeric_df.var().sort_values(ascending=False)
        top_cols = variances.head(max_features).index.tolist()
        numeric_df = numeric_df[top_cols]

    corr_matrix = numeric_df.corr()
    logger.info(
        "Correlation matrix computed for %d features", len(corr_matrix.columns)
    )
    return corr_matrix


def identify_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    max_features: int = 50,
) -> List[tuple[str, str, float]]:
    """Identify pairs of highly correlated features.

    Args:
        df: DataFrame with numeric features.
        threshold: Correlation coefficient threshold (inclusive).
        max_features: Maximum features to analyze for performance.

    Returns:
        List of (feature_a, feature_b, correlation) tuples.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) > max_features:
        variances = numeric_df.var().sort_values(ascending=False)
        top_cols = variances.head(max_features).index.tolist()
        numeric_df = numeric_df[top_cols]

    corr = numeric_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if val >= threshold:
                pairs.append((idx, col, round(float(val), 4)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    logger.info("Found %d highly correlated feature pairs (threshold=%.2f)", len(pairs), threshold)
    return pairs


def print_profile(profile: DatasetProfile) -> str:
    """Format a DatasetProfile as a readable string.

    Args:
        profile: DatasetProfile to format.

    Returns:
        Formatted string summary.
    """
    lines = [
        "=== Dataset Profile ===",
        f"Rows: {profile.n_rows:,}",
        f"Columns: {profile.n_columns}",
        f"Features: {profile.n_features}",
        f"Memory: {profile.memory_usage_mb:.2f} MB",
        f"Total missing values: {profile.total_missing:,}",
        f"Duplicate rows: {profile.duplicate_rows:,}",
        f"Unique labels: {len(profile.unique_labels)}",
        f"Class imbalance ratio: {profile.class_imbalance_ratio:.2f}",
        "",
        "--- Label Distribution ---",
    ]
    for label, count in sorted(profile.label_distribution.items()):
        pct = count / profile.n_rows * 100
        lines.append(f"  {label}: {count:,} ({pct:.1f}%)")

    return "\n".join(lines)
