"""Dataset management module.

Provides functionality for loading, validating, profiling, and splitting
the CICIDS2017 dataset for zero-day DoS detection.
"""

from src.datasets.loader import (
    BENIGN_LABEL,
    KNOWN_ATTACK_LABELS,
    extract_benign_only,
    filter_dos_traffic,
    load_clean_dataset,
    load_raw_dataset,
    separate_features_and_labels,
    standardize_labels,
    strip_column_whitespace,
)
from src.datasets.profiler import (
    DatasetProfile,
    identify_highly_correlated_features,
    print_profile,
    profile_class_distribution,
    profile_dataset,
    profile_feature_correlations,
)
from src.datasets.splitter import (
    load_splits,
    save_splits,
    split_dataset,
    validate_split_class_distribution,
    validate_split_sizes,
)
from src.datasets.validator import (
    ValidationResult,
    run_full_validation,
    validate_data_quality,
    validate_dataset_structure,
    validate_features,
    validate_labels,
)

__all__ = [
    # Loader
    "BENIGN_LABEL",
    "KNOWN_ATTACK_LABELS",
    "extract_benign_only",
    "filter_dos_traffic",
    "load_clean_dataset",
    "load_raw_dataset",
    "separate_features_and_labels",
    "standardize_labels",
    "strip_column_whitespace",
    # Validator
    "ValidationResult",
    "run_full_validation",
    "validate_data_quality",
    "validate_dataset_structure",
    "validate_features",
    "validate_labels",
    # Profiler
    "DatasetProfile",
    "identify_highly_correlated_features",
    "print_profile",
    "profile_class_distribution",
    "profile_dataset",
    "profile_feature_correlations",
    # Splitter
    "load_splits",
    "save_splits",
    "split_dataset",
    "validate_split_class_distribution",
    "validate_split_sizes",
]
