"""Feature engineering and extraction.

Provides functionality for feature extraction, selection, transformation,
and schema validation for the CICIDS2017 network traffic features.
"""

from src.features.extractor import (
    ExtractionResult,
    extract_aggregate_features,
    extract_difference_features,
    extract_ratio_features,
    extract_statistical_features,
    select_top_features,
)
from src.features.engineering import (
    FeatureEngineeringResult,
    run_feature_engineering,
    run_feature_engineering_on_splits,
)
from src.features.importance import FeatureImportanceAnalyzer
from src.features.schema import (
    SchemaValidationResult,
    ensure_feature_consistency,
    get_feature_set_difference,
    validate_feature_schema,
    validate_feature_values,
)
from src.features.selector import (
    get_feature_names,
    remove_redundant_features,
    select_features_by_config,
    select_features_by_correlation,
    select_features_by_variance,
)
from src.features.transformer import (
    add_interaction_features,
    add_ratio_features,
    add_statistical_features,
    apply_feature_transformations,
    log_transform_features,
)

__all__ = [
    # Extractor
    "ExtractionResult",
    "extract_aggregate_features",
    "extract_difference_features",
    "extract_ratio_features",
    "extract_statistical_features",
    "select_top_features",
    # Engineering
    "FeatureEngineeringResult",
    "run_feature_engineering",
    "run_feature_engineering_on_splits",
    # Importance
    "FeatureImportanceAnalyzer",
    # Schema
    "SchemaValidationResult",
    "ensure_feature_consistency",
    "get_feature_set_difference",
    "validate_feature_schema",
    "validate_feature_values",
    # Selector
    "get_feature_names",
    "remove_redundant_features",
    "select_features_by_config",
    "select_features_by_correlation",
    "select_features_by_variance",
    # Transformer
    "add_interaction_features",
    "add_ratio_features",
    "add_statistical_features",
    "apply_feature_transformations",
    "log_transform_features",
]
