"""Data preprocessing and cleaning.

Provides functionality for data cleaning (duplicates, missing values,
infinite values, outliers), label encoding, feature scaling, and a
unified preprocessing pipeline.
"""

from src.preprocessing.cleaner import (
    CleaningResult,
    clean_dataset,
    clip_outliers,
    drop_duplicates,
    handle_infinite_values,
    handle_missing_values,
)
from src.preprocessing.encoder import (
    create_binary_labels,
    decode_labels,
    encode_labels,
    get_label_mapping,
    load_label_encoder,
    save_label_encoder,
)
from src.preprocessing.imputer import (
    ImputationResult,
    create_imputer,
    fit_imputer,
    get_missing_summary,
    impute_missing,
    transform_with_imputer,
)
from src.preprocessing.pipeline import (
    PreprocessingResult,
    run_preprocessing,
    run_preprocessing_on_splits,
)
from src.preprocessing.scaler import (
    fit_scaler,
    fit_transform_features,
    inverse_transform_features,
    load_scaler,
    save_scaler,
    transform_features,
)

__all__ = [
    # Cleaner
    "CleaningResult",
    "clean_dataset",
    "clip_outliers",
    "drop_duplicates",
    "handle_infinite_values",
    "handle_missing_values",
    # Encoder
    "create_binary_labels",
    "decode_labels",
    "encode_labels",
    "get_label_mapping",
    "load_label_encoder",
    "save_label_encoder",
    # Imputer
    "ImputationResult",
    "create_imputer",
    "fit_imputer",
    "get_missing_summary",
    "impute_missing",
    "transform_with_imputer",
    # Pipeline
    "PreprocessingResult",
    "run_preprocessing",
    "run_preprocessing_on_splits",
    # Scaler
    "fit_scaler",
    "fit_transform_features",
    "inverse_transform_features",
    "load_scaler",
    "save_scaler",
    "transform_features",
]
