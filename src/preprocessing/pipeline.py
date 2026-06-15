from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.core.config import get_config
from src.core.exceptions import DataError
from src.preprocessing.cleaner import CleaningResult, clean_dataset
from src.preprocessing.encoder import encode_labels, save_label_encoder
from src.preprocessing.imputer import ImputationResult, impute_missing
from src.preprocessing.scaler import fit_transform_features, save_scaler

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Aggregated result of the full preprocessing pipeline."""

    cleaning: Optional[CleaningResult] = None
    imputation: Optional[ImputationResult] = None
    encoding_applied: bool = False
    scaling_applied: bool = False
    original_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)
    feature_columns: List[str] = field(default_factory=list)
    label_column: str = "Label"
    steps_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def run_preprocessing(
    df: pd.DataFrame,
    label_column: str = "Label",
    feature_columns: Optional[List[str]] = None,
    clean: bool = True,
    impute: bool = True,
    encode: bool = True,
    scale: bool = True,
    save_artifacts: bool = False,
    artifact_prefix: str = "preprocessing",
) -> Tuple[pd.DataFrame, PreprocessingResult]:
    """Run the full preprocessing pipeline: clean → impute → encode → scale.

    This is the primary entry point for preprocessing raw DataFrames through
    the complete pipeline.

    Args:
        df: Raw input DataFrame.
        label_column: Name of the label/target column.
        feature_columns: Specific feature columns to process. If None, uses
            all columns except the label column.
        clean: Whether to run data cleaning.
        impute: Whether to impute missing values.
        encode: Whether to encode labels.
        scale: Whether to scale features.
        save_artifacts: Whether to persist scaler and encoder to disk.
        artifact_prefix: Prefix for saved artifact filenames.

    Returns:
        Tuple of (processed DataFrame, PreprocessingResult).

    Raises:
        DataError: If the input DataFrame is empty or invalid.
    """
    if df.empty:
        raise DataError("Cannot preprocess an empty DataFrame")

    result = PreprocessingResult(
        original_shape=df.shape,
        label_column=label_column,
    )

    logger.info("Starting preprocessing pipeline: shape=%s", df.shape)

    # Step 1: Cleaning
    if clean:
        try:
            df, cleaning_result = clean_dataset(df)
            result.cleaning = cleaning_result
            result.steps_completed.append("clean")
            logger.info("Cleaning complete: %d rows remaining", len(df))
        except DataError as e:
            result.errors.append(f"Cleaning failed: {e}")
            logger.warning("Cleaning failed: %s", e)

    # Step 2: Imputation
    if impute:
        try:
            df, imputation_result = impute_missing(df)
            result.imputation = imputation_result
            result.steps_completed.append("impute")
            logger.info("Imputation complete: strategy=%s", imputation_result.strategy_used)
        except DataError as e:
            result.errors.append(f"Imputation failed: {e}")
            logger.warning("Imputation failed: %s", e)

    # Determine feature columns
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != label_column]
    result.feature_columns = feature_columns

    # Step 3: Label encoding
    if encode and label_column in df.columns:
        try:
            encoded_labels, encoder = encode_labels(df[label_column])
            df[label_column] = encoded_labels
            result.encoding_applied = True
            result.steps_completed.append("encode")

            if save_artifacts:
                encoder_path = f"{artifact_prefix}_label_encoder.joblib"
                save_label_encoder(encoder, encoder_path)
                logger.info("Saved label encoder to %s", encoder_path)

            logger.info("Label encoding complete: classes=%s", list(encoder.classes_))
        except DataError as e:
            result.errors.append(f"Encoding failed: {e}")
            logger.warning("Encoding failed: %s", e)

    # Step 4: Feature scaling
    if scale:
        try:
            numeric_features = [c for c in feature_columns if c in df.columns]
            if numeric_features:
                df_scaled, scaler, scaled_cols = fit_transform_features(
                    df[numeric_features]
                )
                df[numeric_features] = df_scaled
                result.scaling_applied = True
                result.steps_completed.append("scale")

                if save_artifacts:
                    scaler_path = f"{artifact_prefix}_scaler.joblib"
                    save_scaler(scaler, scaler_path)
                    logger.info("Saved scaler to %s", scaler_path)

                logger.info("Feature scaling complete: %d features scaled", len(scaled_cols))
        except DataError as e:
            result.errors.append(f"Scaling failed: {e}")
            logger.warning("Scaling failed: %s", e)

    result.final_shape = df.shape
    result.stats["rows_retained"] = len(df)
    result.stats["columns_retained"] = len(df.columns)

    logger.info(
        "Preprocessing complete: %s → %s, steps=%s",
        result.original_shape, result.final_shape, result.steps_completed,
    )

    return df, result


def run_preprocessing_on_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str = "Label",
    feature_columns: Optional[List[str]] = None,
    save_artifacts: bool = False,
    artifact_prefix: str = "preprocessing",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PreprocessingResult]:
    """Run preprocessing on train/val/test splits consistently.

    Fits cleaning, encoding, and scaling on the training set, then applies
    the same transformations to validation and test sets.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        label_column: Name of the label column.
        feature_columns: Feature columns to process.
        save_artifacts: Whether to persist fitted scalers/encoders.
        artifact_prefix: Prefix for saved artifact filenames.

    Returns:
        Tuple of (train, val, test DataFrames, PreprocessingResult).
    """
    if train_df.empty:
        raise DataError("Training DataFrame is empty")

    result = PreprocessingResult(
        original_shape=train_df.shape,
        label_column=label_column,
    )

    if feature_columns is None:
        feature_columns = [c for c in train_df.columns if c != label_column]
    result.feature_columns = feature_columns

    # Clean
    train_df, cleaning_result = clean_dataset(train_df)
    val_df, _ = clean_dataset(val_df)
    test_df, _ = clean_dataset(test_df)
    result.cleaning = cleaning_result
    result.steps_completed.append("clean")

    # Impute
    train_df, impute_result = impute_missing(train_df)
    val_df, _ = impute_missing(val_df)
    test_df, _ = impute_missing(test_df)
    result.imputation = impute_result
    result.steps_completed.append("impute")

    # Encode
    if label_column in train_df.columns:
        encoded_train, encoder = encode_labels(train_df[label_column])
        train_df[label_column] = encoded_train
        val_df[label_column] = encoder.transform(val_df[label_column].astype(str))
        test_df[label_column] = encoder.transform(test_df[label_column].astype(str))
        result.encoding_applied = True
        result.steps_completed.append("encode")

        if save_artifacts:
            save_label_encoder(encoder, f"{artifact_prefix}_label_encoder.joblib")

    # Scale — fit on train, transform all
    numeric_features = [c for c in feature_columns if c in train_df.columns]
    if numeric_features:
        train_scaled, scaler, scaled_cols = fit_transform_features(train_df[numeric_features])
        train_df[numeric_features] = train_scaled

        from src.preprocessing.scaler import transform_features
        val_df[numeric_features] = transform_features(val_df[numeric_features], scaler)
        test_df[numeric_features] = transform_features(test_df[numeric_features], scaler)

        result.scaling_applied = True
        result.steps_completed.append("scale")

        if save_artifacts:
            save_scaler(scaler, f"{artifact_prefix}_scaler.joblib")

    result.final_shape = train_df.shape

    logger.info(
        "Split preprocessing complete: train=%s, val=%s, test=%s",
        train_df.shape, val_df.shape, test_df.shape,
    )

    return train_df, val_df, test_df, result
