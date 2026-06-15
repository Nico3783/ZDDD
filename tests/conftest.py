"""Shared fixtures for the zero-day DoS detection engine test suite."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml


# ---------------------------------------------------------------------------
# Patch broken imports before any test module loads.
# stream_reader.py imports `DataCleaner` from src.preprocessing.cleaner
# and `FeatureEncoder` from src.preprocessing.encoder — neither exists.
# We inject stubs so the import chain succeeds.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project-like directory structure."""
    for sub in ["data", "models", "logs", "logs/alerts", "config", "results"]:
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def tmp_config_dir(tmp_project: Path) -> Path:
    """Return the temporary config directory and populate it with minimal YAML."""
    cfg_dir = tmp_project / "config"
    return cfg_dir


@pytest.fixture()
def tmp_log_dir(tmp_project: Path) -> Path:
    """Return a temporary log directory."""
    return tmp_project / "logs"


@pytest.fixture()
def tmp_alert_log_dir(tmp_project: Path) -> Path:
    """Return a temporary alert log directory."""
    d = tmp_project / "logs" / "alerts"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def tmp_model_dir(tmp_project: Path) -> Path:
    """Return a temporary model directory."""
    return tmp_project / "models"


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_thresholds_yaml() -> Dict[str, Any]:
    """Return a minimal thresholds config dict."""
    return {
        "anomaly_detection": {
            "contamination": 0.1,
            "score_threshold_percentile": 95.0,
            "min_anomaly_score": 0.0,
            "max_anomaly_score": 1.0,
        },
        "classification": {
            "decision_threshold": 0.5,
            "optimize_threshold": True,
            "optimization_metric": "f1",
        },
        "alerting": {
            "severity_levels": {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.5,
                "low": 0.0,
            },
            "minimum_confidence": 0.5,
            "cooldown_seconds": 10,
            "batch_alerts": True,
            "max_alerts_per_minute": 60,
            "severity_boost_zero_day": 0.2,
            "zero_day_high_confidence": 0.7,
        },
        "streaming": {
            "anomaly_rate_warning_threshold": 5.0,
            "anomaly_rate_critical_threshold": 20.0,
            "window_size_seconds": 60,
        },
        "evaluation": {
            "acceptable_fpr": 0.05,
            "acceptable_latency_ms": 100.0,
            "minimum_throughput": 1000,
            "minimum_accuracy": 0.95,
            "minimum_f1": 0.90,
        },
    }


@pytest.fixture()
def sample_models_yaml() -> Dict[str, Any]:
    """Return a minimal models config dict."""
    return {
        "isolation_forest": {
            "n_estimators": 100,
            "max_samples": "auto",
            "contamination": 0.1,
            "max_features": 1.0,
            "bootstrap": False,
            "n_jobs": 1,
            "random_state": 42,
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": 1,
            "random_state": 42,
            "class_weight": "balanced",
            "criterion": "gini",
        },
    }


@pytest.fixture()
def sample_features_yaml() -> Dict[str, Any]:
    """Return a minimal features config dict."""
    return {
        "flow_features": [
            "Fwd Packet Length Mean",
            "Fwd Packet Length Std",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Std",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
        ],
        "categorical": {
            "label_column": "Label",
            "benign_label": "BENIGN",
            "attack_labels": ["DoS Hulk", "DoS GoldenEye"],
        },
    }


@pytest.fixture()
def sample_logging_yaml() -> Dict[str, Any]:
    """Return a minimal logging config dict."""
    return {
        "level": "INFO",
        "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
    }


@pytest.fixture()
def sample_paths_yaml() -> Dict[str, Any]:
    """Return a minimal paths config dict."""
    return {
        "data": {"raw": "data/raw.csv", "processed": "data/processed/"},
        "models": {"isolation_forest": "models/isolation_forest.joblib"},
        "logs": {"dir": "logs/"},
        "alerts": {"dir": "alerts/"},
    }


@pytest.fixture()
def write_config_files(
    tmp_config_dir: Path,
    sample_thresholds_yaml: Dict[str, Any],
    sample_models_yaml: Dict[str, Any],
    sample_features_yaml: Dict[str, Any],
    sample_logging_yaml: Dict[str, Any],
    sample_paths_yaml: Dict[str, Any],
) -> None:
    """Write minimal YAML config files to the temporary config directory."""
    configs = {
        "thresholds.yaml": sample_thresholds_yaml,
        "models.yaml": sample_models_yaml,
        "features.yaml": sample_features_yaml,
        "logging.yaml": sample_logging_yaml,
        "paths.yaml": sample_paths_yaml,
        "settings.yaml": {"environment": "test", "seed": 42},
        "alerts.yaml": {},
        "dashboard.yaml": {},
    }
    for name, data in configs.items():
        (tmp_config_dir / name).write_text(yaml.dump(data))


# ---------------------------------------------------------------------------
# ConfigLoader mock fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_config_loader(
    sample_thresholds_yaml: Dict[str, Any],
    sample_models_yaml: Dict[str, Any],
    sample_features_yaml: Dict[str, Any],
    sample_logging_yaml: Dict[str, Any],
) -> Generator[MagicMock, None, None]:
    """Patch get_config to return a mock ConfigLoader with test data."""
    cfg_map: Dict[str, Any] = {
        "thresholds": sample_thresholds_yaml,
        "models": sample_models_yaml,
        "features": sample_features_yaml,
        "logging": sample_logging_yaml,
        "paths": {
            "data": {"raw": "data/raw.csv"},
            "logs": {"dir": "logs/"},
            "alerts": {"dir": "alerts/"},
            "alerts_dir": "logs/alerts",
        },
        "settings": {"environment": "test"},
        "alerts": {},
        "dashboard": {},
    }

    mock_loader = MagicMock()
    mock_loader.load = MagicMock(side_effect=lambda name: cfg_map.get(name, {}))
    mock_loader.get = MagicMock(return_value=None)
    mock_loader.load_all = MagicMock(return_value=cfg_map)

    with patch("src.core.config.get_config", return_value=mock_loader), \
         patch("src.core.logger.get_config", return_value=mock_loader), \
         patch("src.classification.random_forest.get_config", return_value=mock_loader):
        yield mock_loader


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_flow_features() -> pd.DataFrame:
    """Return a DataFrame mimicking CICIDS2017 flow features (30 columns)."""
    rng = np.random.RandomState(42)
    n = 200
    columns = [
        "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Flow IAT Mean", "Flow IAT Std",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
        "Avg Fwd Segment Size", "Avg Bwd Segment Size",
        "Subflow Fwd Bytes", "Subflow Bwd Bytes",
        "Subflow Fwd Packets", "Subflow Bwd Packets",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "Fwd Packets/s", "Bwd Packets/s",
        "Packet Length Mean", "Packet Length Std",
        "Packet Length Variance", "Down/Up Ratio", "Average Packet Size",
    ]
    data = rng.rand(n, len(columns)) * 1000
    return pd.DataFrame(data, columns=columns)


@pytest.fixture()
def synthetic_benign_features(synthetic_flow_features: pd.DataFrame) -> pd.DataFrame:
    """Return only benign-like features (suitable for Isolation Forest training)."""
    return synthetic_flow_features.iloc[:150].copy()


@pytest.fixture()
def synthetic_attack_features(synthetic_flow_features: pd.DataFrame) -> pd.DataFrame:
    """Return features with some anomalous patterns (for classification testing)."""
    rng = np.random.RandomState(99)
    attack = synthetic_flow_features.iloc[150:].copy()
    # Make some rows look anomalous (high packet length, high flow duration)
    attack.iloc[:10, attack.columns.get_loc("Fwd Packet Length Mean")] = rng.uniform(5000, 10000, 10)
    attack.iloc[:10, attack.columns.get_loc("Flow Duration")] = rng.uniform(100000, 500000, 10)
    return attack


@pytest.fixture()
def synthetic_labels() -> pd.Series:
    """Return synthetic multi-class labels matching the flow features."""
    labels = ["BENIGN"] * 150 + ["DoS Hulk"] * 25 + ["DoS GoldenEye"] * 15 + ["DoS Slowloris"] * 10
    return pd.Series(labels, name="Label")


@pytest.fixture()
def synthetic_binary_labels() -> pd.Series:
    """Return binary labels (BENIGN vs ATTACK)."""
    labels = ["BENIGN"] * 150 + ["ATTACK"] * 50
    return pd.Series(labels, name="Label")


@pytest.fixture()
def synthetic_labeled_df(
    synthetic_flow_features: pd.DataFrame,
    synthetic_labels: pd.Series,
) -> pd.DataFrame:
    """Return a complete labeled DataFrame."""
    df = synthetic_flow_features.copy()
    df["Label"] = synthetic_labels.values
    return df


@pytest.fixture()
def synthetic_anomaly_scores() -> np.ndarray:
    """Return synthetic anomaly scores (0-1, higher = more anomalous)."""
    rng = np.random.RandomState(42)
    return rng.uniform(0.0, 1.0, 200)


@pytest.fixture()
def synthetic_rf_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Return synthetic RF predictions and probabilities."""
    rng = np.random.RandomState(42)
    n = 200
    classes = ["BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS Slowloris"]
    preds = rng.choice(classes, n)
    # Make first 100 look like correct predictions, rest noisy
    preds[:100] = "BENIGN"
    # Corresponding probability array
    proba = rng.dirichlet(np.ones(len(classes)), n)
    return preds, proba


# ---------------------------------------------------------------------------
# Alert fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_alert_dict() -> Dict[str, Any]:
    """Return a sample alert dictionary."""
    return {
        "timestamp": "2026-01-15T10:30:00",
        "alert_id": "alert-001",
        "severity": "high",
        "predicted_class": "DoS Hulk",
        "anomaly_score": 0.85,
        "classification_confidence": 0.92,
        "is_zero_day": False,
        "sample_id": "sample-001",
        "message": "Known attack detected: DoS Hulk",
    }


@pytest.fixture()
def sample_alert_dicts() -> list[Dict[str, Any]]:
    """Return a list of sample alert dictionaries."""
    return [
        {
            "timestamp": "2026-01-15T10:30:00",
            "alert_id": f"alert-{i:03d}",
            "severity": sev,
            "predicted_class": cls,
            "anomaly_score": score,
            "classification_confidence": conf,
            "is_zero_day": zd,
            "sample_id": f"sample-{i:03d}",
            "message": f"Alert {i}",
        }
        for i, (sev, cls, score, conf, zd) in enumerate([
            ("critical", "ZERO_DAY", 0.95, 0.3, True),
            ("high", "DoS Hulk", 0.85, 0.92, False),
            ("medium", "DoS GoldenEye", 0.6, 0.78, False),
            ("low", "BENIGN", 0.3, 0.6, False),
            ("high", "ZERO_DAY", 0.88, 0.4, True),
        ], start=1)
    ]


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def trained_isolation_forest(
    synthetic_benign_features: pd.DataFrame,
    mock_config_loader: MagicMock,
):
    """Return a trained IsolationForestModel instance."""
    from src.anomaly_detection.isolation_forest import IsolationForestModel

    model = IsolationForestModel()
    model.train(
        synthetic_benign_features,
        contamination=0.1,
        n_estimators=50,
        n_jobs=1,
        random_state=42,
    )
    return model


@pytest.fixture()
def trained_random_forest(
    synthetic_flow_features: pd.DataFrame,
    synthetic_labels: pd.Series,
    mock_config_loader: MagicMock,
):
    """Return a trained RandomForestClassifierModel instance."""
    from src.classification.random_forest import RandomForestClassifierModel

    model = RandomForestClassifierModel()
    model.train(
        synthetic_flow_features,
        synthetic_labels,
        n_estimators=50,
        n_jobs=1,
        random_state=42,
    )
    return model
