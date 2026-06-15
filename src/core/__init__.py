"""Core utilities: logging, configuration, shared infrastructure."""

from src.core.config import ConfigLoader, get_config, resolve_path
from src.core.constants import (
    CONFIG_DIR,
    DATA_DIR,
    DEFAULT_CONFIG_FILE,
    DEFAULT_N_JOBS,
    LOGS_DIR,
    MODELS_DIR,
    PACKAGE_NAME,
    PROJECT_ROOT,
    RANDOM_SEED,
    REPORTS_DIR,
    VERSION,
)
from src.core.exceptions import (
    AlertError,
    ConfigError,
    DataError,
    DashboardError,
    DetectionError,
    EvaluationError,
    ModelError,
    StreamingError,
)
from src.core.logger import setup_logging

__all__ = [
    # config
    "ConfigLoader",
    "get_config",
    "resolve_path",
    # constants
    "CONFIG_DIR",
    "DATA_DIR",
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_N_JOBS",
    "LOGS_DIR",
    "MODELS_DIR",
    "PACKAGE_NAME",
    "PROJECT_ROOT",
    "RANDOM_SEED",
    "REPORTS_DIR",
    "VERSION",
    # exceptions
    "AlertError",
    "ConfigError",
    "DataError",
    "DashboardError",
    "DetectionError",
    "EvaluationError",
    "ModelError",
    "StreamingError",
    # logger
    "setup_logging",
]
