from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TESTS_DIR = PROJECT_ROOT / "tests"

DEFAULT_CONFIG_FILE = CONFIG_DIR / "settings.yaml"

CONFIG_FILES = {
    "settings": CONFIG_DIR / "settings.yaml",
    "paths": CONFIG_DIR / "paths.yaml",
    "features": CONFIG_DIR / "features.yaml",
    "models": CONFIG_DIR / "models.yaml",
    "thresholds": CONFIG_DIR / "thresholds.yaml",
    "alerts": CONFIG_DIR / "alerts.yaml",
    "logging": CONFIG_DIR / "logging.yaml",
    "dashboard": CONFIG_DIR / "dashboard.yaml",
}

RANDOM_SEED = 42
DEFAULT_N_JOBS = -1

PACKAGE_NAME = "zero_day_dos_detection"
VERSION = "0.1.0"
