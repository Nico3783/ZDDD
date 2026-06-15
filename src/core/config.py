from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from src.core.constants import CONFIG_DIR, DEFAULT_CONFIG_FILE
from src.core.exceptions import ConfigError


class ConfigLoader:
    """Loads and merges YAML configuration files from the config directory."""

    def __init__(self, config_dir: Path = CONFIG_DIR) -> None:
        self._config_dir = config_dir
        self._cache: Dict[str, Any] = {}

    def load(self, name: str) -> Dict[str, Any]:
        """Load a single config file by name (without .yaml extension)."""
        if name in self._cache:
            return self._cache[name]

        path = self._config_dir / f"{name}.yaml"
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        self._cache[name] = data
        return data

    def load_all(self) -> Dict[str, Any]:
        """Load and merge all config files into a single dict."""
        merged: Dict[str, Any] = {}
        for path in sorted(self._config_dir.glob("*.yaml")):
            name = path.stem
            merged[name] = self.load(name)
        return merged

    def get(self, *keys: str, default: Any = None) -> Any:
        """Deep-get a value from the merged config. Usage: get('models', 'isolation_forest', 'n_estimators')."""
        cfg = self._merged
        for key in keys:
            if isinstance(cfg, dict):
                cfg = cfg.get(key)
                if cfg is None:
                    return default
            else:
                return default
        return cfg

    @property
    def _merged(self) -> Dict[str, Any]:
        if not self._cache:
            return self.load_all()
        result: Dict[str, Any] = {}
        for name, data in self._cache.items():
            result[name] = data
        return result

    def clear_cache(self) -> None:
        self._cache.clear()


_config_loader: ConfigLoader | None = None


def get_config() -> ConfigLoader:
    """Return a singleton ConfigLoader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def resolve_path(key: str, *subpaths: str) -> Path:
    """Resolve a path from the paths config file, joined with subpaths."""
    cfg = get_config().load("paths")
    raw = cfg.get(key)
    if raw is None:
        raise ConfigError(f"Path key not found: {key}")
    base = Path(raw)
    if not base.is_absolute():
        base = Path(__file__).resolve().parents[2] / base
    os.makedirs(base, exist_ok=True)
    if subpaths:
        base = base / os.path.join(*subpaths)
        os.makedirs(base, exist_ok=True)
    return base
