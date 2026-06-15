"""Unit tests for the core module: constants, config, exceptions, and logger."""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, mock_open

import pytest
import yaml

# src/core/constants.py


class TestConstants:
    """Tests for src.core.constants module."""

    def test_project_root_is_path(self):
        from src.core.constants import PROJECT_ROOT
        assert isinstance(PROJECT_ROOT, Path)

    def test_config_dir_is_path_and_child_of_project_root(self):
        from src.core.constants import PROJECT_ROOT, CONFIG_DIR
        assert isinstance(CONFIG_DIR, Path)
        assert CONFIG_DIR == PROJECT_ROOT / "config"

    def test_data_dir_is_path(self):
        from src.core.constants import DATA_DIR
        assert isinstance(DATA_DIR, Path)

    def test_models_dir_is_path(self):
        from src.core.constants import MODELS_DIR
        assert isinstance(MODELS_DIR, Path)

    def test_logs_dir_is_path(self):
        from src.core.constants import LOGS_DIR
        assert isinstance(LOGS_DIR, Path)

    def test_reports_dir_is_path(self):
        from src.core.constants import REPORTS_DIR
        assert isinstance(REPORTS_DIR, Path)

    def test_tests_dir_is_path(self):
        from src.core.constants import TESTS_DIR
        assert isinstance(TESTS_DIR, Path)

    def test_default_config_file_is_path(self):
        from src.core.constants import DEFAULT_CONFIG_FILE
        assert isinstance(DEFAULT_CONFIG_FILE, Path)

    def test_default_config_file_points_to_settings_yaml(self):
        from src.core.constants import DEFAULT_CONFIG_FILE, CONFIG_DIR
        assert DEFAULT_CONFIG_FILE == CONFIG_DIR / "settings.yaml"

    def test_random_seed_equals_42(self):
        from src.core.constants import RANDOM_SEED
        assert RANDOM_SEED == 42

    def test_default_n_jobs_equals_minus_one(self):
        from src.core.constants import DEFAULT_N_JOBS
        assert DEFAULT_N_JOBS == -1

    def test_package_name(self):
        from src.core.constants import PACKAGE_NAME
        assert isinstance(PACKAGE_NAME, str)
        assert PACKAGE_NAME == "zero_day_dos_detection"

    def test_version(self):
        from src.core.constants import VERSION
        assert isinstance(VERSION, str)
        assert VERSION == "0.1.0"

    def test_config_files_is_dict(self):
        from src.core.constants import CONFIG_FILES
        assert isinstance(CONFIG_FILES, dict)

    def test_config_files_has_expected_keys(self):
        from src.core.constants import CONFIG_FILES
        expected_keys = {
            "settings", "paths", "features", "models",
            "thresholds", "alerts", "logging", "dashboard",
        }
        assert set(CONFIG_FILES.keys()) == expected_keys

    def test_config_files_values_are_paths(self):
        from src.core.constants import CONFIG_FILES
        for key, value in CONFIG_FILES.items():
            assert isinstance(value, Path), f"CONFIG_FILES['{key}'] is not a Path"

    def test_config_files_values_end_with_yaml(self):
        from src.core.constants import CONFIG_FILES
        for key, value in CONFIG_FILES.items():
            assert value.suffix == ".yaml", f"CONFIG_FILES['{key}'] does not end with .yaml"

    def test_dirs_are_subdirectories_of_project_root(self):
        from src.core.constants import (
            PROJECT_ROOT, CONFIG_DIR, DATA_DIR, MODELS_DIR,
            LOGS_DIR, REPORTS_DIR, TESTS_DIR,
        )
        for d in [CONFIG_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR, TESTS_DIR]:
            assert str(d).startswith(str(PROJECT_ROOT)), (
                f"{d} is not under PROJECT_ROOT={PROJECT_ROOT}"
            )

    def test_data_dir_and_models_dir_are_valid_paths(self):
        from src.core.constants import DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR
        for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
            assert d.name in ("data", "models", "logs", "reports")

# src/core/config.py


class TestConfigLoader:
    """Tests for src.core.config.ConfigLoader and related utilities."""

    def test_load_returns_dict(self, tmp_config_dir: Path, write_config_files):
        from src.core.config import ConfigLoader
        loader = ConfigLoader(config_dir=tmp_config_dir)
        result = loader.load("thresholds")
        assert isinstance(result, dict)
        assert "anomaly_detection" in result

    def test_load_caches_result(self, tmp_config_dir: Path, write_config_files):
        from src.core.config import ConfigLoader
        loader = ConfigLoader(config_dir=tmp_config_dir)
        first = loader.load("thresholds")
        second = loader.load("thresholds")
        assert first is second

    def test_load_missing_file_raises_config_error(self, tmp_config_dir: Path):
        from src.core.config import ConfigLoader
        from src.core.exceptions import ConfigError
        loader = ConfigLoader(config_dir=tmp_config_dir)
        with pytest.raises(ConfigError, match="Config file not found"):
            loader.load("nonexistent")

    def test_load_all_returns_dict_of_dicts(self, tmp_config_dir: Path, write_config_files):
        from src.core.config import ConfigLoader
        loader = ConfigLoader(config_dir=tmp_config_dir)
        result = loader.load_all()
        assert isinstance(result, dict)
        assert len(result) > 0
        for key, value in result.items():
            assert isinstance(value, dict), f"load_all()['{key}'] is not a dict"

    def test_clear_cache_clears_internal_cache(self, tmp_config_dir: Path, write_config_files):
        from src.core.config import ConfigLoader
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("thresholds")
        assert "thresholds" in loader._cache
        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_clear_cache_allows_reload(self, tmp_config_dir: Path, write_config_files):
        from src.core.config import ConfigLoader
        loader = ConfigLoader(config_dir=tmp_config_dir)
        first = loader.load("thresholds")
        loader.clear_cache()
        second = loader.load("thresholds")
        assert first == second

    def test_config_dir_property(self, tmp_config_dir: Path):
        from src.core.config import ConfigLoader
        loader = ConfigLoader(config_dir=tmp_config_dir)
        assert loader._config_dir == tmp_config_dir

    def test_load_all_empty_dir(self, tmp_path: Path):
        from src.core.config import ConfigLoader
        empty_dir = tmp_path / "empty_config"
        empty_dir.mkdir()
        loader = ConfigLoader(config_dir=empty_dir)
        result = loader.load_all()
        assert result == {}

    def test_load_nonexistent_dir_raises(self, tmp_path: Path):
        from src.core.config import ConfigLoader
        from src.core.exceptions import ConfigError
        fake_dir = tmp_path / "does_not_exist"
        loader = ConfigLoader(config_dir=fake_dir)
        with pytest.raises(ConfigError, match="Config file not found"):
            loader.load("anything")


class TestGetConfig:
    """Tests for the get_config singleton factory."""

    def test_returns_config_loader_instance(self):
        from src.core.config import get_config, ConfigLoader
        loader = get_config()
        assert isinstance(loader, ConfigLoader)

    def test_returns_same_instance(self):
        from src.core.config import get_config
        first = get_config()
        second = get_config()
        assert first is second

    def test_singleton_not_recreated(self):
        from src.core.config import get_config
        first = get_config()
        first._cache["test_key"] = {"test": True}
        second = get_config()
        assert "test_key" in second._cache


class TestResolvePath:
    """Tests for src.core.config.resolve_path."""

    def test_resolve_path_returns_path(self, tmp_config_dir: Path, write_config_files, tmp_path: Path):
        from src.core.config import ConfigLoader
        loader = ConfigLoader(config_dir=tmp_config_dir)
        paths_cfg = tmp_config_dir / "paths.yaml"
        paths_cfg.write_text(yaml.dump({
            "data_dir": str(tmp_path / "resolved_data"),
        }))

        with patch("src.core.config.get_config", return_value=loader):
            from src.core.config import resolve_path
            result = resolve_path("data_dir")
            assert isinstance(result, Path)

    def test_resolve_path_missing_key_raises(self, tmp_config_dir: Path, write_config_files):
        from src.core.config import ConfigLoader
        from src.core.exceptions import ConfigError
        loader = ConfigLoader(config_dir=tmp_config_dir)
        paths_cfg = tmp_config_dir / "paths.yaml"
        paths_cfg.write_text(yaml.dump({"data": {"raw": "data/raw.csv"}}))

        with patch("src.core.config.get_config", return_value=loader):
            from src.core.config import resolve_path
            with pytest.raises(ConfigError, match="Path key not found"):
                resolve_path("nonexistent_key")

    def test_resolve_path_creates_directories(self, tmp_config_dir: Path, write_config_files, tmp_path: Path):
        from src.core.config import ConfigLoader
        loader = ConfigLoader(config_dir=tmp_config_dir)
        target = tmp_path / "new_dir"
        paths_cfg = tmp_config_dir / "paths.yaml"
        paths_cfg.write_text(yaml.dump({"target": str(target)}))

        with patch("src.core.config.get_config", return_value=loader):
            from src.core.config import resolve_path
            result = resolve_path("target")
            assert result.exists()

    def test_resolve_path_with_subpaths(self, tmp_config_dir: Path, write_config_files, tmp_path: Path):
        from src.core.config import ConfigLoader
        loader = ConfigLoader(config_dir=tmp_config_dir)
        base = tmp_path / "base_dir"
        paths_cfg = tmp_config_dir / "paths.yaml"
        paths_cfg.write_text(yaml.dump({"base": str(base)}))

        with patch("src.core.config.get_config", return_value=loader):
            from src.core.config import resolve_path
            result = resolve_path("base", "sub", "path")
            assert isinstance(result, Path)

# src/core/exceptions.py


class TestExceptions:
    """Tests for all custom exception classes."""

    ALL_EXCEPTIONS = [
        ("ConfigError", "config"),
        ("DataError", "data"),
        ("ModelError", "model"),
        ("StreamingError", "streaming"),
        ("AlertError", "alert"),
        ("DetectionError", "detection"),
        ("EvaluationError", "evaluation"),
        ("DashboardError", "dashboard"),
    ]

    @pytest.mark.parametrize("exc_name,expected_word", ALL_EXCEPTIONS)
    def test_exception_inherits_from_exception(self, exc_name: str, expected_word: str):
        from src.core import exceptions
        exc_cls = getattr(exceptions, exc_name)
        assert issubclass(exc_cls, Exception)

    @pytest.mark.parametrize("exc_name,expected_word", ALL_EXCEPTIONS)
    def test_exception_can_be_raised_and_caught(self, exc_name: str, expected_word: str):
        from src.core import exceptions
        exc_cls = getattr(exceptions, exc_name)
        with pytest.raises(exc_cls):
            raise exc_cls("something went wrong")

    @pytest.mark.parametrize("exc_name,expected_word", ALL_EXCEPTIONS)
    def test_exception_carries_message(self, exc_name: str, expected_word: str):
        from src.core import exceptions
        exc_cls = getattr(exceptions, exc_name)
        msg = f"Test message for {expected_word}"
        exc = exc_cls(msg)
        assert str(exc) == msg

    @pytest.mark.parametrize("exc_name,expected_word", ALL_EXCEPTIONS)
    def test_exception_can_be_caught_by_base_exception(self, exc_name: str, expected_word: str):
        from src.core import exceptions
        exc_cls = getattr(exceptions, exc_name)
        try:
            raise exc_cls("base catch test")
        except BaseException:
            pass  # Should not raise

    @pytest.mark.parametrize("exc_name,expected_word", ALL_EXCEPTIONS)
    def test_exception_without_message(self, exc_name: str, expected_word: str):
        from src.core import exceptions
        exc_cls = getattr(exceptions, exc_name)
        exc = exc_cls()
        assert str(exc) == ""

    def test_config_error_is_not_data_error(self):
        from src.core.exceptions import ConfigError, DataError
        assert not issubclass(ConfigError, DataError)
        assert not issubclass(DataError, ConfigError)

    def test_all_exceptions_have_docstrings(self):
        from src.core import exceptions
        for exc_name, _ in self.ALL_EXCEPTIONS:
            exc_cls = getattr(exceptions, exc_name)
            assert exc_cls.__doc__ is not None, f"{exc_name} has no docstring"

    def test_custom_exceptions_unique_classes(self):
        from src.core import exceptions
        classes = [getattr(exceptions, name) for name, _ in self.ALL_EXCEPTIONS]
        assert len(set(classes)) == len(classes), "Some exception classes are duplicated"

# src/core/logger.py


class TestGetLogger:
    """Tests for src.core.logger.get_logger."""

    def test_returns_logger(self, mock_config_loader):
        from src.core.logger import get_logger
        logger = get_logger("test_unit")
        assert isinstance(logger, logging.Logger)

    def test_returns_same_logger_for_same_name(self, mock_config_loader):
        from src.core.logger import get_logger
        l1 = get_logger("test_same_name")
        l2 = get_logger("test_same_name")
        assert l1 is l2

    def test_different_names_return_different_loggers(self, mock_config_loader):
        from src.core.logger import get_logger
        l1 = get_logger("test_name_a")
        l2 = get_logger("test_name_b")
        assert l1 is not l2

    def test_logger_has_handler(self, mock_config_loader):
        from src.core.logger import get_logger
        logger = get_logger("test_handler_check")
        assert len(logger.handlers) > 0

    def test_logger_name_is_set(self, mock_config_loader):
        from src.core.logger import get_logger
        logger = get_logger("my_custom_logger")
        assert logger.name == "my_custom_logger"

    def test_logger_not_propagated(self, mock_config_loader):
        from src.core.logger import get_logger
        logger = get_logger("test_no_propagate")
        assert logger.propagate is False


class TestSetupLogging:
    """Tests for src.core.logger.setup_logging."""

    def test_setup_returns_logger(self, mock_config_loader):
        from src.core.logger import setup_logging
        logger = setup_logging(name="test_setup_logging")
        assert isinstance(logger, logging.Logger)

    def test_setup_adds_stream_handler(self, mock_config_loader):
        from src.core.logger import setup_logging
        logger = setup_logging(name="test_stream_handler")
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_setup_with_custom_level(self, mock_config_loader):
        from src.core.logger import setup_logging
        logger = setup_logging(name="test_custom_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_sets_log_format(self, mock_config_loader):
        from src.core.logger import setup_logging
        logger = setup_logging(name="test_format")
        for handler in logger.handlers:
            if handler.formatter is not None:
                assert handler.formatter._fmt is not None

    def test_setup_idempotent(self, mock_config_loader):
        from src.core.logger import setup_logging
        name = "test_idempotent"
        l1 = setup_logging(name=name)
        handler_count_after_first = len(l1.handlers)
        l2 = setup_logging(name=name)
        assert len(l2.handlers) == handler_count_after_first

    def test_setup_with_file_handler(self, mock_config_loader, tmp_path: Path):
        from src.core.logger import setup_logging
        log_file = tmp_path / "test.log"
        logger = setup_logging(name="test_file_handler", log_file=log_file)
        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert log_file.exists()


class TestLoggerMixin:
    """Tests for src.core.logger.LoggerMixin."""

    def test_logger_mixin_exists(self):
        try:
            from src.core.logger import LoggerMixin
            assert callable(LoggerMixin)
        except ImportError:
            pytest.skip("LoggerMixin not yet implemented in src.core.logger")

    def test_logger_mixin_provides_logger_attribute(self):
        try:
            from src.core.logger import LoggerMixin
        except ImportError:
            pytest.skip("LoggerMixin not yet implemented in src.core.logger")

        class MyComponent(LoggerMixin):
            pass
        obj = MyComponent()
        assert hasattr(obj, "logger")
        assert isinstance(obj.logger, logging.Logger)

    def test_logger_mixin_logger_name_matches_class(self):
        try:
            from src.core.logger import LoggerMixin
        except ImportError:
            pytest.skip("LoggerMixin not yet implemented in src.core.logger")

        class SomeSpecificClass(LoggerMixin):
            pass
        obj = SomeSpecificClass()
        assert obj.logger.name == "SomeSpecificClass"

    def test_logger_mixin_multiple_instances_share_logger(self):
        try:
            from src.core.logger import LoggerMixin
        except ImportError:
            pytest.skip("LoggerMixin not yet implemented in src.core.logger")

        class SharedLoggerClass(LoggerMixin):
            pass
        obj1 = SharedLoggerClass()
        obj2 = SharedLoggerClass()
        assert obj1.logger is obj2.logger

    def test_logger_mixin_logger_has_handler(self):
        try:
            from src.core.logger import LoggerMixin
        except ImportError:
            pytest.skip("LoggerMixin not yet implemented in src.core.logger")

        class HandlerCheckClass(LoggerMixin):
            pass
        obj = HandlerCheckClass()
        assert len(obj.logger.handlers) > 0

    def test_logger_mixin_different_classes_different_loggers(self):
        try:
            from src.core.logger import LoggerMixin
        except ImportError:
            pytest.skip("LoggerMixin not yet implemented in src.core.logger")

        class ClassA(LoggerMixin):
            pass
        class ClassB(LoggerMixin):
            pass
        assert ClassA().logger is not ClassB().logger
