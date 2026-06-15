from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

from src.core.config import get_config


def setup_logging(
    name: str = "zero_day_dos",
    level: int | None = None,
    log_file: str | Path | None = None,
    stream: TextIO = sys.stdout,
) -> logging.Logger:
    """Configure and return a logger with both stream and optional file handlers.

    Settings are read from config/logging.yaml unless overridden.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    cfg = get_config().load("logging")
    effective_level = level or _resolve_level(cfg.get("level", "INFO"))
    logger.setLevel(effective_level)

    fmt = cfg.get(
        "format",
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    datefmt = cfg.get("date_format", "%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    stream_handler = logging.StreamHandler(stream)
    stream_handler.setLevel(effective_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = log_file or cfg.get("file")
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(effective_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def _resolve_level(level_str: str) -> int:
    return getattr(logging, level_str.upper(), logging.INFO)


def get_logger(name: str = "zero_day_dos") -> logging.Logger:
    """Return an existing logger or set one up with defaults."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logging(name=name)
    return logger
