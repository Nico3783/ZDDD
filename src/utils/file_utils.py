from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from src.core.exceptions import DataError


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise DataError(f"CSV file not found: {p}")
    try:
        return pd.read_csv(p, **kwargs)
    except Exception as e:
        raise DataError(f"Failed to read CSV {p}: {e}") from e


def read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise DataError(f"JSON file not found: {p}")
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as e:
        raise DataError(f"Failed to read JSON {p}: {e}") from e


def read_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise DataError(f"YAML file not found: {p}")
    try:
        with open(p, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise DataError(f"Failed to read YAML {p}: {e}") from e


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    try:
        df.to_csv(p, index=False, **kwargs)
    except Exception as e:
        raise DataError(f"Failed to write CSV {p}: {e}") from e
    return p


def write_json(data: Any, path: str | Path, **kwargs: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    try:
        with open(p, "w") as f:
            json.dump(data, f, **kwargs)
    except Exception as e:
        raise DataError(f"Failed to write JSON {p}: {e}") from e
    return p


def write_yaml(data: Dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    try:
        with open(p, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
    except Exception as e:
        raise DataError(f"Failed to write YAML {p}: {e}") from e
    return p


def get_file_size_mb(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        raise DataError(f"File not found: {p}")
    return p.stat().st_size / (1024 * 1024)


def list_files(
    directory: str | Path,
    pattern: str = "*",
    sort_by: str = "name",
) -> List[Path]:
    p = Path(directory)
    if not p.is_dir():
        raise DataError(f"Directory not found: {p}")
    files = list(p.glob(pattern))
    if sort_by == "name":
        files.sort(key=lambda f: f.name)
    elif sort_by == "size":
        files.sort(key=lambda f: f.stat().st_size)
    elif sort_by == "mtime":
        files.sort(key=lambda f: f.stat().st_mtime)
    return files
