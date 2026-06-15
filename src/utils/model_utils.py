from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

from src.core.exceptions import ModelError
from src.utils.file_utils import ensure_dir


def save_model(model: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    try:
        with open(p, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        raise ModelError(f"Failed to save model to {p}: {e}") from e
    return p


def load_model(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise ModelError(f"Model file not found: {p}")
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise ModelError(f"Failed to load model from {p}: {e}") from e


def model_params(model: Any) -> Dict[str, Any]:
    if hasattr(model, "get_params"):
        return model.get_params()
    raise ModelError(f"Model {type(model).__name__} does not expose get_params")


def model_summary(model: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"type": type(model).__name__}
    try:
        summary["params"] = model.get_params()
    except Exception:
        summary["params"] = {}
    if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
        summary["feature_importances"] = model.feature_importances_.tolist()
    if hasattr(model, "n_features_in_"):
        summary["n_features_in"] = model.n_features_in_
    if hasattr(model, "estimators_"):
        summary["n_estimators"] = len(model.estimators_)
    return summary
