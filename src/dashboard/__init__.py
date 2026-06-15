"""Dashboard module.

Streamlit-based real-time visualization of detection metrics
and alert statistics. Launch with ``streamlit run src/dashboard/app.py``.
"""

from src.dashboard.metrics import MetricsCalculator

__all__ = [
    "MetricsCalculator",
]
