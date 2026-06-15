# Dashboard API

## Overview

The Dashboard API provides the Streamlit-based monitoring interface for visualizing detection results, model performance, and system health in real-time.

## Streamlit Application

### Entry Point

```python
# src/dashboard/app.py
import streamlit as st
from src.dashboard.metrics import MetricsCalculator

st.set_page_config(
    page_title="Zero-Day DoS Detection Engine",
    page_icon="🛡️",
    layout="wide"
)

# Main dashboard layout
col1, col2, col3 = st.columns(3)
```

### Page Structure

| Page | Path | Description |
|------|------|-------------|
| **Main Dashboard** | `pages/dashboard.py` | Real-time detection overview |
| **Model Performance** | `pages/performance.py` | Metrics, ROC, confusion matrix |
| **Alert History** | `pages/alerts.py` | Alert log browser |
| **Data Explorer** | `pages/data.py` | Dataset inspection |
| **System Health** | `pages/health.py` | Pipeline status, latency |

## MetricsCalculator

### `MetricsCalculator`

Computes real-time dashboard metrics from detection results.

```python
from src.dashboard.metrics import MetricsCalculator

calc = MetricsCalculator()

# Summary metrics
metrics = calc.get_summary_metrics()
# Returns:
# {
#    "total_flows": 15234,
#    "total_alerts": 342,
#    "critical_alerts": 28,
#    "detection_rate": 0.87,
#    "false_positive_rate": 0.03,
#    "avg_latency_ms": 45.2,
#    "throughput_fps": 1250
# }

# Time-series data for charts
timeseries = calc.get_alert_timeseries(interval="1min")

# Severity distribution
severity_dist = calc.get_severity_distribution()

# Per-class metrics
class_metrics = calc.get_class_metrics()
```

### Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_summary_metrics()` | dict | Aggregate detection statistics |
| `get_alert_timeseries(interval)` | DataFrame | Alerts over time |
| `get_severity_distribution()` | dict | Alert counts by severity |
| `get_class_metrics()` | DataFrame | Per-class precision/recall/F1 |
| `get_latency_distribution()` | list | Inference latency histogram |
| `get_roc_data()` | dict | ROC curve points for plotting |
| `get_confusion_matrix()` | ndarray | Confusion matrix values |

## Dashboard Components

### Real-Time Metrics Cards

```python
with col1:
    st.metric("Total Flows", f"{metrics['total_flows']:,}")
with col2:
    st.metric("Alerts", metrics['total_alerts'],
              delta=f"{metrics['critical_alerts']} critical")
with col3:
    st.metric("Detection Rate", f"{metrics['detection_rate']:.1%}")
```

### Charts

| Chart | Library | Purpose |
|-------|---------|---------|
| Alert Timeline | Plotly | Alerts over time with severity coloring |
| ROC Curve | Plotly | Model ROC with AUC annotation |
| Confusion Matrix | Plotly Heatmap | Per-class classification accuracy |
| Feature Importance | Plotly Bar | Top 10 features by importance |
| Severity Gauge | Plotly Gauge | Current alert severity level |
| Latency Histogram | Plotly Bar | Inference time distribution |

### Alert Table

```python
st.dataframe(
    alert_df[["timestamp", "severity", "attack_type", "confidence", "flow_id"]],
    use_container_width=True
)
```

## Configuration

Dashboard settings in `config/settings.yaml`:

```yaml
dashboard:
  port: 8501
  refresh_interval: 2
  theme: dark
  max_alerts_display: 1000
  chart_height: 400
```
