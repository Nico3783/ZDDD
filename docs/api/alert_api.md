# Alert API

## Overview

The Alert API provides interfaces for generating, formatting, and exporting detection alerts. It encompasses the AlertManager, AlertLogger, and alert export modules (JSON and CSV loggers).

## AlertManager

### `AlertManager.process_detection_results()`

Main entry point for processing detection results into alerts.

```python
from src.detection_engine.alert_manager import AlertManager

manager = AlertManager(config)
alerts = manager.process_detection_results(
    flow_id="flow_001",
    anomaly_score=0.92,
    predicted_class="DoS Hulk",
    confidence=0.87,
    is_zero_day=False,
    features=flow_features_dict
)
```

**Parameters:**
- `flow_id` (str): Unique identifier for the network flow.
- `anomaly_score` (float): Isolation Forest anomaly score [0, 1].
- `predicted_class` (str): Random Forest predicted class label.
- `confidence` (float): RF prediction confidence [0, 1].
- `is_zero_day` (bool): Whether the detection is flagged as zero-day.
- `features` (dict): Original flow feature values.

**Returns:** List of `Alert` objects.

### `Alert` Object

```python
@dataclass
class Alert:
    alert_id: str          # UUID
    timestamp: datetime    # ISO 8601
    severity: str          # critical|high|medium|low
    attack_type: str       # known class or "zero_day"
    confidence: float      # Detection confidence
    anomaly_score: float   # Raw anomaly score
    source_ip: str         # Source IP (if available)
    dest_ip: str           # Destination IP (if available)
    flow_id: str           # Reference flow ID
    features: dict         # Relevant flow features
    metadata: dict         # Additional context
```

## AlertLogger

### `AlertLogger.log_alert()`

Persists an alert to the configured logging backend.

```python
from src.alerting.logger import AlertLogger

logger = AlertLogger(config)
logger.log_alert(alert)
```

**Actions:**
- Writes structured log entry to `logs/detection.log`.
- Routes alert to JSON and CSV loggers.
- Emits metrics for Prometheus integration.

## JSON Logger

### `JsonLogger`

Exports alerts in JSON Lines format for programmatic consumption.

```python
from src.alerting.json_logger import JsonLogger

json_logger = JsonLogger(output_path="alerts/alerts.json")
json_logger.write(alert)
```

**Output Format:**
```json
{
    "alert_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "timestamp": "2025-06-12T14:30:00.000Z",
    "severity": "critical",
    "attack_type": "zero_day",
    "confidence": 0.95,
    "anomaly_score": 0.93,
    "flow_id": "flow_001",
    "source_ip": "10.0.0.1",
    "dest_ip": "10.0.0.2",
    "features": {
        "flow_duration": 120.5,
        "fwd_pkt_count": 1500,
        "bwd_pkt_count": 200
    }
}
```

## CSV Logger

### `CsvLogger`

Exports alerts in CSV format for spreadsheet analysis.

```python
from src.alerting.csv_logger import CsvLogger

csv_logger = CsvLogger(output_path="alerts/alerts.csv")
csv_logger.write(alert)
```

**CSV Columns:**
`alert_id, timestamp, severity, attack_type, confidence, anomaly_score, flow_id, source_ip, dest_ip`

## Severity Levels

| Level | Anomaly Score | Zero-Day Boost | Response |
|-------|--------------|----------------|----------|
| `critical` | ≥ 0.9 | +0.2 if flagged | Immediate |
| `high` | ≥ 0.7 | +0.2 if flagged | Priority |
| `medium` | ≥ 0.5 | +0.2 if flagged | Standard |
| `low` | < 0.5 | N/A | Log only |
