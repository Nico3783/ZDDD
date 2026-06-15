# Deployment Design

## Deployment Architecture

The system is designed for local deployment on a single machine, packaged as a Docker container for reproducibility and portability.

## Docker Container

### Container Structure

```
zero-day-dos-detector/
├── Dockerfile
├── docker-compose.yml
├── src/                    # Application source code
├── config/                 # Configuration files
├── models/                 # Trained model artifacts
├── data/                   # Dataset files
├── logs/                   # Application logs
├── alerts/                 # Alert output files
└── requirements.txt        # Python dependencies
```

### Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/
COPY models/ models/

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "src/dashboard/app.py"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  detection-engine:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./alerts:/app/alerts
    environment:
      - LOG_LEVEL=INFO
      - DETECTION_THRESHOLD=0.9
```

## Systemd Service

For non-containerized deployment, a systemd service unit is provided:

```ini
[Unit]
Description=Zero-Day DoS Detection Engine
After=network.target

[Service]
Type=simple
User=detector
WorkingDirectory=/opt/zero-day-dos-detection-engine
ExecStart=/opt/venv/bin/python -m src.detection_engine.orchestrator
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Monitoring

### Prometheus Metrics

Key metrics exposed at `localhost:9090/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `flows_processed_total` | Counter | Total flows processed |
| `alerts_generated_total` | Counter | Total alerts generated |
| `detection_latency_seconds` | Histogram | Per-flow detection latency |
| `model_accuracy` | Gauge | Current model accuracy |
| `active_alerts` | Gauge | Number of active alerts |

### Grafana Dashboards

Pre-configured dashboards for:
- **Detection Overview**: Alert rates, severity distribution, flow throughput.
- **Model Performance**: Accuracy, precision, recall trends over time.
- **System Health**: CPU usage, memory consumption, processing latency.

## Directory Layout

```
/opt/zero-day-dos-detection-engine/
├── config/
│   ├── settings.yaml        # Main configuration
│   ├── logging.yaml         # Logging configuration
│   ├── models.yaml          # Model hyperparameters
│   └── features.yaml        # Feature definitions
├── models/
│   ├── isolation_forest.pkl  # Trained IF model
│   └── random_forest.pkl     # Trained RF model
├── data/
│   ├── raw/                  # Original CICIDS2017 CSVs
│   └── processed/            # Preprocessed datasets
├── logs/
│   ├── detection.log         # Detection events
│   └── errors.log            # Error log
├── alerts/
│   ├── alerts.json           # JSON alert log
│   └── alerts.csv            # CSV alert log
└── src/                      # Application code
```
