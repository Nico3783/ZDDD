# Zero-Day DoS Detection Engine

Real-Time Autonomous Anomaly Detection Engine for Identifying Zero-Day Exploits in Denial of Service (DoS) Attacks Using Machine Learning.

**Author:** Oyelude Zion Clifford (CYS/20/4940)
**Supervisor:** Prof. Alowolodu
**Institution:** Federal University of Technology Akure (FUTA)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Simulation](#simulation)
- [Dashboard](#dashboard)
- [Demo](#demo)
- [Testing](#testing)
- [Scripts Reference](#scripts-reference)
- [Project Structure](#project-structure)

---

## Overview

This system detects zero-day DoS attacks in network traffic using a two-stage machine learning pipeline:

1. **Isolation Forest** — unsupervised anomaly detection to identify novel/unknown attack patterns
2. **Random Forest** — supervised multi-class classification to categorize known attack types

The pipeline processes CICIDS2017 network traffic data, extracts 78 flow-based features, trains both models, and provides real-time detection via streaming simulation, backtesting evaluation, and a Streamlit dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                        │
│  Raw CSV → Preprocessing → Feature Extraction → Split   │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
    ┌──────────▼──────────┐ ┌────────▼──────────────┐
    │   ISOLATION FOREST  │ │    RANDOM FOREST      │
    │   (Anomaly Det.)    │ │    (Classifier)       │
    │   • train()         │ │    • train()          │
    │   • predict()       │ │    • predict()        │
    │   • score_samples() │ │    • predict_proba()  │
    └──────────┬──────────┘ └────────┬──────────────┘
               │                      │
    ┌──────────▼──────────────────────▼──────────────┐
    │              DETECTION ENGINE                  │
    │  DetectionEngine → DetectionOrchestrator       │
    │  • detect_batch() • process_dataframe()        │
    └──────────┬──────────────────────┬──────────────┘
               │                      │
    ┌──────────▼──────────┐ ┌────────▼──────────────┐
    │  STREAMING SIMULATOR│ │    ALERT SYSTEM       │
    │  • stream_from_df() │ │    • AlertGenerator   │
    │  • generate_traffic │ │    • AlertLogger      │
    └─────────────────────┘ └───────────────────────┘
               │                      │
    ┌──────────▼──────────────────────▼──────────────┐
    │           EVALUATION & REPORTING               │
    │  PerformanceEvaluator • LatencyTracker         │
    │  ThroughputTracker • Experiment Reports        │
    └────────────────────────────────────────────────┘
```

---

## Prerequisites

- **Python 3.11+**
- **uv** (package manager) — [install](https://docs.astral.sh/uv/getting-started/installation/)
- **Git**
- ~2GB disk space for CICIDS2017 dataset

---

## Environment Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd zero-day-dos-detection-engine
```

### 2. Create virtual environment and install dependencies

```bash
# Using uv (recommended)
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Verify installation

```bash
python -c "import src; print('Package OK')"
python -m pytest tests/ -o "addopts=" --tb=short -q
# Expected: 765 passed, 6 skipped
```

---

## Dataset

### CICIDS2017

The system uses the **CICIDS2017** dataset for training and evaluation.

### Download

```bash
python scripts/download_dataset.py
```

This downloads the raw CSV files to `data/raw/cicids2017/`. The primary file used is:

- `Wednesday-workingHours.pcap_ISCX.csv` (~225MB)

### Dataset contents

| File | Description |
|------|-------------|
| `Monday-WorkingHours.pcap_ISCX.csv` | Normal traffic baseline |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | FTP-Patator, SSH-Patator |
| `Wednesday-workingHours.pcap_ISCX.csv` | **Primary file** — DoS Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Web attacks |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Infiltration |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Bot, PortScan |
| `Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv` | DDoS |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | PortScan |

### Labels

The system focuses on these DoS attack types:

- `BENIGN` — normal traffic
- `DoS Hulk` — HTTP flood
- `DoS GoldenEye` — HTTP keepalive flood
- `DoS Slowloris` — slow HTTP connections
- `DoS Slowhttptest` — slow POST body
- `Heartbleed` — OpenSSL vulnerability

---

## Preprocessing

```bash
python scripts/preprocess_data.py
```

This performs:
- Loading raw CSV data
- Dropping duplicates
- Handling missing/infinite values
- Feature scaling (StandardScaler)
- Encoding labels (LabelEncoder)
- Train/validation/test split (80/10/10)
- Saving processed splits to `data/processed/`

---

## Training

### Train Isolation Forest (Anomaly Detector)

```bash
python scripts/train_iforest.py
```

- Trains on normal (BENIGN) traffic only
- Contamination: 0.1 (10% expected anomalies)
- Saves model to `models/trained/isolation_forest.pkl`

### Train Random Forest (Classifier)

```bash
python scripts/train_random_forest.py
```

- Trains on all labeled traffic (BENIGN + attack types)
- 150 estimators, max_depth=20, balanced class weights
- Saves model to `models/trained/random_forest.pkl`

### Train both models

```bash
python scripts/train_iforest.py && python scripts/train_random_forest.py
```

---

## Evaluation

### Evaluate trained models

```bash
python scripts/evaluate_models.py --verbose
```

Reports accuracy, precision, recall, F1-score, confusion matrix, and per-class metrics.

### Generate comprehensive report

```bash
python scripts/generate_report.py --verbose
```

Generates JSON evaluation reports saved to `reports/evaluation/`.

### Backtesting

```bash
python scripts/run_backtest.py --window-size 100 --verbose
```

Evaluates models using a sliding window approach on historical data to measure real-time detection performance.

---

## Simulation

### Run streaming detection simulation

```bash
python scripts/run_simulation.py --duration 60 --batch-size 100 --verbose
```

Simulates real-time network traffic streaming:
- Loads trained models
- Streams data in configurable batches
- Detects anomalies and generates alerts
- Reports latency and throughput metrics

### Optimize detection thresholds

```bash
python scripts/optimize_thresholds.py
```

Optimizes the anomaly detection threshold using ROC curve analysis.

---

## Dashboard

### Launch Streamlit dashboard

```bash
python scripts/run_dashboard.py --port 8501
```

Opens the interactive dashboard at `http://localhost:8501` with:
- Real-time detection visualization
- Model performance metrics
- Alert history and statistics
- Traffic analysis plots

---

## Demo

### Quick demo (end-to-end)

```bash
python scripts/demo.py
```

This runs a complete end-to-end demonstration:
1. Loads raw CICIDS2017 data (2000 samples)
2. Extracts and preprocesses features
3. Trains Isolation Forest (anomaly detection)
4. Trains Random Forest (multi-class classification)
5. Evaluates both models
6. Generates experiment report
7. Saves report to `reports/`

### Full pipeline

```bash
# 1. Download data
python scripts/download_dataset.py

# 2. Preprocess
python scripts/preprocess_data.py

# 3. Train models
python scripts/train_iforest.py
python scripts/train_random_forest.py

# 4. Evaluate
python scripts/evaluate_models.py --verbose

# 5. Backtest
python scripts/run_backtest.py --verbose

# 6. Generate report
python scripts/generate_report.py --verbose

# 7. Run simulation
python scripts/run_simulation.py --duration 30 --verbose

# 8. Launch dashboard
python scripts/run_dashboard.py
```

---

## Testing

### Run full test suite

```bash
python scripts/run_tests.py --verbose
```

### Run specific test categories

```bash
# Unit tests only
python scripts/run_tests.py --unit-only

# Integration tests only
python scripts/run_tests.py --integration-only

# Module-specific tests
python scripts/run_tests.py --module anomaly_detection
python scripts/run_tests.py --module classification
python scripts/run_tests.py --module evaluation
```

### Run pytest directly

```bash
python -m pytest tests/ -o "addopts=" --tb=short -v
```

---

## Scripts Reference

| Script | Description | Key Arguments |
|--------|-------------|---------------|
| `scripts/demo.py` | End-to-end demonstration | — |
| `scripts/download_dataset.py` | Download CICIDS2017 dataset | — |
| `scripts/preprocess_data.py` | Preprocess raw data | — |
| `scripts/train_iforest.py` | Train Isolation Forest | — |
| `scripts/train_random_forest.py` | Train Random Forest | — |
| `scripts/evaluate_models.py` | Evaluate trained models | `--data`, `--models`, `--verbose` |
| `scripts/generate_report.py` | Generate evaluation report | `--data`, `--models`, `--output`, `--verbose` |
| `scripts/run_backtest.py` | Run backtesting evaluation | `--data`, `--models`, `--window-size`, `--verbose` |
| `scripts/run_simulation.py` | Run streaming simulation | `--data`, `--duration`, `--batch-size`, `--verbose` |
| `scripts/run_dashboard.py` | Launch Streamlit dashboard | `--port`, `--host`, `--verbose` |
| `scripts/run_tests.py` | Run test suite | `--verbose`, `--unit-only`, `--integration-only`, `--module` |
| `scripts/optimize_thresholds.py` | Optimize detection thresholds | — |

---

```

---

## Configuration

All configuration is in `config/` as YAML files:

- **settings.yaml** — environment, seed, streaming, pipeline options
- **paths.yaml** — file paths for data, models, results, logs
- **models.yaml** — hyperparameters for Isolation Forest and Random Forest
- **thresholds.yaml** — anomaly detection thresholds

---

## License

Academic use only — Federal University of Technology Akure.
