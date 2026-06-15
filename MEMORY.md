# MEMORY.md

## Project Context Memory

This file stores long-term implementation context.

Claude must continuously update this file as development progresses.

---

## Project Domain

Cybersecurity

Machine Learning

Network Intrusion Detection

Zero-Day Detection

Denial of Service Detection

---

## Core Detection Pipeline

Dataset
↓
Preprocessing
↓
Feature Engineering
↓
Isolation Forest
↓
Random Forest
↓
Alert Engine
↓
Evaluation Engine
↓
Dashboard

---

## Primary Dataset

CICIDS2017

Primary File:

Wednesday-workingHours.pcap_ISCX.csv

Important Labels:

* BENIGN
* DoS Hulk
* DoS GoldenEye
* DoS Slowloris
* DoS Slowhttptest
* Heartbleed

---

## Machine Learning Models

Anomaly Detection:

* Isolation Forest

Classification:

* Random Forest

---

## Key Success Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* False Positive Rate
* Detection Latency
* Throughput

---

## Architectural Principles

* Modular
* Testable
* Reproducible
* Explainable
* Real-time capable
* Research-aligned

---

## Current Status

### Completed (All Core Phases 1–11)
- Phase 1: `pyproject.toml`, `config/` (8 YAMLs), `src/core/`, `src/utils/`
- Phase 2: `src/datasets/` — loader, validator, profiler, splitter
- Phase 3: `src/preprocessing/` — cleaner, encoder, scaler; `src/features/` — selector, transformer, schema
- Phase 4: `src/anomaly_detection/` — isolation_forest, trainer, inference, threshold
- Phase 5: `src/classification/` — random_forest, trainer, inference, importance
- Phase 6: `src/detection_engine/` — engine, alert_manager, severity, decision_logic
- Phase 7: `src/streaming/` — streamer, scheduler, simulator, stream_reader
- Phase 8: `src/alerting/` — JSON/CSV logger, alert archival, formatter, notifier, generator
- Phase 9: `src/dashboard/` — metrics calculator, Streamlit app with cyber theme
- Phase 10: `src/evaluation/` — PerformanceEvaluator, LatencyTracker, ThroughputTracker, experiment reports
- Phase 11: Testing — 765 tests passing, 0 failures, 6 skipped

### Bug Fixes Found During Testing
- `severity.py`: Zero-day logic changed from "always critical" to score-boost (add 0.2, cap 1.0, then threshold)
- `severity.py`: Added `__init__` handling for nested config dicts (`min_score` extraction)
- `engine.py`: Added `numpy` import, fixed single-sample detection (was always 0.0), added `anomaly_rate` to `DetectionStats`
- `stream_reader.py`: Removed broken `DataCleaner`/`FeatureEncoder` imports

### File Population (Post-Testing)
- 9 Jupyter notebooks populated (01–09)
- 25 docs files populated (architecture, models, API, research, reports, project)
- README.md, CHANGELOG.md, LICENSE at root
- 4 deployment files (Dockerfile, docker-compose.yml, prometheus.yml, systemd service)
- 3 GitHub Actions workflows (tests.yml, lint.yml, build.yml)

### Scripts Fixed & Verified (All 12)
- `scripts/demo.py` — working end-to-end demo
- `scripts/evaluate_models.py` — rewritten with correct APIs, `_find_model()` for .pkl/.joblib
- `scripts/run_simulation.py` — rewritten using StreamSimulator, DetectionOrchestrator, AlertLogger
- `scripts/generate_report.py` — rewritten using src/evaluation/reports.py APIs
- `scripts/run_backtest.py` — created with sliding window backtest using IsolationForestModel
- `scripts/run_dashboard.py` — rewritten to launch Streamlit via subprocess
- `scripts/run_tests.py` — rewritten to shell out to pytest
- `scripts/train_iforest.py`, `scripts/train_random_forest.py`, `scripts/optimize_thresholds.py`, `scripts/preprocess_data.py`, `scripts/download_dataset.py` — verified OK

### Critical API Knowledge
- **Model files**: Actual files on disk are `.pkl` (not `.joblib`). `_find_model()` checks both extensions.
- `save_model()`/`load_model()` in `src/utils/model_utils.py` use `pickle.dump`/`pickle.load`
- `StreamSimulator(random_state=...)` with `.stream_from_dataframe(df, batch_size, delay_seconds, jitter)`
- `DetectionOrchestrator(engine, alert_logger)` with `.process_dataframe(df, return_details=True)` returns `{"alerts": [...]}`
- `DetectionEngine(anomaly_model=..., classifier_model=...)` — takes model instances directly
- `AlertLogger(log_dir=...)` in `src/alerting/logger.py` (NOT `src/logging_engine/`)
- `AlertGenerator` in `src/alerting/generator.py`
- `PerformanceEvaluator` in `src/evaluation/metrics.py` with `.evaluate(y_true, y_pred)`
- `generate_experiment_report(experiment_name, y_true, y_pred, metadata)` in `src/evaluation/reports.py`
- `create_dashboard()` in `src/dashboard/app.py` — launches Streamlit app
- No `src/simulation/` module — streaming is in `src/streaming/simulator.py`
- No `src/dashboard/report.py` — reporting is in `src/evaluation/reports.py`

### Remaining
- None — all phases complete

---

## Environment & Dependencies (Last Updated: 2026-06-12)

- **venv**: `/home/nick/Documents/Projects/Zion/zero-day-dos-detection-engine/venv` — Python 3.13.12
- **Install**: `venv/bin/pip install -r requirements.txt` (core) or `requirements-dev.txt` (with pytest, ruff, mypy)
- **85 packages installed** in venv
- **NumPy 2.x fix**: `np.trapz` → `np.trapezoid` in `src/evaluation/roc.py` and `src/evaluation/metrics.py`
- **Test suite**: 755 passing (721 unit + 34 integration, 6 skipped) — run with `venv/bin/python -m pytest tests/ -o "addopts="`
- **.gitignore** created — covers venv, __pycache__, data/raw CSVs, models/trained, reports, logs, .env, IDE files
