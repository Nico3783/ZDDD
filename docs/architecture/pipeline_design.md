# Pipeline Design

## Overview

The system implements three distinct pipelines, each serving a specific purpose in the detection workflow. Pipelines are composable sequences of processing stages that transform data from raw input to actionable alerts.

## Pipeline 1: Batch Preprocessing Pipeline

Used during model training and evaluation. Processes the complete CICIDS2017 dataset offline.

```
Raw CSV → Cleaner → Encoder → Imputer → Scaler → Splitter → Model Training
```

### Stages

1. **Cleaner** (`preprocessing/cleaner.py`): Removes duplicates, handles missing values, filters invalid flows.
2. **Encoder** (`preprocessing/encoder.py`): Converts categorical labels to numerical representations using label encoding.
3. **Imputer** (`preprocessing/imputer.py`): Fills missing feature values using mean/median strategies.
4. **Scaler** (`preprocessing/scaler.py`): Normalizes features to standard scale (zero mean, unit variance).
5. **Splitter** (`datasets/splitter.py`): Divides data into training (80%) and test (20%) sets with stratification.

## Pipeline 2: Feature Engineering Pipeline

Transforms raw features into model-ready representations.

```
Encoded Data → Feature Selection → Feature Transformation → Feature Extraction
```

### Stages

1. **Feature Selection** (`features/selector.py`): Selects 30 most relevant flow-based features from the CICIDS2017 feature set.
2. **Feature Transformation** (`features/transformer.py`): Applies PCA or other dimensionality reduction techniques.
3. **Feature Importance** (`features/importance.py`): Computes and ranks feature importance using mutual information and model-based methods.

## Pipeline 3: Real-Time Detection Pipeline

Processes simulated streaming traffic through the complete detection workflow.

```
StreamSimulator → MicroBatch → DetectionEngine → AlertManager → Dashboard
```

### Stages

1. **StreamSimulator** (`streaming/simulator.py`): Reads CSV data row-by-row with configurable delays.
2. **MicroBatch** (`streaming/microbatch.py`): Accumulates flows into time-windowed batches.
3. **DetectionEngine** (`detection_engine/engine.py`): Runs anomaly detection and classification.
4. **AlertManager** (`detection_engine/alert_manager.py`): Generates and routes alerts.
5. **Dashboard** (`dashboard/`): Displays real-time metrics and alerts.

## Pipeline Configuration

All pipelines are configured through `config/settings.yaml`:

```yaml
pipeline:
  batch:
    chunk_size: 10000
    parallel_workers: 4
  streaming:
    batch_interval_seconds: 1.0
    max_batch_size: 100
  detection:
    anomaly_threshold: 0.9
    zero_day_boost: 0.2
```
