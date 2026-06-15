# System Architecture Overview

## High-Level Architecture

The zero-day DoS detection engine follows a modular pipeline architecture with clear separation of concerns. Each stage of the pipeline is implemented as an independent module, enabling isolated development, testing, and modification.

```
┌───────────────────────┐
│                        DETECTION ENGINE                             │
├──────────┬──────────┬──────────┬──────────┬──────────┬────────────┤
│  Data    │ Preproc- │ Feature  │ Anomaly  │ Classif- │ Detection  │
│ Ingestion│ essing   │ Engineer │ Detection│ ication  │ Engine     │
├──────────┼──────────┼──────────┼──────────┼──────────┼────────────┤
│ datasets/│ preproc- │ features/│ anomaly_ │ classif- │ detection_ │
│          │ essing/  │          │ detection│ ication/ │ engine/    │
└──────────┴──────────┴──────────┴──────────┴──────────┴────────────┘
                               │
                    ┌──────────┴──────────┐
                    │     Alerting        │
                    │     Dashboard       │
                    └─────────────────────┘
```

## Module Inventory

| Module | Location | Responsibility |
|--------|----------|---------------|
| `core/` | `src/core/` | Configuration, constants, exceptions, logging |
| `utils/` | `src/utils/` | Data utilities, file operations, validation |
| `datasets/` | `src/datasets/` | Data loading, validation, profiling, splitting |
| `preprocessing/` | `src/preprocessing/` | Cleaning, encoding, scaling, imputation |
| `features/` | `src/features/` | Feature selection, transformation, engineering |
| `anomaly_detection/` | `src/anomaly_detection/` | Isolation Forest training, inference, thresholds |
| `classification/` | `src/classification/` | Random Forest training, inference, importance |
| `detection_engine/` | `src/detection_engine/` | Orchestration, decision logic, alerting |
| `streaming/` | `src/streaming/` | Real-time simulation, scheduling, dispatching |
| `alerting/` | `src/alerting/` | Alert generation, formatting, logging, export |
| `evaluation/` | `src/evaluation/` | Metrics, reports, confusion matrices, ROC |
| `dashboard/` | `src/dashboard/` | Streamlit interface, real-time monitoring |

## Design Principles

1. **Modularity**: Each module has a single responsibility and can be tested independently.
2. **Composability**: Modules are combined through the pipeline orchestrator to form complete workflows.
3. **Configurability**: Model parameters, thresholds, and pipeline settings are centralized in configuration.
4. **Observability**: Structured logging and metrics collection throughout the pipeline.
5. **Reproducibility**: Deterministic configurations ensure consistent results across runs.
