# Component Diagram

## Module Relationships

The following diagram illustrates the inter-module dependencies and data flow within the detection engine.

```
                            ┌─────────────────┐
                            │   config/       │
                            │   settings.yaml │
                            └────────┬────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                 │
              ┌─────┴─────┐   ┌─────┴─────┐   ┌──────┴──────┐
              │   core/    │   │  utils/   │   │  datasets/  │
              │ config.py  │   │ data_     │   │  loader.py  │
              │ constants  │   │ utils.py  │   │  validator  │
              │ exceptions │   │ file_     │   │  profiler   │
              │ logger.py  │   │ utils.py  │   │  splitter   │
              └─────┬──────┘   └─────┬─────┘   └──────┬──────┘
                    │                │                 │
                    └────────────────┼─────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                       │
     ┌────────┴────────┐   ┌───────┴────────┐   ┌─────────┴────────┐
     │  preprocessing/ │   │   features/     │   │ anomaly_detection/│
     │  cleaner.py     │   │  selector.py    │   │  isolation_forest │
     │  encoder.py     │   │  transformer    │   │  trainer.py       │
     │  scaler.py      │   │  importance.py  │   │  inference.py     │
     │  imputer.py     │   │  engineering    │   │  threshold.py     │
     │  pipeline.py    │   │  schema.py      │   │  evaluator.py     │
     └────────┬────────┘   └───────┬────────┘   └─────────┬────────┘
              │                     │                       │
              └─────────────────────┼───────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │                │
           ┌────────┴────────┐  ┌──┴──────────┐  ┌──┴───────────┐
           │ classification/ │  │ detection_   │  │  streaming/  │
           │ random_forest   │  │ engine/      │  │  streamer.py │
           │ trainer.py      │  │ engine.py    │  │  scheduler   │
           │ inference.py    │  │ alert_mgr    │  │  dispatcher  │
           │ importance.py   │  │ severity.py  │  │  microbatch  │
           │ evaluator.py    │  │ decision.py  │  │  simulator   │
           └────────┬────────┘  │ orchestrator │  │  stream_rdr  │
                    │           └──────┬───────┘  └──────────────┘
                    │                  │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    │    alerting/    │
                    │  logger.py      │
                    │  formatter.py   │
                    │  generator.py   │
                    │  notifier.py    │
                    │  json_logger    │
                    │  csv_logger     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │   evaluation/   │
                    │  metrics.py     │
                    │  reports.py     │
                    │  confusion.py   │
                    │  roc.py         │
                    │  latency.py     │
                    │  throughput.py  │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │   dashboard/    │
                    │  Streamlit UI   │
                    │  metrics.py     │
                    │  pages/         │
                    └─────────────────┘
```

## Dependency Rules

- `core/` has no internal project dependencies; it is the foundation module.
- `utils/` depends only on `core/`.
- `datasets/` depends on `core/` and `utils/`.
- `preprocessing/` depends on `core/`, `utils/`, and `datasets/`.
- `features/` depends on `preprocessing/` and `datasets/`.
- `anomaly_detection/` and `classification/` depend on `features/` and `datasets/`.
- `detection_engine/` orchestrates all upstream modules.
- `alerting/`, `evaluation/`, and `dashboard/` are downstream consumers.
