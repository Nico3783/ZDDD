# Training Log

## Model Training Sessions

### Session 1: Isolation Forest Baseline

| Property | Value |
|----------|-------|
| **Date** | 2025-06-10 |
| **Model** | Isolation Forest |
| **Training data** | CICIDS2017 Wednesday (all classes) |
| **Training samples** | 2,264,594 |
| **Test samples** | 566,149 |
| **Features** | 30 selected flow-based features |

**Hyperparameters:**
```yaml
n_estimators: 100
max_samples: auto
contamination: 0.1
max_features: 1.0
bootstrap: false
random_state: 42
n_jobs: -1
```

**Results:**
| Metric | Value |
|--------|-------|
| Training time | 87 seconds |
| Model file size | 12.4 MB |
| Detection rate | 83.2% |
| False positive rate | 4.7% |

---

### Session 2: Isolation Forest (Benign-Only Training)

| Property | Value |
|----------|-------|
| **Date** | 2025-06-10 |
| **Model** | Isolation Forest |
| **Training data** | CICIDS2017 Wednesday (BENIGN only) |
| **Training samples** | 1,818,477 |
| **Test samples** | 566,149 (all classes) |
| **Features** | 30 selected flow-based features |

**Hyperparameters:**
```yaml
n_estimators: 150
max_samples: 256
contamination: 0.05
random_state: 42
```

**Results:**
| Metric | Value |
|--------|-------|
| Training time | 62 seconds |
| Detection rate | 88.5% |
| False positive rate | 3.2% |

---

### Session 3: Random Forest Classifier

| Property | Value |
|----------|-------|
| **Date** | 2025-06-11 |
| **Model** | Random Forest Classifier |
| **Training data** | CICIDS2017 Wednesday (all classes) |
| **Training samples** | 2,264,594 |
| **Test samples** | 566,149 |
| **Features** | 30 selected flow-based features |
| **Classes** | 6 (BENIGN + 5 DoS types) |

**Hyperparameters:**
```yaml
n_estimators: 150
criterion: gini
max_depth: null
min_samples_split: 2
min_samples_leaf: 1
max_features: sqrt
class_weight: balanced
random_state: 42
n_jobs: -1
```

**Results:**
| Metric | Value |
|--------|-------|
| Training time | 142 seconds |
| Model file size | 28.7 MB |
| Accuracy | 95.8% |
| Macro F1 | 93.4% |
| Weighted F1 | 95.6% |

---

### Session 4: Combined Pipeline

| Property | Value |
|----------|-------|
| **Date** | 2025-06-11 |
| **Pipeline** | IF + RF combined |
| **Anomaly detection** | Isolation Forest (Session 2) |
| **Classification** | Random Forest (Session 3) |
| **Threshold config** | Critical: 0.9, High: 0.7, Medium: 0.5 |

**Results:**
| Metric | Value |
|--------|-------|
| Zero-day detection rate | 86.3% |
| Known attack accuracy | 95.2% |
| Overall false positive rate | 3.8% |
| Average inference latency | 42 ms |

## Model Artifacts

| Model | File | Size | Date |
|-------|------|------|------|
| Isolation Forest | `models/isolation_forest.pkl` | 12.4 MB | 2025-06-10 |
| Random Forest | `models/random_forest.pkl` | 28.7 MB | 2025-06-11 |
| Scaler | `models/scaler.pkl` | 1.2 KB | 2025-06-10 |
| Encoder | `models/encoder.pkl` | 0.8 KB | 2025-06-10 |
