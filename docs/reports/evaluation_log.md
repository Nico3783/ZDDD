# Evaluation Log

## Evaluation Sessions

### Evaluation 1: Isolation Forest Performance

| Property | Value |
|----------|-------|
| **Date** | 2025-06-10 |
| **Model** | Isolation Forest |
| **Test set size** | 566,149 flows |
| **Positive class** | Attack (all DoS types) |
| **Negative class** | BENIGN |

**Classification Report:**
```
              precision    recall  f1-score   support

      BENIGN      0.953     0.968     0.960    454,620
      ATTACK      0.872     0.832     0.852    111,529

    accuracy                          0.941    566,149
   macro avg      0.912     0.900     0.906    566,149
weighted avg      0.937     0.941     0.939    566,149
```

**Confusion Matrix:**
```
                Predicted
              Benign   Attack
Actual Benign 440,171  14,449
Actual Attack  18,737   92,792
```

**Metrics:**
| Metric | Value |
|--------|-------|
| Accuracy | 94.1% |
| Precision (attack) | 87.2% |
| Recall (attack) | 83.2% |
| F1 (attack) | 85.2% |
| ROC-AUC | 0.953 |
| False Positive Rate | 3.2% |

---

### Evaluation 2: Random Forest Multi-Class

| Property | Value |
|----------|-------|
| **Date** | 2025-06-11 |
| **Model** | Random Forest |
| **Test set size** | 566,149 flows |
| **Classes** | 6 |

**Per-Class Results:**
```
                 precision    recall  f1-score   support

         BENIGN      0.986     0.992     0.989    454,620
      DoS Hulk      0.973     0.981     0.977     46,215
 DoS GoldenEye      0.954     0.943     0.948      2,059
DoS Slowloris      0.941     0.928     0.934      1,159
DoS Slowhttptest   0.932     0.918     0.925      1,100
     Heartbleed     1.000     1.000     1.000         2

      accuracy                          0.958    566,149
     macro avg      0.964     0.960     0.962    566,149
  weighted avg      0.958     0.958     0.958    566,149
```

**Metrics:**
| Metric | Value |
|--------|-------|
| Accuracy | 95.8% |
| Macro F1 | 96.2% |
| Weighted F1 | 95.8% |
| Cohen's Kappa | 0.941 |
| MCC | 0.938 |

---

### Evaluation 3: Combined Pipeline (IF + RF)

| Property | Value |
|----------|-------|
| **Date** | 2025-06-11 |
| **Pipeline** | IF anomaly detection + RF classification |
| **Test set size** | 566,149 flows |
| **Zero-day simulation** | Slowloris + Slowhttptest excluded from RF training |

**Zero-Day Detection Results:**
| Metric | Value |
|--------|-------|
| Zero-day recall (Slowloris) | 82.4% |
| Zero-day recall (Slowhttptest) | 79.8% |
| Combined zero-day recall | 81.1% |
| False positive rate | 4.1% |
| Average detection latency | 45 ms |

**Severity Distribution (Test Set):**
| Severity | Count | Percentage |
|----------|-------|------------|
| Critical | 8,234 | 1.45% |
| High | 15,672 | 2.77% |
| Medium | 42,318 | 7.47% |
| Low | 499,925 | 88.30% |

---

### Evaluation 4: Latency and Throughput

| Property | Value |
|----------|-------|
| **Date** | 2025-06-12 |
| **Hardware** | Intel i7, 16GB RAM |
| **Test duration** | 300 seconds |

**Latency Measurements:**
| Stage | Mean (ms) | P95 (ms) | P99 (ms) |
|-------|-----------|----------|----------|
| Feature transform | 2.1 | 4.3 | 8.7 |
| Isolation Forest | 18.4 | 32.1 | 45.6 |
| Random Forest | 12.7 | 22.8 | 31.2 |
| Alert generation | 0.3 | 0.8 | 1.2 |
| **End-to-end** | **42.3** | **68.4** | **92.1** |

**Throughput:**
| Metric | Value |
|--------|-------|
| Flows per second | 1,247 |
| Batch size (1 sec) | ~1,250 flows |
| Memory usage (peak) | 2.1 GB |
