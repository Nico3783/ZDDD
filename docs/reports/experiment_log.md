# Experiment Log

## Experiments Overview

Multiple configurations were tested to optimize detection performance. Each experiment varies one or more parameters while holding others constant.

---

### Experiment 1: Contamination Rate Sensitivity

**Objective**: Determine optimal contamination parameter for Isolation Forest.

| Run | Contamination | Detection Rate | FPR | F1 |
|-----|--------------|----------------|-----|-----|
| 1a | 0.05 | 71.2% | 1.8% | 0.79 |
| 1b | 0.10 | 83.2% | 3.2% | 0.85 |
| 1c | 0.15 | 87.6% | 5.1% | 0.84 |
| 1d | 0.20 | 90.1% | 7.8% | 0.82 |

**Conclusion**: contamination=0.10 provides the best balance. Higher values increase detection but also increase false positives.

---

### Experiment 2: Number of Estimators

**Objective**: Evaluate impact of n_estimators on Isolation Forest performance.

| Run | n_estimators | Training Time | Detection Rate | FPR |
|-----|-------------|---------------|----------------|-----|
| 2a | 50 | 44s | 80.1% | 3.5% |
| 2b | 100 | 87s | 83.2% | 3.2% |
| 2c | 150 | 131s | 83.8% | 3.1% |
| 2d | 200 | 174s | 83.9% | 3.0% |

**Conclusion**: 100 estimators is sufficient. Diminishing returns beyond 100 with increased training time.

---

### Experiment 3: Random Forest Estimators

**Objective**: Evaluate impact of n_estimators on Random Forest classification.

| Run | n_estimators | Training Time | Accuracy | Macro F1 |
|-----|-------------|---------------|----------|----------|
| 3a | 50 | 52s | 94.2% | 91.8% |
| 3b | 100 | 104s | 95.3% | 93.1% |
| 3c | 150 | 142s | 95.8% | 93.4% |
| 3d | 200 | 218s | 95.9% | 93.5% |

**Conclusion**: 150 estimators provides good accuracy. Beyond 150, improvement is marginal.

---

### Experiment 4: Feature Count Impact

**Objective**: Determine minimum feature count for acceptable performance.

| Run | Features | Accuracy | F1 | Training Time |
|-----|----------|----------|-----|---------------|
| 4a | 10 | 89.4% | 86.2% | 38s |
| 4b | 20 | 93.7% | 91.3% | 72s |
| 4c | 30 | 95.8% | 93.4% | 142s |
| 4d | 40 | 95.9% | 93.5% | 198s |
| 4e | 50 | 95.8% | 93.3% | 261s |

**Conclusion**: 30 features provide near-optimal performance. Additional features add complexity without significant gain.

---

### Experiment 5: Class Weight Strategy

**Objective**: Evaluate impact of class weighting on minority class detection.

| Run | class_weight | Heartbleed F1 | Slowloris F1 | Overall F1 |
|-----|-------------|---------------|--------------|------------|
| 5a | None | 0.82 | 0.87 | 0.91 |
| 5b | balanced | 1.00 | 0.93 | 0.93 |
| 5c | balanced_subsample | 0.98 | 0.92 | 0.92 |

**Conclusion**: `class_weight='balanced'` provides best overall performance and handles minority classes well.

---

### Experiment 6: Zero-Day Simulation

**Objective**: Test zero-day detection by excluding specific attack types from RF training.

**Setup**: Slowloris and Slowhttptest excluded from Random Forest training, treated as unknown.

| Scenario | IF Detection | RF Classification | Combined |
|----------|-------------|-------------------|----------|
| Slowloris (zero-day) | 82.4% | N/A (unknown) | 82.4% |
| Slowhttptest (zero-day) | 79.8% | N/A (unknown) | 79.8% |
| Hulk (known) | 91.2% | 98.1% | 98.1% |
| GoldenEye (known) | 87.5% | 94.3% | 94.3% |

**Conclusion**: Anomaly detection successfully identifies zero-day variants with 80%+ recall. Combined approach maintains high accuracy for known attacks.

---

### Experiment 7: Threshold Tuning

**Objective**: Optimize severity thresholds for balanced alert distribution.

| Config | Critical | High | Medium | Low | Total Alerts |
|--------|----------|------|--------|-----|-------------|
| Conservative | 0.8% | 1.5% | 4.2% | 93.5% | 36,812 |
| Balanced (default) | 1.5% | 2.8% | 7.5% | 88.2% | 66,224 |
| Aggressive | 3.2% | 5.1% | 12.4% | 79.3% | 117,489 |

**Conclusion**: Balanced configuration (critical=0.9, high=0.7, medium=0.5) provides actionable alert volume without overwhelming operators.
