# Model Evaluation

## Evaluation Framework

Model evaluation uses the held-out test set (20% of CICIDS2017) with comprehensive metrics covering classification performance, anomaly detection quality, and operational characteristics.

## Classification Metrics

### Core Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | ≥ 92% |
| **Precision** | TP / (TP + FP) | ≥ 90% |
| **Recall** | TP / (TP + FN) | ≥ 85% |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | ≥ 88% |
| **ROC-AUC** | Area under ROC curve | ≥ 0.95 |

### Multi-Class Metrics

For Random Forest classification across 6 classes:

- **Macro F1**: Unweighted mean of per-class F1 scores.
- **Weighted F1**: Weighted by class support (handles imbalance).
- **Cohen's Kappa**: Agreement beyond chance.
- **Matthews Correlation Coefficient**: Balanced measure even with imbalance.

## Anomaly Detection Metrics

### Isolation Forest Evaluation

| Metric | Description | Target |
|--------|-------------|--------|
| **Detection Rate** | % of attacks correctly identified | ≥ 85% |
| **False Positive Rate** | % of benign flagged as anomalous | ≤ 5% |
| **AUC-ROC** | Area under ROC for anomaly score | ≥ 0.93 |
| **AUC-PR** | Area under Precision-Recall curve | ≥ 0.85 |

### Threshold Impact Analysis

Performance measured at each severity threshold:

```python
thresholds = [0.5, 0.7, 0.9]
for t in thresholds:
    predictions = (anomaly_scores >= t).astype(int)
    report = classification_report(y_true, predictions)
```

## Confusion Matrix

The confusion matrix reveals per-class classification patterns:

```
                 Predicted
              B   DH  DG  DS  SHT  HB
Actual B  [ 98   0   0   1   0    0 ]
       DH [  0  97   1   0   1    0 ]
       DG [  0   2  95   1   1    0 ]
       DS [  1   0   1  96   0    1 ]
       SHT[  0   1   1   0  97    0 ]
       HB [  0   0   0   1   0   98 ]

B=BENIGN, DH=DoS Hulk, DG=GoldenEye, DS=Slowloris,
SHT=Slowhttptest, HB=Heartbleed
```

## Latency and Throughput

| Measurement | Metric | Target |
|-------------|--------|--------|
| Per-flow inference | Latency (ms) | < 100ms |
| Batch throughput | Flows/second | ≥ 1000 |
| End-to-end pipeline | Latency (ms) | < 200ms |
| Dashboard refresh | Interval (s) | ≤ 2s |

## ROC Curve Analysis

The ROC curve is plotted for both models:

- **Isolation Forest**: Binary ROC (benign vs. attack) using anomaly scores.
- **Random Forest**: Multi-class ROC using one-vs-rest approach with class probabilities.

## Evaluation Reports

Evaluation results are saved to:
- `reports/evaluation_log.md` — Detailed test results.
- `reports/findings.md` — Key findings and conclusions.
- `evaluation/` — Generated plots (ROC, confusion matrix).

## Cross-Validation

5-fold stratified cross-validation provides robust performance estimates:

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
```
