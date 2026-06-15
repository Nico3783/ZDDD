# Threshold Strategy

## Overview

Threshold strategy defines how raw model outputs (anomaly scores and classification probabilities) are mapped to actionable severity levels and detection decisions. The system uses a multi-tier approach combining anomaly scores from Isolation Forest with classification confidence from Random Forest.

## Anomaly Score Thresholds

The Isolation Forest anomaly score is normalized to [0, 1] where higher values indicate greater anomaly likelihood.

### Severity Levels

| Severity | Score Range | Description | Response |
|----------|------------|-------------|----------|
| **Critical** | score ≥ 0.9 | Highly anomalous, strong zero-day indicator | Immediate alert, automated response |
| **High** | 0.7 ≤ score < 0.9 | Strongly anomalous, likely attack | Priority alert, investigation |
| **Medium** | 0.5 ≤ score < 0.7 | Moderately anomalous, possible attack | Standard alert, monitoring |
| **Low** | score < 0.5 | Weakly anomalous, likely benign | Log entry, no alert |

### Threshold Configuration

```yaml
thresholds:
  anomaly:
    critical: 0.9
    high: 0.7
    medium: 0.5
  zero_day_boost: 0.2
  classification_confidence: 0.85
```

## Zero-Day Boost Mechanism

When the Random Forest classifier identifies traffic as potentially unknown (low confidence across all known classes), a boost factor is applied to the anomaly score:

```
adjusted_score = anomaly_score + zero_day_boost
```

### Logic

1. If RF confidence for the top predicted class < 0.5, apply the full boost (+0.2).
2. If RF confidence is between 0.5 and 0.85, apply a partial boost (proportional).
3. If RF confidence ≥ 0.85, no boost is applied (high confidence in known classification).

### Rationale

Low RF confidence suggests the traffic pattern may not match any known attack class, increasing the likelihood of a zero-day variant. The boost elevates the severity to ensure such cases receive attention.

## Classification Confidence Threshold

The Random Forest prediction is only accepted when confidence exceeds a threshold:

```python
rf_probabilities = model.predict_proba(X)
max_confidence = rf_probabilities.max()

if max_confidence >= classification_confidence:
    predicted_class = rf_classes[rf_probabilities.argmax()]
    is_known_attack = True
else:
    predicted_class = "UNKNOWN"
    is_known_attack = False
```

## Combined Decision Matrix

| Anomaly Score | RF Known Attack | RF Confidence | Final Decision |
|---------------|----------------|---------------|----------------|
| ≥ 0.9 | Any | Any | Critical alert |
| 0.7-0.9 | Yes | ≥ 0.85 | High alert (known) |
| 0.7-0.9 | Yes | < 0.85 | High alert (zero-day boost) |
| 0.7-0.9 | No | Any | High alert (zero-day) |
| 0.5-0.7 | Yes | ≥ 0.85 | Medium alert (known) |
| 0.5-0.7 | No | Any | Medium alert (zero-day boost) |
| < 0.5 | Any | Any | Low / No alert |

## Threshold Tuning

Thresholds are validated against the CICIDS2017 test set using ROC curve analysis and precision-recall tradeoffs. The default thresholds prioritize high recall (detection rate) over precision to minimize missed zero-day attacks.
