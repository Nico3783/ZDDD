# Isolation Forest

## Overview

The Isolation Forest (IF) algorithm is an unsupervised anomaly detection method that identifies anomalies by isolating observations. Anomalies are few and different, making them easier to isolate than normal points. The algorithm builds an ensemble of random trees (iTrees) where anomalies require fewer splits to isolate, resulting in shorter average path lengths.

## Algorithm Principle

Unlike traditional methods that profile normal data and detect deviations, Isolation Forest explicitly isolates anomalies:

1. Randomly select a feature and a split value between the feature's maximum and minimum.
2. Recursively partition data until each point is isolated or a depth limit is reached.
3. Anomalies have shorter average path lengths from root to leaf across the ensemble.
4. The anomaly score is normalized using the average path length relative to the expected path length of random data.

## Implementation

### Model Configuration

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=100,       # Number of base isolation trees
    max_samples='auto',     # subsample size (min(256, n_samples))
    contamination=0.1,      # Expected proportion of anomalies
    max_features=1.0,       # Features to draw for each tree
    bootstrap=False,        # No bootstrap sampling
    random_state=42,        # Reproducibility
    n_jobs=-1               # Use all CPU cores
)
```

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 100 | Sufficient for stable anomaly scores |
| `max_samples` | 'auto' (256) | Balances performance and accuracy |
| `contamination` | 0.1 | ~10% attack traffic in CICIDS2017 |
| `max_features` | 1.0 | Use all 30 features per tree |
| `random_state` | 42 | Reproducible results |

### Training

```python
# Train on benign traffic only (for zero-day detection)
model.fit(X_train_benign)

# Or train on full training set
model.fit(X_train)

# Save model
import joblib
joblib.dump(model, 'models/isolation_forest.pkl')
```

### Inference

```python
# Predict anomalies (-1 = anomaly, 1 = normal)
predictions = model.predict(X_test)

# Get anomaly scores (lower = more anomalous)
scores = model.decision_function(X_test)

# Convert to 0-1 scale (higher = more anomalous)
anomaly_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
```

## Role in Zero-Day Detection

The Isolation Forest serves as the primary zero-day detection mechanism:

1. Trained predominantly on normal traffic patterns.
2. Detects novel attack patterns that deviate from learned normal behavior.
3. Produces anomaly scores that drive the severity classification.
4. Complements the Random Forest classifier which handles known attack types.

## Performance Characteristics

| Metric | Expected Range |
|--------|---------------|
| Training time (2.24M samples) | 30-120 seconds |
| Inference time per flow | < 5ms |
| Memory usage | ~200-500 MB |
| Detection rate (zero-day) | 80-90% |
| False positive rate | 2-5% |
