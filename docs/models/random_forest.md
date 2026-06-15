# Random Forest

## Overview

Random Forest is an ensemble supervised learning algorithm that constructs multiple decision trees during training and outputs the class that is the mode of the classes of individual trees. For this project, it classifies known DoS attack types (Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed) from benign traffic.

## Algorithm Principle

1. **Bootstrap Aggregation (Bagging)**: Each tree is trained on a random subset of the training data (with replacement).
2. **Feature Randomness**: At each split, only a random subset of features is considered, decorrelating trees.
3. **Majority Voting**: Final prediction is the class with the most votes across all trees.
4. **Out-of-Bag Estimation**: ~37% of data not used for each tree provides internal validation.

## Implementation

### Model Configuration

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=150,           # Number of trees
    criterion='gini',           # Split quality measure
    max_depth=None,             # Nodes expand until pure or min_samples
    min_samples_split=2,        # Minimum samples to split a node
    min_samples_leaf=1,         # Minimum samples in leaf node
    max_features='sqrt',        # Features per split: sqrt(30) ≈ 5
    class_weight='balanced',    # Adjust for class imbalance
    random_state=42,            # Reproducibility
    n_jobs=-1                   # Parallel tree construction
)
```

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 150 | Good accuracy-speed tradeoff |
| `criterion` | 'gini' | Faster than 'entropy', comparable accuracy |
| `max_depth` | None | Allow full growth for complex patterns |
| `max_features` | 'sqrt' | Standard for classification (sqrt(p)) |
| `class_weight` | 'balanced' | Handle DoS class imbalance |
| `random_state` | 42 | Reproducible results |

### Training

```python
# Train on labeled data (all classes)
model.fit(X_train, y_train)

# Save model
import joblib
joblib.dump(model, 'models/random_forest.pkl')
```

### Inference

```python
# Predict class labels
predictions = model.predict(X_test)

# Get class probabilities
probabilities = model.predict_proba(X_test)

# Get feature importances
importances = model.feature_importances_
```

## Class Labels

The Random Forest classifies traffic into the following categories:

| Label | Description |
|-------|-------------|
| BENIGN | Normal network traffic |
| DoS Hulk | High-rate HTTP flood |
| DoS GoldenEye | HTTP connection exhaustion |
| DoS Slowloris | Slow connection hold attack |
| DoS Slowhttptest | Slow HTTP body attack |
| Heartbleed | OpenSSL vulnerability exploit |

## Feature Importance

Top features typically ranked by Random Forest importance:

1. `flow_duration` — Total flow lifetime
2. `fwd_pkt_count` — Forward packet count
3. `bwd_pkt_count` — Backward packet count
4. `fwd_iat_mean` — Mean forward inter-arrival time
5. `fwd_pkt_len_mean` — Mean forward packet length
6. `bwd_byte_count` — Total backward bytes
7. `fwd_flag_count` — Forward SYN/FIN/RST flags
8. `flow_byts_sec` — Bytes per second

## Performance Characteristics

| Metric | Expected Range |
|--------|---------------|
| Training time (2.24M samples) | 60-300 seconds |
| Inference time per flow | < 10ms |
| Memory usage | ~500 MB - 1 GB |
| Classification accuracy | 92-97% |
| Multi-class F1 (macro) | 90-95% |
