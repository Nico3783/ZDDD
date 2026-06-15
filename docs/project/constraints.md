# Constraints

## Technical Constraints

### Single-Machine Deployment

The system is designed and tested on a single-machine environment. All model training, inference, and dashboard rendering execute locally. This limits scalability but is appropriate for the academic scope of the project.

### Python 3.13 Runtime

The implementation targets Python 3.13 specifically. Some libraries may have compatibility limitations with this version. All dependencies are pinned in `requirements.txt` to ensure reproducibility.

### CPU-Only Machine Learning

No GPU acceleration is used. All scikit-learn models (Isolation Forest, Random Forest) run on CPU. While this limits training speed for large datasets, it ensures the system runs on commodity hardware without specialized accelerators.

### Memory Limitations

Processing the full CICIDS2017 dataset requires significant memory. The system must operate within typical workstation constraints (16-32 GB RAM). Large batch sizes or excessive feature engineering may cause memory pressure.

## Dataset Constraints

### CICIDS2017 Limitations

- **Temporal scope**: Traffic captured during a single week in 2017 may not reflect current attack patterns.
- **Synthetic attacks**: Some attack traffic was generated in controlled lab environments, which may differ from real-world attack behavior.
- **Label accuracy**: While generally reliable, dataset labels may contain misclassifications or ambiguous flows.
- **Feature set**: The 80+ raw features require careful selection to avoid overfitting and maintain interpretability.

### Train-Test Split

The 80/20 train-test split means 20% of labeled data is reserved for evaluation. This may reduce the effective training set size, particularly for minority attack classes.

## Academic Constraints

### Evaluation Boundaries

Performance metrics are computed against the CICIDS2017 test set. Results may not generalize to other network environments, traffic patterns, or attack types without additional validation.

### No Real-World Validation

The system has not been tested against live network traffic. Detection rates and false positive measurements are based on synthetic evaluation, not production monitoring.

### Scope Limitation

The project addresses DoS attacks only. Other attack categories (DDoS, botnet, infiltration, port scanning) are outside the current scope.

## Resource Constraints

| Resource | Constraint | Mitigation |
|----------|-----------|------------|
| CPU | No GPU available | Optimized sklearn parameters |
| RAM | 16-32 GB typical | Chunked data processing |
| Storage | Local filesystem only | Efficient CSV/parquet formats |
| Network | No live capture | StreamSimulator from CSV |
| Time | Academic semester | Focused scope on DoS |
| Budget | Zero-cost tools | Open-source stack only |
