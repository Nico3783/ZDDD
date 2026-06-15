# Project Objectives

## Primary Objective

Develop a real-time autonomous anomaly detection engine capable of identifying zero-day Denial of Service (DoS) exploits in network traffic using machine learning techniques, specifically Isolation Forest for anomaly detection and Random Forest for classification of known attack types.

## Specific Objectives

### 1. Anomaly-Based Detection Model

- Implement an Isolation Forest model trained on benign network traffic to detect anomalous flow patterns indicative of zero-day DoS attacks.
- Achieve anomaly detection sensitivity sufficient to identify novel attack vectors without prior signature definitions.
- Maintain a false positive rate below 5% during normal traffic classification.
- Tune contamination parameters to balance detection rate against false alarm frequency.

### 2. Real-Time Traffic Simulation Pipeline

- Build a streaming data ingestion pipeline capable of simulating real-time network traffic from CSV-based datasets.
- Implement micro-batching to process traffic flows at configurable intervals (default: 1-second windows).
- Support both batch preprocessing for model training and stream-based inference for detection simulation.
- Ensure the pipeline handles feature extraction and transformation in real-time with latency under 100ms per flow.

### 3. Benchmark Evaluation

- Evaluate model performance using the CICIDS2017 dataset (Wednesday working hours traffic) as the primary benchmark.
- Report standard ML metrics: accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Measure detection latency and throughput to assess real-time feasibility.
- Compare anomaly detection performance against known baselines from literature.

## Academic Context

This project fulfills the requirements for the B.Sc. Cybersecurity program at the Federal University of Technology Akure (FUTA), under the supervision of Prof. (Mrs.) Alowolodu, student: Oyelude Zion Clifford (CYS/20/4940).

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Zero-day detection rate | ≥ 85% |
| False positive rate | ≤ 5% |
| Detection latency | < 200ms per flow |
| Classification accuracy (known attacks) | ≥ 92% |
| Feature count | 30 flow-based features |
