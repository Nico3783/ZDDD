# Assumptions

## Dataset Assumptions

### CICIDS2017 Representativeness

The CICIDS2017 dataset is assumed to provide a realistic representation of modern network traffic patterns, including both benign communications and various DoS attack vectors. While no dataset perfectly mirrors production network traffic, CICIDS2017 captures sufficient behavioral diversity to train effective detection models.

### Flow-Based Feature Sufficiency

It is assumed that 30 flow-level statistical features derived from network connections are sufficient to capture the behavioral characteristics necessary for distinguishing normal traffic from DoS attacks. These features encode timing patterns, packet sizes, flag distributions, and byte transfer rates that collectively characterize traffic behavior.

### Feature Completeness

The selected feature set (30 features) covers the essential dimensions of network flow behavior:
- Packet inter-arrival times (forward and backward)
- Flow duration and volume
- TCP flag distributions
- Byte and packet count statistics
- Header length distributions

## Model Assumptions

### Contamination Rate

The contamination parameter for Isolation Forest is assumed to be approximately 10%, reflecting the expected proportion of anomalous (attack) traffic in the training data. This assumption is validated against the CICIDS2017 class distribution.

### Training Data Representativeness

Training data is assumed to be representative of the traffic patterns the model will encounter during inference. Benign traffic in the training set captures the baseline behavior of the monitored network environment.

### Feature Independence

While features may exhibit correlations, the models are assumed to perform adequately without requiring explicit feature decorrelation. Random Forest handles correlated features gracefully through its ensemble approach.

## Operational Assumptions

### Single-Machine Processing

The system operates on a single machine with sufficient CPU and memory resources to handle model training and inference. No distributed computing infrastructure is assumed.

### Labeled Training Data

Supervised training of the Random Forest classifier assumes access to accurately labeled attack traffic in the training dataset. Mislabeling is assumed to be minimal in CICIDS2017.

### Attack Novelty

Zero-day attacks are assumed to exhibit behavioral patterns that differ sufficiently from normal traffic to be detectable by the Isolation Forest model, even without prior exposure to similar attack signatures.

## Statistical Assumptions

- Traffic flows follow approximately normal distributions for baseline features.
- Attack traffic deviates from baseline distributions in at least one feature dimension.
- The 80/20 train-test split provides statistically reliable performance estimates.
- Cross-validation results generalize to the broader test population.
