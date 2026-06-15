# Project Scope

## In Scope

### Network-Based Detection

The project focuses exclusively on network-level anomaly detection and DoS attack classification. Detection operates on flow-level network features extracted from packet captures, analyzing traffic patterns, flow statistics, and protocol behavior at the network layer.

### CICIDS2017 Dataset

The primary dataset for training and evaluation is the Canadian Institute for Cybersecurity Intrusion Detection System 2017 (CICIDS2017). Specifically, the Wednesday working hours portion of the dataset is used, which contains both benign traffic and multiple DoS attack types including Hulk, GoldenEye, Slowloris, Slowhttptest, and Heartbleed.

### Machine Learning Models

- **Isolation Forest**: Unsupervised anomaly detection model used to identify zero-day attacks by detecting deviations from normal traffic behavior.
- **Random Forest**: Supervised classification model used to classify known DoS attack types into their respective categories.

### Real-Time Simulation

The project includes a simulated real-time detection pipeline that processes network flow data in a streaming fashion, demonstrating the feasibility of real-time zero-day DoS detection without requiring actual network infrastructure.

### Python Implementation

All code is implemented in Python 3.13 using scikit-learn, pandas, numpy, and Streamlit for visualization. The implementation follows modular design principles with clear separation of concerns across 12+ source modules.

## Out of Scope

### Host-Based Detection

Host-based intrusion detection systems (HIDS) that monitor system calls, file integrity, or local process behavior are not addressed. The project operates solely at the network flow level.

### Encrypted Traffic Analysis

Deep packet inspection of encrypted traffic (TLS/SSL) is not performed. Feature extraction relies on flow-level metadata rather than payload content.

### Production Deployment

While the architecture is designed with production considerations, the system is an academic prototype. Full production deployment including integration with live network interfaces, SOC workflows, and enterprise SIEM systems is not included.

### Evasion Resistance

Advanced adversarial evasion techniques (e.g., traffic shaping, packet fragmentation attacks designed to evade detection) are not the focus of this study.

### Multi-Protocol Coverage

The scope is limited to TCP/IP-based DoS attacks. Attacks targeting UDP, ICMP, or application-layer protocols beyond HTTP are not comprehensively covered.

## Boundaries

| Aspect | Included | Excluded |
|--------|----------|----------|
| Detection type | Network flow analysis | Host-based monitoring |
| Traffic type | TCP/IP, HTTP | Encrypted, UDP floods |
| Dataset | CICIDS2017 | UNSW-NB15 (reference only) |
| Models | IF + RF | Deep learning, SVM |
| Deployment | Local simulation | Cloud/production |
| Evaluation | Academic benchmarks | Real-world traffic |
