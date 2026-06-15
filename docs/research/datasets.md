# Datasets

## Primary Dataset: CICIDS2017

### Overview

The Canadian Institute for Cybersecurity Intrusion Detection System 2017 (CICIDS2017) is the primary dataset for this project. It provides a comprehensive collection of network traffic captures including both benign and malicious flows with accurate ground truth labels.

### Dataset Specifications

| Property | Value |
|----------|-------|
| Total flows | ~2,830,743 |
| Total records | ~2,830,743 (one per flow) |
| Features (raw) | 80+ |
| Features (selected) | 30 |
| Duration | 5 days (Mon-Fri) |
| Attack types | 14 categories |
| DoS types | Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed |
| Labeling | Flow-level labels with timestamps |

### Wednesday Working Hours

The project specifically uses the Wednesday portion, which contains:

- **Morning (9 AM - 12 PM)**: Benign traffic + DoS attacks.
- **Afternoon (1 PM - 4 PM)**: Continued attacks + Heartbleed exploitation.
- **Attack window**: Multiple DoS attack types executed simultaneously.

### Feature Categories

The raw dataset includes features across these categories:
- Flow duration and volume statistics
- Forward and backward packet counts
- Inter-arrival time statistics (mean, std, max, min)
- Packet length statistics (mean, std, max, min)
- Header length statistics
- TCP flag counts (SYN, ACK, FIN, RST, PSH, URG)
- Bulk transfer statistics
- Sub-flow statistics
- Active/idle time statistics

### Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| BENIGN | 2,273,097 | 80.3% |
| DoS Hulk | 231,073 | 8.2% |
| DoS GoldenEye | 10,293 | 0.4% |
| DoS Slowloris | 5,796 | 0.2% |
| DoS Slowhttptest | 5,499 | 0.2% |
| Heartbleed | 11 | < 0.01% |

## Secondary Reference: UNSW-NB15

### Overview

UNSW-NB15 serves as a reference dataset for comparison. Created at the Australian Centre for Cyber Security, it contains modern attack types and realistic background traffic.

### Specifications

| Property | Value |
|----------|-------|
| Total flows | 2,540,044 |
| Features | 49 |
| Attack types | 9 |
| Attack categories | Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms |

## Data Loading

### CICIDS2017 Loader

```python
from src.datasets.loader import DatasetLoader

loader = DatasetLoader(config)
df = loader.load_cicids2017(day="wednesday")
# Returns: pandas DataFrame with raw features + labels
```

### Data Format

All data is stored in CSV format with the following conventions:
- No index column in CSV files.
- Feature names use underscores (e.g., `flow_duration`).
- Labels are uppercase strings (e.g., `BENIGN`, `DoS Hulk`).
- Missing values are represented as `NaN` or `inf`.
