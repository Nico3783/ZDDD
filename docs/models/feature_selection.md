# Feature Selection

## Overview

Feature selection reduces the CICIDS2017 feature set from 80+ raw features to 30 carefully chosen flow-based features. This reduces model complexity, improves training speed, and enhances generalization by removing redundant and irrelevant features.

## Selected Features

The 30 features are organized into five categories:

### Flow Duration Features

| # | Feature | Description | Unit |
|---|---------|-------------|------|
| 1 | `flow_duration` | Total flow duration | seconds |
| 2 | `flow_byts_sec` | Flow bytes per second | bytes/s |
| 3 | `flow_pkts_sec` | Flow packets per second | pkts/s |

### Forward Flow Features

| # | Feature | Description | Unit |
|---|---------|-------------|------|
| 4 | `fwd_pkt_count` | Total forward packets | count |
| 5 | `fwd_byte_count` | Total forward bytes | bytes |
| 6 | `fwd_pkt_len_mean` | Mean forward packet length | bytes |
| 7 | `fwd_pkt_len_std` | Std dev forward packet length | bytes |
| 8 | `fwd_pkt_len_max` | Max forward packet length | bytes |
| 9 | `fwd_iat_mean` | Mean forward inter-arrival time | seconds |
| 10 | `fwd_iat_std` | Std dev forward inter-arrival time | seconds |
| 11 | `fwd_iat_max` | Max forward inter-arrival time | seconds |
| 12 | `fwd_header_len` | Sum of forward header lengths | bytes |
| 13 | `fwd_flag_count` | Count of forward TCP flags | count |

### Backward Flow Features

| # | Feature | Description | Unit |
|---|---------|-------------|------|
| 14 | `bwd_pkt_count` | Total backward packets | count |
| 15 | `bwd_byte_count` | Total backward bytes | bytes |
| 16 | `bwd_pkt_len_mean` | Mean backward packet length | bytes |
| 17 | `bwd_pkt_len_std` | Std dev backward packet length | bytes |
| 18 | `bwd_pkt_len_max` | Max backward packet length | bytes |
| 19 | `bwd_iat_mean` | Mean backward inter-arrival time | seconds |
| 20 | `bwd_iat_std` | Std dev backward inter-arrival time | seconds |
| 21 | `bwd_iat_max` | Max backward inter-arrival time | seconds |
| 22 | `bwd_header_len` | Sum of backward header lengths | bytes |

### Bidirectional Features

| # | Feature | Description | Unit |
|---|---------|-------------|------|
| 23 | `pkt_count` | Total packet count | count |
| 24 | `byte_count` | Total byte count | bytes |
| 25 | `pkt_len_mean` | Mean packet length | bytes |
| 26 | `pkt_len_std` | Std dev packet length | bytes |
| 27 | `iat_mean` | Mean inter-arrival time | seconds |
| 28 | `iat_std` | Std dev inter-arrival time | seconds |

### Flag-Based Features

| # | Feature | Description | Unit |
|---|---------|-------------|------|
| 29 | `syn_flag_count` | SYN flag count | count |
| 30 | `rst_flag_count` | RST flag count | count |

## Feature Selection Methods

### Mutual Information

```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X_train, y_train)
top_30 = mi_scores.argsort()[-30:][::-1]
```

### Random Forest Importance

Feature importance from a preliminary Random Forest model identifies the most discriminative features for attack classification.

### Correlation Analysis

Features with Pearson correlation > 0.95 are considered redundant; one is retained based on domain relevance.

## Feature Configuration

Features are defined in `config/features.yaml`:

```yaml
features:
  selected:
    - flow_duration
    - fwd_pkt_count
    - bwd_pkt_count
    # ... 27 additional features
  target_column: label
  excluded:
    - flow_id
    - src_ip
    - dst_ip
    - timestamp
```
