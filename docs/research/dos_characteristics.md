# DoS Attack Characteristics

## Overview

Denial of Service (DoS) attacks aim to make a network service unavailable to legitimate users by exhausting resources or disrupting protocol operations. Understanding attack characteristics is essential for designing effective detection features and models.

## Attack Categories

### 1. Volumetric Attacks

**Objective**: Overwhelm network bandwidth or server capacity with high-volume traffic.

**Examples**: UDP flood, ICMP flood, HTTP flood (DoS Hulk)

**Detection Indicators**:
- Abnormally high packet rate
- High bytes/second throughput
- Large total flow volume
- Short per-flow duration
- Consistent packet sizes

**Feature Signatures**:
```
flow_pkts_sec >> normal_threshold
flow_byts_sec >> normal_threshold
fwd_pkt_count >> normal_threshold
fwd_pkt_len_mean ≈ constant (uniform payload)
```

### 2. Protocol Attacks

**Objective**: Exhaust server resources (connection tables, state machines) by exploiting protocol behavior.

**Examples**: SYN flood, Ping of Death, Smurf attack

**Detection Indicators**:
- High SYN flag count
- Incomplete TCP handshakes
- Abnormal flag combinations
- Connection state exhaustion

**Feature Signatures**:
```
syn_flag_count >> normal_threshold
rst_flag_count elevated
fwd_flag_count disproportionate to payload
```

### 3. Application Layer Attacks

**Objective**: Exhaust application-level resources (CPU, memory, database connections) with看似 legitimate requests.

**Examples**: HTTP slow POST, Slowloris, Slowhttptest, GoldenEye

**Detection Indicators**:
- Very slow data transmission rates
- Long-lived connections
- Low bytes per second per connection
- Asymmetric traffic patterns
- Partial or incomplete requests

**Feature Signatures**:
```
flow_duration >> normal_threshold
fwd_iat_mean >> normal_threshold
flow_byts_sec << normal_threshold
bwd_pkt_count << fwd_pkt_count
```

### 4. Resource Exhaustion Attacks

**Objective**: Consume specific server resources (file descriptors, memory, CPU cycles) through legitimate-appearing requests.

**Examples**: Hash collision DoS, ReDoS, XML bomb

**Detection Indicators**:
- High CPU utilization per request
- Memory growth over time
- Request patterns targeting vulnerable code paths

## Traffic Behavioral Patterns

### Normal Traffic

- Inter-arrival times follow approximately exponential distribution
- Packet sizes follow bimodal distribution (small ACKs + large data)
- Forward/backward ratio is approximately balanced
- Flow duration varies widely (short requests to long downloads)
- Byte rate is moderate and variable

### DoS Traffic

- Inter-arrival times are often uniform (automated tools)
- Packet sizes are often uniform (same request type)
- Forward/backward ratio is highly asymmetric
- Flow duration clusters around specific values
- Byte rate is either extremely high (volumetric) or extremely low (slow attacks)

## Feature Discriminability

| Feature | Normal | Volumetric | Slow Attack | Protocol |
|---------|--------|-----------|-------------|----------|
| `flow_duration` | Variable | Short | Very long | Variable |
| `flow_pkts_sec` | Moderate | Very high | Very low | High |
| `flow_byts_sec` | Moderate | Very high | Very low | Variable |
| `fwd_iat_mean` | Low | Low | Very high | Low |
| `fwd_pkt_count` | Variable | Very high | Low | High |
| `syn_flag_count` | Low | Low | Low | Very high |
| `fwd_bwd_ratio` | ~1.0 | >>1.0 | >>1.0 | Variable |

## Implications for Detection

1. **Multi-scale analysis**: Attacks span different time scales (millisecond floods to hour-long slow attacks).
2. **Feature diversity**: No single feature detects all attack types; ensemble methods are necessary.
3. **Threshold complexity**: Static thresholds fail; adaptive or ML-based thresholds needed.
4. **Zero-day potential**: Novel attacks may combine characteristics of known types in new ways.
