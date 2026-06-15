# Zero-Day Attack Characteristics

## Definition

A zero-day attack is a previously unknown vulnerability or attack technique that has no existing signature, detection rule, or patch. In the context of DoS detection, zero-day attacks represent novel DoS vectors that evade signature-based and known-pattern detection systems.

## Key Characteristics

### 1. No Prior Signatures

Zero-day attacks cannot be detected by signature-based systems because no prior example exists in the training data or rule sets. This is the defining characteristic that necessitates anomaly-based detection approaches.

**Implication**: The Isolation Forest model must detect attacks based on behavioral deviation from normal traffic, not pattern matching against known attack signatures.

### 2. Behavioral Anomaly

Despite being unknown, zero-day attacks still produce traffic that deviates from normal baseline behavior. The attack must consume resources or disrupt service, which creates observable anomalies in flow statistics.

**Observable deviations**:
- Unusual packet rate patterns
- Abnormal flow duration distributions
- Novel combinations of existing feature values
- Atypical protocol behavior sequences

### 3. Low Initial Detection Rate

By definition, zero-day attacks initially evade most detection systems. The goal of this project is to improve initial detection rates through anomaly-based detection.

**Detection timeline**:
1. **Zero-day phase**: No signatures exist; only anomaly detection can identify.
2. **Early analysis**: Security researchers analyze captured traffic to develop signatures.
3. **Signature deployment**: IDS/IPS systems receive updated rules.
4. **Mature detection**: Signature-based systems effectively detect the attack.

### 4. Feature Space Deviation

Zero-day attacks occupy regions of the feature space that are远离 the normal traffic distribution but may also远离 known attack clusters.

```
Feature Space:

Normal     Known Attacks     Zero-Day
  ●●●          ▲▲▲              ★
 ●●●●●        ▲▲▲▲▲            ★★
  ●●●          ▲▲▲              ★

● = Normal traffic cluster
▲ = Known attack clusters
★ = Zero-day attack region (novel deviation)
```

### 5. Unknown Attack Vector

The specific mechanism of a zero-day DoS attack is unknown. It could be:
- A novel exploitation of a known protocol
- An unexpected combination of existing attack techniques
- A new application-layer vulnerability
- An unforeseen resource exhaustion pattern

## Detection Strategy

### Anomaly-Based Detection

The primary strategy for zero-day detection uses Isolation Forest to identify traffic that deviates from learned normal patterns:

1. **Training**: Train on predominantly benign traffic to learn the normal distribution.
2. **Detection**: Flag flows with low isolation probability as potential anomalies.
3. **Severity**: Assign severity based on anomaly score magnitude.

### Threshold Adaptation

The zero-day boost mechanism enhances detection of potentially novel attacks:

```python
if rf_confidence < 0.5:
    # Low confidence suggests unknown pattern
    adjusted_score = anomaly_score + ZERO_DAY_BOOST
else:
    adjusted_score = anomaly_score
```

### Behavioral Indicators

Common behavioral indicators of zero-day DoS attacks:

| Indicator | Normal Range | Anomalous Range |
|-----------|-------------|-----------------|
| Packet rate | 10-500 pkts/s | <5 or >2000 pkts/s |
| Byte rate | 1 KB/s - 10 MB/s | <100 B/s or >50 MB/s |
| Flow duration | 0.1s - 300s | >600s or <0.01s |
| Forward/backward ratio | 0.5 - 2.0 | >5.0 or <0.2 |
| Flag concentration | Distributed | >80% in one flag type |

## Evaluation Challenge

Evaluating zero-day detection capability requires creating synthetic zero-day scenarios:
1. Exclude specific attack types from training data.
2. Test detection against the excluded attacks as if they were unknown.
3. Measure detection rate for these "simulated zero-day" attacks.

This methodology approximates real zero-day detection performance using the available labeled dataset.
