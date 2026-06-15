# Data Flow

## Batch Processing Flow

The batch pipeline processes the complete CICIDS2017 dataset for model training and evaluation.

```
┌─────────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐
│ Raw CSV     │───▶│ Cleaner  │───▶│ Encoder │───▶│ Imputer │
│ (CICIDS2017)│    │ Remove   │    │ Label   │    │ Mean/   │
│             │    │ dups,    │    │ encode  │    │ median  │
│ ~2.8M rows  │    │ filter   │    │ classes │    │ fill    │
└─────────────┘    └──────────┘    └─────────┘    └─────────┘
                                                  │
                                                  ▼
┌─────────────┐    ┌──────────┐    ┌─────────────────────────────┐
│ Model Ready │◀───│ Splitter │◀───│ Scaler                      │
│ Features    │    │ 80/20    │    │ StandardScaler (z-score)    │
│ + Labels    │    │ stratify │    │ Fit on train, transform both│
└──────┬──────┘    └──────────┘    └─────────────────────────────┘
       │
       ├──────────────────────────────┐
       ▼                              ▼
┌──────────────┐            ┌──────────────┐
│ Train Set    │            │ Test Set     │
│ (80%)        │            │ (20%)        │
│ ~2.24M rows  │            │ ~560K rows   │
└──────┬───────┘            └──────┬───────┘
       │                           │
       ▼                           ▼
┌──────────────┐            ┌──────────────┐
│ Train IF     │            │ Evaluate     │
│ Train RF     │            │ Both Models  │
│ Save models  │            │ Generate     │
│              │            │ metrics      │
└──────────────┘            └──────────────┘
```

## Streaming Detection Flow

The streaming pipeline processes simulated network traffic in real-time.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ CSV File     │───▶│ Stream       │───▶│ MicroBatch   │
│ (test data)  │    │ Simulator    │    │ Accumulator  │
│              │    │ row-by-row   │    │ 1-sec window │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                                │
                                                ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Alert        │◀───│ Detection    │◀───│ Feature      │
│ Manager      │    │ Engine       │    │ Transformer  │
│ Route alerts │    │ IF + RF      │    │ Scale/encode │
└──────┬───────┘    └──────────────┘    └──────────────┘
       │
       ├──────────────────────────┐
       ▼                          ▼
┌──────────────┐          ┌──────────────┐
│ JSON Logger  │          │ CSV Logger   │
│ alerts.json  │          │ alerts.csv   │
└──────────────┘          └──────────────┘
       │
       ▼
┌──────────────┐
│ Dashboard    │
│ Streamlit UI │
│ Real-time    │
└──────────────┘
```

## Data Formats

### Internal Flow Representation

Each network flow is represented as a dictionary or pandas Series with 30 features:

```python
{
    "flow_duration": float,        # Total flow duration in seconds
    "fwd_pkt_count": int,          # Forward packet count
    "bwd_pkt_count": int,          # Backward packet count
    "fwd_byte_count": int,         # Total forward bytes
    "bwd_byte_count": int,         # Total backward bytes
    "fwd_iat_mean": float,         # Mean forward inter-arrival time
    "bwd_iat_mean": float,         # Mean backward inter-arrival time
    "fwd_header_len": int,         # Forward header length sum
    "bwd_header_len": int,         # Backward header length sum
    "fwd_pkt_len_mean": float,     # Mean forward packet length
    "bwd_pkt_len_mean": float,     # Mean backward packet length
    # ... 19 additional features
    "label": str,                  # Traffic class label
    "timestamp": datetime          # Flow timestamp
}
```

### Alert Schema

```json
{
    "alert_id": "uuid",
    "timestamp": "ISO8601",
    "severity": "critical|high|medium|low",
    "attack_type": "known|zero_day",
    "confidence": 0.95,
    "source_ip": "10.0.0.1",
    "dest_ip": "10.0.0.2",
    "flow_features": {...}
}
```
