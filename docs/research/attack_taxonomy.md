# Attack Taxonomy

## DoS Attack Classification

The CICIDS2017 dataset contains five distinct DoS attack types plus benign traffic. Each exhibits unique behavioral characteristics that the detection models must learn to identify.

## Attack Types

### DoS Hulk

**Category**: Volumetric HTTP flood

**Mechanism**: Sends a massive volume of HTTP GET requests to overwhelm the target web server. Each request is valid individually but the aggregate volume exhausts server resources (CPU, memory, connection pools).

**Characteristics**:
- Very high packet rate (>1000 packets/sec)
- High byte throughput
- Short flow duration per connection
- Consistent packet sizes (HTTP GET requests)
- Minimal server response (connection refused or timeout)

**Flow Signature**:
- `flow_pkts_sec` >> normal
- `fwd_pkt_count` very high
- `fwd_pkt_len_mean` ≈ 200-500 bytes (GET request size)
- `flow_duration` relatively short

### DoS GoldenEye

**Category**: HTTP connection exhaustion

**Mechanism**: Opens numerous HTTP connections and holds them open by sending headers slowly. Exhausts the server's connection pool by maintaining many half-open connections.

**Characteristics**:
- Moderate packet rate
- Low byte throughput per connection
- Long flow duration
- Asymmetric: more forward packets than backward
- Frequent keep-alive or partial header sends

**Flow Signature**:
- `flow_duration` >> normal (long-lived connections)
- `fwd_pkt_count` moderate to high
- `fwd_iat_mean` high (slow sending)
- `fwd_byte_count` moderate

### DoS Slowloris

**Category**: Connection holding attack

**Mechanism**: Opens connections and sends HTTP headers very slowly, keeping connections alive. A single attacker can consume hundreds of connections with minimal bandwidth.

**Characteristics**:
- Very low packet rate
- Very low byte throughput
- Very long flow duration
- Single connection per attack thread
- Periodic small data sends to prevent timeout

**Flow Signature**:
- `flow_duration` extremely long
- `fwd_pkt_count` low (intermittent sends)
- `fwd_iat_mean` very high
- `fwd_byte_count` very low
- `flow_byts_sec` extremely low

### DoS Slowhttptest

**Category**: Slow HTTP body attack

**Mechanism**: Similar to Slowloris but targets the HTTP body. Sends POST data very slowly, keeping the server waiting for the complete request body.

**Characteristics**:
- Low packet rate
- Low byte throughput
- Long flow duration
- POST request method
- Periodic small body chunks

**Flow Signature**:
- Similar to Slowloris
- `fwd_pkt_count` slightly higher (body chunks)
- `bwd_pkt_count` higher (server sends 100 Continue)

### Heartbleed

**Category**: Vulnerability exploitation

**Mechanism**: Exploits the OpenSSL Heartbleed vulnerability (CVE-2014-0160) to read arbitrary memory from TLS-enabled servers. While not a DoS attack per se, it causes resource exhaustion and potential service disruption.

**Characteristics**:
- Low packet rate
- Small payload sizes
- TLS handshake followed by heartbeat requests
- Server responds with large memory dumps
- Asymmetric: server sends much more data than client

**Flow Signature**:
- `bwd_byte_count` >> `fwd_byte_count`
- `bwd_pkt_len_mean` high (memory dump responses)
- Distinct TLS-related packet patterns

### BENIGN

**Category**: Normal network traffic

**Characteristics**:
- Variable packet rates and sizes
- Balanced forward/backward traffic
- HTTP request-response patterns
- Typical web browsing behavior
- Mixed protocols and applications

## Detection Challenge

The detection models must distinguish between these attack types and benign traffic using flow-level features, while also identifying zero-day variants that may not match any known signature.
