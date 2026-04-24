# Fed-Intel — Dataset Feature Comparison
>
> NF-CSE-CIC-IDS2018-v3 vs. Gotham 2025

---

## Critical Discovery: Different Abstraction Levels

| Property | NF-CSE-CIC-IDS2018-v3 | Gotham 2025 |
| --- | --- | --- |
| **Level** | **Flow-level** (aggregated) | **Packet-level** (raw) |
| **Features** | 53 | 23 |
| **Year** | Feb 2025 | Feb 2025 |
| **Source** | UQ (University of Queensland) | Zenodo |
| **Rows** | ~20M flows | Varies by device |
| **Labels** | Binary + 6 attack classes | 10 classes (benign + 9 attacks) |
| **Format** | CSV | CSV |

> This means we CAN'T just find "common columns" — the features are fundamentally different. We need to **engineer a shared feature space**.

---

## NF-CSE-CIC-IDS2018-v3 Features (53 Flow-Level)

### Core NetFlow (12 features — from v1)

| # | Feature | Description |
| --- | --- | --- |
| 1 | `IPV4_SRC_ADDR` | Source IP address |
| 2 | `IPV4_DST_ADDR` | Destination IP address |
| 3 | `L4_SRC_PORT` | Source port |
| 4 | `L4_DST_PORT` | Destination port |
| 5 | `PROTOCOL` | IP protocol (TCP=6, UDP=17, ICMP=1) |
| 6 | `L7_PROTO` | Application layer protocol |
| 7 | `IN_BYTES` | Incoming bytes |
| 8 | `OUT_BYTES` | Outgoing bytes |
| 9 | `IN_PKTS` | Incoming packet count |
| 10 | `OUT_PKTS` | Outgoing packet count |
| 11 | `TCP_FLAGS` | TCP flags (SYN, ACK, FIN, etc.) |
| 12 | `FLOW_DURATION_MILLISECONDS` | Flow duration in ms |

### Extended NetFlow (31 features — from v2)

| # | Feature | Description |
| --- | --- | --- |
| 13 | `CLIENT_TCP_FLAGS` | TCP flags from client side |
| 14 | `SERVER_TCP_FLAGS` | TCP flags from server side |
| 15-18 | `DURATION_IN/OUT` | Duration of in/out traffic |
| 19-22 | `MIN/MAX_TTL` | Min/max TTL in/out |
| 23-26 | `LONGEST/SHORTEST_FLOW_PKT` | Longest/shortest packet sizes |
| 27-30 | `MIN/MAX_IP_PKT_LEN` | Min/max IP packet lengths |
| 31-34 | `SRC/DST_TO_SRC/DST_PKTS` | Directional packet ratios |
| 35-38 | `SRC/DST_TO_SRC/DST_BYTES` | Directional byte ratios |
| 39-43 | Other extended fields | Retransmissions, window sizes, etc. |

### Temporal Features (10 features — NEW in v3)

| # | Feature | Description |
| --- | --- | --- |
| 44 | `FLOW_START_MILLISECONDS` | Flow start timestamp |
| 45 | `FLOW_END_MILLISECONDS` | Flow end timestamp |
| 46 | `SRC_TO_DST_IAT_MIN` | Min inter-arrival time (src→dst) |
| 47 | `SRC_TO_DST_IAT_MAX` | Max inter-arrival time (src→dst) |
| 48 | `SRC_TO_DST_IAT_AVG` | Avg inter-arrival time (src→dst) |
| 49 | `SRC_TO_DST_IAT_STDDEV` | Std dev inter-arrival time (src→dst) |
| 50 | `DST_TO_SRC_IAT_MIN` | Min inter-arrival time (dst→src) |
| 51 | `DST_TO_SRC_IAT_MAX` | Max inter-arrival time (dst→src) |
| 52 | `DST_TO_SRC_IAT_AVG` | Avg inter-arrival time (dst→src) |
| 53 | `DST_TO_SRC_IAT_STDDEV` | Std dev inter-arrival time (dst→src) |

### Labels

- `Label` — Binary (0=Benign, 1=Attack)
- `Attack` — Multi-class: Benign, BruteForce, DoS, DDoS, Web Attack, Bot, Infiltration

---

## Gotham 2025 Features (23 Packet-Level)

| # | Feature | Description |
| --- | --- | --- |
| 1 | `frame.time` | Packet arrival timestamp |
| 2 | `frame.len` | Packet length (bytes) |
| 3 | `frame.protocols` | Protocol stack (e.g., "eth:ip:tcp") |
| 4 | `eth.src` | Source MAC address |
| 5 | `eth.dst` | Destination MAC address |
| 6 | `ip.src` | Source IP address |
| 7 | `ip.dst` | Destination IP address |
| 8 | `ip.proto` | IP protocol number |
| 9 | `tcp.srcport` | TCP source port |
| 10 | `tcp.dstport` | TCP destination port |
| 11 | `tcp.flags` | TCP flags |
| 12 | `tcp.len` | TCP payload length |
| 13 | `udp.srcport` | UDP source port |
| 14 | `udp.dstport` | UDP destination port |
| 15 | `udp.len` | UDP payload length |
| 16 | `icmp.type` | ICMP message type |
| 17 | `icmp.code` | ICMP message code |
| 18 | `mqtt.msgtype` | MQTT message type |
| 19 | `mqtt.qos` | MQTT quality of service |
| 20 | `coap.type` | CoAP message type |
| 21 | `coap.code` | CoAP response code |
| 22 | `rtsp.seq` | RTSP sequence number |
| 23 | `rtsp.cmd` | RTSP command |

Labels

- `Label` — Multi-class: Benign, DoS, BruteForce, Scanning, C&C, CoAP Amplification, + others

---

## Alignment Strategy for Federated Learning

Since the two datasets have fundamentally different feature granularities, we have **three options**:

### Option A: Shared Semantic Features (Recommended ⭐)

Engineer a common set of **abstract features** that can be computed from both datasets:

| Shared Feature | From IDS2018 (flow) | From Gotham (packet → aggregate) |
| --- | --- | --- |
| `protocol` | `PROTOCOL` | `ip.proto` |
| `src_port` | `L4_SRC_PORT` | `tcp.srcport` or `udp.srcport` |
| `dst_port` | `L4_DST_PORT` | `tcp.dstport` or `udp.dstport` |
| `tcp_flags` | `TCP_FLAGS` | `tcp.flags` |
| `total_bytes` | `IN_BYTES + OUT_BYTES` | aggregate `frame.len` per flow |
| `total_pkts` | `IN_PKTS + OUT_PKTS` | count packets per flow |
| `flow_duration` | `FLOW_DURATION_MILLISECONDS` | max(frame.time) - min(frame.time) per flow |
| `byte_rate` | total_bytes / flow_duration | total_bytes / flow_duration |
| `pkt_rate` | total_pkts / flow_duration | total_pkts / flow_duration |
| `avg_pkt_size` | total_bytes / total_pkts | mean(frame.len) |

**Result: ~10-15 shared features** that all companies use for FL.

### Option B: Feature Projection (Neural)

Each company trains a **local encoder** that maps its raw features to a shared embedding space. The FL then operates on these embeddings. Inspired by Tri-LLM's semantic projection `ẑ = W·x`.

### Option C: Separate FL + Shared RAG

Run FL only among companies with the same dataset (A+B), and let Company C participate only through the RAG channel. Simpler but weaker.

### Recommendation

**Option A** for Phase 2. It's the most principled approach and demonstrates that your system handles heterogeneous data distributions — a key differentiator for the thesis.
