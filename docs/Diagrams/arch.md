# ACTIS Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│  LOCAL ORGANIZATIONS (Clients: Nodes A, B, C, D)                       │
│                                                                        │
│  ┌──────────────────────────────┐                                      │
│  │ Local Detection Pipeline     │                                      │
│  │                              │                                      │
│  │  [Raw NetFlow/PCAP]          │                                      │
│  │         │                    │                                      │
│  │         ▼                    │                                      │
│  │  [Feature Engineering]       │                                      │
│  │  (41 to 66 dimensions)       │                                      │
│  │         │                    │                                      │
│  │         ▼                    │<···(Global Weights)··············┐   │
│  │  [IDS Classifier]            │                                  ·   │
│  │  (PyTorch Neural Net)        │····(Local Weights + DP Noise)·┐  ·   │
│  │         │                    │                               ·  ·   │
│  │         ▼                    │                               ·  ·   │
│  │  [Zero-Day Detector]         │                               ·  ·   │
│  │  (Autoencoder & Entropy)     │                               ·  ·   │
│  └─────────│────────────────────┘                               ·  ·   │
│            │ (Threat Detected)                                  ·  ·   │
│            ▼                                                    ·  ·   │
│  ┌──────────────────────────────┐                               ·  ·   │
│  │ Privacy & Intelligence Layer │                               ·  ·   │
│  │                              │                               ·  ·   │
│  │  [Agent Analyzer] <···············(Cross-Org Context)····┐   ·  ·   │
│  │  (MITRE Mapping)             │                           ·   ·  ·   │
│  │         │                    │                           ·   ·  ·   │
│  │         ▼                    │                           ·   ·  ·   │
│  │  [Agent Sanitizer]           │                           ·   ·  ·   │
│  │  (PII Scrubbing)             │                           ·   ·  ·   │
│  │         │                    │                           ·   ·  ·   │
│  │         ▼                    │                           ·   ·  ·   │
│  │  [PII Validator]             │                           ·   ·  ·   │
│  │  (Leakage Prevention)        │                           ·   ·  ·   │
│  └─────────║────────────────────┘                           ·   ·  ·   │
└────────────╫────────────────────────────────────────────────┼───┼──┼───┘
             ║ (Sanitized Threat Report)                      ·   ·  ·
             ║                                                ·   ·  ·
┌────────────╫────────────────────────────────────────────────┼───┼──┼───┐
│            ║                                                ·   ·  ·   │
│  GLOBAL ACTIS SERVER                                        ·   ·  ·   │
│            ║                                                ·   ·  ·   │
│            ║        ┌──────────────────────────────────┐    ·   ·  ·   │
│            ║        │  [Federated Learning Server]     │<···┘   ·  ·   │
│            ║        └──────────────────────────────────┘        ·  ·   │
│            ║                                                    ·  ·   │
│  ┌─────────╫────────────────────────────────────────┐           ·  ·   │
│  │ Cross-Org Intelligence Hub                       │           ·  ·   │
│  │         ▼                                        │           ·  ·   │
│  │  [Intel Aggregator]                              │           ·  ·   │
│  │         │                                        │           ·  ·   │
│  │         ▼                                        │           ·  ·   │
│  │  [Global Vector DB] <───┐                        │           ·  ·   │
│  │  (ChromaDB)             │                        │           ·  ·   │
│  │         │               │                        │           ·  ·   │
│  │         ▼               │                        │           ·  ·   │
│  │  [Agentic RAG Engine] ──┘                        │           ·  ·   │
│  │  (Cross-Encoder) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ┘  ·   │
│  │         │                                        │              ·   │
│  │         ▼                                        │              ·   │
│  │  [Immunity Engine]                               │              ·   │
│  │  (Zero-Day Rules) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─ ┘   │
│  └──────────────────────────────────────────────────┘                  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Problem Analysis & Methodology

### 1. Problem Statement

Traditional Intrusion Detection Systems (IDS) suffer from five critical limitations that ACTIS addresses:

| # | Problem | Impact |
|---|---|---|
| P1 | **Data Silos** — Organizations train IDS models only on their own local data, missing attack patterns seen by other organizations. | Low detection accuracy on unseen attack types, especially in non-IID (heterogeneous) data environments. |
| P2 | **Zero-Day Blindness** — Signature-based and supervised classifiers cannot detect previously unseen attack categories. | Novel attacks (e.g., new DDoS variants) bypass defenses entirely until manual signature updates. |
| P3 | **Privacy Violations** — Sharing raw network data between organizations exposes sensitive PII (IP addresses, internal topology, user identifiers). | Legal and regulatory non-compliance (GDPR, HIPAA), making cross-org collaboration infeasible. |
| P4 | **Intelligence Fragmentation** — Threat intelligence generated at one site is not systematically shared or actionable at other sites. | Each organization independently rediscovers the same threats, wasting resources and response time. |
| P5 | **Report Quality** — Automated threat summaries often lack context, MITRE alignment, and semantic coherence. | Security analysts cannot act on low-quality alerts, increasing mean time to respond (MTTR). |

---

### 2. Proposed Methodology

ACTIS solves these problems through a **four-pillar methodology**:

#### Pillar 1: Federated Learning (Solves P1)

**Approach**: Train a shared IDS model across 4 organizations without exchanging raw data.

- **Algorithm**: FedAvg (Federated Averaging) with optional FedProx regularization for non-IID robustness.
- **Architecture**: 4 local clients (Nodes A, B, C, D) and 1 central aggregation server.
- **Data Split**: Nodes A & B receive NF-CSE-CIC-IDS2018-v2 data (enterprise network attacks). Nodes C & D receive NF-BoT-IoT-v2 data (IoT botnet attacks). This creates a deliberately **non-IID** setting.
- **Process per round**:
  1. Server distributes global model weights to all clients.
  2. Each client trains locally for 5 epochs on its private data.
  3. Each client sends updated weights (not data) back to the server.
  4. Server computes weighted average: `W_global = Σ(n_k / N) × W_k`
  5. Repeat for 10 rounds.

#### Pillar 2: Zero-Day Detection (Solves P2)

**Approach**: A dual-signal anomaly detector that catches attacks the classifier has never seen.

- **Signal 1 — Confidence Score**: Measures entropy of the softmax output. High entropy = model is uncertain = possible unknown attack.
  - Formula: `ZDS_conf = λ × entropy(softmax) + (1-λ) × (1 - max_confidence)`
- **Signal 2 — Reconstruction Error**: Per-protocol autoencoders (TCP, UDP, ICMP) trained only on known traffic. Unknown attacks produce high reconstruction error because the autoencoder has never learned their patterns.
- **Combined Score**: `ZDS = α × ZDS_conf + (1-α) × ZDS_recon`
- **Threshold Calibration**: Set at the p-th percentile (e.g., 90th or 95th) of training ZDS scores.
- **Key Innovation**: Protocol-specific autoencoders — a UDP autoencoder distinguishes DDoS (malicious UDP floods) from normal UDP traffic far better than a single global autoencoder.

#### Pillar 3: Privacy-Preserving Intelligence (Solves P3)

**Approach**: A three-layer privacy pipeline ensures zero PII leakage.

| Layer | Component | Technique |
|---|---|---|
| Layer 1 | **Differential Privacy** | Gaussian noise injection into model weights before sharing (`ε-δ` privacy guarantee) |
| Layer 2 | **Agent Sanitizer** | 12 regex patterns scrub IPs, emails, hostnames, MAC addresses, phone numbers, SSNs from threat reports |
| Layer 3 | **PII Validator** | Recursive dictionary scanner rejects any report that still contains detectable PII |

- **Measured Leakage Rate**: 0% across all test cases.
- **DP Layer Behavior**: Active during training (adds noise), deterministic during evaluation (no noise).

#### Pillar 4: Agentic RAG Intelligence (Solves P4 & P5)

**Approach**: Retrieval-Augmented Generation with cross-organizational threat sharing.

- **Storage**: ChromaDB vector database stores sanitized threat reports as 384-dim embeddings (via `all-MiniLM-L6-v2` sentence-transformer).
- **Retrieval**: Semantic similarity search finds related threats across all organizations.
- **Reranking**: Cross-encoder model reranks candidates for higher precision.
- **Abstention**: If the best retrieval score < 0.30 threshold, the system returns "no relevant intelligence found" instead of hallucinating.
- **Immunity**: The Immunity Engine automatically generates Snort/iptables firewall rules from RAG results and pushes them to all other nodes — providing **herd immunity**.
- **Threat Summarization**: Gemini 1.5 Flash LLM generates human-readable threat summaries with MITRE ATT&CK context, with a deterministic fallback for environments without API access.

---

### 3. Datasets

| Dataset | Records | Attack Types | Source | Role in ACTIS |
|---|---|---|---|---|
| NF-CSE-CIC-IDS2018-v2 | ~17M flows | DDoS, DoS, Brute Force, Web Attack, Infiltration, Botnet | Canadian Institute for Cybersecurity | Nodes A & B (enterprise traffic) |
| NF-BoT-IoT-v2 | ~30M flows | DDoS, DoS, Reconnaissance, Theft, Benign | UNSW Sydney | Nodes C & D (IoT traffic) |

- **Features**: 41 raw NetFlow features (protocol, ports, bytes, packets, duration, TCP flags, etc.)
- **Enhanced Features**: 66 after feature engineering (adds ratios, rates, burst indicators)
- **Format**: Apache Parquet (columnar, compressed)

---

### 4. Evaluation Strategy

| Evaluation | Metric | Baseline Comparison | Expected Result |
|---|---|---|---|
| FL 4-Node (10 rounds) | Accuracy, F1 | Local-only training | FL improves accuracy by 5-15% over local-only |
| Zero-Day DDoS | Detection Rate | Tri-LLM paper | 99.5% detection rate (vs 85.6% in Tri-LLM) |
| TCP SYN Flood | Accuracy | ReGAIN paper | 100% (vs 98.82% in ReGAIN) |
| ICMP/UDP Flood | Accuracy | ReGAIN paper | 99.5% (vs 95.95% in ReGAIN) |
| Report Quality | BERTScore F1 | CyberRAG paper | 0.919 (vs 0.94 in CyberRAG) |
| PII Leakage | Leakage Rate | — | 0% |
| RAG Abstention | Hallucination Prevention | — | 100% abstention on off-topic queries |
| Feature Engineering | Dimension Expansion | — | 41 → 66 features |