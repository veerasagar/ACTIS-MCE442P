# SYNAPSE - SYNTHETIC NETWORK-TRAFFIC AGENT PIPELINE FOR FEDERATED EDGE SECURITY

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A privacy-preserving federated intrusion detection system that combines **federated learning**, **retrieval-augmented generation (RAG)**, **zero-day detection**, and **real-time deployment** across multi-organization networks.

---

## Overview

SYNAPSE enables multiple organizations to collaboratively train a shared intrusion detection model **without sharing raw network data**. Each organization retains full control of its traffic — only sanitized threat intelligence and DP-noised model weights leave the local node.

The system detects **9 attack classes** across enterprise and IoT networks, generates MITRE ATT&CK-mapped threat reports, and deploys as a real-time monitoring pipeline.

---

## Key Features

- **4-Node Federated Learning** — Cross-dataset FL across enterprise (CIC-IDS2018) and IoT (BoT-IoT) traffic
- **Zero-Day Detection** — Per-protocol autoencoders (TCP/UDP/ICMP) with 99.5% DDoS detection rate
- **RAG with Abstention** — Cross-encoder reranking + hallucination prevention via confidence thresholds
- **Privacy Stack** — PII sanitization (0% leakage) + (ε, δ)-differential privacy on model gradients
- **66-Dim Feature Engineering** — 25 derived features including TCP flag decomposition and window ratio analysis
- **LLM-Enhanced Reports** — Gemini-powered threat summaries for high-severity events
- **Live Deployment** — Real-time CSV, log tailing, and nfdump pipe support

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- Kaggle API credentials (for dataset download)

### Setup

```bash
git clone https://github.com/your-username/MCE442P.git
cd MCE442P
bash scripts/setup.sh
```

This installs all dependencies and downloads both datasets (~3 GB).

### Run Validations

```bash
# Run all 9 validation phases
bash scripts/run.sh

# Run specific phases (e.g., FL training + zero-day + ReGAIN benchmark)
bash scripts/run.sh 3 5 8
```

| Phase | Validation | Command |
| :---: | :--- | :--- |
| 1 | Data pipeline + feature engineering | `bash scripts/run.sh 1` |
| 2 | Local IDS (no federation) | `bash scripts/run.sh 2` |
| 3 | Federated learning (4 nodes) | `bash scripts/run.sh 3` |
| 4 | PII sanitization + DP noise | `bash scripts/run.sh 4` |
| 5 | Zero-day DDoS detection | `bash scripts/run.sh 5` |
| 6 | RAG + cross-org intelligence | `bash scripts/run.sh 6` |
| 7 | BERTScore report quality | `bash scripts/run.sh 7` |
| 8 | ReGAIN benchmark replication | `bash scripts/run.sh 8` |
| 9 | Full evaluation suite | `bash scripts/run.sh 9` |

### Live Monitor

```bash
# Process a CSV file
python3 -m src.live_monitor --input flows.csv --company A

# Tail a growing log
python3 -m src.live_monitor --tail /var/log/netflows.csv

# Pipe from nfdump
nfdump -r capture.nfcapd -o csv | python3 -m src.live_monitor --stdin
```

---

## Results

### vs Baseline Papers

| Metric | ReGAIN | Tri-LLM | CyberRAG | **SYNAPSE** |
| :--- | :---: | :---: | :---: | :---: |
| TCP SYN Flood Acc | 98.82% | — | — | **100.00%** |
| ICMP/UDP Flood Acc | 95.95% | — | — | **99.45%** |
| FL Accuracy (non-IID) | — | 85.6% | — | **98.4%** |
| Zero-Day DDoS | ~60% | 68% | — | **99.5%** |
| BERTScore F1 | — | — | 0.94 | 0.919 |
| PII Protection | ❌ | ❌ | ❌ | **✅ 0%** |
| Real-Time Deploy | ❌ | ❌ | ❌ | **✅** |

### Federated Learning (4 Nodes, 5 Rounds)

| Node | Dataset | Accuracy | F1 |
| :--- | :--- | :---: | :---: |
| A | CIC-IDS2018 | 76.1% | 0.8205 |
| B | CIC-IDS2018 | 70.8% | 0.7867 |
| C | BoT-IoT | 94.9% | 0.9560 |
| D | BoT-IoT | 94.9% | 0.9580 |

---
