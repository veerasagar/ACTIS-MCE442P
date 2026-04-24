# Problem Analysis & Methodology — Descriptive Explanation

This document provides a paragraph-by-paragraph explanation of the problems ACTIS solves and the methodology it uses. Use this as a reference for your project report, viva, or presentation.

---

## Problem Analysis

### Problem 1: Data Silos in Cybersecurity

In real-world cybersecurity, every organization — whether it is a bank, a hospital, or a government agency — operates its own network and collects its own traffic data. When they build an Intrusion Detection System (IDS), they can only train it on the attacks **they have personally seen**. This means if Organization A has experienced a DDoS attack but Organization B has not, then B's IDS has no knowledge of DDoS patterns and will fail to detect it when it eventually happens. This is called the **data silo problem**. Each organization is isolated in its own bubble of knowledge, and the overall detection capability across the industry remains low because no single organization has visibility into the full threat landscape.

### Problem 2: Zero-Day Attack Blindness

Traditional IDS systems are **supervised classifiers** — they learn to recognize attack patterns from labeled training data. If an attack type was present in the training data, the model can detect it. But what happens when a completely new, never-before-seen attack arrives? The classifier has no training examples for it, so it either misclassifies the attack as benign traffic (letting it through) or assigns it to a wrong attack category. These unknown attacks are called **zero-day attacks** because there are "zero days" of prior knowledge about them. In the real world, zero-day attacks are the most dangerous because they exploit vulnerabilities that no one knows about yet, and no signatures or rules exist to catch them.

### Problem 3: Privacy Barriers to Collaboration

The obvious solution to the data silo problem would be to pool all organizations' data into one central dataset and train a single, powerful model. However, this is **impossible in practice** because network traffic data contains highly sensitive Personally Identifiable Information (PII) — IP addresses reveal network topology, port numbers reveal running services, flow patterns can fingerprint individual users, and payloads may contain credentials or personal data. Sharing this data between organizations would violate privacy regulations like GDPR and HIPAA, and would expose each organization to security risks. So the challenge becomes: **how do we enable collaborative learning without sharing raw data?**

### Problem 4: Intelligence Fragmentation

Even when organizations do detect and analyze threats, the resulting threat intelligence (reports, indicators of compromise, recommended defenses) is typically stored locally and not shared systematically. If Organization A detects a new ransomware variant on Monday, Organization B has no way of knowing about it until they independently discover the same threat — which could be days or weeks later. This **fragmentation of intelligence** means that each organization is fighting threats alone, when collective defense would be far more effective. The cybersecurity industry lacks a system for automatically sharing actionable intelligence across organizational boundaries while preserving privacy.

### Problem 5: Low-Quality Automated Reports

When an IDS detects an attack, it generates an alert. But raw alerts are often just a class label (e.g., "DDoS") with no context about the attack mechanism, no mapping to standardized frameworks like MITRE ATT&CK, no recommended defenses, and no assessment of severity. Security analysts receive hundreds or thousands of such low-quality alerts daily and must manually investigate each one. This alert fatigue leads to slow response times. The challenge is to generate **rich, semantically coherent, and actionable threat reports** automatically — reports that a human analyst can immediately act upon.

---

## Methodology

### Overview

ACTIS addresses all five problems through an integrated system built on four technical pillars: **Federated Learning**, **Zero-Day Detection**, **Privacy-Preserving Intelligence**, and **Agentic RAG**. Each pillar directly solves one or more of the identified problems.

---

### Pillar 1: Federated Learning — Solving the Data Silo Problem

Federated Learning (FL) is a distributed machine learning technique where multiple organizations collaboratively train a shared model **without ever exchanging their raw data**. In ACTIS, we simulate 4 organizations (Nodes A, B, C, and D), each holding a different portion of network traffic data. Nodes A and B hold enterprise network data (from the NF-CSE-CIC-IDS2018 dataset), while Nodes C and D hold IoT botnet data (from the NF-BoT-IoT dataset). This deliberate heterogeneity creates a **non-IID** (non-identically distributed) setting — the most challenging and realistic scenario for federated learning.

The FL process works in rounds. In each round, the central server sends the current global model to all 4 clients. Each client trains the model on its own private data for a few epochs, then sends back only the updated model **weights** (numerical parameters) — not the data itself. The server then computes a weighted average of all clients' weights, where each client's contribution is proportional to its dataset size. This averaged model becomes the new global model, and the process repeats.

The algorithm used is **FedAvg** (Federated Averaging), the most widely adopted FL algorithm. For handling the non-IID data distribution, we also support **FedProx**, which adds a proximal regularization term that prevents any single client's model from drifting too far from the global model.

After 10 rounds of federated training, the global model has effectively learned from all 4 organizations' combined data — including enterprise DDoS attacks, IoT botnets, brute force attacks, and reconnaissance patterns — without any organization having revealed its private data to anyone else.

---

### Pillar 2: Zero-Day Detection — Catching the Unseen

ACTIS uses a **dual-signal hybrid anomaly detection** approach to identify zero-day attacks. This is one of our key innovations.

**Signal 1 — Classifier Confidence**: When the IDS neural network classifies a flow, it produces a probability distribution over all known attack classes via the softmax function. If the model is confident, one class will have a very high probability (e.g., 0.95 for DDoS). But if the flow is from an unknown attack type, the probabilities will be spread more evenly across classes — resulting in high entropy. We measure this uncertainty using the formula: `ZDS_conf = λ × normalized_entropy + (1-λ) × (1 - max_confidence)`. High ZDS_conf means the classifier is confused, which is a signal that this could be a zero-day attack.

**Signal 2 — Reconstruction Error**: We train small autoencoder neural networks on known traffic data — but critically, we train **separate autoencoders for each network protocol** (TCP, UDP, ICMP). An autoencoder learns to compress and reconstruct its training data. If we feed it data that looks different from what it was trained on, it will produce a high reconstruction error. For example, the UDP autoencoder learns the patterns of normal UDP traffic (DNS queries, NTP sync, etc.). When a DDoS UDP flood arrives, the traffic patterns are drastically different — massive packet counts, huge byte volumes, unusual port distributions — and the autoencoder cannot reconstruct them well, producing a high error signal.

**Combined Score**: We combine both signals: `ZDS = α × confidence_score + (1-α) × reconstruction_error`. If this combined score exceeds a calibrated threshold (set at the 90th or 95th percentile of training scores), the flow is flagged as a potential zero-day attack.

The key innovation of per-protocol autoencoders is that DDoS attacks (which primarily use UDP and ICMP) produce dramatically higher reconstruction errors when evaluated against protocol-specific autoencoders compared to a single global autoencoder. This is because the protocol-specific model has a much tighter learned distribution, making anomalies more distinguishable.

---

### Pillar 3: Privacy-Preserving Intelligence — Zero PII Leakage

Privacy is enforced through a **three-layer defense-in-depth** architecture:

**Layer 1 — Differential Privacy (DP)**: Before any model weights are transmitted from a local client to the FL server, calibrated Gaussian noise is injected into the weight matrices. Differential Privacy provides a mathematical guarantee (parameterized by epsilon ε and delta δ) that the shared weights cannot be reverse-engineered to reconstruct the original training data. During training, the DP layer actively adds noise to the neural network's intermediate computations. During evaluation/inference, the layer becomes deterministic (no noise) to ensure consistent predictions.

**Layer 2 — Agent Sanitizer**: When a threat is detected and a report is generated, the Agent Sanitizer scrubs all PII from the report before it leaves the organization. It uses 12 regex patterns to detect and replace: IPv4 addresses, IPv6 addresses, email addresses, MAC addresses, hostnames/FQDNs, phone numbers, Social Security Numbers, credit card numbers, and other sensitive identifiers. Each detected PII element is replaced with a typed placeholder (e.g., `192.168.1.100` becomes `[REDACTED_IP]`).

**Layer 3 — PII Validator**: As a final safety check, the PII Validator recursively scans every field in the sanitized report — including nested dictionaries and lists — using the same regex patterns. If any PII is still detected, the report is **rejected** and not shared. This defense-in-depth approach means that even if the sanitizer misses something, the validator catches it. In all our testing, the measured PII leakage rate is **0%**.

---

### Pillar 4: Agentic RAG Intelligence — Cross-Organizational Immunity

Once a sanitized threat report passes the PII Validator, it enters the **Global Intelligence Hub** on the ACTIS server. This hub consists of four components working together:

**Intel Aggregator**: Receives the sanitized report and normalizes it into a standardized format containing the attack type, MITRE ATT&CK technique ID, severity level, and recommended defenses.

**ChromaDB Vector Store**: The standardized report is converted into a 384-dimensional numerical vector (embedding) using the `all-MiniLM-L6-v2` sentence transformer model, and stored in ChromaDB — a vector database optimized for semantic similarity search. This means we can search for threats by meaning, not just keywords. Searching for "UDP amplification attack" will also find reports about "DNS reflection floods" because they are semantically similar.

**Agentic RAG Engine**: When any organization needs threat intelligence — either triggered by a new detection or by an analyst query — the RAG (Retrieval-Augmented Generation) engine performs a three-step process: (1) retrieve candidate reports from ChromaDB using embedding similarity, (2) rerank candidates using a cross-encoder model for higher precision, and (3) apply an abstention threshold of 0.30 — if no candidate scores above this threshold, the system responds with "no relevant intelligence found" rather than returning irrelevant or hallucinated information. This abstention mechanism is critical for trust and reliability.

**Immunity Engine**: The most impactful component. When Organization A detects a zero-day attack, the Immunity Engine automatically generates preventative firewall rules (in Snort and iptables format) based on the attack characteristics. These rules are distributed to Organizations B, C, and D — protecting them from the same attack **before they even encounter it**. This creates a "herd immunity" effect: one organization's detection becomes every organization's defense.

---

### Threat Summarization

For report quality (Problem 5), ACTIS uses **Gemini 1.5 Flash LLM** to generate human-readable threat summaries enriched with MITRE ATT&CK context, severity assessments, and recommended defenses. Each summary includes the attack type, the MITRE technique ID (e.g., T1498 for Network Denial of Service), the tactic category (e.g., Impact), flow-level indicators, and specific mitigation steps. For environments without LLM API access, a deterministic fallback generates structured summaries using template-based rules — ensuring the system works even offline.

The quality of these summaries is evaluated using **BERTScore**, which measures semantic similarity between generated and reference text using BERT embeddings. Our system achieves an F1 BERTScore of 0.919, which is competitive with CyberRAG's reported 0.94 while offering the additional benefits of federated learning and privacy preservation that CyberRAG does not provide.
