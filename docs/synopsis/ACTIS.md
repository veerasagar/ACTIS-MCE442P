# Synopsis

**College**: RV COLLEGE OF ENGINEERING®
**Department**: COMPUTER SCIENCE AND ENGINEERING
**Program**: M.Tech in CSE / CNE

**Major Project**
**Course Code**: [Course Code]

**Student Name**: Veerasagar
**USN**: [Your USN]

**Project Title**: ACTIS: Agentic Collaborative Threat Intelligence System via Trust-Aware Federated Learning
**Undertaken at**: [Company Name or RVCE]

**Internal Guide Name**: [Internal Guide Name]
**Designation**: [Guide Designation]

**External Guide**: [External Guide Name, if applicable]

---

## SYNOPSIS

### Introduction

As cyber threats scale in sophistication, isolated organizational defenses are consistently outpaced by zero-day attacks. While sharing network telemetry could create a powerful collective defense, strict data privacy regulations (e.g., GDPR) and the risk of exposing Personally Identifiable Information (PII) prohibit pooling raw network traffic. Current collaborative architectures either expose sensitive data through centralization or completely isolate nodes, preventing the discovery of cross-organizational attack patterns. This project proposes **ACTIS (Agentic Collaborative Threat Intelligence System)**, a dual-channel framework that leverages **Trust-Aware Federated Learning (FL)** to collectively train PyTorch-based Intrusion Detection Systems (IDS) without exposing raw data, and a **Federated Retrieval-Augmented Generation (RAG)** pipeline governed by an **Agentic Privacy Engine** to safely structure, sanitize, and share actionable zero-day threat intelligence.

### Objectives

1. **Decentralized IDS Training:** Implement a PyTorch Multilayer Perceptron (MLP) IDS using the FedProx algorithm to counter statistical heterogeneity (non-IID data) across distributed nodes.
2. **Resilient Aggregation:** Develop a "Trust-Aware FedAvg" strategy at the central server to dynamically penalize underperforming or compromised nodes using empirical loss metrics.
3. **Agentic Privacy & PII Sanitization:** Employ an LLM-driven Agentic Privacy Engine (using LangChain and Gemini) with Regex fail-safes to guarantee 0% PII leakage (stripping IPs, MACs, credentials) while mapping threats to MITRE ATT&CK tactics.
4. **Mathematical Privacy Guarantees:** Engineer a Differential Privacy (DP) layer that applies calibrated Gaussian noise to semantic embeddings, ensuring a formal \((\epsilon, \delta)\)-DP guarantee.
5. **Automated Zero-Day Immunity:** Build a Global Federated RAG Knowledge Base (ChromaDB) to perform semantic searches, powering an Immunity Engine that autonomously generates deployable perimeter defenses (e.g., `iptables` rules).

### Methodology and Working

The methodology relies on an end-to-end edge-to-cloud security pipeline structured into several distinct technical stages:

1. **Heterogeneous Data Processing:** NetFlow v2 data from the NF-CSE-CIC-IDS2018-v2 (Enterprise) and NF-BoT-IoT-v2 (IoT) datasets are ingested. The 43-column schema is normalized to a 41-dimensional feature vector. The target variables are mapped to a unified taxonomy (e.g., DDoS, DoS, brute-force, botnet). Inverse-frequency class weighting and StandardScaler normalization are applied locally at each partitioned node (simulating non-IID company environments A, B, and C).
2. **Local Model Training (FedProx):** Each node trains an MLP architecture (hidden layers: 128, 64, 32 with 0.3 Dropout) using the Adam optimizer. To prevent divergence on non-IID data, the local Cross-Entropy loss incorporates a FedProx proximal term \((\frac{\mu}{2} ||w - w_{global}||^2)\), directly regularizing the local weights against the global baseline.
3. **Trust-Aware Global Aggregation:** Local weights and training losses (\(L_i\)) are transmitted to the FL Server. The server computes a trust score \( \tau_i = \frac{1}{L_i + \epsilon} \). The final client aggregation weight is a clamped 50/50 blend of the standard sample volume ratio and the calculated trust score ratio. This guarantees that nodes with anomalous loss profiles cannot maliciously skew the global IDS model.
4. **Agentic Privacy Engine:** When a trained local IDS flags an anomalous flow, the LangChain-orchestrated **Agent Analyzer** queries the Gemini 2.0 Flash LLM to inspect the payload dimensions and port behaviors, outputting a structured MITRE ATT&CK analysis. The **Agent Sanitizer** subsequently redacts structural PII identifiers, utilizing a **PII Validator** in a strict retry loop to guarantee zero leakage.
5. **Differential Privacy Layer:** The sanitized threat report is transformed into a 384-dimensional dense vector using the `all-MiniLM-L6-v2` SentenceTransformer. The DP Layer calculates a noise multiplier \(\sigma = \frac{\Delta f \sqrt{2 \ln(1.25/\delta)}}{\epsilon}\) and injects Gaussian noise into the embedding. This provides deniability against membership inference attacks on the vector store.
6. **Federated RAG & Immunity Engine:** The differentially private embeddings and sanitized JSON metrics are pushed to a central ChromaDB instance containing separate collections for threats, MITRE references, and defense patterns. Using Maximal Marginal Relevance (MMR) search, the server's **Immunity Engine** retrieves zero-day threat patterns discovered by Node A, interprets them, and automatically synthesizes actionable firewall configurations (e.g., `iptables --limit`, ModSecurity WAF rules) to proactively immunize Node B and C against identical lateral movements.

### Software Requirements

- **Programming Language:** Python 3.10+
- **Machine Learning & FL:** PyTorch 2.0+, Scikit-learn, Numpy, Pandas
- **Agentic AI & NLP:** LangChain, `langchain-google-genai` (Google Gemini 2.0 API), `sentence-transformers`
- **Vector Database:** ChromaDB
- **Interfaces & IPC:** Typer, Rich (Terminal Dashboard UI), ZeroMQ

### Hardware Requirements

- **Processor:** Multi-core modern CPU (Intel i7/i9 or AMD Ryzen 7/9 / Apple Silicon)
- **RAM:** Minimum 16 GB (32 GB recommended for simultaneous local training simulations)
- **Storage:** 50 GB NVMe SSD for fast localized Pandas chunking and Parquet queries.
- **GPU (Recommended):** NVIDIA GPU (CUDA supported) for accelerated model training and tensor evaluations.

### Innovation / Contributions

- **Trust-Aware Proximal Federation:** Solves the notorious non-IID data distribution issue in network security by marrying FedProx regularization with a dynamic, loss-inversely-proportional client trust metric.
- **Deniable Semantic Threat Vectorization:** Successfully bounds the mathematical risk of data recovery from shared threat reports by injecting calibrated \((\epsilon, \delta)\)-DP Gaussian noise specifically into the 384-dimensional semantic embedding space.
- **Agentic Redaction Pipeline:** Demonstrates that Large Language Models—when constrained by static Regex validators in a closed retry loop—can reliably understand networking context and achieve 0% PII leakage without rendering the threat summary semantically useless.
- **Zero-Day Actionable RAG:** Pioneers the concept of "Digital Immunity" where semantic similarities across cross-organizational attacks actively trigger the code generation of concrete, rule-based perimeter configurations using Retrieval-Augmented Generation.

---

**Internal Guide**                           **Signature of Dean CSE-Cluster**                **Signature of the HOD**
Name: [Guide Name]                      Name: Dr. Ramakanthkumar P                       Name: Dr. Shanta Rangaswamy
Designation: [Guide Desig.]             Professor and Dean CSE-Cluster                   Prof. & Head of the Dept, CSE
Dept. of CSE, RVCE                           Dept. of CSE, RVCE                                          RVCE
