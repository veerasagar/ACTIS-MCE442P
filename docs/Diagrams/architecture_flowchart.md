# ACTIS — Architecture Diagrams & Explanation Guide

This document provides a detailed explanation of every component in the **Design Flowchart** and the **Design Structure Chart**, along with a glossary of all key technical terms used in the project.

---

## Part 1: Design Flowchart — Component Descriptions

The flowchart shows the end-to-end journey of a single network packet from raw capture to actionable intelligence.

---

### Layer 1: Data Ingestion Layer

| Component | What It Does | How to Explain It |
|---|---|---|
| **Raw NetFlow / PCAP Data** | The starting point. This is the raw network traffic collected from routers, switches, or packet capture tools. | "Our system begins by ingesting raw network telemetry. NetFlow records provide metadata like source IP, destination port, bytes transferred, and protocol. PCAP provides full packet captures." |
| **FedIntelDataLoader** | Loads two large-scale benchmark datasets (NF-CSE-CIC-IDS2018 with ~17M rows, and NF-BoT-IoT with ~30M rows) and partitions them across 4 simulated organizations (Nodes A, B, C, D). | "The data loader simulates a real-world federated environment where each organization only has access to its own local dataset. Nodes A and B get IDS2018 data, while Nodes C and D get BoT-IoT data, creating a non-IID (non-identically distributed) setting." |
| **Feature Engineer** | Takes the 41 raw numerical features and derives 25 additional statistical features (ratios, rates, flags), expanding the feature vector from 41 to 66 dimensions. | "We perform feature engineering to extract hidden patterns. For example, we compute bytes-per-packet ratio, flow asymmetry, and burst indicators. This improves the model's ability to distinguish subtle attack patterns from normal traffic." |
| **Preprocessor** | Handles label encoding, train/test splitting, and class balancing. Creates a unified label space so all 4 nodes share the same class indices. | "The preprocessor ensures that all organizations use a consistent label encoding — so class index 3 always means 'DDoS' across all nodes. This is critical for federated aggregation to work correctly." |

---

### Layer 2: Local Detection Engine

| Component | What It Does | How to Explain It |
|---|---|---|
| **IDS Neural Network Classifier** | A PyTorch-based deep neural network (3 hidden layers with batch normalization, dropout, and ReLU activations) that classifies each flow into one of 9 attack categories or 'benign'. | "The IDS classifier is a multi-class deep neural network. It takes the 66-dimensional feature vector as input and outputs a probability distribution over 9 attack types. The class with the highest probability is the prediction." |
| **Benign → Log** | If the classifier predicts the traffic as benign with high confidence, it is simply logged. No further action is taken. | "Most network traffic is legitimate. If the classifier is confident the flow is benign, we log it and move on. This keeps the system efficient — only suspicious traffic triggers the full analysis pipeline." |
| **Zero-Day Detector** | A dual-signal anomaly detection system that catches attacks the classifier has never seen before. It combines: (1) **Confidence Signal** — high entropy in the softmax output means the classifier is uncertain, and (2) **Reconstruction Signal** — per-protocol autoencoders trained only on known traffic; unseen attacks produce high reconstruction error. | "This is our key innovation. Even if the classifier says 'benign' or misclassifies a new attack, the zero-day detector can still catch it. The autoencoder has only learned what normal TCP, UDP, and ICMP traffic looks like. When a DDoS flood arrives, the autoencoder can't reconstruct it properly, and the high reconstruction error flags it as a potential zero-day." |
| **Agent Analyzer** | Maps detected threats to the MITRE ATT&CK framework (assigns Technique IDs like T1498, Tactics like 'Impact', and Severity levels). Uses Gemini LLM for enhanced semantic analysis with a deterministic fallback. | "Once a threat is detected, the Agent Analyzer enriches it with context. It assigns a MITRE ATT&CK technique ID (the global standard for cyber threat classification), determines the severity level (LOW to CRITICAL), and generates a human-readable description of the attack and recommended defenses." |
| **Agent Sanitizer** | Strips all Personally Identifiable Information (PII) from the threat report using regex patterns and LLM-assisted masking. Replaces IPs with `[REDACTED_IP]`, emails with `[REDACTED_EMAIL]`, etc. | "Before sharing any intelligence with other organizations, we must protect privacy. The sanitizer uses 12 regex patterns to find and replace IP addresses, email addresses, hostnames, MAC addresses, and other sensitive fields. This ensures compliance with data protection regulations." |
| **PII Validator** | A strict final check that recursively scans every field in the sanitized report. If any PII is found, the report is rejected and not shared. | "This is the last line of defense. Even after sanitization, we run a full recursive scan. Our measured leakage rate is 0% — meaning no PII has ever escaped through our pipeline in testing." |

---

### Layer 3: Federated Learning Loop

| Component | What It Does | How to Explain It |
|---|---|---|
| **FL Client** | Each of the 4 organizations runs a local FL Client. It trains the IDS model on its own private data for a few epochs, then sends only the model weights (not the data) to the server. | "In federated learning, the raw data never leaves the organization. Each FL Client trains a local copy of the neural network on its private dataset, then shares only the learned parameters (weights and biases) with the central server." |
| **DP Noise + Weights** | Before sending weights to the server, Differential Privacy noise is injected. This mathematically guarantees that the weights cannot be reverse-engineered to reveal the training data. | "We add calibrated Gaussian noise to the model weights before sharing. This provides a mathematical privacy guarantee — even if an adversary intercepts the weights, they cannot reconstruct the original training data." |
| **FL Server (FedAvg Aggregation)** | Receives weights from all 4 clients, computes a weighted average (proportional to each client's dataset size), and produces a single improved global model. | "The server performs Federated Averaging (FedAvg). It takes the weights from all 4 nodes, computes a weighted average based on how many samples each node trained on, and creates a single global model that has learned from everyone's data without seeing anyone's data." |
| **Global Weights** | The averaged global model is sent back to all clients, who use it as their starting point for the next round of local training. | "After aggregation, the improved global model is distributed back to all organizations. This creates a positive feedback loop — each round of training improves the model for everyone." |

---

### Layer 4: Global Intelligence Hub

| Component | What It Does | How to Explain It |
|---|---|---|
| **Intel Aggregator** | Receives sanitized threat reports from the PII Validator and normalizes them into a consistent format for storage. | "The aggregator is the entry point into the global knowledge base. It takes the sanitized threat report and structures it into a standardized format with fields like attack type, MITRE technique, severity, and recommended defense." |
| **ChromaDB Vector Store** | A vector database that stores threat intelligence as high-dimensional embeddings. Enables semantic similarity search — finding threats that are 'similar in meaning' not just 'matching keywords'. | "We use ChromaDB as our vector database. Each threat report is converted into a 384-dimensional embedding using a sentence transformer model. This allows us to search by meaning — for example, searching 'UDP amplification' will also find reports about 'DNS reflection attacks' because they are semantically similar." |
| **RAG Engine (Cross-Encoder Reranking)** | Retrieval-Augmented Generation engine that queries the vector store, reranks results using a cross-encoder for higher precision, and includes an abstention mechanism (threshold = 0.30) to prevent hallucinations. | "The RAG engine is our intelligence retrieval system. When queried, it first retrieves candidate threats from ChromaDB using embedding similarity, then reranks them using a cross-encoder model for higher accuracy. Critically, if no result scores above our 0.30 threshold, the system abstains rather than returning irrelevant information — this prevents hallucinations." |
| **Immunity Engine** | Takes RAG query results and automatically generates preventative firewall/Snort rules that can be deployed to other organizations to block similar attacks before they happen. | "This is the 'herd immunity' component. When Node A detects a new DDoS variant, the Immunity Engine automatically generates firewall rules that are pushed to Nodes B, C, and D — protecting them from the same attack before they even encounter it." |
| **Firewall Rules to All Nodes** | The generated rules flow back up to all local organizations, completing the intelligence sharing loop. | "This closes the loop. Intelligence flows from detection → analysis → sanitization → aggregation → retrieval → rule generation → distribution. The entire cycle happens automatically without human intervention." |

---

## Part 2: Design Structure Chart — Module Descriptions

The structure chart shows the software architecture — how the codebase is organized into packages and files.

---

### Package: `src/data` (Data Pipeline)

| File | Class | Purpose |
|---|---|---|
| `loader.py` | `FedIntelDataLoader` | Loads Kaggle parquet datasets, partitions into 4 nodes, handles train/test splits |
| `feature_engineer.py` | `FeatureEngineer` | Derives 25 statistical features from 41 raw features (ratios, rates, flags) |
| `preprocessor.py` | `Preprocessor` | Label encoding, unified class mapping, stratified sampling, normalization |
| `mitre_mapper.py` | `MITREMapper` | Maps attack names to MITRE ATT&CK Technique IDs, Tactics, and Severity levels |
| `summarizer.py` | `ThreatSummarizer` | Generates human-readable threat summaries using Gemini LLM with deterministic fallback |

### Package: `src/local_node` (Local Detection)

| File | Class | Purpose |
|---|---|---|
| `ids_model.py` | `IDSModel` | PyTorch neural network (66→256→128→64→N classes) with BatchNorm, Dropout, and DP noise layer |
| `ids_trainer.py` | `IDSTrainer` | Training loop with FedProx support, evaluation metrics, model save/load |
| `zeroday_detector.py` | `ZeroDayDetector` | Per-protocol autoencoders (TCP/UDP/ICMP) + entropy-based confidence scoring for zero-day detection |
| `agent_analyzer.py` | `AgentAnalyzer` | Gemini LLM-powered threat analysis with MITRE mapping and deterministic fallback |
| `agent_sanitizer.py` | `AgentSanitizer` | 12-pattern regex PII scrubbing engine |
| `pii_validator.py` | `PIIValidator` | Recursive dictionary scanner to enforce 0% PII leakage |
| `dp_layer.py` | `DPLayer` | Differential Privacy noise injection layer (active during training, deterministic during eval) |
| `node.py` | `PrivacyNode` | Orchestrator that chains: Analyze → Sanitize → Validate → Share |

### Package: `src/server` (Global Server)

| File | Class | Purpose |
|---|---|---|
| `fl_simulation.py` | `FLSimulation` | End-to-end FL orchestrator: creates 4 clients, runs N rounds of train-aggregate-distribute |
| `fl_server.py` | `FLServer` | FedAvg/FedProx weight aggregation with trust scoring |
| `fl_client.py` | `FLClient` | Wraps IDSTrainer for FL: local training, weight extraction, weight loading |
| `global_kb.py` | `GlobalKnowledgeBase` | ChromaDB wrapper: stores/retrieves threat embeddings using sentence-transformers |
| `rag_engine.py` | `RAGEngine` | Cross-encoder reranking, abstention threshold (0.30), semantic search |
| `aggregator.py` | `Aggregator` | Ingests sanitized reports into the Global Knowledge Base |
| `immunity.py` | `ImmunityEngine` | Generates Snort/iptables firewall rules from threat intelligence |

### Package: `src/evaluation` (Evaluation)

| File | Class | Purpose |
|---|---|---|
| `evaluate.py` | 4 evaluation suites | (1) Local vs FL accuracy, (2) PII leakage + DP tradeoff, (3) Zero-day + BERTScore, (4) Ablation study |

### Standalone Modules

| File | Class | Purpose |
|---|---|---|
| `live_monitor.py` | `LiveMonitor` | Real-time deployment pipeline: Feature Eng → IDS → Zero-Day → PII → Alert |
| `cli.py` | CLI | Command-line interface for training, evaluation, querying, and simulation |

---

## Part 3: Glossary of Key Terms

| Term | Full Form | Meaning |
|---|---|---|
| **ACTIS** | Adaptive Cyber Threat Intelligence System | The name of this project — an intelligent, privacy-preserving, federated cybersecurity system |
| **IDS** | Intrusion Detection System | Software that monitors network traffic and identifies malicious activity |
| **NetFlow** | Network Flow | A protocol developed by Cisco that collects and records all IP network traffic metadata |
| **PCAP** | Packet Capture | A format for capturing raw network packets including full payload data |
| **FL** | Federated Learning | A machine learning approach where the model is trained across multiple devices without sharing raw data |
| **FedAvg** | Federated Averaging | The most common FL algorithm — averages model weights from all clients proportional to their dataset sizes |
| **FedProx** | Federated Proximal | An FL algorithm that adds a regularization term to handle non-IID data across clients |
| **Non-IID** | Non-Independent and Identically Distributed | When different clients have different data distributions (e.g., Node A sees DDoS, Node C sees botnets) |
| **DP** | Differential Privacy | A mathematical framework that adds noise to data or models to prevent re-identification of individuals |
| **Zero-Day** | — | An attack that exploits a previously unknown vulnerability — no prior signatures or patterns exist for it |
| **Autoencoder** | — | A neural network trained to compress and reconstruct input data. If it cannot reconstruct well, the input is likely anomalous |
| **Entropy** | — | A measure of uncertainty in a probability distribution. High entropy = the model is unsure about its prediction |
| **Softmax** | — | A function that converts raw neural network outputs into probabilities that sum to 1.0 |
| **MITRE ATT&CK** | Adversarial Tactics, Techniques, and Common Knowledge | A globally recognized knowledge base of adversary behaviors, organized by tactics and techniques |
| **PII** | Personally Identifiable Information | Any data that can identify an individual — IP addresses, emails, names, phone numbers |
| **RAG** | Retrieval-Augmented Generation | An AI pattern that retrieves relevant documents from a knowledge base before generating a response |
| **Cross-Encoder** | — | A reranking model that jointly encodes the query and each candidate document for higher-precision relevance scoring |
| **Abstention** | — | The system's ability to say "I don't know" instead of returning irrelevant results — prevents hallucinations |
| **ChromaDB** | — | An open-source vector database optimized for storing and querying embeddings (high-dimensional numerical representations of text) |
| **Embedding** | — | A fixed-length numerical vector (e.g., 384 dimensions) that captures the semantic meaning of text |
| **Sentence-Transformer** | — | A model that converts text sentences into embeddings, enabling semantic similarity search |
| **BERTScore** | — | An evaluation metric that measures the semantic similarity between generated and reference text using BERT embeddings |
| **DDoS** | Distributed Denial of Service | An attack where multiple systems flood a target with traffic to make it unavailable |
| **DoS** | Denial of Service | Similar to DDoS but from a single source |
| **Snort** | — | An open-source network intrusion prevention system that can generate real-time traffic alerts |
| **iptables** | — | A Linux command-line firewall utility for configuring packet filter rules |
| **Herd Immunity** | — | In ACTIS context: when one organization detects a zero-day attack, all other organizations are automatically protected through shared firewall rules |
| **PyTorch** | — | An open-source deep learning framework developed by Meta, used for building and training neural networks |
| **BatchNorm** | Batch Normalization | A technique that normalizes layer inputs to speed up training and improve stability |
| **Dropout** | — | A regularization technique that randomly deactivates neurons during training to prevent overfitting |
| **ReLU** | Rectified Linear Unit | An activation function: f(x) = max(0, x). The most common activation in deep learning |
| **Parquet** | — | A columnar file format optimized for large-scale data storage and fast analytical queries |
| **Ablation Study** | — | An experiment where you systematically remove components to measure each one's individual contribution |
| **F1 Score** | — | The harmonic mean of precision and recall. A single metric that balances both false positives and false negatives |
| **Precision** | — | Of all predictions labeled as "attack", what percentage were actually attacks? (avoids false alarms) |
| **Recall** | — | Of all actual attacks, what percentage did the system correctly detect? (avoids missed attacks) |
| **Accuracy** | — | The percentage of total predictions that were correct |
| **Confusion Matrix** | — | A table showing True Positives, True Negatives, False Positives, and False Negatives for each class |
