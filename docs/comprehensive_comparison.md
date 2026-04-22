# Comprehensive System Comparison

| Feature / Metric | ReGAIN | Tri-LLM | CyberRAG | **ACTIS** |
| --- | --- | --- | --- | --- |
| **Core Concept** | RAG + GPT-4 for explainable IDS | 3-LLM semantic prototypes + federated zero-shot | Agentic RAG with modular BERT classifiers | **Federated IDS + RAG + Privacy + Zero-Day + Live Deployment** |
| **Datasets Evaluated** | MAWILab (PCAP) | NF-BoT-IoT-v2 (IoT) | NF-CSE-CIC-IDS2018-v2 | **NF-CIC-IDS2018-v2 & NF-BoT-IoT-v2 (Cross-dataset)** |
| **Input Feature Space** | 41 NetFlow | Semantic embeddings | BERT text features | **66 (41 NetFlow + 25 engineered DDoS-specific features)** |
| **Federated Learning** | ❌ Single Node | ✅ 3 Clients | ❌ Single Node | ✅ **4 Clients (with FedProx Non-IID handling)** |
| **Zero-Day Detection Strategy** | Heuristic rules | ZDS cosine + disagreement | ❌ None | ✅ **Per-protocol autoencoders (MSE reconstruction)** |
| **Zero-Day Detection Rate** | ~60% (est.) | 68% (avg) | — | **99.5% (DDoS on BoT-IoT) / 70.4% (avg)** |
| **Known Attack Detection Accuracy** | 98.82% (TCP SYN Flood) | 85.6% (Non-IID FL worst) | — | **100.0% (TCP SYN Flood) / 98.4% (Non-IID FL worst)** |
| **False Positive Rate (Precision)** | 74.5% precision (ICMP Flood) | — | — | **99.6% precision (ICMP Flood)** |
| **Privacy & Security Framework** | ❌ None | ✅ Federated isolation | ❌ None | ✅ **Federated isolation + PII Sanitization + (ε, δ)-DP Noise** |
| **RAG Retrieval Engine** | Bi-encoder + cross-encoder | Semantic prototypes | BERT + MMR | ✅ **Bi-encoder + cross-encoder + Abstention threshold (0.30)** |
| **LLM Orchestration** | GPT-4 | GPT-4o | Fine-tuned BERT | **Gemini-1.5-Flash** |
| **Operational Costs** | High (Paid API) | High (Paid API) | None (Local) | **Free (Gemini Free API tier)** |
| **Deployment Readiness** | ❌ Offline analysis only | ❌ Offline analysis only | ❌ Offline analysis only | ✅ **Real-time stream processing, CSV batch, live log tailing** |
| **Attack Class Coverage** | 2 (Binary) | 5 Classes | Multi-class | **9 Multi-class** |
