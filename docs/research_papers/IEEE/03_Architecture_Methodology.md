# III. ACTIS Architecture and Methodology

To transition from disjointed anomaly classifiers to an integrated, intelligence-generating defense mesh, the ACTIS framework is fundamentally architected across three interactive layers: Multi-org Federated Telemetry, Privacy & Dissemination control, and the Edge RAG intelligence layer. ACTIS synthesizes discrete network packets into holistic vector intelligence, subsequently disseminating anonymized behavioral knowledge to global security graphs.

## A. Feature Space Engineering
Raw packet flows are captured and mapped into a comprehensive 66-dimensional numerical vector. This robust abstraction expands significantly upon standard 41 NetFlow fields natively extracted by tools like NFStream.
To accommodate modern volumetric threats, specifically varied Distributed Denial of Service (DDoS) forms, ACTIS introduces 25 dynamically derived engineered attributes. Driven strictly by quantitative analysis, specific feature additions include:
- **Flag Decompilation (`tcp_flag_syn`, `tcp_flag_rst`, etc.):** Isolating specific bit-level activations heavily correlative to DoS floods.
- **Asymmetry Factors (`byte_asymmetry`, `pkt_asymmetry`):** Calculating inbound/outbound disparities (e.g., massive incoming SYN packets juxtaposed with negligible outgoing payload size).
- **Throughput Metrics (`tp_dst_src_norm`):** Applying a $\log(\text{DST\_TO\_SRC\_TP})$ function to smooth out sudden traffic intensity peaks while amplifying persistent high-throughput variance.

## B. Trust-Aware Proximal Federated Learning (FedProx)
In federated edge topologies, devices range wildly from constrained factory IoT gateways to high-compute core infrastructure. Thus, the feature distributions natively inherit harsh non-Independent and Identically Distributed (non-IID) qualities [26]. Attempting to use a standard `FedAvg` across heavily imbalanced clients leads to catastrophic parameter drift. 
To counteract localized drift, ACTIS utilizes `FedProx` [36], inherently deploying an additional proximal regularization term $\mu = 0.01$ to strictly penalize nodes updating their parameters too distantly from the global anchor $W_g$.

Building directly on Tri-LLM's loss-driven monitoring scheme, ACTIS introduces dynamic Trust-Aware Client Selection. Assuming client $i$'s empirical alignment loss is $L_i$, the client trust factor $\tau_i$ is evaluated as:
$\tau_i = \frac{1}{L_i + \epsilon}$
A normalization filter determines an active contribution mask. Consequently, during global synchronization (orchestrated over 5 primary convergence rounds with 4 simulated clients: A, B, C, D), ACTIS automatically scales client gradients to mathematically stabilize malicious or wildly variant edge nodes.

## C. The Agentic Privacy Stack: PII Sanitization + Differential Privacy
Central to ACTIS's core objective is robust multi-org intelligence collaboration completely devoid of legal or identity data leakage. We instantiate a strict, two-stage privacy defense.

**1. Deterministic PII Stripping:** The vast majority of IDSs fail to operate securely because reporting relies heavily on localized log inspection. In ACTIS, whenever anomalous traffic flags the local anomaly detector, the telemetry triggers an Agentic LLM process optimized explicitly for PII extraction. The LLM translates raw rows into sanitized Natural Language (NL). Using targeted prompts, the language agent scrubs IP addresses, sensitive hostnames, and domain data—transmuting "192.168.1.10 contacted anomalous server 203.0.113.1" into "Internal host communicated with an external IP flagged for high outbound throughput".
**2. Parameter-Level Differential Privacy:** Once localized Deep Learning Multilayer Perceptrons (MLPs) optimize their weights on local data, the raw gradient differentials are not immediately exported to the central aggregator. ACTIS implements an explicit Local Differential Privacy (LDP) [19], mathematically bounds sensitivity $S$, and introduces a Gaussian noise mechanism to obfuscate membership inference attacks before transmission:
$W_{i}^{\text{noisy}} = W_i + \mathcal{N}(0, \sigma^2 S^2)$

## D. Agentic Retrieval-Augmented Generation (RAG) System
ACTIS’s centralized query system natively incorporates an intelligent RAG orchestrator interacting with ChromaDB. Drawing inspiration directly from both ReGAIN and CyberRAG, the knowledge architecture divides threat information into distinct vector namespaces (telemetry, heuristics, attack signatures). 
When assessing threat semantics, the pipeline utilizes a **bi-encoder + cross-encoder reranking** algorithm [2]. Following an initial approximate nearest neighbor search via the bi-encoder, the cross-encoder (powered by `ms-marco-MiniLM`) scores contextual (Query, Candidate) pairs directly, discarding loosely related noise. Notably, ACTIS institutes an explicit threshold abstention logic (score < 0.30), coercing the agent to halt generation rather than hallucinate threat causality. Combining the refined retrieval output with deterministic LLM reasoning enables ACTIS to automatically correlate zero-day variants directly onto known historical CVE and MITRE frameworks.

## E. Protocol-Specific Autoencoders for Zero-Day Verification
ACTIS abandons naive deep-learning multi-class classifiers when searching for zero-days. Zero-day threats commonly masquerade as benign background signals but deviate fundamentally in structural protocol execution. To identify variants, ACTIS independently channels network records into distinct models strictly defined by transport protocol (TCP, UDP, ICMP).
Rather than labeling threats, each protocol subset trains a decoupled Deep Autoencoder to continually reconstruct expected benign sequences. By calculating the protocol-calibrated Mean Squared Error (MSE), ACTIS defines localized anomaly borders (typically at the 99th percentile of benign reconstruction loss). Spikes bypassing the MSE trigger the RAG orchestration sequence, classifying an event as an unprecedented zero-day instance without requiring static signature evaluation.
