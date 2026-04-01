# III. ACTIS Architecture and Methodology

To transition from disjointed anomaly classifiers to an integrated, intelligence-generating defense mesh, the ACTIS framework is fundamentally architected across three interactive layers.

> [!TIP]
> **Architecture Diagram Placeholder**: Insert a visual diagram here demonstrating the data flow: Raw Telemetry $\rightarrow$ Local Autoencoder & PII LLM Scrubber $\rightarrow$ LDP Noise Injection $\rightarrow$ Trust-Aware FedProx Server.

## A. Feature Space Engineering
Raw packet flows are captured and mapped into a comprehensive 66-dimensional numerical vector. To accommodate modern volumetric threats, specifically varied Distributed Denial of Service (DDoS) forms, ACTIS introduces 25 dynamically derived engineered attributes:
- **Flag Decompilation (`tcp_flag_syn`):** Isolating specific bit-level activations heavily correlative to DoS floods.
- **Asymmetry Factors (`byte_asymmetry`):** Calculating inbound/outbound disparities.
- **Throughput Metrics:** Applying a $\log(\text{DST\_TO\_SRC\_TP})$ function to amplify persistent high-throughput variance.

## B. Algorithm and Federated Orchestration
To stabilize non-IID client drift across unconstrained IoT gateways, ACTIS relies on a highly regulated execution schema bridging Local Zero-Day Detection, Agentic RAG Generation, and Trust-Aware FedProx Parameter Synchronization.

**Algorithm 1: ACTIS Trust-Aware FedProx with PII-Sanitization**
```text
REQUIRE: Edge Nodes N, Protocol Autoencoders AE_p, Global MLP W_g
FOR each Node i in N in parallel:
    X_i = Extract_Features(Raw Traffic)
    IF MSE(AE_p(X_i)) > θ_anomaly:
        S_i = PII_Scrubber(X_i)
        Alert_Intel = Agentic_RAG(S_i)
    END IF
    
    W_i = Local_Train(W_g, X_i, μ)
    W_i_noisy = W_i + GaussianNoise(0, σ²S²)
    Send (W_i_noisy, L_i) to Aggregator
END FOR

Aggregator computes trust factor: τ_i = 1 / (L_i + ε)
W_g = Σ [ (τ_i / Σ τ_j) * W_i_noisy ]
```

## C. Mathematical Formalism for Zero-Day Discovery
ACTIS employs deep Autoencoders partitioned structurally by transport protocols (TCP, UDP, ICMP) to discover zero-day anomalies mathematically, circumventing static classification lists entirely.

The protocol-specific autoencoder minimizes the structural reconstruction error utilizing Mean Squared Error (MSE):
$$ \mathcal{L}_{AE}(X_i) = \frac{1}{d} \sum_{j=1}^{d} \left(x_{i,j} - \hat{x}_{i,j}\right)^2 $$

An active cyber vulnerability is intrinsically triggered if the input heavily distorts structural sequence parameters, triggering the constraint:
$$ \mathcal{L}_{AE}(X_i) > \theta_p $$

Where $\theta_p$ defines the mathematically derived 99th percentile of the purely localized benign structural formulation for a specific protocol $p \in \{\text{TCP, UDP, ICMP}\}$.

## D. The Agentic Privacy Stack
Central to ACTIS's core objective is robust multi-org intelligence collaboration completely devoid of legal or identity data leakage.
### 1. Deterministic PII Stripping
Whenever anomalous traffic mathematically trips the $\theta_p$ condition, the telemetry triggers an Agentic LLM process optimized explicitly for PII extraction, isolating target vectors away from human-identifiable origin nodes.
### 2. Parameter-Level Differential Privacy
Once localized Deep Learning Multilayer Perceptrons (MLPs) optimize their weights on local data, the raw gradient differentials are explicitly obscured by mathematical Local Differential Privacy (LDP) before execution via Algorithm 1.
