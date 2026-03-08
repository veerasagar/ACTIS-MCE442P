# Exhaustive Deep-Dive Analysis — Three Baseline Papers
>
> MCE442P / Fed-Intel Project | Generated: March 2026

---

## PAPER 1: ReGAIN

## Retrieval-Grounded AI Framework for Network Traffic Analysis

**arXiv:** 2512.22223 | **Published:** Dec 23, 2025 | **Pages:** 8 | **Venue:** arXiv cs.LG

---

### 1.1 Problem Statement

Modern network traffic analysis relies on ML/DL classifiers that achieve high detection accuracy but operate as **black boxes**. Analysts can't trust verdicts they can't explain. Existing LLMs help with reasoning but suffer from **hallucinations** when used in purely generative mode — they can make up threat explanations not grounded in actual evidence. RAG mitigates this by grounding LLM outputs to external knowledge, but prior RAG systems:

- Use a **single knowledge base** (no specialization by data type)
- **Lack retrieval quality controls** (no reranking, no abstention)
- Don't produce **cited evidence** — just generated text

### 1.2 Core Thesis

Transform raw heterogeneous network telemetry into natural-language summaries, semantically index them, retrieve the most relevant evidence using a multi-stage pipeline, and have an LLM produce a verdict with explicit citations to that evidence.

---

### 1.3 Data Ingestion and Summarization (Component 1)

**Input:** Heterogeneous traffic — PCAP files, CSV anomaly reports, flow records.

**Normalization Schema (5-tuple):**

```text
r_i = {ts_i, src_i, dst_i, p_i, proto_i, ℓ_i}
```

where:

- `ts_i` = timestamp
- `src_i` = source IP
- `dst_i` = destination IP
- `p_i` = port
- `proto_i` = protocol
- `ℓ_i` = anomaly label

**Deterministic Summarization Function `f_sum`:**

Raw record:

```text
2024-08-15 10:05:23, 192.0.2.7, 203.0.113.5, icmp, label=DoS
```

↓ becomes ↓

```text
"At 10:05:23 on August 15, 2024, host 192.0.2.7 sent an ICMP request
 to 203.0.113.5, flagged as a potential DoS anomaly."
```

**Why NL summaries and not raw rows?**

1. Reduces context window noise from raw log fields
2. Exposes network semantics (endpoints, protocol, timing) in natural language — improves embedding quality
3. Makes cited evidence transparent and human-readable

---

### 1.4 Semantic Vectorization and Knowledge Base (Component 2)

**Embedding:**

```text
v_i = f_embed(s_i) ∈ ℝ^d   (d = 384 dimensions)
```

Using a **transformer-based sentence embedding model** (lightweight, ~384d for throughput balance).

**Knowledge Base Entry:**

```text
e_i = (s_i, v_i, m_i)
```

where `m_i` = structured metadata (src IP, dst IP, src port, dst port, protocol, labels, timestamp).

**Multi-Collection Architecture (3 specialized vector databases):**

| Collection | What it stores |
| --- | --- |
| **Telemetry** | Enriched flow-level and packet-level summaries from PCAPs/log files |
| **Anomaly** | Labeled or auto-detected attack instances with metadata |
| **Heuristic** | Reference patterns, expert rules, known attack signatures |

**Why multi-collection?** Different query types benefit from different corpora. A "explain this SYN flood" query should retrieve from *anomaly* records, not heuristic rules, and vice-versa.

**Vector DB:** ChromaDB

---

### 1.5 Retrieval-Augmented Reasoning Engine (Component 3)

This is ReGAIN's most sophisticated component — a **multi-stage hierarchical pipeline**:

```text
Stage 1: Adaptive Metadata Filtering
   ↓  Filters by src/dst IP, protocol, port, label, time window
   ↓  Narrows search space before any vector comparison

Stage 2: Bi-Encoder (Dense) Retrieval
   ↓  Query is embedded; cosine similarity search across all 3 collections
   ↓  Fast but approximate — retrieves top-k candidates

Stage 3: Cross-Encoder Reranking
   ↓  Each (query, candidate) pair scored by a cross-encoder (computationally heavier)
   ↓  More accurate pairwise relevance than bi-encoder alone

Stage 4: MMR (Maximal Marginal Relevance) Diversity Sampling
   ↓  Removes redundant results — ensures retrieved context is diverse
   ↓  Prevents LLM context saturation with near-duplicate records

Stage 5: Abstention Mechanism
   ↓  If top-k retrieval quality score BELOW threshold → return diagnostic
   ↓  Prevents hallucination by refusing to generate when evidence is weak
```

**Why each stage matters:**

- Without metadata filtering → too many irrelevant hits, slow
- Without reranking → low-quality context fed to LLM
- Without MMR → LLM sees near-duplicate records, misses breadth
- Without abstention → LLM hallucinates when evidence is absent

---

### 1.6 LLM Analysis with Human-in-the-Loop (Component 4)

**LLM used:** GPT-4-class model (via remote API)

**Output includes:**

1. **Natural language verdict** — plain English explanation of what the traffic indicates
2. **Explicit citations** — references to specific record IDs from the knowledge base
3. **Conversational interface** — analyst can ask follow-up questions, refine criteria, query specific time ranges

**Dual-mode interaction:**

- *Automated mode*: system autonomously flags traffic and generates per-record explanations
- *Expert mode*: analyst provides additional context, asks targeted questions

---

### 1.7 Dataset and Evaluation Setup

**Dataset:** MAWILab — real-world internet traffic backbone dataset with diverse anomaly labels.

- 10,000 labeled instances total
- Attack scenarios: **TCP SYN Flood** and **ICMP Ping Flood**

**Baselines compared:**

| Type | Systems |
| --- | --- |
| Rule-based | Snort-style signature matching |
| Classical ML | Random Forest (RF), Support Vector Machine (SVM), k-Nearest Neighbors (KNN) |
| Deep Learning | Long Short-Term Memory (LSTM) |

**Metrics used:** Accuracy, Precision, Recall (F1 implied)

**Results:**

| Attack | System | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- |
| SYN Flood | **ReGAIN** | **98.82%** | **~91%** | **98.6%** |
| SYN Flood | LSTM (best baseline) | 95.1% | ~76.5% | 94.8% |
| Ping Flood | **ReGAIN** | **95.95%** | **74.5%** | **100%** |
| Ping Flood | LSTM (best baseline) | ~94% | ~80% | 95.6% |

**Key findings:**

- ReGAIN improves over best baseline (LSTM) on SYN Flood: +3.7% accuracy, +14.5% precision, +3.8% recall
- ReGAIN achieves **perfect recall on Ping Flood** (100%) at the cost of precision (74.5%) — a deliberate design choice (miss nothing, investigate false alarms)

---

### 1.8 Qualitative Advantages Over Baselines

- RTF classifiers output a label (0/1). ReGAIN outputs a **paragraph explaining why**
- Conversational interface enables **human-AI collaborative threat hunting**
- Abstention prevents **false confidence** — a property no traditional classifier has

---

### 1.9 Limitations

- GPT-4 via remote API → **latency unsuitable for real-time** (batch/forensic analysis only)
- 384-d embeddings sacrifice throughput for quality
- Ping Flood precision gap: benign ICMP traffic resembles flood patterns
- **Single-node only** — no federation, no multi-org sharing
- **No PII protection** — doesn't sanitize sensitive fields

### 1.10 Future Work (stated by authors)

- On-premise models (LLaMA-3-8B, Mistral-7B, Phi-3) for air-gapped environments
- Dynamic similarity thresholds by protocol type
- Rate/temporal filters to separate short benign ICMP bursts from sustained attacks
- Formal evaluation of NL output quality

---
---

## PAPER 2: Tri-LLM

## Cooperative Federated Zero-Shot Intrusion Detection with Semantic Disagreement and Trust-Aware Aggregation

**arXiv:** 2602.00219 | **Published:** Jan 30, 2026 | **Pages:** 17 | **Venue:** arXiv cs.CR

---

### 2.1 Problem Statement

Existing Federated IDS frameworks:

1. Are **classification-driven** — assume fixed attack label spaces (can't detect new attacks)
2. Don't handle **zero-day / open-world** scenarios (distributional shift)
3. **Don't integrate semantic knowledge** from LLMs into the detection pipeline
4. Standard aggregation (FedAvg) is fragile under **non-IID data** and **compromised clients**

**The core insight:** LLMs encode rich knowledge about attacker tactics and behaviors that is hard to capture with statistical feature learning alone. If you can express attacks as *language-derived semantic prototypes*, you can detect attack behaviors you've never seen before via semantic similarity — **without needing a single labeled sample**.

---

### 2.2 System and Adversary Model

**Setup:** N edge clients (IoT gateways, controllers, monitoring nodes) coordinated by a central FL server.

**Privacy constraint:** Raw traffic data and extracted features are **never shared** — only lightweight model parameters (weight matrices) are transmitted.

**Adversary capabilities:**

1. **Data-level evasion** — craft inputs whose embeddings resemble benign traffic:

   ```text
   min_{x'} || f(x'; W_g) - z_benign ||_2
   ```

   The framework is robust to this because superficial perturbations don't change semantic meaning

2. **Model poisoning** — compromised client i submits manipulated update W_adv_i. Impact:

   ```text
   ΔW_g = Σ_{i=1}^{N} α_i (W_i - W_g)
   ```

   Mitigated by trust-weighting — poisoned clients get lower α_i

3. **Zero-day attacks** — no labels or signatures exist. Mitigated by semantic similarity + disagreement-based risk scoring

---

### 2.3 Tri-LLM Semantic Knowledge Construction (the "Secret Sauce")

**The three LLMs and their roles:**

| LLM | Role | Why chosen |
| --- | --- | --- |
| **GPT-4o** | Contextual reasoning, capturing attacker intent | Balanced semantic abstraction |
| **DeepSeek-V3** | Stable, low-variance representations | Stabilizes the semantic space |
| **LLaMA-3-8B** | Expressive, fine-grained context | Richer modeling of evasion strategies |

**For each attack category a ∈ A, three textual descriptions are generated — one per LLM — each emphasizing:**

- *Offensive perspective*: attacker goals and tactics
- *Defensive perspective*: observable network/system symptoms
- *Adversarial strategy perspective*: execution patterns and evasion behaviors

**Per-LLM embedding:**

```text
z_a^(m) = LLM_m(description_a^(m))
```

**Fusion — weighted combination:**

```text
z_a = Σ_m w_m · z_a^(m)   (w_m derived from LLM reliability scores)
```

**Result:** A shared semantic prototype space A = {z_a} for all attack categories, including future ones not yet observed.

**Critical note:** The LLMs are NOT federated — they are external semantic encoders. Only the **lightweight linear projection model W** is trained via FL across clients.

---

### 2.4 Local Feature Representation and Semantic Projection

**Per client i, at time t:**

```text
x_{i,t} ∈ ℝ^d   (raw feature vector)
```

Features include: protocol-level statistics, flow-level summaries, temporal dynamics, host-level indicators.

**Semantic projection:**

```text
ẑ_{i,t} = f(x_{i,t}; W_i) = W_i · x_{i,t}
```

- W_i ∈ ℝ^{k×d} is the local projection matrix
- Chosen as **linear projection** for efficiency on resource-constrained devices
- Maps traffic features into the shared semantic space aligned with LLM-derived prototypes

---

### 2.5 Federated Training Loop

1. Server shares current global model W_g with all clients
2. Each client computes its **semantic alignment loss** (how well its local projections match the prototypes):

   ```text
   L_i = (1/T) Σ_t || ẑ_{i,t} - z_{y_t} ||_2^2
   ```

3. Each client sends updated W_i back to server
4. Server computes **trust score** per client:

   ```text
   τ_i = 1 / (L_i + ε)
   ```

   High loss → low trust → low weight
5. **Trust-aware aggregation:**

   ```text
   α_i = τ_i / Σ_j τ_j
   W_g ← Σ_i α_i · W_i
   ```

6. Server monitors **trust entropy** H = -Σ_i α_i log(α_i) as a stability diagnostic. Sudden entropy collapse = potential client failure or poisoning attack.

---

### 2.6 Zero-Shot Inference and Zero-Day Risk

**Zero-shot attribution** (for known attack categories):

```text
â = argmax_{a ∈ A} cos(ẑ, z_a)
```

**Inter-LLM Semantic Disagreement:**

```text
D_a = (1/M(M-1)) Σ_{m≠n} || z_a^(m) - z_a^(n) ||_2^2
```

Where M=3 (number of LLMs). High disagreement = models have different "opinions" about what this attack looks like → **epistemic uncertainty**.

**Zero-Day Risk Score:**

```text
ZDS(x) = λ · D_â + (1-λ) · (1 - cos(ẑ, z_â))
```

- First term: inter-LLM disagreement (uncertainty about the known class prototype)
- Second term: low cosine similarity (the traffic doesn't match any known prototype well)
- Joint score: high ZDS means "this looks like something we haven't seen before"

**Monotonic relationship confirmed empirically:**

```text
C = f(D_a),   ∂C/∂D_a < 0
```

As disagreement increases, confidence decreases — a well-calibrated uncertainty response.

---

### 2.7 Detailed Experimental Results

**Semantic Diversity vs. Zero-Shot Accuracy (Table IX):**

| Centered Trust Entropy | Accuracy (%) | Interpretation |
| --- | --- | --- |
| -12 | 88.5 | Low diversity, early convergence |
| -8 | 89.2 | Improved stability |
| -4 | 89.9 | Balanced representation learning |
| 0 | 90.6 | Stable convergence region |
| +4 | 91.3 | Enhanced generalization |
| +8 | 92.0 | Increased semantic diversity |
| +12 | 92.7 | Robust zero-day behavior |

**The higher the semantic diversity across clients, the better the zero-shot accuracy.** The system *benefits* from heterogeneous client knowledge.

**Zero-Day Risk vs. Confidence (Table VIII):**

| Disagreement D_a range | Mean Confidence | Interpretation |
| --- | --- | --- |
| < 1.10×10⁻³ | 1.56 | High confidence — established behavior |
| 1.10–1.13×10⁻³ | 1.52 | Moderate uncertainty |
| ≥ 1.13×10⁻³ | 1.46 | Elevated risk — potential zero-day |

**Overall results:**

- **80%+ zero-shot accuracy** on attack categories the system was never trained on
- **10%+ improvement** over similarity-based baselines for zero-day discrimination
- Stable aggregation even with unreliable/poisoned clients

---

### 2.8 Limitations

- LLMs are **static, external encoders** — not agentic (no log reading, no actions taken)
- **No RAG** — semantic embeddings used, not retrieval-augmented generation
- **No PII sanitization** — data stays local via FL, but no active privacy scrubbing
- **No human-readable explanations** — outputs are verdicts, not cited reports
- Requires GPT-4o API for initial prototype generation phase

### 2.9 Future Work (stated by authors)

- Extending to heterogeneous feature spaces across clients
- Incorporating time-aware semantic drift detection
- Applying to real-time streaming traffic
- Reducing dependency on proprietary LLMs for prototype generation

---
---

## PAPER 3: CyberRAG

## An Agentic RAG Cyber Attack Classification and Reporting Tool

**arXiv:** 2507.02498 (v2: Sep 10, 2025) | **Pages:** 18 | **Venue:** arXiv cs.CR

---

### 3.1 Problem Statement

Conventional RAG systems have a critical weakness: **they retrieve context only once** before generating output. They cannot:

- Refine queries based on retrieved results
- Reason iteratively over evidence
- Dynamically adapt when the first retrieval is insufficient

And LLM-based agents, while autonomous, operate as **black boxes** — analysts can't trace why a decision was made.

**The gap:** Need a system that is BOTH agentic (autonomous, iterative, tool-using) AND explainable (transparent reasoning, cited sources), AND specialized (precise for cybersecurity).

---

### 3.2 Attack Scope

CyberRAG targets **web-based attack payloads** — the entry point for most enterprise breaches:

**SQL Injection (SQLi):**

- Classic (direct payload in query string)
- Blind (boolean-based or time-based, no direct feedback)
- Union-based (UNION operator to extract data from other tables)
- Out-of-band (DNS/HTTP exfiltration when direct feedback not available)

**Cross-Site Scripting (XSS):**

- Reflected, stored, DOM-based
- Goal: steal cookies, credentials, execute JavaScript in victim's browser

**Server-Side Template Injection (SSTI):**

- Attacker injects template directives into server-rendered templates (Jinja2, Twig, etc.)
- Can escalate to Remote Code Execution (RCE)

---

### 3.3 System Architecture — Detailed

The **Core LLM Engine** is the central autonomous agent. It:

- Receives a flagged payload from the upstream IDS/IPS
- Decides control flow — which tools to call, in what order
- Iterates: can re-query the RAG tool if initial retrieval is insufficient
- Synthesizes all outputs into the final report

**Component 1 — Classification Tool:**

A modular ensemble of **fine-tuned BERT-family classifiers**, one per attack type:

- Fine-tuned on attack-specific datasets (not raw GPT-4 prompting — domain-adapted weights)
- Each classifier outputs:
  - **Label**: predicted attack type (SQLi / XSS / SSTI)
  - **Confidence score**: 0.0–1.0
  - **Explanatory component**: key features that triggered the classification

Classifiers are evaluated in **parallel** on the same payload. Output is a structured comparison table:

| Payload ID | SQLi score | SSTI score | XSS score |
| --- | --- | --- | --- |
| PD001 | **0.9999** | 0.3956 | 0.0673 |
| PD002 | 0.3999 | **0.9997** | 0.3830 |
| PD003 | 0.3998 | 0.3929 | **0.9999** |

Core LLM selects the **highest confidence** class. No voting — direct confidence-based selection.

**Why modular? Ablation result:**

- Single monolithic model (all attack types combined): **73.4% accuracy**
- Ensemble of specialized classifiers (no RAG): **84.75% accuracy**
- Ensemble + RAG tool: **94.92% accuracy**

Each specialist can be added **without retraining the core agent**. Separation of failures — one bad classifier doesn't break others.

**Component 2 — RAG Tool:**

**3 separate vector stores** (using FAISS), each optimized for a different source domain:

1. **CVE/NVD database** — known vulnerability records
2. **OWASP/MITRE ATT&CK** — standard attack taxonomy, defensive frameworks
3. **Scientific literature + incident reports** — academic context

#### Retrieval strategy: MMR (Maximal Marginal Relevance)

- Balances **relevance** (topically close to query) and **diversity** (non-redundant)
- Prevents the LLM from seeing 5 nearly identical CVE descriptions

#### Process

1. Core LLM generates a natural language query from the classifier output
2. Query is embedded and searched across all 3 stores
3. MMR selects top-k diverse, relevant chunks
4. Chunks are summarized and returned to core LLM

**Component 3 — Report Generation:**

Core LLM synthesizes Classification output + RAG chunks into a structured attack report:

- Inferred attack type + reasoning justification
- Salient payload features that triggered the verdict
- Contextual indicators (script patterns, DOM elements, field characteristics)
- Associated CVEs, threat severity rating
- Recommended mitigation steps

**Component 4 — Interactive Chat:**

Post-report, analyst can ask natural language questions:

- *"Why was this classified as SSTI?"*
- *"How do I patch this vulnerability?"*
- *"What are the other known exploits for this CVE?"*

System adapts responses to analyst expertise level (novice vs. expert).

---

### 3.4 Datasets Used

| Attack Type | Dataset | Source |
| --- | --- | --- |
| XSS | Kaggle XSS dataset | Kaggle (syedsaqlainhussain) |
| SSTI | SSTI dataset | GitHub (francescopirox/ssti_dataset) |
| SQLi | Kaggle SQL Injection dataset | Kaggle (syedsaqlainhussain) |

---

### 3.5 Detailed Evaluation Results

**Classification accuracy by component:**

| Configuration | Accuracy |
| --- | --- |
| Single monolithic model | 73.4% |
| Ensemble of specialists (no RAG) | 84.75% |
| Ensemble + RAG Tool (full CyberRAG) | **94.92%** |

**Per-class accuracy:**

- SQLi: >94%
- XSS: >94%
- SSTI: >94%

**Report quality (NLG metrics):**

| Metric | Score | What it measures |
| --- | --- | --- |
| BLEU | Reported | N-gram overlap with reference |
| ROUGE | Reported | Recall-oriented overlap |
| METEOR | Reported | Synonym-aware overlap |
| **BERTScore** | **0.94** | Semantic similarity to ground truth |
| **GPT-4 Expert Eval** | **4.9/5** | Human-judge quality (automated) |

**Robustness tests:**

- Tested against **adversarial payloads** (deliberately obfuscated inputs)
- Tested against **out-of-distribution** (unseen payload variations)
- Strong performance maintained in both conditions

---

### 3.6 Key Design Insights

1. **Modularity beats monolithics**: 73.4% → 94.92% purely from specialization + RAG integration
2. **RAG is the semantic orchestrator**: Not just context retrieval — it bridges low-level payload analysis to high-level threat knowledge
3. **Knowledge base extensibility without retraining**: Organizations can add internal docs, architecture diagrams, policies as PDFs — zero retraining of the agent or classifiers
4. **Iterative retrieval**: Agentic design allows re-querying when initial evidence is weak — unlike standard one-pass RAG

---

### 3.7 Limitations

- **Upstream IDS dependency**: If the IDS doesn't flag something, CyberRAG never sees it — the pipeline starts too late
- **Known-threat constraint**: Can only classify attacks it has classifiers for; unknown attack families produce no valid output
- **Web attacks only**: Not designed for network-level attacks (DDoS, SYN flood, Ping flood)
- **Single-node**: No federation, no cross-org sharing
- **Knowledge base quality**: Output quality is bounded by what's in the vector stores

### 3.8 Future Work (stated by authors)

- Expand taxonomy to include more attack families
- Add **knowledge graph** integration for deeper contextual reasoning
- Enable **automated response mechanisms** (not just analysis — also action)
- SIEM integration (Splunk, Microsoft Sentinel, etc.)
- Toward **autonomous proactive defense**

---
---

## Synthesis: How All Three Work Together for Fed-Intel

| Fed-Intel Layer | Technique to Borrow | From Which Paper | Exact Component |
| --- | --- | --- | --- |
| Traffic ingestion + NL summary | Deterministic summarization | **ReGAIN** | `f_sum(r_i)` — 5-tuple → NL |
| Local attack classifier | Modular BERT classifiers | **CyberRAG** | Classification Tool |
| Agentic log reader / PII stripper | Core LLM as autonomous agent | **CyberRAG** | Core LLM Engine + tool orchestration |
| Local semantic projection | Linear projection W | **Tri-LLM** | `ẑ = W·x` — maps traffic to semantic space |
| Local vector knowledge base | Multi-collection ChromaDB | **ReGAIN** | Telemetry / Anomaly / Heuristic collections |
| Local retrieval pipeline | MMR + reranking | **ReGAIN + CyberRAG** | Bi-encoder → cross-encoder → MMR |
| Federated weight sharing | Only W shared, not raw data | **Tri-LLM** | FL projection model |
| Trust-aware aggregation | Loss-driven client weighting | **Tri-LLM** | `τ_i = 1/(L_i + ε)` |
| Zero-day detection | Semantic disagreement as risk | **Tri-LLM** | ZDS(x) = λD_â + (1-λ)(1-cos) |
| Global RAG knowledge base | Federated summary store | **ReGAIN** | Multi-collection, now global |
| Report generation | Structured NL report | **CyberRAG** | Report Generation module |
| Analyst interaction | Interactive chat | **CyberRAG** | User Chat component |
| **PII sanitization** | **Nothing — Fed-Intel novelty** | **None (gap)** | **Must be designed from scratch** |

---

## Key Metrics to Reproduce and Surpass

| What you test | Metric | Target | Paper baseline |
| --- | --- | --- | --- |
| Attack detection | Accuracy | >95% | ReGAIN: 95.95–98.82% |
| Attack detection | Recall | >98% | ReGAIN: 98.64–100% |
| Zero-day detection | Zero-shot accuracy | >80% | Tri-LLM: 80–92.7% |
| Report quality | BERTScore | >0.90 | CyberRAG: 0.94 |
| Report quality | Expert evaluation | >4.5/5 | CyberRAG: 4.9/5 |
| **Privacy (your novelty)** | **PII Leakage Rate** | **0%** | **None — no baseline exists** |
