# ACTIS — Complete Commands Reference

> All commands are run from the project root: `cd /path/to/MCE442P`

---

## Phase 0 — Setup & Dependencies

```bash
# Install all dependencies
python3 -m pip install -r requirements.txt

# Install enhancement packages (BERTScore + cross-encoder reranking)
python3 -m pip install bert-score sentence-transformers

# (Optional) Set Gemini API key for LLM-enhanced summaries
echo 'GEMINI_API_KEY=your_key_here' > .env

# Download both datasets (~3 GB)
bash scripts/download_dataset.sh
```

---

## Phase 1 — Data Pipeline Validation

```bash
# Verify 4-node data loading (A, B = IDS2018; C, D = BoT-IoT)
python3 scripts/validate/data_pipeline.py

# Verify feature engineering (41 → 66 features)
python3 scripts/validate/feature_engineering.py
```

---

## Phase 2 — Local IDS (No Federation)

```bash
# Train & evaluate standalone IDS per node
python3 scripts/validate/local_ids.py
```

---

## Phase 3 — Federated Learning (4 Nodes)

```bash
# 5-round FL training
python3 scripts/validate/federated_learning.py --rounds 5

# 10-round FL training (full)
python3 scripts/validate/federated_learning.py --rounds 10 --samples 20000
```

---

## Phase 4 — Privacy Validation

```bash
# PII sanitization + Differential Privacy noise verification
python3 scripts/validate/privacy.py
```

---

## Phase 5 — Zero-Day Detection

```bash
# DDoS zero-day detection benchmark on BoT-IoT
python3 scripts/validate/zeroday_ddos.py
```

---

## Phase 6 — RAG & Threat Intelligence

```bash
# RAG with cross-encoder reranking + abstention + cross-org immunity
python3 scripts/validate/rag_engine.py

# MITRE ATT&CK mapping table
python3 -m src.data.mitre_mapper
```

---

## Phase 7 — Report Quality

```bash
# BERTScore evaluation + threat summarizer test
python3 scripts/validate/report_quality.py
```

---

## Phase 8 — ReGAIN Benchmark Replication

```bash
# TCP SYN Flood + ICMP/UDP Flood (vs ReGAIN paper)
python3 scripts/validate/regain_benchmark.py
```

---

## Phase 9 — Full Evaluation Suite

```bash
# Run all 4 evaluations (~13 min)
python3 -m src.evaluation.evaluate
```

Runs: (1) Local vs federated accuracy, (2) PII leakage + DP tradeoff, (3) Zero-day + BERTScore, (4) Ablation study. Results saved to `results/evaluation_results.json`.

---

## Phase 10 — End-to-End Simulation

```bash
# Full pipeline (5 rounds, 10K samples, 20 threat flows)
python3 -m src.cli simulate --rounds 5 --samples 10000 --threats 20

# Quick demo (3 minutes)
python3 -m src.cli simulate --rounds 3 --samples 5000 --threats 10
```

---

## Phase 11 — Live Deployment

```bash
# Process a CSV of network flows
python3 -m src.live_monitor --input flows.csv --company A

# Save alerts to JSONL
python3 -m src.live_monitor --input flows.csv --output alerts.jsonl --company A

# Tail a growing log file (continuous monitoring)
python3 -m src.live_monitor --tail /var/log/netflows.csv --interval 5

# Pipe from nfdump
nfdump -r capture.nfcapd -o csv | python3 -m src.live_monitor --stdin

# With pre-trained model
python3 -m src.live_monitor --input flows.csv --model models/ids_A.pt --company A
```

Pipeline per flow: Feature engineering (41→66) → IDS classification → Zero-day detection → PII validation → Threat summarization.

---

## CLI Quick Reference

```bash
python3 -m src.cli data download                # Download datasets
python3 -m src.cli fl train --rounds 10          # Train FL model
python3 -m src.cli query "DDoS attacks"          # Query threat KB
python3 -m src.cli eval detection                # Evaluate accuracy
python3 -m src.cli eval pii --samples 500        # Evaluate PII leakage
python3 -m src.cli eval all                      # Run all evaluations
```

---

## Expected Results Summary

| Validation | Metric | Expected |
| --- | --- | --- |
| FL 4-Node (5 rounds) | Co. A Accuracy | ~76% |
| | Co. B Accuracy | ~71% |
| | Co. C Accuracy (BoT-IoT) | ~95% |
| | Co. D Accuracy (BoT-IoT) | ~95% |
| Zero-Day DDoS (BoT-IoT) | Detection Rate (α=0.3, p90) | **99.5%** |
| TCP SYN Flood (vs ReGAIN) | Accuracy | **100%** (vs 98.82%) |
| ICMP/UDP Flood (vs ReGAIN) | Accuracy | **99.5%** (vs 95.95%) |
| BERTScore (vs CyberRAG) | F1 | 0.919 (vs 0.94) |
| PII Leakage | Rate | **0%** |
| Feature Engineering | Dimensions | 41 → **66** |
| RAG Abstention | Hallucination prevention | ✅ threshold=0.30 |
