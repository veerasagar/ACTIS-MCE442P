# ACTIS — Step-by-Step Commands

## 1. Setup

```bash
# Install all dependencies
python3 -m pip install -r requirements.txt

# Install additional packages (if not already installed)
python3 -m pip install chromadb sentence-transformers google-generativeai
```

```bash
# (Optional) Set Gemini API key for LLM-powered agents
echo 'GEMINI_API_KEY=your_key_here' > .env
```

---

## 2. Download Datasets

```bash
bash scripts/download_dataset.sh
```

This downloads **NF-CSE-CIC-IDS2018-v2** and **NF-BoT-IoT-v2** (~3 GB total) into `data/raw/`.

---

## 3. Verify Data Pipeline

```bash
python3 -c "
from src.data.loader import FedIntelDataLoader
loader = FedIntelDataLoader()
data = loader.load_and_partition()
for cid in ['A','B','C']:
    print(f'Company {cid}: {len(data[cid][\"data\"]):,} rows')
"
```

---

## 4. Test Local IDS Model (Standalone)

```bash
python3 -c "
import warnings; warnings.filterwarnings('ignore')
from src.data.loader import FedIntelDataLoader
from src.data.preprocessor import Preprocessor
from src.local_node.ids_model import IDSModel
from src.local_node.ids_trainer import IDSTrainer

loader = FedIntelDataLoader()
data = loader.load_and_partition()

for cid in ['A','C']:
    p = Preprocessor()
    X_tr, X_te, y_tr, y_te = p.prepare(data[cid], max_samples=20000)
    model = IDSModel(input_dim=41, num_classes=p.num_classes)
    trainer = IDSTrainer(model=model)
    trainer.train(X_tr, y_tr, epochs=10)
    m = trainer.evaluate(X_te, y_te)
    print(f'Company {cid}: Acc={m[\"accuracy\"]:.4f}, F1={m[\"f1\"]:.4f}')
"
```

---

## 5. Run Federated Learning (10 Rounds)

```bash
python3 -c "
import warnings; warnings.filterwarnings('ignore')
from src.server.fl_simulation import FLSimulation
sim = FLSimulation(max_samples=20000)
results = sim.run(num_rounds=10)
for cid, r in results['final_eval'].items():
    print(f'Company {cid}: Acc={r[\"accuracy\"]:.4f}, F1={r[\"f1\"]:.4f}')
"
```

---

## 6. Test Privacy Engine (PII Stripping + DP)

```bash
python3 -c "
import warnings; warnings.filterwarnings('ignore')
from src.local_node.node import PrivacyNode

node = PrivacyNode(company_id='A')
test_flow = {'PROTOCOL': 6, 'L4_SRC_PORT': 12345, 'L4_DST_PORT': 80,
             'IN_BYTES': 500000, 'OUT_BYTES': 200, 'IN_PKTS': 1000,
             'OUT_PKTS': 5, 'FLOW_DURATION_MILLISECONDS': 100, 'TCP_FLAGS': 2}
result = node.process_flow(test_flow, 'ddos')
print('Shared:', bool(result))
node.print_summary()
"
```

---

## 7. Test Global RAG + Immunity

```bash
python3 -c "
import warnings; warnings.filterwarnings('ignore')
import shutil
from src.server.global_kb import GlobalKnowledgeBase
from src.server.aggregator import Aggregator
from src.server.rag_engine import RAGEngine
from src.server.immunity import ImmunityEngine

kb = GlobalKnowledgeBase(persist_dir='/tmp/fedintel_kb')
kb.populate_mitre()
agg = Aggregator(kb)

# Simulate threat from Company A
agg.ingest({'analysis': {
    'attack_type': 'ddos', 'attack_description': 'SYN flood on port 80',
    'mitre_technique_id': 'T1498', 'mitre_tactic': 'Impact',
    'severity': 'HIGH', 'company_id': 'A',
    'recommended_defense': 'Rate limit SYN packets',
}})

# Company B queries RAG and gets immunity rules
rag = RAGEngine(kb)
results = rag.find_similar_attacks('ddos', 'B')
print(f'Cross-org intel for B: {len(results)} threats from other companies')

immunity = ImmunityEngine(rag)
rule = immunity.generate_rules('ddos', 'B')
print(f'Firewall rule: {rule[\"rule\"][:60]}...')

shutil.rmtree('/tmp/fedintel_kb', ignore_errors=True)
"
```

---

## 8. Run Full End-to-End Simulation

```bash
python3 -m src.cli simulate --rounds 5 --samples 10000 --threats 20
```

This runs **all 4 stages** in sequence:

1. Federated Learning (5 rounds)
2. Agentic Privacy Engine (20 threat flows per company)
3. Global Federated RAG (ingest + MITRE mapping)
4. Immunity Engine (firewall rules for all companies)

Outputs a Rich dashboard with all metrics.

---

## 9. Run Full Evaluation Suite

```bash
python3 -m src.evaluation.evaluate
```

This runs **4 evaluations** (~13 min):

1. Local-only vs Federated detection accuracy
2. PII leakage rate + DP privacy-utility tradeoff
3. Zero-day detection + threat report quality
4. Ablation study (5 configurations)

Results saved to `results/evaluation_results.json`.

---

## 10. MITRE ATT&CK Mapping Table

```bash
python3 -m src.data.mitre_mapper
```

---

## Individual CLI Commands

```bash
# Download data
python3 -m src.cli data download

# Train FL model
python3 -m src.cli fl train --rounds 10 --samples 20000

# Query the threat knowledge base
python3 -m src.cli query "DDoS attacks on web servers"

# Evaluate detection accuracy
python3 -m src.cli eval detection

# Evaluate PII leakage
python3 -m src.cli eval pii --samples 500

# Run all evaluations
python3 -m src.cli eval all
```

---

## Quick Demo (3 minutes)

For a fast demo showing the complete pipeline:

```bash
python3 -m src.cli simulate --rounds 3 --samples 5000 --threats 10
```

---

## 11. Install Additional Dependencies (High-Impact Fixes)

```bash
# BERTScore for report quality evaluation
python3 -m pip install bert-score

# Cross-encoder reranking for RAG (optional — fallback to bi-encoder if missing)
python3 -m pip install sentence-transformers
```

---

## 12. Test Feature Engineering (58-dim temporal features)

```bash
python3 -c "
import numpy as np, sys; sys.path.insert(0, '.')
from src.data.feature_engineer import FeatureEngineer
fe = FeatureEngineer()
X = np.random.rand(100, 41).astype('float32')
X_enh = fe.transform(X)
print(f'Input: {X.shape}  →  Output: {X_enh.shape}')
print('Features added:', fe.feature_names())
"
```

---

## 13. Test Per-Protocol Zero-Day Detector

```bash
python3 -c "
import warnings; warnings.filterwarnings('ignore')
import numpy as np, sys; sys.path.insert(0, '.')
from src.data.feature_engineer import FeatureEngineer
from src.local_node.ids_model import IDSModel
from src.local_node.zeroday_detector import ZeroDayDetector

fe = FeatureEngineer()
X_raw = np.random.rand(500, 41).astype('float32')
X_raw[:150, 2] = 6    # TCP
X_raw[150:300, 2] = 17 # UDP  (DDoS uses this)
X_raw[300:400, 2] = 1  # ICMP
X_enh = fe.transform(X_raw)
y = np.zeros(500, dtype=int); y[50:] = 1

model = IDSModel(input_dim=X_enh.shape[1], num_classes=9)
det = ZeroDayDetector(model, percentile=90, alpha=0.3)
det.fit_thresholds(X_enh, y)
print('Autoencoders:', [k for k,v in det._autoencoders.items() if v])
print('Threshold:', round(det.threshold, 4))
"
```

---

## 14. Test BERTScore Report Quality

```bash
python3 -c "
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
from src.evaluation.evaluate import _compute_bertscore

preds = ['DDoS attack detected with high-volume UDP flood targeting availability.']
refs  = ['Distributed Denial of Service with volumetric flood. MITRE T1498.']
bs = _compute_bertscore(preds, refs)
print(f'BERTScore  P={bs[\"precision\"]:.4f}  R={bs[\"recall\"]:.4f}  F1={bs[\"f1\"]:.4f}')
"
```

---

## 15. Live Monitor (Real-Time Deployment)

```bash
# Process a CSV of network flows and print alerts
python3 -m src.live_monitor --input flows.csv --company A

# Process CSV and save alerts to JSONL file
python3 -m src.live_monitor --input flows.csv --output alerts.jsonl

# Tail a growing live capture CSV (5-second poll)
python3 -m src.live_monitor --tail /var/log/netflows.csv --interval 5

# Use with a pre-trained saved model
python3 -m src.live_monitor --input flows.csv --model models/ids_A.pt --company A

# Pipe directly from nfdump
nfdump -r capture.nfcapd -o csv | python3 -m src.live_monitor --stdin
```

The live monitor runs the full pipeline per flow:

1. **Feature engineering** (41 → 58 dims)
2. **IDS classification** (IDSModel)
3. **Zero-day detection** (per-protocol ZeroDayDetector)
4. **PII validation** (blocks sensitive data)
5. **Threat summarization** (Gemini LLM for HIGH/CRITICAL, template fallback)

---

## 16. RAG with Cross-Encoder Reranking + Abstention

```bash
python3 -c "
import warnings; warnings.filterwarnings('ignore')
import shutil, sys; sys.path.insert(0, '.')
from src.server.global_kb import GlobalKnowledgeBase
from src.server.rag_engine import RAGEngine

kb = GlobalKnowledgeBase(persist_dir='/tmp/actis_kb_test')
kb.populate_mitre()

# RAG with abstention (returns abstained=True if similarity too low)
rag = RAGEngine(kb, abstention_threshold=0.30, use_reranker=True)
results = rag.query('DDoS amplification attack using UDP')
print(f'Abstained: {results[\"abstained\"]}')
print(f'Best score: {results[\"best_score\"]:.3f}')
print(f'Threats found: {len(results[\"threats\"])}')

shutil.rmtree('/tmp/actis_kb_test', ignore_errors=True)
"
```
