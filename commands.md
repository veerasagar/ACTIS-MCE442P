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
