"""Phase 6 — Validate RAG engine with cross-encoder reranking and abstention."""
import warnings; warnings.filterwarnings('ignore')
import shutil
import sys; sys.path.insert(0, '.')
from src.server.global_kb import GlobalKnowledgeBase
from src.server.aggregator import Aggregator
from src.server.rag_engine import RAGEngine
from src.server.immunity import ImmunityEngine

KB_DIR = "/tmp/actis_kb_validate"

# ── Setup KB ────────────────────────────────────────────────────
kb = GlobalKnowledgeBase(persist_dir=KB_DIR)
kb.populate_mitre()

# ── RAG Abstention ──────────────────────────────────────────────
print("=== RAG Engine — Cross-Encoder Reranking + Abstention ===")
rag = RAGEngine(kb, abstention_threshold=0.30, use_reranker=True)

queries = [
    ("DDoS amplification attack using UDP", False),
    ("completely irrelevant query about cooking recipes", True),
]
for query, expect_abstain in queries:
    r = rag.query(query)
    status = "✓" if r["abstained"] == expect_abstain else "✗"
    print(f"  {status} Query: \"{query[:50]}\"")
    print(f"    Abstained: {r['abstained']}  (expected: {expect_abstain})")
    print(f"    Score: {r['best_score']:.3f}   Threats: {len(r['threats'])}")

# ── Cross-Org Intelligence ──────────────────────────────────────
print()
print("=== Cross-Org Threat Intelligence ===")
agg = Aggregator(kb)
agg.ingest({"analysis": {
    "attack_type": "ddos", "attack_description": "UDP volumetric flood on port 53",
    "mitre_technique_id": "T1498", "mitre_tactic": "Impact",
    "severity": "CRITICAL", "company_id": "A",
    "recommended_defense": "Rate limit UDP 53, enable SYN cookies",
}})

results = rag.find_similar_attacks("ddos", "B")
immunity = ImmunityEngine(rag)
rule = immunity.generate_rules("ddos", "B")

print(f"  Threats shared A → B: {len(results)}")
print(f"  Firewall rule for B:  {rule['rule'][:80]}...")

shutil.rmtree(KB_DIR, ignore_errors=True)
