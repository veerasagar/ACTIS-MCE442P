"""Phase 7 — BERTScore report quality evaluation and threat summarizer test."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
from src.evaluation.evaluate import _compute_bertscore
from src.data.summarizer import ThreatSummarizer

# ── BERTScore ───────────────────────────────────────────────────
print("=== BERTScore Report Quality ===")
preds = ["DDoS attack detected with high-volume UDP flood targeting service availability. MITRE T1498."]
refs = ["Distributed Denial of Service attack detected with volumetric UDP flood pattern impacting availability."]
bs = _compute_bertscore(preds, refs)
print(f"  Precision: {bs['precision']:.4f}")
print(f"  Recall:    {bs['recall']:.4f}")
print(f"  F1:        {bs['f1']:.4f}")
print()
print(f"  CyberRAG baseline: F1 = 0.9400")
print(f"  ACTIS:             F1 = {bs['f1']:.4f}")

# ── Threat Summarizer ──────────────────────────────────────────
print()
print("=== Threat Summarizer (LLM-Enhanced) ===")
s = ThreatSummarizer(use_llm=True)
flow = {
    "PROTOCOL": 17, "L4_SRC_PORT": 1234, "L4_DST_PORT": 53,
    "IN_BYTES": 5000000, "OUT_BYTES": 200, "IN_PKTS": 10000,
    "OUT_PKTS": 5, "FLOW_DURATION_MILLISECONDS": 100, "TCP_FLAGS": 0,
}
r = s.summarize_flow(flow, "ddos", "A")
print(f"  LLM enhanced: {r['llm_enhanced']}")
print(f"  Severity:     {r['severity']}")
print(f"  MITRE:        {r['mitre_technique_id']} ({r['mitre_tactic']})")
print(f"  Summary:      {r['summary'][:120]}...")
