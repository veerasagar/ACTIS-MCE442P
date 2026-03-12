"""
Fed-Intel Aggregator
=====================
Receives sanitized threat summaries from company nodes and
stores them in the global knowledge base.

This is the RAG channel endpoint — the counterpart to Channel A
(federated learning weights). Channel B handles threat intelligence
as privacy-safe JSON summaries.

Usage:
    aggregator = Aggregator(kb)
    aggregator.ingest(sanitized_summary)
    aggregator.ingest_batch(summaries)
"""

import time
from typing import Dict, List
from rich.console import Console

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.server.global_kb import GlobalKnowledgeBase

console = Console()


class Aggregator:
    """
    Threat intelligence aggregator.
    Receives privacy-safe summaries from nodes and stores in global KB.
    """

    def __init__(self, kb: GlobalKnowledgeBase):
        self.kb = kb
        self.stats = {
            "ingested": 0,
            "defenses_generated": 0,
            "companies_seen": set(),
        }

    def ingest(self, summary: Dict) -> str:
        """
        Ingest one sanitized threat summary into the global KB.

        Args:
            summary: output from PrivacyNode.process_flow()

        Returns:
            Document ID in the knowledge base
        """
        analysis = summary.get("analysis", summary)

        # Add to threats collection
        doc_id = self.kb.add_threat(summary)

        # Track stats
        self.stats["ingested"] += 1
        cid = analysis.get("company_id", "unknown")
        self.stats["companies_seen"].add(cid)

        # Auto-generate defense rule from the threat
        defense_rec = analysis.get("recommended_defense", "")
        if defense_rec:
            self.kb.add_defense({
                "attack_type": analysis.get("attack_type", ""),
                "rule": defense_rec,
                "rule_type": "auto_generated",
                "source_company": cid,
            })
            self.stats["defenses_generated"] += 1

        return doc_id

    def ingest_batch(self, summaries: List[Dict]) -> List[str]:
        """Ingest a batch of sanitized summaries."""
        ids = []
        for s in summaries:
            doc_id = self.ingest(s)
            ids.append(doc_id)
        return ids

    def get_stats(self) -> Dict:
        """Get aggregation statistics."""
        return {
            "ingested": self.stats["ingested"],
            "defenses_generated": self.stats["defenses_generated"],
            "companies_contributing": len(self.stats["companies_seen"]),
            "companies": list(self.stats["companies_seen"]),
            "kb_stats": self.kb.get_stats(),
        }
