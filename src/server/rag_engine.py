"""
Fed-Intel RAG Engine
=====================
Retrieval-Augmented Generation engine for querying the global
threat intelligence knowledge base.

Pipeline: Query → Embed → Metadata Filter → Semantic Search → MMR → Top-K

Usage:
    engine = RAGEngine(kb)
    results = engine.query("DDoS attacks on port 80")
    context = engine.build_context(results)
"""

import numpy as np
from typing import Dict, List, Optional
from rich.console import Console

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import RAG_TOP_K, RAG_MMR_LAMBDA
from src.server.global_kb import GlobalKnowledgeBase

console = Console()


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for threat intelligence.

    Retrieves relevant threat summaries and defense patterns from the
    global knowledge base, applies MMR for diversity, and builds
    context for downstream analysis or firewall rule generation.
    """

    def __init__(self, kb: GlobalKnowledgeBase):
        self.kb = kb

    def query(
        self,
        query_text: str,
        top_k: int = RAG_TOP_K,
        filter_attack: Optional[str] = None,
        filter_company: Optional[str] = None,
        include_defenses: bool = True,
        include_mitre: bool = True,
    ) -> Dict:
        """
        Full RAG query against the knowledge base.

        Args:
            query_text: natural language query
            top_k: number of results per collection
            filter_attack: optional attack type filter
            filter_company: optional company filter
            include_defenses: whether to query defense collection
            include_mitre: whether to query MITRE collection

        Returns:
            Dict with threats, defenses, mitre results
        """
        result = {
            "query": query_text,
            "threats": [],
            "defenses": [],
            "mitre": [],
        }

        # Query threats
        if self.kb.threats.count() > 0:
            result["threats"] = self.kb.query_threats(
                query_text, top_k=top_k,
                filter_attack=filter_attack,
                filter_company=filter_company,
            )

        # Query defenses
        if include_defenses and self.kb.defenses.count() > 0:
            result["defenses"] = self.kb.query_defenses(
                query_text, top_k=min(top_k, 3),
            )

        # Query MITRE
        if include_mitre and self.kb.mitre.count() > 0:
            query_emb = self.kb.embed_text(query_text)
            mitre_results = self.kb.mitre.query(
                query_embeddings=[query_emb],
                n_results=min(3, self.kb.mitre.count()),
            )
            result["mitre"] = self.kb._format_results(mitre_results)

        return result

    def build_context(self, results: Dict, max_chars: int = 2000) -> str:
        """
        Build a context string from RAG results for downstream use.

        Args:
            results: output from self.query()
            max_chars: max context length

        Returns:
            Formatted context string
        """
        parts = []

        # Threats
        if results["threats"]:
            parts.append("=== Relevant Threats ===")
            for i, t in enumerate(results["threats"][:5], 1):
                parts.append("{}. {}".format(i, t["document"]))
                meta = t.get("metadata", {})
                if meta.get("severity"):
                    parts.append(
                        "   Severity: {} | Company: {}".format(
                            meta.get("severity", "?"),
                            meta.get("company_id", "?"),
                        )
                    )

        # Defenses
        if results["defenses"]:
            parts.append("\n=== Known Defenses ===")
            for d in results["defenses"][:3]:
                parts.append("- {}".format(d["document"]))

        # MITRE
        if results["mitre"]:
            parts.append("\n=== MITRE ATT&CK References ===")
            for m in results["mitre"][:3]:
                parts.append("- {}".format(m["document"]))

        context = "\n".join(parts)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n... (truncated)"

        return context

    def find_similar_attacks(
        self,
        attack_type: str,
        company_id: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Find similar attacks from OTHER companies (cross-org intelligence).

        This is the core federation value — Company B can learn about
        attacks that first hit Company A.
        """
        if self.kb.threats.count() == 0:
            return []

        query = "{} attack patterns and indicators".format(attack_type)
        results = self.kb.query_threats(
            query, top_k=top_k * 2,
            filter_attack=attack_type,
        )

        # Filter to other companies only
        cross_org = [
            r for r in results
            if r.get("metadata", {}).get("company_id") != company_id
        ]
        return cross_org[:top_k]

    def get_threat_summary(self) -> Dict:
        """Get a summary of the knowledge base for dashboard display."""
        stats = self.kb.get_stats()

        # Get attack type distribution
        attack_types = {}
        if stats["threats"] > 0:
            all_threats = self.kb.threats.get(
                limit=min(stats["threats"], 1000),
                include=["metadatas"],
            )
            if all_threats and all_threats.get("metadatas"):
                for meta in all_threats["metadatas"]:
                    atype = meta.get("attack_type", "unknown")
                    attack_types[atype] = attack_types.get(atype, 0) + 1

        return {
            "total_threats": stats["threats"],
            "total_defenses": stats["defenses"],
            "mitre_refs": stats["mitre_refs"],
            "attack_distribution": attack_types,
        }
