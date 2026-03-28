"""
ACTIS RAG Engine
==================
Retrieval-Augmented Generation engine for querying the global
threat intelligence knowledge base.

Pipeline (ReGAIN-inspired):
  Query → Embed → Metadata Filter → Bi-encoder Semantic Search
       → Cross-Encoder Reranking → MMR → Abstention Check → Top-K

New in v2:
  - Cross-encoder reranking: sentence-transformers re-scores top-k
    candidates for higher precision (not just embedding similarity)
  - Abstention: if best score < threshold, returns empty results
    rather than hallucinating low-confidence defense rules

Usage:
    engine = RAGEngine(kb)
    results = engine.query("DDoS attacks on port 80")
    context = engine.build_context(results)
    if results['abstained']:
        print('Low confidence — no rules generated')
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from rich.console import Console

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import RAG_TOP_K, RAG_MMR_LAMBDA
from src.server.global_kb import GlobalKnowledgeBase

console = Console()

# Lazy-load cross-encoder to avoid hard dependency
_cross_encoder = None
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def _get_cross_encoder():
    """Load the cross-encoder model once (lazy, cached)."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
            console.print(f"  [green]Cross-encoder loaded: {_CROSS_ENCODER_MODEL}[/green]")
        except Exception as e:
            console.print(f"  [yellow]Cross-encoder unavailable ({e}), using bi-encoder only[/yellow]")
            _cross_encoder = False  # Mark as unavailable
    return _cross_encoder if _cross_encoder is not False else None


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for threat intelligence.

    Retrieves relevant threat summaries and defense patterns from the
    global knowledge base, applies MMR for diversity, and builds
    context for downstream analysis or firewall rule generation.
    """

    def __init__(
        self,
        kb: GlobalKnowledgeBase,
        abstention_threshold: float = 0.30,
        use_reranker: bool = True,
    ):
        """
        Args:
            kb: Global knowledge base (ChromaDB wrapper)
            abstention_threshold: if best cosine similarity < this, abstain
                                  from returning results (avoids hallucination)
            use_reranker: whether to apply cross-encoder reranking
        """
        self.kb = kb
        self.abstention_threshold = abstention_threshold
        self.use_reranker = use_reranker

    def _rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Rerank candidates using a cross-encoder.

        Cross-encoder jointly encodes (query, document) — much higher
        precision than bi-encoder cosine similarity, at the cost of
        running one forward pass per candidate.
        """
        ce = _get_cross_encoder() if self.use_reranker else None
        if ce is None or not candidates:
            return candidates

        pairs = [(query, c["document"]) for c in candidates]
        try:
            scores = ce.predict(pairs, show_progress_bar=False)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        except Exception as e:
            console.print(f"  [yellow]Reranking failed ({e}), using original order[/yellow]")
            return candidates

    def _check_abstention(self, results: List[Dict]) -> Tuple[bool, float]:
        """
        Check whether to abstain from returning results.

        Returns:
            (should_abstain, best_score)
        """
        if not results:
            return True, 0.0

        # ChromaDB returns distances; convert to similarity (1 - dist)
        best_dist = results[0].get("distance", 1.0)
        best_sim = 1.0 - min(best_dist, 1.0)

        should_abstain = best_sim < self.abstention_threshold
        return should_abstain, best_sim

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
        Full RAG query with cross-encoder reranking and abstention.

        Args:
            query_text: natural language query
            top_k: number of results per collection
            filter_attack: optional attack type filter
            filter_company: optional company filter
            include_defenses: whether to query defense collection
            include_mitre: whether to query MITRE collection

        Returns:
            Dict with threats, defenses, mitre, abstained, best_score
        """
        result = {
            "query": query_text,
            "threats": [],
            "defenses": [],
            "mitre": [],
            "abstained": False,
            "best_score": 0.0,
        }

        # Query threats (retrieve 2x top_k for reranking)
        if self.kb.threats.count() > 0:
            candidates = self.kb.query_threats(
                query_text, top_k=top_k * 2,
                filter_attack=filter_attack,
                filter_company=filter_company,
            )

            # Abstention check on bi-encoder results
            should_abstain, best_sim = self._check_abstention(candidates)
            result["best_score"] = best_sim

            if should_abstain:
                console.print(
                    f"  [yellow]RAG abstaining — best similarity {best_sim:.3f} "
                    f"< threshold {self.abstention_threshold:.2f}[/yellow]"
                )
                result["abstained"] = True
                return result

            # Cross-encoder reranking
            candidates = self._rerank(query_text, candidates)
            result["threats"] = candidates[:top_k]

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
