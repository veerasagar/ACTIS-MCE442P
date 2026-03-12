"""
Fed-Intel Global Knowledge Base
=================================
ChromaDB-backed vector store with 3 specialized collections:

  1. threat_summaries  — attack descriptions from all nodes
  2. defense_patterns  — auto-generated firewall/defense rules
  3. mitre_reference   — MITRE ATT&CK tactic embeddings

Usage:
    kb = GlobalKnowledgeBase()
    kb.add_threat(sanitized_summary, embedding)
    results = kb.query_threats("DDoS attacks targeting port 80", top_k=5)
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_THREATS,
    CHROMA_COLLECTION_DEFENSE,
    CHROMA_COLLECTION_MITRE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    RAG_TOP_K,
)

console = Console()


class GlobalKnowledgeBase:
    """
    Multi-collection ChromaDB knowledge base for federated threat intelligence.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        import chromadb

        self.persist_dir = str(persist_dir or CHROMA_DIR)
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Initialize 3 collections
        self.threats = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_THREATS,
            metadata={"description": "Sanitized threat summaries from all nodes"},
        )
        self.defenses = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_DEFENSE,
            metadata={"description": "Auto-generated defense/firewall rules"},
        )
        self.mitre = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_MITRE,
            metadata={"description": "MITRE ATT&CK tactic reference"},
        )

        self._embedder = None  # Lazy-load

        console.print(
            "  [green]Global KB initialized:[/green] "
            "{} threats, {} defenses, {} MITRE refs".format(
                self.threats.count(), self.defenses.count(), self.mitre.count()
            )
        )

    def _get_embedder(self):
        """Lazy-load sentence-transformer model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
            console.print("  [dim]Loaded embedding model: {}[/dim]".format(EMBEDDING_MODEL))
        return self._embedder

    def embed_text(self, text: str) -> List[float]:
        """Embed a text string using sentence-transformers."""
        model = self._get_embedder()
        emb = model.encode(text, normalize_embeddings=True)
        return emb.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        model = self._get_embedder()
        embs = model.encode(texts, normalize_embeddings=True)
        return embs.tolist()

    # ─── Threat Summaries ───────────────────────────────────────────────

    def add_threat(
        self,
        summary: Dict,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Add a sanitized threat summary to the knowledge base.

        Args:
            summary: sanitized analysis dict from PrivacyNode
            embedding: optional pre-computed embedding

        Returns:
            ID of the added document
        """
        analysis = summary.get("analysis", summary)
        doc_id = "threat_{}_{}".format(
            analysis.get("company_id", "X"),
            int(time.time() * 1000),
        )

        # Build document text for embedding
        doc_text = "{} attack — {}. MITRE: {} ({}). Severity: {}.".format(
            analysis.get("attack_type", "unknown"),
            analysis.get("attack_description", ""),
            analysis.get("mitre_technique_id", ""),
            analysis.get("mitre_tactic", ""),
            analysis.get("severity", "unknown"),
        )

        # Use provided embedding or compute one
        if embedding is None:
            embedding = self.embed_text(doc_text)
        elif isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        # Metadata
        metadata = {
            "company_id": str(analysis.get("company_id", "unknown")),
            "attack_type": str(analysis.get("attack_type", "")),
            "mitre_id": str(analysis.get("mitre_technique_id", "")),
            "severity": str(analysis.get("severity", "")),
            "timestamp": str(int(time.time())),
        }

        self.threats.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[metadata],
        )

        return doc_id

    def add_threats_batch(self, summaries: List[Dict]) -> List[str]:
        """Add multiple threat summaries."""
        ids = []
        for s in summaries:
            doc_id = self.add_threat(s)
            ids.append(doc_id)
        return ids

    def query_threats(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        filter_attack: Optional[str] = None,
        filter_company: Optional[str] = None,
    ) -> List[Dict]:
        """
        Query threats using semantic search.

        Args:
            query: natural language query
            top_k: number of results
            filter_attack: optional attack type filter
            filter_company: optional company filter

        Returns:
            List of matching threat dicts
        """
        query_emb = self.embed_text(query)

        where_filter = None
        conditions = []
        if filter_attack:
            conditions.append({"attack_type": filter_attack})
        if filter_company:
            conditions.append({"company_id": filter_company})
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        results = self.threats.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, max(self.threats.count(), 1)),
            where=where_filter if where_filter else None,
        )

        return self._format_results(results)

    # ─── Defense Patterns ───────────────────────────────────────────────

    def add_defense(self, rule: Dict) -> str:
        """Add a defense/firewall rule to the knowledge base."""
        doc_id = "defense_{}_{}".format(
            rule.get("attack_type", "x"),
            int(time.time() * 1000),
        )

        doc_text = "Defense against {}: {}".format(
            rule.get("attack_type", "unknown"),
            rule.get("rule", ""),
        )
        embedding = self.embed_text(doc_text)

        metadata = {
            "attack_type": str(rule.get("attack_type", "")),
            "rule_type": str(rule.get("rule_type", "generic")),
            "source_company": str(rule.get("source_company", "")),
            "timestamp": str(int(time.time())),
        }

        self.defenses.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[metadata],
        )
        return doc_id

    def query_defenses(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query defense patterns."""
        query_emb = self.embed_text(query)
        results = self.defenses.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, max(self.defenses.count(), 1)),
        )
        return self._format_results(results)

    # ─── MITRE Reference ────────────────────────────────────────────────

    def populate_mitre(self):
        """Populate MITRE collection from the MITRE mapper."""
        from src.data.mitre_mapper import MITRE_MAP

        if self.mitre.count() >= len(MITRE_MAP):
            return  # Already populated

        for attack_type, info in MITRE_MAP.items():
            doc_text = "{} — {} ({}). {} tactic. {}".format(
                info["technique_id"],
                info["technique"],
                attack_type,
                info["tactic"],
                info.get("description", ""),
            )
            embedding = self.embed_text(doc_text)

            self.mitre.add(
                ids=["mitre_{}".format(attack_type)],
                embeddings=[embedding],
                documents=[doc_text],
                metadatas=[{
                    "attack_type": attack_type,
                    "technique_id": info["technique_id"],
                    "tactic": info["tactic"],
                    "severity": info["severity"],
                }],
            )

        console.print(
            "  [green]Populated {} MITRE references[/green]".format(
                self.mitre.count()
            )
        )

    # ─── Utilities ──────────────────────────────────────────────────────

    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB query results into a clean list."""
        formatted = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return formatted

        for i, doc_id in enumerate(results["ids"][0]):
            entry = {
                "id": doc_id,
                "document": results["documents"][0][i] if results.get("documents") else "",
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "distance": results["distances"][0][i] if results.get("distances") else None,
            }
            formatted.append(entry)
        return formatted

    def get_stats(self) -> Dict:
        """Get KB statistics."""
        return {
            "threats": self.threats.count(),
            "defenses": self.defenses.count(),
            "mitre_refs": self.mitre.count(),
            "persist_dir": self.persist_dir,
        }

    def clear(self):
        """Clear all collections (for testing)."""
        self.client.delete_collection(CHROMA_COLLECTION_THREATS)
        self.client.delete_collection(CHROMA_COLLECTION_DEFENSE)
        self.client.delete_collection(CHROMA_COLLECTION_MITRE)
        # Recreate
        self.__init__(self.persist_dir)
