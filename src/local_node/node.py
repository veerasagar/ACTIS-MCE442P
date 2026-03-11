"""
Fed-Intel Node Pipeline Orchestrator
=====================================
Orchestrates the full privacy-preserving pipeline for one company node:

  Anomalous Flow → Analyzer → Sanitizer → PII Validator → DP Noise → Share

This is the core of Layer 2 (Agentic Privacy Engine). Each node runs
this pipeline on its flagged traffic before sharing with the global server.

Usage:
    node = PrivacyNode(company_id="A")
    results = node.process_batch(flows, labels)
    node.print_summary()
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.local_node.agent_analyzer import AgentAnalyzer
from src.local_node.agent_sanitizer import AgentSanitizer
from src.local_node.pii_validator import PIIValidator
from src.local_node.dp_layer import DPLayer
from src.config import DP_EPSILON, PII_MAX_RETRIES

console = Console()


class PrivacyNode:
    """
    Full pipeline orchestrator for one company node.

    Pipeline stages:
    1. Agent Analyzer — reads anomalous flow, produces structured analysis
    2. Agent Sanitizer — strips PII from analysis
    3. PII Validator — verifies no PII remains (retry loop if needed)
    4. DP Layer — adds calibrated noise to embedding
    5. Output — privacy-safe JSON + noisy embedding, ready for federation
    """

    def __init__(
        self,
        company_id: str,
        epsilon: float = DP_EPSILON,
        max_pii_retries: int = PII_MAX_RETRIES,
    ):
        self.company_id = company_id
        self.max_pii_retries = max_pii_retries

        console.print(
            "\n[bold cyan]═══ Initializing Privacy Node "
            "{} ═══[/bold cyan]".format(company_id)
        )

        # Initialize pipeline components
        self.analyzer = AgentAnalyzer()
        self.sanitizer = AgentSanitizer()
        self.validator = PIIValidator()
        self.dp_layer = DPLayer(epsilon=epsilon)

        # Stats
        self.processed = 0
        self.shared = 0
        self.blocked = 0
        self.timings: List[float] = []

        dp_params = self.dp_layer.get_privacy_params()
        console.print(
            "  [dim]DP guarantee: {}[/dim]".format(dp_params["guarantee"])
        )

    def process_flow(
        self,
        flow_row: Dict,
        attack_label: str,
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[Dict]:
        """
        Process one anomalous flow through the full privacy pipeline.

        Args:
            flow_row: dict of flow features
            attack_label: normalized attack label
            embedding: optional pre-computed embedding (384-d)

        Returns:
            Privacy-safe threat summary dict, or None if blocked
        """
        start = time.time()
        self.processed += 1

        # Stage 1: Analyze
        analysis = self.analyzer.analyze(flow_row, attack_label, self.company_id)

        # Stage 2: Sanitize
        sanitized = self.sanitizer.sanitize(analysis)

        # Stage 3: Validate (with retry loop)
        is_clean = False
        for attempt in range(self.max_pii_retries):
            is_clean, findings = self.validator.validate(sanitized)
            if is_clean:
                break
            # Re-sanitize if PII found
            sanitized = self.sanitizer.sanitize(sanitized)

        if not is_clean:
            # Final fallback: block sharing
            self.blocked += 1
            self.timings.append(time.time() - start)
            return None

        # Stage 4: DP noise on embedding (if provided)
        noisy_embedding = None
        if embedding is not None:
            noisy_embedding = self.dp_layer.add_noise(embedding)

        # Build final output
        output = {
            "company_id": self.company_id,
            "analysis": sanitized,
            "embedding": noisy_embedding,
            "privacy": {
                "pii_validated": True,
                "dp_epsilon": self.dp_layer.epsilon,
                "dp_sigma": round(self.dp_layer.sigma, 4),
            },
        }

        self.shared += 1
        self.timings.append(time.time() - start)
        return output

    def process_batch(
        self,
        flows: List[Dict],
        labels: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Process a batch of anomalous flows.

        Args:
            flows: list of flow feature dicts
            labels: list of attack labels
            embeddings: optional (n, d) embedding array

        Returns:
            List of privacy-safe summaries
        """
        results = []
        for i, (flow, label) in enumerate(zip(flows, labels)):
            emb = embeddings[i] if embeddings is not None else None
            result = self.process_flow(flow, label, emb)
            if result is not None:
                results.append(result)

        return results

    def get_stats(self) -> Dict:
        """Get comprehensive pipeline statistics."""
        avg_time = (
            sum(self.timings) / len(self.timings) if self.timings else 0
        )
        return {
            "company_id": self.company_id,
            "processed": self.processed,
            "shared": self.shared,
            "blocked": self.blocked,
            "share_rate": (
                self.shared / max(self.processed, 1) * 100
            ),
            "pii_leakage_rate": self.validator.get_stats()["leakage_rate"],
            "avg_latency_ms": round(avg_time * 1000, 2),
            "sanitizer_stats": self.sanitizer.get_stats(),
            "dp_stats": self.dp_layer.get_stats(),
            "dp_params": self.dp_layer.get_privacy_params(),
        }

    def print_summary(self):
        """Print a rich summary of pipeline performance."""
        stats = self.get_stats()

        table = Table(
            title="Privacy Node {} — Summary".format(self.company_id),
            show_header=True,
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Flows processed", str(stats["processed"]))
        table.add_row("Flows shared", str(stats["shared"]))
        table.add_row("Flows blocked (PII)", str(stats["blocked"]))
        table.add_row(
            "Share rate",
            "{:.1f}%".format(stats["share_rate"]),
        )
        table.add_row(
            "PII leakage rate",
            "{:.1f}%".format(stats["pii_leakage_rate"]),
        )
        table.add_row(
            "Avg latency",
            "{:.2f}ms".format(stats["avg_latency_ms"]),
        )
        table.add_row(
            "DP guarantee",
            stats["dp_params"]["guarantee"],
        )
        table.add_row(
            "DP σ (noise std)",
            str(stats["dp_params"]["sigma"]),
        )

        console.print(table)
