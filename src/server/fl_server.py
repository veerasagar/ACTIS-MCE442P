"""
Fed-Intel FL Server
====================
Federated aggregation server with two aggregation strategies:
  1. FedAvg — standard weighted average by sample count
  2. Trust-Aware FedAvg — weights influenced by loss + trust scores

Trust score formula: τ_i = 1 / (L_i + ε)
Aggregation weight: w_i = τ_i * n_i / Σ(τ_j * n_j)

Usage:
    server = FLServer(global_model)
    for round in range(num_rounds):
        client_results = [client.fit(server.get_weights()) for client in clients]
        server.aggregate(client_results)
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.local_node.ids_model import IDSModel
from src.config import FL_TRUST_EPSILON

console = Console()


class FLServer:
    """
    Federated Learning aggregation server.

    Aggregates model weights from multiple clients using
    trust-aware FedAvg.
    """

    def __init__(self, global_model: IDSModel):
        self.global_model = global_model
        self.round_history: List[Dict] = []
        self.trust_scores: Dict[str, float] = {}

    def get_weights(self) -> list:
        """Get current global model weights."""
        return self.global_model.get_weights()

    def set_weights(self, weights: list):
        """Set global model weights."""
        self.global_model.set_weights(weights)

    # ─── Aggregation ────────────────────────────────────────────────────

    def aggregate(
        self,
        client_results: List[Tuple[str, list, float, int]],
        strategy: str = "trust_aware",
    ) -> Dict:
        """
        Aggregate client model updates into the global model.

        Args:
            client_results: list of (client_id, weights, loss, n_samples)
            strategy: "fedavg" or "trust_aware"

        Returns:
            Dict with round metrics
        """
        if strategy == "trust_aware":
            return self._trust_aware_aggregate(client_results)
        else:
            return self._fedavg_aggregate(client_results)

    def _fedavg_aggregate(
        self, client_results: List[Tuple[str, list, float, int]]
    ) -> Dict:
        """Standard FedAvg: weighted average by sample count."""
        total_samples = sum(n for _, _, _, n in client_results)

        # Weighted average of all weight tensors
        aggregated = []
        for i in range(len(client_results[0][1])):  # For each weight tensor
            weighted_sum = sum(
                weights[i] * (n / total_samples)
                for _, weights, _, n in client_results
            )
            aggregated.append(weighted_sum)

        self.set_weights(aggregated)

        # Compute metrics
        avg_loss = np.mean([loss for _, _, loss, _ in client_results])
        round_info = {
            "strategy": "fedavg",
            "avg_loss": avg_loss,
            "total_samples": total_samples,
            "client_losses": {cid: loss for cid, _, loss, _ in client_results},
        }
        self.round_history.append(round_info)
        return round_info

    def _trust_aware_aggregate(
        self, client_results: List[Tuple[str, list, float, int]]
    ) -> Dict:
        """
        Trust-aware FedAvg with clamped weights.

        Blends sample-based FedAvg (stability) with trust-based weighting
        (performance), preventing any single client from dominating.

        α = 0.5 blend factor: w_i = α * (n_i/N) + (1-α) * (τ_i/Στ)
        Then clamp: max weight per client = 1/K + 0.15 (K = num clients)
        """
        n_clients = len(client_results)
        total_samples = sum(n for _, _, _, n in client_results)

        # Compute trust scores: τ_i = 1 / (L_i + ε)
        for cid, _, loss, _ in client_results:
            self.trust_scores[cid] = 1.0 / (loss + FL_TRUST_EPSILON)

        total_trust = sum(self.trust_scores[cid] for cid, _, _, _ in client_results)

        # Blend: 50% sample-weighted + 50% trust-weighted
        alpha = 0.5
        raw_weights = {}
        for cid, _, _, n in client_results:
            sample_w = n / total_samples
            trust_w = self.trust_scores[cid] / total_trust
            raw_weights[cid] = alpha * sample_w + (1 - alpha) * trust_w

        # Clamp: no client gets more than (1/K + 0.15)
        max_w = (1.0 / n_clients) + 0.15
        clamped = {cid: min(w, max_w) for cid, w in raw_weights.items()}
        total_clamped = sum(clamped.values())
        agg_weights = {cid: w / total_clamped for cid, w in clamped.items()}

        # Weighted average of model weights
        aggregated = []
        for i in range(len(client_results[0][1])):
            weighted_sum = sum(
                weights[i] * agg_weights[cid]
                for cid, weights, _, _ in client_results
            )
            aggregated.append(weighted_sum)

        self.set_weights(aggregated)

        # Compute metrics
        avg_loss = np.mean([loss for _, _, loss, _ in client_results])
        round_info = {
            "strategy": "trust_aware",
            "avg_loss": avg_loss,
            "trust_scores": dict(self.trust_scores),
            "aggregation_weights": agg_weights,
            "total_samples": total_samples,
            "client_losses": {cid: loss for cid, _, loss, _ in client_results},
        }
        self.round_history.append(round_info)

        # Log
        trust_str = " | ".join(
            "{}:w={:.3f}".format(cid, agg_weights[cid])
            for cid in sorted(agg_weights.keys())
        )
        console.print(
            f"  [green]Aggregated[/green] ({strategy_label(avg_loss)}): {trust_str}"
        )

        return round_info

    def get_round_history(self) -> List[Dict]:
        """Get the full history of FL rounds."""
        return self.round_history

    def get_best_round(self) -> int:
        """Get the round index with the lowest average loss."""
        if not self.round_history:
            return 0
        losses = [r["avg_loss"] for r in self.round_history]
        return int(np.argmin(losses))


def strategy_label(loss: float) -> str:
    """Color-code loss for display."""
    if loss < 0.05:
        return f"[bold green]loss={loss:.4f}[/bold green]"
    elif loss < 0.2:
        return f"[yellow]loss={loss:.4f}[/yellow]"
    else:
        return f"[red]loss={loss:.4f}[/red]"
