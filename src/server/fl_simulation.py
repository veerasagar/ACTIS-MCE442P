"""
ACTIS FL Simulation
====================
Orchestrates the full federated learning simulation across 4 company nodes:
  A, B — Enterprise (CIC-IDS2018): DDoS, DoS, Brute Force, Web, Botnet
  C, D — IoT Sensor (BoT-IoT-v2):  DDoS/UDP, DoS/TCP, Reconnaissance, Theft

Flow per round:
  1. Server sends global weights to all clients
  2. Each client trains locally with FedProx + class-weighted loss
  3. Server aggregates using trust-aware FedAvg
  4. Evaluate global model on each client's test set

Usage:
    from src.server.fl_simulation import FLSimulation
    sim = FLSimulation(max_samples=50000)
    results = sim.run(num_rounds=10)
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.loader import FedIntelDataLoader
from src.data.preprocessor import Preprocessor
from src.local_node.ids_model import IDSModel
from src.server.fl_client import FLClient
from src.server.fl_server import FLServer
from src.config import (
    FL_NUM_ROUNDS,
    FL_NUM_CLIENTS,
    FL_LOCAL_EPOCHS,
    IDS_INPUT_DIM,
    IDS_HIDDEN_LAYERS,
    IDS_DROPOUT,
    MODELS_DIR,
)

console = Console()


class FLSimulation:
    """
    Runs the complete federated learning simulation.

    Creates N clients (one per company), a global model on the server,
    and orchestrates the FL training rounds.
    """

    def __init__(self, max_samples: Optional[int] = None):
        """
        Args:
            max_samples: limit per-company samples (for development/testing)
        """
        self.max_samples = max_samples
        self.clients: Dict[str, FLClient] = {}
        self.server: Optional[FLServer] = None
        self.preprocessors: Dict[str, Preprocessor] = {}
        self.round_metrics: list = []

    def setup(self):
        """Load data, create clients and server."""
        console.print("\n[bold]═══ Fed-Intel FL Simulation Setup ═══[/bold]\n")

        # Load and partition datasets
        loader = FedIntelDataLoader()
        company_data = loader.load_and_partition()

        # Collect ALL unique attack labels across ALL companies so that
        # class ID 3 means "ddos" everywhere regardless of node.
        all_labels = set()
        for cid in company_data.keys():
            labels = company_data[cid]["data"][company_data[cid]["label_col"]].unique()
            all_labels.update(labels)

        unified_labels = sorted(all_labels)
        num_classes = len(unified_labels)
        console.print(f"  Unified labels ({num_classes}): {unified_labels}")

        # ── Create Preprocessors + Clients ───────────────────────────────
        client_data = {}
        for cid in company_data.keys():
            console.print(f"\n[bold cyan]Setting up Client {cid}[/bold cyan]")
            prep = Preprocessor()
            # Pre-fit with the unified labels BEFORE encoding
            prep.set_unified_labels(unified_labels)
            X_train, X_test, y_train, y_test = prep.prepare(
                company_data[cid], max_samples=self.max_samples
            )
            self.preprocessors[cid] = prep

            # Compute class weights for this client's data distribution
            class_weights = prep.compute_class_weights(y_train)
            client_data[cid] = (X_train, X_test, y_train, y_test, prep, class_weights)

        for cid, (X_train, X_test, y_train, y_test, prep, class_weights) in client_data.items():
            client = FLClient(
                company_id=cid,
                X_train=X_train,
                y_train=y_train,
                num_classes=num_classes,
                X_test=X_test,
                y_test=y_test,
                class_names=prep.class_names,
                class_weights=class_weights,
            )
            self.clients[cid] = client

        # Create global model with same num_classes
        global_model = IDSModel(
            input_dim=IDS_INPUT_DIM,
            hidden_layers=IDS_HIDDEN_LAYERS,
            num_classes=num_classes,
            dropout=IDS_DROPOUT,
        )
        self.server = FLServer(global_model)

        console.print(
            f"\n[bold green]✓ Setup complete:[/bold green] "
            f"{len(self.clients)} clients, "
            f"global model {global_model.count_parameters():,} params, "
            f"unified classes: {num_classes}"
        )

    def run(self, num_rounds: int = FL_NUM_ROUNDS) -> Dict:
        """
        Run the full FL simulation.

        Args:
            num_rounds: number of federated training rounds

        Returns:
            Dict with final metrics and round history
        """
        if not self.server:
            self.setup()

        console.print(f"\n[bold]═══ Running {num_rounds} FL Rounds ═══[/bold]\n")
        start_time = time.time()

        for fl_round in range(1, num_rounds + 1):
            console.print(f"\n[bold yellow]── Round {fl_round}/{num_rounds} ──[/bold yellow]")

            # 1. Get current global weights
            global_weights = self.server.get_weights()

            # 2. Each client trains locally
            client_results = []
            for cid, client in self.clients.items():
                # For the first round, don't send global weights
                # (let each client start from its own init)
                weights_to_send = global_weights if fl_round > 1 else None
                weights, loss, n_samples = client.fit(weights_to_send)
                client_results.append((cid, weights, loss, n_samples))

            # 3. Server aggregates
            round_info = self.server.aggregate(client_results, strategy="trust_aware")
            round_info["round"] = fl_round

            # 4. Evaluate global model on each client's test set
            eval_results = self._evaluate_global(fl_round)
            round_info["eval"] = eval_results

            self.round_metrics.append(round_info)

        elapsed = time.time() - start_time
        console.print(f"\n[bold green]✓ FL training complete in {elapsed:.1f}s[/bold green]")

        # Final evaluation
        final_results = self._final_evaluation()

        return {
            "num_rounds": num_rounds,
            "elapsed_seconds": elapsed,
            "round_history": self.round_metrics,
            "final_eval": final_results,
        }

    def _evaluate_global(self, fl_round: int) -> Dict:
        """Evaluate the global model on each client's local test set."""
        global_weights = self.server.get_weights()
        eval_results = {}

        for cid, client in self.clients.items():
            # Temporarily set global weights on client model
            old_weights = client.get_weights()
            client.set_weights(global_weights)

            results = client.evaluate()
            if results:
                eval_results[cid] = {
                    "accuracy": results["accuracy"],
                    "f1": results["f1"],
                }

            # Restore client's own weights (for next round's local training)
            client.set_weights(old_weights)

        # Print round summary
        if eval_results:
            accs = [v["accuracy"] for v in eval_results.values()]
            f1s = [v["f1"] for v in eval_results.values()]
            details = ', '.join(
                '{}:{:.3f}'.format(k, v['accuracy'])
                for k, v in eval_results.items()
            )
            console.print(
                f"  [green]Global eval[/green] — "
                f"Avg Acc: {np.mean(accs):.4f}, "
                f"Avg F1: {np.mean(f1s):.4f} "
                f"({details})"
            )

        return eval_results

    def _final_evaluation(self) -> Dict:
        """Print final evaluation results table."""
        console.print("\n[bold]═══ Final FL Evaluation ═══[/bold]\n")

        global_weights = self.server.get_weights()

        table = Table(title="📊 Global Model Performance (Post-FL)", show_lines=True)
        table.add_column("Company", style="bold cyan")
        table.add_column("Accuracy", justify="right", style="green")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1", justify="right", style="bold green")
        table.add_column("Classes", justify="right")

        final = {}
        for cid, client in self.clients.items():
            client.set_weights(global_weights)
            results = client.evaluate()
            if results:
                final[cid] = results
                table.add_row(
                    f"Company {cid}",
                    f"{results['accuracy']:.4f}",
                    f"{results['precision']:.4f}",
                    f"{results['recall']:.4f}",
                    f"{results['f1']:.4f}",
                    str(len(self.preprocessors[cid].class_names)),
                )

        console.print(table)

        # Print convergence summary
        if self.round_metrics:
            losses = [r["avg_loss"] for r in self.round_metrics]
            console.print(f"\n  Loss curve: {losses[0]:.4f} → {losses[-1]:.4f} "
                           f"(Δ = {losses[0] - losses[-1]:.4f})")
            best_round = self.server.get_best_round()
            console.print(f"  Best round: {best_round + 1} (loss={losses[best_round]:.4f})")

        return final

    def save_global_model(self, path: Optional[str] = None):
        """Save the global model to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = str(MODELS_DIR / "global_ids_model.pt")

        import torch
        torch.save(self.server.global_model.state_dict(), path)
        console.print(f"[green]Global model saved to {path}[/green]")


# ─── Standalone Usage ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # Quick test: 5 rounds, 20K samples per company
    sim = FLSimulation(max_samples=20000)
    results = sim.run(num_rounds=5)
    sim.save_global_model()
