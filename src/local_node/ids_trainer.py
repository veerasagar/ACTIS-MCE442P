"""
Fed-Intel IDS Trainer
======================
Local training loop for the IDS model.
Handles single-node training, evaluation, and metric reporting.

For FL: trains for `local_epochs` per round, then returns weights + loss.

Usage:
    from src.local_node.ids_trainer import IDSTrainer
    trainer = IDSTrainer(model, device="cpu")
    metrics = trainer.train(X_train, y_train, epochs=5)
    results = trainer.evaluate(X_test, y_test)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.local_node.ids_model import IDSModel
from src.config import FL_BATCH_SIZE, FL_LEARNING_RATE, FL_LOCAL_EPOCHS

console = Console()


class IDSTrainer:
    """Trains and evaluates the IDS model locally."""

    def __init__(
        self,
        model: IDSModel,
        device: str = None,
        learning_rate: float = FL_LEARNING_RATE,
        batch_size: int = FL_BATCH_SIZE,
        class_weights: "torch.Tensor" = None,
        fedprox_mu: float = 0.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.fedprox_mu = fedprox_mu
        self._global_params_ref = None  # Snapshot of global weights for FedProx

        # Use class-weighted loss if weights provided
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.5
        )

    def set_global_weights_ref(self):
        """Snapshot current model weights as the FedProx reference point."""
        self._global_params_ref = [
            p.clone().detach() for p in self.model.parameters()
        ]

    # ─── Data Loading ───────────────────────────────────────────────────

    def _make_dataloader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        """Create a PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    # ─── Training ───────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = FL_LOCAL_EPOCHS,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the model for a given number of epochs.

        Args:
            X_train: training features (numpy)
            y_train: training labels (numpy)
            epochs: number of training epochs
            verbose: print per-epoch metrics

        Returns:
            Dict with training metrics (loss, accuracy per epoch)
        """
        self.model.train()
        dataloader = self._make_dataloader(X_train, y_train, shuffle=True)

        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                # FedProx: add proximal term μ/2 * ||w - w_global||²
                if self.fedprox_mu > 0 and self._global_params_ref is not None:
                    prox_term = 0.0
                    for p, g in zip(self.model.parameters(), self._global_params_ref):
                        prox_term += ((p - g) ** 2).sum()
                    loss = loss + (self.fedprox_mu / 2.0) * prox_term

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += X_batch.size(0)

            epoch_loss = total_loss / total
            epoch_acc = correct / total
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_acc)

            self.scheduler.step(epoch_loss)

            if verbose:
                console.print(
                    f"  Epoch {epoch + 1}/{epochs} — "
                    f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
                )

        return history

    # ─── Evaluation ─────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: list = None,
    ) -> Dict:
        """
        Evaluate the model on test data.

        Returns:
            Dict with accuracy, precision, recall, f1, and per-class report
        """
        self.model.eval()
        dataloader = self._make_dataloader(X_test, y_test, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                preds = torch.argmax(self.model(X_batch), dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        results = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
            "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(all_labels, all_preds),
        }

        if class_names:
            # Use labels param so it doesn't break when not all classes appear
            unique_labels = sorted(set(all_labels) | set(all_preds))
            # Only use class_names for labels that actually exist
            label_names = [class_names[i] if i < len(class_names) else str(i)
                           for i in unique_labels]
            results["classification_report"] = classification_report(
                all_labels, all_preds,
                labels=unique_labels,
                target_names=label_names,
                zero_division=0,
            )

        return results

    def print_results(self, results: Dict, title: str = "Evaluation"):
        """Print evaluation results as a Rich table."""
        table = Table(title=f"📊 {title}", show_lines=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Accuracy", f"{results['accuracy']:.4f}")
        table.add_row("Precision (weighted)", f"{results['precision']:.4f}")
        table.add_row("Recall (weighted)", f"{results['recall']:.4f}")
        table.add_row("F1 Score (weighted)", f"{results['f1']:.4f}")

        console.print(table)

        if "classification_report" in results:
            console.print("\n[bold]Per-Class Report:[/bold]")
            console.print(results["classification_report"])

    # ─── FL Helpers ─────────────────────────────────────────────────────

    def train_one_fl_round(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = FL_LOCAL_EPOCHS,
    ) -> Tuple[list, float]:
        """
        Train for one FL round and return weights + loss.

        Returns:
            (weights, final_loss) — weights as list of numpy arrays
        """
        history = self.train(X_train, y_train, epochs=epochs, verbose=False)
        final_loss = history["loss"][-1]
        weights = self.model.get_weights()
        return weights, final_loss

    # ─── Save / Load ────────────────────────────────────────────────────

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "input_dim": self.model.input_dim,
            "num_classes": self.model.num_classes,
        }, path)
        console.print(f"  [green]Model saved to {path}[/green]")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        console.print(f"  [green]Model loaded from {path}[/green]")


# ─── Standalone Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    from src.data.loader import FedIntelDataLoader
    from src.data.preprocessor import Preprocessor

    # Load data
    loader = FedIntelDataLoader()
    data = loader.load_and_partition()

    # Test on Company A (sampled for speed)
    prep = Preprocessor()
    X_train, X_test, y_train, y_test = prep.prepare(
        data["A"], max_samples=20000
    )

    # Create & train model
    model = IDSModel(input_dim=X_train.shape[1], num_classes=prep.num_classes)
    console.print(f"\n[bold]Model: {model.count_parameters():,} parameters[/bold]")

    trainer = IDSTrainer(model)
    console.print("\n[bold]Training Company A (20K samples, 5 epochs):[/bold]")
    history = trainer.train(X_train, y_train, epochs=5)

    # Evaluate
    results = trainer.evaluate(X_test, y_test, class_names=prep.class_names)
    trainer.print_results(results, title="Company A — Local IDS")
