"""
Fed-Intel FL Client
====================
Wraps IDSModel + IDSTrainer into a federated learning client.
Each company node runs one FLClient.

Usage:
    client = FLClient(company_id="A", X_train=..., y_train=..., num_classes=7)
    weights, loss, n_samples = client.fit(global_weights)
    metrics = client.evaluate(X_test, y_test)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.local_node.ids_model import IDSModel
from src.local_node.ids_trainer import IDSTrainer
from src.config import (
    IDS_INPUT_DIM,
    IDS_HIDDEN_LAYERS,
    IDS_DROPOUT,
    FL_LOCAL_EPOCHS,
    FL_LEARNING_RATE,
    FL_BATCH_SIZE,
)

console = Console()


class FLClient:
    """
    Federated Learning client for one company node.

    Each round:
    1. Receives global weights from server
    2. Sets weights on local model
    3. Trains locally for FL_LOCAL_EPOCHS
    4. Returns updated weights + loss + sample count
    """

    def __init__(
        self,
        company_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_classes: int,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        class_weights: Optional[np.ndarray] = None,
    ):
        self.company_id = company_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.n_samples = len(X_train)

        # Convert class weights to tensor
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        # Create model and trainer
        # NOTE: No class_weights in FL training — it causes gradient
        # instability across non-IID clients. Use FedProx instead.
        self.model = IDSModel(
            input_dim=X_train.shape[1],
            hidden_layers=IDS_HIDDEN_LAYERS,
            num_classes=num_classes,
            dropout=IDS_DROPOUT,
        )
        self.trainer = IDSTrainer(
            model=self.model,
            learning_rate=FL_LEARNING_RATE,
            batch_size=FL_BATCH_SIZE,
            fedprox_mu=0.1,  # Proximal term to prevent client drift
        )

    def get_weights(self) -> list:
        """Get current model weights."""
        return self.model.get_weights()

    def set_weights(self, weights: list):
        """Set model weights (from global server)."""
        self.model.set_weights(weights)

    def fit(
        self, global_weights: Optional[list] = None, epochs: int = FL_LOCAL_EPOCHS
    ) -> Tuple[list, float, int]:
        """
        Perform one FL training round.

        Args:
            global_weights: weights from the server (None for first round)
            epochs: local training epochs

        Returns:
            (updated_weights, avg_loss, n_samples)
        """
        # Apply global weights if provided
        if global_weights is not None:
            self.set_weights(global_weights)

        # Snapshot weights for FedProx reference before local training
        self.trainer.set_global_weights_ref()

        # Train locally
        history = self.trainer.train(
            self.X_train, self.y_train, epochs=epochs, verbose=False
        )

        # Return updated weights + final loss + sample count
        updated_weights = self.get_weights()
        final_loss = history["loss"][-1]
        final_acc = history["accuracy"][-1]

        console.print(
            f"  [cyan]Client {self.company_id}[/cyan] — "
            f"Loss: {final_loss:.4f}, Acc: {final_acc:.4f}, "
            f"Samples: {self.n_samples:,}"
        )

        return updated_weights, final_loss, self.n_samples

    def evaluate(self) -> Optional[Dict]:
        """Evaluate model on local test set."""
        if self.X_test is None or self.y_test is None:
            return None

        results = self.trainer.evaluate(
            self.X_test, self.y_test, class_names=self.class_names
        )
        return results
