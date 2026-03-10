"""
Fed-Intel IDS Model
====================
PyTorch MLP for network intrusion detection.
Designed for federated learning — weights can be extracted and aggregated.

Architecture: Input(41) → 128 → 64 → 32 → num_classes
With BatchNorm, Dropout, and ReLU activations.

Usage:
    from src.local_node.ids_model import IDSModel
    model = IDSModel(input_dim=41, num_classes=7)
"""

import torch
import torch.nn as nn
from typing import List


class IDSModel(nn.Module):
    """
    Multi-layer perceptron for network intrusion detection.
    
    Designed for FL: get_weights() / set_weights() enable
    weight extraction and aggregation across federated nodes.
    """

    def __init__(
        self,
        input_dim: int = 41,
        hidden_layers: List[int] = None,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Build layers dynamically
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits (no softmax)."""
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    # ─── FL Weight Management ───────────────────────────────────────────

    def get_weights(self) -> list:
        """Extract model weights as a list of numpy arrays (for FL)."""
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_weights(self, weights: list):
        """Set model weights from a list of numpy arrays (from FL server)."""
        for param, w in zip(self.parameters(), weights):
            param.data = torch.tensor(w, dtype=param.dtype, device=param.device)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Standalone Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = IDSModel(input_dim=41, num_classes=7)
    print(f"Model: {model.count_parameters():,} parameters")
    print(model)

    # Test forward pass
    x = torch.randn(32, 41)
    logits = model(x)
    print(f"\nInput: {x.shape} → Output: {logits.shape}")
    
    preds = model.predict(x)
    print(f"Predictions: {preds.shape}, unique: {preds.unique().tolist()}")

    # Test FL weight extraction
    weights = model.get_weights()
    print(f"\nFL weights: {len(weights)} tensors")
    for i, w in enumerate(weights):
        print(f"  [{i}] {w.shape}")
