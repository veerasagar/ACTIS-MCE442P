"""
Fed-Intel Differential Privacy Layer
======================================
Adds calibrated Gaussian noise to embeddings before sharing,
providing a formal (ε, δ)-differential privacy guarantee.

Noise is added to the embedding vectors of threat summaries
before they are sent to the global ChromaDB, ensuring that
individual flow records cannot be reconstructed.

Usage:
    dp = DPLayer(epsilon=1.0)
    noisy_embedding = dp.add_noise(embedding)
"""

import numpy as np
from typing import Optional
from rich.console import Console

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DP_EPSILON, DP_NOISE_SIGMA

console = Console()


class DPLayer:
    """
    Differential privacy via Gaussian noise on embeddings.

    For (ε, δ)-DP with Gaussian mechanism:
        σ = Δf * √(2 ln(1.25/δ)) / ε

    Where Δf is the L2 sensitivity of the embedding function.
    For normalized embeddings (||e|| ≤ 1), Δf = 2.
    """

    def __init__(
        self,
        epsilon: float = DP_EPSILON,
        delta: float = 1e-5,
        sensitivity: float = 2.0,
        sigma_override: Optional[float] = DP_NOISE_SIGMA,
    ):
        """
        Args:
            epsilon: privacy budget (lower = more private, more noise)
            delta: failure probability
            sensitivity: L2 sensitivity of the embedding function
            sigma_override: if set, uses this sigma directly instead of computing
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

        if sigma_override is not None:
            self.sigma = sigma_override
        else:
            # Gaussian mechanism: σ = Δf * √(2 ln(1.25/δ)) / ε
            self.sigma = (
                sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
            )

        self.stats = {"embeddings_processed": 0, "total_noise_norm": 0.0}

    def add_noise(self, embedding: np.ndarray) -> np.ndarray:
        """
        Add calibrated Gaussian noise to an embedding vector.

        Args:
            embedding: numpy array of shape (d,) or (n, d)

        Returns:
            Noisy embedding with same shape
        """
        noise = np.random.normal(0, self.sigma, size=embedding.shape)

        self.stats["embeddings_processed"] += 1
        self.stats["total_noise_norm"] += float(np.linalg.norm(noise))

        noisy = embedding + noise.astype(embedding.dtype)
        return noisy

    def add_noise_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Add noise to a batch of embeddings (n, d)."""
        noise = np.random.normal(0, self.sigma, size=embeddings.shape)
        self.stats["embeddings_processed"] += len(embeddings)
        self.stats["total_noise_norm"] += float(np.linalg.norm(noise))
        return (embeddings + noise).astype(embeddings.dtype)

    def get_privacy_params(self) -> dict:
        """Return the privacy parameters."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sigma": round(self.sigma, 6),
            "sensitivity": self.sensitivity,
            "guarantee": "(ε={}, δ={})-DP".format(self.epsilon, self.delta),
        }

    def get_stats(self) -> dict:
        """Return processing statistics."""
        n = max(self.stats["embeddings_processed"], 1)
        return {
            "embeddings_processed": self.stats["embeddings_processed"],
            "avg_noise_norm": round(self.stats["total_noise_norm"] / n, 4),
            "sigma": round(self.sigma, 6),
        }

    def measure_utility_impact(
        self, original: np.ndarray, noisy: np.ndarray
    ) -> dict:
        """
        Measure how much the noise affects embedding similarity.

        Args:
            original: clean embedding (d,)
            noisy: noisy embedding (d,)

        Returns:
            Dict with cosine similarity and L2 distance
        """
        cos_sim = float(
            np.dot(original, noisy)
            / (np.linalg.norm(original) * np.linalg.norm(noisy) + 1e-8)
        )
        l2_dist = float(np.linalg.norm(original - noisy))
        return {
            "cosine_similarity": round(cos_sim, 4),
            "l2_distance": round(l2_dist, 4),
            "utility_preserved": cos_sim > 0.9,
        }
