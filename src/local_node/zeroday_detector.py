"""
ACTIS Zero-Day Detector
========================
Detects unseen/zero-day attacks using a hybrid scoring approach:

1. **Confidence Signal** (from Tri-LLM's ZDS):
   ZDS_conf = λ·entropy(softmax) + (1-λ)·(1 - max_confidence)

2. **Feature Distance Signal** (centroid-based):
   ZDS_dist = min distance from sample's hidden representation
              to any known-class centroid

3. **Combined Score**:
   ZDS = α·ZDS_conf_norm + (1-α)·ZDS_dist_norm

Samples with ZDS > threshold are flagged as potential zero-day.

Usage:
    detector = ZeroDayDetector(model, unified_labels)
    detector.fit_thresholds(X_train, y_train)
    results = detector.detect(X_test)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from rich.console import Console

console = Console()


class ZeroDayDetector:
    """
    Hybrid zero-day attack detector combining confidence + feature distance.

    Works alongside the IDS classifier:
    - IDS produces class predictions
    - ZeroDayDetector flags samples where the model is uncertain OR
      where the sample is far from known class centroids in feature space

    Three signals:
    1. **Max confidence**: max(softmax(logits)) — low = uncertain
    2. **Prediction entropy**: -Σ p·log(p) — high = uncertain
    3. **Centroid distance**: distance to nearest class centroid in penultimate layer
    """

    def __init__(
        self,
        model: "IDSModel",
        class_names: Optional[List[str]] = None,
        lambda_: float = 0.5,
        alpha: float = 0.3,
        threshold: Optional[float] = None,
        percentile: float = 95.0,
    ):
        """
        Args:
            model: trained IDS model
            class_names: list of class label names
            lambda_: blend for confidence vs entropy (0=confidence, 1=entropy)
            alpha: blend for confidence-signal vs distance-signal (0=all distance, 1=all confidence)
            threshold: fixed ZDS threshold (if None, auto-calibrated)
            percentile: percentile of training ZDS for threshold
        """
        self.model = model
        self.class_names = class_names or []
        self.lambda_ = lambda_
        self.alpha = alpha
        self.threshold = threshold
        self.percentile = percentile
        self.max_entropy = None
        self.centroids = None  # Per-class centroids in hidden space
        self.centroid_scale = None  # Normalization for distance scores

    def _get_hidden_features(self, X: np.ndarray) -> np.ndarray:
        """Extract penultimate layer features (before output layer)."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            # Forward through all layers except the last
            hidden = X_t
            layers = list(self.model.network)
            for layer in layers[:-1]:  # Skip final Linear layer
                hidden = layer(hidden)
            return hidden.numpy()

    def _compute_centroids(self, X: np.ndarray, y: np.ndarray):
        """Compute per-class centroids from hidden features."""
        hidden = self._get_hidden_features(X)
        classes = np.unique(y)
        self.centroids = {}
        for c in classes:
            mask = y == c
            if mask.sum() > 0:
                self.centroids[c] = hidden[mask].mean(axis=0)

    def _compute_distance_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute min distance from each sample to any class centroid."""
        hidden = self._get_hidden_features(X)
        distances = np.full(len(X), float('inf'))
        for c, centroid in self.centroids.items():
            d = np.linalg.norm(hidden - centroid, axis=1)
            distances = np.minimum(distances, d)
        return distances

    def _compute_scores(
        self, X: np.ndarray, return_components: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute hybrid zero-day scores.

        Returns:
            (zds_scores, max_confidences, entropies)
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            batch_size = 1024
            all_probs = []
            for i in range(0, len(X_t), batch_size):
                logits = self.model(X_t[i:i + batch_size])
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.numpy())
            probs = np.concatenate(all_probs, axis=0)

        # Confidence signal
        max_conf = np.max(probs, axis=1)
        log_probs = np.log(probs + 1e-10)
        entropy = -np.sum(probs * log_probs, axis=1)
        if self.max_entropy is None:
            self.max_entropy = np.log(probs.shape[1])
        norm_entropy = entropy / self.max_entropy

        zds_conf = self.lambda_ * norm_entropy + (1 - self.lambda_) * (1 - max_conf)

        # Distance signal (if centroids available)
        if self.centroids:
            dist_scores = self._compute_distance_scores(X)
            # Normalize to [0, 1] using scale from training
            if self.centroid_scale is not None:
                norm_dist = dist_scores / self.centroid_scale
                norm_dist = np.clip(norm_dist, 0, 1)
            else:
                norm_dist = dist_scores / (np.max(dist_scores) + 1e-10)

            # Hybrid: α · confidence_signal + (1-α) · distance_signal
            zds = self.alpha * zds_conf + (1 - self.alpha) * norm_dist
        else:
            zds = zds_conf

        return zds, max_conf, norm_entropy

    def fit_thresholds(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Calibrate threshold and compute class centroids from training data.

        Returns:
            Calibrated threshold value
        """
        # Compute centroids
        self._compute_centroids(X_train, y_train)

        # Compute distance scale from training data
        dist_train = self._compute_distance_scores(X_train)
        self.centroid_scale = float(np.percentile(dist_train, 99))

        # Compute ZDS on training data
        zds, max_conf, entropy = self._compute_scores(X_train)
        self.threshold = float(np.percentile(zds, self.percentile))

        console.print(
            "  ZDS threshold: {:.4f} (p{}), centroid_scale: {:.4f}".format(
                self.threshold, self.percentile, self.centroid_scale
            )
        )

        return self.threshold

    def detect(
        self, X: np.ndarray, y_true: Optional[np.ndarray] = None
    ) -> Dict:
        """Detect zero-day attacks."""
        if self.threshold is None:
            raise ValueError("Call fit_thresholds() first")

        zds, max_conf, entropy = self._compute_scores(X)
        is_zeroday = zds > self.threshold

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            preds = self.model(X_t).argmax(dim=1).numpy()

        result = {
            "zds_scores": zds,
            "max_confidence": max_conf,
            "entropy": entropy,
            "is_zeroday": is_zeroday,
            "predicted_class": preds,
            "threshold": self.threshold,
            "n_flagged": int(is_zeroday.sum()),
            "n_total": len(X),
            "flag_rate": float(is_zeroday.sum()) / len(X),
        }

        if y_true is not None:
            result["y_true"] = y_true
            result["detection_rate"] = float(is_zeroday.sum()) / max(len(X), 1)

        return result

    def detect_and_report(
        self,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        known_class_idx: Optional[List[int]] = None,
    ) -> Dict:
        """Full detection with per-class breakdown."""
        result = self.detect(X, y_true)

        if y_true is not None and known_class_idx is not None:
            unknown_mask = ~np.isin(y_true, known_class_idx)
            known_mask = np.isin(y_true, known_class_idx)

            if unknown_mask.sum() > 0:
                result["unknown_detection_rate"] = float(
                    result["is_zeroday"][unknown_mask].sum()
                ) / unknown_mask.sum()
                result["unknown_count"] = int(unknown_mask.sum())
                result["unknown_flagged"] = int(result["is_zeroday"][unknown_mask].sum())
            else:
                result["unknown_detection_rate"] = 0.0

            if known_mask.sum() > 0:
                result["known_false_alarm_rate"] = float(
                    result["is_zeroday"][known_mask].sum()
                ) / known_mask.sum()
                result["known_count"] = int(known_mask.sum())
            else:
                result["known_false_alarm_rate"] = 0.0

        return result
