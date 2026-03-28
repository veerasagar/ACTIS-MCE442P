"""
ACTIS Zero-Day Detector
========================
Detects unseen/zero-day attacks using a dual-signal approach:

1. **Confidence Signal** (from Tri-LLM's ZDS):
   ZDS_conf = λ·entropy(softmax) + (1-λ)·(1 - max_confidence)

2. **Reconstruction Signal** (autoencoder-based):
   Train an autoencoder ONLY on known-class data.
   Unseen classes → high reconstruction error (anomaly).

3. **Combined Score**:
   ZDS = α·ZDS_conf_norm + (1-α)·ZDS_recon_norm

The reconstruction signal is robust even when the classifier is
confidently wrong, because unseen attack patterns live in different
regions of the feature space that the autoencoder hasn't learned.

Usage:
    detector = ZeroDayDetector(model)
    detector.fit(X_train_known, y_train_known)
    results = detector.detect(X_test)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
from rich.console import Console

console = Console()


class _Autoencoder(nn.Module):
    """Compact autoencoder for reconstruction-based anomaly detection."""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# Protocol column index in standard 41-feature CIC-IDS2018 ordering
_PROTOCOL_COL_IDX = 2   # PROTOCOL column (6=TCP, 17=UDP, 1=ICMP)
_PROTO_TCP  = 6
_PROTO_UDP  = 17
_PROTO_ICMP = 1


def _get_protocol_mask(X: np.ndarray, proto: int) -> np.ndarray:
    """Boolean mask for rows matching a given protocol value."""
    return X[:, _PROTOCOL_COL_IDX] == proto


class ZeroDayDetector:
    """
    Hybrid zero-day attack detector combining classifier confidence
    with per-protocol autoencoder reconstruction error.

    Key insight: DDoS uses UDP/ICMP floods; DoS uses TCP resource exhaustion.
    Training separate autoencoders per protocol cluster means:
      - UDP autoencoder learns normal UDP traffic patterns
      - When DDoS (malicious UDP) arrives, it has higher recon error
      - This separates DDoS from DoS even when raw features overlap

    Signals:
    1. **Confidence**: entropy + (1 - max_softmax) from IDS classifier
    2. **Reconstruction**: min(MSE over per-protocol autoencoders)

    Combined: ZDS = α·confidence + (1-α)·reconstruction
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
            model: trained IDS classifier
            class_names: list of class label names
            lambda_: blend for confidence vs entropy (0=conf, 1=entropy)
            alpha: blend for confidence vs reconstruction (0=recon, 1=conf)
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
        # Per-protocol autoencoders
        self._autoencoders: Dict[str, Optional[nn.Module]] = {
            "tcp": None, "udp": None, "icmp": None, "other": None
        }
        self._recon_scales: Dict[str, float] = {}

    def _train_protocol_autoencoder(
        self, X: np.ndarray, tag: str, epochs: int = 30, lr: float = 1e-3
    ):
        """Train one autoencoder on a protocol-specific subset of known-class data."""
        if len(X) < 32:  # Not enough samples
            return
        input_dim = X.shape[1]
        ae = _Autoencoder(input_dim)
        optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=min(256, len(X)), shuffle=True)
        ae.train()
        for _ in range(epochs):
            for (batch,) in loader:
                recon = ae(batch)
                loss = F.mse_loss(recon, batch)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        ae.eval()
        self._autoencoders[tag] = ae

    def _compute_recon_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample reconstruction error using the matching
        protocol-specific autoencoder.

        Samples for which no protocol AE exists fall back to "other".
        """
        errors = np.zeros(len(X), dtype=np.float32)
        proto_col = X[:, _PROTOCOL_COL_IDX]

        groups = {
            "tcp":  proto_col == _PROTO_TCP,
            "udp":  proto_col == _PROTO_UDP,
            "icmp": proto_col == _PROTO_ICMP,
        }
        # Everything else → other
        other_mask = ~(groups["tcp"] | groups["udp"] | groups["icmp"])
        groups["other"] = other_mask

        for tag, mask in groups.items():
            if not mask.any():
                continue
            ae = self._autoencoders.get(tag) or self._autoencoders.get("other")
            if ae is None:
                continue
            X_sub = torch.tensor(X[mask], dtype=torch.float32)
            with torch.no_grad():
                recon = ae(X_sub)
                errs = ((X_sub - recon) ** 2).mean(dim=1).numpy()
            # Normalize by this protocol's scale
            scale = self._recon_scales.get(tag, self._recon_scales.get("other", 1.0))
            errors[mask] = errs / max(scale, 1e-8)

        return errors

    def _compute_confidence_scores(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute confidence-based ZDS from the IDS classifier."""
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

        max_conf = np.max(probs, axis=1)
        log_probs = np.log(probs + 1e-10)
        entropy = -np.sum(probs * log_probs, axis=1)
        if self.max_entropy is None:
            self.max_entropy = np.log(probs.shape[1])
        norm_entropy = entropy / self.max_entropy

        zds_conf = self.lambda_ * norm_entropy + (1 - self.lambda_) * (1 - max_conf)
        return zds_conf, max_conf, norm_entropy

    def _compute_scores(
        self, X: np.ndarray, return_components: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute hybrid ZDS = α·confidence + (1-α)·per-protocol-reconstruction."""
        zds_conf, max_conf, entropy = self._compute_confidence_scores(X)

        any_ae = any(v is not None for v in self._autoencoders.values())
        if any_ae:
            recon_errors = self._compute_recon_errors(X)  # already normalized per-proto
            zds = self.alpha * zds_conf + (1 - self.alpha) * np.clip(recon_errors, 0, 1)
        else:
            zds = zds_conf

        return zds, max_conf, entropy

    def fit_thresholds(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Train per-protocol autoencoders and calibrate ZDS threshold.

        Returns:
            Calibrated threshold value
        """
        proto_col = X_train[:, _PROTOCOL_COL_IDX]
        groups = {
            "tcp":   proto_col == _PROTO_TCP,
            "udp":   proto_col == _PROTO_UDP,
            "icmp":  proto_col == _PROTO_ICMP,
            "other": ~((proto_col == _PROTO_TCP) | (proto_col == _PROTO_UDP) | (proto_col == _PROTO_ICMP)),
        }

        for tag, mask in groups.items():
            if mask.sum() >= 32:
                self._train_protocol_autoencoder(X_train[mask], tag)
                # Compute scale from training data for this protocol
                ae = self._autoencoders[tag]
                if ae:
                    X_sub = torch.tensor(X_train[mask], dtype=torch.float32)
                    with torch.no_grad():
                        recon = ae(X_sub)
                        errs = ((X_sub - recon) ** 2).mean(dim=1).numpy()
                    self._recon_scales[tag] = float(np.percentile(errs, 99))

        # Compute ZDS on full training data
        zds, _, _ = self._compute_scores(X_train)
        self.threshold = float(np.percentile(zds, self.percentile))

        proto_counts = {t: int(m.sum()) for t, m in groups.items() if m.sum() > 0}
        console.print(
            "  ZDS threshold: {:.4f} (p{}), protos: {}".format(
                self.threshold, self.percentile, proto_counts
            )
        )
        return self.threshold

    def detect(
        self, X: np.ndarray, y_true: Optional[np.ndarray] = None
    ) -> Dict:
        """Detect zero-day attacks in a batch of samples."""
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
