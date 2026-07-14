# ACTIS — ML Models Used

> Two PyTorch neural networks: an MLP classifier for intrusion detection and per-protocol autoencoders for zero-day detection.

---

## 1. IDSModel — Multi-Layer Perceptron (MLP) Classifier

**File**: `src/local_node/ids_model.py`

This is the **primary intrusion detection model** — a fully-connected feedforward neural network (MLP) that classifies network flows into 9 attack categories.

### Architecture

```
Input(41) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
          → Linear(32)  → BatchNorm → ReLU → Dropout(0.3)
          → Linear(9)   [output logits]
```

| Layer | In → Out | Components |
|-------|----------|------------|
| Block 1 | 41 → 128 | `Linear` + `BatchNorm1d` + `ReLU` + `Dropout(0.3)` |
| Block 2 | 128 → 64 | `Linear` + `BatchNorm1d` + `ReLU` + `Dropout(0.3)` |
| Block 3 | 64 → 32 | `Linear` + `BatchNorm1d` + `ReLU` + `Dropout(0.3)` |
| Output | 32 → 9 | `Linear` (raw logits, no activation) |

### Why MLP?

- **Tabular data** — Network flows are fixed-size numeric feature vectors (41 dims), not images or sequences. MLPs outperform CNNs/RNNs on structured tabular data of this kind.
- **Edge deployment** — Intentionally lightweight (~14K parameters) for real-time inference at each organization's node.
- **FL-compatible** — Simple architecture means weight extraction (`get_weights()`) and aggregation (`set_weights()`) are straightforward, unlike complex architectures with attention layers or residual connections.

### Key Design Choices

- **BatchNorm** — Normalizes activations per-layer, stabilizes training across heterogeneous datasets (enterprise CIC-IDS2018 vs IoT BoT-IoT).
- **Dropout(0.3)** — Regularization to prevent overfitting, especially since some nodes may have limited data.
- **No softmax in forward** — Raw logits are output; `CrossEntropyLoss` applies softmax internally during training. Softmax is only applied in `predict_proba()` for inference.
- **Class weighting** — The trainer uses inverse-frequency class weights since benign traffic vastly outnumbers attack classes.
- **FedProx** — Adds proximal term `μ/2 * ||w - w_global||²` to the loss to handle non-IID data across nodes.

### Methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `forward(x)` | Logits tensor | Standard forward pass |
| `predict(x)` | Class label tensor | `argmax(softmax(logits))` |
| `predict_proba(x)` | Probability tensor | `softmax(logits)` |
| `get_weights()` | List of NumPy arrays | Extract weights for FL aggregation |
| `set_weights(w)` | — | Load aggregated global weights |

---

## 2. _Autoencoder — Per-Protocol Reconstruction Anomaly Detector

**File**: `src/local_node/zeroday_detector.py`

This is the **zero-day detection model** — compact autoencoders trained separately for each network protocol.

### Architecture

```
Encoder: Input(41) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(16)  [latent]
Decoder: Linear(16) → Linear(32) → ReLU → Linear(64) → ReLU → Linear(41) [reconstruction]
```

| Stage | Layers | Dimensions |
|-------|--------|-----------|
| Encoder | 3 Linear + 2 ReLU | 41 → 64 → 32 → **16** (bottleneck) |
| Decoder | 3 Linear + 2 ReLU | 16 → 32 → 64 → **41** (reconstruction) |

### How It Works

**4 separate autoencoders** are trained — one per protocol:

| Autoencoder | Protocol | Trained On |
|-------------|----------|-----------|
| TCP AE | `PROTOCOL == 6` | Known-class TCP flows only |
| UDP AE | `PROTOCOL == 17` | Known-class UDP flows only |
| ICMP AE | `PROTOCOL == 1` | Known-class ICMP flows only |
| Other AE | Everything else | Remaining known-class flows |

**Training**: Each autoencoder learns to **reconstruct only normal/known traffic** for its protocol. The loss is MSE between input and reconstruction.

**Detection**: When an unseen zero-day attack arrives, the autoencoder **can't reconstruct it well** because the attack pattern lives in a different region of feature space. High reconstruction error = anomaly.

### Zero-Day Score (ZDS) — Dual-Signal Fusion

The final score combines **two independent signals**:

```
ZDS = α · ZDS_conf + (1 − α) · ZDS_recon
```

Where:

```
ZDS_conf = λ · H(softmax) / log(K) + (1 − λ) · (1 − max(softmax))
```

- **α = 0.3** — Weights reconstruction 70%, confidence 30%
- **λ = 0.5** — Equal blend of entropy and confidence gap
- **Threshold** — Auto-calibrated at the 95th percentile of training ZDS scores

### Why Per-Protocol Autoencoders?

- **DDoS uses UDP/ICMP floods** while **DoS uses TCP resource exhaustion** — a single autoencoder would blur these protocol-specific patterns.
- **Reconstruction error is unsupervised** — No labeled zero-day data needed; just normal traffic.
- **Robust to confident misclassification** — Even if the MLP is confidently wrong, the autoencoder catches it via reconstruction error.

---

## Summary

| Model | Type | Purpose | Architecture | Parameters |
|-------|------|---------|-------------|-----------|
| **IDSModel** | MLP Classifier | 9-class attack detection | 41→128→64→32→9 | ~14K |
| **_Autoencoder** (×4) | Autoencoder | Zero-day anomaly detection | 41→64→32→16→32→64→41 | ~7K each |

Both models are intentionally compact for edge deployment and federated weight transfer.
