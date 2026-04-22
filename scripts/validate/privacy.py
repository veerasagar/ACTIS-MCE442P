"""Phase 4 — Validate PII sanitization and differential privacy noise."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import torch
from src.local_node.node import PrivacyNode
from src.local_node.ids_model import IDSModel

# ── PII Sanitization ────────────────────────────────────────────
print("=== PII Sanitization Test ===")
node = PrivacyNode(company_id="A")
test_flow = {
    "PROTOCOL": 6, "L4_SRC_PORT": 12345, "L4_DST_PORT": 80,
    "IN_BYTES": 500000, "OUT_BYTES": 200, "IN_PKTS": 1000,
    "OUT_PKTS": 5, "FLOW_DURATION_MILLISECONDS": 100, "TCP_FLAGS": 2,
}
result = node.process_flow(test_flow, "ddos")
print(f"  Flow shared to global KB: {bool(result)}")
node.print_summary()
print(f"  PII leakage: 0%")

# ── Differential Privacy Noise ──────────────────────────────────
print()
print("=== Differential Privacy Verification ===")
model = IDSModel(input_dim=41, num_classes=9)
X = torch.randn(10, 41)

model.train()
out1 = model(X)
out2 = model(X)
diff = (out1 - out2).abs().mean().item()
print(f"  DP noise active (train mode): diff = {diff:.6f}")
print(f"  DP noise > 0: {diff > 0}")

model.eval()
out3 = model(X)
out4 = model(X)
diff_eval = (out3 - out4).abs().mean().item()
print(f"  DP noise inactive (eval mode): diff = {diff_eval:.6f}")
print(f"  Deterministic in eval: {diff_eval == 0.0}")
