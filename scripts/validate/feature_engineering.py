"""Phase 1.2 — Verify feature engineering (41 → 66 features)."""
import numpy as np
import sys; sys.path.insert(0, '.')
from src.data.feature_engineer import FeatureEngineer

fe = FeatureEngineer()
X = np.random.rand(100, 41).astype("float32")
X_enh = fe.transform(X)

print(f"Input shape:  {X.shape}")
print(f"Output shape: {X_enh.shape}  (41 raw + {fe.N_DERIVED} derived)")
print()
print("Derived features:")
for i, name in enumerate(fe.feature_names(), 1):
    print(f"  {i:2d}. {name}")
