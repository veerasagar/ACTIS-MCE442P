"""Phase 5 — DDoS zero-day detection benchmark on BoT-IoT dataset."""
import warnings; warnings.filterwarnings('ignore')
import gc, numpy as np, pandas as pd
import sys; sys.path.insert(0, '.')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.data.feature_engineer import FeatureEngineer
from src.local_node.ids_model import IDSModel
from src.local_node.ids_trainer import IDSTrainer
from src.local_node.zeroday_detector import ZeroDayDetector

fe = FeatureEngineer()

print("Loading NF-BoT-IoT-v2...")
df = pd.read_parquet("data/nf-bot-iot-v2/NF-BoT-IoT-V2.parquet")
bot_map = {"DDoS": "ddos", "DoS": "dos", "Reconnaissance": "reconnaissance",
           "Benign": "benign", "Theft": "theft"}
df["unified"] = df["Attack"].map(bot_map).fillna("benign")
unified = sorted(df["unified"].unique())
lmap = {n: i for i, n in enumerate(unified)}

chunks = [
    df[df["unified"] == c].sample(n=min(len(df[df["unified"] == c]), 2000), random_state=42)
    for c in unified
]
df_s = pd.concat(chunks).sample(frac=1, random_state=42)
del df; gc.collect()

feat_cols = [c for c in df_s.columns
             if c not in ["Attack", "Label", "unified"]
             and pd.api.types.is_numeric_dtype(df_s[c])][:41]
df_s[feat_cols] = df_s[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

X = fe.transform(df_s[feat_cols].values.astype(np.float32))
y = np.array([lmap[l] for l in df_s["unified"]])
X = StandardScaler().fit_transform(X).astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

ddos_idx = unified.index("ddos")
known_idx = [i for i in range(len(unified)) if i != ddos_idx]
train_mask = y_tr != ddos_idx
Xt, yt = X_tr[train_mask], y_tr[train_mask]

model = IDSModel(input_dim=X.shape[1], num_classes=len(unified))
t = IDSTrainer(model=model)
t.train(Xt, yt, epochs=15, verbose=False)

print()
print("=== DDoS Zero-Day Detection (BoT-IoT, per-protocol AE) ===")
print(f"  Classes: {unified}")
print(f"  DDoS held out as unknown (class {ddos_idx})")
print()
for alpha, p in [(0.3, 90), (0.5, 90), (0.3, 95)]:
    d = ZeroDayDetector(model, alpha=alpha, percentile=p)
    d.fit_thresholds(Xt, yt)
    r = d.detect_and_report(X_te, y_true=y_te, known_class_idx=known_idx)
    udr = r.get("unknown_detection_rate", 0) * 100
    far = r.get("known_false_alarm_rate", 0) * 100
    uf = r.get("unknown_flagged", 0)
    uc = r.get("unknown_count", 0)
    print(f"  α={alpha} p={p}:  {uf}/{uc} DDoS detected  Rate={udr:.1f}%  FAR={far:.1f}%")
