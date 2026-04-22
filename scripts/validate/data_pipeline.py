"""Phase 1.1 — Verify data loading across all 4 federated nodes."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
from src.data.loader import FedIntelDataLoader

loader = FedIntelDataLoader()
data = loader.load_and_partition()

print()
print("Node  Dataset                  Rows         Attack Classes")
print("----  -----------------------  -----------  ---------------")
for cid in sorted(data.keys()):
    d = data[cid]
    n = len(d["data"])
    n_atk = d["data"][d["label_col"]].nunique() - 1
    print(f"  {cid}    {d['dataset']:23s}  {n:>11,}  {n_atk}")
