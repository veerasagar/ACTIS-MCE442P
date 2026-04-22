"""Phase 2 — Train & evaluate standalone IDS per node (no federation)."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
from src.data.loader import FedIntelDataLoader
from src.data.preprocessor import Preprocessor
from src.local_node.ids_model import IDSModel
from src.local_node.ids_trainer import IDSTrainer

loader = FedIntelDataLoader()
data = loader.load_and_partition()

# Build unified label set
all_labels = set()
for c in data.values():
    all_labels.update(c["data"][c["label_col"]].unique())
unified = sorted(all_labels)

print()
print("=== Local-Only IDS (No Federation) ===")
print()
print("  Node  Dataset    Accuracy   F1       Precision  Recall")
print("  ----  ---------  --------   ------   ---------  ------")
for cid in sorted(data.keys()):
    p = Preprocessor()
    p.set_unified_labels(unified)
    X_tr, X_te, y_tr, y_te = p.prepare(data[cid], max_samples=15000)
    model = IDSModel(input_dim=41, num_classes=p.num_classes)
    trainer = IDSTrainer(model=model)
    trainer.train(X_tr, y_tr, epochs=10, verbose=False)
    m = trainer.evaluate(X_te, y_te)
    ds = "IDS2018" if cid in ["A", "B"] else "BoT-IoT"
    print(
        f"  {cid}     {ds:9s}  {m['accuracy']*100:6.2f}%    "
        f"{m['f1']:.4f}   {m['precision']:.4f}     {m['recall']:.4f}"
    )
