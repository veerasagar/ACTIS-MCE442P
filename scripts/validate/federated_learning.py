"""Phase 3 — Run 4-node federated learning and report results."""
import warnings; warnings.filterwarnings('ignore')
import argparse
import sys; sys.path.insert(0, '.')
from src.server.fl_simulation import FLSimulation

parser = argparse.ArgumentParser(description="Run federated learning simulation")
parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
parser.add_argument("--samples", type=int, default=15000, help="Max samples per node")
args = parser.parse_args()

sim = FLSimulation(max_samples=args.samples)
results = sim.run(num_rounds=args.rounds)

print()
print(f"=== Federated Learning Results (4 Nodes, {args.rounds} Rounds) ===")
print()
print("  Node  Dataset    Accuracy   F1       Precision  Recall")
print("  ----  ---------  --------   ------   ---------  ------")
for cid in sorted(results["final_eval"].keys()):
    r = results["final_eval"][cid]
    ds = "IDS2018" if cid in ["A", "B"] else "BoT-IoT"
    print(
        f"  {cid}     {ds:9s}  {r['accuracy']*100:6.2f}%    "
        f"{r['f1']:.4f}   {r['precision']:.4f}     {r['recall']:.4f}"
    )
first_loss = results["round_history"][0]["avg_loss"]
last_loss = results["round_history"][-1]["avg_loss"]
print()
print(f"  Loss: {first_loss:.4f} → {last_loss:.4f}  (Δ = {first_loss - last_loss:.4f})")
