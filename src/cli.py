"""
Fed-Intel CLI
==============
Typer-based command-line interface for the Fed-Intel system.

Usage:
    python -m src.cli data download
    python -m src.cli fl train --rounds 10
    python -m src.cli simulate --live
    python -m src.cli query "DDoS attacks"
"""

import typer
from rich.console import Console

console = Console()
app = typer.Typer(
    name="fedintel",
    help="Fed-Intel — Federated Agentic Threat Intelligence",
    add_completion=False,
)

# ─── Subcommand Groups ──────────────────────────────────────────────────────

data_app = typer.Typer(help="Data management commands")
fl_app = typer.Typer(help="Federated learning commands")
eval_app = typer.Typer(help="Evaluation commands")
app.add_typer(data_app, name="data")
app.add_typer(fl_app, name="fl")
app.add_typer(eval_app, name="eval")


# ─── Data Commands ──────────────────────────────────────────────────────────

@data_app.command("download")
def data_download():
    """Download datasets (NF-CSE-CIC-IDS2018-v2 + NF-BoT-IoT-v2)."""
    console.print("[bold]Downloading datasets...[/bold]")
    import subprocess
    subprocess.run(["bash", "scripts/download_dataset.sh"], check=True)
    console.print("[green]✓ Datasets downloaded[/green]")


@data_app.command("preprocess")
def data_preprocess(
    max_samples: int = typer.Option(20000, help="Max samples per company")
):
    """Preprocess and partition datasets."""
    import warnings; warnings.filterwarnings("ignore")
    from src.data.loader import FedIntelDataLoader
    from src.data.preprocessor import Preprocessor

    loader = FedIntelDataLoader()
    data = loader.load_and_partition()

    for cid in ["A", "B", "C"]:
        p = Preprocessor()
        X_train, X_test, y_train, y_test = p.prepare(data[cid], max_samples=max_samples)
        console.print(
            "  Company {}: train={}, test={}, classes={}".format(
                cid, len(X_train), len(X_test), p.num_classes
            )
        )
    console.print("[green]✓ Preprocessing complete[/green]")


@data_app.command("summarize")
def data_summarize(
    preview: int = typer.Option(5, help="Number of summaries to preview"),
):
    """Generate threat summaries from flagged flows."""
    import warnings; warnings.filterwarnings("ignore")
    from src.data.loader import FedIntelDataLoader
    from src.data.summarizer import ThreatSummarizer

    loader = FedIntelDataLoader()
    data = loader.load_and_partition()
    summarizer = ThreatSummarizer()

    for cid in ["A", "C"]:
        df = data[cid]["data"]
        label_col = data[cid]["label_col"]
        attacks = df[df[label_col] != "Benign"].head(preview)

        console.print("\n[bold cyan]Company {} — {} attack samples:[/bold cyan]".format(cid, len(attacks)))
        for _, row in attacks.iterrows():
            label = row[label_col].lower().replace(" ", "_")
            s = summarizer.summarize_flow(row.to_dict(), label, cid)
            console.print("  [yellow]{}[/yellow]: {}".format(label, s.get("summary", "")[:80]))


# ─── FL Commands ────────────────────────────────────────────────────────────

@fl_app.command("train")
def fl_train(
    rounds: int = typer.Option(10, help="Number of FL rounds"),
    samples: int = typer.Option(20000, help="Max samples per company"),
    evaluate: bool = typer.Option(True, help="Show per-round evaluation"),
):
    """Run federated learning training."""
    import warnings; warnings.filterwarnings("ignore")
    from src.server.fl_simulation import FLSimulation

    sim = FLSimulation(max_samples=samples)
    results = sim.run(num_rounds=rounds)

    if evaluate:
        console.print("\n[bold]Final Results:[/bold]")
        for cid, r in results["final_eval"].items():
            console.print(
                "  Company {}: Acc={:.4f}, F1={:.4f}".format(
                    cid, r["accuracy"], r["f1"]
                )
            )


@fl_app.command("status")
def fl_status():
    """Show current FL model status."""
    import os
    from src.config import MODELS_DIR
    model_path = MODELS_DIR / "global_model.pt"
    if model_path.exists():
        size = os.path.getsize(model_path) / 1024
        console.print("  Global model: {} ({:.1f} KB)".format(model_path, size))
    else:
        console.print("  [yellow]No trained model found. Run: fedintel fl train[/yellow]")


# ─── Simulate Command ──────────────────────────────────────────────────────

@app.command("simulate")
def simulate(
    rounds: int = typer.Option(5, help="FL rounds"),
    samples: int = typer.Option(10000, help="Max samples per company"),
    threats: int = typer.Option(20, help="Number of threats to process through privacy pipeline"),
    live: bool = typer.Option(False, help="Show live dashboard"),
):
    """Run full end-to-end Fed-Intel simulation."""
    import warnings; warnings.filterwarnings("ignore")
    _run_simulation(rounds, samples, threats)


def _run_simulation(rounds: int, samples: int, threats_count: int):
    """Core simulation logic."""
    import time
    import shutil
    import numpy as np
    from src.server.fl_simulation import FLSimulation
    from src.local_node.node import PrivacyNode
    from src.server.global_kb import GlobalKnowledgeBase
    from src.server.aggregator import Aggregator
    from src.server.rag_engine import RAGEngine
    from src.server.immunity import ImmunityEngine
    from src.dashboard.live_display import FedIntelDashboard
    from src.data.loader import FedIntelDataLoader
    from src.data.mitre_mapper import MITREMapper

    start = time.time()
    dashboard = FedIntelDashboard()

    console.print("\n[bold magenta]" + "═" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  Fed-Intel — Full End-to-End Simulation[/bold magenta]")
    console.print("[bold magenta]" + "═" * 60 + "[/bold magenta]\n")

    # ── Stage 1: Federated Learning ──────────────────────────────────
    console.print("[bold]STAGE 1: Federated Learning ({} rounds)[/bold]\n".format(rounds))
    sim = FLSimulation(max_samples=samples)
    fl_results = sim.run(num_rounds=rounds)

    for r in fl_results["round_history"]:
        dashboard.update_fl(r)
    dashboard.update_final_eval(fl_results["final_eval"])

    # ── Stage 2: Privacy Engine ──────────────────────────────────────
    console.print("\n[bold]STAGE 2: Agentic Privacy Engine[/bold]\n")

    loader = FedIntelDataLoader()
    company_data = loader.load_and_partition()
    mapper = MITREMapper()

    privacy_nodes = {}
    all_sanitized = []

    for cid in ["A", "B", "C"]:
        node = PrivacyNode(company_id=cid)
        privacy_nodes[cid] = node

        df = company_data[cid]["data"]
        label_col = company_data[cid]["label_col"]
        attacks = df[df[label_col] != "Benign"].sample(
            n=min(threats_count, len(df[df[label_col] != "Benign"])),
            random_state=42,
        )

        flows = [row.to_dict() for _, row in attacks.iterrows()]
        labels = [row[label_col].lower().replace(" ", "_") for _, row in attacks.iterrows()]

        results = node.process_batch(flows, labels)
        all_sanitized.extend(results)
        dashboard.update_privacy(cid, node.get_stats())

        console.print(
            "  [cyan]Node {}[/cyan]: {}/{} flows shared, PII leak: {:.1f}%".format(
                cid, node.shared, node.processed,
                node.validator.get_stats()["leakage_rate"],
            )
        )

    # ── Stage 3: Global Federated RAG ────────────────────────────────
    console.print("\n[bold]STAGE 3: Global Federated RAG[/bold]\n")

    kb = GlobalKnowledgeBase()
    kb.populate_mitre()
    agg = Aggregator(kb)

    # Ingest all sanitized summaries
    agg.ingest_batch(all_sanitized)
    console.print(
        "  Ingested {} threats from {} companies".format(
            agg.stats["ingested"],
            len(agg.stats["companies_seen"]),
        )
    )

    dashboard.update_kb(kb.get_stats())

    # ── Stage 4: Immunity ────────────────────────────────────────────
    console.print("\n[bold]STAGE 4: Immunity Engine[/bold]\n")

    rag = RAGEngine(kb)
    immunity = ImmunityEngine(rag)

    # Each company gets rules based on OTHER companies' threats
    all_rules = []
    for cid in ["A", "B", "C"]:
        rules = immunity.generate_all_rules(cid)
        all_rules.extend(rules)
        console.print(
            "  [cyan]Company {}[/cyan]: {} rules generated".format(
                cid, len(rules)
            )
        )

    dashboard.update_immunity(all_rules[:10])

    # ── Final Dashboard ──────────────────────────────────────────────
    elapsed = time.time() - start
    console.print("\n[bold magenta]" + "═" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  Simulation Complete — {:.1f}s[/bold magenta]".format(elapsed))
    console.print("[bold magenta]" + "═" * 60 + "[/bold magenta]\n")

    dashboard.render()


# ─── Query Command ──────────────────────────────────────────────────────────

@app.command("query")
def query_rag(
    query_text: str = typer.Argument(..., help="Query for threat intelligence"),
    top_k: int = typer.Option(5, help="Number of results"),
):
    """Query the global threat intelligence knowledge base."""
    import warnings; warnings.filterwarnings("ignore")
    from src.server.global_kb import GlobalKnowledgeBase
    from src.server.rag_engine import RAGEngine

    kb = GlobalKnowledgeBase()
    rag = RAGEngine(kb)
    results = rag.query(query_text, top_k=top_k)
    context = rag.build_context(results)

    console.print("\n[bold]Query:[/bold] {}\n".format(query_text))
    console.print(context)
    console.print("\n[dim]{} threats, {} defenses, {} MITRE refs[/dim]".format(
        len(results["threats"]), len(results["defenses"]), len(results["mitre"])
    ))


# ─── Eval Commands ──────────────────────────────────────────────────────────

@eval_app.command("detection")
def eval_detection(
    mode: str = typer.Option("federated", help="local or federated"),
    samples: int = typer.Option(20000, help="Max samples per company"),
):
    """Evaluate IDS detection accuracy."""
    import warnings; warnings.filterwarnings("ignore")

    if mode == "federated":
        from src.server.fl_simulation import FLSimulation
        sim = FLSimulation(max_samples=samples)
        results = sim.run(num_rounds=10)
        console.print("\n[bold]Federated IDS Accuracy:[/bold]")
        for cid, r in results["final_eval"].items():
            console.print(
                "  Company {}: Acc={:.4f}, F1={:.4f}".format(
                    cid, r["accuracy"], r["f1"]
                )
            )
    else:
        console.print("[yellow]Local evaluation — run: fedintel fl train --rounds 0[/yellow]")


@eval_app.command("pii")
def eval_pii(
    samples: int = typer.Option(500, help="Number of samples to test"),
):
    """Test PII leakage rate."""
    import warnings; warnings.filterwarnings("ignore")
    from src.local_node.node import PrivacyNode
    from src.data.loader import FedIntelDataLoader

    loader = FedIntelDataLoader()
    data = loader.load_and_partition()

    for cid in ["A", "C"]:
        node = PrivacyNode(company_id=cid)
        df = data[cid]["data"]
        label_col = data[cid]["label_col"]
        attacks = df[df[label_col] != "Benign"].sample(
            n=min(samples, len(df[df[label_col] != "Benign"])), random_state=42,
        )
        flows = [row.to_dict() for _, row in attacks.iterrows()]
        labels = [row[label_col].lower().replace(" ", "_") for _, row in attacks.iterrows()]
        node.process_batch(flows, labels)
        node.print_summary()


@eval_app.command("all")
def eval_all():
    """Run all evaluation tests."""
    eval_detection(mode="federated", samples=20000)
    eval_pii(samples=100)


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
