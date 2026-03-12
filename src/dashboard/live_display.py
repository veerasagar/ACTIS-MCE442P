"""
Fed-Intel Rich Live Dashboard
===============================
Terminal dashboard showing real-time Fed-Intel simulation status:
  - FL training progress (rounds, loss, accuracy per company)
  - Privacy engine stats (PII leakage, share rate)
  - Global KB stats (threats, defenses, MITRE)
  - Trust scores and aggregation weights
  - Immunity rules generated

Usage:
    dashboard = FedIntelDashboard()
    dashboard.update_fl(round_data)
    dashboard.update_privacy(node_stats)
    dashboard.render()
"""

from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.layout import Layout

console = Console()


class FedIntelDashboard:
    """
    Rich terminal dashboard for Fed-Intel simulation.
    Collects metrics from all components and renders a summary.
    """

    def __init__(self):
        self.fl_rounds: List[Dict] = []
        self.privacy_stats: Dict[str, Dict] = {}
        self.kb_stats: Dict = {}
        self.immunity_rules: List[Dict] = []
        self.final_eval: Dict[str, Dict] = {}

    # ─── Update Methods ─────────────────────────────────────────────────

    def update_fl(self, round_data: Dict):
        """Update with FL round data."""
        self.fl_rounds.append(round_data)

    def update_privacy(self, company_id: str, stats: Dict):
        """Update with privacy node stats."""
        self.privacy_stats[company_id] = stats

    def update_kb(self, stats: Dict):
        """Update with KB stats."""
        self.kb_stats = stats

    def update_immunity(self, rules: List[Dict]):
        """Update with generated immunity rules."""
        self.immunity_rules = rules

    def update_final_eval(self, eval_data: Dict):
        """Update with final evaluation data."""
        self.final_eval = eval_data

    # ─── Render Methods ─────────────────────────────────────────────────

    def render(self):
        """Render the full dashboard."""
        console.print()
        console.print(
            Panel(
                "[bold white]Fed-Intel — Federated Agentic Threat Intelligence[/bold white]",
                style="bold cyan",
            )
        )

        self._render_fl_progress()
        self._render_privacy_stats()
        self._render_kb_stats()
        self._render_immunity_rules()
        self._render_final_eval()

    def _render_fl_progress(self):
        """Render FL training progress."""
        if not self.fl_rounds:
            return

        table = Table(
            title="📊 Federated Learning Convergence",
            show_header=True,
            title_style="bold",
        )
        table.add_column("Round", style="cyan", justify="center")
        table.add_column("Avg Loss", justify="center")
        table.add_column("Company A", justify="center")
        table.add_column("Company B", justify="center")
        table.add_column("Company C", justify="center")

        for r in self.fl_rounds:
            ev = r.get("eval", {})
            loss = r.get("avg_loss", 0)
            loss_color = "green" if loss < 0.2 else "yellow" if loss < 0.5 else "red"

            table.add_row(
                str(r.get("round", "?")),
                "[{}]{:.4f}[/{}]".format(loss_color, loss, loss_color),
                self._acc_cell(ev.get("A", {}).get("accuracy", 0)),
                self._acc_cell(ev.get("B", {}).get("accuracy", 0)),
                self._acc_cell(ev.get("C", {}).get("accuracy", 0)),
            )

        console.print(table)

    def _render_privacy_stats(self):
        """Render privacy engine stats."""
        if not self.privacy_stats:
            return

        table = Table(
            title="🔒 Privacy Engine Stats",
            show_header=True,
            title_style="bold",
        )
        table.add_column("Company", style="cyan")
        table.add_column("Processed", justify="center")
        table.add_column("Shared", justify="center")
        table.add_column("Blocked", justify="center")
        table.add_column("PII Leak %", justify="center")
        table.add_column("DP ε", justify="center")

        for cid in sorted(self.privacy_stats.keys()):
            s = self.privacy_stats[cid]
            leak = s.get("pii_leakage_rate", 0)
            leak_color = "green" if leak == 0 else "red"

            table.add_row(
                "Company {}".format(cid),
                str(s.get("processed", 0)),
                str(s.get("shared", 0)),
                str(s.get("blocked", 0)),
                "[{}]{:.1f}%[/{}]".format(leak_color, leak, leak_color),
                str(s.get("dp_params", {}).get("epsilon", "?")),
            )

        console.print(table)

    def _render_kb_stats(self):
        """Render global KB stats."""
        if not self.kb_stats:
            return

        table = Table(
            title="🧠 Global Knowledge Base",
            show_header=True,
            title_style="bold",
        )
        table.add_column("Collection", style="cyan")
        table.add_column("Count", justify="center")

        table.add_row("Threat Summaries", str(self.kb_stats.get("threats", 0)))
        table.add_row("Defense Patterns", str(self.kb_stats.get("defenses", 0)))
        table.add_row("MITRE References", str(self.kb_stats.get("mitre_refs", 0)))

        console.print(table)

    def _render_immunity_rules(self):
        """Render generated immunity rules."""
        if not self.immunity_rules:
            return

        table = Table(
            title="🛡️ Immunity Rules Generated",
            show_header=True,
            title_style="bold",
        )
        table.add_column("Attack", style="cyan")
        table.add_column("Priority", justify="center")
        table.add_column("Description")
        table.add_column("Intel Sources", justify="center")

        for r in self.immunity_rules:
            prio = r.get("priority", "?")
            prio_color = "red" if prio == "CRITICAL" else "yellow" if prio == "HIGH" else "green"
            table.add_row(
                r.get("attack_type", "?"),
                "[{}]{}[/{}]".format(prio_color, prio, prio_color),
                r.get("description", "")[:60],
                str(r.get("based_on_threats", 0)),
            )

        console.print(table)

    def _render_final_eval(self):
        """Render final evaluation results."""
        if not self.final_eval:
            return

        table = Table(
            title="🏆 Final Model Performance (Post-FL)",
            show_header=True,
            title_style="bold",
        )
        table.add_column("Company", style="cyan")
        table.add_column("Accuracy", justify="center")
        table.add_column("Precision", justify="center")
        table.add_column("Recall", justify="center")
        table.add_column("F1", justify="center")

        for cid in sorted(self.final_eval.keys()):
            m = self.final_eval[cid]
            table.add_row(
                "Company {}".format(cid),
                self._acc_cell(m.get("accuracy", 0)),
                "{:.4f}".format(m.get("precision", 0)),
                "{:.4f}".format(m.get("recall", 0)),
                "{:.4f}".format(m.get("f1", 0)),
            )

        console.print(table)

    @staticmethod
    def _acc_cell(acc: float) -> str:
        """Format accuracy with color."""
        if acc >= 0.95:
            return "[bold green]{:.1f}%[/bold green]".format(acc * 100)
        elif acc >= 0.80:
            return "[green]{:.1f}%[/green]".format(acc * 100)
        elif acc >= 0.60:
            return "[yellow]{:.1f}%[/yellow]".format(acc * 100)
        else:
            return "[red]{:.1f}%[/red]".format(acc * 100)
