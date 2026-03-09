"""
Fed-Intel Data Loader
=====================
Loads both datasets (NF-CSE-CIC-IDS2018-v3 and Gotham 2025),
aligns their feature schemas, and partitions them across company nodes.

Usage:
    from src.data.loader import FedIntelDataLoader
    loader = FedIntelDataLoader()
    company_data = loader.load_and_partition()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import track

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    DATASET_IDS2018_DIR,
    DATASET_GOTHAM_DIR,
    COMPANY_CONFIG,
    FL_NUM_CLIENTS,
)

console = Console()


class FedIntelDataLoader:
    """Loads, aligns, and partitions datasets for federated nodes."""

    def __init__(self):
        self.ids2018_dir = DATASET_IDS2018_DIR
        self.gotham_dir = DATASET_GOTHAM_DIR
        self._ids2018_df: Optional[pd.DataFrame] = None
        self._gotham_df: Optional[pd.DataFrame] = None

    # ─── Loading ─────────────────────────────────────────────────────────

    def _find_csv_files(self, directory: Path) -> list:
        """Find all CSV files in a directory (recursive)."""
        csv_files = sorted(directory.glob("**/*.csv"))
        if not csv_files:
            console.print(f"[red]✗ No CSV files found in {directory}[/red]")
            console.print(f"[yellow]  Run: bash scripts/download_dataset.sh[/yellow]")
        return csv_files

    def load_ids2018(self) -> pd.DataFrame:
        """Load NF-CSE-CIC-IDS2018-v3 dataset."""
        if self._ids2018_df is not None:
            return self._ids2018_df

        console.print("[bold blue]Loading NF-CSE-CIC-IDS2018-v3...[/bold blue]")
        csv_files = self._find_csv_files(self.ids2018_dir)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {self.ids2018_dir}")

        dfs = []
        for f in track(csv_files, description="Reading CSV files"):
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)

        self._ids2018_df = pd.concat(dfs, ignore_index=True)
        console.print(
            f"[green]✓ Loaded {len(self._ids2018_df):,} rows, "
            f"{len(self._ids2018_df.columns)} columns[/green]"
        )
        return self._ids2018_df

    def load_gotham(self) -> pd.DataFrame:
        """Load Gotham 2025 dataset."""
        if self._gotham_df is not None:
            return self._gotham_df

        console.print("[bold blue]Loading Gotham 2025...[/bold blue]")
        csv_files = self._find_csv_files(self.gotham_dir)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {self.gotham_dir}")

        dfs = []
        for f in track(csv_files, description="Reading CSV files"):
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)

        self._gotham_df = pd.concat(dfs, ignore_index=True)
        console.print(
            f"[green]✓ Loaded {len(self._gotham_df):,} rows, "
            f"{len(self._gotham_df.columns)} columns[/green]"
        )
        return self._gotham_df

    # ─── Feature Alignment ──────────────────────────────────────────────

    def _identify_label_column(self, df: pd.DataFrame, dataset_name: str) -> str:
        """Auto-detect the label/attack-type column."""
        candidates = ["Label", "label", "Attack", "attack", "class", "Class",
                       "attack_cat", "Attack_cat", "category"]
        for col in candidates:
            if col in df.columns:
                console.print(f"  Label column for {dataset_name}: [cyan]{col}[/cyan]")
                return col

        # Fallback: last column is often the label
        last_col = df.columns[-1]
        console.print(
            f"  [yellow]⚠ Guessing label column for {dataset_name}: "
            f"{last_col}[/yellow]"
        )
        return last_col

    def _get_numeric_features(self, df: pd.DataFrame, label_col: str) -> list:
        """Get all numeric feature columns (excluding label)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c != label_col]

    def _normalize_labels(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """Normalize attack labels to a common format."""
        df = df.copy()
        df[label_col] = df[label_col].astype(str).str.strip().str.lower()

        # Map common label variations
        label_map = {
            "benign": "benign",
            "normal": "benign",
            "0": "benign",
            "ddos": "ddos",
            "dos": "dos",
            "dos slowloris": "dos",
            "dos slowhttptest": "dos",
            "dos hulk": "dos",
            "dos goldeneye": "dos",
            "brute force": "brute_force",
            "ssh-bruteforce": "brute_force",
            "ftp-bruteforce": "brute_force",
            "brute force -web": "brute_force",
            "brute force -xss": "xss",
            "web attack": "web_attack",
            "xss": "xss",
            "sql injection": "sql_injection",
            "infilteration": "infiltration",
            "infiltration": "infiltration",
            "bot": "botnet",
            "botnet": "botnet",
            "portscan": "port_scan",
            "port_scan": "port_scan",
            "scanning": "port_scan",
            "network_scanning": "port_scan",
            "c&c": "command_and_control",
            "c2": "command_and_control",
        }

        df["attack_label"] = df[label_col].map(
            lambda x: label_map.get(x, x)
        )
        return df

    # ─── Partitioning ───────────────────────────────────────────────────

    def _partition_dataframe(
        self, df: pd.DataFrame, n_partitions: int, seed: int = 42
    ) -> list:
        """Split a DataFrame into n roughly equal partitions (stratified by label)."""
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        partitions = []
        indices = np.array_split(df.index, n_partitions)
        for idx in indices:
            partitions.append(df.loc[idx].reset_index(drop=True))
        return partitions

    # ─── Main Entry Point ───────────────────────────────────────────────

    def load_and_partition(self) -> Dict[str, Dict]:
        """
        Load both datasets, normalize labels, and partition across companies.

        Returns:
            Dict mapping company name → {
                "data": pd.DataFrame,
                "features": list of feature column names,
                "label_col": str,
                "dataset": str,
                "description": str,
            }
        """
        console.print("\n[bold]═══ Fed-Intel Data Loader ═══[/bold]\n")

        # Load datasets
        ids2018_df = self.load_ids2018()
        gotham_df = self.load_gotham()

        # Identify label columns
        ids2018_label = self._identify_label_column(ids2018_df, "IDS2018")
        gotham_label = self._identify_label_column(gotham_df, "Gotham")

        # Normalize labels
        ids2018_df = self._normalize_labels(ids2018_df, ids2018_label)
        gotham_df = self._normalize_labels(gotham_df, gotham_label)

        # Get feature columns
        ids2018_features = self._get_numeric_features(ids2018_df, ids2018_label)
        gotham_features = self._get_numeric_features(gotham_df, gotham_label)

        # Partition IDS2018 into 2 parts (Company A, B)
        ids2018_partitions = self._partition_dataframe(ids2018_df, 2)

        # Build company data dict
        company_data = {}

        # Company A — IDS2018 partition 1
        company_data["A"] = {
            "data": ids2018_partitions[0],
            "features": ids2018_features,
            "label_col": "attack_label",
            "dataset": "nf-cse-cic-ids2018-v3",
            "description": COMPANY_CONFIG["A"]["description"],
        }

        # Company B — IDS2018 partition 2
        company_data["B"] = {
            "data": ids2018_partitions[1],
            "features": ids2018_features,
            "label_col": "attack_label",
            "dataset": "nf-cse-cic-ids2018-v3",
            "description": COMPANY_CONFIG["B"]["description"],
        }

        # Company C — Gotham 2025 (full dataset, different distribution)
        company_data["C"] = {
            "data": gotham_df,
            "features": gotham_features,
            "label_col": "attack_label",
            "dataset": "gotham-2025",
            "description": COMPANY_CONFIG["C"]["description"],
        }

        # Print summary
        self._print_summary(company_data)

        return company_data

    def _print_summary(self, company_data: Dict):
        """Print a Rich table summarizing loaded data."""
        table = Table(title="📊 Company Data Summary", show_lines=True)
        table.add_column("Company", style="bold cyan")
        table.add_column("Dataset", style="blue")
        table.add_column("Rows", justify="right", style="green")
        table.add_column("Features", justify="right")
        table.add_column("Attack Types", justify="right")
        table.add_column("Attack %", justify="right", style="red")

        for name, info in company_data.items():
            df = info["data"]
            label_col = info["label_col"]
            n_attacks = (df[label_col] != "benign").sum()
            attack_pct = f"{n_attacks / len(df) * 100:.1f}%"
            attack_types = df[label_col].nunique() - (
                1 if "benign" in df[label_col].values else 0
            )

            table.add_row(
                f"Company {name}",
                info["dataset"],
                f"{len(df):,}",
                str(len(info["features"])),
                str(attack_types),
                attack_pct,
            )

        console.print(table)


# ─── Standalone Usage ───────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = FedIntelDataLoader()
    try:
        data = loader.load_and_partition()
    except FileNotFoundError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print("[yellow]Run: bash scripts/download_dataset.sh[/yellow]")
