"""
Fed-Intel Data Loader
=====================
Loads both datasets (NF-CSE-CIC-IDS2018-v2 and NF-BoT-IoT-v2),
normalizes labels, and partitions them across company nodes.

Both datasets share the identical 43-column NetFlow v2 schema.

Usage:
    from src.data.loader import FedIntelDataLoader
    loader = FedIntelDataLoader()
    company_data = loader.load_and_partition()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    DATASET_IDS2018_DIR,
    DATASET_BOTIOT_DIR,
    COMPANY_CONFIG,
)

console = Console()

# ─── Label Mapping ──────────────────────────────────────────────────────────
# Maps raw attack labels from both datasets to a unified taxonomy

LABEL_MAP = {
    # Benign
    "benign": "benign",
    "normal": "benign",
    # DDoS
    "ddos attack-hoic": "ddos",
    "ddos attacks-loic-http": "ddos",
    "ddos attack-loic-udp": "ddos",
    "ddos": "ddos",
    # DoS
    "dos attacks-hulk": "dos",
    "dos attacks-goldeneye": "dos",
    "dos attacks-slowhttptest": "dos",
    "dos attacks-slowloris": "dos",
    "dos": "dos",
    # Brute Force
    "ssh-bruteforce": "brute_force",
    "ftp-bruteforce": "brute_force",
    "brute force -web": "brute_force",
    # Web Attacks
    "brute force -xss": "web_attack",
    "sql injection": "web_attack",
    # Botnet / Bot
    "bot": "botnet",
    # Infiltration
    "infilteration": "infiltration",
    "infiltration": "infiltration",
    # IoT-specific (BoT-IoT)
    "reconnaissance": "reconnaissance",
    "theft": "theft",
}

# Columns that are NOT numeric features (used as labels or identifiers)
NON_FEATURE_COLS = {"Label", "Attack", "attack_label", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"}


class FedIntelDataLoader:
    """Loads, normalizes, and partitions datasets for federated nodes."""

    def __init__(self):
        self.ids2018_dir = DATASET_IDS2018_DIR
        self.botiot_dir = DATASET_BOTIOT_DIR
        self._ids2018_df: Optional[pd.DataFrame] = None
        self._botiot_df: Optional[pd.DataFrame] = None

    # ─── Loading ─────────────────────────────────────────────────────────

    def _load_parquet(self, directory: Path, name: str) -> pd.DataFrame:
        """Load a Parquet file from a directory."""
        parquet_files = sorted(directory.glob("**/*.parquet"))
        if not parquet_files:
            console.print(f"[red]✗ No Parquet files found in {directory}[/red]")
            console.print("[yellow]  Run: fedintel data download[/yellow]")
            raise FileNotFoundError(f"No Parquet files in {directory}")

        console.print(f"[bold blue]Loading {name}...[/bold blue]")
        # Use first parquet file (each dataset has one main file)
        df = pd.read_parquet(parquet_files[0])
        console.print(
            f"[green]  ✓ {len(df):,} rows × {len(df.columns)} columns "
            f"({parquet_files[0].name})[/green]"
        )
        return df

    def load_ids2018(self) -> pd.DataFrame:
        """Load NF-CSE-CIC-IDS2018-v2 dataset."""
        if self._ids2018_df is None:
            self._ids2018_df = self._load_parquet(
                self.ids2018_dir, "NF-CSE-CIC-IDS2018-v2"
            )
        return self._ids2018_df

    def load_botiot(self) -> pd.DataFrame:
        """Load NF-BoT-IoT-v2 dataset."""
        if self._botiot_df is None:
            self._botiot_df = self._load_parquet(
                self.botiot_dir, "NF-BoT-IoT-v2"
            )
        return self._botiot_df

    # ─── Label Normalization ────────────────────────────────────────────

    def _normalize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the 'Attack' column to a unified taxonomy."""
        df = df.copy()
        raw_labels = df["Attack"].astype(str).str.strip().str.lower()
        df["attack_label"] = raw_labels.map(lambda x: LABEL_MAP.get(x, x))

        unknown = set(df["attack_label"].unique()) - set(LABEL_MAP.values())
        if unknown:
            console.print(f"  [yellow]⚠ Unknown labels: {unknown}[/yellow]")

        return df

    # ─── Feature Extraction ─────────────────────────────────────────────

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list:
        """Get the list of numeric feature columns (excluding labels)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in NON_FEATURE_COLS]

    # ─── Partitioning ───────────────────────────────────────────────────

    def _partition_dataframe(
        self, df: pd.DataFrame, n_partitions: int, seed: int = 42
    ) -> list:
        """Split DataFrame into n roughly equal partitions."""
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        indices = np.array_split(df.index, n_partitions)
        return [df.loc[idx].reset_index(drop=True) for idx in indices]

    # ─── Main Entry Point ───────────────────────────────────────────────

    def load_and_partition(self) -> Dict[str, Dict]:
        """
        Load both datasets, normalize labels, and partition across companies.

        Returns:
            Dict[str, Dict] with keys "A", "B", "C" — each containing:
                - data: pd.DataFrame
                - features: list of feature column names
                - label_col: "attack_label"
                - dataset: str
                - description: str
        """
        console.print("\n[bold]═══ Fed-Intel Data Loader ═══[/bold]\n")

        # Load
        ids2018_df = self.load_ids2018()
        botiot_df = self.load_botiot()

        # Normalize labels
        ids2018_df = self._normalize_labels(ids2018_df)
        botiot_df = self._normalize_labels(botiot_df)

        # Verify shared schema
        ids_features = self.get_feature_columns(ids2018_df)
        bot_features = self.get_feature_columns(botiot_df)
        shared = set(ids_features) & set(bot_features)
        console.print(f"\n  Shared numeric features: [bold green]{len(shared)}/41[/bold green]")

        # Partition IDS2018 into 2 parts (Company A, B)
        ids2018_partitions = self._partition_dataframe(ids2018_df, 2)

        company_data = {
            "A": {
                "data": ids2018_partitions[0],
                "features": ids_features,
                "label_col": "attack_label",
                "dataset": COMPANY_CONFIG["A"]["dataset"],
                "description": COMPANY_CONFIG["A"]["description"],
            },
            "B": {
                "data": ids2018_partitions[1],
                "features": ids_features,
                "label_col": "attack_label",
                "dataset": COMPANY_CONFIG["B"]["dataset"],
                "description": COMPANY_CONFIG["B"]["description"],
            },
            "C": {
                "data": botiot_df,
                "features": bot_features,
                "label_col": "attack_label",
                "dataset": COMPANY_CONFIG["C"]["dataset"],
                "description": COMPANY_CONFIG["C"]["description"],
            },
        }

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
            lbl = info["label_col"]
            n_attacks = (df[lbl] != "benign").sum()
            attack_pct = f"{n_attacks / len(df) * 100:.1f}%"
            n_types = df[lbl].nunique() - (1 if "benign" in df[lbl].values else 0)

            table.add_row(
                f"Company {name}",
                info["dataset"],
                f"{len(df):,}",
                str(len(info["features"])),
                str(n_types),
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
        console.print("[yellow]Run: fedintel data download[/yellow]")
