"""
ACTIS Live Monitor
===================
Real-time network flow detection and alerting.

Processes a stream of NetFlow records (from CSV, stdin, or a live sniffer)
and runs the full ACTIS pipeline:
  1. Feature engineering (58-dim)
  2. IDS classification
  3. Zero-day detection (ZeroDayDetector)
  4. PII sanitization
  5. Threat summarization (LLM-enhanced for HIGH/CRITICAL)
  6. Alert output (console / JSON file)

This enables actual deployment — not just simulation.

Usage:
    # Stream from CSV file
    python -m src.live_monitor --input flows.csv --model models/ids_model.pt

    # Stream from stdin (nfdump output piped in)
    nfdump -r capture.nfcapd -o csv | python -m src.live_monitor --stdin

    # Continuous tail of a growing CSV log
    python -m src.live_monitor --tail /var/log/netflows.csv --interval 5
"""

import os
import sys
import json
import time
import signal
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

from src.data.feature_engineer import FeatureEngineer
from src.data.summarizer import ThreatSummarizer
from src.local_node.ids_model import IDSModel
from src.local_node.ids_trainer import IDSTrainer
from src.local_node.zeroday_detector import ZeroDayDetector
from src.local_node.pii_validator import PIIValidator

console = Console()

# Unified label set (must match training)
UNIFIED_LABELS = sorted([
    "benign", "botnet", "brute_force", "ddos", "dos",
    "infiltration", "reconnaissance", "theft", "web_attack",
])

# Alert severity to color
SEVERITY_COLOR = {
    "CRITICAL": "bold red",
    "HIGH":     "red",
    "MEDIUM":   "yellow",
    "LOW":      "green",
    "NONE":     "dim",
}


class LiveMonitor:
    """
    Real-time ACTIS detection engine.

    Loads a pre-trained IDS model + ZeroDayDetector, then processes
    incoming NetFlow records from any source.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        company_id: str = "A",
        zds_percentile: float = 90.0,
        alert_output: Optional[str] = None,
    ):
        self.company_id = company_id
        self.alert_output = alert_output
        self.fe = FeatureEngineer()
        self.summarizer = ThreatSummarizer(use_llm=True)
        self.validator = PIIValidator()
        self.num_classes = len(UNIFIED_LABELS)
        self.label_map = {i: name for i, name in enumerate(UNIFIED_LABELS)}

        # Stats
        self.stats = {
            "total": 0, "alerts": 0, "zero_day": 0,
            "blocked": 0, "start_time": time.time()
        }

        # Load or create model
        input_dim = self.fe.output_dim
        self.model = IDSModel(input_dim=input_dim, num_classes=self.num_classes)
        if model_path and Path(model_path).exists():
            import torch
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            console.print(f"  [green]Model loaded: {model_path}[/green]")
        else:
            console.print("  [yellow]No model path provided — using untrained model. "
                          "Run training first.[/yellow]")

        self.detector = ZeroDayDetector(
            self.model, percentile=zds_percentile, alpha=0.3
        )
        self._detector_fitted = False

    def fit_detector(self, X_known: np.ndarray, y_known: np.ndarray):
        """Fit zero-day detector on known-class training data."""
        console.print("  [dim]Fitting ZeroDayDetector...[/dim]")
        self.detector.fit_thresholds(X_known, y_known)
        self._detector_fitted = True

    def process_flow(self, row: Dict) -> Optional[Dict]:
        """
        Process a single NetFlow record.

        Args:
            row: Dict of feature_name → value

        Returns:
            Alert dict if attack detected, None if benign
        """
        import torch
        import torch.nn.functional as F

        self.stats["total"] += 1

        # Feature extraction
        feat_cols = [
            "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
            "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS",
            "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
            "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL",
            "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN",
            "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
            "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
            "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
            "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
            "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
            "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
            "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
            "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
            "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
        ]
        raw = np.array([float(row.get(c, 0)) for c in feat_cols], dtype=np.float32)
        raw = np.nan_to_num(raw, nan=0, posinf=1e6, neginf=0).reshape(1, -1)
        enhanced = self.fe.transform(raw)

        # IDS prediction
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(enhanced, dtype=torch.float32)
            logits = self.model(X_t)
            probs = F.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.label_map[pred_idx]
        confidence = float(probs[pred_idx])

        # Skip benign
        if pred_label == "benign" and confidence > 0.7:
            return None

        # Zero-day check
        is_zeroday = False
        zds_score = 0.0
        if self._detector_fitted:
            zds, _, _ = self.detector._compute_scores(enhanced)
            zds_score = float(zds[0])
            is_zeroday = bool(zds_score > self.detector.threshold)

        self.stats["alerts"] += 1
        if is_zeroday:
            self.stats["zero_day"] += 1

        # Threat summary (LLM-enhanced for HIGH/CRITICAL)
        summary = self.summarizer.summarize_flow(row, pred_label, self.company_id)

        # PII check
        is_clean, pii_found = self.validator.validate(summary)
        if not is_clean:
            self.stats["blocked"] += 1
            summary["summary"] = "[PII REDACTED] " + summary["summary"]

        alert = {
            "timestamp": datetime.now().isoformat(),
            "company": self.company_id,
            "predicted_attack": pred_label,
            "confidence": round(confidence, 4),
            "is_zero_day": is_zeroday,
            "zds_score": round(zds_score, 4),
            "severity": summary.get("severity", "UNKNOWN"),
            "mitre": summary.get("mitre_technique_id", ""),
            "protocol": summary.get("protocol", ""),
            "dst_service": summary.get("dst_service", ""),
            "summary": summary.get("summary", ""),
            "llm_enhanced": summary.get("llm_enhanced", False),
            "pii_clean": is_clean,
        }

        if self.alert_output:
            with open(self.alert_output, "a") as f:
                f.write(json.dumps(alert) + "\n")

        return alert

    def process_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 100,
        verbose: bool = True,
    ) -> List[Dict]:
        """Process a DataFrame of flows, return list of alerts."""
        alerts = []
        total = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            alert = self.process_flow(row.to_dict())
            if alert:
                alerts.append(alert)
                if verbose:
                    self._print_alert(alert)

            if verbose and (i + 1) % batch_size == 0:
                elapsed = time.time() - self.stats["start_time"]
                rate = (i + 1) / max(elapsed, 0.001)
                console.print(
                    f"  [dim]Progress {i+1}/{total} ({rate:.0f} flows/s) "
                    f"| Alerts: {self.stats['alerts']} "
                    f"| ZDs: {self.stats['zero_day']}[/dim]"
                )

        return alerts

    def tail_csv(self, path: str, interval: float = 5.0):
        """
        Continuously read new rows from a growing CSV file.
        Simulates live packet-capture deployment.
        """
        console.print(f"\n[bold]ACTIS Live Monitor — tailing {path}[/bold]")
        console.print(f"  Poll interval: {interval}s | Ctrl+C to stop\n")

        seen_rows = 0
        running = True

        def _stop(sig, frame):
            nonlocal running
            running = False
            console.print("\n[yellow]Stopping monitor...[/yellow]")

        signal.signal(signal.SIGINT, _stop)

        while running:
            try:
                df = pd.read_csv(path)
                new_rows = df.iloc[seen_rows:]
                if len(new_rows) > 0:
                    alerts = self.process_dataframe(new_rows, verbose=True)
                    seen_rows = len(df)
                    self._print_summary()
                else:
                    console.print(f"  [dim]{datetime.now().strftime('%H:%M:%S')} "
                                  f"— waiting for new flows...[/dim]")
                time.sleep(interval)
            except FileNotFoundError:
                console.print(f"  [red]File not found: {path}[/red]")
                time.sleep(interval)

    def _print_alert(self, alert: Dict):
        """Print a single alert to console."""
        sev = alert.get("severity", "UNKNOWN")
        color = SEVERITY_COLOR.get(sev, "white")
        zd_tag = " [bold magenta][ZERO-DAY][/bold magenta]" if alert["is_zero_day"] else ""
        llm_tag = " [dim][LLM][/dim]" if alert.get("llm_enhanced") else ""
        console.print(
            f"  [{color}][{sev}][/{color}]{zd_tag}{llm_tag} "
            f"• {alert['predicted_attack'].upper()} "
            f"({alert['confidence']:.0%}) "
            f"• {alert['protocol']}→{alert['dst_service']} "
            f"• {alert['mitre']}"
        )
        if alert.get("summary"):
            console.print(f"    [dim]{alert['summary'][:120]}...[/dim]")

    def _print_summary(self):
        """Print running summary stats."""
        elapsed = time.time() - self.stats["start_time"]
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_row("Flows processed:", str(self.stats["total"]))
        tbl.add_row("Alerts generated:", f"[red]{self.stats['alerts']}[/red]")
        tbl.add_row("Zero-day suspects:", f"[magenta]{self.stats['zero_day']}[/magenta]")
        tbl.add_row("PII blocked:", str(self.stats["blocked"]))
        tbl.add_row("Throughput:", f"{self.stats['total']/max(elapsed,0.001):.0f} flows/s")
        console.print(Panel(tbl, title="[bold]ACTIS Stats[/bold]", border_style="dim"))


def main():
    parser = argparse.ArgumentParser(description="ACTIS Live Network Flow Monitor")
    parser.add_argument("--input",    type=str, help="CSV file to process once")
    parser.add_argument("--tail",     type=str, help="CSV file to tail continuously")
    parser.add_argument("--stdin",    action="store_true", help="Read CSV from stdin")
    parser.add_argument("--model",    type=str, default=None, help="Path to saved IDS model .pt")
    parser.add_argument("--company",  type=str, default="A", help="Company ID (A/B/C)")
    parser.add_argument("--output",   type=str, default=None, help="JSONL alert output file")
    parser.add_argument("--interval", type=float, default=5.0, help="Poll interval for --tail (s)")
    args = parser.parse_args()

    console.print(Panel(
        "[bold magenta]ACTIS — Adaptive Cyber Threat Intelligence System[/bold magenta]\n"
        "[dim]Real-time network flow monitoring[/dim]",
        border_style="magenta"
    ))

    monitor = LiveMonitor(
        model_path=args.model,
        company_id=args.company,
        alert_output=args.output,
    )

    if args.tail:
        monitor.tail_csv(args.tail, interval=args.interval)
    elif args.input:
        console.print(f"\n[bold]Processing {args.input}...[/bold]")
        df = pd.read_csv(args.input)
        alerts = monitor.process_dataframe(df)
        monitor._print_summary()
        console.print(f"\n[green]Done — {len(alerts)} alerts generated[/green]")
    elif args.stdin:
        console.print("\n[bold]Reading from stdin...[/bold]")
        df = pd.read_csv(sys.stdin)
        alerts = monitor.process_dataframe(df)
        monitor._print_summary()
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
