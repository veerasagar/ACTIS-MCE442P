"""
Fed-Intel Threat Summarizer
=============================
Converts raw network flow rows into natural language threat summaries.

Two modes:
  1. Deterministic (template-based) — fast, no API needed, used for bulk processing
  2. LLM-enhanced (Gemini) — richer analysis, used for high-severity threats

Usage:
    from src.data.summarizer import ThreatSummarizer
    summarizer = ThreatSummarizer()
    summary = summarizer.summarize_flow(row, features, attack_label)
"""

import json
from pathlib import Path
from typing import Dict, Optional
from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.mitre_mapper import MITREMapper

console = Console()

# ─── Protocol Mappings ──────────────────────────────────────────────────────

PROTOCOL_MAP = {
    1: "ICMP", 2: "IGMP", 6: "TCP", 17: "UDP", 47: "GRE", 50: "ESP",
    51: "AH", 58: "ICMPv6", 89: "OSPF", 132: "SCTP",
}

COMMON_PORTS = {
    20: "FTP-Data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
    53: "DNS", 67: "DHCP", 68: "DHCP", 80: "HTTP", 110: "POP3",
    123: "NTP", 143: "IMAP", 161: "SNMP", 443: "HTTPS", 445: "SMB",
    993: "IMAPS", 995: "POP3S", 1433: "MSSQL", 1883: "MQTT",
    3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL", 5900: "VNC",
    6379: "Redis", 8080: "HTTP-Proxy", 8443: "HTTPS-Alt",
    8883: "MQTT-TLS", 27017: "MongoDB",
}


def _port_service(port: int) -> str:
    """Get service name for a port number."""
    return COMMON_PORTS.get(port, f"port-{port}")


def _format_bytes(n: int) -> str:
    """Human-readable byte count."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / (1024 * 1024):.1f} MB"


def _format_duration(ms: int) -> str:
    """Human-readable duration from milliseconds."""
    if ms < 1000:
        return f"{ms} ms"
    elif ms < 60000:
        return f"{ms / 1000:.1f} sec"
    else:
        return f"{ms / 60000:.1f} min"


class ThreatSummarizer:
    """Generates natural language summaries from network flow data."""

    def __init__(self):
        self.mitre = MITREMapper()

    # ─── Deterministic Summarizer ───────────────────────────────────────

    def summarize_flow(
        self,
        row: Dict,
        attack_label: str,
        company: str = "Unknown",
    ) -> Dict:
        """
        Generate a structured threat summary from a single flow row.

        Args:
            row: dict of feature name → value for one flow
            attack_label: normalized attack label
            company: company identifier (A, B, C)

        Returns:
            Dict with structured threat summary
        """
        mitre = self.mitre.map(attack_label)
        protocol = PROTOCOL_MAP.get(int(row.get("PROTOCOL", 0)), "Unknown")
        src_port = int(row.get("L4_SRC_PORT", 0))
        dst_port = int(row.get("L4_DST_PORT", 0))
        in_bytes = int(row.get("IN_BYTES", 0))
        out_bytes = int(row.get("OUT_BYTES", 0))
        in_pkts = int(row.get("IN_PKTS", 0))
        out_pkts = int(row.get("OUT_PKTS", 0))
        duration = int(row.get("FLOW_DURATION_MILLISECONDS", 0))
        tcp_flags = int(row.get("TCP_FLAGS", 0))

        # Build natural language summary
        nl_summary = self._build_nl_summary(
            attack_label=attack_label,
            protocol=protocol,
            src_port=src_port,
            dst_port=dst_port,
            in_bytes=in_bytes,
            out_bytes=out_bytes,
            in_pkts=in_pkts,
            out_pkts=out_pkts,
            duration=duration,
            tcp_flags=tcp_flags,
            mitre=mitre,
            company=company,
        )

        # Structured output (used for ChromaDB metadata + RAG)
        return {
            "summary": nl_summary,
            "attack_type": attack_label,
            "mitre_technique_id": mitre["technique_id"],
            "mitre_tactic": mitre["tactic"],
            "severity": mitre["severity"],
            "severity_score": self.mitre.get_severity_score(attack_label),
            "protocol": protocol,
            "dst_service": _port_service(dst_port),
            "total_bytes": in_bytes + out_bytes,
            "total_pkts": in_pkts + out_pkts,
            "duration_ms": duration,
            "company": company,
            "flow_stats": {
                "in_bytes": in_bytes,
                "out_bytes": out_bytes,
                "in_pkts": in_pkts,
                "out_pkts": out_pkts,
                "byte_ratio": round(out_bytes / max(in_bytes, 1), 2),
                "pkt_rate": round(
                    (in_pkts + out_pkts) / max(duration / 1000, 0.001), 1
                ),
            },
        }

    def _build_nl_summary(
        self, attack_label, protocol, src_port, dst_port,
        in_bytes, out_bytes, in_pkts, out_pkts, duration, tcp_flags,
        mitre, company,
    ) -> str:
        """Build a deterministic natural language threat summary."""

        total_bytes = in_bytes + out_bytes
        total_pkts = in_pkts + out_pkts
        pkt_rate = total_pkts / max(duration / 1000, 0.001)
        dst_service = _port_service(dst_port)

        # Attack-specific descriptions
        if attack_label == "ddos":
            pattern = (
                f"High-volume distributed denial of service detected. "
                f"{_format_bytes(total_bytes)} transferred in "
                f"{_format_duration(duration)} ({pkt_rate:.0f} pkt/s). "
                f"Characteristic of {mitre['sub_technique']}."
            )
        elif attack_label == "dos":
            pattern = (
                f"Endpoint denial of service attack on {dst_service}. "
                f"{total_pkts:,} packets sent in {_format_duration(duration)}, "
                f"exhausting service resources."
            )
        elif attack_label == "brute_force":
            pattern = (
                f"Brute force credential attack targeting {dst_service} "
                f"({protocol}:{dst_port}). {total_pkts:,} login attempts "
                f"over {_format_duration(duration)}."
            )
        elif attack_label == "web_attack":
            pattern = (
                f"Web application attack targeting {dst_service}. "
                f"Potential SQL injection or XSS payload in {protocol} traffic. "
                f"{_format_bytes(in_bytes)} inbound."
            )
        elif attack_label == "botnet":
            pattern = (
                f"Botnet command-and-control communication detected. "
                f"{protocol} traffic to {dst_service}, "
                f"byte ratio (out/in): {out_bytes / max(in_bytes, 1):.2f}."
            )
        elif attack_label == "infiltration":
            pattern = (
                f"Network infiltration using legitimate {protocol} protocol. "
                f"Lateral movement via {dst_service}. "
                f"{_format_bytes(total_bytes)} exchanged."
            )
        elif attack_label == "reconnaissance":
            pattern = (
                f"Active network scanning and port enumeration. "
                f"{total_pkts:,} probes via {protocol}. "
                f"Scanning {dst_service} service."
            )
        elif attack_label == "theft":
            pattern = (
                f"Data exfiltration detected. {_format_bytes(out_bytes)} "
                f"outbound via {protocol}:{dst_port}. "
                f"Potential data theft over C2 channel."
            )
        else:
            pattern = (
                f"Anomalous {protocol} traffic pattern on {dst_service}. "
                f"{_format_bytes(total_bytes)}, {total_pkts:,} packets. "
                f"Requires further analysis."
            )

        return (
            f"[{mitre['severity']}] {mitre['technique']} ({mitre['technique_id']}) "
            f"— {pattern}"
        )

    # ─── Batch Summarization ────────────────────────────────────────────

    def summarize_batch(
        self,
        df,
        features: list,
        label_col: str,
        company: str,
        max_rows: Optional[int] = None,
    ) -> list:
        """
        Summarize multiple anomalous flows.

        Args:
            df: DataFrame with flow data
            features: list of feature column names
            label_col: name of the attack label column
            company: company identifier
            max_rows: limit number of summaries

        Returns:
            List of summary dicts
        """
        # Filter to attack rows only
        attacks = df[df[label_col] != "benign"]

        if max_rows:
            attacks = attacks.head(max_rows)

        summaries = []
        for _, row in attacks.iterrows():
            row_dict = row.to_dict()
            summary = self.summarize_flow(
                row=row_dict,
                attack_label=row_dict[label_col],
                company=company,
            )
            summaries.append(summary)

        console.print(
            f"[green]  Generated {len(summaries)} threat summaries "
            f"for Company {company}[/green]"
        )
        return summaries


# ─── Standalone Usage ───────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.loader import FedIntelDataLoader

    loader = FedIntelDataLoader()
    data = loader.load_and_partition()

    summarizer = ThreatSummarizer()

    # Preview summaries from each company
    for company_id in ["A", "B", "C"]:
        info = data[company_id]
        console.print(f"\n[bold]══ Company {company_id} ══[/bold]")

        summaries = summarizer.summarize_batch(
            df=info["data"],
            features=info["features"],
            label_col=info["label_col"],
            company=company_id,
            max_rows=3,
        )

        for s in summaries:
            console.print(f"\n  [cyan]{s['summary']}[/cyan]")
            console.print(f"  MITRE: {s['mitre_technique_id']} | "
                           f"Tactic: {s['mitre_tactic']} | "
                           f"Severity: {s['severity']}")
