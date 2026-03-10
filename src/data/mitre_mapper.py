"""
Fed-Intel MITRE ATT&CK Mapper
===============================
Maps normalized attack labels to MITRE ATT&CK technique IDs and tactic categories.

This uses a static mapping based on the attack types present in
NF-CSE-CIC-IDS2018-v2 and NF-BoT-IoT-v2 datasets.

Usage:
    from src.data.mitre_mapper import MITREMapper
    mapper = MITREMapper()
    info = mapper.map("ddos")
    # → {"technique_id": "T1498", "technique": "Network DoS", "tactic": "Impact", ...}
"""

from typing import Dict, Optional
from rich.console import Console
from rich.table import Table

console = Console()


# ─── MITRE ATT&CK Mapping Table ────────────────────────────────────────────
# Based on MITRE ATT&CK Enterprise Matrix (v14+)
# Maps our normalized attack labels → technique IDs

MITRE_MAP: Dict[str, Dict] = {
    "ddos": {
        "technique_id": "T1498",
        "technique": "Network Denial of Service",
        "sub_technique": "T1498.001 - Direct Network Flood",
        "tactic": "Impact",
        "severity": "HIGH",
        "description": "Distributed denial of service via volumetric flooding (HOIC, LOIC).",
    },
    "dos": {
        "technique_id": "T1499",
        "technique": "Endpoint Denial of Service",
        "sub_technique": "T1499.002 - Service Exhaustion Flood",
        "tactic": "Impact",
        "severity": "HIGH",
        "description": "DoS targeting application endpoints (Hulk, GoldenEye, Slowloris).",
    },
    "brute_force": {
        "technique_id": "T1110",
        "technique": "Brute Force",
        "sub_technique": "T1110.001 - Password Guessing",
        "tactic": "Credential Access",
        "severity": "MEDIUM",
        "description": "Credential brute force via SSH or FTP password guessing.",
    },
    "web_attack": {
        "technique_id": "T1190",
        "technique": "Exploit Public-Facing Application",
        "sub_technique": "T1190 - SQL Injection / XSS",
        "tactic": "Initial Access",
        "severity": "CRITICAL",
        "description": "Web application exploitation via SQL injection or cross-site scripting.",
    },
    "botnet": {
        "technique_id": "T1583",
        "technique": "Acquire Infrastructure: Botnet",
        "sub_technique": "T1583.005 - Botnet",
        "tactic": "Resource Development",
        "severity": "HIGH",
        "description": "Compromised hosts acting as part of a botnet.",
    },
    "infiltration": {
        "technique_id": "T1071",
        "technique": "Application Layer Protocol",
        "sub_technique": "T1071.001 - Web Protocols",
        "tactic": "Command and Control",
        "severity": "CRITICAL",
        "description": "Internal network infiltration using legitimate application protocols.",
    },
    "reconnaissance": {
        "technique_id": "T1595",
        "technique": "Active Scanning",
        "sub_technique": "T1595.001 - Scanning IP Blocks",
        "tactic": "Reconnaissance",
        "severity": "LOW",
        "description": "Network scanning and port enumeration for target discovery.",
    },
    "theft": {
        "technique_id": "T1041",
        "technique": "Exfiltration Over C2 Channel",
        "sub_technique": "T1041",
        "tactic": "Exfiltration",
        "severity": "CRITICAL",
        "description": "Data theft / exfiltration over command and control channels.",
    },
    "benign": {
        "technique_id": "N/A",
        "technique": "Normal Traffic",
        "sub_technique": "N/A",
        "tactic": "N/A",
        "severity": "NONE",
        "description": "Legitimate network traffic.",
    },
}

# Severity → numeric score (for trust scoring and prioritization)
SEVERITY_SCORES = {
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


class MITREMapper:
    """Maps attack labels to MITRE ATT&CK techniques."""

    def __init__(self):
        self.mapping = MITRE_MAP

    def map(self, attack_label: str) -> Dict:
        """
        Map a normalized attack label to MITRE ATT&CK info.

        Args:
            attack_label: normalized label (e.g., "ddos", "brute_force")

        Returns:
            Dict with technique_id, technique, tactic, severity, etc.
        """
        label = attack_label.lower().strip()
        if label in self.mapping:
            return self.mapping[label]

        # Unknown attack — return a generic entry
        return {
            "technique_id": "UNKNOWN",
            "technique": f"Unknown: {attack_label}",
            "sub_technique": "N/A",
            "tactic": "Unknown",
            "severity": "MEDIUM",
            "description": f"Unclassified attack pattern: {attack_label}",
        }

    def get_severity_score(self, attack_label: str) -> int:
        """Get numeric severity score (0-4)."""
        info = self.map(attack_label)
        return SEVERITY_SCORES.get(info["severity"], 2)

    def get_technique_id(self, attack_label: str) -> str:
        """Get just the MITRE ATT&CK technique ID."""
        return self.map(attack_label)["technique_id"]

    def get_tactic(self, attack_label: str) -> str:
        """Get the tactic category."""
        return self.map(attack_label)["tactic"]

    def print_mapping_table(self):
        """Print a Rich table of all mappings."""
        table = Table(title="🛡️ MITRE ATT&CK Mapping", show_lines=True)
        table.add_column("Label", style="bold cyan")
        table.add_column("Technique ID", style="red")
        table.add_column("Technique", style="white")
        table.add_column("Tactic", style="blue")
        table.add_column("Severity", style="bold")

        severity_colors = {
            "NONE": "dim", "LOW": "green", "MEDIUM": "yellow",
            "HIGH": "red", "CRITICAL": "bold red",
        }

        for label, info in self.mapping.items():
            if label == "benign":
                continue
            color = severity_colors.get(info["severity"], "white")
            table.add_row(
                label,
                info["technique_id"],
                info["technique"],
                info["tactic"],
                f"[{color}]{info['severity']}[/{color}]",
            )

        console.print(table)


# ─── Standalone Usage ───────────────────────────────────────────────────────

if __name__ == "__main__":
    mapper = MITREMapper()
    mapper.print_mapping_table()

    # Test
    for label in ["ddos", "brute_force", "web_attack", "theft", "benign"]:
        info = mapper.map(label)
        console.print(f"\n[bold]{label}[/bold] → {info['technique_id']} "
                       f"({info['tactic']}) [{info['severity']}]")
