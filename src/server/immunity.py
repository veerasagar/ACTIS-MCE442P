"""
Fed-Intel Immunity Engine
==========================
Generates firewall/defense rules from RAG-retrieved threat intelligence.
Uses the RAG engine to find relevant threats and produces actionable
defense configurations.

This is how federation translates to concrete defense:
  Company A gets attacked → summary shared → Company B gets firewall rules

Usage:
    engine = ImmunityEngine(rag)
    rules = engine.generate_rules("ddos", company_id="B")
"""

from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.server.rag_engine import RAGEngine

console = Console()

# ─── Rule Templates ────────────────────────────────────────────────────────

RULE_TEMPLATES = {
    "ddos": {
        "iptables": "iptables -A INPUT -p tcp --syn -m limit --limit 25/s --limit-burst 50 -j ACCEPT\n"
                    "iptables -A INPUT -p tcp --syn -j DROP",
        "description": "Rate-limit SYN packets to 25/s with burst of 50. Drop excess.",
        "priority": "CRITICAL",
    },
    "dos": {
        "iptables": "iptables -A INPUT -p tcp --dport {port} -m connlimit --connlimit-above 50 -j REJECT",
        "description": "Limit concurrent connections per source to 50.",
        "priority": "HIGH",
    },
    "brute_force": {
        "iptables": "iptables -A INPUT -p tcp --dport 22 -m recent --set --name sshbrute\n"
                    "iptables -A INPUT -p tcp --dport 22 -m recent --update --seconds 60 "
                    "--hitcount 4 --name sshbrute -j DROP",
        "description": "Block IPs with >3 SSH attempts in 60 seconds.",
        "priority": "HIGH",
    },
    "web_attack": {
        "iptables": "# Deploy WAF (ModSecurity) with OWASP CRS ruleset\n"
                    "iptables -A INPUT -p tcp --dport 80 -m string --algo bm "
                    '--string "SELECT" -j DROP',
        "description": "Block common SQLi/XSS patterns. Deploy ModSecurity WAF.",
        "priority": "HIGH",
    },
    "botnet": {
        "iptables": "# Block known C2 communication patterns\n"
                    "iptables -A OUTPUT -p tcp --dport 6667 -j DROP\n"
                    "iptables -A OUTPUT -p tcp --dport 8080 -m string --algo bm "
                    '--string "POST /gate" -j DROP',
        "description": "Block IRC C2 channels and common botnet gate endpoints.",
        "priority": "CRITICAL",
    },
    "reconnaissance": {
        "iptables": "iptables -A INPUT -p icmp --icmp-type echo-request -m limit "
                    "--limit 1/s -j ACCEPT\n"
                    "iptables -A INPUT -p icmp --icmp-type echo-request -j DROP",
        "description": "Rate-limit ICMP echo requests to 1/s to prevent scanning.",
        "priority": "MEDIUM",
    },
    "theft": {
        "iptables": "# Enable DLP: block large outbound transfers\n"
                    "iptables -A OUTPUT -p tcp -m length --length 10000:65535 "
                    "-m limit --limit 10/m -j ACCEPT\n"
                    "iptables -A OUTPUT -p tcp -m length --length 10000:65535 -j LOG",
        "description": "Monitor and limit large outbound data transfers.",
        "priority": "CRITICAL",
    },
    "infiltration": {
        "iptables": "# Segment network and log lateral movement\n"
                    "iptables -A FORWARD -s 10.0.1.0/24 -d 10.0.2.0/24 -j LOG\n"
                    "iptables -A FORWARD -s 10.0.1.0/24 -d 10.0.2.0/24 -j DROP",
        "description": "Block and log cross-subnet lateral movement.",
        "priority": "HIGH",
    },
}


class ImmunityEngine:
    """
    Generates firewall rules from federated threat intelligence.

    Uses RAG to find relevant threats from OTHER companies,
    then generates defense rules based on template + context.
    """

    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
        self.generated_rules: List[Dict] = []

    def generate_rules(
        self,
        attack_type: str,
        company_id: str,
        port: int = 80,
    ) -> Dict:
        """
        Generate firewall rules for a specific attack type.

        Uses RAG to find intelligence from other companies,
        then generates actionable defense rules.

        Args:
            attack_type: normalized attack type
            company_id: company requesting defense
            port: target port for port-specific rules

        Returns:
            Dict with rules, context, and metadata
        """
        # Get cross-org intelligence
        similar = self.rag.find_similar_attacks(attack_type, company_id)

        # Get template rule
        template = RULE_TEMPLATES.get(attack_type, {
            "iptables": "# No specific rule template for {}".format(attack_type),
            "description": "Generic monitoring rule",
            "priority": "MEDIUM",
        })

        # Format rule with port
        rule_text = template["iptables"].format(port=port)

        result = {
            "attack_type": attack_type,
            "company_id": company_id,
            "rule": rule_text,
            "description": template["description"],
            "priority": template["priority"],
            "based_on_threats": len(similar),
            "cross_org_intelligence": [
                {
                    "source": s.get("metadata", {}).get("company_id"),
                    "description": s.get("document", "")[:100],
                    "severity": s.get("metadata", {}).get("severity"),
                }
                for s in similar[:3]
            ],
        }

        self.generated_rules.append(result)

        # Also store in KB for future reference
        self.rag.kb.add_defense({
            "attack_type": attack_type,
            "rule": rule_text,
            "rule_type": "immunity_generated",
            "source_company": company_id,
        })

        return result

    def generate_all_rules(self, company_id: str) -> List[Dict]:
        """Generate rules for all known attack types."""
        rules = []
        for attack_type in RULE_TEMPLATES:
            rule = self.generate_rules(attack_type, company_id)
            rules.append(rule)
        return rules

    def print_rules(self, rules: Optional[List[Dict]] = None):
        """Print rules in a formatted table."""
        rules = rules or self.generated_rules

        table = Table(
            title="🛡️ Immunity Rules for Federation",
            show_header=True,
        )
        table.add_column("Attack Type", style="cyan")
        table.add_column("Priority", style="red")
        table.add_column("Description", style="white")
        table.add_column("Intel Sources", style="yellow")

        for r in rules:
            table.add_row(
                r["attack_type"],
                r["priority"],
                r["description"],
                str(r["based_on_threats"]),
            )

        console.print(table)

    def get_stats(self) -> Dict:
        """Get immunity engine statistics."""
        return {
            "rules_generated": len(self.generated_rules),
            "attack_types_covered": len(set(
                r["attack_type"] for r in self.generated_rules
            )),
        }
