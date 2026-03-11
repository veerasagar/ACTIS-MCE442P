"""
Fed-Intel Agent Analyzer
========================
LLM-powered agent that reads anomalous network flows and produces
structured attack analysis with MITRE ATT&CK mapping.

When a Gemini API key is available, uses LLM for rich analysis.
Falls back to deterministic analysis (using summarizer + MITRE mapper)
when no API key is configured.

Usage:
    analyzer = AgentAnalyzer()
    result = analyzer.analyze(flow_row, attack_label, company_id)
"""

import json
from pathlib import Path
from typing import Dict, Optional
from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import GEMINI_API_KEY, LLM_PRIMARY_MODEL, LLM_TEMPERATURE
from src.data.mitre_mapper import MITREMapper
from src.data.summarizer import ThreatSummarizer

console = Console()

# ─── Prompt Templates ───────────────────────────────────────────────────────

ANALYZER_SYSTEM_PROMPT = """You are an expert cybersecurity analyst for a federated threat intelligence network.
Your task: analyze a network flow that was flagged as malicious by an IDS.

RULES:
1. Output ONLY valid JSON — no markdown, no explanation outside JSON.
2. Be precise about the attack technique and MITRE ATT&CK mapping.
3. Focus on actionable intelligence that helps OTHER organizations defend.
4. Do NOT include any IP addresses, MAC addresses, or PII in your output."""

ANALYZER_USER_PROMPT = """Analyze this flagged network flow:

Attack Label: {attack_label}
Company: {company_id}
Protocol: {protocol}
Src Port: {src_port} → Dst Port: {dst_port}
Bytes In/Out: {in_bytes}/{out_bytes}
Packets In/Out: {in_pkts}/{out_pkts}
Duration: {duration_ms}ms
TCP Flags: {tcp_flags}

Produce a JSON object with these fields:
{{
  "attack_type": "string - normalized attack category",
  "mitre_technique_id": "string - e.g. T1498",
  "mitre_tactic": "string - e.g. Impact",
  "severity": "string - low/medium/high/critical",
  "confidence": "float 0-1",
  "attack_description": "string - 1-2 sentence description of the attack pattern",
  "indicators": ["list of observable indicators without PII"],
  "recommended_defense": "string - 1 sentence defense recommendation"
}}"""


class AgentAnalyzer:
    """
    Agentic attack analyzer.
    Uses Gemini LLM when available, deterministic fallback otherwise.
    """

    def __init__(self):
        self.mapper = MITREMapper()
        self.summarizer = ThreatSummarizer()
        self._llm = None

        if GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                self._llm = genai.GenerativeModel(
                    LLM_PRIMARY_MODEL,
                    generation_config={"temperature": LLM_TEMPERATURE},
                )
                console.print("  [green]Analyzer: Gemini LLM loaded[/green]")
            except Exception as e:
                console.print(f"  [yellow]Analyzer: LLM init failed ({e}), using deterministic mode[/yellow]")
        else:
            console.print("  [dim]Analyzer: No API key — deterministic mode[/dim]")

    def analyze(
        self, flow_row: Dict, attack_label: str, company_id: str
    ) -> Dict:
        """
        Analyze a flagged network flow.

        Args:
            flow_row: dict of flow features
            attack_label: normalized attack label (e.g. 'ddos')
            company_id: company identifier (A/B/C)

        Returns:
            Structured analysis dict
        """
        if self._llm:
            return self._llm_analyze(flow_row, attack_label, company_id)
        return self._deterministic_analyze(flow_row, attack_label, company_id)

    def _llm_analyze(
        self, flow_row: Dict, attack_label: str, company_id: str
    ) -> Dict:
        """Use Gemini LLM for rich attack analysis."""
        try:
            prompt = ANALYZER_USER_PROMPT.format(
                attack_label=attack_label,
                company_id=company_id,
                protocol=flow_row.get("PROTOCOL", "unknown"),
                src_port=flow_row.get("L4_SRC_PORT", "?"),
                dst_port=flow_row.get("L4_DST_PORT", "?"),
                in_bytes=flow_row.get("IN_BYTES", 0),
                out_bytes=flow_row.get("OUT_BYTES", 0),
                in_pkts=flow_row.get("IN_PKTS", 0),
                out_pkts=flow_row.get("OUT_PKTS", 0),
                duration_ms=flow_row.get("FLOW_DURATION_MILLISECONDS", 0),
                tcp_flags=flow_row.get("TCP_FLAGS", 0),
            )

            response = self._llm.generate_content(
                [ANALYZER_SYSTEM_PROMPT, prompt]
            )

            # Parse JSON from response
            text = response.text.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(text)
            result["source"] = "llm"
            result["company_id"] = company_id
            return result

        except Exception as e:
            console.print(f"  [yellow]LLM analysis failed: {e}, falling back[/yellow]")
            return self._deterministic_analyze(flow_row, attack_label, company_id)

    def _deterministic_analyze(
        self, flow_row: Dict, attack_label: str, company_id: str
    ) -> Dict:
        """Deterministic analysis using MITRE mapper + summarizer."""
        # MITRE mapping
        mitre = self.mapper.map(attack_label)

        # Generate summary
        summary = self.summarizer.summarize_flow(flow_row, attack_label, company_id)

        # Build structured result
        in_bytes = flow_row.get("IN_BYTES", 0)
        out_bytes = flow_row.get("OUT_BYTES", 0)

        indicators = []
        dst_port = flow_row.get("L4_DST_PORT", 0)
        if dst_port:
            indicators.append("target_port:{}".format(dst_port))
        if in_bytes > 100000:
            indicators.append("high_volume_inbound:{}B".format(in_bytes))
        if flow_row.get("IN_PKTS", 0) > 100:
            indicators.append("high_packet_rate")

        return {
            "attack_type": attack_label,
            "mitre_technique_id": mitre["technique_id"],
            "mitre_tactic": mitre["tactic"],
            "severity": mitre["severity"],
            "confidence": 0.85,
            "attack_description": summary.get("summary", "Flagged as {}".format(attack_label)),
            "indicators": indicators,
            "recommended_defense": self._get_defense(attack_label),
            "source": "deterministic",
            "company_id": company_id,
        }

    @staticmethod
    def _get_defense(attack_label: str) -> str:
        """Get a default defense recommendation."""
        defenses = {
            "ddos": "Deploy rate limiting and SYN cookie protection",
            "dos": "Implement connection throttling and traffic shaping",
            "brute_force": "Enable account lockout and multi-factor authentication",
            "web_attack": "Deploy WAF rules for XSS/SQLi pattern filtering",
            "botnet": "Block C2 communication patterns and quarantine infected hosts",
            "infiltration": "Implement network segmentation and endpoint detection",
            "reconnaissance": "Deploy honeypots and limit ICMP/port scan responses",
            "theft": "Enable DLP controls and encrypt sensitive data at rest",
        }
        return defenses.get(attack_label, "Monitor and investigate flagged traffic")
