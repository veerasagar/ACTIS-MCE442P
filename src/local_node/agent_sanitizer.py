"""
Fed-Intel Agent Sanitizer
==========================
LLM-powered agent that strips PII from threat analysis results.
Removes IP addresses, MAC addresses, hostnames, usernames,
and raw payloads while preserving actionable threat intelligence.

When a Gemini API key is available, uses LLM for context-aware sanitization.
Falls back to regex-based stripping otherwise.

Usage:
    sanitizer = AgentSanitizer()
    clean = sanitizer.sanitize(analysis_result)
"""

import re
import json
from pathlib import Path
from typing import Dict, List
from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import GEMINI_API_KEY, LLM_PRIMARY_MODEL, LLM_TEMPERATURE

console = Console()

# ─── PII Patterns ───────────────────────────────────────────────────────────

PII_PATTERNS = {
    "ipv4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    "ipv6": re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"),
    "mac": re.compile(r"\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "hostname": re.compile(
        r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)"
        r"+(?:com|net|org|io|edu|gov|mil|co|us|uk|de|fr|jp|cn|ru|br|in)\b"
    ),
    "filepath": re.compile(r"(?:/[a-zA-Z0-9._-]+){3,}"),
    "username_pattern": re.compile(r"\buser(?:name)?[=:]\s*\S+", re.IGNORECASE),
    "password_pattern": re.compile(r"\bpass(?:word)?[=:]\s*\S+", re.IGNORECASE),
}

# Replacement tokens
PII_REPLACEMENTS = {
    "ipv4": "[REDACTED_IP]",
    "ipv6": "[REDACTED_IP]",
    "mac": "[REDACTED_MAC]",
    "email": "[REDACTED_EMAIL]",
    "hostname": "[REDACTED_HOST]",
    "filepath": "[REDACTED_PATH]",
    "username_pattern": "[REDACTED_CRED]",
    "password_pattern": "[REDACTED_CRED]",
}

SANITIZER_SYSTEM_PROMPT = """You are a privacy sanitization agent for a federated threat intelligence network.
Your task: remove ALL personally identifiable information (PII) from threat analysis data.

STRICT RULES:
1. Replace ALL IP addresses with [REDACTED_IP]
2. Replace ALL MAC addresses with [REDACTED_MAC]  
3. Replace ALL hostnames/domains with [REDACTED_HOST]
4. Replace ALL email addresses with [REDACTED_EMAIL]
5. Replace ALL usernames/passwords with [REDACTED_CRED]
6. Replace ALL file paths with [REDACTED_PATH]
7. Keep attack descriptions, MITRE IDs, severity, and defense recommendations INTACT
8. Output ONLY valid JSON — the sanitized version of the input"""


class AgentSanitizer:
    """
    Privacy sanitization agent.
    Strips PII from threat analysis results before federation.
    """

    def __init__(self):
        self._llm = None
        self.stats = {"total": 0, "pii_found": 0, "pii_removed": 0}

        if GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                self._llm = genai.GenerativeModel(
                    LLM_PRIMARY_MODEL,
                    generation_config={"temperature": 0.0},  # Deterministic for safety
                )
                console.print("  [green]Sanitizer: Gemini LLM loaded[/green]")
            except Exception as e:
                console.print(
                    "  [yellow]Sanitizer: LLM init failed ({}), using regex mode[/yellow]".format(e)
                )
        else:
            console.print("  [dim]Sanitizer: No API key — regex mode[/dim]")

    def sanitize(self, analysis: Dict) -> Dict:
        """
        Sanitize a threat analysis result by removing all PII.

        Args:
            analysis: dict from AgentAnalyzer.analyze()

        Returns:
            Sanitized dict with PII replaced by tokens
        """
        self.stats["total"] += 1

        if self._llm:
            result = self._llm_sanitize(analysis)
        else:
            result = self._regex_sanitize(analysis)

        return result

    def _llm_sanitize(self, analysis: Dict) -> Dict:
        """Use LLM for context-aware sanitization."""
        try:
            prompt = "Sanitize this threat analysis JSON. Remove ALL PII:\n\n{}".format(
                json.dumps(analysis, indent=2)
            )
            response = self._llm.generate_content(
                [SANITIZER_SYSTEM_PROMPT, prompt]
            )

            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(text)

            # Double-check with regex (defense in depth)
            result = self._regex_sanitize(result)
            return result

        except Exception as e:
            console.print("  [yellow]LLM sanitize failed: {}, using regex[/yellow]".format(e))
            return self._regex_sanitize(analysis)

    def _regex_sanitize(self, analysis: Dict) -> Dict:
        """Regex-based PII stripping — applied to all string values."""
        sanitized = {}
        for key, value in analysis.items():
            if isinstance(value, str):
                sanitized[key] = self._strip_pii_from_string(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._strip_pii_from_string(v) if isinstance(v, str) else v
                    for v in value
                ]
            elif isinstance(value, dict):
                sanitized[key] = self._regex_sanitize(value)
            else:
                sanitized[key] = value

        return sanitized

    def _strip_pii_from_string(self, text: str) -> str:
        """Apply all PII regex patterns to a string."""
        for pii_type, pattern in PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                self.stats["pii_found"] += len(matches)
                self.stats["pii_removed"] += len(matches)
                text = pattern.sub(PII_REPLACEMENTS[pii_type], text)
        return text

    def get_stats(self) -> Dict:
        """Return sanitization statistics."""
        return dict(self.stats)

    def reset_stats(self):
        """Reset counters."""
        self.stats = {"total": 0, "pii_found": 0, "pii_removed": 0}
