"""
Fed-Intel PII Validator
========================
Regex-based validation gate that checks if ANY PII remains
in sanitized threat summaries before they leave the node.

This is the last line of defense — if the sanitizer missed anything,
the validator catches it and blocks the summary from being shared.

Usage:
    validator = PIIValidator()
    is_clean, findings = validator.validate(sanitized_dict)
"""

import re
from typing import Dict, Tuple, List
from rich.console import Console

console = Console()

# ─── PII Detection Patterns ────────────────────────────────────────────────

PII_DETECTORS = {
    "ipv4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    "ipv6": re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"),
    "mac": re.compile(r"\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "private_ip": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
        r"192\.168\.\d{1,3}\.\d{1,3})\b"
    ),
    "hostname": re.compile(
        r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)"
        r"+(?:com|net|org|io|edu|gov|mil|co|us|uk|de|fr|jp|cn|ru|br|in)\b"
    ),
    "filepath_unix": re.compile(r"(?:/[a-zA-Z0-9._-]+){3,}"),
    "filepath_win": re.compile(r"[A-Z]:\\(?:[a-zA-Z0-9._-]+\\){2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
}

# Patterns that should be ignored (our own redaction tokens)
REDACTION_TOKENS = re.compile(r"\[REDACTED_\w+\]")


class PIIValidator:
    """
    Validates that sanitized data contains no PII.
    Acts as the final gate before data leaves the node.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.stats = {"validated": 0, "passed": 0, "failed": 0, "pii_found": []}

    def validate(self, data: Dict) -> Tuple[bool, List[Dict]]:
        """
        Validate a sanitized dict for remaining PII.

        Args:
            data: sanitized analysis dict

        Returns:
            (is_clean, findings) where findings is a list of
            {"field": ..., "pii_type": ..., "match": ...}
        """
        self.stats["validated"] += 1
        findings = []

        self._scan_dict(data, findings, prefix="")

        if findings:
            self.stats["failed"] += 1
            self.stats["pii_found"].extend(findings)
        else:
            self.stats["passed"] += 1

        return len(findings) == 0, findings

    def _scan_dict(self, data: Dict, findings: List, prefix: str):
        """Recursively scan dict for PII."""
        for key, value in data.items():
            field_path = "{}.{}".format(prefix, key) if prefix else key

            if isinstance(value, str):
                self._scan_string(value, field_path, findings)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        self._scan_string(
                            item, "{}[{}]".format(field_path, i), findings
                        )
                    elif isinstance(item, dict):
                        self._scan_dict(
                            item, findings,
                            prefix="{}[{}]".format(field_path, i)
                        )
            elif isinstance(value, dict):
                self._scan_dict(value, findings, prefix=field_path)

    def _scan_string(self, text: str, field: str, findings: List):
        """Scan a string for PII patterns."""
        # Remove redaction tokens before scanning
        clean_text = REDACTION_TOKENS.sub("", text)

        for pii_type, pattern in PII_DETECTORS.items():
            matches = pattern.findall(clean_text)
            for match in matches:
                findings.append({
                    "field": field,
                    "pii_type": pii_type,
                    "match": match,
                })

    def get_stats(self) -> Dict:
        """Return validation statistics."""
        return {
            "validated": self.stats["validated"],
            "passed": self.stats["passed"],
            "failed": self.stats["failed"],
            "leakage_rate": (
                self.stats["failed"] / max(self.stats["validated"], 1) * 100
            ),
        }

    def reset_stats(self):
        """Reset counters."""
        self.stats = {"validated": 0, "passed": 0, "failed": 0, "pii_found": []}
