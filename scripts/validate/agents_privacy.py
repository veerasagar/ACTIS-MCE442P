"""Phase 4b — Agent Analyzer + Agent Sanitizer + PII Validator."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
from src.local_node.agent_analyzer import AgentAnalyzer
from src.local_node.agent_sanitizer import AgentSanitizer
from src.local_node.pii_validator import PIIValidator

flow = {'PROTOCOL': 17, 'L4_SRC_PORT': 1234, 'L4_DST_PORT': 53,
        'IN_BYTES': 5000000, 'OUT_BYTES': 200, 'IN_PKTS': 10000,
        'OUT_PKTS': 5, 'FLOW_DURATION_MILLISECONDS': 100, 'TCP_FLAGS': 0}

# Agent Analyzer
print('=== Agent Analyzer ===')
analyzer = AgentAnalyzer()
analysis = analyzer.analyze(flow, 'ddos', 'A')
print(f'  Source:     {analysis["source"]}')
print(f'  Attack:     {analysis["attack_type"]}')
print(f'  MITRE:      {analysis["mitre_technique_id"]} ({analysis["mitre_tactic"]})')
print(f'  Severity:   {analysis["severity"]}')
print(f'  Confidence: {analysis["confidence"]}')
print(f'  Defense:    {analysis["recommended_defense"]}')
print(f'  Indicators: {analysis["indicators"]}')

# Agent Sanitizer
print()
print('=== Agent Sanitizer ===')
sanitizer = AgentSanitizer()
dirty = dict(analysis)
dirty['attack_description'] = 'DDoS from 192.168.1.100 to server admin@corp.com'
sanitized = sanitizer.sanitize(dirty)
print(f'  Before: {dirty["attack_description"]}')
print(f'  After:  {sanitized["attack_description"]}')
stats = sanitizer.get_stats()
print(f'  Sanitized: {stats["sanitized"]}  Scrubbed: {stats["pii_scrubbed"]}')

# PII Validator
print()
print('=== PII Validator ===')
validator = PIIValidator()
is_clean, findings = validator.validate(sanitized)
print(f'  Clean: {is_clean}  Findings: {len(findings)}')
dirty_test = {'desc': 'Attack from 10.0.0.5 user john@evil.com'}
is_clean2, findings2 = validator.validate(dirty_test)
print(f'  Dirty test clean: {is_clean2}  PII found: {len(findings2)}')
for f in findings2:
    print(f'    {f["pii_type"]}: {f["match"]}')
v_stats = validator.get_stats()
print(f'  Leakage rate: {v_stats["leakage_rate"]:.1f}%')
