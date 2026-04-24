"""Phase 6b — MITRE ATT&CK Mapper validation."""
import sys; sys.path.insert(0, '.')
from src.data.mitre_mapper import MITREMapper

mapper = MITREMapper()

print('=== MITRE ATT&CK Mapping ===')
print()
attacks = ['ddos', 'dos', 'brute_force', 'web_attack', 'botnet',
           'infiltration', 'reconnaissance', 'theft']
print(f'  {"Attack":<18s} {"Technique":<10s} {"Tactic":<20s} {"Severity"}')
print(f'  {"-"*18} {"-"*10} {"-"*20} {"-"*8}')
for atk in attacks:
    m = mapper.map(atk)
    print(f'  {atk:<18s} {m["technique_id"]:<10s} {m["tactic"]:<20s} {m["severity"]}')

print()
mapper.print_mapping_table()
