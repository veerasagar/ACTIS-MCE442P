"""Phase 9 — Live Monitor demo (processes synthetic flows)."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from src.live_monitor import LiveMonitor

print('=== Live Monitor Demo ===')
print()

monitor = LiveMonitor(company_id='A')

# Create synthetic test flows
test_flows = [
    {'PROTOCOL': 17, 'L4_SRC_PORT': 1234, 'L4_DST_PORT': 53,
     'IN_BYTES': 5000000, 'OUT_BYTES': 200, 'IN_PKTS': 10000,
     'OUT_PKTS': 5, 'FLOW_DURATION_MILLISECONDS': 100, 'TCP_FLAGS': 0},
    {'PROTOCOL': 6, 'L4_SRC_PORT': 4444, 'L4_DST_PORT': 22,
     'IN_BYTES': 500, 'OUT_BYTES': 500, 'IN_PKTS': 20,
     'OUT_PKTS': 20, 'FLOW_DURATION_MILLISECONDS': 5000, 'TCP_FLAGS': 27},
    {'PROTOCOL': 6, 'L4_SRC_PORT': 8080, 'L4_DST_PORT': 80,
     'IN_BYTES': 100, 'OUT_BYTES': 50000, 'IN_PKTS': 2,
     'OUT_PKTS': 100, 'FLOW_DURATION_MILLISECONDS': 200, 'TCP_FLAGS': 219},
]

print(f'Processing {len(test_flows)} synthetic flows...')
print()
for i, flow in enumerate(test_flows):
    alert = monitor.process_flow(flow)
    if alert:
        print(f'  Flow {i+1}: [{alert["severity"]}] {alert["predicted_attack"].upper()} '
              f'(conf={alert["confidence"]:.0%}) MITRE={alert["mitre"]} '
              f'zero_day={alert["is_zero_day"]}')
    else:
        print(f'  Flow {i+1}: Benign (no alert)')

print()
monitor._print_summary()
