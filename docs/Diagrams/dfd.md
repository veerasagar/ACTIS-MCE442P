# ACTIS Data Flow Diagram

```text
[Raw Network Traffic]
           │
           ▼
 ┌──────────────────────────┐
 │                          │
 │  1. Local Detection      │  ──(Local Model Weights)──> [Federated Learning]
 │  (Scans for Anomalies)   │
 │                          │<──(Global Model Weights)──  [Federated Learning]
 └──────────────────────────┘
           │
    (Raw Threat Event)
           │
           ▼
 ┌──────────────────────────┐
 │                          │
 │  2. Privacy Filter       │
 │  (Strips PII & Maps Data)│
 │                          │
 └──────────────────────────┘
           │
  (Sanitized Threat Report)
           │
           ▼
 ┌──────────────────────────┐
 │                          │
 │  3. Global Intel Server  │  <══(Cross-Check)══> [Global Threat Database]
 │  (Builds Immunity Rules) │
 │                          │
 └──────────────────────────┘
           │
(New Preventative Rules)
           │
           ▼
   [Local Firewalls]
```
