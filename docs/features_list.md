# ACTIS — 66 Feature List

> 41 Raw NetFlow features + 25 Derived features computed by `src/data/feature_engineer.py`

---

## Raw Features (1–41)

These are the numeric columns extracted from the CIC-IDS2018-v2 and BoT-IoT-v2 Parquet datasets by the `Preprocessor`.

| # | Feature Name | Description |
|---|-------------|-------------|
| 1 | `L4_SRC_PORT` | Layer-4 source port |
| 2 | `L4_DST_PORT` | Layer-4 destination port |
| 3 | `PROTOCOL` | IP protocol number (6=TCP, 17=UDP, 1=ICMP) |
| 4 | `L7_PROTO` | Layer-7 application protocol ID |
| 5 | `IN_BYTES` | Incoming byte count |
| 6 | `IN_PKTS` | Incoming packet count |
| 7 | `OUT_BYTES` | Outgoing byte count |
| 8 | `OUT_PKTS` | Outgoing packet count |
| 9 | `TCP_FLAGS` | Cumulative TCP flags (bitmask) |
| 10 | `CLIENT_TCP_FLAGS` | Client-side TCP flags |
| 11 | `SERVER_TCP_FLAGS` | Server-side TCP flags |
| 12 | `FLOW_DURATION_MILLISECONDS` | Total flow duration in ms |
| 13 | `DURATION_IN` | Inbound duration |
| 14 | `DURATION_OUT` | Outbound duration |
| 15 | `MIN_TTL` | Minimum time-to-live |
| 16 | `MAX_TTL` | Maximum time-to-live |
| 17 | `LONGEST_FLOW_PKT` | Longest packet in the flow |
| 18 | `SHORTEST_FLOW_PKT` | Shortest packet in the flow |
| 19 | `MIN_IP_PKT_LEN` | Minimum IP packet length |
| 20 | `MAX_IP_PKT_LEN` | Maximum IP packet length |
| 21 | `SRC_TO_DST_SECOND_BYTES` | Source-to-destination bytes per second |
| 22 | `DST_TO_SRC_SECOND_BYTES` | Destination-to-source bytes per second |
| 23 | `RETRANSMITTED_IN_BYTES` | Retransmitted inbound bytes |
| 24 | `RETRANSMITTED_IN_PKTS` | Retransmitted inbound packets |
| 25 | `RETRANSMITTED_OUT_BYTES` | Retransmitted outbound bytes |
| 26 | `RETRANSMITTED_OUT_PKTS` | Retransmitted outbound packets |
| 27 | `SRC_TO_DST_AVG_THROUGHPUT` | Average throughput (source → destination) |
| 28 | `DST_TO_SRC_AVG_THROUGHPUT` | Average throughput (destination → source) |
| 29 | `NUM_PKTS_UP_TO_128_BYTES` | Packet count: ≤128 bytes |
| 30 | `NUM_PKTS_128_TO_256_BYTES` | Packet count: 128–256 bytes |
| 31 | `NUM_PKTS_256_TO_512_BYTES` | Packet count: 256–512 bytes |
| 32 | `NUM_PKTS_512_TO_1024_BYTES` | Packet count: 512–1024 bytes |
| 33 | `NUM_PKTS_1024_TO_1514_BYTES` | Packet count: 1024–1514 bytes |
| 34 | `TCP_WIN_MAX_IN` | Maximum TCP window size (inbound) |
| 35 | `TCP_WIN_MAX_OUT` | Maximum TCP window size (outbound) |
| 36 | `ICMP_TYPE` | ICMP message type |
| 37 | `ICMP_IPV4_TYPE` | ICMP IPv4 type code |
| 38 | `DNS_QUERY_ID` | DNS query identifier |
| 39 | `DNS_QUERY_TYPE` | DNS query type |
| 40 | `DNS_TTL_ANSWER` | DNS answer TTL |
| 41 | `FTP_COMMAND_RET_CODE` | FTP command return code |

> **Note:** Features 36–41 are the remaining numeric columns from the Parquet schema selected at runtime by `Preprocessor.get_feature_columns()`. The exact names may vary depending on dataset version; the above are the standard NF-v2 column names.

---

## Derived Features (42–66)

Computed by `FeatureEngineer.transform()` and appended to the raw feature matrix.

### Temporal / Byte-Level (42–58)

| # | Feature Name | Formula | Purpose |
|---|-------------|---------|---------|
| 42 | `bytes_per_pkt_in` | `IN_BYTES / IN_PKTS` | Avg inbound packet size |
| 43 | `bytes_per_pkt_out` | `OUT_BYTES / OUT_PKTS` | Avg outbound packet size |
| 44 | `pkt_size_ratio` | `bpp_in / bpp_out` | In/out packet size ratio |
| 45 | `byte_asymmetry` | `(IN_BYTES − OUT_BYTES) / (IN_BYTES + OUT_BYTES)` | Directional byte imbalance |
| 46 | `pkt_asymmetry` | `(IN_PKTS − OUT_PKTS) / (IN_PKTS + OUT_PKTS)` | Directional packet imbalance |
| 47 | `throughput_ratio` | `SRC_TO_DST_AVG_THROUGHPUT / DST_TO_SRC_AVG_THROUGHPUT` | Throughput directionality |
| 48 | `duration_log` | `log1p(FLOW_DURATION_MILLISECONDS)` | Log-scaled flow duration |
| 49 | `byte_rate_in` | `IN_BYTES / (FLOW_DURATION_MILLISECONDS + 1)` | Inbound byte rate |
| 50 | `byte_rate_out` | `OUT_BYTES / (FLOW_DURATION_MILLISECONDS + 1)` | Outbound byte rate |
| 51 | `pkt_size_variance` | `MAX_IP_PKT_LEN − MIN_IP_PKT_LEN` | Packet size spread |
| 52 | `is_udp` | `PROTOCOL == 17` | UDP protocol indicator |
| 53 | `is_icmp` | `PROTOCOL == 1` | ICMP protocol indicator |
| 54 | `is_tcp` | `PROTOCOL == 6` | TCP protocol indicator |
| 55 | `retransmit_ratio_in` | `RETRANSMITTED_IN_BYTES / IN_BYTES` | Inbound retransmission rate |
| 56 | `retransmit_ratio_out` | `RETRANSMITTED_OUT_BYTES / OUT_BYTES` | Outbound retransmission rate |
| 57 | `burst_score` | `byte_rate_in / (1 + byte_rate_out)` | Burst traffic indicator |
| 58 | `dst_wellknown` | `L4_DST_PORT < 1024` | Well-known port flag |

### DDoS-Specific (59–66)

Based on empirical analysis — DDoS has `TCP_FLAGS` median of 219 (many flags set) vs DoS median of 27 (few flags).

| # | Feature Name | Formula | Purpose |
|---|-------------|---------|---------|
| 59 | `tcp_flag_syn` | `(TCP_FLAGS & 0x02) > 0` | SYN flag present (bit 1) |
| 60 | `tcp_flag_ack` | `(TCP_FLAGS & 0x10) > 0` | ACK flag present (bit 4) |
| 61 | `tcp_flag_psh` | `(TCP_FLAGS & 0x08) > 0` | PSH flag present (bit 3) |
| 62 | `tcp_flag_fin` | `(TCP_FLAGS & 0x01) > 0` | FIN flag present (bit 0) |
| 63 | `tcp_flag_rst` | `(TCP_FLAGS & 0x04) > 0` | RST flag present (bit 2) |
| 64 | `tcp_flag_count` | `popcount(TCP_FLAGS) / 6` | Normalized flag bit count |
| 65 | `tcp_win_ratio` | `TCP_WIN_MAX_IN / TCP_WIN_MAX_OUT` | TCP window size ratio |
| 66 | `tp_dst_src_norm` | `log1p(DST_TO_SRC_AVG_THROUGHPUT)` | Log-normalized response throughput |

---

## Summary

| Category | Count | Range |
|----------|-------|-------|
| Raw NetFlow features | 41 | 1–41 |
| Derived temporal / byte-level | 17 | 42–58 |
| Derived DDoS-specific | 8 | 59–66 |
| **Total** | **66** | **1–66** |
