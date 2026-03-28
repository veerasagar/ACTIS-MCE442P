"""
ACTIS Feature Engineer
=======================
Computes temporal and protocol-aware derived features from raw NetFlow data.
Based on empirical analysis of CIC-IDS2018, these features expose
DDoS vs DoS distinctions invisible to raw byte/packet counts.

Key findings from data analysis (DDoS vs DoS medians):
  - TCP_FLAGS:          219 vs 27  (8× ratio) — strongest DDoS signal
  - TCP_WIN_MAX_IN:   65535 vs 26883           — full window vs partial
  - DST_TO_SRC_TP:   2.9× higher for DDoS     — response amplification
  - IN_BYTES/PKTS:    lower for DDoS           — many short flows

Usage:
    fe = FeatureEngineer()
    X_enhanced = fe.transform(X_raw, col_names)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from rich.console import Console

console = Console()

# ── Expected column indices in CIC-IDS2018 standard ordering ────────────────
# (Matches the 41 numeric cols extracted by Preprocessor)
COL_MAP = {
    "L4_SRC_PORT": 0,
    "L4_DST_PORT": 1,
    "PROTOCOL": 2,        # 6=TCP, 17=UDP, 1=ICMP
    "L7_PROTO": 3,
    "IN_BYTES": 4,
    "IN_PKTS": 5,
    "OUT_BYTES": 6,
    "OUT_PKTS": 7,
    "TCP_FLAGS": 8,
    "CLIENT_TCP_FLAGS": 9,
    "SERVER_TCP_FLAGS": 10,
    "FLOW_DURATION_MILLISECONDS": 11,
    "DURATION_IN": 12,
    "DURATION_OUT": 13,
    "MIN_TTL": 14,
    "MAX_TTL": 15,
    "LONGEST_FLOW_PKT": 16,
    "SHORTEST_FLOW_PKT": 17,
    "MIN_IP_PKT_LEN": 18,
    "MAX_IP_PKT_LEN": 19,
    "SRC_TO_DST_SECOND_BYTES": 20,
    "DST_TO_SRC_SECOND_BYTES": 21,
    "RETRANSMITTED_IN_BYTES": 22,
    "RETRANSMITTED_IN_PKTS": 23,
    "RETRANSMITTED_OUT_BYTES": 24,
    "RETRANSMITTED_OUT_PKTS": 25,
    "SRC_TO_DST_AVG_THROUGHPUT": 26,
    "DST_TO_SRC_AVG_THROUGHPUT": 27,
    # Packet-size histogram bins (cols 28-32)
    "NUM_PKTS_UP_TO_128_BYTES": 28,
    "NUM_PKTS_128_TO_256_BYTES": 29,
    "NUM_PKTS_256_TO_512_BYTES": 30,
    "NUM_PKTS_512_TO_1024_BYTES": 31,
    "NUM_PKTS_1024_TO_1514_BYTES": 32,
    "TCP_WIN_MAX_IN": 33,
    "TCP_WIN_MAX_OUT": 34,
}

EPS = 1e-8


class FeatureEngineer:
    """
    Computes derived temporal, protocol-aware, and DDoS-specific
    features from NetFlow data.

    Features added (25 total):
      === Temporal / byte-level (1-17) ===
      1.  bytes_per_pkt_in        — avg bytes per inbound packet
      2.  bytes_per_pkt_out       — avg bytes per outbound packet
      3.  pkt_size_ratio          — in_pkt_size / out_pkt_size
      4.  byte_asymmetry          — (IN - OUT) / (IN + OUT)
      5.  pkt_asymmetry           — (in_pkts - out_pkts) / total
      6.  throughput_ratio        — src→dst / dst→src throughput
      7.  duration_log            — log1p(flow_duration_ms)
      8.  byte_rate_in            — in_bytes / (dur_ms + 1)
      9.  byte_rate_out           — out_bytes / (dur_ms + 1)
      10. pkt_size_variance       — max_pkt_len - min_pkt_len
      11. is_udp                  — PROTOCOL == 17
      12. is_icmp                 — PROTOCOL == 1
      13. is_tcp                  — PROTOCOL == 6
      14. retransmit_ratio_in     — retrans_in / in_bytes
      15. retransmit_ratio_out    — retrans_out / out_bytes
      16. burst_score             — byte_rate_in / (1 + byte_rate_out)
      17. dst_wellknown           — L4_DST_PORT < 1024

      === DDoS-specific (18-25) — from empirical analysis ===
      18. tcp_flag_syn            — bit 1 of TCP_FLAGS (SYN)
      19. tcp_flag_ack            — bit 4 of TCP_FLAGS (ACK)
      20. tcp_flag_psh            — bit 3 of TCP_FLAGS (PSH)
      21. tcp_flag_fin            — bit 0 of TCP_FLAGS (FIN)
      22. tcp_flag_rst            — bit 2 of TCP_FLAGS (RST)
      23. tcp_flag_count          — popcount(TCP_FLAGS) / 6
          DDoS median=219 (many flags), DoS median=27 (few flags)
      24. tcp_win_ratio           — TCP_WIN_MAX_IN / (TCP_WIN_MAX_OUT+1)
          DDoS=65535 (max), DoS=26883 (partial)
      25. tp_dst_to_src_norm      — log1p(DST_TO_SRC_AVG_THROUGHPUT)
          DDoS 2.9× higher response throughput than DoS
    """

    N_DERIVED = 25  # 17 temporal + 8 DDoS-specific

    def __init__(self):
        self.col_map = COL_MAP

    def transform(
        self, X: np.ndarray, col_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Append derived features to the input feature matrix.

        Args:
            X: (n, 41) raw NetFlow features
            col_names: optional list of column names to override COL_MAP

        Returns:
            (n, 41 + N_DERIVED) enhanced feature matrix
        """
        # ── Build dynamic index map if col_names provided ────────────────
        if col_names:
            idx = {name.upper(): i for i, name in enumerate(col_names)}
        else:
            idx = {k.upper(): v for k, v in self.col_map.items()}

        def _get(name: str) -> np.ndarray:
            """Safe column getter — returns zeros if column not found."""
            col_upper = name.upper()
            if col_upper in idx:
                return X[:, idx[col_upper]].astype(np.float32)
            return np.zeros(len(X), dtype=np.float32)

        # ── Raw values ───────────────────────────────────────────────────
        in_bytes   = _get("IN_BYTES")
        out_bytes  = _get("OUT_BYTES")
        in_pkts    = _get("IN_PKTS")
        out_pkts   = _get("OUT_PKTS")
        dur_ms     = _get("FLOW_DURATION_MILLISECONDS")
        proto      = _get("PROTOCOL")
        max_pkt    = _get("MAX_IP_PKT_LEN")
        min_pkt    = _get("MIN_IP_PKT_LEN")
        ret_in_b   = _get("RETRANSMITTED_IN_BYTES")
        ret_out_b  = _get("RETRANSMITTED_OUT_BYTES")
        dst_port   = _get("L4_DST_PORT")
        s2d_tp     = _get("SRC_TO_DST_AVG_THROUGHPUT")
        d2s_tp     = _get("DST_TO_SRC_AVG_THROUGHPUT")
        tcp_flags  = _get("TCP_FLAGS").astype(np.int32)
        win_in     = _get("TCP_WIN_MAX_IN")
        win_out    = _get("TCP_WIN_MAX_OUT")

        # ── Temporal / byte-level derived (1-17) ─────────────────────────
        bpp_in             = in_bytes  / (in_pkts  + EPS)
        bpp_out            = out_bytes / (out_pkts + EPS)
        pkt_size_ratio     = bpp_in / (bpp_out + EPS)
        total_bytes        = in_bytes + out_bytes
        in_out_byte_asym   = (in_bytes - out_bytes) / (total_bytes + EPS)
        total_pkts         = in_pkts + out_pkts
        in_out_pkt_asym    = (in_pkts - out_pkts) / (total_pkts + EPS)
        tp_ratio           = s2d_tp / (d2s_tp + EPS)
        dur_norm           = np.log1p(np.abs(dur_ms))
        byte_rate_in       = in_bytes  / (dur_ms + 1)
        byte_rate_out      = out_bytes / (dur_ms + 1)
        pkt_size_var       = max_pkt - min_pkt
        is_udp             = (proto == 17).astype(np.float32)
        is_icmp            = (proto == 1).astype(np.float32)
        is_tcp             = (proto == 6).astype(np.float32)
        retrans_ratio_in   = ret_in_b  / (in_bytes  + EPS)
        retrans_ratio_out  = ret_out_b / (out_bytes + EPS)
        burst_score        = byte_rate_in / (1 + byte_rate_out)
        dst_wellknown      = (dst_port < 1024).astype(np.float32)

        # ── DDoS-specific derived (18-25) ─────────────────────────────────
        # TCP_FLAGS bit decomposition — DDoS median=219 (many flags), DoS=27
        flag_syn  = ((tcp_flags & 0x02) > 0).astype(np.float32)  # bit 1
        flag_ack  = ((tcp_flags & 0x10) > 0).astype(np.float32)  # bit 4
        flag_psh  = ((tcp_flags & 0x08) > 0).astype(np.float32)  # bit 3
        flag_fin  = ((tcp_flags & 0x01) > 0).astype(np.float32)  # bit 0
        flag_rst  = ((tcp_flags & 0x04) > 0).astype(np.float32)  # bit 2
        # popcount: number of set bits / 6 — DDoS sets many bits simultaneously
        flag_count = np.array(
            [bin(int(f)).count('1') for f in tcp_flags], dtype=np.float32
        ) / 6.0
        # TCP window ratio — DDoS has max window (65535) vs DoS partial (26883)
        tcp_win_ratio = win_in / (win_out + EPS)
        # Response throughput (log) — DDoS 2.9× higher than DoS
        tp_dst_src_norm = np.log1p(np.abs(d2s_tp))

        derived = np.column_stack([
            # Temporal (1-17)
            bpp_in, bpp_out, pkt_size_ratio,
            in_out_byte_asym, in_out_pkt_asym,
            tp_ratio, dur_norm,
            byte_rate_in, byte_rate_out,
            pkt_size_var,
            is_udp, is_icmp, is_tcp,
            retrans_ratio_in, retrans_ratio_out,
            burst_score, dst_wellknown,
            # DDoS-specific (18-25)
            flag_syn, flag_ack, flag_psh, flag_fin, flag_rst,
            flag_count, tcp_win_ratio, tp_dst_src_norm,
        ])

        # Clip extreme values in derived features
        derived = np.clip(np.nan_to_num(derived, nan=0, posinf=1e6, neginf=-1e6), -1e6, 1e6)

        return np.concatenate([X, derived.astype(np.float32)], axis=1)

    @property
    def output_dim(self) -> int:
        """Total feature dimension after transformation."""
        return 41 + self.N_DERIVED

    @staticmethod
    def feature_names() -> List[str]:
        """Names of all derived features added."""
        return [
            # Temporal (1-17)
            "bytes_per_pkt_in", "bytes_per_pkt_out", "pkt_size_ratio",
            "byte_asymmetry", "pkt_asymmetry", "throughput_ratio",
            "duration_log", "byte_rate_in", "byte_rate_out",
            "pkt_size_variance", "is_udp", "is_icmp", "is_tcp",
            "retransmit_ratio_in", "retransmit_ratio_out",
            "burst_score", "dst_wellknown",
            # DDoS-specific (18-25)
            "tcp_flag_syn", "tcp_flag_ack", "tcp_flag_psh",
            "tcp_flag_fin", "tcp_flag_rst",
            "tcp_flag_count", "tcp_win_ratio", "tp_dst_src_norm",
        ]
