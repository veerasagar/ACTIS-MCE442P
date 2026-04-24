#!/bin/bash
# ═══════════════════════════════════════════════════════════
# ACTIS — Run Script
# Runs all validation phases sequentially and prints results
#
# Usage:
#   bash scripts/run.sh           # Run all phases
#   bash scripts/run.sh 3         # Run only phase 3
#   bash scripts/run.sh 3 5 8     # Run phases 3, 5, and 8
# ═══════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VALIDATE_DIR="$SCRIPT_DIR/validate"

cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

run_phase() {
    local phase=$1
    local desc=$2
    local cmd=$3
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  Phase $phase — $desc${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo ""
    eval "$cmd"
    echo ""
    echo -e "${GREEN}  ✅ Phase $phase complete${NC}"
}

# Determine which phases to run
if [ $# -eq 0 ]; then
    PHASES="1 2 3 4 5 6 7 8 9 10 11 12"
else
    PHASES="$@"
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        🛡️  ACTIS — Validation Runner                 ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Phases: $PHASES"
echo "╚══════════════════════════════════════════════════════╝"

for phase in $PHASES; do
    case $phase in
        1)
            run_phase 1 "Data Pipeline + Feature Engineering" \
                "python3 $VALIDATE_DIR/data_pipeline.py && echo '' && python3 $VALIDATE_DIR/feature_engineering.py"
            ;;
        2)
            run_phase 2 "Local IDS (No Federation)" \
                "python3 $VALIDATE_DIR/local_ids.py"
            ;;
        3)
            run_phase 3 "Federated Learning (4 Nodes, 5 Rounds)" \
                "python3 $VALIDATE_DIR/federated_learning.py --rounds 5"
            ;;
        4)
            run_phase 4 "Privacy (PII + Differential Privacy)" \
                "python3 $VALIDATE_DIR/privacy.py"
            ;;
        10)
            run_phase 10 "Agents + PII Validator" \
                "python3 $VALIDATE_DIR/agents_privacy.py"
            ;;
        5)
            run_phase 5 "Zero-Day DDoS Detection (BoT-IoT)" \
                "python3 $VALIDATE_DIR/zeroday_ddos.py"
            ;;
        6)
            run_phase 6 "RAG Engine + Cross-Org Intelligence" \
                "python3 $VALIDATE_DIR/rag_engine.py"
            ;;
        11)
            run_phase 11 "MITRE ATT&CK Mapper" \
                "python3 $VALIDATE_DIR/mitre_mapper.py"
            ;;
        7)
            run_phase 7 "Report Quality (BERTScore)" \
                "python3 $VALIDATE_DIR/report_quality.py"
            ;;
        8)
            run_phase 8 "ReGAIN Benchmark Replication" \
                "python3 $VALIDATE_DIR/regain_benchmark.py"
            ;;
        9)
            run_phase 9 "Full Evaluation Suite" \
                "python3 -m src.evaluation.evaluate"
            ;;
        12)
            run_phase 12 "Live Monitor Demo" \
                "python3 $VALIDATE_DIR/live_monitor.py"
            ;;
        *)
            echo "Unknown phase: $phase (valid: 1-12)"
            ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        🎉  All requested phases complete!            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
