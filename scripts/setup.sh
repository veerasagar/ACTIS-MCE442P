#!/bin/bash
# ═══════════════════════════════════════════════════════════
# ACTIS — Setup Script
# Installs all dependencies and downloads both datasets
#
# Usage:  bash scripts/setup.sh
# ═══════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

cd "$PROJECT_ROOT"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        🛡️  ACTIS — Setup                             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Install dependencies ────────────────────────────────
echo "📦 Installing Python dependencies..."
python3 -m pip install -r requirements.txt --quiet
python3 -m pip install bert-score sentence-transformers --quiet
echo "✅ All dependencies installed"
echo ""

# ── 2. Create directories ──────────────────────────────────
mkdir -p models results data chromadb_store
echo "✅ Created directories: models/ results/ data/ chromadb_store/"
echo ""

# ── 3. Download datasets ───────────────────────────────────
IDS2018_DIR="$DATA_DIR/nf-cse-cic-ids2018-v2"
if [ -n "$(ls -A "$IDS2018_DIR" 2>/dev/null)" ]; then
    echo "✅ NF-CSE-CIC-IDS2018-v2 already exists, skipping."
else
    echo "📥 Downloading NF-CSE-CIC-IDS2018-v2..."
    mkdir -p "$IDS2018_DIR"
    kaggle datasets download -d dhoogla/nfcsecicids2018v2 -p "$IDS2018_DIR" --unzip
    echo "✅ Downloaded to $IDS2018_DIR"
fi

BOTIOT_DIR="$DATA_DIR/nf-bot-iot-v2"
if [ -n "$(ls -A "$BOTIOT_DIR" 2>/dev/null)" ]; then
    echo "✅ NF-BoT-IoT-v2 already exists, skipping."
else
    echo "📥 Downloading NF-BoT-IoT-v2..."
    mkdir -p "$BOTIOT_DIR"
    kaggle datasets download -d dhoogla/nfbotiotv2 -p "$BOTIOT_DIR" --unzip
    echo "✅ Downloaded to $BOTIOT_DIR"
fi

echo ""
echo "📊 Dataset sizes:"
du -sh "$IDS2018_DIR" 2>/dev/null || echo "  NF-CSE-CIC-IDS2018-v2: not found"
du -sh "$BOTIOT_DIR" 2>/dev/null || echo "  NF-BoT-IoT-v2: not found"

# ── 4. Verify installation ─────────────────────────────────
echo ""
echo "🔍 Verifying imports..."
python3 -c "
import torch, sklearn, pandas, numpy, chromadb, rich
print(f'  torch:         {torch.__version__}')
print(f'  scikit-learn:  {sklearn.__version__}')
print(f'  pandas:        {pandas.__version__}')
print(f'  numpy:         {numpy.__version__}')
try:
    import bert_score; print(f'  bert-score:    {bert_score.__version__}')
except: print('  bert-score:    not installed')
"

echo ""
echo "🎉 Setup complete! Run: bash scripts/run.sh"
