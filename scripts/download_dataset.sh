#!/bin/bash
# Fed-Intel Dataset Download Script
# Downloads both NF-CSE-CIC-IDS2018-v3 and Gotham 2025 from Kaggle
#
# Prerequisites: pip install kaggle
# Set KAGGLE_USERNAME and KAGGLE_KEY in ~/.kaggle/kaggle.json

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

echo "╔══════════════════════════════════════════════════╗"
echo "║        🛡️  Fed-Intel Dataset Downloader           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# --- Dataset 1: NF-CSE-CIC-IDS2018-v3 ---
IDS2018_DIR="$DATA_DIR/nf-cse-cic-ids2018-v3"
if [ -z "$(ls -A "$IDS2018_DIR" 2>/dev/null)" ]; then
    echo "📥 Downloading NF-CSE-CIC-IDS2018-v3..."
    kaggle datasets download -d dhoogla/nfcsecicids2018v3 -p "$IDS2018_DIR" --unzip
    echo "✅ NF-CSE-CIC-IDS2018-v3 downloaded to $IDS2018_DIR"
else
    echo "✅ NF-CSE-CIC-IDS2018-v3 already exists, skipping."
fi

echo ""

# --- Dataset 2: Gotham 2025 ---
GOTHAM_DIR="$DATA_DIR/gotham-2025"
if [ -z "$(ls -A "$GOTHAM_DIR" 2>/dev/null)" ]; then
    echo "📥 Downloading Gotham 2025..."
    kaggle datasets download -d emilymuller/gotham-network-intrusion-detection-dataset -p "$GOTHAM_DIR" --unzip
    echo "✅ Gotham 2025 downloaded to $GOTHAM_DIR"
else
    echo "✅ Gotham 2025 already exists, skipping."
fi

echo ""
echo "📊 Dataset sizes:"
du -sh "$IDS2018_DIR" 2>/dev/null || echo "  NF-CSE-CIC-IDS2018-v3: not found"
du -sh "$GOTHAM_DIR" 2>/dev/null || echo "  Gotham 2025: not found"
echo ""
echo "🎉 All datasets ready!"
