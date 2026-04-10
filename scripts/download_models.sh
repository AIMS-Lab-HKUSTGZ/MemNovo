#!/bin/bash
#
# Download pre-trained model checkpoints for MemNovo
#
# This script downloads:
# - InstaNovo v1.1.0 checkpoint
# - Casanovo v5.0.0 checkpoint
#
# PrimeNovo currently uses a separate checkpoint (`model_massive.ckpt`) that is
# expected to be placed manually under `weights/`.
#
# Usage:
#   bash scripts/download_models.sh
#

set -e

# Create weights directory
WEIGHTS_DIR="weights"
mkdir -p "$WEIGHTS_DIR"

echo "========================================="
echo "MemNovo Model Downloader"
echo "========================================="
echo ""

# InstaNovo checkpoint
INSTANOVO_URL="https://github.com/instadeepai/InstaNovo/releases/download/v1.1.0/instanovo_yeast.pt"
INSTANOVO_FILE="$WEIGHTS_DIR/instanovo-v1.1.0.ckpt"

if [ -f "$INSTANOVO_FILE" ]; then
    echo "[OK] InstaNovo checkpoint already exists: $INSTANOVO_FILE"
else
    echo "[INFO] Downloading InstaNovo v1.1.0 checkpoint..."
    echo "       URL: $INSTANOVO_URL"

    # Try with wget first, then curl
    if command -v wget &> /dev/null; then
        wget -O "$INSTANOVO_FILE" "$INSTANOVO_URL" || {
            echo "[WARN] wget failed, trying curl..."
            curl -L -o "$INSTANOVO_FILE" "$INSTANOVO_URL"
        }
    else
        curl -L -o "$INSTANOVO_FILE" "$INSTANOVO_URL"
    fi

    if [ -f "$INSTANOVO_FILE" ]; then
        echo "[OK] Downloaded InstaNovo checkpoint"
    else
        echo "[ERROR] Failed to download InstaNovo checkpoint"
        echo ""
        echo "Please download manually from:"
        echo "  $INSTANOVO_URL"
        echo ""
        echo "And save to: $INSTANOVO_FILE"
    fi
fi

echo ""

# Casanovo checkpoint
CASANOVO_URL="https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_pretrained_weights.ckpt"
CASANOVO_FILE="$WEIGHTS_DIR/casanovo_v5_0_0.ckpt"

if [ -f "$CASANOVO_FILE" ]; then
    echo "[OK] Casanovo checkpoint already exists: $CASANOVO_FILE"
else
    echo "[INFO] Downloading Casanovo v5.0.0 checkpoint..."
    echo "       URL: $CASANOVO_URL"

    if command -v wget &> /dev/null; then
        wget -O "$CASANOVO_FILE" "$CASANOVO_URL" || {
            echo "[WARN] wget failed, trying curl..."
            curl -L -o "$CASANOVO_FILE" "$CASANOVO_URL"
        }
    else
        curl -L -o "$CASANOVO_FILE" "$CASANOVO_URL"
    fi

    if [ -f "$CASANOVO_FILE" ]; then
        echo "[OK] Downloaded Casanovo checkpoint"
    else
        echo "[ERROR] Failed to download Casanovo checkpoint"
        echo ""
        echo "Please download manually from:"
        echo "  $CASANOVO_URL"
        echo ""
        echo "And save to: $CASANOVO_FILE"
    fi
fi

echo ""
echo "========================================="
echo "Download complete!"
echo ""
echo "Checkpoints saved to:"
echo "  - $INSTANOVO_FILE"
echo "  - $CASANOVO_FILE"
echo ""
echo "PrimeNovo note:"
echo "  - place model_massive.ckpt at weights/model_massive.ckpt manually"
echo ""
echo "You can now run MemNovo with:"
echo "  python scripts/run_inference.py --config configs/memnovo.yaml --input <your_data.mgf>"
echo "========================================="
