#!/usr/bin/env bash
# Download Unitree G1 robot assets and Replica Dataset scenes.
# Usage: bash scripts/setup_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

echo "=== Hide-and-Seek Robot: Data Setup ==="
echo "Project directory: $PROJECT_DIR"

# -----------------------------------------------
# 1. Unitree G1 robot description (URDF + meshes)
# -----------------------------------------------
G1_DIR="$DATA_DIR/robots/g1"
if [ -d "$G1_DIR/g1_description" ]; then
    echo "[SKIP] G1 robot description already exists at $G1_DIR"
else
    echo "[1/2] Downloading Unitree G1 robot description..."
    mkdir -p "$G1_DIR"
    TMP_DIR=$(mktemp -d)

    # Sparse checkout to only get g1_description (avoid downloading all robots)
    git clone --filter=blob:none --sparse --depth=1 \
        https://github.com/unitreerobotics/unitree_ros.git "$TMP_DIR/unitree_ros"
    cd "$TMP_DIR/unitree_ros"
    git sparse-checkout set robots/g1_description
    cp -r robots/g1_description "$G1_DIR/g1_description"

    rm -rf "$TMP_DIR"
    echo "[OK] G1 robot description saved to $G1_DIR/g1_description"
fi

# -----------------------------------------------
# 2. Replica Dataset (start with apartment_0)
# -----------------------------------------------
REPLICA_RAW="$DATA_DIR/scenes/replica_raw"
if [ -d "$REPLICA_RAW/apartment_0" ]; then
    echo "[SKIP] Replica apartment_0 already exists at $REPLICA_RAW"
else
    echo "[2/2] Downloading Replica Dataset (apartment_0)..."
    mkdir -p "$REPLICA_RAW"
    TMP_DIR=$(mktemp -d)

    git clone --depth=1 https://github.com/facebookresearch/Replica-Dataset.git "$TMP_DIR/Replica-Dataset"
    cd "$TMP_DIR/Replica-Dataset"

    # The download script requires the Replica EULA to be accepted.
    # It downloads scene data via URLs in the repo.
    if [ -f "./download.sh" ]; then
        echo "Running Replica download script for apartment_0..."
        bash ./download.sh apartment_0 "$REPLICA_RAW"
    else
        echo "[WARN] Replica download.sh not found."
        echo "Please manually download Replica scenes to: $REPLICA_RAW"
        echo "See: https://github.com/facebookresearch/Replica-Dataset"
    fi

    rm -rf "$TMP_DIR"
    echo "[OK] Replica scenes saved to $REPLICA_RAW"
fi

echo ""
echo "=== Data setup complete ==="
echo "G1 robot:  $G1_DIR/g1_description/"
echo "Replica:   $REPLICA_RAW/"
echo ""
echo "Next steps:"
echo "  1. Run 'python scripts/convert_replica.py' to convert scenes to USD"
echo "  2. Run 'python scripts/test_env.py' to verify the environment"
