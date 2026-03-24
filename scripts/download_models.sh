#!/usr/bin/env bash
# Download pre-trained nCPU model weights from GitHub releases.
# Usage: ./scripts/download_models.sh [target_dir]
#
# Default target: ~/.ncpu/models
set -euo pipefail

REPO="WispAyr/nCPU"
TAG="${NCPU_RELEASE_TAG:-latest}"
TARGET="${1:-$HOME/.ncpu/models}"
MODEL_DIRS="alu decode math memory os register shifts"

echo "==> Downloading nCPU models to $TARGET"
mkdir -p "$TARGET"

if command -v gh &>/dev/null; then
    echo "Using gh CLI..."
    if [ "$TAG" = "latest" ]; then
        gh release download --repo "$REPO" --pattern "models-*.tar.gz" --dir /tmp --clobber
    else
        gh release download "$TAG" --repo "$REPO" --pattern "models-*.tar.gz" --dir /tmp --clobber
    fi
    for f in /tmp/models-*.tar.gz; do
        tar xzf "$f" -C "$TARGET"
        rm "$f"
    done
else
    echo "Using curl..."
    if [ "$TAG" = "latest" ]; then
        URL="https://github.com/$REPO/releases/latest/download/models.tar.gz"
    else
        URL="https://github.com/$REPO/releases/download/$TAG/models.tar.gz"
    fi
    curl -fSL "$URL" | tar xz -C "$TARGET"
fi

echo "==> Models downloaded to $TARGET"
echo ""
echo "Set NCPU_PATH=$TARGET/.. or ensure models are at $TARGET/"
echo "Subdirectories expected: $MODEL_DIRS"

# Verify
missing=0
for d in $MODEL_DIRS; do
    if [ ! -d "$TARGET/$d" ]; then
        echo "WARNING: missing $TARGET/$d"
        missing=1
    fi
done
[ $missing -eq 0 ] && echo "==> All model directories present ✓"
