#!/bin/bash
# Hailo-8 Compilation Script for nCPU Models
# Run on NOC Pi (192.168.195.238) with Hailo AI Software Suite installed
#
# Prerequisites:
#   pip install hailo_sdk_client hailo_sdk_common
#   apt install hailo-firmware hailo-pcie-driver
#
# Usage: ./compile_hailo.sh [model_name]

set -euo pipefail

ONNX_DIR="${ONNX_DIR:-./onnx}"
HAR_DIR="${HAR_DIR:-./har}"
HEF_DIR="${HEF_DIR:-./hef}"

mkdir -p "$HAR_DIR" "$HEF_DIR"

compile_model() {
    local name="$1"
    local onnx="$ONNX_DIR/${name}.onnx"

    if [ ! -f "$onnx" ]; then
        echo "SKIP $name — ONNX not found"
        return
    fi

    echo "=== Compiling $name ==="

    # Step 1: Parse ONNX → HAR (Hailo Archive)
    hailo parser onnx "$onnx" \
        --hw-arch hailo8 \
        --har-path "$HAR_DIR/${name}.har"

    # Step 2: Optimize (quantize to INT8)
    # For these tiny ALU models, we use random calibration data
    # since the models are trained to binary precision anyway
    hailo optimize "$HAR_DIR/${name}.har" \
        --hw-arch hailo8 \
        --har-path "$HAR_DIR/${name}_optimized.har" \
        --use-random-calib-set

    # Step 3: Compile → HEF (Hailo Executable Format)
    hailo compiler "$HAR_DIR/${name}_optimized.har" \
        --hw-arch hailo8 \
        --hef-path "$HEF_DIR/${name}.hef"

    echo "✅ $name → $HEF_DIR/${name}.hef"
}

if [ $# -gt 0 ]; then
    compile_model "$1"
else
    # Compile all — prioritize ALU core models
    for model in arithmetic divide carry_combine compare multiply logical lsl lsr asr rol sincos exp log sqrt atan2; do
        compile_model "$model" || echo "⚠️  $model failed"
    done
fi

echo ""
echo "Done. HEF files ready for deployment."
echo "Copy to target: scp $HEF_DIR/*.hef target:/opt/ncpu/models/"
