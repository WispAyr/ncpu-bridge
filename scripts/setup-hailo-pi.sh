#!/bin/bash
# Setup script for Hailo-8 on NOC Pi (192.168.195.238)
# Run this ON the Pi as root/sudo
set -e

echo "═══════════════════════════════════════════════════"
echo " nCPU Hailo-8 Setup — NOC Pi"
echo "═══════════════════════════════════════════════════"

# 1. Check if Hailo device exists
echo -e "\n[1/5] Checking Hailo-8 hardware..."
if [ -e /dev/hailo0 ]; then
    echo "  ✅ /dev/hailo0 found"
else
    echo "  ❌ /dev/hailo0 not found"
    echo "  Check: lspci | grep Hailo"
    echo "  If PCIe device visible but no /dev/hailo0, install HailoRT driver:"
    echo "    wget https://hailo.ai/downloads/hailort-pcie-driver_4.23.0_all.deb"
    echo "    sudo dpkg -i hailort-pcie-driver_4.23.0_all.deb"
    echo "    sudo reboot"
    exit 1
fi

# 2. Install HailoRT runtime (if not present)
echo -e "\n[2/5] Checking HailoRT..."
if command -v hailortcli &>/dev/null; then
    echo "  ✅ hailortcli found: $(hailortcli --version 2>/dev/null || echo 'version unknown')"
else
    echo "  ❌ hailortcli not found. Install HailoRT:"
    echo "    # Download from https://hailo.ai/developer-zone/"
    echo "    # For Raspberry Pi 5 (aarch64):"
    echo "    sudo dpkg -i hailort_4.23.0_arm64.deb"
    echo "    pip3 install hailort-4.23.0-cp311-cp311-linux_aarch64.whl"
fi

# 3. Check Python bindings
echo -e "\n[3/5] Checking HailoRT Python..."
if python3 -c "from hailo_platform import HEF; print('  ✅ hailo_platform OK')" 2>/dev/null; then
    true
else
    echo "  ❌ hailo_platform not available"
    echo "  Install: pip3 install hailort"
    echo "  Or from .whl: pip3 install hailort-4.23.0-cp311-cp311-linux_aarch64.whl"
fi

# 4. Firmware check
echo -e "\n[4/5] Checking firmware..."
hailortcli fw-control identify 2>/dev/null || echo "  Could not identify firmware"

# 5. Setup workspace
echo -e "\n[5/5] Setting up workspace..."
WORK_DIR=/home/pi/ncpu-hailo
mkdir -p "$WORK_DIR"/{onnx_models,hef_models,venv}

# Create/update venv
if [ ! -f "$WORK_DIR/venv/bin/python" ]; then
    python3 -m venv "$WORK_DIR/venv"
fi
source "$WORK_DIR/venv/bin/activate"
pip install --quiet onnxruntime numpy

echo -e "\n═══════════════════════════════════════════════════"
echo " Setup complete. Next steps:"
echo "  1. Copy ONNX models:  scp exported_models/onnx/*.onnx pi@$(hostname -I | awk '{print $1}'):$WORK_DIR/onnx_models/"
echo "  2. Compile on x86:    python -m bridge.hailo_deploy compile --docker"
echo "  3. Copy HEFs:         scp exported_models/hef/*.hef pi@$(hostname -I | awk '{print $1}'):$WORK_DIR/hef_models/"
echo "  4. Benchmark:         python -m bridge.hailo_deploy benchmark"
echo "═══════════════════════════════════════════════════"
