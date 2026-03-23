"""Hailo-8 Backend — export nCPU models for hardware neural acceleration.

Pipeline: PyTorch (.pt) → ONNX (.onnx) → Hailo HAR → HEF (Hailo Executable)

This module handles:
1. ONNX export of all ALU models using real model classes
2. Verification: PyTorch vs ONNX output comparison
3. Benchmarking: PyTorch vs ONNX Runtime speed
4. Hailo compilation instructions

Usage:
    python -m bridge.hailo_backend export      # Export all models to ONNX
    python -m bridge.hailo_backend verify      # Verify ONNX matches PyTorch
    python -m bridge.hailo_backend benchmark   # Compare inference backends
    python -m bridge.hailo_backend compile     # Compile ONNX → HEF (needs Hailo SDK)
    python -m bridge.hailo_backend info        # Show model architectures
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

import torch
import torch.nn as nn

MODELS_DIR = NCPU_PATH / "models"
EXPORT_DIR = Path("/Users/noc/projects/ncpu-bridge/exported_models")

# Import actual model classes
from ncpu.model.neural_ops import (
    NeuralFullAdder,
    NeuralMultiplierLUT,
    NeuralCompare,
    NeuralLogical,
    NeuralCarryCombine,
    NeuralShiftNet,
)
from ncpu.model.architectures import (
    NeuralSinCos,
    NeuralSqrt,
    NeuralExp,
    NeuralLog,
    NeuralAtan2,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX-compatible wrappers for models with non-standard forward()
# ═══════════════════════════════════════════════════════════════════════════════

class FullAdderONNX(nn.Module):
    """Wraps NeuralFullAdder to include the sigmoid in the exported graph."""
    def __init__(self, model: NeuralFullAdder):
        super().__init__()
        self.full_adder = model.full_adder

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.full_adder(bits))


class CompareONNX(nn.Module):
    """Wraps NeuralCompare to include sigmoid."""
    def __init__(self, model: NeuralCompare):
        super().__init__()
        self.refine = model.refine

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.refine(features))


class CarryCombineONNX(nn.Module):
    """Wraps NeuralCarryCombine to include sigmoid."""
    def __init__(self, model: NeuralCarryCombine):
        super().__init__()
        self.net = model.net

    def forward(self, gp_pairs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(gp_pairs))


class MultiplierLUTONNX(nn.Module):
    """ONNX-exportable wrapper for NeuralMultiplierLUT.

    Takes two one-hot vectors [B, 256] and returns sigmoid(table lookup) [B, 16].
    Uses matmul-based gather: one_hot_a @ table @ one_hot_b^T equivalent.
    """
    def __init__(self, model: NeuralMultiplierLUT):
        super().__init__()
        # table shape: [256, 256, 16]
        self.register_buffer("table", model.lut.table.data.clone())

    def forward(self, a_onehot: torch.Tensor, b_onehot: torch.Tensor) -> torch.Tensor:
        # a_onehot: [B, 256], b_onehot: [B, 256]
        # Gather: result[b] = sum_i sum_j a[b,i] * b[b,j] * table[i,j]
        # Efficient: (a @ table.view(256, -1)).view(B, 256, 16) then sum with b
        B = a_onehot.shape[0]
        # [B, 256] @ [256, 256*16] -> [B, 256*16] -> [B, 256, 16]
        intermediate = torch.matmul(a_onehot, self.table.view(256, -1)).view(B, 256, 16)
        # [B, 1, 256] @ [B, 256, 16] -> [B, 1, 16] -> [B, 16]
        result = torch.matmul(b_onehot.unsqueeze(1), intermediate).squeeze(1)
        return torch.sigmoid(result)


class LogicalONNX(nn.Module):
    """ONNX-exportable wrapper for NeuralLogical.

    Takes op_onehot [B, 7] and idx_onehot [B, 4], returns sigmoid(truth_table lookup) [B, 1].
    """
    def __init__(self, model: NeuralLogical):
        super().__init__()
        # truth_tables: [7, 4]
        self.register_buffer("truth_tables", model.truth_tables.data.clone())

    def forward(self, op_onehot: torch.Tensor, idx_onehot: torch.Tensor) -> torch.Tensor:
        # op_onehot: [B, 7], idx_onehot: [B, 4]
        # selected_row = op_onehot @ truth_tables -> [B, 4]
        selected = torch.matmul(op_onehot, self.truth_tables)  # [B, 4]
        # dot with idx_onehot -> [B, 1]
        result = (selected * idx_onehot).sum(dim=1, keepdim=True)
        return torch.sigmoid(result)


class ShiftNetONNX(nn.Module):
    """ONNX-exportable wrapper for NeuralShiftNet.

    Takes concatenated [value_bits(64), shift_bits(64)] = [B, 128].
    Returns [B, 64] output bits.
    """
    def __init__(self, model: NeuralShiftNet):
        super().__init__()
        self.temperature = model.temperature
        self.shift_decoder = model.shift_decoder
        self.index_net = model.index_net
        self.validity_net = model.validity_net
        # Pre-register the eye matrix as a buffer
        self.register_buffer("positions", torch.eye(64))

    def forward(self, combined_input: torch.Tensor) -> torch.Tensor:
        # combined_input: [B, 128] = [value_bits(64), shift_bits(64)]
        # For simplicity, process batch size 1
        value_bits = combined_input[0, :64]       # [64]
        shift_bits = combined_input[0, 64:]        # [64]

        # Decode shift
        shift_enc = self.shift_decoder(shift_bits.unsqueeze(0))[0]  # [64]
        shift_soft = torch.softmax(shift_enc, dim=0)                 # [64]

        # Build combined inputs for all 64 positions
        shift_expanded = shift_soft.unsqueeze(0).expand(64, -1)      # [64, 64]
        combined = torch.cat([self.positions, shift_expanded], dim=1) # [64, 128]

        # Index network
        idx_logits = self.index_net(combined)                         # [64, 64]
        idx_weights = torch.softmax(idx_logits / self.temperature, dim=1)
        bit_vals = (idx_weights * value_bits.unsqueeze(0)).sum(dim=1) # [64]

        # Validity gate
        valid_logits = self.validity_net(combined)                    # [64, 1]
        valid = torch.sigmoid(valid_logits.squeeze(1))                # [64]

        result = bit_vals * (valid > 0.5).float()
        return result.unsqueeze(0)  # [1, 64]


class ShiftNetASRONNX(nn.Module):
    """ONNX wrapper for ASR shift model (different arch: hidden=512, BatchNorm, fill_net)."""
    def __init__(self, state_dict):
        super().__init__()
        self.temperature = nn.Parameter(state_dict["temperature"].clone())
        self.shift_decoder = nn.Sequential(
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
        )
        self.index_net = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
        )
        self.fill_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.register_buffer("positions", torch.eye(64))
        # Load weights
        self.load_state_dict(state_dict)

    def forward(self, combined_input: torch.Tensor) -> torch.Tensor:
        value_bits = combined_input[0, :64]
        shift_bits = combined_input[0, 64:]
        shift_enc = self.shift_decoder(shift_bits.unsqueeze(0))[0]
        shift_soft = torch.softmax(shift_enc, dim=0)
        shift_expanded = shift_soft.unsqueeze(0).expand(64, -1)
        combined = torch.cat([self.positions, shift_expanded], dim=1)
        idx_logits = self.index_net(combined)
        idx_weights = torch.softmax(idx_logits / self.temperature, dim=1)
        bit_vals = (idx_weights * value_bits.unsqueeze(0)).sum(dim=1)
        fill_logits = self.fill_net(combined)
        fill = torch.sigmoid(fill_logits.squeeze(1))
        result = bit_vals * (fill > 0.5).float()
        return result.unsqueeze(0)


class ShiftNetROLONNX(nn.Module):
    """ONNX wrapper for ROL shift model (hidden=512, BatchNorm, no validity/fill net)."""
    def __init__(self, state_dict):
        super().__init__()
        self.temperature = nn.Parameter(state_dict["temperature"].clone())
        self.shift_decoder = nn.Sequential(
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
        )
        self.index_net = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
        )
        self.register_buffer("positions", torch.eye(64))
        self.load_state_dict(state_dict)

    def forward(self, combined_input: torch.Tensor) -> torch.Tensor:
        value_bits = combined_input[0, :64]
        shift_bits = combined_input[0, 64:]
        shift_enc = self.shift_decoder(shift_bits.unsqueeze(0))[0]
        shift_soft = torch.softmax(shift_enc, dim=0)
        shift_expanded = shift_soft.unsqueeze(0).expand(64, -1)
        combined = torch.cat([self.positions, shift_expanded], dim=1)
        idx_logits = self.index_net(combined)
        idx_weights = torch.softmax(idx_logits / self.temperature, dim=1)
        bit_vals = (idx_weights * value_bits.unsqueeze(0)).sum(dim=1)
        return bit_vals.unsqueeze(0)


class SqrtONNX(nn.Module):
    """Wrapper for NeuralSqrt that duplicates input for BatchNorm compatibility."""
    def __init__(self, model: NeuralSqrt):
        super().__init__()
        self.initial = model.initial
        self.refine = model.refine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1] where B >= 2 for BatchNorm
        y0 = self.initial(x)
        return self.refine(torch.cat([x, y0], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

def _load_model(name: str):
    """Load a model by name, returning (model, wrapper, dummy_inputs, input_names)."""

    if name == "arithmetic":
        model = NeuralFullAdder(hidden_dim=128)
        state = torch.load(MODELS_DIR / "alu" / "arithmetic.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        wrapper = FullAdderONNX(model)
        wrapper.eval()
        dummy = torch.randn(1, 3)
        return model, wrapper, (dummy,), ["input"], {"input": {0: "batch"}, "output": {0: "batch"}}

    elif name == "divide":
        model = NeuralFullAdder(hidden_dim=64)
        state = torch.load(MODELS_DIR / "alu" / "divide.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        wrapper = FullAdderONNX(model)
        wrapper.eval()
        dummy = torch.randn(1, 3)
        return model, wrapper, (dummy,), ["input"], {"input": {0: "batch"}, "output": {0: "batch"}}

    elif name == "carry_combine":
        model = NeuralCarryCombine(hidden_dim=64)
        state = torch.load(MODELS_DIR / "alu" / "carry_combine.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        wrapper = CarryCombineONNX(model)
        wrapper.eval()
        dummy = torch.randn(1, 4)
        return model, wrapper, (dummy,), ["input"], {"input": {0: "batch"}, "output": {0: "batch"}}

    elif name == "compare":
        model = NeuralCompare()
        state = torch.load(MODELS_DIR / "alu" / "compare.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        wrapper = CompareONNX(model)
        wrapper.eval()
        dummy = torch.randn(1, 3)
        return model, wrapper, (dummy,), ["input"], {"input": {0: "batch"}, "output": {0: "batch"}}

    elif name == "multiply":
        model = NeuralMultiplierLUT()
        state = torch.load(MODELS_DIR / "alu" / "multiply.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        wrapper = MultiplierLUTONNX(model)
        wrapper.eval()
        a_oh = torch.zeros(1, 256); a_oh[0, 3] = 1.0
        b_oh = torch.zeros(1, 256); b_oh[0, 7] = 1.0
        return model, wrapper, (a_oh, b_oh), ["a_onehot", "b_onehot"], {
            "a_onehot": {0: "batch"}, "b_onehot": {0: "batch"}, "output": {0: "batch"}
        }

    elif name == "logical":
        model = NeuralLogical()
        state = torch.load(MODELS_DIR / "alu" / "logical.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        wrapper = LogicalONNX(model)
        wrapper.eval()
        op_oh = torch.zeros(1, 7); op_oh[0, 0] = 1.0  # AND
        idx_oh = torch.zeros(1, 4); idx_oh[0, 3] = 1.0  # a=1, b=1
        return model, wrapper, (op_oh, idx_oh), ["op_onehot", "idx_onehot"], {
            "op_onehot": {0: "batch"}, "idx_onehot": {0: "batch"}, "output": {0: "batch"}
        }

    elif name in ("lsl", "lsr"):
        model = NeuralShiftNet()
        pt_file = "lsl.pt" if name == "lsl" else "lsr.pt"
        state = torch.load(MODELS_DIR / "shifts" / pt_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        wrapper = ShiftNetONNX(model)
        wrapper.eval()
        dummy = torch.randn(1, 128)
        return model, wrapper, (dummy,), ["input"], {"output": {0: "batch"}}

    elif name == "sincos":
        model = NeuralSinCos()
        ckpt = torch.load(MODELS_DIR / "math" / "sincos.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        dummy = torch.randn(1, 1)
        return model, model, (dummy,), ["input"], {"input": {0: "batch"}, "output": {0: "batch"}}

    elif name == "exp":
        model = NeuralExp()
        ckpt = torch.load(MODELS_DIR / "math" / "exp.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        dummy = torch.randn(1, 1)
        return model, model, (dummy,), ["input"], {"input": {0: "batch"}, "output": {0: "batch"}}

    elif name == "log":
        model = NeuralLog()
        ckpt = torch.load(MODELS_DIR / "math" / "log.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        dummy = torch.tensor([[0.5]])
        return model, model, (dummy,), ["input"], {"input": {0: "batch"}, "output": {0: "batch"}}

    elif name == "sqrt":
        model = NeuralSqrt()
        ckpt = torch.load(MODELS_DIR / "math" / "sqrt.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        wrapper = SqrtONNX(model)
        wrapper.eval()
        # BatchNorm needs batch>=2
        dummy = torch.tensor([[4.0], [9.0]])
        return model, wrapper, (dummy,), ["input"], {"output": {0: "batch"}}

    elif name == "atan2":
        model = NeuralAtan2()
        state = torch.load(MODELS_DIR / "math" / "atan2.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.eval()
        # BatchNorm needs batch>=2
        dummy = torch.randn(2, 6)
        return model, model, (dummy,), ["input"], {"output": {0: "batch"}}

    else:
        raise ValueError(f"Unknown model: {name}")


# All exportable models
ALL_MODELS = [
    "arithmetic", "divide", "carry_combine", "compare",
    "multiply", "logical", "lsl", "lsr",
    "sincos", "exp", "log", "sqrt", "atan2",
]

# Models that exist as .pt files but use ShiftNet architecture (check for asr, rol)
SHIFT_MODELS = {"asr": "shifts/asr.pt", "rol": "shifts/rol.pt"}


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_to_onnx(models: list[str] | None = None) -> dict:
    """Export nCPU models to ONNX format using real model classes."""
    onnx_dir = EXPORT_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = ALL_MODELS

    results = {}
    for name in models:
        try:
            _, wrapper, dummy_inputs, input_names, dynamic_axes = _load_model(name)

            onnx_path = onnx_dir / f"{name}.onnx"
            torch.onnx.export(
                wrapper,
                dummy_inputs if len(dummy_inputs) > 1 else dummy_inputs[0],
                str(onnx_path),
                input_names=input_names,
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=18,
            )

            results[name] = {
                "status": "ok",
                "onnx_path": str(onnx_path),
                "size_bytes": onnx_path.stat().st_size,
            }
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}

    # Try asr/rol with their specific architectures
    for name, rel_path in SHIFT_MODELS.items():
        pt_path = MODELS_DIR / rel_path
        if not pt_path.exists():
            results[name] = {"status": "skip", "reason": "file not found"}
            continue
        try:
            state = torch.load(pt_path, map_location="cpu", weights_only=True)
            if name == "asr":
                wrapper = ShiftNetASRONNX(state)
            elif name == "rol":
                wrapper = ShiftNetROLONNX(state)
            else:
                continue
            wrapper.eval()
            dummy = torch.randn(1, 128)
            onnx_path = onnx_dir / f"{name}.onnx"
            torch.onnx.export(
                wrapper, dummy, str(onnx_path),
                input_names=["input"], output_names=["output"],
                dynamic_axes={"output": {0: "batch"}},
                opset_version=18,
            )
            results[name] = {"status": "ok", "onnx_path": str(onnx_path), "size_bytes": onnx_path.stat().st_size}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY
# ═══════════════════════════════════════════════════════════════════════════════

def verify_onnx() -> dict:
    """Verify ONNX models produce same output as PyTorch."""
    import onnxruntime as ort

    onnx_dir = EXPORT_DIR / "onnx"
    results = {}

    for name in ALL_MODELS:
        onnx_path = onnx_dir / f"{name}.onnx"
        if not onnx_path.exists():
            results[name] = {"status": "skip", "reason": "ONNX not exported"}
            continue

        try:
            model, wrapper, dummy_inputs, input_names, _ = _load_model(name)

            # PyTorch output
            with torch.no_grad():
                if len(dummy_inputs) > 1:
                    pt_out = wrapper(*dummy_inputs).numpy()
                else:
                    pt_out = wrapper(dummy_inputs[0]).numpy()

            # ONNX Runtime output
            sess = ort.InferenceSession(str(onnx_path))
            feed = {n: dummy_inputs[i].numpy() for i, n in enumerate(input_names)}
            ort_out = sess.run(None, feed)[0]

            max_diff = float(np.max(np.abs(pt_out - ort_out)))
            results[name] = {
                "status": "ok" if max_diff < 1e-5 else "mismatch",
                "max_diff": max_diff,
                "pt_shape": list(pt_out.shape),
                "ort_shape": list(ort_out.shape),
            }
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark() -> dict:
    """Benchmark PyTorch vs ONNX Runtime speed for all models."""
    try:
        import onnxruntime as ort
        has_ort = True
    except ImportError:
        has_ort = False

    onnx_dir = EXPORT_DIR / "onnx"
    results = {}
    iterations = 1000

    for name in ALL_MODELS:
        try:
            _, wrapper, dummy_inputs, input_names, _ = _load_model(name)
        except Exception:
            continue

        # PyTorch benchmark
        for _ in range(10):
            with torch.no_grad():
                wrapper(*dummy_inputs) if len(dummy_inputs) > 1 else wrapper(dummy_inputs[0])

        t0 = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                wrapper(*dummy_inputs) if len(dummy_inputs) > 1 else wrapper(dummy_inputs[0])
        pt_ms = (time.time() - t0) / iterations * 1000

        ort_ms = None
        onnx_path = onnx_dir / f"{name}.onnx"
        if has_ort and onnx_path.exists():
            sess = ort.InferenceSession(str(onnx_path))
            feed = {n: dummy_inputs[i].numpy() for i, n in enumerate(input_names)}
            for _ in range(10):
                sess.run(None, feed)
            t0 = time.time()
            for _ in range(iterations):
                sess.run(None, feed)
            ort_ms = (time.time() - t0) / iterations * 1000

        results[name] = {
            "pytorch_ms": round(pt_ms, 4),
            "onnx_ms": round(ort_ms, 4) if ort_ms else None,
            "speedup": round(pt_ms / ort_ms, 1) if ort_ms else None,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# INFO
# ═══════════════════════════════════════════════════════════════════════════════

def get_model_info() -> list[dict]:
    """Get info about all models."""
    infos = []
    for name in ALL_MODELS + list(SHIFT_MODELS.keys()):
        try:
            model, wrapper, dummy_inputs, _, _ = _load_model(name)
            params = sum(p.numel() for p in wrapper.parameters()) + sum(b.numel() for b in wrapper.buffers())
            onnx_path = EXPORT_DIR / "onnx" / f"{name}.onnx"
            infos.append({
                "name": name,
                "exists": True,
                "params": params,
                "onnx_exported": onnx_path.exists(),
                "onnx_size": onnx_path.stat().st_size if onnx_path.exists() else 0,
            })
        except Exception as e:
            infos.append({"name": name, "exists": False, "error": str(e)})
    return infos


# ═══════════════════════════════════════════════════════════════════════════════
# HAILO COMPILATION SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

HAILO_COMPILE_SCRIPT = """#!/bin/bash
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
    hailo parser onnx "$onnx" \\
        --hw-arch hailo8 \\
        --har-path "$HAR_DIR/${name}.har"

    # Step 2: Optimize (quantize to INT8)
    # For these tiny ALU models, we use random calibration data
    # since the models are trained to binary precision anyway
    hailo optimize "$HAR_DIR/${name}.har" \\
        --hw-arch hailo8 \\
        --har-path "$HAR_DIR/${name}_optimized.har" \\
        --use-random-calib-set

    # Step 3: Compile → HEF (Hailo Executable Format)
    hailo compiler "$HAR_DIR/${name}_optimized.har" \\
        --hw-arch hailo8 \\
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
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"

    if cmd == "info":
        print("nCPU Model Registry")
        print("=" * 60)
        infos = get_model_info()
        total_params = 0
        for info in infos:
            exists = "✅" if info.get("exists") else "❌"
            onnx = "📦" if info.get("onnx_exported") else "  "
            params = f"{info.get('params', 0):,}"
            print(f"  {exists} {onnx} {info['name']:20s} params={params:>10s}")
            total_params += info.get("params", 0)
        print(f"\n  Total parameters: {total_params:,}")

    elif cmd == "export":
        print("Exporting nCPU models to ONNX (opset 18)...")
        print("=" * 60)
        results = export_to_onnx()
        ok = err = 0
        for name, result in results.items():
            if result["status"] == "ok":
                ok += 1
                size = result["size_bytes"]
                print(f"  ✅ {name:20s} → {result['onnx_path']} ({size:,} bytes)")
            else:
                err += 1
                print(f"  ❌ {name:20s} — {result.get('reason', result.get('error', '?'))}")
        print(f"\n  Exported: {ok}, Failed: {err}")

    elif cmd == "verify":
        print("Verifying ONNX vs PyTorch...")
        print("=" * 60)
        results = verify_onnx()
        for name, result in results.items():
            if result["status"] == "ok":
                print(f"  ✅ {name:20s} max_diff={result['max_diff']:.2e}")
            elif result["status"] == "mismatch":
                print(f"  ⚠️  {name:20s} max_diff={result['max_diff']:.2e} MISMATCH!")
            else:
                print(f"  ❌ {name:20s} — {result.get('reason', result.get('error', '?'))}")

    elif cmd == "benchmark":
        print("Benchmarking PyTorch vs ONNX Runtime...")
        print("=" * 60)
        results = benchmark()
        for name, r in results.items():
            pt = f"{r['pytorch_ms']:.4f}ms"
            ort = f"{r['onnx_ms']:.4f}ms" if r['onnx_ms'] else "N/A"
            spd = f"{r['speedup']}x" if r['speedup'] else "-"
            print(f"  {name:20s}  PyTorch: {pt:10s}  ONNX: {ort:10s}  Speedup: {spd}")

    elif cmd == "compile":
        # Write compilation script
        script_path = EXPORT_DIR / "compile_hailo.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(HAILO_COMPILE_SCRIPT)
        script_path.chmod(0o755)
        print(f"Hailo compilation script written to: {script_path}")
        print()
        print("To compile on NOC Pi (192.168.195.238):")
        print(f"  1. scp -r {EXPORT_DIR}/ pi@192.168.195.238:~/ncpu-export/")
        print(f"  2. ssh pi@192.168.195.238")
        print(f"  3. cd ~/ncpu-export && ./compile_hailo.sh")

    else:
        print("Usage: python -m bridge.hailo_backend [info|export|verify|benchmark|compile]")


if __name__ == "__main__":
    main()
