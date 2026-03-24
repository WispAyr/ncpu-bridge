# nCPU-Bridge Benchmark Suite

Comprehensive benchmarks for the neural computing stack — measures latency, throughput, accuracy, and scaling characteristics of all neural ALU operations and system modules.

## Prerequisites

```bash
pip install torch onnxruntime numpy
```

Models must be present:
- PyTorch: `/Users/noc/projects/nCPU/models/`
- ONNX: `exported_models/onnx/`

## Benchmarks

### 1. ALU Operations (`bench_alu.py`)

Per-operation latency and throughput for all 10 neural ALU ops (ADD, SUB, MUL, DIV, CMP, AND, OR, XOR, SHL, SHR). Compares PyTorch inference vs ONNX Runtime, including batched ONNX throughput. Also verifies accuracy against Python arithmetic.

```bash
python benchmarks/bench_alu.py
```

**Output:** `benchmarks/results_alu.md`

### 2. Module Benchmarks (`bench_modules.py`)

End-to-end benchmarks for higher-level modules built on the neural ALU:

| Module | What it tests |
|--------|---------------|
| Neural Hash | CRC32 computation (all bitwise ops are neural) |
| Neural Sort | Bubble/selection/merge sort with neural comparisons |
| Neural DB | INSERT, SELECT, WHERE, COUNT with neural index |
| C Compiler | Compile + execute C programs on neural ALU |
| Neural Crypto | Key derivation, stream cipher encrypt/decrypt |

```bash
python benchmarks/bench_modules.py
```

**Output:** `benchmarks/results_modules.md`

### 3. Scaling Benchmarks (`bench_scaling.py`)

How performance scales with:
- **Bit width:** 8-bit native vs 16-bit chained operations
- **Batch size:** ONNX throughput at batch 1–1000
- **Input size:** Sort time vs array length (O(n²) with neural CMPs)

```bash
python benchmarks/bench_scaling.py
```

**Output:** `benchmarks/results_scaling.md`

### Run All

```bash
cd /Users/noc/projects/ncpu-bridge
python benchmarks/bench_alu.py && python benchmarks/bench_modules.py && python benchmarks/bench_scaling.py
```

## Results

Markdown tables are written to `benchmarks/results_*.md`, formatted for direct inclusion in the whitepaper.

## Platform

Benchmarked on Apple Silicon (ARM64) with:
- PyTorch CPU inference (MPS not used for small ops — overhead exceeds compute)
- ONNX Runtime CPUExecutionProvider
- Single-threaded, no batching unless specified
