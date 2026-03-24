# nCPU Bridge

**A bridge between neural computation and real-world infrastructure monitoring.**

## 📄 Whitepaper

**[Neural Computing Primitives: Building a Complete Computing Stack on Trained Neural Networks](paper/paper.pdf)**

We route every computing primitive — arithmetic, comparison, bitwise, and shift operations — through trained PyTorch neural networks. Built atop the nCPU project's 66 verified ALU operations, ncpu-bridge implements 44 system-level modules spanning a C compiler, TCP stack, filesystem, virtual machine, database, kernel, and more.

**Key results:**
- **44 system modules** built entirely on neural primitives (compiler, TCP/IP, filesystem, SQL database, kernel, Forth interpreter, ARM64 decoder)
- **34 trained models** totalling ~2.5M parameters — 100% correctness across all operations
- **Cross-substrate verification:** 524,288 test cases produce bit-identical results across PyTorch (Apple Silicon) and ONNX Runtime (Raspberry Pi ARM64)
- **ONNX batched inference:** 3.9M ops/sec on Raspberry Pi 5 (ARM64), 10.8M ops/sec on Apple Silicon
- **Hailo-8 NPU** connected and ready for HEF compilation — projected >50M ops/sec

📊 [Verification Report](verification/VERIFICATION_REPORT.md) · 📄 [Paper (PDF)](paper/paper.pdf)

## Installation

```bash
pip install -e "."

# With server support:
pip install -e ".[server]"

# With development tools:
pip install -e ".[dev]"
```

### Model Weights

The neural ALU models are required at runtime. Three options:

**Option A: Download from GitHub release**
```bash
./scripts/download_models.sh
# Models go to ~/.ncpu/models/
```

**Option B: Clone nCPU as a sibling directory**
```bash
cd ..
git clone https://github.com/WispAyr/nCPU.git
# Auto-discovered at ../nCPU relative to ncpu-bridge
```

**Option C: Set NCPU_PATH explicitly**
```bash
export NCPU_PATH=/path/to/nCPU
```

Path resolution order: `NCPU_PATH` env var → sibling `../nCPU` → `~/.ncpu/models` → bundled ONNX models.

## Quick Start

```bash
# Calculate via neural ALU
ncpu-bridge calculate "48 * 365"

# Verify a computation
ncpu-bridge verify mul 48 365 17520

# Run as a module
python -m bridge.compute

# Run sentinel health checks
python -m bridge.sentinel

# Neural CRC32 benchmark
python -m bridge.neural_hash benchmark

# Compile and run C on neural GPU
python -m bridge.c_compiler
```

## What's In Here

| Module | What It Does | Neural Ops Used |
|--------|-------------|-----------------|
| `bridge.compute` | Core nCPU bridge — wraps neural ALU operations | ADD, SUB, MUL, DIV, CMP, AND, OR, XOR, SHL, SHR |
| `bridge.sentinel` | Infrastructure health checks via neural computation | CMP (threshold checks), SUB (time deltas) |
| `bridge.auto_tune` | SOME loop — auto-adjusts thresholds from outcomes | ADD, DIV, CMP (statistical analysis) |
| `bridge.neural_hash` | CRC32 computed through neural bitwise ops | XOR, AND, SHR (matches stdlib exactly) |
| `bridge.c_compiler` | C subset → nCPU assembly → neural GPU execution | All ops (compiled programs run neurally) |
| `bridge.neural_net_stack` | TCP-like protocol: handshake, checksums, AIMD | ADD, SUB, CMP, SHL (Fletcher-16, flow control) |
| `bridge.neural_crypto` | Stream cipher, KDF, MAC, Diffie-Hellman | XOR, ADD, SHL, MUL (all crypto primitives) |
| `bridge.neural_fs` | Virtual filesystem: inodes, blocks, paths | AND, OR, XOR (bitmap), CMP (path resolution) |
| `bridge.neural_vm` | VM: heap malloc/free, stack, processes | CMP (bounds), ADD/SUB (pointers), first-fit alloc |

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `NCPU_PATH` | Path to nCPU project root | Auto-discovered |
| `BRIDGE_PATH` | Path to ncpu-bridge root | Auto-detected |
| `CLAWD_DATA` | Path for data files (outcomes, thresholds) | `~/.ncpu/data` |
| `MEMDB_PATH` | Path to memdb binary | `memdb` |

## Architecture

```
C Source Code
    ↓ (c_compiler)
nCPU Assembly (MOV, ADD, CMP, JMP...)
    ↓ (neural GPU)
Trained .pt Models (one per ALU operation)
    ↓ (PyTorch inference)
Correct Results (verified against stdlib)
```

## The Absurd Truth

This is a CPU made of neural networks. A CRC32 that takes 6,373x longer than `zlib.crc32` because every XOR is a trained model doing tensor multiplication. A C compiler that targets registers implemented as PyTorch inference calls.

It works. Every result is correct. It's just magnificently inefficient.

## License

MIT
