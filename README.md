# nCPU Bridge

**A bridge between neural computation and real-world infrastructure monitoring.**

Every arithmetic, comparison, and bitwise operation runs through trained PyTorch neural networks on the [nCPU](https://github.com/WispAyr/nCPU) — a CPU built entirely from neural ALU models.

## What's In Here

| Module | What It Does | Neural Ops Used |
|--------|-------------|-----------------|
| `bridge.compute` | Core nCPU bridge — wraps neural ALU operations | ADD, SUB, MUL, DIV, CMP, AND, OR, XOR, SHL, SHR |
| `bridge.sentinel` | Infrastructure health checks via neural computation | CMP (threshold checks), SUB (time deltas) |
| `bridge.auto_tune` | SOME loop — auto-adjusts thresholds from outcomes | ADD, DIV, CMP (statistical analysis) |
| `bridge.neural_state_machine` | Obligation lifecycle state machine | MUL, ADD (state encoding), CMP (transition lookup) |
| `bridge.neural_hash` | CRC32 computed through neural bitwise ops | XOR, AND, SHR (matches stdlib exactly) |
| `bridge.c_compiler` | C subset → nCPU assembly → neural GPU execution | All ops (compiled programs run neurally) |
| `bridge.neural_compress` | RLE + delta + hybrid compression | CMP (run detection), SUB (deltas), ADD (counting) |

## Quick Start

```bash
# Requires nCPU repo at /Users/noc/projects/nCPU with trained models
cd ncpu-bridge
source /Users/noc/projects/nCPU/.venv/bin/activate
export PYTHONPATH=/Users/noc/projects/nCPU:.

# Run sentinel health checks
python -m bridge.sentinel

# Auto-tune thresholds
python -m bridge.auto_tune analyse

# Demo the state machine
python -m bridge.neural_state_machine

# Neural CRC32 benchmark
python -m bridge.neural_hash benchmark

# Compile and run C on neural GPU
python -m bridge.c_compiler

# Compress data neurally
python -m bridge.neural_compress demo
```

## The Absurd Truth

This is a CPU made of neural networks monitoring actual production infrastructure. A CRC32 that takes 6,373x longer than `zlib.crc32` because every XOR is a trained model doing tensor multiplication. A C compiler that targets registers implemented as PyTorch inference calls.

It works. Every result is correct. It's just magnificently inefficient.

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

## Data Flow

```
Sentinel Checks → Outcomes (JSONL)
    ↓
Auto-Tuner → Threshold Proposals
    ↓
State Machine → Obligation Lifecycle
    ↓
Neural Hash → Integrity Verification
    ↓
Compressor → Archived Metrics
```
