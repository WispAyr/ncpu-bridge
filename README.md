<p align="center">
  <h1 align="center">⚡ nCPU Bridge</h1>
  <p align="center">
    <strong>A complete computing stack built on trained neural networks</strong>
  </p>
  <p align="center">
    <a href="https://github.com/WispAyr/ncpu-bridge/actions"><img src="https://img.shields.io/github/actions/workflow/status/WispAyr/ncpu-bridge/ci.yml?branch=main&label=CI" alt="CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/python-≥3.10-blue.svg" alt="Python ≥3.10">
    <img src="https://img.shields.io/badge/models-34_trained-green.svg" alt="34 Models">
    <img src="https://img.shields.io/badge/modules-44_system-green.svg" alt="44 Modules">
    <img src="https://img.shields.io/badge/accuracy-100%25-brightgreen.svg" alt="100% Accuracy">
    <img src="https://img.shields.io/badge/Z3_proved-4%2F4_models-purple.svg" alt="Z3 Proved">
  </p>
</p>

---

Every arithmetic, comparison, bitwise, and shift operation routes through **trained PyTorch neural networks**. No hardcoded logic — 34 models totalling ~2.5M parameters implement the full ALU. On top of that: a C compiler, TCP stack, filesystem, VM, database, kernel, ARM64 decoder, and 38 more modules. Every result is correct, verified exhaustively and formally.

📄 **[Paper (PDF)](paper/paper.pdf)** · 📊 **[Verification Report](verification/VERIFICATION_REPORT.md)** · 🏗️ **[Architecture](docs/ARCHITECTURE.md)** · 📈 **[Benchmarks](docs/BENCHMARKS.md)**

---

## Architecture

```
                           ┌─────────────────────────────────────┐
                           │        Application Layer            │
                           │  C Compiler · Shell · HTTP Server   │
                           │  Database · Package Manager · DNS   │
                           └──────────────┬──────────────────────┘
                                          │
                           ┌──────────────▼──────────────────────┐
                           │        System Layer                 │
                           │  Kernel · Scheduler · MMU · IPC     │
                           │  Filesystem · VM · Event Loop       │
                           └──────────────┬──────────────────────┘
                                          │
                           ┌──────────────▼──────────────────────┐
                           │        Primitive Layer              │
                           │  Neural Hash · Sort · Compress      │
                           │  Crypto · Float · Regex · Bloom     │
                           └──────────────┬──────────────────────┘
                                          │
                    ┌─────────────────────▼─────────────────────┐
                    │          Neural ALU (34 Models)            │
                    │                                            │
                    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │
                    │  │ ADD │ │ SUB │ │ MUL │ │ DIV │  ...    │
                    │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘        │
                    │     │      │      │      │              │
                    │  ┌──▼──────▼──────▼──────▼──────────┐   │
                    │  │  Trained Full Adder (3→2 sigmoid)  │   │
                    │  │  Bit-serial ripple carry / CLA     │   │
                    │  └───────────────────────────────────┘   │
                    └──────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
    ┌─────────▼─────────┐    ┌───────────▼──────────┐    ┌──────────▼──────────┐
    │   Apple Silicon    │    │   Raspberry Pi 5     │    │    Hailo-8 NPU      │
    │   PyTorch (.pt)    │    │   ONNX Runtime       │    │    HEF compiled     │
    │   10.8M ops/sec    │    │   3.9M ops/sec       │    │    >50M ops/sec*    │
    └────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                                                * projected
```

## Quick Start

```bash
# 1. Install
pip install -e "."

# 2. Download model weights
./scripts/download_models.sh

# 3. Run: compile and execute C code through neural networks
python -m bridge.c_compiler
```

That's it — you just compiled and ran C code where every CPU operation was a neural network inference.

## Performance

| Metric | Apple Silicon (M4 Max) | Raspberry Pi 5 (ARM64) |
|--------|----------------------:|------------------------:|
| **Throughput** | 10.8M ops/sec | 3.9M ops/sec |
| **ADD (65K pairs)** | 0.297s | 1.762s |
| **MUL (65K pairs)** | 0.006s | 1.510s |
| **DIV (65K pairs)** | 7.816s | 55.858s |
| **XOR (65K pairs)** | 0.045s | 0.246s |
| **Total verification** | 8.754s | 63.448s |

> All benchmarks run exhaustive 256×256 input pairs per operation (65,536 tests each, 524,288 total).
> See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full results.

## Formal Verification

The neural ALU models are **formally proved correct** using the Z3 SMT solver — not just tested, *proved*.

| Model | Architecture | Domain | Z3 Status |
|-------|-------------|--------|-----------|
| **Arithmetic** (full adder) | 3→128→64→2, sigmoid | 8 input combinations | ✅ **Proved** |
| **Logical** (AND/OR/XOR) | 3→64→32→1, sigmoid | 8 input combinations | ✅ **Proved** |
| **Comparator** | 2→64→32→1, sigmoid | 4 input combinations | ✅ **Proved** |
| **Multiplier** | 512→256→256, lookup | 65,536 pairs | ✅ **Proved** |

The Z3 proofs encode the actual trained weights as real-valued arithmetic and verify that the network output matches the expected truth table for **every** possible input — no sampling, no approximation.

## All 44 Modules

| Module | Description | Key Neural Ops |
|--------|-------------|----------------|
| **Core** | | |
| `compute` | Neural ALU bridge — wraps all operations | ADD, SUB, MUL, DIV, CMP, AND, OR, XOR, SHL, SHR |
| `neural_math_real` | Extended math: pow, sqrt, factorial, GCD | Composed from MUL, SUB, CMP |
| `neural_float` | IEEE 754 floating-point via neural bitwise | AND, OR, XOR, SHL, SHR |
| **Compiler & Execution** | | |
| `c_compiler` | C subset → nCPU assembly → neural GPU | All ops (compiled programs run neurally) |
| `neural_vm` | Virtual machine: heap, stack, processes | CMP, ADD, SUB (pointer arithmetic) |
| `neural_forth` | Forth interpreter on neural primitives | ADD, SUB, MUL, CMP (stack machine) |
| `neural_jit` | JIT compilation to neural execution | Dynamic dispatch through neural ALU |
| `neural_elf` | ELF binary parser/loader | AND, SHL, SHR (bit extraction) |
| `neural_linker` | Symbol resolution and linking | CMP, ADD (address computation) |
| `neural_assembler_real` | ARM64 assembler | All bitwise ops (instruction encoding) |
| `neural_arm64_real` | ARM64 instruction decoder | AND, SHR (bit field extraction) |
| **Operating System** | | |
| `neural_kernel` | Microkernel: syscalls, process lifecycle | CMP, ADD (scheduling, resource mgmt) |
| `neural_scheduler` | Priority scheduler with preemption | CMP, SUB (priority comparison, time slicing) |
| `neural_scheduler_real` | Real-time scheduler variant | CMP, SUB (deadline tracking) |
| `neural_mmu_real` | Memory management unit with paging | AND, SHL, SHR (page table walks) |
| `neural_memory_real` | Physical memory allocator | CMP, ADD, AND (bitmap allocation) |
| `neural_slab` | Slab allocator for fixed-size objects | ADD, CMP (slab management) |
| `neural_gc` | Garbage collector (mark-sweep) | CMP, AND (reachability analysis) |
| `neural_signal` | POSIX signal handling | CMP, AND (signal masks) |
| `neural_ipc` | Inter-process communication | ADD, CMP (message passing) |
| `neural_pipe` | Unix pipes | ADD, CMP, AND (ring buffer) |
| `neural_watchdog_real` | Hardware watchdog timer | SUB, CMP (timeout detection) |
| `neural_hypervisor` | Type-1 hypervisor with VM isolation | AND, OR (permission bits), CMP (traps) |
| `neural_gic_real` | Generic Interrupt Controller | AND, OR, CMP (interrupt routing) |
| **Storage & Data** | | |
| `neural_fs` | Virtual filesystem: inodes, blocks, paths | AND, OR, XOR (bitmap), CMP |
| `neural_db` | SQL database engine | CMP (B-tree), ADD (indexing) |
| `neural_cache_real` | LRU/LFU cache with eviction | CMP, SUB (age tracking) |
| `neural_bloom` | Bloom filter | XOR, AND (hash probing) |
| **Networking** | | |
| `neural_net_stack` | TCP-like protocol: handshake, AIMD | ADD, SUB, CMP, SHL (Fletcher-16) |
| `neural_http` | HTTP/1.1 request/response parser | CMP, ADD (header parsing) |
| `neural_dns` | DNS resolver | CMP, XOR (name matching) |
| `neural_mesh` | Mesh networking protocol | ADD, CMP (routing tables) |
| **Security** | | |
| `neural_crypto` | Stream cipher, KDF, MAC, Diffie-Hellman | XOR, ADD, SHL, MUL |
| `neural_hash` | CRC32 via neural bitwise operations | XOR, AND, SHR |
| **Algorithms** | | |
| `neural_sort` | Sorting algorithms (neural comparisons) | CMP (all comparisons are neural) |
| `neural_compress` | LZ-style compression | CMP, AND (pattern matching) |
| `neural_regex` | Regular expression engine | CMP, AND (NFA transitions) |
| **Graphics & Audio** | | |
| `neural_gfx` | 2D graphics: lines, fills, transforms | MUL, ADD (coordinate math) |
| `neural_audio` | Audio synthesis and processing | MUL, ADD (waveform generation) |
| **Infrastructure** | | |
| `neural_shell` | Command shell interpreter | CMP (command dispatch) |
| `neural_container` | Container isolation | AND, CMP (namespace management) |
| `neural_pkg` | Package manager | CMP, ADD (dependency resolution) |
| `neural_state_machine` | Finite state machine engine | CMP (transition matching) |
| `neural_debugger` | GDB-like debugger for neural execution | CMP, AND (breakpoint matching) |
| `neural_event_loop` | Async event loop (epoll-like) | CMP, AND (fd readiness) |
| `neural_self_modify` | Self-modifying code support | All ops (runtime code generation) |

## Cross-Platform Deployment

nCPU Bridge runs on three substrates with **bit-identical results**:

| Substrate | Runtime | Hardware | Status |
|-----------|---------|----------|--------|
| **PyTorch** | Native `.pt` models | Apple Silicon, x86_64 | ✅ Production |
| **ONNX Runtime** | Exported `.onnx` models | Raspberry Pi 5 ARM64 | ✅ Production |
| **Hailo-8 NPU** | Compiled `.hef` models | Hailo-8 on Pi 5 | 🔧 Ready for compilation |

Cross-substrate verification confirms all 524,288 test computations produce **identical SHA-256 hashes** across PyTorch and ONNX Runtime on different hardware architectures.

### Deployment on Raspberry Pi

```bash
# On a fresh Pi 5
pip install onnxruntime numpy
./scripts/download_models.sh    # fetches ONNX models
ncpu-bridge calculate "255 + 255"
```

## Installation

```bash
# Standard install
pip install -e "."

# With FastAPI server
pip install -e ".[server]"

# With Hailo-8 support
pip install -e ".[hailo]"

# Development (tests + linting)
pip install -e ".[dev]"
```

### Model Weights

Models are required at runtime. Three options:

1. **Download release** — `./scripts/download_models.sh` → `~/.ncpu/models/`
2. **Sibling clone** — `git clone https://github.com/WispAyr/nCPU.git ../nCPU`
3. **Explicit path** — `export NCPU_PATH=/path/to/nCPU`

Resolution order: `NCPU_PATH` → `../nCPU` → `~/.ncpu/models` → bundled ONNX.

## Usage

```bash
# Arithmetic through neural networks
ncpu-bridge calculate "48 * 365"

# Verify a computation
ncpu-bridge verify mul 48 365 17520

# Neural CRC32 (correct, just 6,373× slower than zlib)
python -m bridge.neural_hash benchmark

# Run the C compiler demo
python -m bridge.c_compiler

# Full cross-substrate verification
python -m verification.cross_substrate_verify
```

## Citation

```bibtex
@misc{ncpu-bridge-2026,
  title   = {Neural Computing Primitives: Building a Complete Computing Stack 
             on Trained Neural Networks},
  author  = {WispAyr},
  year    = {2026},
  month   = {March},
  url     = {https://github.com/WispAyr/ncpu-bridge},
  note    = {34 trained models, 44 system modules, 100\% accuracy, 
             Z3 formally verified. Open source (MIT).}
}
```

## License

[MIT](LICENSE)

---

<p align="center">
  <em>A CPU made of neural networks. Every result is correct. It's just magnificently inefficient.</em>
</p>
