# I Built an Entire Computer Out of Neural Networks. It Works. It's Also Magnificently Slow.

*Can you replace every logic gate with a neural network and still build a functioning computer?*

I spent the last few months finding out. The answer is yes — and the journey there was more interesting than I expected.

## The Idea

A modern CPU does billions of additions per second using transistor logic gates. Simple, fast, proven. But what if you swapped every one of those gates for a small neural network that had *learned* how to add? And then built everything else — a compiler, a TCP stack, a filesystem, a database, a kernel — on top of that foundation?

That's [ncpu-bridge](https://github.com/WispAyr/ncpu-bridge). Every arithmetic, comparison, bitwise, and shift operation runs through a trained PyTorch model. No traditional logic. Just weights and inference.

```python
from bridge.compute import NCPUBridge
cpu = NCPUBridge()
result = cpu.add(42, 17)     # -> 59  (neural network inference)
result = cpu.xor(0xAB, 0xCD) # -> 0x66 (neural network inference)
```

Simple API. Extremely cursed implementation.

## What's Under the Hood

The system uses **34 trained models** totalling about 2.5 million parameters, spanning eight different neural architectures:

- **MLPs** for arithmetic and comparison
- **LUT networks** for multiplication and logical ops
- **Multi-head networks** for bit shifts
- **A 1.7M-parameter Transformer** that decodes ARM64 instructions
- **LSTMs** for cache replacement and watchdog timing
- **CNNs** for assembly tokenization
- **Embedding networks** for memory management
- **Hyperdimensional vectors** (512-dim VSA) for register tracking

Each model learns its operation across the full 8-bit input domain — that's 65,536 input pairs per binary operation, trained to 100% accuracy.

## 44 Modules, All Neural

On top of those primitives, I built **44 system-level modules**. The highlights:

- **A C compiler** — lexer, parser, code generator, all running on neural arithmetic
- **TCP/IP stack** — with Fletcher-16 checksums (chosen over CRC32 because it's ~2 neural ops per byte vs ~40 for CRC)
- **Filesystem** — inode-based, with neural block allocation
- **Database** — B-tree indices, SQL queries, all computed through neural forward passes
- **Kernel** — boots 11 subsystems in 0.588 seconds, entirely through neural ALU operations
- **Forth interpreter, virtual machine, scheduler, garbage collector, signal handler...**

Every intermediate computation — every add, every compare, every bitwise AND inside a hash function — is a neural network inference call.

## The Verification Story

Building it is one thing. Proving it works is another.

I ran exhaustive cross-substrate verification: all 65,536 input pairs for each of 8 binary operations, across three different substrates:

1. **Reference**: Plain Python arithmetic (ground truth)
2. **PyTorch** on Apple M4 Max (macOS, arm64)
3. **ONNX Runtime** on Raspberry Pi 5 (Linux, aarch64)

Every output was serialised to canonical JSON and SHA-256 hashed.

**Result: 8/8 operations produce bit-identical hashes across PyTorch and ONNX Runtime.** That's 524,288 verified computations across two completely different inference engines, operating systems, hardware architectures, and Python versions.

The neural weights encode deterministic mathematical functions. The substrate — PyTorch or ONNX, macOS or Linux, Apple Silicon or ARM Cortex — is irrelevant. **The weights ARE the computation.**

## About That Performance

Let's be honest about the numbers:

| Platform | Latency per op | vs Native |
|----------|---------------|-----------|
| Native CPU | < 1 ns | 1× |
| ONNX on ARM (Cortex-A76) | 8–12 μs | ~10,000× |
| PyTorch on Mac (M4 Max) | ~247 μs | ~247,000× |

Yes, that's ten thousand to a quarter million times slower than a regular CPU. The neural kernel boots in 0.588 seconds, which sounds fast until you remember a real kernel boots billions of operations in less time.

This is not a practical replacement for silicon. It was never meant to be. The point is correctness — proving that the neural abstraction is *sufficient* for Turing-complete computation at the systems level.

## The Path Forward

That said, the performance gap has a known solution: hardware neural accelerators.

The project already exports all 15 core ONNX models, and I have a Hailo-8 (26 TOPS) detected and operational. The pipeline is:

```
PyTorch (.pt) → ONNX (.onnx) → Hailo HEF (.hef) → Hailo-8 Runtime
```

Projected throughput: >1M neural ops/sec, pushing toward sub-microsecond latency per operation. The ONNX-to-HEF conversion step needs the Hailo Dataflow Compiler (x86_64 Linux only), which is the current bottleneck — but it's an engineering problem, not a theoretical one.

## Some Things I Found Interesting

- The **ARM64 Transformer** independently discovered that ADD and SUB have 0.9998 cosine similarity in its learned embeddings (arithmetic cluster), while RET and NOP sit at 0.7823 — semantically different despite similar binary encoding. The model learned instruction semantics from raw data.
- The **LSTM cache** hits 90% on loop patterns and correctly evicts cold scan data while retaining the hot set. It learned LRU-like behaviour without being told what LRU is.
- **Neural CRC32** matches `zlib.crc32` exactly. It's also 6,373× slower. But it's *correct*.
- The **VSA register file** uses 512-dimensional hypervectors with zero cross-correlation between registers — a fundamentally different computational paradigm for register tracking.

## What's Left

It's not all working perfectly. Six math models (sin, cos, sqrt, exp, log, atan2) collapsed during training — they all produce constant output. They need retraining with Huber loss and learning rate warmup. And the 16/32-bit carry chaining (Kogge-Stone parallel prefix) is designed but not yet exhaustively verified.

## Why This Matters

This project isn't about replacing CPUs. It's about a question: *is the neural network abstraction sufficient for general computation?*

The answer is yes, provably, with 524,288 verified test cases and bit-identical cross-substrate results to back it up. Every module — from compiler to kernel — produces correct output through pure neural inference.

And with hardware accelerators closing the performance gap, "correct but slow" is starting to edge toward "correct and interesting."

---

**The code is open source: [github.com/WispAyr/ncpu-bridge](https://github.com/WispAyr/ncpu-bridge)**

*Written by Ewan — [WispAyr](https://github.com/WispAyr)*
