# nCPU Bridge Demos

Self-hosting demos for the neural computing stack. Every arithmetic operation, comparison, and memory access runs through trained neural networks.

## Prerequisites

```bash
cd /Users/noc/projects/ncpu-bridge
# Ensure nCPU models are trained (in /Users/noc/projects/nCPU/models/)
```

## Demos

### 1. Self-Hosting Demo (`self_host.py`)
Compiles C programs with the neural C compiler, executes on the neural GPU, and verifies results against native Python.

```bash
python demos/self_host.py
```

**What it does:** Compiles 4 C programs (arithmetic, bitwise, fibonacci, compound expressions) → shows generated assembly → runs on neural GPU → compares output with Python.

### 2. Kernel Boot (`kernel_boot.py`)
Boots the full neural OS kernel with all 11 subsystems.

```bash
python demos/kernel_boot.py
```

**Subsystems:** Process Manager, Garbage Collector, Filesystem, Message Queue, Kernel Pipe, DNS Resolver, HTTP Server, Task Scheduler, Framebuffer, Crypto Engine, State Machine.

### 3. Full Stack (`full_stack.py`)
End-to-end: kernel boot → store C source in filesystem → compile → execute → store result in neural DB → query it back.

```bash
python demos/full_stack.py
```

**Pipeline:** Boot kernel → `fs.create("/home/demo/multiply.c")` → compile `6 * 7` → execute → verify → `db.insert(results)` → `db.select(results)` → `db.aggregate(SUM)`.
