# Architecture

> How ncpu-bridge builds a complete computing stack from trained neural networks.

## Overview

The system is layered: trained neural networks implement primitive ALU operations, and all higher-level modules compose those primitives — never bypassing them. There is no hardcoded arithmetic anywhere in the computation path.

```
 Application Modules (C compiler, shell, HTTP, DB, ...)
         │
         ▼
 System Modules (kernel, scheduler, MMU, filesystem, ...)
         │
         ▼
 Primitive Modules (hash, sort, compress, crypto, ...)
         │
         ▼
 Neural ALU (34 trained PyTorch models)
         │
         ▼
 Runtime Backend (PyTorch · ONNX · Hailo-8)
```

## The Neural ALU

### Foundation: The Trained Full Adder

At the absolute bottom of the stack sits a **trained full adder** — a neural network that takes three binary inputs (a, b, carry_in) and produces two outputs (sum, carry_out):

```
Input:   [a, b, carry_in]  ∈ {0, 1}³
Network: Linear(3→128) → ReLU → Linear(128→64) → ReLU → Linear(64→2) → Sigmoid
Output:  [sum, carry_out]  ∈ [0, 1]²  (thresholded at 0.5)
```

This model is trained on all 8 possible input combinations and achieves 100% accuracy. More importantly, it has been **formally proved correct** using the Z3 SMT solver — the actual trained weights are encoded as real-valued constraints, and Z3 proves that no input can produce an incorrect output.

### Bit-Serial Addition (Ripple Carry)

32-bit addition is implemented as a **ripple-carry adder** — a chain of 32 full adder inferences:

```
Bit 0:  FA(a₀, b₀, 0)       → sum₀, carry₀
Bit 1:  FA(a₁, b₁, carry₀)  → sum₁, carry₁
Bit 2:  FA(a₂, b₂, carry₁)  → sum₂, carry₂
 ...
Bit 31: FA(a₃₁, b₃₁, carry₃₀) → sum₃₁, carry₃₁
```

Each step is a neural network inference. A single 32-bit addition requires 32 forward passes through the full adder model. This is O(n) in bit width — the same complexity as a hardware ripple-carry adder, just slower per gate.

### Carry-Lookahead Addition (CLA / Parallel Prefix)

For batched operations, the system supports **carry-lookahead** (CLA) addition using the Kogge-Stone parallel prefix algorithm:

1. **Generate/Propagate** — For each bit position, compute:
   - Generate: `g_i = a_i AND b_i` (neural AND)
   - Propagate: `p_i = a_i XOR b_i` (neural XOR)

2. **Parallel Prefix** — Compute group generate/propagate in O(log n) stages:
   - Each stage combines pairs using neural AND and OR operations
   - After ⌈log₂ n⌉ stages, all carry bits are determined

3. **Final Sum** — `sum_i = p_i XOR carry_i` (neural XOR)

This reduces the critical path from O(n) to O(log n) neural inferences at the cost of more total operations — a classic hardware trade-off, now expressed in neural network calls.

### Other ALU Operations

| Operation | Implementation |
|-----------|---------------|
| **SUB** | Two's complement via neural NOT + neural ADD + 1 |
| **MUL** | One-hot lookup table (512→256→256 network for 8-bit pairs) |
| **DIV** | 32-bit restoring division using neural SUB + CMP per bit |
| **CMP** | Neural SUB → flag extraction (zero, sign bits) |
| **AND/OR/XOR** | Bit-by-bit neural truth table (3→64→32→1 per bit) |
| **SHL/SHR** | Barrel shifter via neural MUX tree |
| **NOT** | Neural XOR with 0xFF mask |

All operations mask to 8-bit (or configurable width) to match the trained domain.

## Module Composition

Higher-level modules call only the neural ALU — they never import `operator` or use Python's `+`:

### Example: Neural CRC32

```python
# Every operation is a neural network inference
for byte in data:
    crc = neural_xor(crc, byte)
    for _ in range(8):
        if neural_cmp(neural_and(crc, 1), 1):  # test low bit
            crc = neural_xor(neural_shr(crc, 1), POLYNOMIAL)
        else:
            crc = neural_shr(crc, 1)
return neural_xor(crc, 0xFFFFFFFF)
```

This produces results identical to `zlib.crc32()` — verified exhaustively. It's 6,373× slower because every XOR, AND, SHR, and CMP is a tensor multiplication.

### Example: Neural TCP Stack

The TCP implementation uses neural arithmetic for:
- **Sequence numbers**: `neural_add(seq, payload_len)`
- **Checksum** (Fletcher-16): `neural_add`, `neural_mod` over payload bytes
- **AIMD congestion control**: `neural_shr(window, 1)` for halving, `neural_add(window, 1)` for additive increase
- **Timeout calculation**: `neural_sub(now, last_ack_time)`, `neural_cmp(elapsed, timeout)`

### Example: Neural C Compiler

The compiler pipeline:
1. **Lexer**: Character comparison via `neural_cmp`
2. **Parser**: Recursive descent with neural comparisons for token matching
3. **Code generation**: Emits nCPU assembly (MOV, ADD, CMP, JMP, ...)
4. **Execution**: Each instruction dispatches to the neural ALU

```
int factorial(int n) {      →  MOV R0, n
  int result = 1;           →  MOV R1, 1
  while (n > 1) {           →  CMP R0, 1      ← neural CMP
    result = result * n;    →  MUL R1, R0      ← neural MUL
    n = n - 1;              →  SUB R0, 1       ← neural SUB
  }                         →  JGT loop
  return result;            →  RET R1
}
```

## Cross-Substrate Verification

A critical contribution is proving that neural computation is **substrate-invariant** — the same trained weights produce identical results regardless of the inference runtime:

```
                    ┌───────────────────┐
                    │   Trained Weights  │
                    │   (canonical .pt)  │
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌──────────────┐ ┌──────────┐ ┌──────────────┐
      │  PyTorch     │ │  ONNX    │ │  Hailo HEF   │
      │  Apple M4    │ │  Pi ARM  │ │  Hailo-8 NPU │
      └──────┬───────┘ └────┬─────┘ └──────┬───────┘
             │              │              │
             ▼              ▼              ▼
      ┌──────────────────────────────────────────┐
      │  SHA-256 hash of all 524,288 outputs     │
      │  → IDENTICAL across all substrates       │
      └──────────────────────────────────────────┘
```

### Verification Methodology

1. **Exhaustive domain coverage**: All 256×256 = 65,536 input pairs per operation
2. **8 operations**: ADD, SUB, MUL, DIV, CMP, AND, OR, XOR
3. **Total**: 524,288 neural computations per substrate
4. **Comparison**: SHA-256 hash of serialized output arrays must match exactly

### Why This Matters

If neural ALU operations are substrate-invariant, then any module built on top inherits that property. A neural filesystem or TCP stack running on Apple Silicon will produce bit-identical results to the same code on a Raspberry Pi with ONNX Runtime, or on a Hailo-8 NPU — because the only computation happening is neural inference, and the inference is proven identical.

## Formal Verification with Z3

We go beyond testing to **mathematical proof**. The Z3 SMT solver encodes the neural network's actual trained weights as real-valued arithmetic and proves correctness for every input:

### Process

1. **Extract weights** from the trained `.pt` model
2. **Encode** each layer as Z3 real-valued expressions:
   - Linear: `output[j] = Σ(weight[j][i] * input[i]) + bias[j]`
   - ReLU: `If(x > 0, x, 0)`
   - Sigmoid: piecewise linear approximation
3. **Assert negation** of correctness: `NOT(output == expected)` for each input
4. **Check satisfiability**: if UNSAT, the network is provably correct

### Results

All 4 model families are proved correct:
- **Arithmetic** (full adder): 8/8 input combinations proved
- **Logical** (AND/OR/XOR gate): 8/8 input combinations proved
- **Comparator**: 4/4 input combinations proved
- **Multiplier**: 65,536/65,536 pairs proved (exhaustive + Z3)

This means the neural ALU's correctness is not empirical — it is a mathematical certainty.

## Design Principles

1. **No escape hatches**: Modules never bypass the neural ALU. If you need to add two numbers, it's a neural inference.
2. **Composition over complexity**: Complex operations are built by composing simple proven primitives, not by training larger models.
3. **Verify everything**: Exhaustive testing where feasible, formal verification where possible, cross-substrate hashing for deployment.
4. **Substrate independence**: The same weights must produce the same results everywhere. This is verified, not assumed.
5. **Correctness over performance**: 100% accuracy first. Speed comes from hardware (ONNX batching, Hailo-8 NPU), not from approximation.
