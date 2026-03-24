#!/usr/bin/env python3
"""PU2 Local Verification - Batched PyTorch on Apple Silicon.

Runs all 65,536 input pairs (0-255 × 0-255) for 8 binary operations
using the same neural models and algorithms as NeuralOps, but batched
for speed. Outputs SHA-256 hashes in JSON format.
"""
import sys, os, json, hashlib, time, platform
from datetime import datetime, timezone

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/Users/noc/projects/nCPU')
from ncpu.model.neural_ops import NeuralOps, NeuralFullAdder, NeuralMultiplierLUT, NeuralLogical

MODELS_DIR = '/Users/noc/projects/nCPU/models'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OPERATIONS = ['add', 'sub', 'mul', 'div', 'cmp', 'and', 'or', 'xor']
N_BITS = 32
N = 65536  # 256 * 256

# Build input pairs
A_VALS = np.repeat(np.arange(256, dtype=np.int64), 256)
B_VALS = np.tile(np.arange(256, dtype=np.int64), 256)


def ints_to_bits(vals, n_bits=N_BITS):
    """[N] int64 -> [N, n_bits] float32, LSB first."""
    bits = np.zeros((len(vals), n_bits), dtype=np.float32)
    for i in range(n_bits):
        bits[:, i] = (vals >> i) & 1
    return torch.from_numpy(bits)


def bits_to_ints(bits, n_bits=N_BITS):
    """[N, n_bits] tensor -> [N] int64 numpy array."""
    b = (bits > 0.5).long().numpy()
    result = np.zeros(b.shape[0], dtype=np.int64)
    for i in range(n_bits):
        result += b[:, i].astype(np.int64) << i
    mask = result >= (1 << (n_bits - 1))
    result[mask] -= (1 << n_bits)
    return result


def load_adder():
    """Load arithmetic.pt full adder model."""
    model = NeuralFullAdder(hidden_dim=128)
    state = torch.load(os.path.join(MODELS_DIR, 'alu/arithmetic.pt'), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_multiplier():
    model = NeuralMultiplierLUT()
    state = torch.load(os.path.join(MODELS_DIR, 'alu/multiply.pt'), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_logical():
    model = NeuralLogical()
    state = torch.load(os.path.join(MODELS_DIR, 'alu/logical.pt'), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def batch_ripple_add(adder, bits_a, bits_b, carry_in=0.0):
    """Batched ripple-carry addition. bits_a, bits_b: [N, n_bits] tensors."""
    n = bits_a.shape[0]
    result = torch.zeros(n, N_BITS)
    carry = torch.full((n,), carry_in)

    with torch.no_grad():
        for i in range(N_BITS):
            inp = torch.stack([bits_a[:, i], bits_b[:, i], carry], dim=1)  # [N, 3]
            out = adder(inp)  # [N, 2] (sigmoid already in model)
            result[:, i] = (out[:, 0] > 0.5).float()
            carry = (out[:, 1] > 0.5).float()
    return result


def run_add(adder):
    bits_a = ints_to_bits(A_VALS)
    bits_b = ints_to_bits(B_VALS)
    result_bits = batch_ripple_add(adder, bits_a, bits_b, 0.0)
    return bits_to_ints(result_bits)


def run_sub(adder):
    bits_a = ints_to_bits(A_VALS)
    bits_b = 1.0 - ints_to_bits(B_VALS)
    result_bits = batch_ripple_add(adder, bits_a, bits_b, 1.0)
    return bits_to_ints(result_bits)


def run_mul(multiplier):
    """Multiply using neural LUT - batch all 65536 pairs."""
    with torch.no_grad():
        a_idx = torch.tensor(A_VALS, dtype=torch.long)
        b_idx = torch.tensor(B_VALS, dtype=torch.long)
        logits = multiplier.lut.table[a_idx, b_idx]  # [N, 16]
        bits = (torch.sigmoid(logits) > 0.5).float()
    # Convert to int
    result = np.zeros(N, dtype=np.int64)
    b_np = bits.numpy()
    for i in range(16):
        result += (b_np[:, i] > 0.5).astype(np.int64) << i
    return result


def run_div(adder):
    """Restoring division using batched neural sub."""
    a_abs = np.abs(A_VALS).astype(np.int64)
    b_abs = np.abs(B_VALS).astype(np.int64)

    quotient = np.zeros(N, dtype=np.int64)
    remainder = np.zeros(N, dtype=np.int64)

    for i in range(31, -1, -1):
        remainder = (remainder << 1) | ((a_abs >> i) & 1)
        # neural sub: remainder - b_abs
        bits_r = ints_to_bits(remainder)
        bits_b_comp = 1.0 - ints_to_bits(b_abs)
        diff_bits = batch_ripple_add(adder, bits_r, bits_b_comp, 1.0)
        diff = bits_to_ints(diff_bits)
        positive = diff >= 0
        remainder = np.where(positive, diff, remainder)
        quotient = np.where(positive, quotient | (1 << i), quotient)
        if i % 8 == 0:
            print(f"    div bit {i}/31")

    result = np.where(B_VALS == 0, 0, quotient)
    return result


def run_cmp(adder):
    """CMP = SUB then extract flags."""
    diff = run_sub(adder)
    # Return as list of [zero_flag, sign_flag]
    return [[bool(d == 0), bool(d < 0)] for d in diff]


def run_bitwise(logical, op_idx):
    """Bitwise op using logical model, batched per bit."""
    bits_a = ints_to_bits(A_VALS)
    bits_b = ints_to_bits(B_VALS)
    result_bits = torch.zeros(N, N_BITS)

    with torch.no_grad():
        for i in range(N_BITS):
            a_bit = (bits_a[:, i] > 0.5).long()
            b_bit = (bits_b[:, i] > 0.5).long()
            idx = a_bit * 2 + b_bit  # [N]
            out = torch.sigmoid(logical.truth_tables[op_idx, idx])  # [N]
            result_bits[:, i] = (out > 0.5).float()

    return bits_to_ints(result_bits)


def hash_outputs(outputs):
    canonical = json.dumps(outputs, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def run_reference():
    """Plain Python reference."""
    results = {}
    for op in OPERATIONS:
        outputs = []
        for a in range(256):
            for b in range(256):
                if op == 'add': outputs.append((a + b) & 0xFF)
                elif op == 'sub': outputs.append((a - b) & 0xFF)
                elif op == 'mul': outputs.append((a * b) & 0xFF)
                elif op == 'div': outputs.append(a // b if b != 0 else 0)
                elif op == 'cmp': outputs.append([a == b, a < b])
                elif op == 'and': outputs.append(a & b)
                elif op == 'or': outputs.append(a | b)
                elif op == 'xor': outputs.append(a ^ b)
        results[op] = outputs
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"=== PU2 PyTorch Verification ===")
    print(f"Timestamp: {timestamp}")
    print(f"Machine: {platform.node()} ({platform.machine()})")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Load models
    print("Loading models...")
    adder = load_adder()
    multiplier = load_multiplier()
    logical = load_logical()
    print("Models loaded.\n")

    # Reference
    print("Computing reference...")
    ref = run_reference()
    print("Reference done.\n")

    # Neural ops
    neural_results = {}
    timings = {}

    for op in OPERATIONS:
        print(f"Running {op}...")
        t0 = time.time()
        if op == 'add':
            raw = run_add(adder)
            neural_results[op] = raw.tolist()
        elif op == 'sub':
            raw = run_sub(adder)
            neural_results[op] = raw.tolist()
        elif op == 'mul':
            raw = run_mul(multiplier)
            neural_results[op] = raw.tolist()
        elif op == 'div':
            raw = run_div(adder)
            neural_results[op] = raw.tolist()
        elif op == 'cmp':
            neural_results[op] = run_cmp(adder)
        elif op == 'and':
            raw = run_bitwise(logical, 0)
            neural_results[op] = raw.tolist()
        elif op == 'or':
            raw = run_bitwise(logical, 1)
            neural_results[op] = raw.tolist()
        elif op == 'xor':
            raw = run_bitwise(logical, 2)
            neural_results[op] = raw.tolist()
        elapsed = time.time() - t0
        timings[op] = elapsed
        print(f"  {op}: {elapsed:.2f}s")

    # Hash and compare
    result = {
        'substrate': 'pu2-pytorch',
        'hostname': platform.node(),
        'arch': platform.machine(),
        'python': platform.python_version(),
        'framework': f'pytorch-{torch.__version__}',
        'timestamp': timestamp,
        'ops': {}
    }

    print("\n=== HASHES ===")
    for op in OPERATIONS:
        h = hash_outputs(neural_results[op])
        ref_h = hash_outputs(ref[op])
        result['ops'][op] = {
            'hash': h,
            'reference_hash': ref_h,
            'time_s': round(timings[op], 3),
            'matches_reference': h == ref_h,
        }
        match = "✅" if h == ref_h else "⚠️"
        print(f"  {op}: {h[:16]}.. (ref: {ref_h[:16]}..) {match}")

    # Save
    out_path = os.path.join(RESULTS_DIR, 'pu2_pytorch_results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Also save reference
    ref_result = {
        'substrate': 'reference',
        'hostname': platform.node(),
        'ops': {}
    }
    for op in OPERATIONS:
        ref_result['ops'][op] = {'hash': hash_outputs(ref[op])}
    ref_path = os.path.join(RESULTS_DIR, 'reference_results.json')
    with open(ref_path, 'w') as f:
        json.dump(ref_result, f, indent=2)

    return result


if __name__ == '__main__':
    main()
