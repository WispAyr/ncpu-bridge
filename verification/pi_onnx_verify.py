#!/usr/bin/env python3
"""Pi ONNX Verification - Batched ONNX Runtime on ARM64.

Runs all 65,536 input pairs (0-255 x 0-255) for 8 binary operations
using ONNX Runtime with the exported nCPU models. Implements the same
algorithms as NeuralOps (ripple-carry add, restoring division, etc.)
but using ONNX sessions instead of PyTorch models.

Output: JSON with SHA-256 hashes per operation.
"""
import os, sys, json, hashlib, time, platform, socket
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed. Run: pip3 install onnxruntime --user")
    sys.exit(1)

ONNX_DIR = os.environ.get('ONNX_DIR', os.path.expanduser('~/ncpu-hailo/onnx_models'))
OPERATIONS = ['add', 'sub', 'mul', 'div', 'cmp', 'and', 'or', 'xor']
N_BITS = 32
N = 65536

A_VALS = np.repeat(np.arange(256, dtype=np.int64), 256)
B_VALS = np.tile(np.arange(256, dtype=np.int64), 256)


def ints_to_bits(vals, n_bits=N_BITS):
    bits = np.zeros((len(vals), n_bits), dtype=np.float32)
    for i in range(n_bits):
        bits[:, i] = ((vals >> i) & 1).astype(np.float32)
    return bits


def bits_to_ints(bits, n_bits=N_BITS):
    b = (bits > 0.5).astype(np.int64)
    result = np.zeros(b.shape[0], dtype=np.int64)
    for i in range(n_bits):
        result += b[:, i] << i
    mask = result >= (1 << (n_bits - 1))
    result[mask] -= (1 << n_bits)
    return result


class ONNXOps:
    def __init__(self, onnx_dir):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 4
        print(f"Loading ONNX models from {onnx_dir}...")
        self.adder = ort.InferenceSession(os.path.join(onnx_dir, 'arithmetic.onnx'), so)
        self.multiplier = ort.InferenceSession(os.path.join(onnx_dir, 'multiply.onnx'), so)
        self.logical = ort.InferenceSession(os.path.join(onnx_dir, 'logical.onnx'), so)
        print("Models loaded.")

    def _adder_call(self, inp):
        """inp: [N, 3] float32 -> [N, 2] float32 (sum_bit, carry)"""
        return self.adder.run(None, {'input': inp})[0]

    def _mul_call(self, a_oh, b_oh):
        """a_oh: [N, 256], b_oh: [N, 256] -> [N, 16]"""
        return self.multiplier.run(None, {'a_onehot': a_oh, 'b_onehot': b_oh})[0]

    def _logical_call(self, op_oh, idx_oh):
        """op_oh: [N, 7], idx_oh: [N, 4] -> [N, 1]"""
        return self.logical.run(None, {'op_onehot': op_oh, 'idx_onehot': idx_oh})[0]

    def batch_ripple_add(self, bits_a, bits_b, carry_in=0.0):
        n = bits_a.shape[0]
        result = np.zeros((n, N_BITS), dtype=np.float32)
        carry = np.full(n, carry_in, dtype=np.float32)

        for i in range(N_BITS):
            inp = np.stack([bits_a[:, i], bits_b[:, i], carry], axis=1).astype(np.float32)
            out = self._adder_call(inp)
            result[:, i] = (out[:, 0] > 0.5).astype(np.float32)
            carry = (out[:, 1] > 0.5).astype(np.float32)
        return result

    def add(self):
        bits_a = ints_to_bits(A_VALS)
        bits_b = ints_to_bits(B_VALS)
        result_bits = self.batch_ripple_add(bits_a, bits_b, 0.0)
        return bits_to_ints(result_bits)

    def sub(self):
        bits_a = ints_to_bits(A_VALS)
        bits_b = 1.0 - ints_to_bits(B_VALS)
        result_bits = self.batch_ripple_add(bits_a, bits_b, 1.0)
        return bits_to_ints(result_bits)

    def mul(self):
        a_oh = np.zeros((N, 256), dtype=np.float32)
        b_oh = np.zeros((N, 256), dtype=np.float32)
        a_oh[np.arange(N), A_VALS.astype(int)] = 1.0
        b_oh[np.arange(N), B_VALS.astype(int)] = 1.0
        out = self._mul_call(a_oh, b_oh)  # [N, 16]
        result = np.zeros(N, dtype=np.int64)
        for i in range(16):
            result += (out[:, i] > 0.5).astype(np.int64) << i
        return result

    def div(self):
        a_abs = np.abs(A_VALS).astype(np.int64)
        b_abs = np.abs(B_VALS).astype(np.int64)
        quotient = np.zeros(N, dtype=np.int64)
        remainder = np.zeros(N, dtype=np.int64)

        for i in range(31, -1, -1):
            remainder = (remainder << 1) | ((a_abs >> i) & 1)
            bits_r = ints_to_bits(remainder)
            bits_b_comp = 1.0 - ints_to_bits(b_abs)
            diff_bits = self.batch_ripple_add(bits_r, bits_b_comp, 1.0)
            diff = bits_to_ints(diff_bits)
            positive = diff >= 0
            remainder = np.where(positive, diff, remainder)
            quotient = np.where(positive, quotient | (1 << i), quotient)
            if i % 8 == 0:
                print(f"    div bit {i}/31")

        return np.where(B_VALS == 0, 0, quotient)

    def cmp(self):
        diff = self.sub()
        return [[bool(d == 0), bool(d < 0)] for d in diff]

    def bitwise(self, op_idx):
        bits_a = ints_to_bits(A_VALS)
        bits_b = ints_to_bits(B_VALS)
        result_bits = np.zeros((N, N_BITS), dtype=np.float32)

        op_oh = np.zeros((N, 7), dtype=np.float32)
        op_oh[:, op_idx] = 1.0

        for i in range(N_BITS):
            a_bit = (bits_a[:, i] > 0.5).astype(int)
            b_bit = (bits_b[:, i] > 0.5).astype(int)
            idx = a_bit * 2 + b_bit
            idx_oh = np.zeros((N, 4), dtype=np.float32)
            idx_oh[np.arange(N), idx] = 1.0
            out = self._logical_call(op_oh, idx_oh)
            result_bits[:, i] = (out[:, 0] > 0.5).astype(np.float32)

        return bits_to_ints(result_bits)


def hash_outputs(outputs):
    canonical = json.dumps(outputs, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def main():
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"=== Pi ONNX Verification ===")
    print(f"Timestamp: {timestamp}")
    print(f"Machine: {platform.node()} ({platform.machine()})")
    print(f"Python: {platform.python_version()}")
    print(f"ONNX Runtime: {ort.__version__}")
    print()

    ops = ONNXOps(ONNX_DIR)

    result = {
        'substrate': 'pi-onnx',
        'hostname': platform.node(),
        'arch': platform.machine(),
        'python': platform.python_version(),
        'framework': f'onnxruntime-{ort.__version__}',
        'timestamp': timestamp,
        'ops': {}
    }

    for op_name in OPERATIONS:
        print(f"Running {op_name}...")
        t0 = time.time()
        if op_name == 'add':
            raw = ops.add().tolist()
        elif op_name == 'sub':
            raw = ops.sub().tolist()
        elif op_name == 'mul':
            raw = ops.mul().tolist()
        elif op_name == 'div':
            raw = ops.div().tolist()
        elif op_name == 'cmp':
            raw = ops.cmp()
        elif op_name == 'and':
            raw = ops.bitwise(0).tolist()
        elif op_name == 'or':
            raw = ops.bitwise(1).tolist()
        elif op_name == 'xor':
            raw = ops.bitwise(2).tolist()

        elapsed = time.time() - t0
        h = hash_outputs(raw)
        result['ops'][op_name] = {
            'hash': h,
            'time_s': round(elapsed, 3),
        }
        print(f"  {op_name}: {h[:16]}.. ({elapsed:.2f}s)")

    # Output as JSON
    print("\n" + json.dumps(result, indent=2))

    # Also save to file
    out_path = os.path.expanduser('~/ncpu-hailo/pi_onnx_results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")
    return result


if __name__ == '__main__':
    main()
