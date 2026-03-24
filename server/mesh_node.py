#!/usr/bin/env python3
"""Lightweight ONNX mesh node server for neural ALU operations.

Runs on each Raspberry Pi. Accepts neural ALU operations via HTTP,
executes them through ONNX Runtime (bit-level neural inference),
returns results as JSON.

Deps: stdlib + numpy + onnxruntime (both already on the Pis).

Usage:
    python3 mesh_node.py [--port 9200] [--models /home/pi/ncpu-bridge/onnx]
"""

import argparse
import json
import os
import time
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import onnxruntime as ort

N_BITS = 32


def ints_to_bits(vals, n_bits=N_BITS):
    """Convert integer array to bit representation [N, n_bits]."""
    vals = np.asarray(vals, dtype=np.int64).reshape(-1)
    bits = np.zeros((len(vals), n_bits), dtype=np.float32)
    for i in range(n_bits):
        bits[:, i] = ((vals >> i) & 1).astype(np.float32)
    return bits


def bits_to_ints(bits, n_bits=N_BITS):
    """Convert bit array [N, n_bits] back to integers."""
    b = (bits > 0.5).astype(np.int64)
    result = np.zeros(b.shape[0], dtype=np.int64)
    for i in range(n_bits):
        result += b[:, i] << i
    mask = result >= (1 << (n_bits - 1))
    result[mask] -= (1 << n_bits)
    return result


class NeuralALU:
    """Neural ALU using ONNX models — same algorithms as nCPU."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        so = ort.SessionOptions()
        so.inter_op_num_threads = 2
        so.intra_op_num_threads = 2
        prov = ["CPUExecutionProvider"]

        self.adder = ort.InferenceSession(str(self.model_dir / "arithmetic.onnx"), so, providers=prov)
        self.multiplier = ort.InferenceSession(str(self.model_dir / "multiply.onnx"), so, providers=prov)
        self.logical = ort.InferenceSession(str(self.model_dir / "logical.onnx"), so, providers=prov)
        self.ops_count = 0
        self.total_inference_ms = 0.0

    def _adder_call(self, inp):
        """inp: [N, 3] -> [N, 2] (sum_bit, carry)"""
        return self.adder.run(None, {"input": inp})[0]

    def _mul_call(self, a_oh, b_oh):
        """a_oh: [N, 256], b_oh: [N, 256] -> [N, 16]"""
        return self.multiplier.run(None, {"a_onehot": a_oh, "b_onehot": b_oh})[0]

    def _logical_call(self, op_oh, idx_oh):
        """op_oh: [N, 7], idx_oh: [N, 4] -> [N, 1]"""
        return self.logical.run(None, {"op_onehot": op_oh, "idx_onehot": idx_oh})[0]

    def _ripple_add(self, bits_a, bits_b, carry_in=0.0):
        """Ripple-carry addition through neural adder."""
        n = bits_a.shape[0]
        result = np.zeros((n, N_BITS), dtype=np.float32)
        carry = np.full(n, carry_in, dtype=np.float32)
        for i in range(N_BITS):
            inp = np.stack([bits_a[:, i], bits_b[:, i], carry], axis=1).astype(np.float32)
            out = self._adder_call(inp)
            result[:, i] = (out[:, 0] > 0.5).astype(np.float32)
            carry = (out[:, 1] > 0.5).astype(np.float32)
        return result

    def add(self, a: int, b: int) -> int:
        bits_a = ints_to_bits([a])
        bits_b = ints_to_bits([b])
        result_bits = self._ripple_add(bits_a, bits_b, 0.0)
        return int(bits_to_ints(result_bits)[0])

    def sub(self, a: int, b: int) -> int:
        bits_a = ints_to_bits([a])
        bits_b = 1.0 - ints_to_bits([b])
        result_bits = self._ripple_add(bits_a, bits_b, 1.0)
        return int(bits_to_ints(result_bits)[0])

    def mul(self, a: int, b: int) -> int:
        # One-hot encode (8-bit values)
        a_val = int(a) & 0xFF
        b_val = int(b) & 0xFF
        a_oh = np.zeros((1, 256), dtype=np.float32)
        b_oh = np.zeros((1, 256), dtype=np.float32)
        a_oh[0, a_val] = 1.0
        b_oh[0, b_val] = 1.0
        out = self._mul_call(a_oh, b_oh)  # [1, 16]
        result = 0
        for i in range(16):
            if out[0, i] > 0.5:
                result += 1 << i
        return result

    def div(self, a: int, b: int) -> int:
        if b == 0:
            return 0
        a_abs = abs(a)
        b_abs = abs(b)
        quotient = 0
        remainder = 0
        for i in range(31, -1, -1):
            remainder = (remainder << 1) | ((a_abs >> i) & 1)
            # Neural subtraction to test remainder >= b
            bits_r = ints_to_bits([remainder])
            bits_b_comp = 1.0 - ints_to_bits([b_abs])
            diff_bits = self._ripple_add(bits_r, bits_b_comp, 1.0)
            diff = int(bits_to_ints(diff_bits)[0])
            if diff >= 0:
                remainder = diff
                quotient |= (1 << i)
        return quotient

    def cmp(self, a: int, b: int) -> dict:
        diff = self.sub(a, b)
        return {"zf": int(diff == 0), "sf": int(diff < 0)}

    def _bitwise(self, op_idx: int, a: int, b: int) -> int:
        bits_a = ints_to_bits([a])
        bits_b = ints_to_bits([b])
        result_bits = np.zeros((1, N_BITS), dtype=np.float32)
        op_oh = np.zeros((1, 7), dtype=np.float32)
        op_oh[0, op_idx] = 1.0
        for i in range(N_BITS):
            a_bit = int(bits_a[0, i] > 0.5)
            b_bit = int(bits_b[0, i] > 0.5)
            idx = a_bit * 2 + b_bit
            idx_oh = np.zeros((1, 4), dtype=np.float32)
            idx_oh[0, idx] = 1.0
            out = self._logical_call(op_oh, idx_oh)
            result_bits[0, i] = (out[0, 0] > 0.5)
        return int(bits_to_ints(result_bits)[0])

    def and_op(self, a: int, b: int) -> int:
        return self._bitwise(0, a, b)

    def or_op(self, a: int, b: int) -> int:
        return self._bitwise(1, a, b)

    def xor_op(self, a: int, b: int) -> int:
        return self._bitwise(2, a, b)

    def run_op(self, operation: str, a: int, b: int = 0) -> dict:
        """Execute an operation and return result dict."""
        dispatch = {
            "add": self.add,
            "sub": self.sub,
            "mul": self.mul,
            "div": self.div,
            "cmp": self.cmp,
            "and": self.and_op,
            "or": self.or_op,
            "xor": self.xor_op,
        }
        fn = dispatch.get(operation)
        if not fn:
            raise ValueError(f"Unknown operation: {operation}. Available: {list(dispatch.keys())}")

        t0 = time.perf_counter()
        result = fn(a, b)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.ops_count += 1
        self.total_inference_ms += elapsed_ms

        out = {"inference_ms": round(elapsed_ms, 3)}
        if isinstance(result, dict):
            out["result"] = 0
            out["flags"] = result
        else:
            out["result"] = result
        return out


SUPPORTED_OPS = ["add", "sub", "mul", "div", "cmp", "and", "or", "xor"]


class MeshNodeHandler(BaseHTTPRequestHandler):
    alu: NeuralALU = None
    node_id: str = ""
    start_time: float = 0.0

    def log_message(self, fmt, *args):
        pass

    def _json_response(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {
                "status": "ok",
                "node_id": self.node_id,
                "uptime_s": round(time.time() - self.start_time, 1),
                "ops_served": self.alu.ops_count,
                "avg_inference_ms": round(
                    self.alu.total_inference_ms / max(self.alu.ops_count, 1), 3
                ),
                "hostname": socket.gethostname(),
            })
        elif self.path == "/ops":
            self._json_response(200, {"operations": SUPPORTED_OPS})
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/compute":
            self._json_response(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        op = body.get("operation")
        a = body.get("a", 0)
        b = body.get("b", 0)

        if not op:
            self._json_response(400, {"error": "missing 'operation'"})
            return

        try:
            result = self.alu.run_op(op, int(a), int(b))
            result["node_id"] = self.node_id
            self._json_response(200, result)
        except Exception as e:
            self._json_response(500, {"error": str(e), "node_id": self.node_id})


def serve(port: int = 9200, model_dir: str = "/home/pi/ncpu-bridge/onnx", node_id: str = ""):
    if not node_id:
        node_id = f"{socket.gethostname()}-{port}"

    print(f"[mesh-node] Loading ONNX models from {model_dir}...")
    alu = NeuralALU(model_dir)

    # Quick smoke test
    try:
        r = alu.add(2, 3)
        print(f"[mesh-node] Smoke test: 2 + 3 = {r} {'✓' if r == 5 else '✗'}")
    except Exception as e:
        print(f"[mesh-node] Smoke test failed: {e}")

    MeshNodeHandler.alu = alu
    MeshNodeHandler.node_id = node_id
    MeshNodeHandler.start_time = time.time()

    server = HTTPServer(("0.0.0.0", port), MeshNodeHandler)
    print(f"[mesh-node] {node_id} listening on :{port}")
    print(f"[mesh-node] Operations: {SUPPORTED_OPS}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[mesh-node] Shutting down")
        server.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Neural Mesh Node Server")
    p.add_argument("--port", type=int, default=9200)
    p.add_argument("--models", default="/home/pi/ncpu-bridge/onnx")
    p.add_argument("--id", default="", help="Node identifier")
    args = p.parse_args()
    serve(port=args.port, model_dir=args.models, node_id=args.id)
