#!/usr/bin/env python3
"""bench_alu.py — Individual ALU operation benchmarks (PyTorch vs ONNX Runtime).

Measures latency and throughput for each neural ALU operation.
"""

import sys
import time
import random
import statistics
import numpy as np

sys.path.insert(0, "/Users/noc/projects/nCPU")
sys.path.insert(0, "/Users/noc/projects/ncpu-bridge")

import torch
import onnxruntime as ort
from bridge.compute import NCPUBridge

ONNX_DIR = "/Users/noc/projects/ncpu-bridge/exported_models/onnx"
WARMUP = 50
ITERATIONS = 500
BATCH_SIZES = [1, 10, 100]


def time_op(func, iterations=ITERATIONS, warmup=WARMUP):
    """Time a function, return (mean_ms, median_ms, ops_per_sec, timings)."""
    for _ in range(warmup):
        func()
    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        timings.append((time.perf_counter() - t0) * 1000)
    mean = statistics.mean(timings)
    median = statistics.median(timings)
    ops_sec = 1000.0 / mean if mean > 0 else float("inf")
    return mean, median, ops_sec, timings


def bench_pytorch_ops(bridge: NCPUBridge):
    """Benchmark all PyTorch-backed ALU operations."""
    results = {}
    # Generate random 8-bit test values
    pairs = [(random.randint(0, 255), random.randint(1, 255)) for _ in range(ITERATIONS + WARMUP)]
    idx = [0]

    def make_op(func, two_arg=True):
        def run():
            a, b = pairs[idx[0] % len(pairs)]
            idx[0] += 1
            return func(a, b) if two_arg else func(a, b)
        return run

    ops = {
        "ADD": lambda: bridge.add(random.randint(0, 255), random.randint(0, 255)),
        "SUB": lambda: bridge.sub(random.randint(0, 255), random.randint(0, 255)),
        "MUL": lambda: bridge.mul(random.randint(0, 255), random.randint(0, 255)),
        "DIV": lambda: bridge.div(random.randint(0, 255), random.randint(1, 255)),
        "CMP": lambda: bridge.cmp(random.randint(0, 255), random.randint(0, 255)),
        "AND": lambda: bridge.bitwise_and(random.randint(0, 255), random.randint(0, 255)),
        "OR":  lambda: bridge.bitwise_or(random.randint(0, 255), random.randint(0, 255)),
        "XOR": lambda: bridge.bitwise_xor(random.randint(0, 255), random.randint(0, 255)),
        "SHL": lambda: bridge.shl(random.randint(0, 255), random.randint(0, 7)),
        "SHR": lambda: bridge.shr(random.randint(0, 255), random.randint(0, 7)),
    }

    for name, func in ops.items():
        mean, median, ops_sec, _ = time_op(func)
        results[name] = {"mean_ms": mean, "median_ms": median, "ops_sec": ops_sec}
        print(f"  PyTorch {name:>4s}: {mean:.3f}ms avg, {median:.3f}ms med, {ops_sec:,.0f} ops/sec")

    return results


def bench_onnx_models():
    """Benchmark ONNX Runtime inference for each exported model."""
    import os
    results = {}
    models = sorted(f for f in os.listdir(ONNX_DIR) if f.endswith(".onnx"))

    for model_file in models:
        name = model_file.replace(".onnx", "")
        path = os.path.join(ONNX_DIR, model_file)
        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            inputs = sess.get_inputs()

            # Build dummy input matching the model's expected shape
            feed = {}
            for inp in inputs:
                shape = [s if isinstance(s, int) else 1 for s in inp.shape]
                feed[inp.name] = np.random.randn(*shape).astype(np.float32)

            def run_onnx(s=sess, f=feed):
                s.run(None, f)

            mean, median, ops_sec, _ = time_op(run_onnx)
            results[name] = {"mean_ms": mean, "median_ms": median, "ops_sec": ops_sec}
            print(f"  ONNX {name:>15s}: {mean:.3f}ms avg, {median:.3f}ms med, {ops_sec:,.0f} ops/sec")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"  ONNX {name:>15s}: ERROR - {e}")

    return results


def bench_onnx_batched():
    """Benchmark ONNX with different batch sizes (where supported)."""
    results = {}
    # Only test dynamic-batch models
    batch_models = ["arithmetic", "compare", "divide", "logical", "carry_combine", "multiply"]
    
    for model_name in batch_models:
        path = f"{ONNX_DIR}/{model_name}.onnx"
        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            inputs = sess.get_inputs()
            
            for batch_size in BATCH_SIZES:
                feed = {}
                for inp in inputs:
                    shape = [s if isinstance(s, int) else batch_size for s in inp.shape]
                    feed[inp.name] = np.random.randn(*shape).astype(np.float32)
                
                def run(s=sess, f=feed):
                    s.run(None, f)
                
                mean, median, ops_sec, _ = time_op(run, iterations=200)
                key = f"{model_name}_batch{batch_size}"
                results[key] = {
                    "mean_ms": mean, "median_ms": median,
                    "ops_sec": ops_sec * batch_size,  # effective ops/sec
                    "batch_size": batch_size
                }
                print(f"  ONNX {model_name}[batch={batch_size}]: {mean:.3f}ms, {ops_sec * batch_size:,.0f} effective ops/sec")
        except Exception as e:
            print(f"  ONNX {model_name} batch: ERROR - {e}")
    
    return results


def accuracy_check(bridge: NCPUBridge):
    """Verify neural ALU accuracy against Python arithmetic."""
    results = {}
    tests = 200
    
    ops = {
        "ADD": (bridge.add, lambda a, b: a + b),
        "SUB": (bridge.sub, lambda a, b: a - b),
        "MUL": (bridge.mul, lambda a, b: a * b),
        "DIV": (bridge.div, lambda a, b: a // b if b != 0 else 0),
        "AND": (bridge.bitwise_and, lambda a, b: a & b),
        "OR":  (bridge.bitwise_or, lambda a, b: a | b),
        "XOR": (bridge.bitwise_xor, lambda a, b: a ^ b),
    }
    
    for name, (neural_fn, ref_fn) in ops.items():
        correct = 0
        for _ in range(tests):
            a, b = random.randint(0, 255), random.randint(1, 255)
            try:
                neural = neural_fn(a, b)
                expected = ref_fn(a, b)
                if neural == expected:
                    correct += 1
            except:
                pass
        pct = correct / tests * 100
        results[name] = pct
        print(f"  {name}: {pct:.1f}% ({correct}/{tests})")
    
    return results


def format_markdown(pytorch_results, onnx_results, onnx_batch, accuracy):
    """Generate markdown tables."""
    lines = ["# nCPU ALU Benchmark Results\n"]
    
    # PyTorch table
    lines.append("## PyTorch (CPU) — Per-Operation Latency\n")
    lines.append("| Operation | Mean (ms) | Median (ms) | Ops/sec | Accuracy |")
    lines.append("|-----------|-----------|-------------|---------|----------|")
    for op in ["ADD", "SUB", "MUL", "DIV", "CMP", "AND", "OR", "XOR", "SHL", "SHR"]:
        if op in pytorch_results:
            r = pytorch_results[op]
            acc = f"{accuracy.get(op, 'N/A'):.1f}%" if op in accuracy else "N/A"
            lines.append(f"| {op} | {r['mean_ms']:.3f} | {r['median_ms']:.3f} | {r['ops_sec']:,.0f} | {acc} |")
    
    # ONNX table
    lines.append("\n## ONNX Runtime (CPU) — Per-Model Latency\n")
    lines.append("| Model | Mean (ms) | Median (ms) | Ops/sec |")
    lines.append("|-------|-----------|-------------|---------|")
    for name, r in sorted(onnx_results.items()):
        if "error" not in r:
            lines.append(f"| {name} | {r['mean_ms']:.3f} | {r['median_ms']:.3f} | {r['ops_sec']:,.0f} |")
    
    # Batched ONNX
    lines.append("\n## ONNX Batched — Effective Throughput\n")
    lines.append("| Model | Batch | Latency (ms) | Effective Ops/sec |")
    lines.append("|-------|-------|-------------|-------------------|")
    for key, r in sorted(onnx_batch.items()):
        name = key.rsplit("_batch", 1)[0]
        lines.append(f"| {name} | {r['batch_size']} | {r['mean_ms']:.3f} | {r['ops_sec']:,.0f} |")
    
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("nCPU ALU Benchmark Suite")
    print("=" * 60)
    
    print("\n[1/4] Loading NCPUBridge...")
    bridge = NCPUBridge()
    
    print("\n[2/4] PyTorch ALU Operations")
    pytorch_results = bench_pytorch_ops(bridge)
    
    print("\n[3/4] ONNX Runtime Models")
    onnx_results = bench_onnx_models()
    
    print("\n[3b/4] ONNX Batched Throughput")
    onnx_batch = bench_onnx_batched()
    
    print("\n[4/4] Accuracy Verification")
    accuracy = accuracy_check(bridge)
    
    # Generate markdown
    md = format_markdown(pytorch_results, onnx_results, onnx_batch, accuracy)
    
    out_path = "/Users/noc/projects/ncpu-bridge/benchmarks/results_alu.md"
    with open(out_path, "w") as f:
        f.write(md)
    print(f"\nResults written to {out_path}")
    
    return pytorch_results, onnx_results, onnx_batch, accuracy


if __name__ == "__main__":
    main()
