#!/usr/bin/env python3
"""bench_scaling.py — How nCPU performance scales with input size.

Tests 8-bit, 16-bit, and 32-bit operations using the bridge's native
set_width() support (masks results at the configured bit width).
"""

import sys
import time
import random
import statistics

sys.path.insert(0, "/Users/noc/projects/nCPU")
sys.path.insert(0, "/Users/noc/projects/ncpu-bridge")

import numpy as np
import onnxruntime as ort
from bridge.compute import NCPUBridge

ONNX_DIR = "/Users/noc/projects/ncpu-bridge/exported_models/onnx"
ITERATIONS = 200
WARMUP = 20


def time_op(func, iterations=ITERATIONS, warmup=WARMUP):
    for _ in range(warmup):
        func()
    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        timings.append((time.perf_counter() - t0) * 1000)
    mean = statistics.mean(timings)
    return {"mean_ms": mean, "median_ms": statistics.median(timings),
            "ops_sec": 1000.0 / mean if mean > 0 else 0}


def bench_width_scaling(bridge: NCPUBridge):
    """Compare operations at 8-bit, 16-bit, and 32-bit widths using native set_width()."""
    results = {}
    
    widths = {
        8:  (0, 255),
        16: (0, 65535),
        32: (0, 2**31 - 1),
    }
    
    ops_config = [
        ("add", "ADD", bridge.add, True),
        ("mul", "MUL", bridge.mul, True),
        ("xor", "XOR", bridge.bitwise_xor, True),
    ]
    
    for op_key, op_label, op_fn, _ in ops_config:
        for bits, (lo, hi) in widths.items():
            bridge.set_width(bits)
            max_val = min(hi, 65535)  # keep inputs reasonable for MUL
            r = time_op(lambda mv=max_val: op_fn(random.randint(0, mv), random.randint(0, mv)))
            key = f"{op_key}_{bits}bit"
            results[key] = r
            results[key]["width"] = bits
            print(f"  {op_label} {bits:>2d}-bit:  {r['mean_ms']:.3f}ms, {r['ops_sec']:,.0f} ops/sec")
    
    bridge.set_width(32)  # reset
    
    return results


def bench_onnx_batch_scaling():
    """How ONNX throughput scales with batch size."""
    results = {}
    sess = ort.InferenceSession(f"{ONNX_DIR}/arithmetic.onnx", providers=["CPUExecutionProvider"])
    
    for batch in [1, 10, 50, 100, 500, 1000]:
        feed = {"input": np.random.randn(batch, 3).astype(np.float32)}
        
        def run(s=sess, f=feed):
            s.run(None, f)
        
        r = time_op(run, iterations=200)
        r["batch_size"] = batch
        r["effective_ops_sec"] = r["ops_sec"] * batch
        results[f"batch_{batch}"] = r
        print(f"  Batch {batch:>5d}: {r['mean_ms']:.3f}ms latency, {r['effective_ops_sec']:,.0f} effective ops/sec")
    
    return results


def bench_sort_scaling():
    """How sort time scales with input size."""
    from bridge.neural_sort import NeuralSort
    ns = NeuralSort()
    results = {}
    
    for n in [3, 5, 8, 10, 15, 20]:
        data = [random.randint(0, 255) for _ in range(n)]
        
        timings = []
        for _ in range(3):
            t0 = time.perf_counter()
            ns.bubble_sort(list(data))
            timings.append((time.perf_counter() - t0) * 1000)
        
        mean = statistics.mean(timings)
        results[f"sort_n{n}"] = {"n": n, "mean_ms": mean, "ops_sec": 1000.0 / mean if mean > 0 else 0}
        print(f"  Sort n={n:>3d}: {mean:.1f}ms (O(n²) → ~{n*n} neural CMPs)")
    
    return results


def format_markdown(scaling_8v16, onnx_batch, sort_scaling):
    lines = ["# nCPU Scaling Benchmark Results\n"]
    
    lines.append("## Width Scaling (8 / 16 / 32-bit via set_width)\n")
    lines.append("| Operation | Width | Mean (ms) | Ops/sec |")
    lines.append("|-----------|-------|-----------|---------|")
    
    pairs = [("add", "ADD"), ("mul", "MUL"), ("xor", "XOR")]
    for key, label in pairs:
        for bits in [8, 16, 32]:
            r = scaling_8v16.get(f"{key}_{bits}bit", {})
            if r:
                lines.append(f"| {label} | {bits}-bit | {r['mean_ms']:.3f} | {r['ops_sec']:,.0f} |")
    
    lines.append("\n## ONNX Batch Scaling (arithmetic model)\n")
    lines.append("| Batch Size | Latency (ms) | Effective Ops/sec | Speedup vs batch=1 |")
    lines.append("|------------|-------------|-------------------|---------------------|")
    base_ops = None
    for key, r in sorted(onnx_batch.items(), key=lambda x: x[1]["batch_size"]):
        if base_ops is None:
            base_ops = r["effective_ops_sec"]
        speedup = r["effective_ops_sec"] / base_ops if base_ops > 0 else 0
        lines.append(f"| {r['batch_size']} | {r['mean_ms']:.3f} | {r['effective_ops_sec']:,.0f} | {speedup:.1f}x |")
    
    lines.append("\n## Sort Scaling (Bubble Sort, neural CMP)\n")
    lines.append("| Input Size | Time (ms) | Neural CMPs (~n²) | Sorts/sec |")
    lines.append("|------------|-----------|-------------------|-----------|")
    for key, r in sorted(sort_scaling.items(), key=lambda x: x[1]["n"]):
        lines.append(f"| {r['n']} | {r['mean_ms']:.1f} | ~{r['n']**2} | {r['ops_sec']:.1f} |")
    
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("nCPU Scaling Benchmark Suite")
    print("=" * 60)
    
    bridge = NCPUBridge()
    
    print("\n[1/3] Width Scaling (8/16/32-bit via set_width)")
    scaling_8v16 = bench_width_scaling(bridge)
    
    print("\n[2/3] ONNX Batch Scaling")
    onnx_batch = bench_onnx_batch_scaling()
    
    print("\n[3/3] Sort Scaling (n = input size)")
    sort_scaling = bench_sort_scaling()
    
    md = format_markdown(scaling_8v16, onnx_batch, sort_scaling)
    out_path = "/Users/noc/projects/ncpu-bridge/benchmarks/results_scaling.md"
    with open(out_path, "w") as f:
        f.write(md)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
