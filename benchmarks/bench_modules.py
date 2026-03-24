#!/usr/bin/env python3
"""bench_modules.py — End-to-end benchmarks for key nCPU-bridge modules.

Tests: neural_hash (CRC32), neural_sort, neural_db, c_compiler, neural_crypto.
"""

import sys
import os
from pathlib import Path
import time
import random
import statistics

sys.path.insert(0, os.environ.get("NCPU_PATH", str(Path(__file__).resolve().parent.parent.parent / "nCPU")))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


ITERATIONS = 10  # Module ops are expensive (many neural ops each)


def time_module(name, func, iterations=ITERATIONS, warmup=2):
    """Time a module-level operation."""
    for _ in range(warmup):
        try:
            func()
        except Exception:
            pass
    
    timings = []
    results = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            result = func()
            results.append(result)
        except Exception as e:
            results.append(f"ERROR: {e}")
        timings.append((time.perf_counter() - t0) * 1000)
    
    mean = statistics.mean(timings)
    median = statistics.median(timings)
    return {
        "name": name,
        "mean_ms": mean,
        "median_ms": median,
        "min_ms": min(timings),
        "max_ms": max(timings),
        "ops_sec": 1000.0 / mean if mean > 0 else 0,
        "iterations": iterations,
    }


def bench_neural_hash():
    """Benchmark neural CRC32 hashing."""
    from bridge.neural_hash import NeuralHash
    nh = NeuralHash()
    
    results = {}
    
    # Hash short string
    r = time_module("hash_short", lambda: nh.crc32_bytes(b"hello"), iterations=5, warmup=1)
    results["hash_5byte"] = r
    print(f"  Hash 'hello' (5 bytes): {r['mean_ms']:.1f}ms avg")
    
    # Hash longer string
    data_16 = b"0123456789abcdef"
    r = time_module("hash_16byte", lambda: nh.crc32_bytes(data_16), iterations=3, warmup=1)
    results["hash_16byte"] = r
    print(f"  Hash 16 bytes: {r['mean_ms']:.1f}ms avg")
    
    # Table build time (the expensive neural part)
    r = time_module("table_build", lambda: NeuralHash(), iterations=3, warmup=0)
    results["table_build"] = r
    print(f"  Table build (256 entries, neural): {r['mean_ms']:.1f}ms avg")
    
    return results


def bench_neural_sort():
    """Benchmark neural sorting algorithms."""
    from bridge.neural_sort import NeuralSort
    ns = NeuralSort()
    
    results = {}
    
    for size in [5, 10, 20]:
        data = [random.randint(0, 255) for _ in range(size)]
        
        # Bubble sort
        r = time_module(f"bubble_{size}", lambda d=list(data): ns.bubble_sort(d), iterations=5, warmup=1)
        results[f"bubble_sort_{size}"] = r
        print(f"  Bubble sort (n={size}): {r['mean_ms']:.1f}ms, ~{r['ops_sec']:.1f} sorts/sec")
        
        # Selection sort
        r = time_module(f"selection_{size}", lambda d=list(data): ns.selection_sort(d), iterations=5, warmup=1)
        results[f"selection_sort_{size}"] = r
        print(f"  Selection sort (n={size}): {r['mean_ms']:.1f}ms")
        
        # Merge sort
        try:
            r = time_module(f"merge_{size}", lambda d=list(data): ns.merge_sort(d), iterations=5, warmup=1)
            results[f"merge_sort_{size}"] = r
            print(f"  Merge sort (n={size}): {r['mean_ms']:.1f}ms")
        except Exception as e:
            print(f"  Merge sort (n={size}): ERROR - {e}")
    
    return results


def bench_neural_db():
    """Benchmark neural database operations."""
    from bridge.neural_db import NeuralDB
    
    results = {}
    
    # Create DB and insert
    db = NeuralDB()
    db.create_index("value")
    
    # Insert benchmark
    r = time_module("db_insert",
                    lambda: db.insert(value=random.randint(0, 255), score=random.randint(0, 100)),
                    iterations=20, warmup=2)
    results["insert"] = r
    print(f"  INSERT: {r['mean_ms']:.1f}ms avg, {r['ops_sec']:.0f} inserts/sec")
    
    # SELECT benchmark
    r = time_module("db_select_all", lambda: db.select(), iterations=10, warmup=1)
    results["select_all"] = r
    print(f"  SELECT * : {r['mean_ms']:.1f}ms avg")
    
    # SELECT with WHERE
    try:
        r = time_module("db_select_where",
                        lambda: db.select(where={"value": ("gt", 128)}),
                        iterations=10, warmup=1)
        results["select_where"] = r
        print(f"  SELECT WHERE: {r['mean_ms']:.1f}ms avg")
    except Exception as e:
        print(f"  SELECT WHERE: ERROR - {e}")
    
    return results


def bench_c_compiler():
    """Benchmark the neural C compiler (compile + execute)."""
    from bridge.c_compiler import NeuralCCompiler
    
    results = {}
    compiler = NeuralCCompiler()
    
    # Simple addition
    prog_add = "int a = 10;\nint b = 20;\nint c = a + b;\nreturn c;"
    r = time_module("compile_add", lambda: compiler.compile_and_run(prog_add), iterations=10, warmup=2)
    results["compile_add"] = r
    print(f"  Compile+run add: {r['mean_ms']:.1f}ms avg")
    
    # More complex program
    prog_complex = """
int x = 42;
int y = 13;
int z = x * y;
int w = z - x;
return w;
"""
    r = time_module("compile_complex", lambda: compiler.compile_and_run(prog_complex), iterations=10, warmup=2)
    results["compile_complex"] = r
    print(f"  Compile+run complex: {r['mean_ms']:.1f}ms avg")
    
    # With loop
    prog_loop = """
int sum = 0;
int i = 0;
while (i < 5) {
    sum = sum + i;
    i = i + 1;
}
return sum;
"""
    r = time_module("compile_loop", lambda: compiler.compile_and_run(prog_loop), iterations=5, warmup=1)
    results["compile_loop"] = r
    print(f"  Compile+run loop: {r['mean_ms']:.1f}ms avg")
    
    return results


def bench_neural_crypto():
    """Benchmark neural crypto operations."""
    from bridge.neural_crypto import NeuralKeyDerivation, NeuralStreamCipher
    
    results = {}
    
    # Key derivation
    kdf = NeuralKeyDerivation()
    r = time_module("kdf_8byte", lambda: kdf.derive_key(42, 8), iterations=5, warmup=1)
    results["kdf_8byte"] = r
    print(f"  KDF (8 bytes): {r['mean_ms']:.1f}ms avg")
    
    # Stream cipher encrypt
    cipher = NeuralStreamCipher()
    plaintext = b"Hello"
    
    r = time_module("encrypt_5byte", lambda: cipher.encrypt(plaintext, key_seed=42), iterations=5, warmup=1)
    results["encrypt_5byte"] = r
    print(f"  Encrypt 5 bytes: {r['mean_ms']:.1f}ms avg")
    
    # Decrypt
    ciphertext = cipher.encrypt(plaintext, key_seed=42)
    r = time_module("decrypt_5byte", lambda: cipher.decrypt(ciphertext, key_seed=42), iterations=5, warmup=1)
    results["decrypt_5byte"] = r
    print(f"  Decrypt 5 bytes: {r['mean_ms']:.1f}ms avg")
    
    return results


def format_markdown(all_results):
    """Generate markdown report."""
    lines = ["# nCPU Module Benchmark Results\n"]
    
    for section, results in all_results.items():
        lines.append(f"\n## {section}\n")
        lines.append("| Operation | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Ops/sec |")
        lines.append("|-----------|-----------|-------------|----------|----------|---------|")
        for key, r in results.items():
            lines.append(
                f"| {key} | {r['mean_ms']:.1f} | {r['median_ms']:.1f} | "
                f"{r['min_ms']:.1f} | {r['max_ms']:.1f} | {r['ops_sec']:.1f} |"
            )
    
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("nCPU Module Benchmark Suite")
    print("=" * 60)
    
    all_results = {}
    
    print("\n[1/5] Neural Hash (CRC32)")
    try:
        all_results["Neural Hash (CRC32)"] = bench_neural_hash()
    except Exception as e:
        print(f"  FAILED: {e}")
    
    print("\n[2/5] Neural Sort")
    try:
        all_results["Neural Sort"] = bench_neural_sort()
    except Exception as e:
        print(f"  FAILED: {e}")
    
    print("\n[3/5] Neural DB")
    try:
        all_results["Neural DB (SQL)"] = bench_neural_db()
    except Exception as e:
        print(f"  FAILED: {e}")
    
    print("\n[4/5] C Compiler")
    try:
        all_results["C Compiler (compile+execute)"] = bench_c_compiler()
    except Exception as e:
        print(f"  FAILED: {e}")
    
    print("\n[5/5] Neural Crypto")
    try:
        all_results["Neural Crypto"] = bench_neural_crypto()
    except Exception as e:
        print(f"  FAILED: {e}")
    
    md = format_markdown(all_results)
    out_path = str(Path(__file__).resolve().parent / "results_modules.md")
    with open(out_path, "w") as f:
        f.write(md)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
