#!/usr/bin/env python3
"""nCPU Bridge Latency Profiler — raw Python + HTTP benchmarks."""

import os
import sys
import json
import time
import random
import statistics
import argparse
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request, urllib.error
    HAS_REQUESTS = False

# ── Raw Python benchmark ──

def _http_post(url, data):
    """HTTP POST that works with or without requests lib."""
    if HAS_REQUESTS:
        r = requests.post(url, json=data)
        return r.json()
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

def bench_raw(n=1000):
    """Benchmark raw Python compute (no HTTP)."""
    from bridge.config import get_ncpu_path, get_bridge_path
    NCPU_PATH = get_ncpu_path()
    BRIDGE_PATH = get_bridge_path()
    for p in [str(NCPU_PATH), str(BRIDGE_PATH)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    
    from bridge.compute import NCPUBridge
    from bridge.obligations import ObligationChecker
    
    bridge = NCPUBridge(ncpu_path=str(NCPU_PATH))
    oblg = ObligationChecker(bridge=bridge)
    
    ops = {
        "add": bridge.add, "sub": bridge.sub,
        "mul": bridge.mul, "div": bridge.div,
        "and": bridge.bitwise_and, "or": bridge.bitwise_or,
        "xor": bridge.bitwise_xor, "shl": bridge.shl,
    }
    
    results = {}
    for name, fn in ops.items():
        latencies = []
        for _ in range(n):
            a, b = random.randint(1, 100), random.randint(1, 100)
            if name == "shl":
                b = random.randint(0, 4)
            t0 = time.perf_counter_ns()
            fn(a, b)
            latencies.append((time.perf_counter_ns() - t0) / 1000)
        latencies.sort()
        results[name] = {
            "min": round(latencies[0], 2),
            "avg": round(statistics.mean(latencies), 2),
            "max": round(latencies[-1], 2),
            "p99": round(latencies[int(n * 0.99)], 2),
        }
    
    # CMP
    latencies = []
    for _ in range(n):
        a, b = random.randint(1, 100), random.randint(1, 100)
        t0 = time.perf_counter_ns()
        bridge.cmp(a, b)
        latencies.append((time.perf_counter_ns() - t0) / 1000)
    latencies.sort()
    results["cmp"] = {
        "min": round(latencies[0], 2),
        "avg": round(statistics.mean(latencies), 2),
        "max": round(latencies[-1], 2),
        "p99": round(latencies[int(n * 0.99)], 2),
    }
    
    # Obligation check (chained ops)
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        oblg.check_interval(1000000, 1000000 + random.randint(0, 7200), 3600)
        latencies.append((time.perf_counter_ns() - t0) / 1000)
    latencies.sort()
    results["obligation_check"] = {
        "min": round(latencies[0], 2),
        "avg": round(statistics.mean(latencies), 2),
        "max": round(latencies[-1], 2),
        "p99": round(latencies[int(n * 0.99)], 2),
    }
    
    return results

def bench_http(base_url, n=1000):
    """Benchmark HTTP endpoints."""
    ops = ["add", "sub", "mul", "div", "and", "or", "xor", "shl", "cmp"]
    results = {}
    
    for op in ops:
        latencies = []
        for _ in range(n):
            a, b = random.randint(1, 100), random.randint(1, 100)
            if op == "shl":
                b = random.randint(0, 4)
            t0 = time.perf_counter_ns()
            _http_post(f"{base_url}/compute", {"op": op, "a": a, "b": b})
            latencies.append((time.perf_counter_ns() - t0) / 1000)
        latencies.sort()
        results[op] = {
            "min": round(latencies[0], 2),
            "avg": round(statistics.mean(latencies), 2),
            "max": round(latencies[-1], 2),
            "p99": round(latencies[int(n * 0.99)], 2),
        }
    
    # Batch of 10
    latencies = []
    for _ in range(n // 10):
        batch = [{"op": random.choice(["add", "sub", "mul"]), "a": random.randint(1, 50), "b": random.randint(1, 50)} for _ in range(10)]
        t0 = time.perf_counter_ns()
        _http_post(f"{base_url}/compute/batch", {"ops": batch})
        latencies.append((time.perf_counter_ns() - t0) / 1000)
    latencies.sort()
    results["batch_10"] = {
        "min": round(latencies[0], 2),
        "avg": round(statistics.mean(latencies), 2),
        "max": round(latencies[-1], 2),
        "p99": round(latencies[int(len(latencies) * 0.99)], 2),
    }
    
    # Obligation check
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        _http_post(f"{base_url}/obligation/check", {
            "last_run_epoch": 1000000,
            "now_epoch": 1000000 + random.randint(0, 7200),
            "interval_seconds": 3600,
        })
        latencies.append((time.perf_counter_ns() - t0) / 1000)
    latencies.sort()
    results["obligation_check"] = {
        "min": round(latencies[0], 2),
        "avg": round(statistics.mean(latencies), 2),
        "max": round(latencies[-1], 2),
        "p99": round(latencies[int(n * 0.99)], 2),
    }
    
    return results

def print_table(title, results):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  {'Op':<20} {'Min µs':>10} {'Avg µs':>10} {'Max µs':>10} {'P99 µs':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for op, stats in results.items():
        print(f"  {op:<20} {stats['min']:>10.1f} {stats['avg']:>10.1f} {stats['max']:>10.1f} {stats['p99']:>10.1f}")

def main():
    parser = argparse.ArgumentParser(description="nCPU Bridge Benchmark")
    parser.add_argument("--url", default="http://localhost:3950", help="Server URL")
    parser.add_argument("--n", type=int, default=1000, help="Ops per benchmark")
    parser.add_argument("--raw-only", action="store_true", help="Skip HTTP benchmark")
    parser.add_argument("--http-only", action="store_true", help="Skip raw benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()
    
    all_results = {}
    
    if not args.http_only:
        print("Running raw Python benchmark..." if not args.json else "", end="", flush=True)
        raw = bench_raw(args.n)
        all_results["raw"] = raw
        if not args.json:
            print_table("Raw Python (no HTTP overhead)", raw)
    
    if not args.raw_only:
        print("\nRunning HTTP benchmark..." if not args.json else "", end="", flush=True)
        http = bench_http(args.url, args.n)
        all_results["http"] = http
        if not args.json:
            print_table(f"HTTP ({args.url})", http)
    
    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        print(f"\n✅ Benchmark complete ({args.n} ops each)")

if __name__ == "__main__":
    main()
