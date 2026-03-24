#!/usr/bin/env python3
"""Neural Mesh Demo — distributed inference across two Raspberry Pis.

Runs from the Mac. Demonstrates:
1. Load balancing across nodes
2. Latency: single node vs distributed
3. Fault tolerance (one node goes down)

Usage:
    python3 demos/neural_mesh_demo.py
"""

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.error import URLError

# ── Configuration ───────────────────────────────────────────────────
NODES = [
    {"id": "pi-alpha", "url": "http://192.168.195.238:9200"},
    {"id": "pi-bravo", "url": "http://192.168.195.238:9201"},
]

OPERATIONS = [
    ("add", 42, 58),
    ("sub", 100, 37),
    ("mul", 12, 8),
    ("div", 144, 12),
    ("cmp", 50, 50),
    ("and", 0xFF, 0x0F),
    ("or", 0xA0, 0x05),
    ("xor", 0xFF, 0xAA),
    ("add", 1000, 2000),
    ("mul", 7, 7),
    ("sub", 255, 128),
    ("div", 100, 10),
    ("add", 0, 0),
    ("sub", 0, 0),
    ("cmp", 100, 200),
    ("and", 0xAA, 0x55),
    ("or", 0x0F, 0xF0),
    ("xor", 0xFF, 0xFF),
    ("mul", 15, 15),
    ("add", 127, 128),
]


def rpc(node_url: str, operation: str, a: int, b: int, timeout: float = 5.0) -> dict:
    """Send a compute request to a mesh node."""
    payload = json.dumps({"operation": operation, "a": a, "b": b}).encode()
    req = Request(
        f"{node_url}/compute",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    resp = urlopen(req, timeout=timeout)
    rtt_ms = (time.perf_counter() - t0) * 1000
    result = json.loads(resp.read())
    result["rtt_ms"] = round(rtt_ms, 2)
    return result


def health_check(node: dict) -> dict:
    """Check if a node is alive."""
    try:
        resp = urlopen(f"{node['url']}/health", timeout=3)
        data = json.loads(resp.read())
        return {"id": node["id"], "alive": True, **data}
    except Exception as e:
        return {"id": node["id"], "alive": False, "error": str(e)}


def banner(title: str):
    w = 60
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


def demo_health():
    """Check all nodes."""
    banner("1. MESH HEALTH CHECK")
    alive = []
    for node in NODES:
        h = health_check(node)
        status = "✅ ONLINE" if h["alive"] else "❌ OFFLINE"
        print(f"  {node['id']:12s} {node['url']:30s} {status}")
        if h["alive"]:
            print(f"    hostname: {h.get('hostname', '?')}, models loaded: {h.get('models_loaded', '?')}, ops served: {h.get('ops_served', 0)}")
            alive.append(node)
    return alive


def demo_single_node(alive_nodes: list) -> list:
    """Run all ops on a single node, measure total time."""
    banner("2. SINGLE NODE BENCHMARK")
    node = alive_nodes[0]
    print(f"  Target: {node['id']} ({node['url']})")
    results = []
    t0 = time.perf_counter()
    for op, a, b in OPERATIONS:
        try:
            r = rpc(node["url"], op, a, b)
            results.append(r)
            print(f"    {op:5s}({a}, {b}) = {r['result']:>12.2f}  rtt={r['rtt_ms']:6.1f}ms  inference={r['inference_ms']:5.2f}ms  [{r['node_id']}]")
        except Exception as e:
            print(f"    {op:5s}({a}, {b}) = ERROR: {e}")
    total = (time.perf_counter() - t0) * 1000
    print(f"\n  Total: {total:.1f}ms for {len(OPERATIONS)} ops (sequential, single node)")
    print(f"  Avg RTT: {sum(r['rtt_ms'] for r in results) / max(len(results),1):.1f}ms")
    return results


def demo_distributed(alive_nodes: list) -> list:
    """Distribute ops across nodes with round-robin + threading."""
    banner("3. DISTRIBUTED BENCHMARK (parallel)")
    if len(alive_nodes) < 2:
        print("  ⚠️  Only 1 node alive — skipping distributed comparison")
        return []

    results = []
    t0 = time.perf_counter()

    def run_one(idx_op):
        idx, (op, a, b) = idx_op
        node = alive_nodes[idx % len(alive_nodes)]
        return rpc(node["url"], op, a, b)

    with ThreadPoolExecutor(max_workers=len(alive_nodes)) as pool:
        futures = {pool.submit(run_one, (i, op)): (i, op) for i, op in enumerate(OPERATIONS)}
        for f in as_completed(futures):
            i, (op, a, b) = futures[f]
            try:
                r = f.result()
                results.append(r)
                print(f"    {op:5s}({a}, {b}) = {r['result']:>12.2f}  rtt={r['rtt_ms']:6.1f}ms  [{r['node_id']}]")
            except Exception as e:
                print(f"    {op}({a}, {b}) = ERROR: {e}")

    total = (time.perf_counter() - t0) * 1000
    print(f"\n  Total: {total:.1f}ms for {len(OPERATIONS)} ops (distributed, parallel)")
    print(f"  Avg RTT: {sum(r['rtt_ms'] for r in results) / max(len(results),1):.1f}ms")

    # Per-node breakdown
    by_node = {}
    for r in results:
        nid = r.get("node_id", "?")
        by_node.setdefault(nid, []).append(r)
    for nid, rs in by_node.items():
        print(f"  Node {nid}: {len(rs)} ops, avg RTT {sum(r['rtt_ms'] for r in rs)/len(rs):.1f}ms")

    return results


def demo_fault_tolerance(alive_nodes: list):
    """Show graceful degradation when a node is unreachable."""
    banner("4. FAULT TOLERANCE")
    fake_node = {"id": "pi-ghost", "url": "http://192.168.195.238:9999"}
    test_nodes = [fake_node] + alive_nodes

    print(f"  Nodes: {[n['id'] for n in test_nodes]}")
    print(f"  pi-ghost is intentionally unreachable\n")

    successes = 0
    failures = 0

    for i, (op, a, b) in enumerate(OPERATIONS[:6]):
        node = test_nodes[i % len(test_nodes)]
        try:
            r = rpc(node["url"], op, a, b, timeout=2.0)
            print(f"    ✅ {op}({a},{b}) on {node['id']}: {r['result']:.2f}")
            successes += 1
        except Exception:
            # Fallback to next alive node
            fallback = alive_nodes[0]
            try:
                r = rpc(fallback["url"], op, a, b)
                print(f"    🔄 {op}({a},{b}) on {node['id']} FAILED → fallback {fallback['id']}: {r['result']:.2f}")
                successes += 1
            except Exception as e2:
                print(f"    ❌ {op}({a},{b}) TOTAL FAILURE: {e2}")
                failures += 1

    print(f"\n  Results: {successes} succeeded, {failures} failed")
    print(f"  Fault tolerance: {'PASSED ✅' if failures == 0 else 'DEGRADED ⚠️'}")


def demo_latency_comparison(single_results: list, dist_results: list):
    """Compare single vs distributed performance."""
    banner("5. LATENCY COMPARISON")
    if not dist_results:
        print("  No distributed results to compare")
        return

    single_total = sum(r["rtt_ms"] for r in single_results)
    dist_total = sum(r["rtt_ms"] for r in dist_results)
    # But distributed ran in parallel, so wall-clock is different
    single_avg = single_total / max(len(single_results), 1)
    dist_avg = dist_total / max(len(dist_results), 1)

    print(f"  Single node:  {len(single_results)} ops, avg RTT {single_avg:.1f}ms, total RTT {single_total:.1f}ms")
    print(f"  Distributed:  {len(dist_results)} ops, avg RTT {dist_avg:.1f}ms, total RTT {dist_total:.1f}ms")
    print(f"  Speedup (parallel wall-clock): operations ran concurrently across {len(NODES)} nodes")
    
    # Per-node inference time comparison
    single_inf = sum(r.get("inference_ms", 0) for r in single_results) / max(len(single_results), 1)
    dist_inf = sum(r.get("inference_ms", 0) for r in dist_results) / max(len(dist_results), 1)
    print(f"  Avg inference: single={single_inf:.2f}ms, distributed={dist_inf:.2f}ms")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         NEURAL MESH — Distributed nCPU Demo             ║")
    print("║         2 × Raspberry Pi + ONNX Runtime                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    alive = demo_health()
    if not alive:
        print("\n❌ No nodes available. Exiting.")
        sys.exit(1)

    single_results = demo_single_node(alive)
    dist_results = demo_distributed(alive)
    demo_fault_tolerance(alive)
    demo_latency_comparison(single_results, dist_results)

    banner("DEMO COMPLETE")
    print(f"  Nodes tested: {len(alive)}/{len(NODES)}")
    print(f"  Operations: {len(OPERATIONS)}")
    print()


if __name__ == "__main__":
    main()
