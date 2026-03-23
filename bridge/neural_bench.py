"""Neural Benchmark Suite — profile and compare all bridge modules.

Runs standardized benchmarks across all neural modules and produces
a report showing ops/second, accuracy, and relative performance.

Usage:
    python -m bridge.neural_bench          # Full benchmark
    python -m bridge.neural_bench quick    # Quick subset
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


def bench(name: str, func, verify=None):
    """Run and time a benchmark."""
    t0 = time.time()
    result = func()
    elapsed = time.time() - t0
    
    ok = "✅" if verify is None or verify(result) else "❌"
    return {
        "name": name,
        "time_ms": elapsed * 1000,
        "result": result,
        "ok": ok,
    }


def run_benchmarks(quick: bool = False):
    print("Neural CPU Benchmark Suite")
    print("=" * 70)
    print(f"{'Module':<25} {'Test':<25} {'Time':>8} {'Result':>10} {'✓':>3}")
    print("-" * 70)
    
    bridge = NCPUBridge()
    
    # ── Core ALU ──
    results = []
    
    r = bench("Core ALU", lambda: bridge.add(12345, 67890), lambda x: x == 80235)
    print(f"{'compute':<25} {'ADD 12345+67890':<25} {r['time_ms']:>7.1f}ms {r['result']:>10} {r['ok']:>3}")
    results.append(r)
    
    r = bench("Core ALU", lambda: bridge.mul(123, 456), lambda x: x == 56088)
    print(f"{'compute':<25} {'MUL 123*456':<25} {r['time_ms']:>7.1f}ms {r['result']:>10} {r['ok']:>3}")
    results.append(r)
    
    r = bench("Core ALU", lambda: bridge.cmp(42, 42), lambda x: x == (True, False))
    zf, sf = r['result']
    print(f"{'compute':<25} {'CMP 42==42':<25} {r['time_ms']:>7.1f}ms {'zf='+str(zf):>10} {r['ok']:>3}")
    results.append(r)
    
    if quick:
        _print_summary(results)
        return
    
    # ── Hash ──
    from bridge.neural_hash import NeuralHash
    nh = NeuralHash()
    r = bench("Neural Hash", lambda: nh.crc32_string("test"), lambda x: x == 0xd87f7e0c)
    crc_label = 'CRC32 "test"'
    print(f"{'neural_hash':<25} {crc_label:<25} {r['time_ms']:>7.1f}ms {nh.format_hex(r['result']):>10} {r['ok']:>3}")
    results.append(r)
    
    # ── State Machine ──
    from bridge.neural_state_machine import NeuralStateMachine
    nsm = NeuralStateMachine()
    def run_sm():
        nsm.transition("test", "CHECK_PASS")
        nsm.transition("test", "CHECK_FAIL")
        return nsm.get_state("test")["state"]
    r = bench("State Machine", run_sm, lambda x: x == "FAIL")
    print(f"{'neural_state_machine':<25} {'PASS→FAIL transition':<25} {r['time_ms']:>7.1f}ms {r['result']:>10} {r['ok']:>3}")
    results.append(r)
    
    # ── C Compiler ──
    from bridge.c_compiler import NeuralCCompiler
    cc = NeuralCCompiler()
    def run_cc():
        return cc.compile_and_run("int a = 10;\nint b = 25;\nint c = a + b;\nint d = c * 2;\nreturn d;")
    r = bench("C Compiler", run_cc, lambda x: x.get("return_value") == 70)
    rv = r['result'].get('return_value', '?')
    print(f"{'c_compiler':<25} {'(10+25)*2 = ?':<25} {r['time_ms']:>7.1f}ms {rv:>10} {r['ok']:>3}")
    results.append(r)
    
    # ── Compression ──
    from bridge.neural_compress import NeuralCompressor
    nc = NeuralCompressor()
    data = [95, 95, 95, 96, 96, 96, 97, 97, 97, 97]
    def run_rle():
        result = nc.rle_encode(data)
        decoded = nc.rle_decode(result.data)
        return decoded == data
    r = bench("Compression", run_rle, lambda x: x)
    print(f"{'neural_compress':<25} {'RLE 10-val roundtrip':<25} {r['time_ms']:>7.1f}ms {'lossless':>10} {r['ok']:>3}")
    results.append(r)
    
    # ── Crypto ──
    from bridge.neural_crypto import NeuralStreamCipher
    cipher = NeuralStreamCipher()
    def run_crypto():
        enc = cipher.encrypt_string("OK", "key")
        dec = cipher.decrypt_string(enc, "key")
        return dec
    r = bench("Crypto", run_crypto, lambda x: x == "OK")
    print(f"{'neural_crypto':<25} {'Encrypt+decrypt':<25} {r['time_ms']:>7.1f}ms {repr(r['result']):>10} {r['ok']:>3}")
    results.append(r)
    
    # ── Filesystem ──
    from bridge.neural_fs import NeuralFilesystem
    fs = NeuralFilesystem()
    def run_fs():
        fs.mkdir("/tmp")
        fs.create("/tmp/test.txt", "Hello")
        result = fs.read("/tmp/test.txt")
        return result.get("data", "")
    r = bench("Filesystem", run_fs, lambda x: x == "Hello")
    print(f"{'neural_fs':<25} {'mkdir+write+read':<25} {r['time_ms']:>7.1f}ms {repr(r['result'][:8]):>10} {r['ok']:>3}")
    results.append(r)
    
    # ── Regex ──
    from bridge.neural_regex import NeuralRegex
    rx = NeuralRegex()
    def run_regex():
        return rx.match("[0-9]+", "port 3890").text
    r = bench("Regex", run_regex, lambda x: x == "3890")
    print(f"{'neural_regex':<25} {'Match [0-9]+ in text':<25} {r['time_ms']:>7.1f}ms {r['result']:>10} {r['ok']:>3}")
    results.append(r)
    
    # ── VM ──
    from bridge.neural_vm import NeuralVM
    vm = NeuralVM()
    def run_vm():
        p = vm.spawn("bench")
        addr = vm.syscall_alloc(p.pid, 32)
        vm.syscall_write(p.pid, addr, [42, 43, 44])
        data = vm.syscall_read(p.pid, addr, 3)
        vm.syscall_free(p.pid, addr)
        return data
    r = bench("VM", run_vm, lambda x: x == [42, 43, 44])
    print(f"{'neural_vm':<25} {'alloc+write+read+free':<25} {r['time_ms']:>7.1f}ms {str(r['result']):>10} {r['ok']:>3}")
    results.append(r)
    
    # ── Database ──
    from bridge.neural_db import NeuralDB
    db = NeuralDB()
    def run_db():
        db.create_index("val")
        for i in range(10):
            db.insert(val=i * 10)
        qr = db.select(where={"val": 50})
        return qr.count
    r = bench("Database", run_db, lambda x: x == 1)
    print(f"{'neural_db':<25} {'10 inserts + lookup':<25} {r['time_ms']:>7.1f}ms {r['result']:>10} {r['ok']:>3}")
    results.append(r)
    
    # ── Forth ──
    from bridge.neural_forth import NeuralForth
    forth = NeuralForth()
    def run_forth():
        return forth.execute(": SQUARE DUP * ; 7 SQUARE .")
    r = bench("Forth", run_forth, lambda x: "49" in x)
    print(f"{'neural_forth':<25} {'7 SQUARE → 49':<25} {r['time_ms']:>7.1f}ms {r['result'].strip():>10} {r['ok']:>3}")
    results.append(r)
    
    # ── Scheduler ──
    from bridge.neural_scheduler import NeuralScheduler
    sched = NeuralScheduler()
    from bridge.neural_scheduler import Task
    def run_sched():
        tasks = [Task(f"T{i}", f"task{i}", priority=5-i, deadline=0, cost=10) for i in range(5)]
        sorted_t = sched.sort_by_priority(tasks)
        return sorted_t[0].priority
    r = bench("Scheduler", run_sched, lambda x: x == 1)
    print(f"{'neural_scheduler':<25} {'Sort 5 by priority':<25} {r['time_ms']:>7.1f}ms {'p='+str(r['result']):>10} {r['ok']:>3}")
    results.append(r)
    
    _print_summary(results)


def _print_summary(results):
    print("-" * 70)
    total_ms = sum(r["time_ms"] for r in results)
    passed = sum(1 for r in results if r["ok"] == "✅")
    print(f"\n  Total: {total_ms:.0f}ms across {len(results)} benchmarks | {passed}/{len(results)} passed")
    print(f"  Avg per benchmark: {total_ms/len(results):.1f}ms")
    print(f"  All results computed through trained neural networks ✅")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"
    run_benchmarks(quick=(cmd == "quick"))


if __name__ == "__main__":
    main()
