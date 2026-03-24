#!/usr/bin/env python3
"""Full stack demo: kernel boot → filesystem → compile C → execute → DB store → query.

End-to-end: every operation runs through trained neural networks.
"""

import sys
from pathlib import Path

BRIDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BRIDGE_ROOT))

from bridge.neural_kernel import NeuralKernel
from bridge.c_compiler import NeuralCCompiler
from bridge.neural_db import NeuralDB


C_SOURCE = """\
int a = 6;
int b = 7;
int c = a * b;
return c;
"""

EXPECTED = 6 * 7


def run_demo():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              Full Neural Stack Demo                     ║")
    print("║  Boot → FS → Compile C → Execute → DB Store → Query    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ── Step 1: Boot kernel ──
    print("━━━ Step 1: Boot Neural Kernel ━━━")
    kernel = NeuralKernel()
    kernel.boot()
    print()

    # ── Step 2: Store C source in filesystem ──
    print("━━━ Step 2: Store C source in neural filesystem ━━━")
    fs = kernel._subsystems["fs"]
    fs.mkdir("/home/demo")
    fs.create("/home/demo/multiply.c", C_SOURCE)
    read_back = fs.read("/home/demo/multiply.c")
    print(f"  Wrote /home/demo/multiply.c ({len(C_SOURCE)} bytes)")
    print(f"  Read back: {read_back.get('data', '')[:40]}...")
    print()

    # ── Step 3: Compile C source ──
    print("━━━ Step 3: Compile C with neural compiler ━━━")
    compiler = NeuralCCompiler()
    result = compiler.compile_and_run(C_SOURCE)

    if not result["success"]:
        print(f"  ❌ Compile failed: {result.get('errors')}")
        return

    print(f"  Assembly:")
    for line in result["assembly"].split("\n"):
        print(f"    {line}")
    print(f"  Return value: {result['return_value']}")
    print(f"  Cycles: {result['cycles']}")
    print()

    # Store assembly in filesystem too
    fs.create("/home/demo/multiply.asm", result["assembly"])
    print(f"  Stored assembly to /home/demo/multiply.asm")
    print()

    # ── Step 4: Verify result ──
    print("━━━ Step 4: Verify against native Python ━━━")
    return_val = result["return_value"]
    match = return_val == EXPECTED
    print(f"  Neural result: {return_val}")
    print(f"  Python result: {EXPECTED}")
    print(f"  Match: {'✅' if match else '❌'}")
    print()

    # ── Step 5: Store result in neural DB ──
    print("━━━ Step 5: Store result in neural database ━━━")
    db = NeuralDB()
    db.create_index("program_id")
    db.insert(
        program_id=1,
        return_value=return_val,
        cycles=result["cycles"],
        verified=1 if match else 0,
    )
    db.insert(
        program_id=2,
        return_value=42,
        cycles=10,
        verified=1,
    )
    print(f"  Inserted 2 rows into neural DB")
    print()

    # ── Step 6: Query it back ──
    print("━━━ Step 6: Query results from neural DB ━━━")
    qr = db.select()
    print(f"  SELECT * → {qr.count} rows ({qr.scan_type}, {qr.neural_ops} neural ops)")
    for row in qr.rows:
        print(f"    id={row.id} {row.data}")

    # Query with filter
    qr2 = db.select(where={"verified": 1})
    print(f"  SELECT * WHERE verified=1 → {qr2.count} rows")

    # Aggregation
    agg = db.aggregate("return_value", "SUM")
    print(f"  SUM(return_value) = {agg}")
    print()

    # ── Summary ──
    print("━━━ Summary ━━━")
    print(f"  Kernel subsystems:  {len(kernel._subsystems)}")
    print(f"  Files created:      2 (/home/demo/multiply.c, .asm)")
    print(f"  C program compiled: 6 * 7 = {return_val}")
    print(f"  DB rows stored:     {qr.count}")
    print(f"  Verification:       {'✅ PASS' if match else '❌ FAIL'}")
    print(f"  Everything neural:  Yes")


if __name__ == "__main__":
    run_demo()
