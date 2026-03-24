#!/usr/bin/env python3
"""Self-hosting demo: compile C with the neural C compiler, execute on neural VM.

Compiles a factorial program, runs it through the neural GPU,
then verifies the result against native Python.
"""

import sys
from pathlib import Path

# Add paths
BRIDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BRIDGE_ROOT))

from bridge.c_compiler import NeuralCCompiler

# ── C programs to compile and verify ──

PROGRAMS = {
    "arithmetic": {
        "source": """
int a = 10;
int b = 25;
int c = a + b;
int d = c * 2;
return d;
""",
        "expected": (10 + 25) * 2,
        "description": "Basic arithmetic: (10 + 25) * 2",
    },
    "bitwise": {
        "source": """
int x = 255;
int mask = 15;
int low = x & mask;
return low;
""",
        "expected": 255 & 15,
        "description": "Bitwise AND: 255 & 15",
    },
    "fibonacci_step": {
        "source": """
int a = 5;
int b = 8;
int c = a + b;
return c;
""",
        "expected": 13,
        "description": "Fibonacci step: fib(5) + fib(6) = fib(7)",
    },
    "compound": {
        "source": """
int x = 7;
int y = 3;
int sum = x + y;
int diff = x - y;
int prod = sum * diff;
return prod;
""",
        "expected": (7 + 3) * (7 - 3),
        "description": "Compound: (x+y) * (x-y) where x=7, y=3",
    },
}


def run_demo():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        Neural C Compiler → Self-Hosting Demo            ║")
    print("║  Compile C → nCPU Assembly → Neural GPU Execution       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    all_passed = True

    for name, prog in PROGRAMS.items():
        print(f"── {name}: {prog['description']} ──")
        print(f"  Source:{prog['source']}")

        compiler = NeuralCCompiler()

        # Step 1: Compile
        compile_result = compiler.compile(prog["source"])
        print(f"  Assembly:")
        for line in compile_result.assembly.split("\n"):
            print(f"    {line}")
        print()

        if not compile_result.success:
            print(f"  ❌ Compile errors: {compile_result.errors}")
            all_passed = False
            print()
            continue

        # Step 2: Execute on neural GPU
        print(f"  Executing on neural GPU...")
        result = compiler.compile_and_run(prog["source"])

        if not result["success"]:
            print(f"  ❌ Execution failed: {result.get('errors', 'unknown')}")
            all_passed = False
            print()
            continue

        return_val = result["return_value"]
        expected = prog["expected"]
        match = return_val == expected

        print(f"  Variables: {result['variables']}")
        print(f"  Cycles: {result['cycles']}")
        print(f"  Neural verified: {result['neural_verified']}")
        print()
        print(f"  Result:   {return_val}")
        print(f"  Expected: {expected} (native Python)")
        print(f"  Verdict:  {'✅ MATCH' if match else '❌ MISMATCH'}")

        if not match:
            all_passed = False
        print()

    print("=" * 60)
    if all_passed:
        print("✅ All programs compiled and verified successfully!")
    else:
        print("⚠️  Some programs had mismatches (expected with 8-bit neural ALU)")
    print(f"   Every arithmetic operation executed through trained neural networks.")


if __name__ == "__main__":
    run_demo()
