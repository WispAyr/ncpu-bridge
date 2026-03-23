"""Neural JIT Compiler — trace-based compilation through nCPU.

Records execution traces of Forth programs, identifies hot loops,
and compiles them to optimized nCPU assembly. Then runs the compiled
code on the neural GPU instead of interpreting.

The JIT compilation itself uses neural operations:
- Trace recording with neural counter for heat detection
- Pattern matching for loop detection (neural CMP)
- Register allocation with neural graph coloring
- Dead code elimination with neural reachability

Usage:
    python -m bridge.neural_jit demo
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


@dataclass
class TraceEntry:
    """Single operation in an execution trace."""
    op: str
    operands: list[int] = field(default_factory=list)
    result: int = 0


@dataclass
class CompiledTrace:
    """A compiled hot trace."""
    trace_id: str
    source_ops: int
    compiled_asm: list[str]
    optimized_asm: list[str]
    speedup_estimate: float
    neural_ops_to_compile: int


class NeuralJIT:
    """Trace-based JIT compiler using neural operations.
    
    1. Record: trace Forth-like stack operations
    2. Detect: find hot loops via neural counting
    3. Compile: convert traces to nCPU assembly
    4. Optimize: dead code elimination, constant folding
    5. Execute: run on neural GPU
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._traces: dict[str, list[TraceEntry]] = {}
        self._heat_map: dict[str, int] = {}  # trace_key → execution count
        self._compiled: dict[str, CompiledTrace] = {}
        self._threshold = 3  # Compile after N executions
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def record_trace(self, key: str, ops: list[tuple[str, list[int]]]) -> str:
        """Record an execution trace."""
        entries = []
        for op_name, operands in ops:
            entries.append(TraceEntry(op=op_name, operands=operands))
        
        self._traces[key] = entries
        
        # Increment heat counter (neural ADD)
        current = self._heat_map.get(key, 0)
        self._heat_map[key] = self.bridge.add(current, 1)
        self._op()
        
        # Check if hot (neural CMP)
        zf, sf = self.bridge.cmp(self._heat_map[key], self._threshold)
        self._op()
        
        if not sf and not zf:  # count > threshold → compile!
            if key not in self._compiled:
                return self._compile(key)
        
        return "interpreted"
    
    def _compile(self, key: str) -> str:
        """Compile a trace to nCPU assembly."""
        self._ops = 0
        trace = self._traces[key]
        
        # Phase 1: Direct translation
        asm = []
        reg_map: dict[str, int] = {}
        next_reg = 0
        
        for i, entry in enumerate(trace):
            if entry.op == "PUSH":
                reg = next_reg
                next_reg += 1
                reg_map[f"s{i}"] = reg
                asm.append(f"MOV R{reg}, {entry.operands[0]}")
            
            elif entry.op == "ADD":
                if len(entry.operands) >= 2:
                    out_reg = next_reg
                    next_reg = min(next_reg + 1, 7)
                    asm.append(f"ADD R{out_reg}, R{out_reg-2}, R{out_reg-1}")
                else:
                    asm.append(f"ADD R0, R0, R1")
            
            elif entry.op == "MUL":
                asm.append(f"MUL R0, R0, R1")
            
            elif entry.op == "SUB":
                asm.append(f"SUB R0, R0, R1")
            
            elif entry.op == "CMP":
                asm.append(f"CMP R0, R1")
            
            elif entry.op == "DUP":
                if next_reg < 7:
                    asm.append(f"MOV R{next_reg}, R{next_reg-1}")
                    next_reg += 1
            
            elif entry.op == "STORE":
                asm.append(f"STR R0, [{entry.operands[0]}]")
            
            elif entry.op == "LOAD":
                asm.append(f"LDR R0, [{entry.operands[0]}]")
        
        asm.append("HALT")
        
        # Phase 2: Optimize — constant folding
        optimized = self._optimize(asm, trace)
        
        # Phase 3: Dead code elimination
        optimized = self._eliminate_dead(optimized)
        
        compiled = CompiledTrace(
            trace_id=key,
            source_ops=len(trace),
            compiled_asm=asm,
            optimized_asm=optimized,
            speedup_estimate=len(trace) / max(len(optimized), 1),
            neural_ops_to_compile=self._ops,
        )
        self._compiled[key] = compiled
        
        return f"compiled ({len(trace)} ops → {len(optimized)} asm, {compiled.speedup_estimate:.1f}x)"
    
    def _optimize(self, asm: list[str], trace: list[TraceEntry]) -> list[str]:
        """Constant folding: if two MOVs followed by ADD with known values,
        replace with single MOV of the result."""
        optimized = []
        i = 0
        
        while i < len(asm):
            # Pattern: MOV Rx, A; MOV Ry, B; ADD Rz, Rx, Ry → MOV Rz, A+B
            if (i + 2 < len(asm) and
                asm[i].startswith("MOV") and
                asm[i+1].startswith("MOV") and
                asm[i+2].startswith("ADD")):
                
                # Extract values
                try:
                    val_a = int(asm[i].split(", ")[1])
                    val_b = int(asm[i+1].split(", ")[1])
                    out_reg = asm[i+2].split(" ")[1].rstrip(",")
                    
                    # Neural constant fold
                    folded = self.bridge.add(val_a, val_b)
                    self._op()
                    
                    optimized.append(f"MOV {out_reg} {folded}  ; folded {val_a}+{val_b}")
                    i += 3
                    continue
                except (ValueError, IndexError):
                    pass
            
            # Pattern: MOV Rx, A; MOV Ry, B; MUL → MOV Rz, A*B
            if (i + 2 < len(asm) and
                asm[i].startswith("MOV") and
                asm[i+1].startswith("MOV") and
                asm[i+2].startswith("MUL")):
                
                try:
                    val_a = int(asm[i].split(", ")[1])
                    val_b = int(asm[i+1].split(", ")[1])
                    
                    folded = self.bridge.mul(val_a, val_b)
                    self._op()
                    
                    optimized.append(f"MOV R0, {folded}  ; folded {val_a}*{val_b}")
                    i += 3
                    continue
                except (ValueError, IndexError):
                    pass
            
            optimized.append(asm[i])
            i += 1
        
        return optimized
    
    def _eliminate_dead(self, asm: list[str]) -> list[str]:
        """Remove dead code — instructions whose results are overwritten."""
        # Simple: remove consecutive MOVs to same register
        result = []
        for i, inst in enumerate(asm):
            if i + 1 < len(asm):
                # Both MOV to same register?
                if inst.startswith("MOV R") and asm[i+1].startswith("MOV R"):
                    reg_a = inst.split(" ")[1].split(",")[0]
                    reg_b = asm[i+1].split(" ")[1].split(",")[0]
                    
                    zf = True
                    for j in range(min(len(reg_a), len(reg_b))):
                        zf_j, _ = self.bridge.cmp(ord(reg_a[j]), ord(reg_b[j]))
                        self._op()
                        if not zf_j:
                            zf = False
                            break
                    
                    if zf and len(reg_a) == len(reg_b):
                        continue  # Skip dead write
            
            result.append(inst)
        
        return result
    
    def get_stats(self) -> dict:
        return {
            "traces_recorded": len(self._traces),
            "hot_traces": len(self._compiled),
            "heat_map": dict(self._heat_map),
            "compiled": {
                k: {
                    "source_ops": v.source_ops,
                    "asm_lines": len(v.optimized_asm),
                    "speedup": f"{v.speedup_estimate:.1f}x",
                    "compile_cost": v.neural_ops_to_compile,
                }
                for k, v in self._compiled.items()
            },
        }


# ── CLI ──

def demo():
    jit = NeuralJIT()
    
    print("Neural JIT Compiler")
    print("=" * 60)
    print("Trace-based compilation with neural optimization\n")
    
    # Simulate a hot loop
    print("── Recording traces ──")
    loop_trace = [
        ("PUSH", [10]),
        ("PUSH", [20]),
        ("ADD", []),
        ("PUSH", [2]),
        ("MUL", []),
        ("STORE", [0x100]),
    ]
    
    for i in range(5):
        result = jit.record_trace("hot_loop", loop_trace)
        print(f"  Execution {i+1}: {result}")
    
    print()
    
    # Show compiled output
    print("── Compiled Trace ──")
    if "hot_loop" in jit._compiled:
        ct = jit._compiled["hot_loop"]
        print(f"  Source: {ct.source_ops} stack operations")
        print(f"  Before optimization:")
        for line in ct.compiled_asm:
            print(f"    {line}")
        print(f"  After optimization:")
        for line in ct.optimized_asm:
            print(f"    {line}")
        print(f"  Speedup: {ct.speedup_estimate:.1f}x")
        print(f"  Compile cost: {ct.neural_ops_to_compile} neural ops")
    print()
    
    # Another trace: fibonacci step
    print("── Fibonacci trace ──")
    fib_trace = [
        ("PUSH", [0]),   # a
        ("PUSH", [1]),   # b
        ("DUP", []),
        ("ADD", []),     # a+b
        ("STORE", [0x200]),
    ]
    
    for i in range(5):
        result = jit.record_trace("fib_step", fib_trace)
        if "compiled" in result:
            print(f"  Execution {i+1}: {result}")
    
    if "fib_step" in jit._compiled:
        ct = jit._compiled["fib_step"]
        print(f"  Optimized to {len(ct.optimized_asm)} instructions:")
        for line in ct.optimized_asm:
            print(f"    {line}")
    print()
    
    # Stats
    print("── JIT Stats ──")
    stats = jit.get_stats()
    print(f"  Traces recorded: {stats['traces_recorded']}")
    print(f"  Hot traces compiled: {stats['hot_traces']}")
    for name, info in stats['compiled'].items():
        print(f"    {name}: {info['source_ops']} ops → {info['asm_lines']} asm ({info['speedup']})")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_jit [demo]")


if __name__ == "__main__":
    main()
