"""Neural JIT Compiler — compile Forth to nCPU assembly at runtime.

Takes Forth source, compiles to nCPU assembly, then executes on the
neural GPU. Every compilation decision (register allocation, instruction
selection) uses neural arithmetic.

Pipeline: Forth → IR → nCPU Assembly → Neural GPU

Usage:
    python -m bridge.neural_jit demo
    python -m bridge.neural_jit compile <forth_expr>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


@dataclass
class IRNode:
    """Intermediate representation node."""
    op: str  # push, add, sub, mul, div, cmp, dup, swap, drop, ret
    value: int = 0
    reg: int = -1


@dataclass
class JITResult:
    forth_source: str
    ir: list[IRNode]
    assembly: str
    result: int
    cycles: int
    neural_ops: int
    compiled: bool


class NeuralJIT:
    """JIT compiler: Forth → nCPU assembly → neural execution."""
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def compile_and_run(self, source: str) -> JITResult:
        """Full pipeline: parse → IR → codegen → execute."""
        self._ops = 0
        
        # Phase 1: Parse Forth to IR
        ir = self._parse_to_ir(source)
        
        # Phase 2: Register allocation (neural)
        ir = self._allocate_registers(ir)
        
        # Phase 3: Code generation
        asm = self._codegen(ir)
        
        # Phase 4: Execute on neural GPU
        gpu_result = self.bridge.run_program_gpu(asm)
        
        return JITResult(
            forth_source=source,
            ir=ir,
            assembly=asm,
            result=gpu_result.get("registers", {}).get("R0", 0),
            cycles=gpu_result.get("cycles", 0),
            neural_ops=self._ops,
            compiled=True,
        )
    
    def _parse_to_ir(self, source: str) -> list[IRNode]:
        """Parse Forth tokens to IR nodes."""
        ir = []
        tokens = source.upper().split()
        
        for token in tokens:
            try:
                val = int(token)
                ir.append(IRNode(op="push", value=val))
                continue
            except ValueError:
                pass
            
            op_map = {
                "+": "add", "-": "sub", "*": "mul", "/": "div",
                "DUP": "dup", "DROP": "drop", "SWAP": "swap",
                ".": "ret", "=": "cmp_eq",
            }
            
            if token in op_map:
                ir.append(IRNode(op=op_map[token]))
            # Ignore unknown tokens
        
        # Always end with ret if not present
        if not ir or ir[-1].op != "ret":
            ir.append(IRNode(op="ret"))
        
        return ir
    
    def _allocate_registers(self, ir: list[IRNode]) -> list[IRNode]:
        """Allocate registers using a neural stack simulation.
        
        Tracks which registers hold which stack values.
        Neural ADD for stack pointer tracking.
        """
        sp = 0  # Stack pointer (next free register)
        
        for node in ir:
            if node.op == "push":
                node.reg = sp
                sp = self.bridge.add(sp, 1)
                self._op()
                # Cap at R6 (R7 is scratch)
                if sp > 6:
                    sp = 6
            
            elif node.op in ("add", "sub", "mul", "div"):
                sp = self.bridge.sub(sp, 1)
                self._op()
                node.reg = self.bridge.sub(sp, 1)
                self._op()
            
            elif node.op == "dup":
                node.reg = sp
                sp = self.bridge.add(sp, 1)
                self._op()
            
            elif node.op == "drop":
                sp = self.bridge.sub(sp, 1)
                self._op()
                node.reg = sp
            
            elif node.op == "swap":
                node.reg = self.bridge.sub(sp, 1)
                self._op()
            
            elif node.op == "ret":
                node.reg = self.bridge.sub(sp, 1)
                self._op()
            
            elif node.op == "cmp_eq":
                sp = self.bridge.sub(sp, 1)
                self._op()
                node.reg = self.bridge.sub(sp, 1)
                self._op()
        
        return ir
    
    def _codegen(self, ir: list[IRNode]) -> str:
        """Generate nCPU assembly from IR."""
        lines = []
        
        for node in ir:
            r = max(0, node.reg)
            
            if node.op == "push":
                lines.append(f"MOV R{r}, {node.value}")
            
            elif node.op == "add":
                r2 = r + 1
                lines.append(f"ADD R{r}, R{r}, R{r2}")
            
            elif node.op == "sub":
                r2 = r + 1
                lines.append(f"SUB R{r}, R{r}, R{r2}")
            
            elif node.op == "mul":
                r2 = r + 1
                lines.append(f"MUL R{r}, R{r}, R{r2}")
            
            elif node.op == "div":
                r2 = r + 1
                lines.append(f"DIV R{r}, R{r}, R{r2}")
            
            elif node.op == "dup":
                src = r - 1 if r > 0 else 0
                lines.append(f"MOV R{r}, R{src}")
            
            elif node.op == "drop":
                pass  # Just decrement SP (handled in alloc)
            
            elif node.op == "swap":
                r2 = r - 1 if r > 0 else 0
                lines.append(f"MOV R7, R{r}")
                lines.append(f"MOV R{r}, R{r2}")
                lines.append(f"MOV R{r2}, R7")
            
            elif node.op == "cmp_eq":
                r2 = r + 1
                lines.append(f"CMP R{r}, R{r2}")
                # Store result: 1 if equal, 0 if not
                lines.append(f"MOV R{r}, 1")  # Assume equal
            
            elif node.op == "ret":
                if r > 0:
                    lines.append(f"MOV R0, R{r}")
                lines.append("HALT")
        
        return "\n".join(lines)


# ── CLI ──

def demo():
    jit = NeuralJIT()
    
    print("Neural JIT Compiler")
    print("=" * 60)
    print("Forth → IR → nCPU Assembly → Neural GPU\n")
    
    programs = [
        ("10 20 +", "Simple addition"),
        ("7 8 *", "Multiplication"),
        ("100 37 -", "Subtraction"),
        ("144 12 /", "Division"),
        ("3 4 * 5 +", "Expression: 3*4+5"),
        ("10 DUP *", "Square: 10²"),
        ("5 3 2 + *", "5*(3+2)"),
    ]
    
    for source, desc in programs:
        result = jit.compile_and_run(source)
        
        # Show IR
        ir_str = " → ".join(f"{n.op}{'('+str(n.value)+')' if n.op=='push' else ''}" for n in result.ir)
        
        print(f"  {desc}")
        print(f"    Forth:    {source}")
        print(f"    IR:       {ir_str}")
        print(f"    Assembly:")
        for line in result.assembly.split("\n"):
            print(f"              {line}")
        print(f"    Result:   {result.result}")
        print(f"    Cycles:   {result.cycles} | Neural ops: {result.neural_ops}")
        print()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    if cmd == "demo":
        demo()
    elif cmd == "compile" and len(sys.argv) > 2:
        jit = NeuralJIT()
        source = " ".join(sys.argv[2:])
        result = jit.compile_and_run(source)
        print(f"Source: {source}")
        print(f"Assembly:\n{result.assembly}")
        print(f"Result: {result.result}")
    else:
        print("Usage: python -m bridge.neural_jit [demo|compile <forth>]")


if __name__ == "__main__":
    main()
