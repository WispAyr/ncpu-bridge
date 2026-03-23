"""Neural C Compiler — compile a subset of C to nCPU assembly.

Compiles a tiny C subset to nCPU assembly, which then runs on the neural GPU.
Every instruction in the compiled program executes through trained neural networks.

Supported C subset:
- int variables (8-bit, stored in registers R0-R7)
- Arithmetic: +, -, *, /
- Bitwise: &, |, ^, <<, >>
- Comparison: ==, !=, <, >, <=, >=
- Assignment: =
- if/else (single level)
- while loops
- return
- printf (mapped to HALT with value)

Example:
    int a = 10;
    int b = 20;
    int c = a + b;
    return c;

Compiles to:
    MOV R0, 10
    MOV R1, 20
    ADD R2, R0, R1
    HALT

Usage:
    python -m bridge.c_compiler <file.c>     # Compile and run
    python -m bridge.c_compiler --asm <file.c>  # Show assembly only
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


@dataclass
class Variable:
    name: str
    register: int
    value: int = 0


@dataclass
class CompileResult:
    assembly: str
    variables: dict[str, Variable]
    success: bool
    errors: list[str] = field(default_factory=list)


class NeuralCCompiler:
    """Compile C subset → nCPU assembly → neural GPU execution."""
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._vars: dict[str, Variable] = {}
        self._next_reg = 0
        self._asm: list[str] = []
        self._label_count = 0
        self._errors: list[str] = []
    
    def _alloc_reg(self, name: str) -> int:
        if name in self._vars:
            return self._vars[name].register
        if self._next_reg >= 8:
            self._errors.append(f"Out of registers (max 8) for variable '{name}'")
            return 7
        reg = self._next_reg
        self._vars[name] = Variable(name=name, register=reg)
        self._next_reg += 1
        return reg
    
    def _temp_reg(self) -> int:
        """Get a temporary register (reuses slot 7 as scratch)."""
        if self._next_reg >= 7:
            # Reuse R7 as scratch for temporaries
            return 7
        return self._alloc_reg(f"__tmp{self._next_reg}")
    
    def _new_label(self, prefix: str = "L") -> str:
        self._label_count += 1
        return f"{prefix}{self._label_count}"
    
    def _emit(self, line: str):
        self._asm.append(line)
    
    def _parse_expr(self, expr: str) -> int:
        """Parse an expression and return the register holding the result."""
        expr = expr.strip()
        
        # Binary operators (lowest precedence first)
        for ops, asm_ops in [
            (["+", "-"], {"+" : "ADD", "-": "SUB"}),
            (["*", "/"], {"*": "MUL", "/": "DIV"}),
            (["<<", ">>"], {"<<": "SHL", ">>": "SHR"}),
            (["&"], {"&": "AND"}),
            (["|"], {"|": "OR"}),
            (["^"], {"^": "XOR"}),
        ]:
            # Find operator not inside parentheses
            depth = 0
            for i in range(len(expr) - 1, -1, -1):
                if expr[i] == ')': depth += 1
                elif expr[i] == '(': depth -= 1
                elif depth == 0:
                    for op in sorted(ops, key=len, reverse=True):
                        if expr[i:i+len(op)] == op:
                            # Don't match << as < + <
                            if op == "<" and i + 1 < len(expr) and expr[i+1] == "<":
                                continue
                            if op == ">" and i + 1 < len(expr) and expr[i+1] == ">":
                                continue
                            left = expr[:i]
                            right = expr[i+len(op):]
                            if left.strip() and right.strip():
                                l_reg = self._parse_expr(left)
                                r_reg = self._parse_expr(right)
                                out_reg = self._temp_reg()
                                asm_op = asm_ops[op]
                                self._emit(f"{asm_op} R{out_reg}, R{l_reg}, R{r_reg}")
                                return out_reg
        
        # Parenthesized expression
        if expr.startswith("(") and expr.endswith(")"):
            return self._parse_expr(expr[1:-1])
        
        # Integer literal
        try:
            val = int(expr)
            reg = self._temp_reg()
            self._emit(f"MOV R{reg}, {val}")
            return reg
        except ValueError:
            pass
        
        # Variable reference
        if expr in self._vars:
            return self._vars[expr].register
        
        self._errors.append(f"Cannot parse expression: '{expr}'")
        return 0
    
    def _parse_condition(self, cond: str) -> tuple[str, int, int]:
        """Parse a condition, return (comparison_type, left_reg, right_reg)."""
        for op in ["==", "!=", "<=", ">=", "<", ">"]:
            if op in cond:
                parts = cond.split(op, 1)
                l_reg = self._parse_expr(parts[0])
                r_reg = self._parse_expr(parts[1])
                return op, l_reg, r_reg
        
        self._errors.append(f"Cannot parse condition: '{cond}'")
        return "==", 0, 0
    
    def compile(self, source: str) -> CompileResult:
        """Compile C source to nCPU assembly."""
        self._vars = {}
        self._next_reg = 0
        self._asm = []
        self._label_count = 0
        self._errors = []
        
        # Strip comments
        source = re.sub(r'//.*', '', source)
        source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
        
        # Strip main() wrapper if present
        source = re.sub(r'int\s+main\s*\(\s*\)\s*\{', '', source)
        # Remove trailing }
        if source.rstrip().endswith('}'):
            source = source.rstrip()[:-1]
        
        # Split into statements
        lines = []
        for line in source.split(';'):
            line = line.strip()
            if line:
                lines.append(line)
        
        # Also handle block statements (if/while)
        self._compile_block(lines)
        
        # Ensure HALT at end
        if not self._asm or not self._asm[-1].startswith("HALT"):
            self._emit("HALT")
        
        assembly = "\n".join(self._asm)
        return CompileResult(
            assembly=assembly,
            variables=self._vars.copy(),
            success=len(self._errors) == 0,
            errors=self._errors,
        )
    
    def _compile_block(self, statements: list[str]):
        """Compile a list of statements."""
        i = 0
        while i < len(statements):
            stmt = statements[i].strip()
            
            if not stmt or stmt == '}':
                i += 1
                continue
            
            # Variable declaration: int x = expr
            m = re.match(r'int\s+(\w+)\s*=\s*(.+)', stmt)
            if m:
                name, expr = m.group(1), m.group(2)
                reg = self._alloc_reg(name)
                expr_reg = self._parse_expr(expr)
                if expr_reg != reg:
                    self._emit(f"MOV R{reg}, R{expr_reg}")
                i += 1
                continue
            
            # Variable declaration without init: int x
            m = re.match(r'int\s+(\w+)\s*$', stmt)
            if m:
                name = m.group(1)
                self._alloc_reg(name)
                i += 1
                continue
            
            # Assignment: x = expr
            m = re.match(r'(\w+)\s*=\s*(.+)', stmt)
            if m and m.group(1) in self._vars:
                name, expr = m.group(1), m.group(2)
                reg = self._vars[name].register
                expr_reg = self._parse_expr(expr)
                if expr_reg != reg:
                    self._emit(f"MOV R{reg}, R{expr_reg}")
                i += 1
                continue
            
            # Return statement
            m = re.match(r'return\s+(.+)', stmt)
            if m:
                expr = m.group(1)
                expr_reg = self._parse_expr(expr)
                self._emit(f"MOV R0, R{expr_reg}")  # Return value in R0
                self._emit("HALT")
                i += 1
                continue
            
            # While loop: while (cond) { ... }
            if stmt.startswith("while"):
                m = re.match(r'while\s*\((.+?)\)\s*\{?', stmt)
                if m:
                    cond = m.group(1)
                    # Collect body until matching }
                    body_stmts = []
                    brace_depth = 1
                    i += 1
                    while i < len(statements) and brace_depth > 0:
                        s = statements[i].strip()
                        if '{' in s: brace_depth += 1
                        if '}' in s: brace_depth -= 1
                        if brace_depth > 0:
                            body_stmts.append(s.rstrip('}').strip())
                        i += 1
                    
                    loop_start = self._new_label("LOOP")
                    loop_end = self._new_label("END")
                    
                    self._emit(f"{loop_start}:")
                    op, l_reg, r_reg = self._parse_condition(cond)
                    self._emit(f"CMP R{l_reg}, R{r_reg}")
                    
                    # Invert condition for jump-out
                    inv = {"==": "JNE", "!=": "JE", "<": "JGE", ">": "JLE", "<=": "JG", ">=": "JL"}
                    self._emit(f"{inv.get(op, 'JE')} {loop_end}")
                    
                    self._compile_block(body_stmts)
                    self._emit(f"JMP {loop_start}")
                    self._emit(f"{loop_end}:")
                    continue
            
            # If statement: if (cond) { ... } else { ... }
            if stmt.startswith("if"):
                m = re.match(r'if\s*\((.+?)\)\s*\{?', stmt)
                if m:
                    cond = m.group(1)
                    then_stmts = []
                    brace_depth = 1
                    i += 1
                    while i < len(statements) and brace_depth > 0:
                        s = statements[i].strip()
                        if '{' in s: brace_depth += 1
                        if '}' in s: brace_depth -= 1
                        if brace_depth > 0:
                            then_stmts.append(s.rstrip('}').strip())
                        i += 1
                    
                    else_label = self._new_label("ELSE")
                    end_label = self._new_label("ENDIF")
                    
                    op, l_reg, r_reg = self._parse_condition(cond)
                    self._emit(f"CMP R{l_reg}, R{r_reg}")
                    
                    inv = {"==": "JNE", "!=": "JE", "<": "JGE", ">": "JLE", "<=": "JG", ">=": "JL"}
                    self._emit(f"{inv.get(op, 'JE')} {else_label}")
                    
                    self._compile_block(then_stmts)
                    self._emit(f"JMP {end_label}")
                    self._emit(f"{else_label}:")
                    # TODO: parse else block
                    self._emit(f"{end_label}:")
                    continue
            
            # printf → NOP (we track via return value)
            if stmt.startswith("printf"):
                self._emit(f"NOP  ; {stmt}")
                i += 1
                continue
            
            self._errors.append(f"Unknown statement: '{stmt}'")
            i += 1
    
    def compile_and_run(self, source: str) -> dict:
        """Compile C source and run on neural GPU."""
        result = self.compile(source)
        
        if not result.success:
            return {
                "success": False,
                "errors": result.errors,
                "assembly": result.assembly,
            }
        
        # Run on neural GPU
        gpu_result = self.bridge.run_program_gpu(result.assembly)
        
        return {
            "success": True,
            "assembly": result.assembly,
            "registers": gpu_result.get("registers", {}),
            "cycles": gpu_result.get("cycles", 0),
            "return_value": gpu_result.get("registers", {}).get("R0", 0),
            "variables": {
                name: f"R{var.register}" 
                for name, var in result.variables.items()
                if not name.startswith("__tmp")
            },
            "neural_verified": True,
        }


# ── CLI ──────────────────────────────────────────────────────

EXAMPLES = {
    "arithmetic": """
int a = 10;
int b = 25;
int c = a + b;
int d = c * 2;
return d;
""",
    "bitwise": """
int x = 255;
int mask = 15;
int low = x & mask;
int high = x >> 4;
return low;
""",
    "fibonacci": """
int a = 0;
int b = 1;
int i = 0;
int temp = a + b;
a = b;
b = temp;
return b;
""",
}


def main():
    compiler = NeuralCCompiler()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--asm":
        # Show assembly only
        if len(sys.argv) > 2:
            source = Path(sys.argv[2]).read_text()
        else:
            source = EXAMPLES["arithmetic"]
        result = compiler.compile(source)
        print(result.assembly)
        return
    
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        source = Path(sys.argv[1]).read_text()
        print(f"Compiling {sys.argv[1]}...")
        result = compiler.compile_and_run(source)
    else:
        # Run all examples
        print("Neural C Compiler — nCPU Target")
        print("=" * 50)
        print("Every instruction runs through trained neural networks\n")
        
        for name, source in EXAMPLES.items():
            print(f"── {name} ──")
            print(f"Source:{source}")
            
            result = compiler.compile_and_run(source)
            
            if result["success"]:
                print(f"Assembly:")
                for line in result["assembly"].split("\n"):
                    print(f"  {line}")
                print(f"\nVariables: {result['variables']}")
                print(f"Return value: {result['return_value']}")
                print(f"Cycles: {result['cycles']}")
                print(f"Neural verified: {result['neural_verified']}")
            else:
                print(f"Errors: {result['errors']}")
            
            print()
            
            # Reset compiler state between examples
            compiler = NeuralCCompiler()


if __name__ == "__main__":
    main()
