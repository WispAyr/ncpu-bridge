"""Neural Forth — a Forth interpreter running on the neural CPU.

Forth is a stack-based language — perfect for the nCPU because every
operation (push, pop, add, dup, swap) maps directly to neural ALU ops.

Supported words:
  Arithmetic: + - * / MOD
  Stack:      DUP DROP SWAP OVER ROT
  Comparison: = < > <> 
  Logic:      AND OR XOR NOT
  Control:    IF...THEN, IF...ELSE...THEN, DO...LOOP, BEGIN...UNTIL
  I/O:        . (print top), .S (print stack), CR (newline), EMIT (char)
  Define:     : name ... ;
  Variables:  VARIABLE name, ! (store), @ (fetch)

Every arithmetic and comparison operation goes through neural networks.

Usage:
    python -m bridge.neural_forth demo
    python -m bridge.neural_forth repl
    python -m bridge.neural_forth run <file>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


class NeuralForth:
    """Forth interpreter with neural ALU backend.
    
    Every + is a neural ADD, every < is a neural CMP,
    every AND is a neural bitwise_and. The stack machine
    runs entirely on trained neural networks.
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self.stack: list[int] = []
        self.return_stack: list[int] = []
        self.variables: dict[str, int] = {}  # name → memory address
        self.memory: dict[int, int] = {}  # address → value
        self._next_addr = 1000
        self.definitions: dict[str, list[str]] = {}
        self.output: list[str] = []
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def push(self, val: int):
        self.stack.append(val)
    
    def pop(self) -> int:
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack.pop()
    
    def peek(self) -> int:
        if not self.stack:
            raise RuntimeError("Stack empty")
        return self.stack[-1]
    
    def execute(self, source: str) -> str:
        """Execute Forth source code. Returns output."""
        self.output = []
        self._ops = 0
        tokens = self._tokenize(source)
        self._exec_tokens(tokens, 0)
        return " ".join(self.output)
    
    def _tokenize(self, source: str) -> list[str]:
        """Split source into tokens."""
        return source.upper().split()
    
    def _exec_tokens(self, tokens: list[str], start: int) -> int:
        """Execute tokens starting at index. Returns next index."""
        i = start
        while i < len(tokens):
            token = tokens[i]
            
            # Number literal
            try:
                val = int(token)
                self.push(val)
                i += 1
                continue
            except ValueError:
                pass
            
            # Word definition: : name ... ;
            if token == ':':
                name = tokens[i + 1]
                body = []
                j = i + 2
                while j < len(tokens) and tokens[j] != ';':
                    body.append(tokens[j])
                    j += 1
                self.definitions[name] = body
                i = j + 1
                continue
            
            # Variable definition
            if token == 'VARIABLE':
                name = tokens[i + 1]
                self.variables[name] = self._next_addr
                self.memory[self._next_addr] = 0
                self._next_addr = self.bridge.add(self._next_addr, 1)
                self._op()
                i += 2
                continue
            
            # Variable reference (push address)
            if token in self.variables:
                self.push(self.variables[token])
                i += 1
                continue
            
            # User-defined word
            if token in self.definitions:
                self._exec_tokens(self.definitions[token], 0)
                i += 1
                continue
            
            # Built-in words
            result = self._exec_builtin(token, tokens, i)
            if result is not None:
                i = result
                continue
            
            # Unknown word
            self.output.append(f"?{token}")
            i += 1
        
        return i
    
    def _exec_builtin(self, token: str, tokens: list[str], i: int) -> Optional[int]:
        """Execute a built-in word. Returns next token index or None."""
        
        # ── Arithmetic (all neural) ──
        if token == '+':
            b, a = self.pop(), self.pop()
            self.push(self.bridge.add(a, b))
            self._op()
            return i + 1
        
        if token == '-':
            b, a = self.pop(), self.pop()
            self.push(self.bridge.sub(a, b))
            self._op()
            return i + 1
        
        if token == '*':
            b, a = self.pop(), self.pop()
            self.push(self.bridge.mul(a, b))
            self._op()
            return i + 1
        
        if token == '/':
            b, a = self.pop(), self.pop()
            self.push(self.bridge.div(a, b))
            self._op()
            return i + 1
        
        if token == 'MOD':
            b, a = self.pop(), self.pop()
            q = self.bridge.div(a, b)
            self._op()
            r = self.bridge.sub(a, self.bridge.mul(q, b))
            self._op()
            self.push(r)
            return i + 1
        
        # ── Comparison (neural CMP) ──
        if token == '=':
            b, a = self.pop(), self.pop()
            zf, _ = self.bridge.cmp(a, b)
            self._op()
            self.push(-1 if zf else 0)  # Forth: -1 = true, 0 = false
            return i + 1
        
        if token == '<':
            b, a = self.pop(), self.pop()
            _, sf = self.bridge.cmp(a, b)
            self._op()
            self.push(-1 if sf else 0)
            return i + 1
        
        if token == '>':
            b, a = self.pop(), self.pop()
            zf, sf = self.bridge.cmp(a, b)
            self._op()
            self.push(-1 if (not sf and not zf) else 0)
            return i + 1
        
        if token == '<>':  # not equal
            b, a = self.pop(), self.pop()
            zf, _ = self.bridge.cmp(a, b)
            self._op()
            self.push(0 if zf else -1)
            return i + 1
        
        # ── Logic (neural bitwise) ──
        if token == 'AND':
            b, a = self.pop(), self.pop()
            self.push(self.bridge.bitwise_and(a, b))
            self._op()
            return i + 1
        
        if token == 'OR':
            b, a = self.pop(), self.pop()
            self.push(self.bridge.bitwise_or(a, b))
            self._op()
            return i + 1
        
        if token == 'XOR':
            b, a = self.pop(), self.pop()
            self.push(self.bridge.bitwise_xor(a, b))
            self._op()
            return i + 1
        
        if token == 'NOT':
            a = self.pop()
            self.push(self.bridge.bitwise_xor(a, -1))
            self._op()
            return i + 1
        
        # ── Stack ops ──
        if token == 'DUP':
            self.push(self.peek())
            return i + 1
        
        if token == 'DROP':
            self.pop()
            return i + 1
        
        if token == 'SWAP':
            b, a = self.pop(), self.pop()
            self.push(b)
            self.push(a)
            return i + 1
        
        if token == 'OVER':
            b, a = self.pop(), self.pop()
            self.push(a)
            self.push(b)
            self.push(a)
            return i + 1
        
        if token == 'ROT':
            c, b, a = self.pop(), self.pop(), self.pop()
            self.push(b)
            self.push(c)
            self.push(a)
            return i + 1
        
        # ── Memory ──
        if token == '!':  # store: value addr !
            addr = self.pop()
            val = self.pop()
            self.memory[addr] = val
            return i + 1
        
        if token == '@':  # fetch: addr @
            addr = self.pop()
            self.push(self.memory.get(addr, 0))
            return i + 1
        
        # ── I/O ──
        if token == '.':
            self.output.append(str(self.pop()))
            return i + 1
        
        if token == '.S':
            self.output.append(f"<{len(self.stack)}> {' '.join(str(x) for x in self.stack)}")
            return i + 1
        
        if token == 'CR':
            self.output.append("\n")
            return i + 1
        
        if token == 'EMIT':
            self.output.append(chr(self.pop()))
            return i + 1
        
        # ── Control: IF...ELSE...THEN ──
        if token == 'IF':
            cond = self.pop()
            zf, _ = self.bridge.cmp(cond, 0)
            self._op()
            
            # Find matching ELSE or THEN
            depth = 1
            j = i + 1
            else_pos = None
            then_pos = None
            
            while j < len(tokens) and depth > 0:
                if tokens[j] == 'IF':
                    depth += 1
                elif tokens[j] == 'THEN':
                    depth -= 1
                    if depth == 0:
                        then_pos = j
                elif tokens[j] == 'ELSE' and depth == 1:
                    else_pos = j
                j += 1
            
            if zf:  # condition is false (0)
                if else_pos:
                    # Execute ELSE branch
                    self._exec_tokens(tokens[else_pos + 1:then_pos], 0)
                # else: skip to THEN
            else:
                # Execute IF branch
                end = else_pos if else_pos else then_pos
                self._exec_tokens(tokens[i + 1:end], 0)
            
            return then_pos + 1 if then_pos else i + 1
        
        # ── Control: DO...LOOP ──
        if token == 'DO':
            limit = self.pop()
            index = self.pop()
            
            # Find matching LOOP
            j = i + 1
            depth = 1
            while j < len(tokens):
                if tokens[j] == 'DO':
                    depth += 1
                elif tokens[j] == 'LOOP':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            
            loop_body = tokens[i + 1:j]
            loop_end = j + 1
            
            # Execute loop
            while True:
                zf, sf = self.bridge.cmp(index, limit)
                self._op()
                if zf or not sf:  # index >= limit
                    break
                
                self.return_stack.append(index)
                self._exec_tokens(loop_body, 0)
                self.return_stack.pop()
                
                index = self.bridge.add(index, 1)
                self._op()
            
            return loop_end
        
        if token == 'I':
            # Loop index
            if self.return_stack:
                self.push(self.return_stack[-1])
            return i + 1
        
        return None


# ── CLI ──

def demo():
    forth = NeuralForth()
    
    print("Neural Forth Interpreter")
    print("=" * 60)
    print("Every arithmetic and comparison → trained neural network\n")
    
    programs = [
        ("Basic arithmetic", "10 20 + ."),
        ("Stack ops", "1 2 3 .S"),
        ("Multiplication", "7 8 * ."),
        ("Division + mod", "17 5 / . 17 5 MOD ."),
        ("Comparison", "42 42 = ."),
        ("Define word", ": SQUARE DUP * ; 7 SQUARE ."),
        ("Conditional", "1 IF 42 ELSE 0 THEN ."),
        ("False branch", "0 IF 42 ELSE 99 THEN ."),
        ("Loop (sum 0-4)", "0 5 0 DO I + LOOP ."),
        ("Factorial-ish", ": FACT DUP 1 > IF DUP 1 - FACT * THEN ; 5 FACT ."),
        ("Variable", "VARIABLE X 42 X ! X @ ."),
        ("Bitwise", "255 15 AND . 170 85 XOR ."),
        ("Fibonacci", ": FIB DUP 2 < IF DROP 1 ELSE DUP 1 - FIB SWAP 2 - FIB + THEN ; 7 FIB ."),
    ]
    
    for desc, code in programs:
        forth_inst = NeuralForth()
        try:
            output = forth_inst.execute(code)
            print(f"  {desc:20s} │ {code:45s} │ → {output:10s} │ {forth_inst._ops} ops")
        except Exception as e:
            print(f"  {desc:20s} │ {code:45s} │ → ERROR: {e}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    if cmd == "demo":
        demo()
    elif cmd == "repl":
        forth = NeuralForth()
        print("Neural Forth REPL (type 'bye' to exit)")
        while True:
            try:
                line = input("> ")
                if line.strip().upper() == "BYE":
                    break
                output = forth.execute(line)
                if output.strip():
                    print(output)
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"Error: {e}")
    elif cmd == "run" and len(sys.argv) > 2:
        forth = NeuralForth()
        source = Path(sys.argv[2]).read_text()
        output = forth.execute(source)
        print(output)
    else:
        print("Usage: python -m bridge.neural_forth [demo|repl|run <file>]")


if __name__ == "__main__":
    main()
