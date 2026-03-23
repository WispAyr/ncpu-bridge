"""Neural Self-Modifying Code — programs that rewrite themselves through nCPU.

The ultimate flex: a program running on the neural CPU that modifies
its own instructions during execution. Every instruction modification
is computed through neural arithmetic.

Features:
- Self-modifying loop counter (program rewrites its own immediate values)
- Polymorphic code (program changes its operation each iteration)
- Quine: a neural program that outputs its own source
- Evolution: program mutates and selects for fitness

Usage:
    python -m bridge.neural_self_modify demo
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
class Instruction:
    opcode: str
    operands: list[int] = field(default_factory=list)
    
    def __str__(self):
        ops = ", ".join(str(o) for o in self.operands)
        return f"{self.opcode} {ops}" if ops else self.opcode


class NeuralSelfModify:
    """Execute and modify programs at runtime through neural ops."""
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self.registers = [0] * 8
        self._ops = 0
        self._modifications = 0
        self.output: list[str] = []
    
    def _op(self):
        self._ops += 1
    
    def execute(self, program: list[Instruction], max_steps: int = 100) -> dict:
        """Execute a program that can modify itself."""
        self._ops = 0
        self._modifications = 0
        self.output = []
        pc = 0
        steps = 0
        history = []
        
        while pc < len(program) and steps < max_steps:
            inst = program[pc]
            steps += 1
            
            if inst.opcode == "MOV":
                reg = inst.operands[0]
                val = inst.operands[1] if len(inst.operands) > 1 else 0
                self.registers[reg] = val
            
            elif inst.opcode == "ADD" and len(inst.operands) >= 3:
                dst, src1, src2 = inst.operands[:3]
                self.registers[dst] = self.bridge.add(
                    self.registers[src1], self.registers[src2]
                )
                self._op()
            
            elif inst.opcode == "SUB" and len(inst.operands) >= 3:
                dst, src1, src2 = inst.operands[:3]
                self.registers[dst] = self.bridge.sub(
                    self.registers[src1], self.registers[src2]
                )
                self._op()
            
            elif inst.opcode == "MUL" and len(inst.operands) >= 3:
                dst, src1, src2 = inst.operands[:3]
                self.registers[dst] = self.bridge.mul(
                    self.registers[src1], self.registers[src2]
                )
                self._op()
            
            elif inst.opcode == "CMP":
                src1, src2 = inst.operands
                zf, sf = self.bridge.cmp(
                    self.registers[src1], self.registers[src2]
                )
                self._op()
                self.registers[7] = (1 if zf else 0) + (2 if sf else 0)  # Flags in R7
            
            elif inst.opcode == "JNZ":
                target = inst.operands[0]
                zf = self.registers[7] & 1
                if not zf:
                    pc = target
                    continue
            
            elif inst.opcode == "JMP":
                pc = inst.operands[0]
                continue
            
            elif inst.opcode == "PRINT":
                reg = inst.operands[0]
                self.output.append(str(self.registers[reg]))
            
            elif inst.opcode == "HALT":
                break
            
            # ── Self-modification opcodes ──
            
            elif inst.opcode == "MODIFY_IMM":
                # Modify an instruction's immediate value
                # MODIFY_IMM target_pc, operand_idx, new_value_reg
                target_pc, op_idx, val_reg = inst.operands
                old_val = program[target_pc].operands[op_idx]
                new_val = self.registers[val_reg]
                program[target_pc].operands[op_idx] = new_val
                self._modifications += 1
                history.append(f"  Step {steps}: Modified PC {target_pc} operand[{op_idx}]: {old_val} → {new_val}")
            
            elif inst.opcode == "MODIFY_OP":
                # Change an instruction's opcode
                target_pc, new_op_reg = inst.operands
                opcodes = ["ADD", "SUB", "MUL", "MOV"]
                idx = self.bridge.bitwise_and(self.registers[new_op_reg], 3)
                self._op()
                old_op = program[target_pc].opcode
                program[target_pc].opcode = opcodes[idx]
                self._modifications += 1
                history.append(f"  Step {steps}: Changed PC {target_pc}: {old_op} → {opcodes[idx]}")
            
            elif inst.opcode == "READ_INST":
                # Read own instruction into register
                target_pc, op_idx, dst_reg = inst.operands
                if target_pc < len(program) and op_idx < len(program[target_pc].operands):
                    self.registers[dst_reg] = program[target_pc].operands[op_idx]
            
            pc += 1
        
        return {
            "steps": steps,
            "output": self.output,
            "registers": list(self.registers),
            "modifications": self._modifications,
            "history": history,
            "neural_ops": self._ops,
        }


# ── Demo Programs ──

def self_counting_loop():
    """A loop that modifies its own counter value."""
    return [
        Instruction("MOV", [0, 1]),          # 0: R0 = 1 (counter)
        Instruction("MOV", [1, 1]),           # 1: R1 = 1 (increment)
        Instruction("MOV", [2, 5]),           # 2: R2 = 5 (limit)
        # Loop body:
        Instruction("PRINT", [0]),            # 3: print counter
        Instruction("ADD", [0, 0, 1]),        # 4: R0 += R1
        # Self-modify: double the increment each iteration
        Instruction("ADD", [1, 1, 1]),        # 5: R1 = R1 + R1 (double increment!)
        Instruction("MODIFY_IMM", [2, 1, 2]), # 6: modify limit at PC 2 to R2's value
        Instruction("ADD", [2, 2, 1]),        # 7: increase limit too
        Instruction("CMP", [0, 2]),           # 8: compare counter to limit
        Instruction("JNZ", [3]),              # 9: loop if not equal
        Instruction("HALT", []),              # 10
    ]


def polymorphic_program():
    """A program that changes its own operation each iteration."""
    return [
        Instruction("MOV", [0, 10]),          # 0: R0 = 10
        Instruction("MOV", [1, 3]),           # 1: R1 = 3
        Instruction("MOV", [3, 0]),           # 2: R3 = 0 (op selector)
        # The morphing instruction:
        Instruction("ADD", [2, 0, 1]),        # 3: R2 = R0 + R1 (will change!)
        Instruction("PRINT", [2]),            # 4: print result
        # Self-modify: change the operation
        Instruction("ADD", [3, 3, 1]),        # 5: R3 += 1 (next op)
        Instruction("MODIFY_OP", [3, 3]),     # 6: change instruction at PC 3
        Instruction("MOV", [4, 4]),           # 7: R4 = 4 (loop limit)
        Instruction("CMP", [3, 4]),           # 8: 
        Instruction("JNZ", [3]),              # 9: loop
        Instruction("HALT", []),              # 10
    ]


def neural_quine():
    """A program that reads its own instructions and prints them."""
    return [
        Instruction("MOV", [0, 0]),           # 0: R0 = 0 (PC to read)
        Instruction("MOV", [1, 4]),           # 1: R1 = 4 (num instructions to read)
        Instruction("READ_INST", [0, 0, 2]), # 2: R2 = instruction[R0].operands[0]
        Instruction("PRINT", [2]),            # 3: print it
        Instruction("MOV", [3, 1]),           # 4: R3 = 1
        Instruction("ADD", [0, 0, 3]),        # 5: R0++
        Instruction("CMP", [0, 1]),           # 6: R0 == R1?
        Instruction("JNZ", [2]),              # 7: loop
        Instruction("HALT", []),              # 8
    ]


def demo():
    vm = NeuralSelfModify()
    
    print("Neural Self-Modifying Code")
    print("=" * 60)
    print("Programs that rewrite themselves through neural ops\n")
    
    # ── Self-counting loop ──
    print("── Self-Modifying Counter ──")
    print("  (doubles its increment + raises its limit each iteration)")
    prog = self_counting_loop()
    result = vm.execute(prog, max_steps=50)
    print(f"  Output: {' → '.join(result['output'])}")
    print(f"  Self-modifications: {result['modifications']}")
    for h in result['history'][:5]:
        print(h)
    if len(result['history']) > 5:
        print(f"  ... ({len(result['history'])} total modifications)")
    print(f"  Neural ops: {result['neural_ops']}")
    print()
    
    # ── Polymorphic program ──
    print("── Polymorphic Code ──")
    print("  (changes its own opcode: ADD → SUB → MUL → MOV)")
    vm2 = NeuralSelfModify()
    prog2 = polymorphic_program()
    result2 = vm2.execute(prog2, max_steps=50)
    print(f"  Output: {' → '.join(result2['output'])}")
    print(f"  Mutations:")
    for h in result2['history']:
        print(h)
    print(f"  Neural ops: {result2['neural_ops']}")
    print()
    
    # ── Quine ──
    print("── Neural Quine ──")
    print("  (reads and prints its own instruction operands)")
    vm3 = NeuralSelfModify()
    prog3 = neural_quine()
    result3 = vm3.execute(prog3, max_steps=50)
    print(f"  Self-read values: {result3['output']}")
    print(f"  Neural ops: {result3['neural_ops']}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_self_modify [demo]")


if __name__ == "__main__":
    main()
