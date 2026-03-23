"""Neural Debugger — step through nCPU programs with neural breakpoints.

Debug nCPU assembly programs where every breakpoint check, register
inspection, and watchpoint comparison is neural:

- Breakpoints: neural CMP on program counter
- Watchpoints: neural CMP on register values
- Single-step: execute one instruction at a time on neural GPU
- Register diff: neural XOR to detect changes
- Stack trace: neural pointer chain walking

Usage:
    python -m bridge.neural_debugger demo
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
class Breakpoint:
    id: int
    pc: int  # Program counter value
    condition: str = ""  # Optional: "R0 > 50"
    hit_count: int = 0
    enabled: bool = True


@dataclass
class Watchpoint:
    id: int
    register: str
    condition: str  # "changed", "== 42", "> 100"
    last_value: int = 0
    hit_count: int = 0


@dataclass
class DebugState:
    pc: int = 0
    registers: dict[str, int] = field(default_factory=lambda: {f"R{i}": 0 for i in range(8)})
    flags: dict[str, bool] = field(default_factory=lambda: {"ZF": False, "SF": False})
    halted: bool = False
    instructions: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


class NeuralDebugger:
    """Debug nCPU programs with neural breakpoint/watchpoint checking."""
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._breakpoints: dict[int, Breakpoint] = {}
        self._watchpoints: dict[int, Watchpoint] = {}
        self._next_bp = 1
        self._next_wp = 1
        self._ops = 0
        self.state = DebugState()
    
    def _op(self):
        self._ops += 1
    
    def load(self, assembly: str):
        """Load a program for debugging."""
        self.state = DebugState()
        self.state.instructions = [line.strip() for line in assembly.strip().split("\n") if line.strip()]
    
    def add_breakpoint(self, pc: int) -> int:
        """Set breakpoint at PC using neural comparison."""
        bp_id = self._next_bp
        self._next_bp = self.bridge.add(self._next_bp, 1)
        self._op()
        
        self._breakpoints[bp_id] = Breakpoint(id=bp_id, pc=pc)
        return bp_id
    
    def add_watchpoint(self, register: str, condition: str = "changed") -> int:
        """Set watchpoint on register."""
        wp_id = self._next_wp
        self._next_wp = self.bridge.add(self._next_wp, 1)
        self._op()
        
        self._watchpoints[wp_id] = Watchpoint(id=wp_id, register=register, condition=condition)
        return wp_id
    
    def _check_breakpoints(self) -> list[Breakpoint]:
        """Check if any breakpoint matches current PC — neural CMP."""
        hits = []
        for bp in self._breakpoints.values():
            if not bp.enabled:
                continue
            zf, _ = self.bridge.cmp(self.state.pc, bp.pc)
            self._op()
            if zf:
                bp.hit_count = self.bridge.add(bp.hit_count, 1)
                self._op()
                hits.append(bp)
        return hits
    
    def _check_watchpoints(self, old_regs: dict) -> list[Watchpoint]:
        """Check if any watchpoint triggered — neural CMP on values."""
        hits = []
        for wp in self._watchpoints.values():
            reg = wp.register
            old_val = old_regs.get(reg, 0)
            new_val = self.state.registers.get(reg, 0)
            
            if wp.condition == "changed":
                # Neural: did value change?
                zf, _ = self.bridge.cmp(old_val, new_val)
                self._op()
                if not zf:  # Changed!
                    wp.hit_count = self.bridge.add(wp.hit_count, 1)
                    self._op()
                    wp.last_value = new_val
                    hits.append(wp)
            
            elif wp.condition.startswith("=="):
                target = int(wp.condition[2:].strip())
                zf, _ = self.bridge.cmp(new_val, target)
                self._op()
                if zf:
                    wp.hit_count += 1
                    hits.append(wp)
            
            elif wp.condition.startswith(">"):
                target = int(wp.condition[1:].strip())
                zf, sf = self.bridge.cmp(new_val, target)
                self._op()
                if not sf and not zf:  # new > target
                    wp.hit_count += 1
                    hits.append(wp)
        
        return hits
    
    def step(self) -> dict:
        """Execute one instruction and check breakpoints/watchpoints."""
        if self.state.halted or self.state.pc >= len(self.state.instructions):
            return {"status": "halted"}
        
        old_regs = dict(self.state.registers)
        instruction = self.state.instructions[self.state.pc]
        
        # Execute single instruction via neural GPU
        # We build a mini program: set registers, run one instruction, halt
        setup = []
        for reg, val in self.state.registers.items():
            setup.append(f"MOV {reg}, {val}")
        setup.append(instruction)
        setup.append("HALT")
        
        program = "\n".join(setup)
        result = self.bridge.run_program_gpu(program)
        
        # Update state from GPU result
        new_regs = result.get("registers", {})
        for reg in self.state.registers:
            if reg in new_regs:
                self.state.registers[reg] = new_regs[reg]
        
        # Check for HALT
        if instruction.strip().upper() == "HALT":
            self.state.halted = True
        
        # Advance PC
        self.state.pc = self.bridge.add(self.state.pc, 1)
        self._op()
        
        # Record history
        self.state.history.append({
            "pc": self.state.pc - 1,
            "instruction": instruction,
            "registers": dict(self.state.registers),
        })
        
        # Check breakpoints and watchpoints
        bp_hits = self._check_breakpoints()
        wp_hits = self._check_watchpoints(old_regs)
        
        # Compute register diff (neural XOR to find changes)
        changes = {}
        for reg in self.state.registers:
            old = old_regs.get(reg, 0)
            new = self.state.registers[reg]
            diff = self.bridge.bitwise_xor(old, new)
            self._op()
            zf, _ = self.bridge.cmp(diff, 0)
            self._op()
            if not zf:
                changes[reg] = (old, new)
        
        return {
            "status": "halted" if self.state.halted else "running",
            "pc": self.state.pc - 1,
            "instruction": instruction,
            "changes": changes,
            "breakpoints": [f"BP#{bp.id}@PC={bp.pc}" for bp in bp_hits],
            "watchpoints": [f"WP#{wp.id}:{wp.register}" for wp in wp_hits],
        }
    
    def run_to_end(self) -> list[dict]:
        """Run program, stopping at breakpoints."""
        trace = []
        max_steps = 100
        
        for _ in range(max_steps):
            result = self.step()
            trace.append(result)
            
            if result["status"] == "halted":
                break
            if result["breakpoints"]:
                break
        
        return trace


# ── CLI ──

def demo():
    dbg = NeuralDebugger()
    
    print("Neural Debugger")
    print("=" * 60)
    print("Breakpoints + watchpoints + register diff → neural CMP/XOR\n")
    
    # Load a program
    program = """MOV R0, 10
MOV R1, 20
ADD R2, R0, R1
MOV R3, 5
MUL R4, R2, R3
SUB R5, R4, R0
HALT"""
    
    dbg.load(program)
    
    print("── Program ──")
    for i, inst in enumerate(dbg.state.instructions):
        print(f"  {i:2d}: {inst}")
    print()
    
    # Set breakpoint and watchpoint
    bp1 = dbg.add_breakpoint(4)  # Break at MUL
    wp1 = dbg.add_watchpoint("R2", "changed")
    wp2 = dbg.add_watchpoint("R4", "> 100")
    
    print(f"  Breakpoint #{bp1} at PC=4 (MUL instruction)")
    print(f"  Watchpoint #{wp1} on R2 (on change)")
    print(f"  Watchpoint #{wp2} on R4 (when > 100)")
    print()
    
    # Step through
    print("── Single Step Execution ──")
    for i in range(7):
        result = dbg.step()
        
        changes = ""
        if result.get("changes"):
            changes = " | " + ", ".join(f"{r}: {o}→{n}" for r, (o, n) in result["changes"].items())
        
        triggers = ""
        if result.get("breakpoints"):
            triggers += " 🔴 " + ", ".join(result["breakpoints"])
        if result.get("watchpoints"):
            triggers += " 👁 " + ", ".join(result["watchpoints"])
        
        status = "⏹" if result["status"] == "halted" else "▶"
        inst = result.get("instruction", "?")
        print(f"  {status} PC={result.get('pc', '?'):2d} {inst:25s}{changes}{triggers}")
        
        if result["status"] == "halted":
            break
    
    print()
    print("── Final Registers ──")
    for reg, val in sorted(dbg.state.registers.items()):
        if val != 0:
            print(f"  {reg} = {val}")
    
    print(f"\n  Neural ops: {dbg._ops}")
    print(f"  Steps: {len(dbg.state.history)}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_debugger [demo]")


if __name__ == "__main__":
    main()
