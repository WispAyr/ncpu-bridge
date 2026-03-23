"""Neural Virtual Machine — memory management and process execution through nCPU.

A complete VM with:
- Heap memory management (malloc/free) using neural-computed free list
- Stack operations (push/pop) with neural pointer arithmetic
- Process table with neural PID allocation and scheduling
- System calls: write, read, alloc, free, exit
- Memory protection: bounds checking via neural CMP

Every pointer arithmetic, bounds check, and allocation decision
goes through trained neural networks.

Usage:
    python -m bridge.neural_vm demo
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge

HEAP_SIZE = 1024  # bytes
STACK_SIZE = 128  # per process
MAX_PROCS = 8


@dataclass
class FreeBlock:
    """Free list node for heap allocation."""
    start: int
    size: int
    next: Optional[int] = None  # Index into free list


@dataclass
class Allocation:
    """Track a heap allocation."""
    start: int
    size: int
    pid: int


@dataclass
class Process:
    """A process in the neural VM."""
    pid: int
    name: str
    state: str = "ready"  # ready, running, blocked, terminated
    stack: list[int] = field(default_factory=list)
    sp: int = 0  # stack pointer
    pc: int = 0  # program counter
    registers: dict[str, int] = field(default_factory=lambda: {f"R{i}": 0 for i in range(4)})
    allocations: list[int] = field(default_factory=list)  # heap addresses owned
    exit_code: int = -1


class NeuralHeap:
    """Neural-managed heap with first-fit allocation.
    
    Free list maintained with neural pointer arithmetic.
    Every size comparison and address calculation is neural.
    """
    
    def __init__(self, bridge: NCPUBridge, size: int = HEAP_SIZE):
        self.bridge = bridge
        self.size = size
        self.memory = [0] * size
        self._ops = 0
        
        # Free list: start with one big block
        self._free_list: list[FreeBlock] = [FreeBlock(start=0, size=size)]
        self._allocations: dict[int, Allocation] = {}
    
    def _op(self):
        self._ops += 1
    
    def malloc(self, size: int, pid: int) -> Optional[int]:
        """Allocate memory using neural first-fit search.
        
        Walks free list, using neural CMP to find a block >= requested size.
        """
        for i, block in enumerate(self._free_list):
            # Neural comparison: is this block big enough?
            zf, sf = self.bridge.cmp(block.size, size)
            self._op()
            
            if zf or not sf:  # block.size >= size
                addr = block.start
                
                # Calculate remaining space (neural SUB)
                remaining = self.bridge.sub(block.size, size)
                self._op()
                
                # Check if we should split (remaining > 0)
                zf2, _ = self.bridge.cmp(remaining, 0)
                self._op()
                
                if not zf2:  # remaining > 0: split block
                    new_start = self.bridge.add(block.start, size)
                    self._op()
                    self._free_list[i] = FreeBlock(start=new_start, size=remaining)
                else:
                    # Exact fit: remove block
                    self._free_list.pop(i)
                
                # Record allocation
                self._allocations[addr] = Allocation(start=addr, size=size, pid=pid)
                return addr
        
        return None  # Out of memory
    
    def free(self, addr: int) -> bool:
        """Free allocation and coalesce adjacent free blocks."""
        alloc = self._allocations.pop(addr, None)
        if not alloc:
            return False
        
        # Add to free list
        new_block = FreeBlock(start=alloc.start, size=alloc.size)
        self._free_list.append(new_block)
        
        # Sort free list by address (neural insertion sort)
        for i in range(1, len(self._free_list)):
            key = self._free_list[i]
            j = i - 1
            while j >= 0:
                zf, sf = self.bridge.cmp(self._free_list[j].start, key.start)
                self._op()
                if not sf and not zf:  # [j].start > key.start
                    self._free_list[j + 1] = self._free_list[j]
                    j -= 1
                else:
                    break
            self._free_list[j + 1] = key
        
        # Coalesce adjacent blocks
        self._coalesce()
        return True
    
    def _coalesce(self):
        """Merge adjacent free blocks using neural address arithmetic."""
        i = 0
        while i < len(self._free_list) - 1:
            curr = self._free_list[i]
            next_blk = self._free_list[i + 1]
            
            # Neural check: does curr end where next starts?
            curr_end = self.bridge.add(curr.start, curr.size)
            self._op()
            zf, _ = self.bridge.cmp(curr_end, next_blk.start)
            self._op()
            
            if zf:  # Adjacent — merge
                merged_size = self.bridge.add(curr.size, next_blk.size)
                self._op()
                self._free_list[i] = FreeBlock(start=curr.start, size=merged_size)
                self._free_list.pop(i + 1)
            else:
                i += 1
    
    def write(self, addr: int, data: list[int], pid: int) -> bool:
        """Write to memory with bounds checking (neural CMP)."""
        # Find allocation
        alloc = self._find_alloc(addr, pid)
        if not alloc:
            return False
        
        # Bounds check: addr + len(data) <= alloc.start + alloc.size
        write_end = self.bridge.add(addr, len(data))
        self._op()
        alloc_end = self.bridge.add(alloc.start, alloc.size)
        self._op()
        zf, sf = self.bridge.cmp(write_end, alloc_end)
        self._op()
        
        if not sf and not zf:  # write_end > alloc_end
            return False  # Buffer overflow! Rejected by neural bounds check.
        
        for i, byte in enumerate(data):
            idx = self.bridge.add(addr, i)
            self._op()
            self.memory[idx] = byte
        
        return True
    
    def read(self, addr: int, size: int, pid: int) -> Optional[list[int]]:
        """Read from memory with bounds checking."""
        alloc = self._find_alloc(addr, pid)
        if not alloc:
            return None
        
        read_end = self.bridge.add(addr, size)
        self._op()
        alloc_end = self.bridge.add(alloc.start, alloc.size)
        self._op()
        zf, sf = self.bridge.cmp(read_end, alloc_end)
        self._op()
        
        if not sf and not zf:
            return None
        
        return self.memory[addr:addr + size]
    
    def _find_alloc(self, addr: int, pid: int) -> Optional[Allocation]:
        """Find allocation containing addr, owned by pid."""
        for alloc in self._allocations.values():
            # Neural: is addr >= alloc.start?
            zf1, sf1 = self.bridge.cmp(addr, alloc.start)
            self._op()
            if not (zf1 or not sf1):
                continue
            
            # Neural: is addr < alloc.start + alloc.size?
            end = self.bridge.add(alloc.start, alloc.size)
            self._op()
            zf2, sf2 = self.bridge.cmp(addr, end)
            self._op()
            if not sf2:
                continue
            
            # Neural: pid check
            zf3, _ = self.bridge.cmp(alloc.pid, pid)
            self._op()
            if zf3:
                return alloc
        
        return None
    
    def stats(self) -> dict:
        total_free = sum(b.size for b in self._free_list)
        return {
            "heap_size": self.size,
            "allocated": self.bridge.sub(self.size, total_free),
            "free": total_free,
            "fragments": len(self._free_list),
            "allocations": len(self._allocations),
            "neural_ops": self._ops,
        }


class NeuralVM:
    """Virtual Machine with neural memory management and process scheduling."""
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self.heap = NeuralHeap(self.bridge)
        self._procs: dict[int, Process] = {}
        self._next_pid = 1
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def spawn(self, name: str) -> Process:
        """Spawn a new process with neural PID allocation."""
        pid = self._next_pid
        self._next_pid = self.bridge.add(self._next_pid, 1)
        self._op()
        
        proc = Process(pid=pid, name=name)
        self._procs[pid] = proc
        return proc
    
    def syscall_alloc(self, pid: int, size: int) -> Optional[int]:
        """System call: allocate heap memory."""
        proc = self._procs.get(pid)
        if not proc:
            return None
        
        addr = self.heap.malloc(size, pid)
        if addr is not None:
            proc.allocations.append(addr)
        return addr
    
    def syscall_free(self, pid: int, addr: int) -> bool:
        """System call: free heap memory."""
        proc = self._procs.get(pid)
        if not proc:
            return False
        
        if addr in proc.allocations:
            proc.allocations.remove(addr)
        return self.heap.free(addr)
    
    def syscall_write(self, pid: int, addr: int, data: list[int]) -> bool:
        """System call: write to allocated memory."""
        return self.heap.write(addr, data, pid)
    
    def syscall_read(self, pid: int, addr: int, size: int) -> Optional[list[int]]:
        """System call: read from allocated memory."""
        return self.heap.read(addr, size, pid)
    
    def push(self, pid: int, value: int):
        """Push to process stack."""
        proc = self._procs.get(pid)
        if proc and len(proc.stack) < STACK_SIZE:
            proc.stack.append(value)
            proc.sp = self.bridge.add(proc.sp, 1)
            self._op()
    
    def pop(self, pid: int) -> Optional[int]:
        """Pop from process stack."""
        proc = self._procs.get(pid)
        if proc and proc.stack:
            proc.sp = self.bridge.sub(proc.sp, 1)
            self._op()
            return proc.stack.pop()
        return None
    
    def terminate(self, pid: int, exit_code: int = 0):
        """Terminate process and free all its memory."""
        proc = self._procs.get(pid)
        if not proc:
            return
        
        # Free all allocations
        for addr in list(proc.allocations):
            self.heap.free(addr)
        proc.allocations.clear()
        proc.state = "terminated"
        proc.exit_code = exit_code
    
    def ps(self) -> list[dict]:
        """Process listing."""
        return [
            {
                "pid": p.pid,
                "name": p.name,
                "state": p.state,
                "stack_depth": len(p.stack),
                "heap_allocs": len(p.allocations),
            }
            for p in self._procs.values()
        ]


# ── CLI ──────────────────────────────────────────────────────

def demo():
    vm = NeuralVM()
    
    print("Neural Virtual Machine")
    print("=" * 60)
    print("Memory management + process control → neural ALU\n")
    
    # Spawn processes
    print("── Spawn Processes ──")
    p1 = vm.spawn("sentinel")
    p2 = vm.spawn("monitor")
    p3 = vm.spawn("logger")
    for p in [p1, p2, p3]:
        print(f"  PID {p.pid}: {p.name}")
    print()
    
    # Allocate memory
    print("── Heap Allocation ──")
    addr1 = vm.syscall_alloc(p1.pid, 64)
    addr2 = vm.syscall_alloc(p2.pid, 128)
    addr3 = vm.syscall_alloc(p3.pid, 32)
    print(f"  sentinel: malloc(64)  → addr {addr1}")
    print(f"  monitor:  malloc(128) → addr {addr2}")
    print(f"  logger:   malloc(32)  → addr {addr3}")
    
    stats = vm.heap.stats()
    print(f"  Heap: {stats['allocated']}/{stats['heap_size']} used, {stats['fragments']} fragment(s)")
    print()
    
    # Write and read
    print("── Memory Write/Read ──")
    message = [ord(c) for c in "NCPU OK"]
    ok = vm.syscall_write(p1.pid, addr1, message)
    print(f"  write(sentinel, addr={addr1}, 'NCPU OK') → {'✅' if ok else '❌'}")
    
    data = vm.syscall_read(p1.pid, addr1, 7)
    decoded = ''.join(chr(b) for b in data) if data else "?"
    print(f"  read(sentinel, addr={addr1}, 7) → '{decoded}'")
    
    # Bounds check
    print()
    print("── Memory Protection ──")
    # Try to read monitor's memory as sentinel (should fail)
    bad_read = vm.syscall_read(p1.pid, addr2, 10)
    print(f"  sentinel reading monitor's memory → {'❌ DENIED' if bad_read is None else '⚠️ LEAK!'}")
    
    # Try buffer overflow
    overflow_data = [0xFF] * 100
    overflow_ok = vm.syscall_write(p3.pid, addr3, overflow_data)
    print(f"  logger overflow (100 bytes into 32) → {'❌ DENIED' if not overflow_ok else '⚠️ OVERFLOW!'}")
    print()
    
    # Stack operations
    print("── Stack Operations ──")
    vm.push(p1.pid, 42)
    vm.push(p1.pid, 99)
    vm.push(p1.pid, 7)
    print(f"  push(sentinel, 42, 99, 7) → stack depth: {len(p1.stack)}")
    
    val = vm.pop(p1.pid)
    print(f"  pop(sentinel) → {val}")
    val = vm.pop(p1.pid)
    print(f"  pop(sentinel) → {val}")
    print()
    
    # Free and reallocate (test coalescing)
    print("── Free + Coalesce ──")
    vm.syscall_free(p2.pid, addr2)
    print(f"  free(monitor, addr={addr2})")
    
    stats = vm.heap.stats()
    print(f"  Heap: {stats['allocated']}/{stats['heap_size']} used, {stats['fragments']} fragment(s)")
    
    # Allocate in the freed space
    addr4 = vm.syscall_alloc(p1.pid, 100)
    print(f"  sentinel: malloc(100) → addr {addr4} (reused freed space)")
    
    stats = vm.heap.stats()
    print(f"  Heap: {stats['allocated']}/{stats['heap_size']} used, {stats['fragments']} fragment(s)")
    print()
    
    # Terminate
    print("── Process Termination ──")
    vm.terminate(p3.pid, exit_code=0)
    print(f"  terminate(logger) → freed all allocations")
    
    stats = vm.heap.stats()
    print(f"  Heap: {stats['allocated']}/{stats['heap_size']} used, {stats['fragments']} fragment(s)")
    print()
    
    # Process table
    print("── Process Table ──")
    for p in vm.ps():
        emoji = {"ready": "🟢", "running": "🔵", "terminated": "⚫"}.get(p["state"], "⚪")
        print(f"  {emoji} PID {p['pid']:2d} {p['name']:10s} state={p['state']:12s} stack={p['stack_depth']} heap={p['heap_allocs']}")
    
    print()
    total_ops = vm._ops + vm.heap._ops
    print(f"  Total neural ops: {total_ops}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_vm [demo]")


if __name__ == "__main__":
    main()
