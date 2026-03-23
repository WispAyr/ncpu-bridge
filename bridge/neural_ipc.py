"""Neural IPC — inter-process communication through nCPU.

Pipes, message queues, and shared memory — all managed with neural ops:
- Pipe: circular buffer with neural read/write pointers
- Message queue: priority queue with neural CMP for ordering
- Shared memory: mapped regions with neural access control
- Semaphore: neural counter for synchronization

Usage:
    python -m bridge.neural_ipc demo
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


class NeuralPipe:
    """Circular buffer pipe with neural pointer arithmetic.
    
    write_ptr and read_ptr managed through neural ADD and MOD.
    Fullness/emptiness checked with neural CMP.
    """
    
    def __init__(self, bridge: NCPUBridge, capacity: int = 64):
        self.bridge = bridge
        self.capacity = capacity
        self.buffer = [0] * capacity
        self.write_ptr = 0
        self.read_ptr = 0
        self.count = 0
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def _neural_mod(self, val: int, mod: int) -> int:
        """Neural modulo: val - (val/mod)*mod."""
        q = self.bridge.div(val, mod)
        self._op()
        return self.bridge.sub(val, self.bridge.mul(q, mod))
    
    def write(self, data: list[int]) -> int:
        """Write bytes to pipe. Returns bytes written."""
        written = 0
        for byte in data:
            # Neural: is pipe full?
            zf, _ = self.bridge.cmp(self.count, self.capacity)
            self._op()
            if zf:  # count == capacity → full
                break
            
            self.buffer[self.write_ptr] = byte
            self.write_ptr = self._neural_mod(
                self.bridge.add(self.write_ptr, 1), self.capacity
            )
            self._op()
            self.count = self.bridge.add(self.count, 1)
            self._op()
            written += 1
        
        return written
    
    def read(self, n: int) -> list[int]:
        """Read up to n bytes from pipe."""
        data = []
        for _ in range(n):
            # Neural: is pipe empty?
            zf, _ = self.bridge.cmp(self.count, 0)
            self._op()
            if zf:  # count == 0 → empty
                break
            
            byte = self.buffer[self.read_ptr]
            data.append(byte)
            self.read_ptr = self._neural_mod(
                self.bridge.add(self.read_ptr, 1), self.capacity
            )
            self._op()
            self.count = self.bridge.sub(self.count, 1)
            self._op()
        
        return data


@dataclass
class Message:
    priority: int
    sender: int  # PID
    data: list[int]
    timestamp: float = 0.0


class NeuralMessageQueue:
    """Priority message queue with neural ordering.
    
    Messages sorted by priority using neural CMP.
    """
    
    def __init__(self, bridge: NCPUBridge, max_size: int = 32):
        self.bridge = bridge
        self.max_size = max_size
        self._queue: list[Message] = []
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def send(self, priority: int, sender: int, data: list[int]) -> bool:
        """Send a message. Inserted in priority order (neural CMP)."""
        zf, _ = self.bridge.cmp(len(self._queue), self.max_size)
        self._op()
        if zf:
            return False  # Queue full
        
        msg = Message(priority=priority, sender=sender, data=data, timestamp=time.time())
        
        # Find insertion point (neural binary-ish search)
        pos = len(self._queue)
        for i in range(len(self._queue)):
            zf, sf = self.bridge.cmp(self._queue[i].priority, priority)
            self._op()
            if not sf and not zf:  # queue[i].priority > msg.priority → insert before
                pos = i
                break
        
        self._queue.insert(pos, msg)
        return True
    
    def recv(self) -> Optional[Message]:
        """Receive highest priority message (lowest number)."""
        if not self._queue:
            return None
        return self._queue.pop(0)
    
    def peek(self) -> Optional[Message]:
        return self._queue[0] if self._queue else None
    
    @property
    def size(self) -> int:
        return len(self._queue)


class NeuralSemaphore:
    """Counting semaphore with neural counter."""
    
    def __init__(self, bridge: NCPUBridge, initial: int = 1):
        self.bridge = bridge
        self.value = initial
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def acquire(self) -> bool:
        """Try to acquire (decrement). Returns True if acquired."""
        zf, sf = self.bridge.cmp(self.value, 0)
        self._op()
        
        if sf or zf:  # value <= 0
            return False
        
        self.value = self.bridge.sub(self.value, 1)
        self._op()
        return True
    
    def release(self):
        """Release (increment)."""
        self.value = self.bridge.add(self.value, 1)
        self._op()


class NeuralSharedMemory:
    """Shared memory region with neural access control."""
    
    def __init__(self, bridge: NCPUBridge, size: int = 256):
        self.bridge = bridge
        self.size = size
        self.memory = [0] * size
        self._owners: set[int] = set()
        self._sem = NeuralSemaphore(bridge, initial=1)
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def attach(self, pid: int):
        """Attach process to shared memory."""
        self._owners.add(pid)
    
    def detach(self, pid: int):
        self._owners.discard(pid)
    
    def write(self, pid: int, offset: int, data: list[int]) -> bool:
        """Write with neural bounds checking and locking."""
        if pid not in self._owners:
            return False
        
        # Acquire semaphore
        if not self._sem.acquire():
            return False
        
        # Neural bounds check
        end = self.bridge.add(offset, len(data))
        self._op()
        zf, sf = self.bridge.cmp(end, self.size)
        self._op()
        
        if not sf and not zf:  # end > size
            self._sem.release()
            return False
        
        for i, byte in enumerate(data):
            idx = self.bridge.add(offset, i)
            self._op()
            self.memory[idx] = byte
        
        self._sem.release()
        return True
    
    def read(self, pid: int, offset: int, length: int) -> Optional[list[int]]:
        """Read with neural bounds checking."""
        if pid not in self._owners:
            return None
        
        end = self.bridge.add(offset, length)
        self._op()
        zf, sf = self.bridge.cmp(end, self.size)
        self._op()
        
        if not sf and not zf:
            return None
        
        return self.memory[offset:offset + length]


# ── CLI ──

def demo():
    bridge = NCPUBridge()
    
    print("Neural IPC — Inter-Process Communication")
    print("=" * 60)
    print("Pipes, queues, semaphores, shared memory → neural ALU\n")
    
    # ── Pipe ──
    print("── Neural Pipe (circular buffer) ──")
    pipe = NeuralPipe(bridge, capacity=16)
    
    msg = [ord(c) for c in "Hello"]
    written = pipe.write(msg)
    print(f"  Write 'Hello' ({written} bytes) → ptr={pipe.write_ptr}, count={pipe.count}")
    
    data = pipe.read(3)
    print(f"  Read 3 bytes → '{''.join(chr(b) for b in data)}' ptr={pipe.read_ptr}, count={pipe.count}")
    
    data = pipe.read(10)
    print(f"  Read rest → '{''.join(chr(b) for b in data)}' count={pipe.count}")
    
    # Wrap-around test
    big = list(range(10))
    pipe.write(big)
    pipe.read(5)
    pipe.write([99, 98, 97])
    data = pipe.read(8)
    print(f"  Wrap-around: wrote 10, read 5, wrote 3, read 8 → {data}")
    print(f"  Ops: {pipe._ops}")
    print()
    
    # ── Message Queue ──
    print("── Neural Message Queue ──")
    mq = NeuralMessageQueue(bridge)
    
    mq.send(priority=3, sender=1, data=[1, 2, 3])
    mq.send(priority=1, sender=2, data=[4, 5])
    mq.send(priority=5, sender=1, data=[6])
    mq.send(priority=0, sender=3, data=[7, 8, 9])
    mq.send(priority=2, sender=2, data=[10])
    
    print(f"  Sent 5 messages with priorities 3,1,5,0,2")
    print(f"  Receive order (highest priority first):")
    while mq.size > 0:
        msg = mq.recv()
        print(f"    p={msg.priority} from PID {msg.sender}: {msg.data}")
    print(f"  Ops: {mq._ops}")
    print()
    
    # ── Semaphore ──
    print("── Neural Semaphore ──")
    sem = NeuralSemaphore(bridge, initial=2)
    
    r1 = sem.acquire()
    r2 = sem.acquire()
    r3 = sem.acquire()  # Should fail
    print(f"  Initial value: 2")
    print(f"  Acquire 1: {'✅' if r1 else '❌'} (value={sem.value})")
    print(f"  Acquire 2: {'✅' if r2 else '❌'} (value={sem.value})")
    print(f"  Acquire 3: {'❌ blocked' if not r3 else '⚠️'} (value={sem.value})")
    
    sem.release()
    r4 = sem.acquire()
    print(f"  Release + Acquire: {'✅' if r4 else '❌'} (value={sem.value})")
    print(f"  Ops: {sem._ops}")
    print()
    
    # ── Shared Memory ──
    print("── Neural Shared Memory ──")
    shm = NeuralSharedMemory(bridge, size=64)
    
    shm.attach(pid=1)
    shm.attach(pid=2)
    
    ok = shm.write(1, 0, [72, 101, 108, 108, 111])  # "Hello"
    print(f"  PID 1 writes 'Hello' at offset 0: {'✅' if ok else '❌'}")
    
    data = shm.read(2, 0, 5)
    print(f"  PID 2 reads 5 bytes: '{''.join(chr(b) for b in data)}'")
    
    # Unauthorized access
    data = shm.read(3, 0, 5)
    print(f"  PID 3 (not attached) reads: {'❌ DENIED' if data is None else '⚠️'}")
    
    # Bounds check
    ok = shm.write(1, 60, [1, 2, 3, 4, 5, 6])
    print(f"  PID 1 overflow write at offset 60 (6 bytes into 64): {'❌ DENIED' if not ok else '⚠️'}")
    print(f"  Ops: {shm._ops}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_ipc [demo]")


if __name__ == "__main__":
    main()
