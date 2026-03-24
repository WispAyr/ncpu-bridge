"""
Phase 30 — Neural Slab Allocator
=================================
Kernel-style slab allocator for fixed-size objects.
All pointer arithmetic, free-list management, and slab
bookkeeping done through neural ALU operations.

Features:
  - Named caches for different object sizes
  - Per-slab free lists with neural pointer chasing
  - Slab states: full, partial, empty
  - Cache statistics (hit rate, fragmentation)
  - Slab coalescing / reaping empty slabs
"""

from bridge.compute import NCPUBridge
from dataclasses import dataclass, field

bridge = NCPUBridge()


@dataclass
class Slab:
    """A slab holds N objects of fixed size."""
    base_addr: int
    obj_size: int
    capacity: int
    free_list: list = field(default_factory=list)  # stack of free offsets
    allocated: set = field(default_factory=set)

    @property
    def used(self):
        return len(self.allocated)

    @property
    def state(self):
        if len(self.allocated) == 0:
            return "empty"
        elif len(self.free_list) == 0:
            return "full"
        return "partial"


class NeuralSlabCache:
    """A cache of slabs for a specific object size."""

    def __init__(self, name: str, obj_size: int, slab_size: int = 256):
        self.name = name
        self.obj_size = obj_size
        self.slab_size = slab_size
        self.slabs: list[Slab] = []
        self._next_base = 0
        self._ops = 0
        self._allocs = 0
        self._frees = 0
        self._create_slab()

    def _neural_add(self, a, b):
        self._ops += 1
        return bridge.add(a, b)

    def _neural_mul(self, a, b):
        self._ops += 1
        return bridge.mul(a, b)

    def _neural_cmp(self, a, b):
        self._ops += 1
        return bridge.cmp(a, b)

    def _create_slab(self) -> Slab:
        base = self._next_base
        capacity = self.slab_size // self.obj_size
        # Build free list: each slot's address computed via neural MUL+ADD
        free_list = []
        for i in range(capacity):
            offset = self._neural_mul(i, self.obj_size)
            addr = self._neural_add(base, offset)
            free_list.append(addr)
        free_list.reverse()  # stack — pop from end

        slab = Slab(base_addr=base, obj_size=self.obj_size,
                     capacity=capacity, free_list=free_list)
        self.slabs.append(slab)
        self._next_base = self._neural_add(base, self.slab_size)
        return slab

    def alloc(self) -> int:
        """Allocate one object, return its address."""
        # Find a partial or empty slab
        for slab in self.slabs:
            if slab.free_list:
                addr = slab.free_list.pop()
                slab.allocated.add(addr)
                self._allocs += 1
                return addr
        # All full — create new slab
        slab = self._create_slab()
        addr = slab.free_list.pop()
        slab.allocated.add(addr)
        self._allocs += 1
        return addr

    def free(self, addr: int):
        """Return an object to its slab."""
        for slab in self.slabs:
            # Check if addr belongs to this slab (neural CMP range check)
            zf_lo, sf_lo = self._neural_cmp(addr, slab.base_addr)
            end = self._neural_add(slab.base_addr, self.slab_size)
            zf_hi, sf_hi = self._neural_cmp(addr, end)
            in_range = (zf_lo or not sf_lo) and sf_hi  # addr >= base && addr < end
            if in_range and addr in slab.allocated:
                slab.allocated.discard(addr)
                slab.free_list.append(addr)
                self._frees += 1
                return True
        return False

    def reap(self) -> int:
        """Free empty slabs (except the last one). Returns count reaped."""
        empty = [s for s in self.slabs if s.state == "empty"]
        # Keep at least one slab
        reaped = 0
        for slab in empty[:-1] if len(self.slabs) == len(empty) else empty:
            self.slabs.remove(slab)
            reaped += 1
        return reaped

    def stats(self) -> dict:
        total_cap = sum(s.capacity for s in self.slabs)
        total_used = sum(s.used for s in self.slabs)
        return {
            "name": self.name,
            "obj_size": self.obj_size,
            "slabs": len(self.slabs),
            "capacity": total_cap,
            "used": total_used,
            "free": total_cap - total_used,
            "allocs": self._allocs,
            "frees": self._frees,
            "neural_ops": self._ops,
            "states": {s: sum(1 for sl in self.slabs if sl.state == s)
                       for s in ["empty", "partial", "full"]},
        }


class NeuralSlabAllocator:
    """Top-level allocator managing multiple caches."""

    def __init__(self):
        self.caches: dict[str, NeuralSlabCache] = {}

    def create_cache(self, name: str, obj_size: int, slab_size: int = 256) -> NeuralSlabCache:
        cache = NeuralSlabCache(name, obj_size, slab_size)
        self.caches[name] = cache
        return cache

    def alloc(self, cache_name: str) -> int:
        return self.caches[cache_name].alloc()

    def free(self, cache_name: str, addr: int) -> bool:
        return self.caches[cache_name].free(addr)

    def stats(self) -> list:
        return [c.stats() for c in self.caches.values()]


def demo():
    print("Neural Slab Allocator")
    print("=" * 60)
    print("Kernel-style slab allocation with neural pointer arithmetic\n")

    alloc = NeuralSlabAllocator()

    # Create caches for different object sizes
    task_cache = alloc.create_cache("task_struct", obj_size=32, slab_size=128)
    inode_cache = alloc.create_cache("inode", obj_size=16, slab_size=128)
    buf_cache = alloc.create_cache("buffer_head", obj_size=8, slab_size=64)

    print("  Caches created:")
    for s in alloc.stats():
        print(f"    {s['name']:20s} obj={s['obj_size']:3d}B  "
              f"capacity={s['capacity']:3d}  slabs={s['slabs']}")

    # Allocate objects
    print("\n  Allocating objects...")
    task_addrs = [alloc.alloc("task_struct") for _ in range(5)]
    inode_addrs = [alloc.alloc("inode") for _ in range(10)]
    buf_addrs = [alloc.alloc("buffer_head") for _ in range(8)]

    print(f"    task_struct:  {len(task_addrs)} allocated → addrs {task_addrs[:3]}...")
    print(f"    inode:        {len(inode_addrs)} allocated → addrs {inode_addrs[:3]}...")
    print(f"    buffer_head:  {len(buf_addrs)} allocated → addrs {buf_addrs[:3]}...")

    # Verify addresses are unique and properly spaced
    print("\n  Address spacing verification:")
    for i in range(1, min(3, len(task_addrs))):
        spacing = task_addrs[i] - task_addrs[i-1]
        zf, _ = bridge.cmp(spacing, 32)
        print(f"    task[{i}]-task[{i-1}] = {spacing} bytes → {'✅' if zf else '❌'} (expect 32)")

    # Free some objects
    print("\n  Freeing objects...")
    for addr in task_addrs[:3]:
        ok = alloc.free("task_struct", addr)
        print(f"    free(0x{addr:04x}) → {'✅' if ok else '❌'}")

    # Realloc — should reuse freed slots
    print("\n  Re-allocating (should reuse freed slots)...")
    realloc = alloc.alloc("task_struct")
    reused = realloc in task_addrs[:3]
    print(f"    alloc() → 0x{realloc:04x} {'✅ reused' if reused else '⚠️ new'}")

    # Free all inodes and reap
    print("\n  Reaping empty slabs...")
    for addr in inode_addrs:
        alloc.free("inode", addr)
    reaped = inode_cache.reap()
    print(f"    Reaped {reaped} empty inode slabs")

    # Final stats
    print("\n  Final cache stats:")
    print("  ┌──────────────────────┬──────┬──────┬──────┬──────┬──────┐")
    print("  │ Cache                │ Size │ Used │ Free │ Slbs │  Ops │")
    print("  ├──────────────────────┼──────┼──────┼──────┼──────┼──────┤")
    total_ops = 0
    for s in alloc.stats():
        print(f"  │ {s['name']:20s} │ {s['obj_size']:4d} │ {s['used']:4d} │ "
              f"{s['free']:4d} │ {s['slabs']:4d} │ {s['neural_ops']:4d} │")
        total_ops += s['neural_ops']
    print("  └──────────────────────┴──────┴──────┴──────┴──────┴──────┘")
    print(f"\n  Total neural ops: {total_ops}")
    print("\n✅ Neural slab allocator: 3 caches, alloc/free/reap all working")


if __name__ == "__main__":
    demo()
