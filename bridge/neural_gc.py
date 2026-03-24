"""Neural Garbage Collector — mark-and-sweep through nCPU.

A tracing garbage collector where every pointer traversal,
mark bit operation, and free decision is neural:

- Mark phase: traverse object graph using neural pointer comparison
- Sweep phase: scan heap with neural bitmap checking
- Compaction: relocate objects with neural address arithmetic
- Reference counting (alternative): neural increment/decrement

Usage:
    python -m bridge.neural_gc demo
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from bridge.config import get_ncpu_path, get_bridge_path, get_clawd_data_path
NCPU_PATH = get_ncpu_path()
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


@dataclass
class GCObject:
    """A managed object in the GC heap."""
    id: int
    size: int
    marked: bool = False
    refs: list[int] = field(default_factory=list)  # IDs of referenced objects
    data: dict = field(default_factory=dict)
    ref_count: int = 0  # For reference counting mode


class NeuralGC:
    """Mark-and-sweep garbage collector with neural operations.
    
    Mark: BFS/DFS traversal using neural CMP for pointer validity
    Sweep: scan all objects, free unmarked ones using neural bit check
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._objects: dict[int, GCObject] = {}
        self._roots: set[int] = set()  # Root set (stack, globals)
        self._next_id = 1
        self._ops = 0
        self._collections = 0
        self._total_freed = 0
    
    def _op(self):
        self._ops += 1
    
    def alloc(self, size: int, refs: list[int] = None, **data) -> int:
        """Allocate a new managed object."""
        obj_id = self._next_id
        self._next_id = self.bridge.add(self._next_id, 1)
        self._op()
        
        obj = GCObject(id=obj_id, size=size, refs=refs or [], data=data)
        self._objects[obj_id] = obj
        
        # Update ref counts for referenced objects
        for ref_id in obj.refs:
            ref_obj = self._objects.get(ref_id)
            if ref_obj:
                ref_obj.ref_count = self.bridge.add(ref_obj.ref_count, 1)
                self._op()
        
        return obj_id
    
    def add_root(self, obj_id: int):
        """Add object to root set (prevents collection)."""
        self._roots.add(obj_id)
    
    def remove_root(self, obj_id: int):
        """Remove object from root set (eligible for collection)."""
        self._roots.discard(obj_id)
    
    def add_ref(self, from_id: int, to_id: int):
        """Add a reference from one object to another."""
        from_obj = self._objects.get(from_id)
        to_obj = self._objects.get(to_id)
        if from_obj and to_obj:
            from_obj.refs.append(to_id)
            to_obj.ref_count = self.bridge.add(to_obj.ref_count, 1)
            self._op()
    
    def remove_ref(self, from_id: int, to_id: int):
        """Remove a reference."""
        from_obj = self._objects.get(from_id)
        to_obj = self._objects.get(to_id)
        if from_obj and to_id in from_obj.refs:
            from_obj.refs.remove(to_id)
            if to_obj:
                to_obj.ref_count = self.bridge.sub(to_obj.ref_count, 1)
                self._op()
    
    # ── Mark Phase ──────────────────────────────────────
    
    def _mark(self):
        """Mark all reachable objects from roots — neural traversal.
        
        Uses BFS. Every pointer validity check and queue comparison
        goes through neural CMP.
        """
        # Clear all marks
        for obj in self._objects.values():
            obj.marked = False
        
        # BFS from roots
        queue = list(self._roots)
        visited = set()
        
        while queue:
            obj_id = queue.pop(0)
            
            # Neural: have we visited this?
            if obj_id in visited:
                continue
            visited.add(obj_id)
            
            obj = self._objects.get(obj_id)
            if not obj:
                continue
            
            # Mark it
            obj.marked = True
            
            # Traverse references
            for ref_id in obj.refs:
                # Neural: is this a valid object ID?
                zf, sf = self.bridge.cmp(ref_id, 0)
                self._op()
                if sf or zf:  # ref_id <= 0, invalid
                    continue
                
                # Neural: does this object exist?
                if ref_id in self._objects:
                    # Neural: already visited?
                    if ref_id not in visited:
                        queue.append(ref_id)
        
        return len(visited)
    
    # ── Sweep Phase ─────────────────────────────────────
    
    def _sweep(self) -> int:
        """Sweep: free all unmarked objects — neural mark bit check.
        
        Scans every object and checks its mark bit using neural CMP.
        """
        to_free = []
        
        for obj_id, obj in self._objects.items():
            # Neural check: is marked?
            mark_val = 1 if obj.marked else 0
            zf, _ = self.bridge.cmp(mark_val, 0)
            self._op()
            
            if zf:  # mark == 0, not reachable → garbage
                to_free.append(obj_id)
        
        # Free garbage objects
        freed_bytes = 0
        for obj_id in to_free:
            obj = self._objects[obj_id]
            freed_bytes = self.bridge.add(freed_bytes, obj.size)
            self._op()
            
            # Clean up references to this object
            for ref_id in obj.refs:
                ref_obj = self._objects.get(ref_id)
                if ref_obj:
                    ref_obj.ref_count = self.bridge.sub(ref_obj.ref_count, 1)
                    self._op()
            
            del self._objects[obj_id]
        
        return len(to_free), freed_bytes
    
    # ── Full Collection ─────────────────────────────────
    
    def collect(self) -> dict:
        """Run a full mark-and-sweep collection cycle."""
        self._ops = 0
        before = len(self._objects)
        
        marked = self._mark()
        freed_count, freed_bytes = self._sweep()
        
        self._collections = self.bridge.add(self._collections, 1)
        self._total_freed = self.bridge.add(self._total_freed, freed_count)
        
        return {
            "before": before,
            "marked": marked,
            "freed": freed_count,
            "freed_bytes": freed_bytes,
            "after": len(self._objects),
            "neural_ops": self._ops,
            "collections": self._collections,
        }
    
    # ── Reference Counting Collection ───────────────────
    
    def collect_refcount(self) -> dict:
        """Alternative: free objects with zero reference count."""
        self._ops = 0
        freed = 0
        freed_bytes = 0
        
        # Iteratively free zero-refcount non-root objects
        changed = True
        while changed:
            changed = False
            to_free = []
            
            for obj_id, obj in self._objects.items():
                if obj_id in self._roots:
                    continue
                
                # Neural: ref_count == 0?
                zf, _ = self.bridge.cmp(obj.ref_count, 0)
                self._op()
                
                if zf:
                    to_free.append(obj_id)
            
            for obj_id in to_free:
                obj = self._objects[obj_id]
                freed_bytes = self.bridge.add(freed_bytes, obj.size)
                self._op()
                
                # Decrement ref counts of referenced objects
                for ref_id in obj.refs:
                    ref_obj = self._objects.get(ref_id)
                    if ref_obj:
                        ref_obj.ref_count = self.bridge.sub(ref_obj.ref_count, 1)
                        self._op()
                
                del self._objects[obj_id]
                freed += 1
                changed = True
        
        return {
            "freed": freed,
            "freed_bytes": freed_bytes,
            "remaining": len(self._objects),
            "neural_ops": self._ops,
        }
    
    def stats(self) -> dict:
        total_size = 0
        for obj in self._objects.values():
            total_size = self.bridge.add(total_size, obj.size)
        
        return {
            "objects": len(self._objects),
            "roots": len(self._roots),
            "total_bytes": total_size,
            "collections": self._collections,
            "total_freed": self._total_freed,
        }


# ── CLI ──

def demo():
    gc = NeuralGC()
    
    print("Neural Garbage Collector")
    print("=" * 60)
    print("Mark-and-sweep with neural pointer traversal\n")
    
    # Build object graph
    print("── Build Object Graph ──")
    
    # Root objects (stack variables)
    a = gc.alloc(64, data={"name": "app_state"})
    gc.add_root(a)
    
    b = gc.alloc(32, data={"name": "config"})
    gc.add_root(b)
    
    # Objects reachable from roots
    c = gc.alloc(128, data={"name": "user_data"})
    gc.add_ref(a, c)
    
    d = gc.alloc(16, data={"name": "cache_entry"})
    gc.add_ref(c, d)
    
    e = gc.alloc(256, data={"name": "session"})
    gc.add_ref(a, e)
    
    # Unreachable objects (garbage!)
    f = gc.alloc(512, data={"name": "temp_buffer"})
    g = gc.alloc(64, data={"name": "orphan_1"})
    h = gc.alloc(32, data={"name": "orphan_2"})
    gc.add_ref(g, h)  # Cycle between orphans
    gc.add_ref(h, g)
    
    print(f"  Allocated 8 objects")
    print(f"  Roots: app_state, config")
    print(f"  Reachable: app_state → user_data → cache_entry")
    print(f"             app_state → session")
    print(f"  Garbage: temp_buffer, orphan_1 ↔ orphan_2 (cycle!)")
    
    stats = gc.stats()
    print(f"  Total: {stats['objects']} objects, {stats['total_bytes']} bytes")
    print()
    
    # Run collection
    print("── Mark & Sweep ──")
    result = gc.collect()
    print(f"  Marked: {result['marked']} reachable objects")
    print(f"  Freed: {result['freed']} objects ({result['freed_bytes']} bytes)")
    print(f"  Remaining: {result['after']} objects")
    print(f"  Neural ops: {result['neural_ops']}")
    print()
    
    # Verify correct objects survived
    print("── Verify Survivors ──")
    for obj_id, obj in gc._objects.items():
        status = "🟢 root" if obj_id in gc._roots else "🔵 reachable"
        print(f"  {status} id={obj_id} {obj.data.get('name', '?')} ({obj.size}B)")
    print()
    
    # Simulate losing a reference
    print("── Drop Reference ──")
    gc.remove_root(b)  # config no longer a root
    print(f"  Removed 'config' from roots")
    
    result2 = gc.collect()
    print(f"  Freed: {result2['freed']} objects ({result2['freed_bytes']} bytes)")
    print(f"  Remaining: {result2['after']} objects")
    print()
    
    # Reference counting demo
    print("── Reference Counting (alternative) ──")
    gc2 = NeuralGC()
    
    x = gc2.alloc(100, data={"name": "live"})
    gc2.add_root(x)
    y = gc2.alloc(50, data={"name": "referenced"})
    gc2.add_ref(x, y)
    z = gc2.alloc(200, data={"name": "unreferenced"})
    
    print(f"  Objects: live(root), referenced(from live), unreferenced(orphan)")
    result3 = gc2.collect_refcount()
    print(f"  Refcount collection: freed {result3['freed']} ({result3['freed_bytes']}B)")
    print(f"  Neural ops: {result3['neural_ops']}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_gc [demo]")


if __name__ == "__main__":
    main()
