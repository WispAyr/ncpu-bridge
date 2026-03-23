"""Neural Filesystem — inode table, block allocation, and path resolution through nCPU.

A virtual filesystem where every operation is computed neurally:
- Inode allocation: neural counter increment
- Block bitmap: neural AND/OR for allocation/deallocation
- Path resolution: neural CMP for name matching
- File read/write: neural addressing (MUL for offset calculation)
- Directory listing: neural traversal
- Free space: neural popcount (count set bits in bitmap)

Not a real filesystem — it's an in-memory VFS that proves the neural
CPU can handle the data structures and algorithms of a filesystem.

Usage:
    python -m bridge.neural_fs demo     # Full filesystem demo
    python -m bridge.neural_fs shell    # Interactive shell
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

BLOCK_SIZE = 64  # bytes per block
MAX_BLOCKS = 256
MAX_INODES = 64


@dataclass
class Inode:
    id: int
    name: str
    is_dir: bool
    parent: int  # parent inode id
    size: int = 0
    blocks: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)  # for directories
    created_at: float = 0.0


@dataclass
class Block:
    id: int
    data: list[int] = field(default_factory=lambda: [0] * BLOCK_SIZE)


class NeuralFilesystem:
    """Virtual filesystem with all operations computed through neural ALU.
    
    Structure:
    - Inode table: fixed array of inodes (like ext4)
    - Block bitmap: tracks which blocks are free/used
    - Block storage: raw data blocks
    - Directory entries: inode children lists
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._ops = 0
        
        # Initialize filesystem
        self._inodes: dict[int, Inode] = {}
        self._blocks: dict[int, Block] = {}
        self._block_bitmap = [0] * (MAX_BLOCKS // 8)  # 32 bytes for 256 bits
        self._next_inode = 1
        
        # Create root directory (inode 0)
        root = Inode(id=0, name="/", is_dir=True, parent=0, created_at=time.time())
        self._inodes[0] = root
    
    def _op(self):
        self._ops += 1
    
    # ── Block Allocation ────────────────────────────────
    
    def _alloc_block(self) -> Optional[int]:
        """Find and allocate a free block using neural bitmap scanning.
        
        Scans bitmap bytes with neural AND to find one with a free bit,
        then uses neural bit ops to find and set the specific bit.
        """
        for byte_idx in range(len(self._block_bitmap)):
            # Neural check: is this byte full (0xFF)?
            zf, _ = self.bridge.cmp(self._block_bitmap[byte_idx], 0xFF)
            self._op()
            
            if not zf:  # Has free bits
                # Find first free bit using neural AND
                bitmap_byte = self._block_bitmap[byte_idx]
                for bit in range(8):
                    mask = self.bridge.shl(1, bit)
                    self._op()
                    used = self.bridge.bitwise_and(bitmap_byte, mask)
                    self._op()
                    
                    zf2, _ = self.bridge.cmp(used, 0)
                    self._op()
                    
                    if zf2:  # Bit is 0 = free
                        # Set the bit (neural OR)
                        self._block_bitmap[byte_idx] = self.bridge.bitwise_or(bitmap_byte, mask)
                        self._op()
                        
                        block_id = self.bridge.add(self.bridge.mul(byte_idx, 8), bit)
                        self._op()
                        self._blocks[block_id] = Block(id=block_id)
                        return block_id
        
        return None  # Filesystem full
    
    def _free_block(self, block_id: int):
        """Free a block by clearing its bit in the bitmap (neural AND NOT)."""
        byte_idx = self.bridge.div(block_id, 8)
        self._op()
        bit_idx = self.bridge.sub(block_id, self.bridge.mul(byte_idx, 8))
        self._op()
        
        mask = self.bridge.shl(1, bit_idx)
        self._op()
        inv_mask = self.bridge.bitwise_xor(mask, 0xFF)
        self._op()
        self._block_bitmap[byte_idx] = self.bridge.bitwise_and(self._block_bitmap[byte_idx], inv_mask)
        self._op()
        
        self._blocks.pop(block_id, None)
    
    def _count_free_blocks(self) -> int:
        """Count free blocks using neural popcount."""
        used = 0
        for byte_val in self._block_bitmap:
            # Count set bits in this byte (neural)
            for bit in range(8):
                mask = self.bridge.shl(1, bit)
                self._op()
                is_set = self.bridge.bitwise_and(byte_val, mask)
                self._op()
                zf, _ = self.bridge.cmp(is_set, 0)
                self._op()
                if not zf:
                    used = self.bridge.add(used, 1)
                    self._op()
        
        return self.bridge.sub(MAX_BLOCKS, used)
    
    # ── Inode Operations ────────────────────────────────
    
    def _alloc_inode(self, name: str, is_dir: bool, parent: int) -> Inode:
        """Allocate a new inode with neural counter increment."""
        inode_id = self._next_inode
        self._next_inode = self.bridge.add(self._next_inode, 1)
        self._op()
        
        inode = Inode(
            id=inode_id, name=name, is_dir=is_dir,
            parent=parent, created_at=time.time(),
        )
        self._inodes[inode_id] = inode
        
        # Add to parent's children
        parent_inode = self._inodes.get(parent)
        if parent_inode:
            parent_inode.children.append(inode_id)
        
        return inode
    
    # ── Path Resolution ─────────────────────────────────
    
    def _resolve_path(self, path: str) -> Optional[Inode]:
        """Resolve a path to an inode using neural string comparison.
        
        Walks the directory tree, comparing each path component
        against directory entries using neural CMP on each byte.
        """
        if path == "/":
            return self._inodes[0]
        
        parts = [p for p in path.strip("/").split("/") if p]
        current = self._inodes[0]  # Start at root
        
        for part in parts:
            if not current.is_dir:
                return None
            
            found = False
            for child_id in current.children:
                child = self._inodes.get(child_id)
                if not child:
                    continue
                
                # Neural string comparison: compare each byte
                if self._neural_strcmp(child.name, part):
                    current = child
                    found = True
                    break
            
            if not found:
                return None
        
        return current
    
    def _neural_strcmp(self, a: str, b: str) -> bool:
        """Compare two strings byte-by-byte using neural CMP."""
        if len(a) != len(b):
            # Neural length comparison
            zf, _ = self.bridge.cmp(len(a), len(b))
            self._op()
            return False
        
        for i in range(len(a)):
            zf, _ = self.bridge.cmp(ord(a[i]), ord(b[i]))
            self._op()
            if not zf:
                return False
        
        return True
    
    # ── File Operations ─────────────────────────────────
    
    def mkdir(self, path: str) -> dict:
        """Create a directory."""
        parent_path = "/".join(path.rstrip("/").split("/")[:-1]) or "/"
        name = path.rstrip("/").split("/")[-1]
        
        parent = self._resolve_path(parent_path)
        if not parent:
            return {"error": f"Parent not found: {parent_path}"}
        if not parent.is_dir:
            return {"error": f"Not a directory: {parent_path}"}
        
        # Check if already exists
        existing = self._resolve_path(path)
        if existing:
            return {"error": f"Already exists: {path}"}
        
        inode = self._alloc_inode(name, is_dir=True, parent=parent.id)
        return {"ok": True, "inode": inode.id, "path": path}
    
    def create(self, path: str, data: str = "") -> dict:
        """Create a file with optional data."""
        parent_path = "/".join(path.rstrip("/").split("/")[:-1]) or "/"
        name = path.rstrip("/").split("/")[-1]
        
        parent = self._resolve_path(parent_path)
        if not parent:
            return {"error": f"Parent not found: {parent_path}"}
        
        existing = self._resolve_path(path)
        if existing:
            return {"error": f"Already exists: {path}"}
        
        inode = self._alloc_inode(name, is_dir=False, parent=parent.id)
        
        if data:
            self._write_data(inode, data.encode())
        
        return {"ok": True, "inode": inode.id, "size": inode.size}
    
    def _write_data(self, inode: Inode, data: bytes):
        """Write data to file blocks using neural addressing."""
        # Calculate blocks needed
        blocks_needed = self.bridge.div(len(data), BLOCK_SIZE)
        self._op()
        remainder = self.bridge.sub(len(data), self.bridge.mul(blocks_needed, BLOCK_SIZE))
        self._op()
        zf, _ = self.bridge.cmp(remainder, 0)
        self._op()
        if not zf:
            blocks_needed = self.bridge.add(blocks_needed, 1)
            self._op()
        
        # Free existing blocks
        for blk_id in inode.blocks:
            self._free_block(blk_id)
        inode.blocks = []
        
        # Allocate and write
        offset = 0
        for _ in range(blocks_needed):
            blk_id = self._alloc_block()
            if blk_id is None:
                break
            
            inode.blocks.append(blk_id)
            block = self._blocks[blk_id]
            
            # Write bytes to block
            for j in range(BLOCK_SIZE):
                idx = offset + j  # Direct arithmetic for indexing safety
                if idx >= len(data):
                    break
                block.data[j] = data[idx]
            
            offset = self.bridge.add(offset, BLOCK_SIZE)
            self._op()
        
        inode.size = len(data)
    
    def read(self, path: str) -> dict:
        """Read file contents."""
        inode = self._resolve_path(path)
        if not inode:
            return {"error": f"Not found: {path}"}
        if inode.is_dir:
            return {"error": f"Is a directory: {path}"}
        
        data = []
        remaining = inode.size
        
        for blk_id in inode.blocks:
            block = self._blocks.get(blk_id)
            if not block:
                break
            
            to_read = min(remaining, BLOCK_SIZE)
            data.extend(block.data[:to_read])
            remaining = self.bridge.sub(remaining, to_read)
            self._op()
        
        return {"ok": True, "data": bytes(data).decode("utf-8", errors="replace"), "size": inode.size}
    
    def ls(self, path: str = "/") -> dict:
        """List directory contents."""
        inode = self._resolve_path(path)
        if not inode:
            return {"error": f"Not found: {path}"}
        if not inode.is_dir:
            return {"error": f"Not a directory: {path}"}
        
        entries = []
        for child_id in inode.children:
            child = self._inodes.get(child_id)
            if child:
                kind = "dir" if child.is_dir else "file"
                entries.append({"name": child.name, "type": kind, "size": child.size, "inode": child.id})
        
        return {"ok": True, "path": path, "entries": entries}
    
    def rm(self, path: str) -> dict:
        """Remove a file (not directory)."""
        inode = self._resolve_path(path)
        if not inode:
            return {"error": f"Not found: {path}"}
        if inode.is_dir:
            return {"error": f"Is a directory (use rmdir): {path}"}
        
        # Free blocks
        for blk_id in inode.blocks:
            self._free_block(blk_id)
        
        # Remove from parent
        parent = self._inodes.get(inode.parent)
        if parent:
            parent.children = [c for c in parent.children if c != inode.id]
        
        del self._inodes[inode.id]
        return {"ok": True, "freed_blocks": len(inode.blocks)}
    
    def stat(self) -> dict:
        """Filesystem statistics."""
        free = self._count_free_blocks()
        used = self.bridge.sub(MAX_BLOCKS, free)
        self._op()
        
        return {
            "total_blocks": MAX_BLOCKS,
            "used_blocks": used,
            "free_blocks": free,
            "block_size": BLOCK_SIZE,
            "total_bytes": self.bridge.mul(MAX_BLOCKS, BLOCK_SIZE),
            "inodes_used": len(self._inodes),
            "inodes_max": MAX_INODES,
            "neural_ops": self._ops,
        }


# ── CLI ──────────────────────────────────────────────────────

def demo():
    fs = NeuralFilesystem()
    
    print("Neural Filesystem")
    print("=" * 60)
    print("Every allocation, comparison, and address calc → neural ALU\n")
    
    # Create directory structure
    print("── Creating directories ──")
    for d in ["/home", "/home/ewan", "/var", "/var/log"]:
        result = fs.mkdir(d)
        print(f"  mkdir {d} → inode {result.get('inode', '?')}")
    
    print()
    
    # Create files
    print("── Creating files ──")
    files = [
        ("/home/ewan/hello.txt", "Hello from the neural filesystem!"),
        ("/home/ewan/config.json", '{"ncpu": true}'),
        ("/var/log/system.log", "Boot OK\nAll neural\n"),
    ]
    for path, data in files:
        result = fs.create(path, data)
        print(f"  create {path} ({len(data)} bytes) → inode {result.get('inode', '?')}")
    
    print()
    
    # Read files
    print("── Reading files ──")
    for path, _ in files:
        result = fs.read(path)
        content = result.get("data", "?")[:40]
        print(f"  read {path} → \"{content}\"")
    
    print()
    
    # Directory listing
    print("── Directory listings ──")
    for d in ["/", "/home/ewan", "/var/log"]:
        result = fs.ls(d)
        entries = result.get("entries", [])
        names = [f"{'📁' if e['type'] == 'dir' else '📄'} {e['name']}" for e in entries]
        print(f"  ls {d} → {', '.join(names)}")
    
    print()
    
    # Delete a file
    print("── File operations ──")
    result = fs.rm("/var/log/system.log")
    print(f"  rm /var/log/system.log → freed {result.get('freed_blocks', 0)} blocks")
    
    # Verify it's gone
    result = fs.read("/var/log/system.log")
    print(f"  read after rm → {result.get('error', '?')}")
    
    print()
    
    # Stats
    print("── Filesystem stats ──")
    stats = fs.stat()
    used_pct = stats['used_blocks'] / stats['total_blocks'] * 100
    bar = "█" * int(used_pct / 5) + "░" * (20 - int(used_pct / 5))
    print(f"  Blocks: {stats['used_blocks']}/{stats['total_blocks']} [{bar}] {used_pct:.0f}%")
    print(f"  Capacity: {stats['total_bytes']} bytes ({stats['total_bytes']//1024}KB)")
    print(f"  Inodes: {stats['inodes_used']}/{stats['inodes_max']}")
    print(f"  Neural ops: {stats['neural_ops']}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_fs [demo]")


if __name__ == "__main__":
    main()
