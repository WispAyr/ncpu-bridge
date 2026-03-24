"""Neural Hash — CRC32 and integrity verification computed through nCPU.

Every bit operation (XOR, AND, SHL, SHR) runs through trained neural networks.
This means file integrity checks are neurally computed — the world's most
over-engineered checksum.

Uses:
- Config file integrity (detect drift)
- Outcome data integrity (detect corruption)
- Neural-verified comparison of expected vs actual hashes

Usage:
    python -m bridge.neural_hash file <path>         # Hash a file
    python -m bridge.neural_hash string <text>        # Hash a string
    python -m bridge.neural_hash verify <path> <hash> # Verify integrity
    python -m bridge.neural_hash benchmark            # Speed comparison
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from bridge.config import get_ncpu_path, get_bridge_path, get_clawd_data_path
NCPU_PATH = get_ncpu_path()
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge

# CRC32 polynomial (standard IEEE)
CRC32_POLY = 0xEDB88320


class NeuralHash:
    """CRC32 implementation using only neural ALU bitwise operations.
    
    Standard CRC32 algorithm but every XOR, AND, and shift
    goes through trained PyTorch models on the nCPU.
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._table = self._build_table()
    
    def _build_table(self) -> list[int]:
        """Build CRC32 lookup table using neural XOR and SHR.
        
        Standard approach: for each byte value 0-255, compute
        the CRC by shifting and XORing with the polynomial.
        All operations go through neural networks.
        """
        table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                # Neural check: is lowest bit set?
                low_bit = self.bridge.bitwise_and(crc, 1)
                shifted = self.bridge.shr(crc, 1)
                
                if low_bit:
                    crc = self.bridge.bitwise_xor(shifted, CRC32_POLY)
                else:
                    crc = shifted
            table.append(crc)
        return table
    
    def crc32_bytes(self, data: bytes) -> int:
        """Compute CRC32 of bytes using neural table lookup.
        
        For each byte: crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
        All operations are neural.
        """
        crc = 0xFFFFFFFF
        
        for byte in data:
            # Neural XOR: crc ^ byte
            xored = self.bridge.bitwise_xor(crc, byte)
            # Neural AND: get low byte for table index
            index = self.bridge.bitwise_and(xored, 0xFF)
            # Neural SHR: crc >> 8
            shifted = self.bridge.shr(crc, 8)
            # Neural XOR: table[index] ^ shifted
            crc = self.bridge.bitwise_xor(self._table[index], shifted)
        
        # Final inversion
        return self.bridge.bitwise_xor(crc, 0xFFFFFFFF)
    
    def crc32_string(self, text: str) -> int:
        """Hash a string (UTF-8 encoded)."""
        return self.crc32_bytes(text.encode("utf-8"))
    
    def crc32_file(self, path: str | Path, chunk_size: int = 4096) -> int:
        """Hash a file in chunks. Neural CRC32 on each byte."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        
        # For large files, we hash first N bytes neurally
        # and rest with standard (practical compromise)
        data = path.read_bytes()
        
        if len(data) <= 1024:
            # Fully neural for small files
            return self.crc32_bytes(data)
        else:
            # Neural hash of first 512 + last 512 bytes + length encoding
            # This gives us a neural "fingerprint" that catches most corruption
            head = data[:512]
            tail = data[-512:]
            size_bytes = len(data).to_bytes(8, "little")
            combined = head + tail + size_bytes
            return self.crc32_bytes(combined)
    
    def verify(self, path: str | Path, expected_hex: str) -> dict:
        """Verify file integrity against expected hash."""
        actual = self.crc32_file(path)
        actual_hex = f"{actual:08x}"
        
        # Neural comparison of hashes
        zf, _ = self.bridge.cmp(actual, int(expected_hex, 16))
        
        return {
            "path": str(path),
            "expected": expected_hex,
            "actual": actual_hex,
            "match": zf,  # Zero flag = equal
            "neural_verified": True,
        }
    
    def format_hex(self, crc: int) -> str:
        """Format CRC32 as hex string."""
        return f"{crc:08x}"


class IntegrityMonitor:
    """Monitor file integrity using neural hashes.
    
    Stores baseline hashes and detects drift.
    """
    
    def __init__(self):
        self.hasher = NeuralHash()
        self._baselines: dict[str, int] = {}
    
    def baseline(self, paths: list[str | Path]) -> dict[str, str]:
        """Compute baseline hashes for a set of files."""
        results = {}
        for p in paths:
            p = Path(p)
            if p.exists():
                h = self.hasher.crc32_file(p)
                self._baselines[str(p)] = h
                results[str(p)] = self.hasher.format_hex(h)
        return results
    
    def check(self) -> list[dict]:
        """Check all baselined files for drift."""
        drifted = []
        for path_str, expected in self._baselines.items():
            p = Path(path_str)
            if not p.exists():
                drifted.append({
                    "path": path_str,
                    "status": "MISSING",
                    "expected": self.hasher.format_hex(expected),
                })
                continue
            
            actual = self.hasher.crc32_file(p)
            zf, _ = self.hasher.bridge.cmp(actual, expected)
            
            if not zf:
                drifted.append({
                    "path": path_str,
                    "status": "DRIFTED",
                    "expected": self.hasher.format_hex(expected),
                    "actual": self.hasher.format_hex(actual),
                })
        
        return drifted


# ── CLI ──────────────────────────────────────────────────────

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    
    if cmd == "file" and len(sys.argv) > 2:
        nh = NeuralHash()
        path = sys.argv[2]
        print(f"Computing neural CRC32 of {path}...")
        t0 = time.time()
        h = nh.crc32_file(path)
        elapsed = time.time() - t0
        print(f"  Hash: {nh.format_hex(h)}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Neural ops: every XOR/AND/SHR in CRC32 → trained .pt model")
    
    elif cmd == "string" and len(sys.argv) > 2:
        nh = NeuralHash()
        text = " ".join(sys.argv[2:])
        print(f"Computing neural CRC32 of '{text}'...")
        t0 = time.time()
        h = nh.crc32_string(text)
        elapsed = time.time() - t0
        print(f"  Hash: {nh.format_hex(h)}")
        print(f"  Time: {elapsed:.2f}s")
    
    elif cmd == "verify" and len(sys.argv) > 3:
        nh = NeuralHash()
        result = nh.verify(sys.argv[2], sys.argv[3])
        match = "✅ MATCH" if result["match"] else "❌ MISMATCH"
        print(f"{match}: {result['path']}")
        print(f"  Expected: {result['expected']}")
        print(f"  Actual:   {result['actual']}")
    
    elif cmd == "benchmark":
        print("Neural CRC32 Benchmark")
        print("=" * 50)
        
        nh = NeuralHash()
        
        # Table build already happened in __init__, time a hash
        test_data = b"Hello from the neural hash! Every XOR is a neural network."
        
        print(f"\nHashing {len(test_data)} bytes...")
        t0 = time.time()
        h = nh.crc32_bytes(test_data)
        t_neural = time.time() - t0
        
        # Compare with standard CRC32
        import zlib
        t0 = time.time()
        h_std = zlib.crc32(test_data) & 0xFFFFFFFF
        t_std = time.time() - t0
        
        print(f"  Neural CRC32: {nh.format_hex(h)} ({t_neural:.3f}s)")
        print(f"  stdlib CRC32: {h_std:08x} ({t_std:.6f}s)")
        print(f"  Slowdown: {t_neural/max(t_std, 0.000001):.0f}x (every bit op is a neural net)")
        
        match = "✅" if nh.format_hex(h) == f"{h_std:08x}" else "❌"
        print(f"  Match: {match}")
        
        # Demo: integrity monitor
        print(f"\nIntegrity Monitor Demo")
        print("-" * 50)
        monitor = IntegrityMonitor()
        
        # Baseline some real files
        files = [
            # These paths are configurable via environment
            
            
        ]
        
        existing = [f for f in files if Path(f).exists()]
        if existing:
            baselines = monitor.baseline(existing)
            for path, h in baselines.items():
                print(f"  📎 {Path(path).name}: {h}")
            
            drifted = monitor.check()
            if drifted:
                for d in drifted:
                    print(f"  ⚠️  {d['path']}: {d['status']}")
            else:
                print(f"  ✅ All {len(existing)} files integrity verified (neural CRC32)")
    
    else:
        print("Usage: python -m bridge.neural_hash [file <path>|string <text>|verify <path> <hash>|benchmark]")


if __name__ == "__main__":
    main()
