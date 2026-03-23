"""Neural Memory Compression — RLE and delta encoding through nCPU.

Compresses data using only neural ALU operations:
- Run-Length Encoding (RLE): neural comparison to detect runs
- Delta Encoding: neural subtraction to store differences
- Hybrid: delta + RLE for time-series data (sensor readings, metrics)

Every comparison, subtraction, and count increment goes through
trained neural networks. The world's most computationally expensive
compression algorithm.

Usage:
    python -m bridge.neural_compress demo           # Run demos
    python -m bridge.neural_compress rle <values>   # RLE compress
    python -m bridge.neural_compress delta <values>  # Delta compress
    python -m bridge.neural_compress file <path>     # Compress file metrics
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


@dataclass
class CompressionResult:
    original_size: int
    compressed_size: int
    ratio: float
    method: str
    data: list
    neural_ops: int
    time_seconds: float


class NeuralCompressor:
    """Compress data using only neural ALU operations."""
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._ops = 0
    
    def _count_op(self):
        self._ops += 1
    
    # ── Run-Length Encoding ──────────────────────────────
    
    def rle_encode(self, values: list[int]) -> CompressionResult:
        """RLE encode: [1,1,1,2,2,3] → [(1,3),(2,2),(3,1)]
        
        Every comparison (is current == previous?) goes through neural CMP.
        Every count increment goes through neural ADD.
        """
        if not values:
            return CompressionResult(0, 0, 1.0, "rle", [], 0, 0.0)
        
        self._ops = 0
        t0 = time.time()
        
        encoded = []
        current = values[0]
        count = 1
        
        for i in range(1, len(values)):
            # Neural comparison: is this value same as current?
            zf, _ = self.bridge.cmp(values[i], current)
            self._count_op()
            
            if zf:  # Same value — increment count neurally
                count = self.bridge.add(count, 1)
                self._count_op()
            else:
                encoded.append((current, count))
                current = values[i]
                count = 1
        
        encoded.append((current, count))
        
        elapsed = time.time() - t0
        compressed_size = len(encoded) * 2  # pairs
        
        return CompressionResult(
            original_size=len(values),
            compressed_size=compressed_size,
            ratio=compressed_size / max(len(values), 1),
            method="rle",
            data=encoded,
            neural_ops=self._ops,
            time_seconds=elapsed,
        )
    
    def rle_decode(self, encoded: list[tuple[int, int]]) -> list[int]:
        """Decode RLE data back to original values."""
        result = []
        for value, count in encoded:
            result.extend([value] * count)
        return result
    
    # ── Delta Encoding ──────────────────────────────────
    
    def delta_encode(self, values: list[int]) -> CompressionResult:
        """Delta encode: [100,102,105,103] → [100, 2, 3, -2]
        
        Every difference computed through neural SUB.
        Great for slowly-changing metrics (disk %, temperature, etc.)
        """
        if not values:
            return CompressionResult(0, 0, 1.0, "delta", [], 0, 0.0)
        
        self._ops = 0
        t0 = time.time()
        
        deltas = [values[0]]  # First value stored as-is (base)
        
        for i in range(1, len(values)):
            # Neural subtraction: delta = current - previous
            delta = self.bridge.sub(values[i], values[i - 1])
            self._count_op()
            deltas.append(delta)
        
        elapsed = time.time() - t0
        
        # Compression benefit: deltas are typically smaller numbers
        # which need fewer bits. We measure "effective bits"
        max_delta = max(abs(d) for d in deltas[1:]) if len(deltas) > 1 else 0
        max_original = max(abs(v) for v in values)
        
        # Approximate compression: bits needed for deltas vs originals
        import math
        bits_original = math.ceil(math.log2(max(max_original, 1) + 1)) * len(values)
        bits_delta = (math.ceil(math.log2(max(max_original, 1) + 1))  # base value
                      + math.ceil(math.log2(max(max_delta, 1) + 1)) * (len(values) - 1))  # deltas
        
        return CompressionResult(
            original_size=bits_original,
            compressed_size=bits_delta,
            ratio=bits_delta / max(bits_original, 1),
            method="delta",
            data=deltas,
            neural_ops=self._ops,
            time_seconds=elapsed,
        )
    
    def delta_decode(self, deltas: list[int]) -> list[int]:
        """Decode delta-encoded data."""
        if not deltas:
            return []
        values = [deltas[0]]
        for i in range(1, len(deltas)):
            values.append(self.bridge.add(values[-1], deltas[i]))
        return values
    
    # ── Hybrid: Delta + RLE ─────────────────────────────
    
    def hybrid_encode(self, values: list[int]) -> CompressionResult:
        """Hybrid compression: delta encode first, then RLE the deltas.
        
        Perfect for metrics that stay flat then jump:
        [95, 95, 95, 96, 96, 96, 96, 97] 
        → deltas: [95, 0, 0, 1, 0, 0, 0, 1]
        → RLE: [(95,1), (0,2), (1,1), (0,3), (1,1)]
        
        All operations neural.
        """
        self._ops = 0
        t0 = time.time()
        
        # Step 1: Delta encode (neural SUB)
        delta_result = self.delta_encode(values)
        deltas = delta_result.data
        
        # Step 2: RLE the deltas (neural CMP + ADD)
        rle_result = self.rle_encode(deltas)
        
        elapsed = time.time() - t0
        total_ops = delta_result.neural_ops + rle_result.neural_ops
        
        return CompressionResult(
            original_size=len(values),
            compressed_size=rle_result.compressed_size,
            ratio=rle_result.compressed_size / max(len(values), 1),
            method="hybrid(delta+rle)",
            data=rle_result.data,
            neural_ops=total_ops,
            time_seconds=elapsed,
        )
    
    def hybrid_decode(self, encoded: list[tuple[int, int]]) -> list[int]:
        """Decode hybrid data."""
        deltas = self.rle_decode(encoded)
        return self.delta_decode(deltas)
    
    # ── Metrics Compression ─────────────────────────────
    
    def compress_metrics(self, metrics: list[dict]) -> dict:
        """Compress a list of metric dicts (from outcomes/sentinel data).
        
        Extracts numeric fields, delta+RLE compresses each series,
        stores metadata for reconstruction.
        """
        if not metrics:
            return {"compressed": [], "schema": {}}
        
        # Extract numeric series by key
        series: dict[str, list[int]] = {}
        for m in metrics:
            for key, val in m.items():
                if isinstance(val, (int, float)):
                    series.setdefault(key, []).append(int(val))
        
        compressed = {}
        total_ops = 0
        total_original = 0
        total_compressed = 0
        
        for key, values in series.items():
            if len(values) < 3:
                compressed[key] = {"method": "raw", "data": values}
                continue
            
            result = self.hybrid_encode(values)
            compressed[key] = {
                "method": result.method,
                "data": result.data,
                "ratio": round(result.ratio, 3),
            }
            total_ops += result.neural_ops
            total_original += result.original_size
            total_compressed += result.compressed_size
        
        return {
            "fields": compressed,
            "records": len(metrics),
            "total_ratio": round(total_compressed / max(total_original, 1), 3),
            "neural_ops": total_ops,
        }


# ── CLI ──────────────────────────────────────────────────────

def demo():
    nc = NeuralCompressor()
    
    print("Neural Memory Compression")
    print("=" * 60)
    print("Every comparison and arithmetic op → trained neural network\n")
    
    # ── Demo 1: RLE on repeated values ──
    print("── RLE: Repeated sensor readings ──")
    data = [95, 95, 95, 95, 96, 96, 96, 97, 97, 97, 97, 97]
    result = nc.rle_encode(data)
    print(f"  Input:  {data} ({result.original_size} values)")
    print(f"  Output: {result.data} ({result.compressed_size} values)")
    print(f"  Ratio:  {result.ratio:.1%} | Ops: {result.neural_ops} | Time: {result.time_seconds:.3f}s")
    
    # Verify roundtrip
    decoded = nc.rle_decode(result.data)
    print(f"  Roundtrip: {'✅' if decoded == data else '❌'}")
    print()
    
    # ── Demo 2: Delta on slowly changing metrics ──
    print("── Delta: Disk usage over 10 readings ──")
    data = [62, 62, 63, 63, 63, 64, 64, 64, 64, 65]
    result = nc.delta_encode(data)
    print(f"  Input:  {data}")
    print(f"  Deltas: {result.data}")
    print(f"  Ratio:  {result.ratio:.1%} | Ops: {result.neural_ops}")
    
    decoded = nc.delta_decode(result.data)
    print(f"  Roundtrip: {'✅' if decoded == data else '❌'}")
    print()
    
    # ── Demo 3: Hybrid on real-world pattern ──
    print("── Hybrid: Memory readings (flat periods + jumps) ──")
    data = [512, 512, 512, 512, 510, 510, 510, 480, 480, 480, 480, 480, 490, 490, 512, 512, 512]
    result = nc.hybrid_encode(data)
    print(f"  Input:    {data} ({len(data)} values)")
    print(f"  Encoded:  {result.data} ({result.compressed_size} values)")
    print(f"  Ratio:    {result.ratio:.1%} | Ops: {result.neural_ops} | Time: {result.time_seconds:.3f}s")
    
    decoded = nc.hybrid_decode(result.data)
    print(f"  Roundtrip: {'✅' if decoded == data else '❌'}")
    print()
    
    # ── Demo 4: Compress actual outcome data ──
    print("── Metrics: Compressing outcome data ──")
    outcomes_path = Path("/Users/noc/clawd/data/ncpu-outcomes.jsonl")
    if outcomes_path.exists():
        outcomes = []
        with outcomes_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        o = json.loads(line)
                        if "input_data" in o:
                            outcomes.append(o["input_data"])
                    except json.JSONDecodeError:
                        continue
        
        if outcomes:
            result = nc.compress_metrics(outcomes[:50])  # First 50
            print(f"  Records:     {result['records']}")
            print(f"  Overall:     {result['total_ratio']:.1%} compression")
            print(f"  Neural ops:  {result['neural_ops']}")
            print(f"  Fields:")
            for key, info in result["fields"].items():
                if info["method"] != "raw":
                    print(f"    {key}: {info['method']} → {info['ratio']:.1%}")
        else:
            print("  (no numeric outcome data found)")
    else:
        print("  (outcomes file not found)")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    if cmd == "demo":
        demo()
    elif cmd == "rle" and len(sys.argv) > 2:
        nc = NeuralCompressor()
        values = [int(x) for x in sys.argv[2:]]
        result = nc.rle_encode(values)
        print(f"RLE: {result.data}")
        print(f"Ratio: {result.ratio:.1%} | Ops: {result.neural_ops}")
    elif cmd == "delta" and len(sys.argv) > 2:
        nc = NeuralCompressor()
        values = [int(x) for x in sys.argv[2:]]
        result = nc.delta_encode(values)
        print(f"Deltas: {result.data}")
        print(f"Ratio: {result.ratio:.1%} | Ops: {result.neural_ops}")
    else:
        print("Usage: python -m bridge.neural_compress [demo|rle <values>|delta <values>]")


if __name__ == "__main__":
    main()
