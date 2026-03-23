"""Neural Database — indexed storage with query engine through nCPU.

A key-value store with:
- B-tree-like sorted index (neural CMP for all comparisons)
- Sequential scan with neural filtering
- Aggregations: COUNT, SUM, AVG, MIN, MAX (all neural arithmetic)
- INSERT, SELECT, DELETE operations
- Index lookup in O(log n) via neural binary search

Usage:
    python -m bridge.neural_db demo
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


@dataclass
class Row:
    id: int
    data: dict[str, int]  # column → value (ints only for neural ops)


@dataclass 
class QueryResult:
    rows: list[Row]
    count: int
    neural_ops: int
    scan_type: str  # "index" or "full_scan"


class NeuralIndex:
    """Sorted index with neural binary search.
    
    Maintains a sorted array of (key, row_id) pairs.
    All comparisons use neural CMP.
    """
    
    def __init__(self, bridge: NCPUBridge, column: str):
        self.bridge = bridge
        self.column = column
        self._entries: list[tuple[int, int]] = []  # (key_value, row_id)
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def insert(self, key: int, row_id: int):
        """Insert into sorted index using neural binary search for position."""
        pos = self._find_insert_pos(key)
        self._entries.insert(pos, (key, row_id))
    
    def _find_insert_pos(self, key: int) -> int:
        """Binary search for insertion point — all comparisons neural."""
        lo, hi = 0, len(self._entries)
        
        while lo < hi:
            mid = self.bridge.div(self.bridge.add(lo, hi), 2)
            self._op()
            
            zf, sf = self.bridge.cmp(self._entries[mid][0], key)
            self._op()
            
            if sf:  # entries[mid] < key
                lo = self.bridge.add(mid, 1)
                self._op()
            else:
                hi = mid
        
        return lo
    
    def lookup(self, key: int) -> list[int]:
        """Find all row_ids matching key — neural binary search."""
        pos = self._find_insert_pos(key)
        results = []
        
        # Scan forward for all matches
        while pos < len(self._entries):
            zf, _ = self.bridge.cmp(self._entries[pos][0], key)
            self._op()
            if zf:
                results.append(self._entries[pos][1])
                pos += 1
            else:
                break
        
        return results
    
    def range_scan(self, lo: int, hi: int) -> list[int]:
        """Find all row_ids with key in [lo, hi] — neural range scan."""
        pos = self._find_insert_pos(lo)
        results = []
        
        while pos < len(self._entries):
            key = self._entries[pos][0]
            
            # Neural: key <= hi?
            zf, sf = self.bridge.cmp(key, hi)
            self._op()
            if not sf or zf:  # key <= hi
                results.append(self._entries[pos][1])
                pos += 1
            else:
                break
        
        return results
    
    def delete(self, key: int, row_id: int):
        """Remove entry from index."""
        self._entries = [(k, r) for k, r in self._entries if not (k == key and r == row_id)]


class NeuralDB:
    """Database with neural query execution.
    
    Every comparison, aggregation, and index operation
    goes through trained neural networks.
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._rows: dict[int, Row] = {}
        self._next_id = 1
        self._indexes: dict[str, NeuralIndex] = {}
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def create_index(self, column: str):
        """Create an index on a column."""
        idx = NeuralIndex(self.bridge, column)
        
        # Index existing rows
        for row in self._rows.values():
            if column in row.data:
                idx.insert(row.data[column], row.id)
        
        self._indexes[column] = idx
    
    def insert(self, **data: int) -> int:
        """Insert a row. Returns row ID."""
        row_id = self._next_id
        self._next_id = self.bridge.add(self._next_id, 1)
        self._op()
        
        row = Row(id=row_id, data=data)
        self._rows[row_id] = row
        
        # Update indexes
        for col, idx in self._indexes.items():
            if col in data:
                idx.insert(data[col], row_id)
        
        return row_id
    
    def select(self, where: Optional[dict] = None, order_by: Optional[str] = None,
               limit: Optional[int] = None) -> QueryResult:
        """Select rows with optional filtering, ordering, and limit.
        
        WHERE: {column: value} or {column: (op, value)} where op is 'eq','gt','lt','gte','lte'
        """
        self._ops = 0
        
        # Try index lookup
        scan_type = "full_scan"
        if where and len(where) == 1:
            col, val = list(where.items())[0]
            if col in self._indexes:
                if isinstance(val, int):
                    row_ids = self._indexes[col].lookup(val)
                    scan_type = "index"
                elif isinstance(val, tuple) and len(val) == 2:
                    op, v = val
                    if op in ('gte', 'lte'):
                        lo = v if op == 'gte' else 0
                        hi = v if op == 'lte' else 999999
                        row_ids = self._indexes[col].range_scan(lo, hi)
                        scan_type = "index_range"
                    else:
                        row_ids = list(self._rows.keys())
                else:
                    row_ids = list(self._rows.keys())
            else:
                row_ids = list(self._rows.keys())
        else:
            row_ids = list(self._rows.keys())
        
        # Filter
        results = []
        for rid in row_ids:
            row = self._rows.get(rid)
            if not row:
                continue
            
            if where and scan_type == "full_scan":
                match = True
                for col, val in where.items():
                    if col not in row.data:
                        match = False
                        break
                    
                    if isinstance(val, int):
                        zf, _ = self.bridge.cmp(row.data[col], val)
                        self._op()
                        if not zf:
                            match = False
                            break
                    elif isinstance(val, tuple):
                        op, v = val
                        actual = row.data[col]
                        zf, sf = self.bridge.cmp(actual, v)
                        self._op()
                        
                        if op == 'eq' and not zf:
                            match = False; break
                        elif op == 'gt' and (sf or zf):
                            match = False; break
                        elif op == 'lt' and not sf:
                            match = False; break
                        elif op == 'gte' and sf and not zf:
                            match = False; break
                        elif op == 'lte' and not sf and not zf:
                            match = False; break
                
                if not match:
                    continue
            
            results.append(row)
        
        # Order by (neural insertion sort)
        if order_by and results:
            for i in range(1, len(results)):
                key = results[i]
                j = i - 1
                while j >= 0:
                    a_val = results[j].data.get(order_by, 0)
                    b_val = key.data.get(order_by, 0)
                    zf, sf = self.bridge.cmp(a_val, b_val)
                    self._op()
                    if not sf and not zf:  # a > b
                        results[j + 1] = results[j]
                        j -= 1
                    else:
                        break
                results[j + 1] = key
        
        # Limit
        if limit:
            results = results[:limit]
        
        idx_ops = sum(idx._ops for idx in self._indexes.values())
        return QueryResult(
            rows=results,
            count=len(results),
            neural_ops=self._ops + idx_ops,
            scan_type=scan_type,
        )
    
    def aggregate(self, column: str, func: str, where: Optional[dict] = None) -> dict:
        """Aggregate: COUNT, SUM, AVG, MIN, MAX — all neural arithmetic."""
        self._ops = 0
        
        qr = self.select(where=where)
        values = [r.data.get(column, 0) for r in qr.rows if column in r.data]
        
        if not values:
            return {"result": 0, "func": func, "neural_ops": self._ops}
        
        if func == "COUNT":
            return {"result": len(values), "func": func, "neural_ops": self._ops}
        
        elif func == "SUM":
            total = values[0]
            for v in values[1:]:
                total = self.bridge.add(total, v)
                self._op()
            return {"result": total, "func": func, "neural_ops": self._ops}
        
        elif func == "AVG":
            total = values[0]
            for v in values[1:]:
                total = self.bridge.add(total, v)
                self._op()
            avg = self.bridge.div(total, len(values))
            self._op()
            return {"result": avg, "func": func, "neural_ops": self._ops}
        
        elif func == "MIN":
            result = values[0]
            for v in values[1:]:
                zf, sf = self.bridge.cmp(v, result)
                self._op()
                if sf:  # v < result
                    result = v
            return {"result": result, "func": func, "neural_ops": self._ops}
        
        elif func == "MAX":
            result = values[0]
            for v in values[1:]:
                zf, sf = self.bridge.cmp(v, result)
                self._op()
                if not sf and not zf:  # v > result
                    result = v
            return {"result": result, "func": func, "neural_ops": self._ops}
        
        return {"error": f"Unknown function: {func}"}
    
    def delete(self, where: dict) -> int:
        """Delete rows matching condition. Returns count deleted."""
        qr = self.select(where=where)
        count = 0
        for row in qr.rows:
            # Remove from indexes
            for col, idx in self._indexes.items():
                if col in row.data:
                    idx.delete(row.data[col], row.id)
            del self._rows[row.id]
            count += 1
        return count


# ── CLI ──

def demo():
    db = NeuralDB()
    
    print("Neural Database Engine")
    print("=" * 60)
    print("Every comparison, aggregation, and index lookup → neural ALU\n")
    
    # Create table with index
    print("── Schema ──")
    db.create_index("port")
    db.create_index("status")
    print("  Table: services (name, port, status, latency_ms)")
    print("  Indexes: port, status")
    print()
    
    # Insert data
    print("── INSERT ──")
    services = [
        {"name": ord('P'), "port": 3000, "status": 1, "latency_ms": 12},   # POS
        {"name": ord('S'), "port": 3890, "status": 1, "latency_ms": 5},    # SentryFlow
        {"name": ord('N'), "port": 1984, "status": 1, "latency_ms": 3},    # NVR/go2rtc
        {"name": ord('A'), "port": 5773, "status": 0, "latency_ms": 0},    # AirWave (down)
        {"name": ord('D'), "port": 3700, "status": 1, "latency_ms": 8},    # Docs
        {"name": ord('K'), "port": 3750, "status": 1, "latency_ms": 15},   # Kiosk
        {"name": ord('R'), "port": 3500, "status": 0, "latency_ms": 0},    # Remotion (down)
        {"name": ord('W'), "port": 3710, "status": 1, "latency_ms": 22},   # Workers
    ]
    
    for s in services:
        rid = db.insert(**s)
        name = chr(s["name"])
        status = "UP" if s["status"] else "DOWN"
        print(f"  id={rid}: {name} port={s['port']} {status} {s['latency_ms']}ms")
    print()
    
    # SELECT with index
    print("── SELECT (index lookup) ──")
    qr = db.select(where={"port": 3890})
    for r in qr.rows:
        print(f"  port=3890 → id={r.id} name={chr(r.data['name'])} ({qr.scan_type}, {qr.neural_ops} ops)")
    print()
    
    # SELECT with filter
    print("── SELECT (full scan with filter) ──")
    qr = db.select(where={"status": 1}, order_by="latency_ms")
    print(f"  WHERE status=1 ORDER BY latency_ms ({qr.scan_type}, {qr.neural_ops} ops):")
    for r in qr.rows:
        print(f"    {chr(r.data['name'])} port={r.data['port']} latency={r.data['latency_ms']}ms")
    print()
    
    # SELECT with comparison
    print("── SELECT (comparison) ──")
    qr = db.select(where={"latency_ms": ("gt", 10)})
    print(f"  WHERE latency_ms > 10 ({qr.count} rows, {qr.neural_ops} ops):")
    for r in qr.rows:
        print(f"    {chr(r.data['name'])} latency={r.data['latency_ms']}ms")
    print()
    
    # Aggregations
    print("── AGGREGATIONS ──")
    for func in ["COUNT", "SUM", "AVG", "MIN", "MAX"]:
        result = db.aggregate("latency_ms", func, where={"status": 1})
        print(f"  {func}(latency_ms) WHERE status=1 → {result['result']} ({result['neural_ops']} ops)")
    print()
    
    # Delete
    print("── DELETE ──")
    deleted = db.delete(where={"status": 0})
    print(f"  DELETE WHERE status=0 → {deleted} rows removed")
    
    qr = db.select()
    print(f"  Remaining rows: {qr.count}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_db [demo]")


if __name__ == "__main__":
    main()
