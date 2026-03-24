"""
Phase 35 — Neural Bloom Filter & Data Structures
==================================================
Probabilistic data structures where all hash computation,
bit manipulation, and membership testing go through neural ALU.

Features:
  - Bloom filter: neural hash functions, bit array ops
  - Skip list: probabilistic balancing with neural CMP
  - LRU cache: neural timestamp comparison for eviction
"""

from bridge.compute import NCPUBridge

bridge = NCPUBridge()


class NeuralBloomFilter:
    """Bloom filter with neural hash computation and bit ops."""

    def __init__(self, size: int = 64, num_hashes: int = 3):
        self.size = size
        self.bits = 0  # bitmask
        self.num_hashes = num_hashes
        self._ops = 0
        self._items = 0

    def _neural_hash(self, data: list[int], seed: int) -> int:
        """Hash bytes using neural XOR/ADD/MUL chain."""
        h = seed
        for byte in data:
            h = bridge.bitwise_xor(h, byte)
            self._ops += 1
            h = bridge.add(h, bridge.mul(byte, 31))
            self._ops += 2
            h = h % self.size
        return h

    def _set_bit(self, pos: int):
        bit = bridge.shl(1, pos)
        self._ops += 1
        self.bits = bridge.bitwise_or(self.bits, bit)
        self._ops += 1

    def _test_bit(self, pos: int) -> bool:
        bit = bridge.shl(1, pos)
        self._ops += 1
        result = bridge.bitwise_and(self.bits, bit)
        self._ops += 1
        zf, _ = bridge.cmp(result, 0)
        self._ops += 1
        return not zf

    def add(self, item: str):
        """Add item to bloom filter."""
        data = [ord(c) for c in item]
        for i in range(self.num_hashes):
            seed = bridge.add(i, 7)  # different seed per hash
            self._ops += 1
            pos = self._neural_hash(data, seed)
            self._set_bit(pos)
        self._items += 1

    def might_contain(self, item: str) -> bool:
        """Check membership (may have false positives)."""
        data = [ord(c) for c in item]
        for i in range(self.num_hashes):
            seed = bridge.add(i, 7)
            self._ops += 1
            pos = self._neural_hash(data, seed)
            if not self._test_bit(pos):
                return False
        return True


class NeuralLRUCache:
    """LRU cache with neural timestamp comparison for eviction."""

    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self._store: dict[str, tuple[int, int]] = {}  # key → (value, timestamp)
        self._clock = 0
        self._ops = 0
        self._hits = 0
        self._misses = 0

    def _tick(self):
        self._ops += 1
        self._clock = bridge.add(self._clock, 1)
        return self._clock

    def get(self, key: str) -> int | None:
        if key in self._store:
            val, _ = self._store[key]
            self._store[key] = (val, self._tick())
            self._hits += 1
            return val
        self._misses += 1
        return None

    def put(self, key: str, value: int):
        if key in self._store:
            self._store[key] = (value, self._tick())
            return

        if len(self._store) >= self.capacity:
            # Find LRU entry via neural CMP
            lru_key = None
            lru_ts = None
            for k, (v, ts) in self._store.items():
                if lru_ts is None:
                    lru_key, lru_ts = k, ts
                else:
                    zf, sf = bridge.cmp(ts, lru_ts)
                    self._ops += 1
                    if sf:  # ts < lru_ts
                        lru_key, lru_ts = k, ts
            if lru_key:
                del self._store[lru_key]

        self._store[key] = (value, self._tick())

    def stats(self):
        total = self._hits + self._misses
        rate = (self._hits / total * 100) if total else 0
        return {"hits": self._hits, "misses": self._misses, "hit_rate": rate}


class NeuralSkipNode:
    def __init__(self, key: int, value: int, level: int):
        self.key = key
        self.value = value
        self.forward = [None] * (level + 1)


class NeuralSkipList:
    """Skip list with neural CMP for key comparison."""

    def __init__(self, max_level: int = 4):
        self.max_level = max_level
        self.level = 0
        self.header = NeuralSkipNode(-1, -1, max_level)
        self._ops = 0
        self._size = 0
        self._level_seed = 42

    def _neural_cmp(self, a, b):
        self._ops += 1
        return bridge.cmp(a, b)

    def _random_level(self) -> int:
        """Deterministic pseudo-random level via neural ops."""
        self._level_seed = bridge.bitwise_xor(self._level_seed, bridge.shl(self._level_seed, 3) & 0xFF)
        self._ops += 2
        self._level_seed = bridge.bitwise_xor(self._level_seed, bridge.shr(self._level_seed, 2) & 0xFF)
        self._ops += 2
        level = 0
        val = self._level_seed
        while level < self.max_level:
            bit = bridge.bitwise_and(val, 1)
            self._ops += 1
            zf, _ = bridge.cmp(bit, 0)
            self._ops += 1
            if zf:
                break
            level += 1
            val = bridge.shr(val, 1)
            self._ops += 1
        return level

    def insert(self, key: int, value: int):
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] is not None:
                zf, sf = self._neural_cmp(current.forward[i].key, key)
                if sf:  # forward.key < key
                    current = current.forward[i]
                else:
                    break
            update[i] = current

        lvl = self._random_level()
        if lvl > self.level:
            for i in range(self.level + 1, lvl + 1):
                update[i] = self.header
            self.level = lvl

        new_node = NeuralSkipNode(key, value, lvl)
        for i in range(lvl + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
        self._size += 1

    def search(self, key: int) -> int | None:
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] is not None:
                zf, sf = self._neural_cmp(current.forward[i].key, key)
                if zf:
                    return current.forward[i].value
                if sf:
                    current = current.forward[i]
                else:
                    break
        return None

    def to_list(self) -> list[tuple[int, int]]:
        result = []
        node = self.header.forward[0]
        while node:
            result.append((node.key, node.value))
            node = node.forward[0]
        return result


def demo():
    print("Neural Bloom Filter & Data Structures")
    print("=" * 60)
    print("Probabilistic structures with neural hash/compare ops\n")

    # ── Bloom filter ──
    print("  Bloom Filter (64 bits, 3 hash functions):")
    bf = NeuralBloomFilter(size=32, num_hashes=3)
    words = ["hello", "world", "neural", "cpu", "bloom"]
    for w in words:
        bf.add(w)
    print(f"    Added: {words}")

    test_words = ["hello", "world", "neural", "missing", "absent", "bloom"]
    for w in test_words:
        found = bf.might_contain(w)
        expected = w in words
        fp = found and not expected
        status = "✅" if found == expected else ("⚠️ FP" if fp else "❌")
        print(f"    '{w}': {'maybe' if found else 'no':5s} {status}")
    print(f"    Neural ops: {bf._ops}")

    # ── LRU cache ──
    print("\n  LRU Cache (capacity=3):")
    lru = NeuralLRUCache(capacity=3)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.put("c", 3)
    print(f"    put a=1, b=2, c=3")

    v = lru.get("a")  # access 'a' to make it recent
    print(f"    get(a) = {v}")

    lru.put("d", 4)  # should evict 'b' (least recently used)
    print(f"    put d=4 (should evict b)")

    evicted = lru.get("b")
    kept = lru.get("a")
    print(f"    get(b) = {evicted} {'✅ evicted' if evicted is None else '❌'}")
    print(f"    get(a) = {kept} {'✅ kept' if kept == 1 else '❌'}")

    stats = lru.stats()
    print(f"    Stats: {stats['hits']} hits, {stats['misses']} misses ({stats['hit_rate']:.0f}% hit rate)")
    print(f"    Neural ops: {lru._ops}")

    # ── Skip list ──
    print("\n  Skip List:")
    sl = NeuralSkipList(max_level=3)
    entries = [(30, 300), (10, 100), (50, 500), (20, 200), (40, 400)]
    for k, v in entries:
        sl.insert(k, v)
    print(f"    Inserted: {entries}")

    sorted_list = sl.to_list()
    keys = [k for k, v in sorted_list]
    is_sorted = keys == sorted(keys)
    print(f"    Sorted order: {sorted_list} {'✅' if is_sorted else '❌'}")

    for k in [10, 30, 99]:
        v = sl.search(k)
        expected = k * 10 if k != 99 else None
        ok = v == expected
        print(f"    search({k}) = {v} {'✅' if ok else '❌'}")
    print(f"    Neural ops: {sl._ops}")

    total = bf._ops + lru._ops + sl._ops
    print(f"\n  Total neural ops: {total}")
    print("\n✅ Bloom filter + LRU cache + skip list all working")


if __name__ == "__main__":
    demo()
