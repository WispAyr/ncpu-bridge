"""
Phase 41 — Real Neural Cache (LSTM Replacement Policy)
========================================================
Uses nCPU's CacheReplacementNet — an LSTM that observes access patterns
and scores each cache line for eviction.

Architecture:
  Access history → LSTM(4→64) → hidden state
  Per-line features: [recency, frequency, dirty, tag]
  Combined: [hidden; line_feat] → Linear(68,64) → ReLU → Linear(64,1)
  Output: eviction score per line (highest = evict)
"""

import sys
import torch
import random
import math
from pathlib import Path

from bridge.config import get_ncpu_path, get_bridge_path, get_clawd_data_path
NCPU_PATH = get_ncpu_path()
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from ncpu.os.neuros.cache import CacheReplacementNet
from bridge.compute import NCPUBridge

bridge = NCPUBridge()


class RealNeuralCache:
    """Cache with LSTM-learned replacement policy from nCPU."""

    def __init__(self, num_lines: int = 8):
        self.num_lines = num_lines
        self._store: dict[int, int] = {}
        self._access_history: list[list[float]] = []  # recent accesses as features
        self._freq: dict[int, int] = {}
        self._last_access: dict[int, int] = {}
        self._dirty: set[int] = set()
        self._tick = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        self.net = CacheReplacementNet(
            access_feature_dim=4, hidden_dim=64, line_feature_dim=4
        )

        model_path = NCPU_PATH / "models" / "os" / "cache_replace.pt"
        if model_path.exists():
            self.net.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            self._pretrained = True
        else:
            self._pretrained = False
        self.net.eval()

        self._params = sum(p.numel() for p in self.net.parameters())

    def _record_access(self, tag: int):
        """Record access for LSTM history."""
        feat = [
            tag / 1000.0,
            math.log1p(self._freq.get(tag, 0)),
            1.0 if tag in self._dirty else 0.0,
            self._tick / 100.0,
        ]
        self._access_history.append(feat)
        if len(self._access_history) > 64:
            self._access_history = self._access_history[-64:]

    def _line_features(self) -> torch.Tensor:
        """Build [num_lines, 4] feature matrix for current cache lines."""
        tags = list(self._store.keys())
        features = torch.zeros(len(tags), 4)
        for i, tag in enumerate(tags):
            recency = (self._tick - self._last_access.get(tag, 0)) / 100.0
            freq = math.log1p(self._freq.get(tag, 0))
            dirty = 1.0 if tag in self._dirty else 0.0
            features[i] = torch.tensor([recency, freq, dirty, tag / 1000.0])
        return features

    def access(self, tag: int, write: bool = False) -> tuple[int | None, bool]:
        self._tick += 1

        if tag in self._store:
            self._hits += 1
            self._freq[tag] = self._freq.get(tag, 0) + 1
            self._last_access[tag] = self._tick
            self._record_access(tag)
            if write:
                self._dirty.add(tag)
            return self._store[tag], True

        self._misses += 1

        if len(self._store) >= self.num_lines:
            self._evict_neural()

        value = bridge.mul(tag, 7)
        self._store[tag] = value
        self._freq[tag] = 1
        self._last_access[tag] = self._tick
        self._record_access(tag)
        if write:
            self._dirty.add(tag)
        return value, False

    def _evict_neural(self):
        """Use LSTM to choose eviction victim."""
        if not self._access_history or not self._store:
            # Fallback: evict oldest
            oldest = min(self._store, key=lambda t: self._last_access.get(t, 0))
            self._remove(oldest)
            return

        # Build history tensor [1, seq_len, 4]
        history = torch.tensor(self._access_history[-32:]).unsqueeze(0)
        line_feats = self._line_features()

        with torch.no_grad():
            scores = self.net(history, line_feats)

        # Evict highest-scored line
        evict_idx = scores.argmax().item()
        tags = list(self._store.keys())
        if evict_idx < len(tags):
            self._remove(tags[evict_idx])
        else:
            self._remove(tags[0])

    def _remove(self, tag: int):
        del self._store[tag]
        self._freq.pop(tag, None)
        self._last_access.pop(tag, None)
        self._dirty.discard(tag)
        self._evictions += 1

    def stats(self):
        total = self._hits + self._misses
        return {
            "hits": self._hits, "misses": self._misses,
            "hit_rate": self._hits / max(total, 1) * 100,
            "evictions": self._evictions,
            "size": len(self._store),
        }


def demo():
    print("Real Neural Cache (LSTM Replacement Policy)")
    print("=" * 60)
    print(f"Architecture: LSTM(4→64) + scorer MLP(68→64→1)")

    cache = RealNeuralCache(num_lines=8)
    print(f"Parameters: {cache._params:,}")
    print(f"Pretrained: {cache._pretrained}")
    print(f"Capacity: {cache.num_lines} lines\n")

    # Workload 1: Sequential
    print("  Workload 1: Sequential scan (0-15)")
    for tag in range(16):
        cache.access(tag)
    s1 = cache.stats()
    print(f"    Hit rate: {s1['hit_rate']:.1f}%  Evictions: {s1['evictions']}")

    # Workload 2: Hot set + cold
    print("\n  Workload 2: Hot set (0-3) + cold scan")
    c2 = RealNeuralCache(num_lines=8)
    for _ in range(5):
        for tag in [0, 1, 2, 3]:
            c2.access(tag)
    for tag in [0, 1, 20, 2, 21, 3, 22, 0, 23, 1, 24, 2, 25, 3]:
        c2.access(tag)
    s2 = c2.stats()
    hot = sum(1 for t in [0,1,2,3] if t in c2._store)
    print(f"    Hit rate: {s2['hit_rate']:.1f}%  Evictions: {s2['evictions']}  Hot retained: {hot}/4")

    # Workload 3: Zipf
    print("\n  Workload 3: Zipf distribution")
    c3 = RealNeuralCache(num_lines=8)
    random.seed(42)
    zipf = [0]*20 + [1]*10 + [2]*5 + list(range(3,11))
    random.shuffle(zipf)
    for tag in zipf:
        c3.access(tag)
    s3 = c3.stats()
    print(f"    Hit rate: {s3['hit_rate']:.1f}%  Evictions: {s3['evictions']}")

    # Workload 4: Looping (temporal locality)
    print("\n  Workload 4: Loop (0-5 repeated 10x)")
    c4 = RealNeuralCache(num_lines=8)
    for _ in range(10):
        for tag in range(6):
            c4.access(tag)
    s4 = c4.stats()
    print(f"    Hit rate: {s4['hit_rate']:.1f}%  Evictions: {s4['evictions']}")

    print("\n  ┌──────────────────┬──────────┬───────────┐")
    print("  │ Workload         │ Hit Rate │ Evictions │")
    print("  ├──────────────────┼──────────┼───────────┤")
    for name, s in [("Sequential", s1), ("Hot+Cold", s2), ("Zipf", s3), ("Loop", s4)]:
        print(f"  │ {name:16s} │ {s['hit_rate']:6.1f}%  │ {s['evictions']:9d} │")
    print("  └──────────────────┴──────────┴───────────┘")

    print(f"\n  LSTM params: {cache._params:,}")
    print(f"\n✅ Real Neural Cache: LSTM-learned replacement across 4 workloads")


if __name__ == "__main__":
    demo()
