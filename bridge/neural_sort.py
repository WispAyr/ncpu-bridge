"""
Phase 29 — Neural Sorting Algorithms
=====================================
Multiple sorting algorithms where every comparison and swap
goes through the neural ALU.

Algorithms:
  - Bubble sort
  - Selection sort  
  - Insertion sort (already in scheduler, standalone here)
  - Merge sort (neural merge step)
  - Quicksort (neural partition)
  - Counting sort (neural increment)

All comparisons via bridge.cmp, all arithmetic via bridge.add/sub.
"""

from bridge.compute import NCPUBridge
import time

bridge = NCPUBridge()


class NeuralSort:
    """Sorting with every comparison through neural CMP."""

    def __init__(self):
        self._ops = 0
        self._comparisons = 0
        self._swaps = 0

    def _cmp_lt(self, a: int, b: int) -> bool:
        """a < b via neural CMP."""
        self._ops += 1
        self._comparisons += 1
        zf, sf = bridge.cmp(a, b)
        return sf and not zf

    def _cmp_le(self, a: int, b: int) -> bool:
        self._ops += 1
        self._comparisons += 1
        zf, sf = bridge.cmp(a, b)
        return sf or zf

    def _cmp_gt(self, a: int, b: int) -> bool:
        self._ops += 1
        self._comparisons += 1
        zf, sf = bridge.cmp(a, b)
        return not sf and not zf

    def _neural_add(self, a, b):
        self._ops += 1
        return bridge.add(a, b)

    def _neural_sub(self, a, b):
        self._ops += 1
        return bridge.sub(a, b)

    def _reset_stats(self):
        self._ops = 0
        self._comparisons = 0
        self._swaps = 0

    def bubble_sort(self, arr: list) -> list:
        """Neural bubble sort."""
        self._reset_stats()
        a = list(arr)
        n = len(a)
        for i in range(n):
            for j in range(n - 1 - i):
                if self._cmp_gt(a[j], a[j + 1]):
                    a[j], a[j + 1] = a[j + 1], a[j]
                    self._swaps += 1
        return a

    def selection_sort(self, arr: list) -> list:
        """Neural selection sort."""
        self._reset_stats()
        a = list(arr)
        n = len(a)
        for i in range(n - 1):
            min_idx = i
            for j in range(i + 1, n):
                if self._cmp_lt(a[j], a[min_idx]):
                    min_idx = j
            if min_idx != i:
                a[i], a[min_idx] = a[min_idx], a[i]
                self._swaps += 1
        return a

    def insertion_sort(self, arr: list) -> list:
        """Neural insertion sort."""
        self._reset_stats()
        a = list(arr)
        for i in range(1, len(a)):
            key = a[i]
            j = i - 1
            while j >= 0 and self._cmp_gt(a[j], key):
                a[j + 1] = a[j]
                self._swaps += 1
                j -= 1
            a[j + 1] = key
        return a

    def merge_sort(self, arr: list) -> list:
        """Neural merge sort."""
        self._reset_stats()
        return self._merge_sort_impl(list(arr))

    def _merge_sort_impl(self, a: list) -> list:
        if len(a) <= 1:
            return a
        mid = len(a) // 2
        left = self._merge_sort_impl(a[:mid])
        right = self._merge_sort_impl(a[mid:])
        return self._merge(left, right)

    def _merge(self, left: list, right: list) -> list:
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if self._cmp_le(left[i], right[j]):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def quicksort(self, arr: list) -> list:
        """Neural quicksort."""
        self._reset_stats()
        a = list(arr)
        self._quicksort_impl(a, 0, len(a) - 1)
        return a

    def _quicksort_impl(self, a: list, lo: int, hi: int):
        if lo >= hi:
            return
        pivot = a[hi]
        i = lo
        for j in range(lo, hi):
            if self._cmp_le(a[j], pivot):
                a[i], a[j] = a[j], a[i]
                self._swaps += 1
                i += 1
        a[i], a[hi] = a[hi], a[i]
        self._swaps += 1
        self._quicksort_impl(a, lo, i - 1)
        self._quicksort_impl(a, i + 1, hi)

    def counting_sort(self, arr: list, max_val: int = 255) -> list:
        """Neural counting sort — uses neural ADD for counting."""
        self._reset_stats()
        counts = [0] * (max_val + 1)
        for v in arr:
            counts[v] = self._neural_add(counts[v], 1)
        result = []
        for val in range(max_val + 1):
            zf, _ = bridge.cmp(counts[val], 0)
            self._ops += 1
            if not zf:
                for _ in range(counts[val]):
                    result.append(val)
        return result


def demo():
    print("Neural Sorting Algorithms")
    print("=" * 60)
    print("Every comparison goes through the neural ALU\n")

    sorter = NeuralSort()
    test_data = [42, 17, 93, 5, 28, 71, 3, 56, 88, 12]
    expected = sorted(test_data)

    algorithms = [
        ("Bubble Sort",    sorter.bubble_sort),
        ("Selection Sort", sorter.selection_sort),
        ("Insertion Sort", sorter.insertion_sort),
        ("Merge Sort",     sorter.merge_sort),
        ("Quicksort",      sorter.quicksort),
        ("Counting Sort",  lambda a: sorter.counting_sort(a, max_val=100)),
    ]

    results = []
    for name, fn in algorithms:
        t0 = time.time()
        result = fn(test_data)
        elapsed = time.time() - t0
        correct = result == expected
        stats = (sorter._comparisons, sorter._swaps, sorter._ops)
        results.append((name, correct, elapsed, stats))
        status = "✅" if correct else "❌"
        print(f"  {status} {name:20s} {elapsed*1000:6.0f}ms  "
              f"cmps={stats[0]:3d} swaps={stats[1]:3d} ops={stats[2]:3d}  "
              f"→ {result[:5]}...")

    print(f"\n  Input:    {test_data}")
    print(f"  Expected: {expected}")

    # Comparison table
    print("\n  ┌──────────────────────┬────────┬──────┬───────┬──────┐")
    print("  │ Algorithm            │  Time  │ Cmps │ Swaps │  Ops │")
    print("  ├──────────────────────┼────────┼──────┼───────┼──────┤")
    for name, correct, elapsed, (cmps, swaps, ops) in results:
        c = "✅" if correct else "❌"
        print(f"  │ {c} {name:18s} │ {elapsed*1000:5.0f}ms │ {cmps:4d} │ {swaps:5d} │ {ops:4d} │")
    print("  └──────────────────────┴────────┴──────┴───────┴──────┘")

    all_correct = all(r[1] for r in results)
    print(f"\n{'✅' if all_correct else '❌'} {sum(1 for r in results if r[1])}/{len(results)} algorithms correct")


if __name__ == "__main__":
    demo()
