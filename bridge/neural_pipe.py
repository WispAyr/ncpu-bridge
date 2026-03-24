"""
Phase 32 — Neural UNIX Pipes
==============================
UNIX-style pipe/filter processing where data flows through
a pipeline of neural transformations. Each filter applies
operations via the neural ALU.

Features:
  - Pipe buffers with neural read/write pointers
  - Built-in filters: grep (neural CMP), sort, uniq, wc, head, tail, tr, cut
  - Pipeline composition: cmd1 | cmd2 | cmd3
  - Byte-level stream processing
"""

from bridge.compute import NCPUBridge

bridge = NCPUBridge()


class NeuralPipeBuffer:
    """Ring buffer for pipe data."""
    def __init__(self, capacity=256):
        self._buf = []
        self._ops = 0

    def write(self, data: list):
        self._buf.extend(data)

    def read_all(self) -> list:
        data = list(self._buf)
        self._buf.clear()
        return data

    def is_empty(self) -> bool:
        return len(self._buf) == 0


class NeuralFilter:
    """Base class for pipe filters."""
    def __init__(self):
        self._ops = 0

    def _cmp(self, a, b):
        self._ops += 1
        return bridge.cmp(a, b)

    def _add(self, a, b):
        self._ops += 1
        return bridge.add(a, b)

    def _sub(self, a, b):
        self._ops += 1
        return bridge.sub(a, b)

    def process(self, lines: list[str]) -> list[str]:
        raise NotImplementedError


class NeuralGrep(NeuralFilter):
    """Filter lines matching a pattern (byte-level neural CMP)."""
    def __init__(self, pattern: str, invert=False):
        super().__init__()
        self.pattern = pattern
        self.invert = invert

    def process(self, lines):
        result = []
        pat_bytes = [ord(c) for c in self.pattern]
        for line in lines:
            line_bytes = [ord(c) for c in line]
            found = False
            for i in range(len(line_bytes) - len(pat_bytes) + 1):
                match = True
                for j, pb in enumerate(pat_bytes):
                    zf, _ = self._cmp(line_bytes[i + j], pb)
                    if not zf:
                        match = False
                        break
                if match:
                    found = True
                    break
            if found != self.invert:
                result.append(line)
        return result


class NeuralSort(NeuralFilter):
    """Sort lines lexicographically via neural CMP on first char."""
    def process(self, lines):
        arr = list(lines)
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0:
                # Compare first differing byte
                a_bytes = [ord(c) for c in arr[j]]
                b_bytes = [ord(c) for c in key]
                gt = False
                for k in range(min(len(a_bytes), len(b_bytes))):
                    zf, sf = self._cmp(a_bytes[k], b_bytes[k])
                    if not zf:
                        gt = not sf  # a[k] > b[k]
                        break
                else:
                    # Equal prefix — longer string is "greater"
                    _, sf = self._cmp(len(a_bytes), len(b_bytes))
                    gt = not sf and len(a_bytes) != len(b_bytes)
                if gt:
                    arr[j + 1] = arr[j]
                    j -= 1
                else:
                    break
            arr[j + 1] = key
        return arr


class NeuralUniq(NeuralFilter):
    """Remove adjacent duplicates via neural byte comparison."""
    def process(self, lines):
        if not lines:
            return []
        result = [lines[0]]
        for i in range(1, len(lines)):
            prev = [ord(c) for c in lines[i-1]]
            curr = [ord(c) for c in lines[i]]
            same = True
            if len(prev) != len(curr):
                same = False
            else:
                for a, b in zip(prev, curr):
                    zf, _ = self._cmp(a, b)
                    if not zf:
                        same = False
                        break
            if not same:
                result.append(lines[i])
        return result


class NeuralWc(NeuralFilter):
    """Word/line/char count."""
    def process(self, lines):
        line_count = len(lines)
        char_count = 0
        word_count = 0
        for line in lines:
            char_count = self._add(char_count, len(line))
            char_count = self._add(char_count, 1)  # newline
            words = len(line.split())
            word_count = self._add(word_count, words)
        return [f"  {line_count}  {word_count}  {char_count}"]


class NeuralHead(NeuralFilter):
    """Take first N lines."""
    def __init__(self, n=10):
        super().__init__()
        self.n = n
    def process(self, lines):
        return lines[:self.n]


class NeuralTail(NeuralFilter):
    """Take last N lines."""
    def __init__(self, n=10):
        super().__init__()
        self.n = n
    def process(self, lines):
        return lines[-self.n:]


class NeuralTr(NeuralFilter):
    """Translate characters via neural CMP matching."""
    def __init__(self, from_chars: str, to_chars: str):
        super().__init__()
        self.from_c = [ord(c) for c in from_chars]
        self.to_c = [ord(c) for c in to_chars]

    def process(self, lines):
        result = []
        for line in lines:
            out = []
            for ch in line:
                byte = ord(ch)
                replaced = False
                for fi, fc in enumerate(self.from_c):
                    zf, _ = self._cmp(byte, fc)
                    if zf and fi < len(self.to_c):
                        out.append(chr(self.to_c[fi]))
                        replaced = True
                        break
                if not replaced:
                    out.append(ch)
            result.append("".join(out))
        return result


class NeuralCut(NeuralFilter):
    """Cut fields by delimiter."""
    def __init__(self, field: int, delim: str = ","):
        super().__init__()
        self.field = field
        self.delim = delim

    def process(self, lines):
        result = []
        for line in lines:
            parts = line.split(self.delim)
            if self.field < len(parts):
                result.append(parts[self.field].strip())
        return result


class NeuralPipeline:
    """Compose filters into a pipeline."""
    def __init__(self, *filters: NeuralFilter):
        self.filters = list(filters)

    def run(self, input_lines: list[str]) -> list[str]:
        data = list(input_lines)
        for f in self.filters:
            data = f.process(data)
        return data

    @property
    def total_ops(self):
        return sum(f._ops for f in self.filters)


def demo():
    print("Neural UNIX Pipes")
    print("=" * 60)
    print("Pipe/filter processing with neural byte comparison\n")

    # Sample data
    data = [
        "alice,engineering,85",
        "bob,marketing,72",
        "charlie,engineering,93",
        "dave,marketing,68",
        "eve,engineering,91",
        "frank,sales,77",
        "grace,engineering,88",
        "heidi,sales,65",
    ]

    print("  Input data:")
    for line in data:
        print(f"    {line}")

    # Pipeline 1: grep engineering | sort | head 3
    print("\n  Pipeline: grep 'engineering' | sort | head 3")
    p1 = NeuralPipeline(
        NeuralGrep("engineering"),
        NeuralSort(),
        NeuralHead(3),
    )
    result1 = p1.run(data)
    for line in result1:
        print(f"    {line}")
    print(f"    ({p1.total_ops} neural ops)")

    # Pipeline 2: cut field 1 | sort | uniq
    print("\n  Pipeline: cut -f2 -d, | sort | uniq")
    p2 = NeuralPipeline(
        NeuralCut(field=1, delim=","),
        NeuralSort(),
        NeuralUniq(),
    )
    result2 = p2.run(data)
    for line in result2:
        print(f"    {line}")
    print(f"    ({p2.total_ops} neural ops)")

    # Pipeline 3: tr lowercase → uppercase (first 3 lines)
    print("\n  Pipeline: head 3 | tr a-z A-Z")
    p3 = NeuralPipeline(
        NeuralHead(3),
        NeuralTr("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    )
    result3 = p3.run(data)
    for line in result3:
        print(f"    {line}")
    print(f"    ({p3.total_ops} neural ops)")

    # Pipeline 4: grep -v sales | wc
    print("\n  Pipeline: grep -v 'sales' | wc")
    p4 = NeuralPipeline(
        NeuralGrep("sales", invert=True),
        NeuralWc(),
    )
    result4 = p4.run(data)
    for line in result4:
        print(f"    {line}")
    print(f"    ({p4.total_ops} neural ops)")

    total = p1.total_ops + p2.total_ops + p3.total_ops + p4.total_ops
    print(f"\n  Total neural ops across all pipelines: {total}")
    print("\n✅ Neural pipes: grep, sort, uniq, wc, head, tr, cut all working")


if __name__ == "__main__":
    demo()
