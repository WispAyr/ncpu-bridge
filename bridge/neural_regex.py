"""Neural Regex Engine — pattern matching computed through nCPU.

A simplified regex engine where every character comparison and
state transition goes through trained neural networks.

Supported patterns:
- Literal characters: abc
- Dot (any char): .
- Star (zero or more): a*
- Plus (one or more): a+
- Question (zero or one): a?
- Character classes: [abc], [a-z]
- Anchors: ^ (start), $ (end)

Every comparison between input chars and pattern chars uses neural CMP.
State machine transitions use neural arithmetic.

Usage:
    python -m bridge.neural_regex demo
    python -m bridge.neural_regex match <pattern> <text>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bridge.config import get_ncpu_path, get_bridge_path, get_clawd_data_path
NCPU_PATH = get_ncpu_path()
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


@dataclass
class MatchResult:
    matched: bool
    start: int = 0
    end: int = 0
    text: str = ""
    neural_ops: int = 0


class NeuralRegex:
    """Regex engine with neural character comparison.
    
    Compiles pattern to a simple NFA, then executes matching
    using neural CMP for every character test.
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def _neural_char_eq(self, a: int, b: int) -> bool:
        """Compare two characters using neural CMP."""
        zf, _ = self.bridge.cmp(a, b)
        self._op()
        return zf
    
    def _neural_char_range(self, ch: int, lo: int, hi: int) -> bool:
        """Check if char is in range [lo, hi] using neural CMP."""
        # ch >= lo
        zf1, sf1 = self.bridge.cmp(ch, lo)
        self._op()
        ge_lo = zf1 or not sf1
        
        # ch <= hi
        zf2, sf2 = self.bridge.cmp(ch, hi)
        self._op()
        le_hi = zf2 or sf2
        
        return ge_lo and le_hi
    
    def _match_char(self, ch: int, pattern_char: str) -> bool:
        """Match a single character against a pattern element."""
        if pattern_char == '.':
            return True  # Dot matches anything
        
        if pattern_char.startswith('[') and pattern_char.endswith(']'):
            # Character class
            inner = pattern_char[1:-1]
            if len(inner) == 3 and inner[1] == '-':
                # Range: [a-z]
                return self._neural_char_range(ch, ord(inner[0]), ord(inner[2]))
            else:
                # Set: [abc]
                for c in inner:
                    if self._neural_char_eq(ch, ord(c)):
                        return True
                return False
        
        # Literal
        return self._neural_char_eq(ch, ord(pattern_char))
    
    def _parse_pattern(self, pattern: str) -> list[tuple[str, str]]:
        """Parse pattern into (element, quantifier) pairs.
        
        Returns list of (char_spec, quantifier) where quantifier is
        '', '*', '+', or '?'.
        """
        tokens = []
        i = 0
        
        while i < len(pattern):
            if pattern[i] == '^':
                tokens.append(('^', ''))
                i += 1
            elif pattern[i] == '$':
                tokens.append(('$', ''))
                i += 1
            elif pattern[i] == '[':
                # Find matching ]
                j = pattern.index(']', i) + 1
                char_spec = pattern[i:j]
                quant = ''
                if j < len(pattern) and pattern[j] in '*+?':
                    quant = pattern[j]
                    j += 1
                tokens.append((char_spec, quant))
                i = j
            elif pattern[i] == '.':
                quant = ''
                if i + 1 < len(pattern) and pattern[i + 1] in '*+?':
                    quant = pattern[i + 1]
                    i += 1
                tokens.append(('.', quant))
                i += 1
            else:
                quant = ''
                if i + 1 < len(pattern) and pattern[i + 1] in '*+?':
                    quant = pattern[i + 1]
                    i += 1
                tokens.append((pattern[i], quant))
                i += 1
        
        return tokens
    
    def _match_at(self, tokens: list[tuple[str, str]], text: str, pos: int) -> Optional[int]:
        """Try to match tokens starting at position `pos`.
        
        Returns end position if matched, None otherwise.
        All character comparisons are neural.
        """
        ti = 0  # Token index
        si = pos  # String index
        
        while ti < len(tokens):
            char_spec, quant = tokens[ti]
            
            # Anchor: start
            if char_spec == '^':
                zf, _ = self.bridge.cmp(pos, 0)
                self._op()
                if not zf:
                    return None
                ti += 1
                continue
            
            # Anchor: end
            if char_spec == '$':
                zf, _ = self.bridge.cmp(si, len(text))
                self._op()
                if not zf:
                    return None
                ti += 1
                continue
            
            if quant == '*':
                # Zero or more: greedy match
                count = 0
                while si < len(text) and self._match_char(ord(text[si]), char_spec):
                    count = self.bridge.add(count, 1)
                    self._op()
                    si += 1
                
                # Try to match rest with decreasing si
                while True:
                    rest = self._match_at(tokens[ti + 1:], text, si)
                    if rest is not None:
                        return rest
                    if count == 0:
                        break
                    si -= 1
                    count -= 1
                return None
            
            elif quant == '+':
                # One or more: must match at least once
                if si >= len(text) or not self._match_char(ord(text[si]), char_spec):
                    return None
                si += 1
                
                # Then greedy like *
                count = 1
                while si < len(text) and self._match_char(ord(text[si]), char_spec):
                    count = self.bridge.add(count, 1)
                    self._op()
                    si += 1
                
                while True:
                    rest = self._match_at(tokens[ti + 1:], text, si)
                    if rest is not None:
                        return rest
                    if count <= 1:
                        break
                    si -= 1
                    count -= 1
                return None
            
            elif quant == '?':
                # Zero or one
                if si < len(text) and self._match_char(ord(text[si]), char_spec):
                    rest = self._match_at(tokens[ti + 1:], text, si + 1)
                    if rest is not None:
                        return rest
                # Try without
                rest = self._match_at(tokens[ti + 1:], text, si)
                return rest
            
            else:
                # Exact match required
                if si >= len(text):
                    return None
                if not self._match_char(ord(text[si]), char_spec):
                    return None
                si += 1
            
            ti += 1
        
        return si
    
    def match(self, pattern: str, text: str) -> MatchResult:
        """Match pattern against text. Returns first match."""
        self._ops = 0
        tokens = self._parse_pattern(pattern)
        
        # Check for start anchor
        anchored_start = tokens and tokens[0][0] == '^'
        
        if anchored_start:
            end = self._match_at(tokens, text, 0)
            if end is not None:
                return MatchResult(True, 0, end, text[0:end], self._ops)
            return MatchResult(False, neural_ops=self._ops)
        
        # Try matching at each position
        for start in range(len(text)):
            end = self._match_at(tokens, text, start)
            if end is not None:
                return MatchResult(True, start, end, text[start:end], self._ops)
        
        return MatchResult(False, neural_ops=self._ops)
    
    def find_all(self, pattern: str, text: str) -> list[MatchResult]:
        """Find all non-overlapping matches."""
        self._ops = 0
        tokens = self._parse_pattern(pattern)
        results = []
        pos = 0
        
        while pos < len(text):
            end = self._match_at(tokens, text, pos)
            if end is not None and end > pos:
                results.append(MatchResult(True, pos, end, text[pos:end], self._ops))
                pos = end
            else:
                pos += 1
        
        return results


# ── CLI ──────────────────────────────────────────────────────

def demo():
    rx = NeuralRegex()
    
    print("Neural Regex Engine")
    print("=" * 60)
    print("Every character comparison → neural CMP\n")
    
    tests = [
        ("hello", "say hello world", "literal match"),
        ("h.llo", "say hello world", "dot wildcard"),
        ("^say", "say hello world", "start anchor"),
        ("world$", "say hello world", "end anchor"),
        ("[0-9]+", "error 404 found", "digit class"),
        ("a*b", "aaaab", "star quantifier"),
        ("go+d", "goood", "plus quantifier"),
        ("colou?r", "color", "question mark"),
        ("colou?r", "colour", "question mark (with u)"),
        ("[a-z]+@[a-z]+", "mail ewan@parkwise ok", "email-like"),
        ("n.PU", "nCPU is neural", "nCPU pattern"),
    ]
    
    for pattern, text, desc in tests:
        result = rx.match(pattern, text)
        if result.matched:
            print(f"  /{pattern}/ in \"{text}\"")
            print(f"    ✅ matched \"{result.text}\" at [{result.start}:{result.end}] ({result.neural_ops} ops) — {desc}")
        else:
            print(f"  /{pattern}/ in \"{text}\"")
            print(f"    ❌ no match ({result.neural_ops} ops) — {desc}")
    
    print()
    
    # Find all
    print("── Find All ──")
    text = "POS at 10.10.10.238 port 3000 and NVR at 10.10.10.2"
    pattern = "[0-9]+"
    matches = rx.find_all(pattern, text)
    print(f"  /{pattern}/ in \"{text}\"")
    for m in matches:
        print(f"    → \"{m.text}\" at [{m.start}:{m.end}]")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    if cmd == "demo":
        demo()
    elif cmd == "match" and len(sys.argv) > 3:
        rx = NeuralRegex()
        result = rx.match(sys.argv[2], sys.argv[3])
        if result.matched:
            print(f"Match: \"{result.text}\" at [{result.start}:{result.end}] ({result.neural_ops} ops)")
        else:
            print(f"No match ({result.neural_ops} ops)")
    else:
        print("Usage: python -m bridge.neural_regex [demo|match <pattern> <text>]")


if __name__ == "__main__":
    main()
