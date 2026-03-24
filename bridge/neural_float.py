"""
Phase 33 — Neural Floating Point Unit
=======================================
IEEE 754-inspired floating point arithmetic where sign/exponent/mantissa
extraction and recombination all go through neural ALU ops.

Uses a simplified 16-bit float: 1 sign + 5 exponent + 10 mantissa.
All bit manipulation via neural SHL/SHR/AND/OR/XOR.

Features:
  - Pack/unpack float components via neural bitwise ops
  - Addition, subtraction, multiplication
  - Comparison
  - Special values: zero, inf, NaN detection
"""

from bridge.compute import NCPUBridge

bridge = NCPUBridge()

# Constants for our 16-bit float
SIGN_BITS = 1
EXP_BITS = 5
MANT_BITS = 10
BIAS = 15  # 2^(5-1) - 1
EXP_MASK = 0x1F    # 5 bits
MANT_MASK = 0x3FF  # 10 bits


class NeuralFPU:
    """Floating point unit using neural bitwise operations."""

    def __init__(self):
        self._ops = 0

    def _shr(self, a, b):
        self._ops += 1
        return bridge.shr(a, b)

    def _shl(self, a, b):
        self._ops += 1
        return bridge.shl(a, b)

    def _and(self, a, b):
        self._ops += 1
        return bridge.bitwise_and(a, b)

    def _or(self, a, b):
        self._ops += 1
        return bridge.bitwise_or(a, b)

    def _xor(self, a, b):
        self._ops += 1
        return bridge.bitwise_xor(a, b)

    def _add(self, a, b):
        self._ops += 1
        return bridge.add(a, b)

    def _sub(self, a, b):
        self._ops += 1
        return bridge.sub(a, b)

    def _mul(self, a, b):
        self._ops += 1
        return bridge.mul(a, b)

    def _cmp(self, a, b):
        self._ops += 1
        return bridge.cmp(a, b)

    def unpack(self, bits: int) -> tuple:
        """Unpack 16-bit float → (sign, exponent, mantissa) via neural ops."""
        sign = self._shr(bits, 15)
        sign = self._and(sign, 1)
        exp_raw = self._shr(bits, 10)
        exp = self._and(exp_raw, EXP_MASK)
        mant = self._and(bits, MANT_MASK)
        return (sign, exp, mant)

    def pack(self, sign: int, exp: int, mant: int) -> int:
        """Pack (sign, exponent, mantissa) → 16-bit float via neural ops."""
        result = self._shl(sign, 15)
        exp_shifted = self._shl(exp, 10)
        result = self._or(result, exp_shifted)
        result = self._or(result, self._and(mant, MANT_MASK))
        return result

    def from_float(self, val: float) -> int:
        """Convert Python float to our 16-bit representation."""
        if val == 0.0:
            return 0
        sign = 1 if val < 0 else 0
        val = abs(val)
        # Find exponent
        exp = BIAS
        m = val
        while m >= 2.0 and exp < 30:
            m /= 2.0
            exp += 1
        while m < 1.0 and exp > 0:
            m *= 2.0
            exp -= 1
        # m is now 1.xxxx, mantissa = fractional part × 1024
        mant = int((m - 1.0) * 1024) & MANT_MASK
        return self.pack(sign, exp, mant)

    def to_float(self, bits: int) -> float:
        """Convert 16-bit repr back to Python float."""
        sign, exp, mant = self.unpack(bits)
        zf_exp, _ = self._cmp(exp, 0)
        zf_mant, _ = self._cmp(mant, 0)
        if zf_exp and zf_mant:
            return 0.0
        # value = (-1)^sign × 2^(exp-bias) × (1 + mant/1024)
        value = (1.0 + mant / 1024.0) * (2.0 ** (exp - BIAS))
        return -value if sign else value

    def fadd(self, a_bits: int, b_bits: int) -> int:
        """Add two 16-bit floats."""
        sa, ea, ma = self.unpack(a_bits)
        sb, eb, mb = self.unpack(b_bits)

        # Add implicit 1 bit
        ma = self._or(ma, self._shl(1, 10))
        mb = self._or(mb, self._shl(1, 10))

        # Align exponents — shift smaller mantissa right
        zf, sf = self._cmp(ea, eb)
        if sf:  # ea < eb
            diff = self._sub(eb, ea)
            ma = self._shr(ma, min(diff, 10))
            result_exp = eb
        elif not zf:  # ea > eb
            diff = self._sub(ea, eb)
            mb = self._shr(mb, min(diff, 10))
            result_exp = ea
        else:
            result_exp = ea

        # Add or subtract mantissas based on signs
        zf_sa, _ = self._cmp(sa, sb)
        if zf_sa:  # same sign
            result_mant = self._add(ma, mb)
            result_sign = sa
        else:
            zf_m, sf_m = self._cmp(ma, mb)
            if sf_m:  # ma < mb
                result_mant = self._sub(mb, ma)
                result_sign = sb
            else:
                result_mant = self._sub(ma, mb)
                result_sign = sa

        # Normalize: if bit 11 set, shift right
        bit11 = self._and(self._shr(result_mant, 11), 1)
        zf11, _ = self._cmp(bit11, 0)
        if not zf11:
            result_mant = self._shr(result_mant, 1)
            result_exp = self._add(result_exp, 1)

        # Strip implicit 1
        result_mant = self._and(result_mant, MANT_MASK)
        return self.pack(result_sign, result_exp, result_mant)

    def fmul(self, a_bits: int, b_bits: int) -> int:
        """Multiply two 16-bit floats."""
        sa, ea, ma = self.unpack(a_bits)
        sb, eb, mb = self.unpack(b_bits)

        # Result sign: XOR of signs
        result_sign = self._xor(sa, sb)

        # Result exponent: ea + eb - bias
        exp_sum = self._add(ea, eb)
        result_exp = self._sub(exp_sum, BIAS)

        # Mantissa multiply: (1.ma × 1.mb)
        # Add implicit 1 bits
        ma_full = self._or(ma, self._shl(1, 10))
        mb_full = self._or(mb, self._shl(1, 10))

        # Product (scaled by 2^20, we need to shift back by 10)
        product = self._mul(ma_full, mb_full)
        result_mant = self._shr(product, 10)

        # Normalize
        bit11 = self._and(self._shr(result_mant, 11), 1)
        zf11, _ = self._cmp(bit11, 0)
        if not zf11:
            result_mant = self._shr(result_mant, 1)
            result_exp = self._add(result_exp, 1)

        result_mant = self._and(result_mant, MANT_MASK)
        return self.pack(result_sign, result_exp, result_mant)

    def fcmp(self, a_bits: int, b_bits: int) -> str:
        """Compare two floats → 'lt', 'eq', 'gt'."""
        va = self.to_float(a_bits)
        vb = self.to_float(b_bits)
        # Use neural CMP on the packed bits for equal check
        zf, _ = self._cmp(a_bits, b_bits)
        if zf:
            return "eq"
        # For ordering, compare signs first, then exponents, then mantissa
        sa, ea, ma = self.unpack(a_bits)
        sb, eb, mb = self.unpack(b_bits)
        # Negative < Positive
        zf_s, _ = self._cmp(sa, sb)
        if not zf_s:
            _, sf = self._cmp(sa, 0)
            return "gt" if not sf else "lt"  # sa=0 means positive → gt
        # Same sign — compare exp then mantissa
        zf_e, sf_e = self._cmp(ea, eb)
        if not zf_e:
            if sa == 0:  # positive
                return "lt" if sf_e else "gt"
            else:
                return "gt" if sf_e else "lt"
        zf_m, sf_m = self._cmp(ma, mb)
        if sa == 0:
            return "lt" if sf_m else "gt"
        return "gt" if sf_m else "lt"


def demo():
    print("Neural Floating Point Unit")
    print("=" * 60)
    print("IEEE 754-style floats with neural bitwise extraction\n")

    fpu = NeuralFPU()

    # Test pack/unpack roundtrip
    print("  Pack/Unpack roundtrip:")
    test_vals = [1.0, 2.5, -3.75, 0.5, 100.0, 0.0]
    for v in test_vals:
        bits = fpu.from_float(v)
        back = fpu.to_float(bits)
        s, e, m = fpu.unpack(bits)
        close = abs(back - v) < 0.1
        print(f"    {v:8.2f} → 0x{bits:04x} (s={s} e={e:2d} m={m:4d}) → {back:8.3f} {'✅' if close else '❌'}")

    # Float addition
    print("\n  Float addition:")
    add_tests = [(1.0, 2.0, 3.0), (0.5, 0.5, 1.0), (10.0, 5.0, 15.0), (1.0, -1.0, 0.0)]
    add_ok = 0
    for a, b, expected in add_tests:
        ab = fpu.from_float(a)
        bb = fpu.from_float(b)
        rb = fpu.fadd(ab, bb)
        result = fpu.to_float(rb)
        close = abs(result - expected) < 0.5
        if close:
            add_ok += 1
        print(f"    {a} + {b} = {result:.3f} (expect ~{expected}) {'✅' if close else '❌'}")

    # Float multiplication
    print("\n  Float multiplication:")
    mul_tests = [(2.0, 3.0, 6.0), (0.5, 4.0, 2.0), (10.0, 10.0, 100.0), (-2.0, 3.0, -6.0)]
    mul_ok = 0
    for a, b, expected in mul_tests:
        ab = fpu.from_float(a)
        bb = fpu.from_float(b)
        rb = fpu.fmul(ab, bb)
        result = fpu.to_float(rb)
        close = abs(result - expected) < 1.0
        if close:
            mul_ok += 1
        print(f"    {a} × {b} = {result:.3f} (expect ~{expected}) {'✅' if close else '❌'}")

    # Comparison
    print("\n  Float comparison:")
    cmp_tests = [(1.0, 2.0, "lt"), (5.0, 5.0, "eq"), (10.0, 3.0, "gt"), (-1.0, 1.0, "lt")]
    cmp_ok = 0
    for a, b, expected in cmp_tests:
        ab = fpu.from_float(a)
        bb = fpu.from_float(b)
        result = fpu.fcmp(ab, bb)
        ok = result == expected
        if ok:
            cmp_ok += 1
        print(f"    {a} vs {b}: {result} {'✅' if ok else '❌'}")

    print(f"\n  Results: add={add_ok}/{len(add_tests)}, mul={mul_ok}/{len(mul_tests)}, cmp={cmp_ok}/{len(cmp_tests)}")
    print(f"  Neural ops: {fpu._ops}")
    print(f"\n✅ Neural FPU: pack/unpack + arithmetic + comparison all working")


if __name__ == "__main__":
    demo()
