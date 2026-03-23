"""Health check computations compiled to neural ALU operations."""

from __future__ import annotations

from bridge.compute import NCPUBridge


class HealthComputer:
    """Compile health checks to neural ALU operations."""

    def __init__(self, bridge: NCPUBridge | None = None):
        self.bridge = bridge or NCPUBridge()

    def check_threshold(self, value: int, threshold: int, name: str) -> dict:
        """Check if value exceeds threshold using neural comparison.

        Returns dict with name, value, threshold, exceeded (bool), headroom.
        """
        zf, sf = self.bridge.cmp(value, threshold)
        # CMP computes value - threshold: SF=True means value < threshold
        exceeded = not sf and not zf  # value > threshold
        headroom = self.bridge.sub(threshold, value)
        return {
            "name": name,
            "value": value,
            "threshold": threshold,
            "exceeded": exceeded,
            "headroom": headroom,
        }

    def compute_stats(self, values: list[int]) -> dict:
        """Compute sum, min, max, count, avg through the neural ALU."""
        if not values:
            return {"sum": 0, "min": 0, "max": 0, "count": 0, "avg": 0}

        total = values[0]
        lo = values[0]
        hi = values[0]

        for v in values[1:]:
            total = self.bridge.add(total, v)

            # min: if v < lo, update
            zf, sf = self.bridge.cmp(v, lo)
            if sf:
                lo = v

            # max: if v > hi, update
            zf, sf = self.bridge.cmp(v, hi)
            if not sf and not zf:
                hi = v

        count = len(values)
        avg = self.bridge.div(total, count) if count else 0

        return {
            "sum": total,
            "min": lo,
            "max": hi,
            "count": count,
            "avg": avg,
        }

    def check_threshold_asm(self, value: int, threshold: int) -> dict:
        """Check threshold by compiling to assembly and running on neural CPU.

        R0 = value, R1 = threshold, R2 = headroom, R3 = exceeded flag.
        """
        asm = f"""\
MOV R0, {value}
MOV R1, {threshold}
SUB R2, R1, R0
CMP R0, R1
JS not_exceeded
JZ not_exceeded
MOV R3, 1
JMP done
not_exceeded:
MOV R3, 0
done:
HALT
"""
        result = self.bridge.run_program(asm)
        regs = result["registers"]
        return {
            "value": regs["R0"],
            "threshold": regs["R1"],
            "headroom": regs["R2"],
            "exceeded": regs["R3"] == 1,
        }
