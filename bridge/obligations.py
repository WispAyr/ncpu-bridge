"""Obligation checking through verified neural computation."""

from __future__ import annotations

from bridge.compute import NCPUBridge


class ObligationChecker:
    """Run obligation checks through the neural ALU."""

    def __init__(self, bridge: NCPUBridge | None = None):
        self.bridge = bridge or NCPUBridge()

    def check_interval(
        self, last_run_epoch: int, now_epoch: int, interval_seconds: int
    ) -> dict:
        """Check if an obligation is overdue using neural arithmetic.

        Returns dict with elapsed, interval, overdue (bool), seconds_until_due.
        """
        elapsed = self.bridge.sub(now_epoch, last_run_epoch)
        seconds_until_due = self.bridge.sub(interval_seconds, elapsed)

        # overdue when elapsed >= interval
        zf, sf = self.bridge.cmp(elapsed, interval_seconds)
        overdue = zf or (not sf)  # elapsed == interval or elapsed > interval

        return {
            "elapsed": elapsed,
            "interval": interval_seconds,
            "overdue": overdue,
            "seconds_until_due": max(seconds_until_due, 0),
        }

    def compute_trend(
        self, pass_counts: list[int], fail_counts: list[int]
    ) -> dict:
        """Compute obligation health trends through the neural ALU.

        Takes parallel lists of pass/fail counts per period.
        Returns totals, rates, and trend direction.
        """
        if not pass_counts or not fail_counts:
            return {
                "total_pass": 0,
                "total_fail": 0,
                "total_checks": 0,
                "pass_rate_pct": 0,
                "trend": "unknown",
            }

        total_pass = pass_counts[0]
        for p in pass_counts[1:]:
            total_pass = self.bridge.add(total_pass, p)

        total_fail = fail_counts[0]
        for f in fail_counts[1:]:
            total_fail = self.bridge.add(total_fail, f)

        total_checks = self.bridge.add(total_pass, total_fail)

        # pass_rate = (total_pass * 100) / total_checks
        scaled = self.bridge.mul(total_pass, 100)
        pass_rate = self.bridge.div(scaled, total_checks) if total_checks else 0

        # trend: compare first half vs second half pass counts
        n = len(pass_counts)
        if n < 2:
            trend = "stable"
        else:
            mid = n // 2
            first_half = pass_counts[0]
            for p in pass_counts[1:mid]:
                first_half = self.bridge.add(first_half, p)
            second_half = pass_counts[mid]
            for p in pass_counts[mid + 1 :]:
                second_half = self.bridge.add(second_half, p)

            zf, sf = self.bridge.cmp(second_half, first_half)
            if zf:
                trend = "stable"
            elif sf:
                trend = "declining"
            else:
                trend = "improving"

        return {
            "total_pass": total_pass,
            "total_fail": total_fail,
            "total_checks": total_checks,
            "pass_rate_pct": pass_rate,
            "trend": trend,
        }

    def check_interval_asm(
        self, last_run_epoch: int, now_epoch: int, interval_seconds: int
    ) -> dict:
        """Check obligation via assembly on the neural CPU.

        R0 = elapsed, R1 = interval, R2 = seconds_until_due, R3 = overdue flag.
        """
        asm = f"""\
MOV R4, {now_epoch}
MOV R5, {last_run_epoch}
SUB R0, R4, R5
MOV R1, {interval_seconds}
SUB R2, R1, R0
CMP R0, R1
JS not_overdue
MOV R3, 1
JMP done
not_overdue:
MOV R3, 0
done:
HALT
"""
        result = self.bridge.run_program(asm)
        regs = result["registers"]
        return {
            "elapsed": regs["R0"],
            "interval": regs["R1"],
            "seconds_until_due": max(regs["R2"], 0),
            "overdue": regs["R3"] == 1,
        }
