"""Skynet integration — wires nCPU bridge + SOME feedback into heartbeat workflow.

Usage:
    python -m bridge.skynet_integration run-checks
    python -m bridge.skynet_integration stats
    python -m bridge.skynet_integration trend
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
BRIDGE_PATH = Path("/Users/noc/projects/ncpu-bridge")
MEMDB = "/Users/noc/clawd/tools/memdb"

if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))
if str(BRIDGE_PATH) not in sys.path:
    sys.path.insert(0, str(BRIDGE_PATH))

from bridge.compute import NCPUBridge
from bridge.health import HealthComputer
from bridge.obligations import ObligationChecker
from bridge.feedback_loop import SkynetFeedbackLoop, TaskOutcome


STALENESS_THRESHOLDS = {
    "critical": 7200,
    "high": 14400,
    "medium": 86400,
    "low": 172800,
}


def get_obligations() -> list[dict]:
    """Get obligations from memdb."""
    try:
        result = subprocess.run(
            [MEMDB, "obligation", "list", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        return json.loads(result.stdout) if result.stdout.strip() else []
    except Exception:
        return []


def get_system_stats() -> dict:
    """Get system health metrics."""
    import shutil

    disk = shutil.disk_usage("/")
    disk_pct = int(disk.used / disk.total * 100)

    try:
        import subprocess as sp
        vm = sp.run(["vm_stat"], capture_output=True, text=True, timeout=3)
        for line in vm.stdout.splitlines():
            if "Pages free" in line:
                pages = int(line.split(":")[1].strip().rstrip("."))
                mem_free_mb = pages * 4096 // (1024 * 1024)
                break
        else:
            mem_free_mb = 0
    except Exception:
        mem_free_mb = 0

    return {"disk_pct": disk_pct, "mem_free_mb": mem_free_mb}


def run_checks():
    """Run all neural-verified checks with SOME feedback recording."""
    bridge = NCPUBridge()
    health = HealthComputer(bridge)
    checker = ObligationChecker(bridge)
    feedback = SkynetFeedbackLoop()

    now = int(time.time())
    results = {"obligations": [], "health": [], "feedback": []}

    # ── Obligation checks ─────────────────────────────────
    obligations = get_obligations()
    for ob in obligations:
        name = ob.get("name", "unknown")
        severity = ob.get("severity", "medium")
        last_status = ob.get("last_status", "unknown")
        checked = ob.get("checked", "")
        fail_count = ob.get("fail_count", 0)

        if not checked:
            continue

        try:
            dt = datetime.fromisoformat(checked)
            checked_epoch = int(dt.timestamp())
        except Exception:
            continue

        threshold = STALENESS_THRESHOLDS.get(severity, 86400)

        t0 = time.perf_counter()
        elapsed = bridge.sub(now, checked_epoch)
        zf, sf = bridge.cmp(elapsed, threshold)
        stale = zf or (not sf)
        hours_ago = bridge.div(elapsed, 3600)
        exec_ms = (time.perf_counter() - t0) * 1000

        ok = last_status != "fail" and not stale

        ob_result = {
            "name": name,
            "severity": severity,
            "status": last_status,
            "stale": stale,
            "hours_ago": hours_ago,
            "fail_count": fail_count,
            "ok": ok,
        }
        results["obligations"].append(ob_result)

        # Record to SOME feedback
        outcome = TaskOutcome(
            task_name=f"obligation:{name}",
            category="obligation_check",
            success=ok,
            neural_verified=True,
            execution_time_ms=round(exec_ms, 2),
            input_data={"epoch_now": now, "epoch_checked": checked_epoch, "threshold": threshold},
            output_data=ob_result,
        )
        suggestion = feedback.record_outcome(outcome)
        results["feedback"].append(suggestion)

    # ── Health checks ─────────────────────────────────────
    sys_stats = get_system_stats()

    t0 = time.perf_counter()
    disk_check = health.check_threshold(sys_stats["disk_pct"], 90, "disk_usage")
    exec_ms = (time.perf_counter() - t0) * 1000

    results["health"].append({
        "name": "disk_usage",
        "value": sys_stats["disk_pct"],
        "threshold": 90,
        "ok": not disk_check["exceeded"],
    })

    feedback.record_outcome(TaskOutcome(
        task_name="health:disk_usage",
        category="health_check",
        success=not disk_check["exceeded"],
        neural_verified=True,
        execution_time_ms=round(exec_ms, 2),
        input_data={"disk_pct": sys_stats["disk_pct"]},
        output_data=disk_check,
    ))

    t0 = time.perf_counter()
    zf, sf = bridge.cmp(sys_stats["mem_free_mb"], 200)
    mem_low = sf  # < 200MB
    exec_ms = (time.perf_counter() - t0) * 1000

    results["health"].append({
        "name": "memory_free",
        "value": sys_stats["mem_free_mb"],
        "threshold": 200,
        "ok": not mem_low,
    })

    feedback.record_outcome(TaskOutcome(
        task_name="health:memory_free",
        category="health_check",
        success=not mem_low,
        neural_verified=True,
        execution_time_ms=round(exec_ms, 2),
        input_data={"mem_free_mb": sys_stats["mem_free_mb"]},
        output_data={"mem_free_mb": sys_stats["mem_free_mb"], "low": mem_low},
    ))

    # ── Summary ───────────────────────────────────────────
    session_stats = feedback.get_session_stats()
    results["session_stats"] = session_stats

    ob_ok = sum(1 for o in results["obligations"] if o["ok"])
    ob_total = len(results["obligations"])
    health_ok = sum(1 for h in results["health"] if h["ok"])
    health_total = len(results["health"])

    # Output
    print(json.dumps({
        "obligations": {"ok": ob_ok, "total": ob_total, "details": results["obligations"]},
        "health": {"ok": health_ok, "total": health_total, "details": results["health"]},
        "feedback": session_stats,
        "neural_verified": True,
    }, indent=2))

    # Exit code
    if session_stats.get("critical_failures", 0) > 0:
        sys.exit(2)
    elif ob_ok < ob_total or health_ok < health_total:
        sys.exit(1)
    sys.exit(0)


def show_stats():
    """Show feedback session stats."""
    feedback = SkynetFeedbackLoop()
    trend = feedback.get_trend()
    print(json.dumps(trend, indent=2))


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m bridge.skynet_integration [run-checks|stats|trend]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "run-checks":
        run_checks()
    elif cmd in ("stats", "trend"):
        show_stats()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
