"""SOME Self-Improvement Engine for Skynet.

Analyses accumulated trajectory data (ncpu-outcomes.jsonl) and generates
actionable improvements. This is the closed loop:

    Execute → Record → Analyse → Adapt → Execute better

Usage:
    python -m bridge.self_improve analyse          # Analyse outcomes, suggest improvements
    python -m bridge.self_improve report            # Full self-improvement report
    python -m bridge.self_improve thresholds        # Suggest threshold adjustments based on data
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OUTCOMES_PATH = Path("/Users/noc/clawd/data/ncpu-outcomes.jsonl")
IMPROVEMENTS_PATH = Path("/Users/noc/clawd/data/ncpu-improvements.jsonl")


def load_outcomes(path: Path = OUTCOMES_PATH) -> list[dict]:
    """Load all recorded outcomes."""
    outcomes = []
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        outcomes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return outcomes


@dataclass
class TaskProfile:
    """Statistical profile of a specific task."""
    name: str
    total_runs: int
    successes: int
    failures: int
    success_rate: float
    avg_execution_ms: float
    last_success: float  # timestamp
    last_failure: float  # timestamp
    failure_streak: int  # consecutive failures at end
    category: str


def build_task_profiles(outcomes: list[dict]) -> dict[str, TaskProfile]:
    """Build per-task statistical profiles from outcomes."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for o in outcomes:
        by_task[o.get("task_name", "unknown")].append(o)

    profiles = {}
    for name, runs in by_task.items():
        runs.sort(key=lambda x: x.get("timestamp", 0))
        successes = sum(1 for r in runs if r.get("success"))
        failures = len(runs) - successes

        exec_times = [r.get("execution_time_ms", 0) for r in runs]
        avg_ms = sum(exec_times) / len(exec_times) if exec_times else 0

        last_success = max(
            (r.get("timestamp", 0) for r in runs if r.get("success")), default=0
        )
        last_failure = max(
            (r.get("timestamp", 0) for r in runs if not r.get("success")), default=0
        )

        # Count consecutive failures at the end
        streak = 0
        for r in reversed(runs):
            if not r.get("success"):
                streak += 1
            else:
                break

        profiles[name] = TaskProfile(
            name=name,
            total_runs=len(runs),
            successes=successes,
            failures=failures,
            success_rate=round(successes / len(runs) * 100, 1) if runs else 0,
            avg_execution_ms=round(avg_ms, 2),
            last_success=last_success,
            last_failure=last_failure,
            failure_streak=streak,
            category=runs[0].get("category", "unknown"),
        )

    return profiles


def analyse_patterns(profiles: dict[str, TaskProfile]) -> list[dict]:
    """Detect patterns and generate improvement suggestions."""
    suggestions = []

    for name, p in profiles.items():
        # Pattern: Always failing → something is structurally wrong
        if p.total_runs >= 3 and p.success_rate == 0:
            suggestions.append({
                "type": "structural_failure",
                "severity": "critical",
                "task": name,
                "message": f"'{name}' has NEVER succeeded in {p.total_runs} runs. "
                           f"This likely needs a config/threshold change, not a retry.",
                "action": "review_threshold",
            })

        # Pattern: Recently started failing → regression
        elif p.failure_streak >= 3 and p.success_rate > 50:
            suggestions.append({
                "type": "regression",
                "severity": "high",
                "task": name,
                "message": f"'{name}' was {p.success_rate}% successful but has failed "
                           f"{p.failure_streak} times in a row. Possible regression.",
                "action": "investigate",
            })

        # Pattern: Intermittent failures → flaky check
        elif 30 < p.success_rate < 90 and p.total_runs >= 5:
            suggestions.append({
                "type": "flaky",
                "severity": "medium",
                "task": name,
                "message": f"'{name}' is flaky at {p.success_rate}% success rate. "
                           f"Consider widening threshold or fixing root cause.",
                "action": "adjust_threshold",
            })

        # Pattern: Getting slower → performance degradation
        # (would need time-series analysis of execution_time_ms)

    return suggestions


def suggest_threshold_adjustments(outcomes: list[dict]) -> list[dict]:
    """Suggest threshold changes based on observed values."""
    adjustments = []

    # Group health check values by task
    health_values: dict[str, list[int]] = defaultdict(list)
    for o in outcomes:
        if o.get("category") == "health_check" and o.get("input_data"):
            name = o.get("task_name", "")
            for key, val in o["input_data"].items():
                if isinstance(val, (int, float)):
                    health_values[f"{name}:{key}"].append(val)

    for key, values in health_values.items():
        if len(values) < 5:
            continue
        avg = sum(values) / len(values)
        mn, mx = min(values), max(values)

        # If the value is consistently near or above threshold
        # suggest adjusting
        adjustments.append({
            "metric": key,
            "samples": len(values),
            "avg": round(avg, 1),
            "min": mn,
            "max": mx,
            "range": mx - mn,
        })

    return adjustments


def generate_report(outcomes: list[dict]) -> dict:
    """Generate full self-improvement report."""
    if not outcomes:
        return {"status": "no_data", "message": "No outcomes recorded yet."}

    profiles = build_task_profiles(outcomes)
    suggestions = analyse_patterns(profiles)
    thresholds = suggest_threshold_adjustments(outcomes)

    # Overall stats
    total = len(outcomes)
    successes = sum(1 for o in outcomes if o.get("success"))
    neural = sum(1 for o in outcomes if o.get("neural_verified"))

    # Category breakdown
    by_cat = Counter(o.get("category", "unknown") for o in outcomes)

    # Time range
    timestamps = [o.get("timestamp", 0) for o in outcomes]
    duration_hours = (max(timestamps) - min(timestamps)) / 3600 if timestamps else 0

    return {
        "summary": {
            "total_outcomes": total,
            "success_rate": round(successes / total * 100, 1),
            "neural_verified_rate": round(neural / total * 100, 1),
            "duration_hours": round(duration_hours, 1),
            "unique_tasks": len(profiles),
            "categories": dict(by_cat),
        },
        "task_profiles": {
            name: {
                "runs": p.total_runs,
                "success_rate": p.success_rate,
                "avg_ms": p.avg_execution_ms,
                "failure_streak": p.failure_streak,
            }
            for name, p in sorted(profiles.items(), key=lambda x: x[1].success_rate)
        },
        "suggestions": suggestions,
        "threshold_data": thresholds,
    }


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    outcomes = load_outcomes()

    if cmd == "analyse":
        profiles = build_task_profiles(outcomes)
        suggestions = analyse_patterns(profiles)
        if not suggestions:
            print("No improvement suggestions — all tasks performing within parameters.")
        else:
            for s in suggestions:
                print(f"[{s['severity'].upper()}] {s['type']}: {s['message']}")
                print(f"  → Action: {s['action']}")
                print()

    elif cmd == "thresholds":
        adjustments = suggest_threshold_adjustments(outcomes)
        print(json.dumps(adjustments, indent=2))

    elif cmd == "report":
        report = generate_report(outcomes)
        print(json.dumps(report, indent=2))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
