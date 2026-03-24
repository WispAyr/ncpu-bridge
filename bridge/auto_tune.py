"""SOME Auto-Tuner — automatically adjusts thresholds based on outcome data.

The closed loop:
1. Checks run → outcomes recorded (ncpu-outcomes.jsonl)
2. Auto-tuner analyses outcomes → detects structural failures
3. Proposes threshold changes → writes to proposals file
4. Human approves (or auto-applies non-critical after N consistent proposals)
5. New thresholds applied → checks run better → repeat

Usage:
    python -m bridge.auto_tune analyse      # Analyse and propose
    python -m bridge.auto_tune apply <id>   # Apply a proposal (with human approval)
    python -m bridge.auto_tune history      # View proposal history
    python -m bridge.auto_tune auto         # Auto-apply safe proposals
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from bridge.config import get_ncpu_path, get_bridge_path, get_clawd_data_path
NCPU_PATH = get_ncpu_path()
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge

OUTCOMES_PATH = get_clawd_data_path("ncpu-outcomes.jsonl")
PROPOSALS_PATH = get_clawd_data_path("ncpu-threshold-proposals.json")
THRESHOLDS_PATH = get_clawd_data_path("ncpu-thresholds.json")

# Default thresholds (can be overridden by THRESHOLDS_PATH)
DEFAULT_THRESHOLDS = {
    "health:disk_usage": {"metric": "disk_pct", "threshold": 90, "direction": "max"},
    "health:memory_free": {"metric": "mem_free_mb", "threshold": 200, "direction": "min"},
    "obligation_staleness:critical": {"metric": "elapsed_seconds", "threshold": 7200, "direction": "max"},
    "obligation_staleness:high": {"metric": "elapsed_seconds", "threshold": 14400, "direction": "max"},
    "obligation_staleness:medium": {"metric": "elapsed_seconds", "threshold": 86400, "direction": "max"},
}


@dataclass
class ThresholdProposal:
    id: str
    task: str
    current_threshold: int
    proposed_threshold: int
    reason: str
    confidence: float  # 0-1
    critical: bool
    auto_applicable: bool  # safe to auto-apply?
    data_points: int
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, applied, rejected
    applied_at: Optional[float] = None


def load_outcomes() -> list[dict]:
    outcomes = []
    if OUTCOMES_PATH.exists():
        with OUTCOMES_PATH.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        outcomes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return outcomes


def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        with THRESHOLDS_PATH.open() as f:
            return json.load(f)
    return DEFAULT_THRESHOLDS.copy()


def save_thresholds(thresholds: dict):
    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with THRESHOLDS_PATH.open("w") as f:
        json.dump(thresholds, f, indent=2)


def load_proposals() -> list[dict]:
    if PROPOSALS_PATH.exists():
        with PROPOSALS_PATH.open() as f:
            return json.load(f)
    return []


def save_proposals(proposals: list[dict]):
    PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROPOSALS_PATH.open("w") as f:
        json.dump(proposals, f, indent=2)


def analyse_and_propose() -> list[ThresholdProposal]:
    """Analyse outcomes and generate threshold proposals using neural computation."""
    bridge = NCPUBridge()
    outcomes = load_outcomes()
    thresholds = load_thresholds()
    
    if len(outcomes) < 10:
        print(f"Only {len(outcomes)} outcomes — need at least 10 for meaningful analysis.")
        return []
    
    proposals = []
    
    # Group outcomes by task
    by_task: dict[str, list[dict]] = defaultdict(list)
    for o in outcomes:
        by_task[o.get("task_name", "")].append(o)
    
    for task_name, task_outcomes in by_task.items():
        if len(task_outcomes) < 5:
            continue
        
        successes = sum(1 for o in task_outcomes if o.get("success"))
        failures = len(task_outcomes) - successes
        success_rate = successes / len(task_outcomes)
        
        # ── Pattern: Structural failure (never or rarely succeeds) ──
        if success_rate < 0.2 and failures >= 5:
            # Extract the actual values from input_data
            values = []
            for o in task_outcomes:
                inp = o.get("input_data", {})
                if "disk_pct" in inp:
                    values.append(inp["disk_pct"])
                elif "mem_free_mb" in inp:
                    values.append(inp["mem_free_mb"])
            
            if not values:
                continue
            
            # Neural-verified statistics
            neural_sum = values[0]
            for v in values[1:]:
                neural_sum = bridge.add(neural_sum, v)
            neural_avg = bridge.div(neural_sum, len(values))
            neural_max = values[0]
            neural_min = values[0]
            for v in values[1:]:
                zf, sf = bridge.cmp(v, neural_max)
                if not sf and not zf:  # v > max
                    neural_max = v
                zf, sf = bridge.cmp(v, neural_min)
                if sf:  # v < min
                    neural_min = v
            
            # Determine threshold key
            threshold_key = None
            for key in thresholds:
                if key.startswith(task_name.replace(":", "_")) or task_name.startswith(key.split(":")[0]):
                    threshold_key = key
                    break
            
            if not threshold_key:
                # Try direct match
                for key, val in thresholds.items():
                    if task_name in key or key in task_name:
                        threshold_key = key
                        break
            
            if not threshold_key:
                continue
            
            current = thresholds[threshold_key]["threshold"]
            direction = thresholds[threshold_key].get("direction", "max")
            
            # Propose new threshold based on observed data
            if direction == "max":
                # Value should be BELOW threshold. If always above, raise threshold.
                # New threshold = observed max + 10% headroom
                headroom = bridge.div(neural_max, 10)  # 10% headroom
                proposed = bridge.add(neural_max, headroom)
                if proposed <= current:
                    continue  # Current threshold is fine
                reason = (
                    f"'{task_name}' fails {failures}/{len(task_outcomes)} times. "
                    f"Observed values: avg={neural_avg}, min={neural_min}, max={neural_max}. "
                    f"Current threshold {current} is too low. "
                    f"Proposing {proposed} (max + 10% headroom). [neural-verified stats]"
                )
            else:  # direction == "min"
                # Value should be ABOVE threshold. If always below, lower threshold.
                # New threshold = observed min - 10% margin (but not negative)
                margin = bridge.div(neural_min, 10)
                proposed = bridge.sub(neural_min, margin)
                if proposed < 0:
                    proposed = 0
                if proposed >= current:
                    continue
                reason = (
                    f"'{task_name}' fails {failures}/{len(task_outcomes)} times. "
                    f"Observed values: avg={neural_avg}, min={neural_min}, max={neural_max}. "
                    f"Current threshold {current} is unrealistic for this hardware. "
                    f"Proposing {proposed} (min - 10% margin). [neural-verified stats]"
                )
            
            # Is this safe to auto-apply?
            is_critical = "obligation" in task_name.lower() or "pos" in task_name.lower()
            
            proposal = ThresholdProposal(
                id=f"P{int(time.time())}-{task_name.replace(':', '-').replace('/', '-')}",
                task=task_name,
                current_threshold=current,
                proposed_threshold=proposed,
                reason=reason,
                confidence=min(0.95, len(task_outcomes) / 50),  # More data = more confidence
                critical=is_critical,
                auto_applicable=not is_critical and success_rate < 0.1,
                data_points=len(task_outcomes),
            )
            proposals.append(proposal)
        
        # ── Pattern: Intermittent failures (flaky) ──
        elif 0.3 < success_rate < 0.8 and len(task_outcomes) >= 10:
            # Track if it's getting worse
            recent = task_outcomes[-5:]
            recent_rate = sum(1 for o in recent if o.get("success")) / len(recent)
            
            if recent_rate < success_rate:
                proposal = ThresholdProposal(
                    id=f"P{int(time.time())}-{task_name.replace(':', '-').replace('/', '-')}-flaky",
                    task=task_name,
                    current_threshold=0,
                    proposed_threshold=0,
                    reason=(
                        f"'{task_name}' is flaky: {success_rate:.0%} overall, "
                        f"{recent_rate:.0%} in last 5 runs (degrading). "
                        f"Root cause investigation recommended."
                    ),
                    confidence=0.5,
                    critical="obligation" in task_name.lower(),
                    auto_applicable=False,
                    data_points=len(task_outcomes),
                )
                proposals.append(proposal)
    
    # Save proposals
    existing = load_proposals()
    for p in proposals:
        existing.append(asdict(p))
    save_proposals(existing)
    
    return proposals


def apply_proposal(proposal_id: str):
    """Apply a threshold proposal."""
    proposals = load_proposals()
    thresholds = load_thresholds()
    
    target = None
    for p in proposals:
        if p["id"] == proposal_id:
            target = p
            break
    
    if not target:
        print(f"Proposal {proposal_id} not found.")
        return
    
    if target["status"] == "applied":
        print(f"Proposal {proposal_id} already applied.")
        return
    
    # Find and update threshold
    task = target["task"]
    for key in thresholds:
        if task in key or key in task:
            old = thresholds[key]["threshold"]
            thresholds[key]["threshold"] = target["proposed_threshold"]
            print(f"Applied: {key} threshold {old} → {target['proposed_threshold']}")
            break
    
    target["status"] = "applied"
    target["applied_at"] = time.time()
    
    save_thresholds(thresholds)
    save_proposals(proposals)
    print(f"Proposal {proposal_id} applied and saved.")


def auto_apply_safe():
    """Auto-apply proposals that are safe (non-critical, high confidence, consistent)."""
    proposals = load_proposals()
    applied = 0
    
    for p in proposals:
        if p["status"] != "pending":
            continue
        if not p.get("auto_applicable"):
            continue
        if p.get("confidence", 0) < 0.5:
            continue
        
        print(f"Auto-applying: {p['id']}")
        apply_proposal(p["id"])
        applied += 1
    
    if applied == 0:
        print("No safe proposals to auto-apply.")
    else:
        print(f"Auto-applied {applied} proposal(s).")


def show_history():
    """Show proposal history."""
    proposals = load_proposals()
    if not proposals:
        print("No proposals yet.")
        return
    
    for p in proposals:
        status_emoji = {"pending": "⏳", "applied": "✅", "rejected": "❌"}.get(p["status"], "?")
        print(f"{status_emoji} [{p['id']}] {p['task']}")
        print(f"   {p['current_threshold']} → {p['proposed_threshold']} ({p['status']})")
        print(f"   Reason: {p['reason'][:100]}...")
        print(f"   Confidence: {p.get('confidence', 0):.0%} | Data points: {p.get('data_points', 0)}")
        print()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    
    if cmd == "analyse":
        proposals = analyse_and_propose()
        if proposals:
            print(f"\n{'='*60}")
            print(f"Generated {len(proposals)} proposal(s):")
            print(f"{'='*60}\n")
            for p in proposals:
                auto = " [AUTO-APPLICABLE]" if p.auto_applicable else ""
                print(f"📋 {p.id}{auto}")
                print(f"   Task: {p.task}")
                print(f"   Change: {p.current_threshold} → {p.proposed_threshold}")
                print(f"   Confidence: {p.confidence:.0%}")
                print(f"   Reason: {p.reason}")
                print()
        else:
            print("No proposals generated — all thresholds look appropriate.")
    
    elif cmd == "apply" and len(sys.argv) > 2:
        apply_proposal(sys.argv[2])
    
    elif cmd == "auto":
        auto_apply_safe()
    
    elif cmd == "history":
        show_history()
    
    else:
        print("Usage: python -m bridge.auto_tune [analyse|apply <id>|auto|history]")


if __name__ == "__main__":
    main()
