"""SOME (Self-Optimizing Machine Engine) feedback loop for Skynet.

Records task outcomes, builds trajectory data, and enables gradient-based
self-improvement over time. Every execution result feeds back into the system.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import sys
import numpy as np

from bridge.config import get_ncpu_path, get_bridge_path, get_clawd_data_path
NCPU_PATH = get_ncpu_path()
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from ncpu.self_optimizing.engine import SelfOptimizingEngine, Task, ExecutionResult
from ncpu.self_optimizing.gradient_feedback import (
    GradientFeedbackSystem,
    ExecutionSignal,
    FeedbackType,
)
from ncpu.self_optimizing.trajectory_logger import TrajectoryLogger
from ncpu.self_optimizing.ncpu_adaptation_backend import (
    NCPUAdaptationBackend,
    NCPUAdaptationConfig,
)


# ── Skynet Task Categories ────────────────────────────────────

TASK_CATEGORIES = {
    "obligation_check": {"weight": 1.0, "critical": True},
    "health_check": {"weight": 0.8, "critical": True},
    "heartbeat": {"weight": 0.5, "critical": False},
    "computation": {"weight": 1.0, "critical": False},
    "verification": {"weight": 1.2, "critical": True},
}


@dataclass
class TaskOutcome:
    """Record of a task execution with neural-verified result."""

    task_name: str
    category: str
    success: bool
    neural_verified: bool = False
    execution_time_ms: float = 0.0
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class SkynetFeedbackLoop:
    """Self-improvement feedback loop using SOME + nCPU.

    Records task outcomes, generates gradient signals from execution results,
    and maintains trajectory logs for future self-optimization.

    Flow:
    1. Task executes (obligation check, health check, etc.)
    2. Result recorded as TaskOutcome
    3. Gradient feedback generated (success → reinforce, failure → adapt)
    4. Trajectory logged for later distillation
    5. Adaptation backend suggests parameter adjustments
    """

    def __init__(
        self,
        trajectory_path: str = "",
        outcomes_path: str = "",
    ):
        self.trajectory_logger = TrajectoryLogger(path=trajectory_path)
        self.outcomes_path = Path(outcomes_path)
        self.outcomes_path.parent.mkdir(parents=True, exist_ok=True)

        self.feedback = GradientFeedbackSystem()
        self.adaptation = NCPUAdaptationBackend(
            NCPUAdaptationConfig(
                compression_type="top_k",
                top_k_ratio=0.1,
                max_gradient_steps=3,
                verify_failure_boost=1.2,
            )
        )

        # Session stats
        self._outcomes: list[TaskOutcome] = []
        self._session_start = time.time()

    def record_outcome(self, outcome: TaskOutcome) -> dict:
        """Record a task outcome and generate feedback signal.

        Returns adaptation suggestions based on the gradient signal.
        """
        self._outcomes.append(outcome)

        # Persist to JSONL
        with self.outcomes_path.open("a") as f:
            f.write(json.dumps(asdict(outcome)) + "\n")

        # Generate gradient signal
        signal = self._outcome_to_signal(outcome)

        # Generate adaptation suggestion
        category_info = TASK_CATEGORIES.get(outcome.category, {})
        suggestion = {
            "task": outcome.task_name,
            "category": outcome.category,
            "success": outcome.success,
            "neural_verified": outcome.neural_verified,
            "signal_strength": signal.get("strength", 0.0),
            "critical": category_info.get("critical", False),
        }

        # For failures on critical tasks, boost learning rate
        if not outcome.success and category_info.get("critical"):
            suggestion["action"] = "investigate"
            suggestion["urgency"] = "high"
            suggestion["reason"] = f"Critical task '{outcome.task_name}' failed"
        elif not outcome.success:
            suggestion["action"] = "log"
            suggestion["urgency"] = "low"
        else:
            suggestion["action"] = "reinforce"
            suggestion["urgency"] = "none"

        return suggestion

    def _outcome_to_signal(self, outcome: TaskOutcome) -> dict:
        """Convert a task outcome to a gradient-like signal."""
        category_info = TASK_CATEGORIES.get(outcome.category, {})
        weight = category_info.get("weight", 1.0)

        if outcome.success:
            strength = weight * 1.0
            direction = "reinforce"
        else:
            strength = weight * -1.0 * (1.5 if category_info.get("critical") else 1.0)
            direction = "correct"

        return {
            "strength": strength,
            "direction": direction,
            "category": outcome.category,
            "neural_verified": outcome.neural_verified,
            "execution_time_ms": outcome.execution_time_ms,
        }

    def get_session_stats(self) -> dict:
        """Get stats for the current feedback session."""
        if not self._outcomes:
            return {"total": 0, "success_rate": 0.0}

        successes = sum(1 for o in self._outcomes if o.success)
        neural = sum(1 for o in self._outcomes if o.neural_verified)
        critical_fails = sum(
            1 for o in self._outcomes
            if not o.success and TASK_CATEGORIES.get(o.category, {}).get("critical")
        )

        return {
            "total": len(self._outcomes),
            "successes": successes,
            "failures": len(self._outcomes) - successes,
            "success_rate": round(successes / len(self._outcomes) * 100, 1),
            "neural_verified": neural,
            "neural_rate": round(neural / len(self._outcomes) * 100, 1),
            "critical_failures": critical_fails,
            "session_duration_s": round(time.time() - self._session_start, 1),
            "avg_execution_ms": round(
                sum(o.execution_time_ms for o in self._outcomes) / len(self._outcomes), 1
            ),
        }

    def get_trend(self, last_n: int = 100) -> dict:
        """Analyze recent outcomes for performance trends.

        Reads from the persistent outcomes file.
        """
        outcomes = []
        if self.outcomes_path.exists():
            with self.outcomes_path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            outcomes.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        recent = outcomes[-last_n:] if outcomes else []
        if not recent:
            return {"trend": "no_data", "total_recorded": 0}

        # Split into halves for trend analysis
        mid = len(recent) // 2
        first_half = recent[:mid] if mid > 0 else recent
        second_half = recent[mid:] if mid > 0 else []

        first_rate = (
            sum(1 for o in first_half if o.get("success")) / len(first_half) * 100
            if first_half
            else 0
        )
        second_rate = (
            sum(1 for o in second_half if o.get("success")) / len(second_half) * 100
            if second_half
            else 0
        )

        if second_rate > first_rate + 5:
            trend = "improving"
        elif second_rate < first_rate - 5:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "total_recorded": len(outcomes),
            "recent_count": len(recent),
            "first_half_success_pct": round(first_rate, 1),
            "second_half_success_pct": round(second_rate, 1),
            "categories": list(set(o.get("category", "unknown") for o in recent)),
        }
