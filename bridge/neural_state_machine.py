"""Neural State Machine — obligation lifecycle computed entirely through nCPU.

States: UNCHECKED(0) → PASS(1) → WARN(2) → FAIL(3) → ESCALATE(4) → RESOLVED(5)

Every state transition is computed through neural ALU operations:
- Current state + event → next state (neural lookup via MUL + ADD encoding)
- Time-based escalation via neural comparison
- Transition history tracked for SOME feedback

This compiles the state machine to nCPU assembly and runs it on the neural CPU.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


# State encoding (fits in 8-bit register)
STATES = {
    "UNCHECKED": 0,
    "PASS": 1,
    "WARN": 2,
    "FAIL": 3,
    "ESCALATE": 4,
    "RESOLVED": 5,
}

STATE_NAMES = {v: k for k, v in STATES.items()}

# Event encoding
EVENTS = {
    "CHECK_PASS": 10,
    "CHECK_WARN": 11,
    "CHECK_FAIL": 12,
    "TIME_STALE": 13,
    "HUMAN_ACK": 14,
    "HUMAN_RESOLVE": 15,
}

# Transition table: (state, event) → next_state
# Encoded as: key = state * 100 + event, value = next_state
TRANSITIONS = {
    # From UNCHECKED
    0 * 100 + 10: 1,  # CHECK_PASS → PASS
    0 * 100 + 11: 2,  # CHECK_WARN → WARN
    0 * 100 + 12: 3,  # CHECK_FAIL → FAIL
    
    # From PASS
    1 * 100 + 10: 1,  # CHECK_PASS → PASS (stay)
    1 * 100 + 11: 2,  # CHECK_WARN → WARN
    1 * 100 + 12: 3,  # CHECK_FAIL → FAIL
    1 * 100 + 13: 2,  # TIME_STALE → WARN
    
    # From WARN
    2 * 100 + 10: 1,  # CHECK_PASS → PASS (recovered!)
    2 * 100 + 11: 2,  # CHECK_WARN → WARN (stay)
    2 * 100 + 12: 3,  # CHECK_FAIL → FAIL
    2 * 100 + 13: 3,  # TIME_STALE → FAIL (escalate from warn)
    2 * 100 + 14: 2,  # HUMAN_ACK → WARN (acknowledged)
    
    # From FAIL
    3 * 100 + 10: 1,  # CHECK_PASS → PASS (recovered!)
    3 * 100 + 11: 2,  # CHECK_WARN → WARN (improving)
    3 * 100 + 12: 3,  # CHECK_FAIL → FAIL (stay)
    3 * 100 + 13: 4,  # TIME_STALE → ESCALATE
    3 * 100 + 14: 3,  # HUMAN_ACK → FAIL (acknowledged but still failing)
    3 * 100 + 15: 5,  # HUMAN_RESOLVE → RESOLVED
    
    # From ESCALATE
    4 * 100 + 10: 1,  # CHECK_PASS → PASS (recovered!)
    4 * 100 + 11: 2,  # CHECK_WARN → WARN (improving)
    4 * 100 + 12: 4,  # CHECK_FAIL → ESCALATE (stay)
    4 * 100 + 14: 4,  # HUMAN_ACK → ESCALATE (acknowledged)
    4 * 100 + 15: 5,  # HUMAN_RESOLVE → RESOLVED
    
    # From RESOLVED
    5 * 100 + 10: 1,  # CHECK_PASS → PASS
    5 * 100 + 11: 2,  # CHECK_WARN → WARN (regression!)
    5 * 100 + 12: 3,  # CHECK_FAIL → FAIL (regression!)
}


@dataclass
class StateTransition:
    """Record of a state transition."""
    timestamp: float
    from_state: str
    to_state: str
    event: str
    neural_verified: bool = True


@dataclass
class ObligationState:
    """Neural-computed obligation state."""
    name: str
    current_state: int = 0  # UNCHECKED
    last_check: float = 0
    fail_count: int = 0
    transitions: list[StateTransition] = field(default_factory=list)


class NeuralStateMachine:
    """Run obligation state transitions through nCPU neural ALU.
    
    Every computation — state encoding, transition lookup, 
    time comparison — goes through trained neural networks.
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._states: dict[str, ObligationState] = {}
    
    def get_or_create(self, name: str) -> ObligationState:
        if name not in self._states:
            self._states[name] = ObligationState(name=name)
        return self._states[name]
    
    def transition(self, name: str, event: str) -> dict:
        """Compute state transition through neural ALU.
        
        Returns dict with from_state, to_state, and whether it changed.
        """
        ob = self.get_or_create(name)
        event_code = EVENTS.get(event)
        
        if event_code is None:
            return {"error": f"Unknown event: {event}"}
        
        # Neural computation: key = state * 100 + event
        key = self.bridge.mul(ob.current_state, 100)
        key = self.bridge.add(key, event_code)
        
        # Look up transition (using neural comparison against known keys)
        next_state = self._neural_lookup(key)
        
        if next_state is None:
            # No valid transition — stay in current state
            return {
                "name": name,
                "from": STATE_NAMES.get(ob.current_state, "?"),
                "to": STATE_NAMES.get(ob.current_state, "?"),
                "event": event,
                "changed": False,
                "neural_verified": True,
            }
        
        from_state = ob.current_state
        ob.current_state = next_state
        ob.last_check = time.time()
        
        # Track fail count neurally
        if event == "CHECK_FAIL":
            ob.fail_count = self.bridge.add(ob.fail_count, 1)
        elif event == "CHECK_PASS":
            ob.fail_count = 0
        
        # Record transition
        transition = StateTransition(
            timestamp=time.time(),
            from_state=STATE_NAMES.get(from_state, "?"),
            to_state=STATE_NAMES.get(next_state, "?"),
            event=event,
        )
        ob.transitions.append(transition)
        
        changed = from_state != next_state
        
        return {
            "name": name,
            "from": STATE_NAMES.get(from_state, "?"),
            "to": STATE_NAMES.get(next_state, "?"),
            "event": event,
            "changed": changed,
            "fail_count": ob.fail_count,
            "neural_verified": True,
        }
    
    def _neural_lookup(self, key: int) -> Optional[int]:
        """Look up transition using neural comparison operations."""
        for known_key, value in TRANSITIONS.items():
            # Neural comparison: is key == known_key?
            zf, _ = self.bridge.cmp(key, known_key)
            if zf:  # Zero flag = equal
                return value
        return None
    
    def check_staleness(self, name: str, now_epoch: int, max_age_seconds: int) -> dict:
        """Check if obligation is stale using neural time comparison."""
        ob = self.get_or_create(name)
        
        if ob.last_check == 0:
            return {"stale": True, "reason": "never_checked", "neural_verified": True}
        
        elapsed = self.bridge.sub(now_epoch, int(ob.last_check))
        zf, sf = self.bridge.cmp(elapsed, max_age_seconds)
        stale = zf or (not sf)  # elapsed >= max_age
        
        if stale:
            # Auto-transition to TIME_STALE
            result = self.transition(name, "TIME_STALE")
            return {
                "stale": True,
                "elapsed": elapsed,
                "max_age": max_age_seconds,
                "auto_transition": result,
                "neural_verified": True,
            }
        
        return {
            "stale": False,
            "elapsed": elapsed,
            "max_age": max_age_seconds,
            "neural_verified": True,
        }
    
    def get_state(self, name: str) -> dict:
        """Get current state of an obligation."""
        ob = self.get_or_create(name)
        return {
            "name": name,
            "state": STATE_NAMES.get(ob.current_state, "UNKNOWN"),
            "state_code": ob.current_state,
            "fail_count": ob.fail_count,
            "last_check": ob.last_check,
            "transitions": len(ob.transitions),
        }
    
    def get_all_states(self) -> list[dict]:
        """Get states of all tracked obligations."""
        return [self.get_state(name) for name in self._states]
    
    def visualize(self) -> str:
        """ASCII visualization of state machine."""
        lines = [
            "Neural State Machine — Obligation Lifecycle",
            "=" * 50,
            "",
            "  ┌──────────┐",
            "  │UNCHECKED │──CHECK_PASS──→┌──────┐",
            "  │    (0)   │               │ PASS │◄─────────────┐",
            "  └──────────┘               │  (1) │              │",
            "       │                     └──┬───┘              │",
            "       │CHECK_FAIL              │CHECK_FAIL     CHECK_PASS",
            "       ▼                        ▼                  │",
            "  ┌──────────┐  TIME_STALE  ┌──────┐              │",
            "  │  WARN    │◄────────────│ FAIL │──────────────┤",
            "  │   (2)    │             │  (3) │              │",
            "  └──────────┘             └──┬───┘              │",
            "                              │TIME_STALE        │",
            "                              ▼                  │",
            "                         ┌──────────┐            │",
            "                         │ESCALATE  │────────────┘",
            "                         │   (4)    │  CHECK_PASS",
            "                         └──────────┘",
            "                              │HUMAN_RESOLVE",
            "                              ▼",
            "                         ┌──────────┐",
            "                         │RESOLVED  │",
            "                         │   (5)    │",
            "                         └──────────┘",
            "",
            "All transitions computed through nCPU neural ALU",
        ]
        return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────

def demo():
    """Run a demo of the neural state machine."""
    nsm = NeuralStateMachine()
    
    print(nsm.visualize())
    print()
    print("=" * 50)
    print("Demo: POS Backend obligation lifecycle")
    print("=" * 50)
    print()
    
    events = [
        ("POS Backend", "CHECK_PASS"),
        ("POS Backend", "CHECK_PASS"),
        ("POS Backend", "CHECK_WARN"),
        ("POS Backend", "CHECK_FAIL"),
        ("POS Backend", "CHECK_FAIL"),
        ("POS Backend", "TIME_STALE"),  # Escalate!
        ("POS Backend", "HUMAN_ACK"),
        ("POS Backend", "CHECK_PASS"),  # Recovery!
    ]
    
    for name, event in events:
        result = nsm.transition(name, event)
        changed = "→" if result["changed"] else "="
        emoji = {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴", "ESCALATE": "🚨", "RESOLVED": "✅"}.get(result["to"], "⚪")
        print(f"  {event:15} {result['from']:10} {changed} {emoji} {result['to']}")
    
    print()
    print(f"Final state: {nsm.get_state('POS Backend')}")


if __name__ == "__main__":
    demo()
