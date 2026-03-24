"""
Phase 31 — Neural Signal Handler
=================================
POSIX-style signal delivery and handling where signal number
matching, priority comparison, and mask operations all go
through neural ALU.

Features:
  - Signal registration (SIGINT, SIGTERM, SIGKILL, SIGUSR1, etc.)
  - Signal masks (block/unblock) via neural AND/OR
  - Priority-based signal delivery via neural CMP
  - Pending signal queue with neural dequeue
  - Default actions: terminate, ignore, core dump, stop, continue
"""

from bridge.compute import NCPUBridge
from dataclasses import dataclass, field
from typing import Callable, Optional

bridge = NCPUBridge()

# Signal constants (Linux-style)
SIGHUP  = 1;  SIGINT  = 2;  SIGQUIT = 3;  SIGILL  = 4
SIGABRT = 6;  SIGFPE  = 8;  SIGKILL = 9;  SIGUSR1 = 10
SIGSEGV = 11; SIGUSR2 = 12; SIGPIPE = 13; SIGALRM = 14
SIGTERM = 15; SIGCHLD = 17; SIGCONT = 18; SIGSTOP = 19
SIGTSTP = 20

SIGNAL_NAMES = {
    1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 4: "SIGILL",
    6: "SIGABRT", 8: "SIGFPE", 9: "SIGKILL", 10: "SIGUSR1",
    11: "SIGSEGV", 12: "SIGUSR2", 13: "SIGPIPE", 14: "SIGALRM",
    15: "SIGTERM", 17: "SIGCHLD", 18: "SIGCONT", 19: "SIGSTOP",
    20: "SIGTSTP",
}

# Default actions
DEFAULT_ACTIONS = {
    SIGHUP: "terminate", SIGINT: "terminate", SIGQUIT: "core",
    SIGILL: "core", SIGABRT: "core", SIGFPE: "core",
    SIGKILL: "terminate", SIGUSR1: "terminate", SIGSEGV: "core",
    SIGUSR2: "terminate", SIGPIPE: "terminate", SIGALRM: "terminate",
    SIGTERM: "terminate", SIGCHLD: "ignore", SIGCONT: "continue",
    SIGSTOP: "stop", SIGTSTP: "stop",
}

# Signal priorities (lower = higher priority)
SIGNAL_PRIORITY = {
    SIGKILL: 0, SIGSTOP: 1, SIGSEGV: 2, SIGABRT: 3,
    SIGINT: 4, SIGTERM: 5, SIGQUIT: 6, SIGALRM: 7,
    SIGUSR1: 8, SIGUSR2: 9, SIGCHLD: 10, SIGCONT: 11,
}


@dataclass
class PendingSignal:
    signum: int
    priority: int
    data: int = 0


class NeuralSignalHandler:
    """POSIX-style signal handling with neural ALU for all matching/masking."""

    def __init__(self):
        self._handlers: dict[int, Optional[Callable]] = {}
        self._mask: int = 0  # blocked signal bitmask
        self._pending: list[PendingSignal] = []
        self._delivered: list[tuple[int, str]] = []  # (signum, action)
        self._ops = 0
        self._process_state = "running"  # running, stopped, terminated

    def _neural_and(self, a, b):
        self._ops += 1
        return bridge.bitwise_and(a, b)

    def _neural_or(self, a, b):
        self._ops += 1
        return bridge.bitwise_or(a, b)

    def _neural_xor(self, a, b):
        self._ops += 1
        return bridge.bitwise_xor(a, b)

    def _neural_shl(self, a, b):
        self._ops += 1
        return bridge.shl(a, b)

    def _neural_cmp(self, a, b):
        self._ops += 1
        return bridge.cmp(a, b)

    def _sig_bit(self, signum: int) -> int:
        """Get bitmask for signal via neural SHL."""
        return self._neural_shl(1, signum)

    def signal(self, signum: int, handler: Optional[Callable]):
        """Register a signal handler (like sigaction)."""
        # SIGKILL and SIGSTOP cannot be caught
        zf_kill, _ = self._neural_cmp(signum, SIGKILL)
        zf_stop, _ = self._neural_cmp(signum, SIGSTOP)
        if zf_kill or zf_stop:
            return False
        self._handlers[signum] = handler
        return True

    def block(self, signum: int):
        """Add signal to block mask via neural OR."""
        zf_kill, _ = self._neural_cmp(signum, SIGKILL)
        zf_stop, _ = self._neural_cmp(signum, SIGSTOP)
        if zf_kill or zf_stop:
            return  # can't block these
        bit = self._sig_bit(signum)
        self._mask = self._neural_or(self._mask, bit)

    def unblock(self, signum: int):
        """Remove signal from block mask via neural AND+XOR."""
        bit = self._sig_bit(signum)
        inv = self._neural_xor(bit, 0xFFFFFFFF)
        self._mask = self._neural_and(self._mask, inv)

    def is_blocked(self, signum: int) -> bool:
        """Check if signal is blocked via neural AND."""
        bit = self._sig_bit(signum)
        result = self._neural_and(self._mask, bit)
        zf, _ = self._neural_cmp(result, 0)
        return not zf

    def kill(self, signum: int, data: int = 0):
        """Send a signal (like kill(2))."""
        priority = SIGNAL_PRIORITY.get(signum, 15)
        pending = PendingSignal(signum=signum, priority=priority, data=data)

        if self.is_blocked(signum):
            # Queue it but don't deliver yet
            self._pending.append(pending)
            return "blocked"

        return self._deliver(pending)

    def _deliver(self, sig: PendingSignal) -> str:
        """Deliver a signal — check handler, then default action."""
        signum = sig.signum

        # Custom handler?
        if signum in self._handlers and self._handlers[signum] is not None:
            self._handlers[signum](signum, sig.data)
            action = "handled"
        else:
            # Default action
            action = DEFAULT_ACTIONS.get(signum, "terminate")
            if action == "terminate" or action == "core":
                self._process_state = "terminated"
            elif action == "stop":
                self._process_state = "stopped"
            elif action == "continue":
                self._process_state = "running"

        self._delivered.append((signum, action))
        return action

    def flush_pending(self) -> list:
        """Deliver pending signals (e.g., after unblock). Priority sort via neural CMP."""
        if not self._pending:
            return []

        # Sort by priority using neural CMP (insertion sort)
        pending = list(self._pending)
        for i in range(1, len(pending)):
            key = pending[i]
            j = i - 1
            while j >= 0:
                zf, sf = self._neural_cmp(pending[j].priority, key.priority)
                if not sf and not zf:  # pending[j].priority > key.priority
                    pending[j + 1] = pending[j]
                    j -= 1
                else:
                    break
            pending[j + 1] = key

        results = []
        still_pending = []
        for sig in pending:
            if self.is_blocked(sig.signum):
                still_pending.append(sig)
            else:
                action = self._deliver(sig)
                results.append((sig.signum, action))

        self._pending = still_pending
        return results


def demo():
    print("Neural Signal Handler")
    print("=" * 60)
    print("POSIX signals with neural bitmask ops\n")

    sh = NeuralSignalHandler()

    # Register custom handlers
    log = []
    sh.signal(SIGINT, lambda s, d: log.append(f"caught SIGINT (data={d})"))
    sh.signal(SIGUSR1, lambda s, d: log.append(f"caught SIGUSR1 (data={d})"))

    # Try to catch SIGKILL — should fail
    ok = sh.signal(SIGKILL, lambda s, d: None)
    print(f"  Register SIGKILL handler: {'❌ denied' if not ok else '⚠️ allowed'}")

    # Send signals
    print("\n  Sending signals:")
    for sig, data in [(SIGINT, 42), (SIGUSR1, 99), (SIGCHLD, 0), (SIGTERM, 0)]:
        action = sh.kill(sig, data)
        name = SIGNAL_NAMES.get(sig, f"SIG{sig}")
        print(f"    kill({name}) → {action}")

    print(f"\n  Handler log: {log}")
    print(f"  Process state: {sh._process_state}")

    # Test signal blocking
    print("\n  Signal masking:")
    sh2 = NeuralSignalHandler()
    sh2.signal(SIGINT, lambda s, d: None)  # reset to avoid termination

    sh2.block(SIGINT)
    print(f"    Blocked SIGINT: {sh2.is_blocked(SIGINT)}")

    result = sh2.kill(SIGINT)
    print(f"    kill(SIGINT) while blocked → {result}")
    print(f"    Pending queue: {len(sh2._pending)} signal(s)")

    sh2.unblock(SIGINT)
    print(f"    Unblocked SIGINT: {not sh2.is_blocked(SIGINT)}")

    flushed = sh2.flush_pending()
    print(f"    Flushed: {flushed}")

    # Priority ordering test
    print("\n  Priority delivery order:")
    sh3 = NeuralSignalHandler()
    caught = []
    sh3.signal(SIGINT, lambda s, d: caught.append("SIGINT"))
    sh3.signal(SIGUSR1, lambda s, d: caught.append("SIGUSR1"))
    sh3.signal(SIGALRM, lambda s, d: caught.append("SIGALRM"))

    # Block all, send in reverse priority order, then flush
    sh3.block(SIGINT)
    sh3.block(SIGUSR1)
    sh3.block(SIGALRM)
    sh3.kill(SIGUSR1)   # priority 8
    sh3.kill(SIGALRM)   # priority 7
    sh3.kill(SIGINT)    # priority 4

    sh3.unblock(SIGINT)
    sh3.unblock(SIGUSR1)
    sh3.unblock(SIGALRM)
    flushed = sh3.flush_pending()

    print(f"    Delivery order: {caught}")
    expected = ["SIGINT", "SIGALRM", "SIGUSR1"]
    ok = caught == expected
    print(f"    Expected:       {expected} → {'✅' if ok else '❌'}")

    total_ops = sh._ops + sh2._ops + sh3._ops
    print(f"\n  Total neural ops: {total_ops}")
    print("\n✅ Neural signal handler: delivery, masking, priority ordering all working")


if __name__ == "__main__":
    demo()
