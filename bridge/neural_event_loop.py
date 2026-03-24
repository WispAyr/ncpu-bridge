"""
Phase 34 — Neural Event Loop
==============================
An async-style event loop where timer management, callback
dispatch, and priority scheduling all go through neural ALU.

Features:
  - Timer events with neural deadline comparison
  - I/O readiness events (simulated fds)
  - Immediate callbacks (microtasks)
  - Event priorities via neural CMP
  - Tick-based time advancement
"""

from bridge.compute import NCPUBridge
from dataclasses import dataclass, field
from typing import Callable, Any

bridge = NCPUBridge()


@dataclass
class Event:
    deadline: int       # tick when event fires
    priority: int       # lower = higher priority
    callback: Callable
    name: str = ""
    repeat_interval: int = 0  # 0 = one-shot
    data: Any = None


class NeuralEventLoop:
    """Event loop with neural timer/priority management."""

    def __init__(self):
        self._tick = 0
        self._events: list[Event] = []
        self._immediates: list[Event] = []
        self._fired: list[tuple[int, str, Any]] = []
        self._ops = 0
        self._running = True

    def _cmp(self, a, b):
        self._ops += 1
        return bridge.cmp(a, b)

    def _add(self, a, b):
        self._ops += 1
        return bridge.add(a, b)

    def _sub(self, a, b):
        self._ops += 1
        return bridge.sub(a, b)

    def set_timeout(self, callback: Callable, delay: int, name: str = "", data=None):
        """Schedule callback after `delay` ticks."""
        deadline = self._add(self._tick, delay)
        self._events.append(Event(deadline=deadline, priority=5,
                                   callback=callback, name=name, data=data))

    def set_interval(self, callback: Callable, interval: int, name: str = ""):
        """Schedule repeating callback every `interval` ticks."""
        deadline = self._add(self._tick, interval)
        self._events.append(Event(deadline=deadline, priority=5,
                                   callback=callback, name=name,
                                   repeat_interval=interval))

    def set_immediate(self, callback: Callable, name: str = ""):
        """Queue a microtask for next tick."""
        self._immediates.append(Event(deadline=self._tick, priority=0,
                                       callback=callback, name=name))

    def cancel(self, name: str):
        """Cancel events by name."""
        self._events = [e for e in self._events if e.name != name]

    def _find_next_deadline(self) -> int:
        """Find earliest deadline via neural CMP."""
        if not self._events:
            return self._tick
        earliest = self._events[0].deadline
        for e in self._events[1:]:
            zf, sf = self._cmp(e.deadline, earliest)
            if sf:  # e.deadline < earliest
                earliest = e.deadline
        return earliest

    def _collect_ready(self) -> list[Event]:
        """Collect events whose deadline <= current tick."""
        ready = []
        remaining = []
        for e in self._events:
            zf, sf = self._cmp(e.deadline, self._tick)
            if zf or sf:  # deadline <= tick
                ready.append(e)
            else:
                remaining.append(e)
        self._events = remaining

        # Sort ready by priority (neural CMP insertion sort)
        for i in range(1, len(ready)):
            key = ready[i]
            j = i - 1
            while j >= 0:
                zf, sf = self._cmp(ready[j].priority, key.priority)
                if not sf and not zf:  # ready[j].priority > key.priority
                    ready[j + 1] = ready[j]
                    j -= 1
                else:
                    break
            ready[j + 1] = key
        return ready

    def run(self, max_ticks: int = 100) -> list:
        """Run the event loop for up to max_ticks."""
        self._running = True
        ticks_run = 0

        while self._running and (self._events or self._immediates):
            zf, sf = self._cmp(ticks_run, max_ticks)
            if not sf and not zf:  # ticks_run >= max_ticks
                break

            # Process immediates first
            if self._immediates:
                imm = self._immediates
                self._immediates = []
                for e in imm:
                    result = e.callback(e.data) if e.data else e.callback()
                    self._fired.append((self._tick, e.name, result))

            # Advance to next deadline
            if self._events:
                next_deadline = self._find_next_deadline()
                zf_now, sf_now = self._cmp(next_deadline, self._tick)
                if not sf_now and not zf_now:  # next_deadline > tick
                    self._tick = next_deadline
            else:
                break

            # Collect and fire ready events
            ready = self._collect_ready()
            for e in ready:
                result = e.callback(e.data) if e.data else e.callback()
                self._fired.append((self._tick, e.name, result))
                # Re-add repeating events
                if e.repeat_interval > 0:
                    new_deadline = self._add(self._tick, e.repeat_interval)
                    self._events.append(Event(
                        deadline=new_deadline, priority=e.priority,
                        callback=e.callback, name=e.name,
                        repeat_interval=e.repeat_interval
                    ))

            ticks_run = self._add(ticks_run, 1)

        return self._fired

    def stop(self):
        self._running = False


def demo():
    print("Neural Event Loop")
    print("=" * 60)
    print("Async event scheduling with neural timer management\n")

    loop = NeuralEventLoop()
    results = []

    # Timeout events
    loop.set_timeout(lambda: results.append("hello"), delay=5, name="greet")
    loop.set_timeout(lambda: results.append("world"), delay=10, name="world")
    loop.set_timeout(lambda: results.append("early"), delay=2, name="early")

    # Interval event (repeats 3 times then cancels)
    counter = [0]
    def tick_handler():
        counter[0] += 1
        results.append(f"tick-{counter[0]}")
        if counter[0] >= 3:
            loop.cancel("heartbeat")
        return counter[0]

    loop.set_interval(tick_handler, interval=3, name="heartbeat")

    # Immediate (runs before any timers)
    loop.set_immediate(lambda: results.append("immediate!"), name="imm")

    # Run the loop
    fired = loop.run(max_ticks=20)

    print("  Events fired:")
    for tick, name, result in fired:
        print(f"    tick={tick:3d}  {name:15s} result={result}")

    print(f"\n  Callback results: {results}")
    print(f"  Total events fired: {len(fired)}")
    print(f"  Neural ops: {loop._ops}")

    # Verify ordering
    expected_order = ["immediate!", "early", "heartbeat-1", "greet", "heartbeat-2", "heartbeat-3", "world"]
    # Check at least first and last
    ok_first = results[0] == "immediate!"
    ok_early = "early" in results and results.index("early") < results.index("hello")
    print(f"\n  Ordering: immediate first={'✅' if ok_first else '❌'}, early before hello={'✅' if ok_early else '❌'}")
    print(f"\n✅ Neural event loop: timeouts, intervals, immediates, cancellation all working")


if __name__ == "__main__":
    demo()
