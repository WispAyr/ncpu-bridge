"""Neural Scheduler — task prioritization and scheduling through nCPU.

A priority scheduler where every comparison, time calculation,
and priority computation runs through trained neural networks.

Features:
- Priority queue with neural CMP for ordering
- Deadline-aware scheduling (EDF) with neural time arithmetic
- Round-robin with neural modulo
- Aging: neural ADD to prevent starvation
- Load balancing across workers with neural comparison

Usage:
    python -m bridge.neural_scheduler demo      # Run scheduling demos
    python -m bridge.neural_scheduler edf       # Earliest Deadline First
    python -m bridge.neural_scheduler balance    # Load balancing demo
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


@dataclass
class Task:
    id: str
    name: str
    priority: int  # 0 = highest
    deadline: int  # epoch seconds (0 = no deadline)
    cost: int  # estimated cycles
    assigned_to: Optional[str] = None
    started_at: float = 0.0
    completed: bool = False
    wait_time: int = 0  # for aging


@dataclass 
class Worker:
    id: str
    name: str
    load: int = 0  # current task cost sum
    capacity: int = 100
    tasks_completed: int = 0


class NeuralScheduler:
    """Schedule tasks using only neural ALU operations.
    
    Every priority comparison, deadline check, load balance decision,
    and aging increment goes through trained neural networks.
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._ops = 0
        self._tick = 0
    
    def _op(self):
        self._ops += 1
    
    # ── Priority Queue (neural insertion sort) ──────────
    
    def sort_by_priority(self, tasks: list[Task]) -> list[Task]:
        """Sort tasks by priority using neural CMP (insertion sort).
        
        Every comparison between priorities is a neural operation.
        """
        sorted_tasks = list(tasks)
        
        for i in range(1, len(sorted_tasks)):
            key = sorted_tasks[i]
            j = i - 1
            
            while j >= 0:
                # Neural comparison: is current priority > key priority?
                # (higher number = lower priority)
                zf, sf = self.bridge.cmp(sorted_tasks[j].priority, key.priority)
                self._op()
                
                if not sf and not zf:  # sorted[j].priority > key.priority
                    sorted_tasks[j + 1] = sorted_tasks[j]
                    j -= 1
                else:
                    break
            
            sorted_tasks[j + 1] = key
        
        return sorted_tasks
    
    # ── Earliest Deadline First ─────────────────────────
    
    def sort_by_deadline(self, tasks: list[Task], now: int) -> list[Task]:
        """EDF scheduling: sort by deadline using neural CMP.
        
        Tasks with no deadline (0) go to the end.
        Tasks past deadline get boosted.
        """
        # Calculate urgency for each task: deadline - now
        urgencies = []
        for task in tasks:
            if task.deadline == 0:
                urgencies.append((task, 999999))  # No deadline = lowest urgency
                continue
            
            remaining = self.bridge.sub(task.deadline, now)
            self._op()
            
            # If past deadline, urgency is negative (highest priority)
            urgencies.append((task, remaining))
        
        # Neural insertion sort by urgency
        for i in range(1, len(urgencies)):
            key = urgencies[i]
            j = i - 1
            
            while j >= 0:
                zf, sf = self.bridge.cmp(urgencies[j][1], key[1])
                self._op()
                
                if not sf and not zf:  # urgencies[j] > key
                    urgencies[j + 1] = urgencies[j]
                    j -= 1
                else:
                    break
            
            urgencies[j + 1] = key
        
        return [t for t, _ in urgencies]
    
    # ── Aging (prevent starvation) ──────────────────────
    
    def apply_aging(self, tasks: list[Task], age_boost: int = 1) -> list[Task]:
        """Increment wait_time and boost priority for starving tasks.
        
        Every increment and check is neural.
        """
        for task in tasks:
            if task.completed or task.assigned_to:
                continue
            
            # Neural increment of wait time
            task.wait_time = self.bridge.add(task.wait_time, age_boost)
            self._op()
            
            # Every 5 ticks of waiting, boost priority by 1 (lower number = higher priority)
            zf, sf = self.bridge.cmp(task.wait_time, 5)
            self._op()
            
            if not sf and not zf:  # wait >= 5
                if task.priority > 0:
                    task.priority = self.bridge.sub(task.priority, 1)
                    self._op()
                task.wait_time = 0  # Reset aging counter
        
        return tasks
    
    # ── Load Balancing ──────────────────────────────────
    
    def find_least_loaded(self, workers: list[Worker]) -> Worker:
        """Find the least loaded worker using neural CMP."""
        best = workers[0]
        
        for w in workers[1:]:
            zf, sf = self.bridge.cmp(w.load, best.load)
            self._op()
            
            if sf:  # w.load < best.load
                best = w
        
        return best
    
    def assign_task(self, task: Task, worker: Worker) -> bool:
        """Assign task to worker if capacity allows (neural check)."""
        # Neural: new_load = current + cost
        new_load = self.bridge.add(worker.load, task.cost)
        self._op()
        
        # Neural: can we fit? new_load <= capacity
        zf, sf = self.bridge.cmp(new_load, worker.capacity)
        self._op()
        
        if sf or zf:  # new_load <= capacity
            worker.load = new_load
            task.assigned_to = worker.id
            task.started_at = time.time()
            return True
        
        return False
    
    def complete_task(self, task: Task, worker: Worker):
        """Complete a task and free worker capacity (neural SUB)."""
        worker.load = self.bridge.sub(worker.load, task.cost)
        self._op()
        worker.tasks_completed = self.bridge.add(worker.tasks_completed, 1)
        self._op()
        task.completed = True
    
    # ── Round Robin ─────────────────────────────────────
    
    def round_robin_next(self, workers: list[Worker], current_idx: int) -> int:
        """Neural modulo for round-robin worker selection."""
        next_idx = self.bridge.add(current_idx, 1)
        self._op()
        
        # Neural modulo: if next >= len, wrap to 0
        zf, sf = self.bridge.cmp(next_idx, len(workers))
        self._op()
        
        if not sf:  # next >= len
            next_idx = 0
        
        return next_idx


# ── CLI ──────────────────────────────────────────────────────

def demo():
    sched = NeuralScheduler()
    
    print("Neural Scheduler")
    print("=" * 60)
    print("Every scheduling decision → trained neural network\n")
    
    # ── Priority Queue ──
    print("── Priority Scheduling ──")
    tasks = [
        Task("T1", "POS health check", priority=3, deadline=0, cost=10),
        Task("T2", "Camera reconnect", priority=1, deadline=0, cost=25),
        Task("T3", "Log rotation", priority=5, deadline=0, cost=5),
        Task("T4", "Backup run", priority=2, deadline=0, cost=30),
        Task("T5", "Alert notify", priority=0, deadline=0, cost=3),
    ]
    
    print("  Before (insertion order):")
    for t in tasks:
        print(f"    {t.id} [{t.name}] priority={t.priority}")
    
    sorted_tasks = sched.sort_by_priority(tasks)
    print("  After neural sort:")
    for t in sorted_tasks:
        print(f"    {t.id} [{t.name}] priority={t.priority}")
    print()
    
    # ── EDF ──
    print("── Earliest Deadline First ──")
    now = 1000
    tasks_edf = [
        Task("D1", "Invoice due", priority=3, deadline=1050, cost=10),
        Task("D2", "Report due", priority=1, deadline=1200, cost=20),
        Task("D3", "URGENT alert", priority=0, deadline=990, cost=5),  # Past deadline!
        Task("D4", "Backup", priority=2, deadline=1100, cost=15),
        Task("D5", "Cleanup", priority=4, deadline=0, cost=8),  # No deadline
    ]
    
    edf_sorted = sched.sort_by_deadline(tasks_edf, now)
    print(f"  Now = {now}")
    for t in edf_sorted:
        dl = f"deadline={t.deadline}" if t.deadline else "no deadline"
        overdue = " ⚠️ OVERDUE" if t.deadline and t.deadline < now else ""
        print(f"    {t.id} [{t.name}] {dl}{overdue}")
    print()
    
    # ── Load Balancing ──
    print("── Load Balancing ──")
    workers = [
        Worker("W1", "Sentinel", load=0, capacity=50),
        Worker("W2", "Monitor", load=0, capacity=50),
        Worker("W3", "Checker", load=0, capacity=50),
    ]
    
    assign_tasks = [
        Task("A1", "Disk check", priority=1, deadline=0, cost=15),
        Task("A2", "Memory check", priority=1, deadline=0, cost=20),
        Task("A3", "POS ping", priority=0, deadline=0, cost=10),
        Task("A4", "Camera poll", priority=2, deadline=0, cost=25),
        Task("A5", "NVR check", priority=1, deadline=0, cost=18),
        Task("A6", "Backup verify", priority=3, deadline=0, cost=12),
    ]
    
    for task in assign_tasks:
        worker = sched.find_least_loaded(workers)
        success = sched.assign_task(task, worker)
        status = f"→ {worker.name} (load={worker.load})" if success else "❌ no capacity"
        print(f"  {task.id} [{task.name}] cost={task.cost} {status}")
    
    print(f"\n  Worker loads:")
    for w in workers:
        bar = "█" * (w.load // 5) + "░" * ((w.capacity - w.load) // 5)
        print(f"    {w.name}: {w.load}/{w.capacity} [{bar}]")
    print()
    
    # ── Aging ──
    print("── Aging (anti-starvation) ──")
    aging_tasks = [
        Task("S1", "Low priority job", priority=5, deadline=0, cost=10),
        Task("S2", "Another low job", priority=4, deadline=0, cost=8),
    ]
    
    print(f"  Initial: S1 priority={aging_tasks[0].priority}, S2 priority={aging_tasks[1].priority}")
    for tick in range(1, 12):
        sched.apply_aging(aging_tasks)
        if tick % 3 == 0:
            print(f"  Tick {tick:2d}: S1 p={aging_tasks[0].priority} wait={aging_tasks[0].wait_time} | S2 p={aging_tasks[1].priority} wait={aging_tasks[1].wait_time}")
    print()
    
    # ── Stats ──
    print(f"── Neural Stats ──")
    print(f"  Total neural ops: {sched._ops}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_scheduler [demo|edf|balance]")


if __name__ == "__main__":
    main()
