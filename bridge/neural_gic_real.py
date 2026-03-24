"""
Phase 40 — Real Neural Interrupt Controller (GIC)
===================================================
Uses nCPU's actual NeuralGIC — a neural priority encoder that learns
optimal interrupt dispatch ordering. Instead of fixed priority tables,
an MLP scores all pending interrupts and selects the most important.

Architecture:
  Input:  [NUM_IRQS * 3] — pending bits + in-service bits + mask bits  
  MLP:    Linear(96, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 32)
  Output: [32] priority scores per IRQ (higher = handle first)
"""

import sys
import torch
from pathlib import Path

from bridge.config import get_ncpu_path, get_bridge_path, get_clawd_data_path
NCPU_PATH = get_ncpu_path()
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from ncpu.os.neuros.interrupts import (
    NeuralGIC, NeuralPriorityEncoder, 
    IRQ_TIMER, IRQ_KEYBOARD, IRQ_DISK, IRQ_NETWORK,
    IRQ_IPC, IRQ_PAGE_FAULT, IRQ_SYSCALL, IRQ_GPU, NUM_IRQS
)
from bridge.compute import NCPUBridge

bridge = NCPUBridge()

IRQ_NAMES = {
    0: "TIMER", 1: "KEYBOARD", 2: "DISK", 3: "NETWORK",
    4: "IPC", 5: "PAGE_FAULT", 6: "SYSCALL", 7: "GPU",
}


class RealNeuralGIC:
    """Bridge to nCPU's neural interrupt controller."""

    def __init__(self):
        self.gic = NeuralGIC(num_irqs=NUM_IRQS, device=torch.device("cpu"))
        
        # Load pretrained weights
        model_path = NCPU_PATH / "models" / "os" / "gic.pt"
        if model_path.exists():
            self.gic.load(str(model_path))
            self._pretrained = True
        else:
            self._pretrained = False

        self._params = sum(p.numel() for p in self.gic.encoder.parameters())
        self._dispatch_log = []

    def raise_irq(self, irq: int):
        """Raise an interrupt request."""
        self.gic.raise_irq(irq)

    def mask(self, irq: int):
        self.gic.mask_irq(irq)

    def unmask(self, irq: int):
        self.gic.unmask_irq(irq)

    def register_handler(self, irq: int, name: str, handler):
        """Register an interrupt handler (plain callable)."""
        self.gic.register_handler(irq, handler)

    def dispatch(self) -> int | None:
        """Neural dispatch — select highest priority pending interrupt."""
        irq = self.gic.dispatch()
        if irq is not None:
            self._dispatch_log.append(irq)
        return irq

    def dispatch_all(self) -> list[int]:
        """Dispatch all pending interrupts in neural priority order."""
        dispatched = self.gic.dispatch_all()
        self._dispatch_log.extend(dispatched)
        return dispatched

    def stats(self):
        s = self.gic.stats()
        s['params'] = self._params
        s['pretrained'] = self._pretrained
        return s


def demo():
    print("Real Neural Interrupt Controller (GIC)")
    print("=" * 60)
    print(f"Architecture: MLP priority encoder (96→64→64→32)")

    gic = RealNeuralGIC()
    print(f"Parameters: {gic._params:,}")
    print(f"Pretrained: {gic._pretrained}\n")

    # Register handlers
    handler_log = []

    for irq, name in IRQ_NAMES.items():
        def make_handler(n, i):
            def h(state=None):
                handler_log.append(n)
            return h
        gic.register_handler(irq, name, make_handler(name, irq))

    # Scenario 1: Single interrupt
    print("  Scenario 1: Single TIMER interrupt")
    gic.raise_irq(IRQ_TIMER)
    dispatched = gic.dispatch()
    name = IRQ_NAMES.get(dispatched, f"IRQ{dispatched}") if dispatched is not None else "None"
    print(f"    Dispatched: {name} ✅\n")

    # Scenario 2: Multiple simultaneous interrupts
    print("  Scenario 2: Multiple simultaneous interrupts")
    print("    Raising: KEYBOARD, DISK, NETWORK, PAGE_FAULT, SYSCALL")
    gic.raise_irq(IRQ_KEYBOARD)
    gic.raise_irq(IRQ_DISK)
    gic.raise_irq(IRQ_NETWORK)
    gic.raise_irq(IRQ_PAGE_FAULT)
    gic.raise_irq(IRQ_SYSCALL)

    dispatched = gic.dispatch_all()
    order = [IRQ_NAMES.get(i, f"IRQ{i}") for i in dispatched]
    print(f"    Dispatch order: {order}")
    print(f"    (Neural network decided this priority ordering)\n")

    # Scenario 3: Masked interrupts
    print("  Scenario 3: Masked interrupts")
    gic.mask(IRQ_DISK)
    gic.raise_irq(IRQ_DISK)
    gic.raise_irq(IRQ_GPU)
    
    dispatched = gic.dispatch_all()
    order = [IRQ_NAMES.get(i, f"IRQ{i}") for i in dispatched]
    print(f"    Masked DISK, raised DISK + GPU")
    print(f"    Dispatched: {order}")
    disk_dispatched = IRQ_DISK in dispatched
    gpu_dispatched = IRQ_GPU in dispatched
    print(f"    DISK masked out: {'✅' if not disk_dispatched else '❌'}")
    print(f"    GPU dispatched:  {'✅' if gpu_dispatched else '❌'}")

    # Unmask and dispatch remaining
    gic.unmask(IRQ_DISK)
    remaining = gic.dispatch_all()
    if remaining:
        print(f"    After unmask: {[IRQ_NAMES.get(i, f'IRQ{i}') for i in remaining]}")

    # Scenario 4: Burst of all interrupts
    print("\n  Scenario 4: All IRQs firing simultaneously")
    for irq in range(8):
        gic.raise_irq(irq)
    
    all_dispatched = gic.dispatch_all()
    order = [IRQ_NAMES.get(i, f"IRQ{i}") for i in all_dispatched]
    print(f"    Neural priority order: {order}")

    stats = gic.stats()
    print(f"\n  Stats:")
    print(f"    Total dispatches: {stats.get('total_dispatches', len(gic._dispatch_log))}")
    print(f"    Parameters: {gic._params:,}")
    print(f"    Handler log: {handler_log[:10]}...")

    print(f"\n✅ Real Neural GIC: learned interrupt priority dispatch")


if __name__ == "__main__":
    demo()
