"""Neural Hypervisor — run multiple neural kernel instances simultaneously.

A type-2 hypervisor where every VM isolation check, resource quota
enforcement, and guest scheduling decision is neural:

- VM creation with neural ID allocation
- CPU time accounting via neural ADD
- Memory quota enforcement via neural CMP
- Inter-VM messaging via neural message queue
- Guest suspension/resume via neural state tracking
- Resource usage reporting

Usage:
    python -m bridge.neural_hypervisor demo
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from io import StringIO
import contextlib

from bridge.config import get_ncpu_path, get_bridge_path, get_clawd_data_path
NCPU_PATH = get_ncpu_path()
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


@dataclass
class VMConfig:
    name: str
    mem_quota: int = 512     # bytes
    cpu_shares: int = 100    # relative weight
    max_procs: int = 4


@dataclass
class GuestVM:
    id: int
    config: VMConfig
    state: str = "created"   # created, running, suspended, terminated
    cpu_used: int = 0        # neural-tracked cycles
    mem_used: int = 0
    uptime_start: float = 0.0
    boot_log: list[str] = field(default_factory=list)
    kernel: object = None    # NeuralKernel instance


class NeuralHypervisor:
    """Type-2 hypervisor managing multiple neural kernel VMs."""

    def __init__(self):
        self.bridge = NCPUBridge()
        self._vms: dict[int, GuestVM] = {}
        self._next_vmid = 1
        self._ops = 0
        self._total_cpu = 0
        self._total_mem = 512 * 8  # 4KB host memory

    def _op(self):
        self._ops += 1

    def _alloc_vmid(self) -> int:
        vmid = self._next_vmid
        self._next_vmid = self.bridge.add(self._next_vmid, 1)
        self._op()
        return vmid

    def create_vm(self, config: VMConfig) -> GuestVM:
        """Create a new VM with neural quota checks."""
        # Neural: check total memory won't be exceeded
        new_total = self.bridge.add(self._total_mem_used(), config.mem_quota)
        self._op()
        zf, sf = self.bridge.cmp(new_total, self._total_mem)
        self._op()
        if not sf and not zf:
            raise RuntimeError(f"Insufficient host memory (need {config.mem_quota})")

        vmid = self._alloc_vmid()
        vm = GuestVM(id=vmid, config=config)
        self._vms[vmid] = vm
        return vm

    def _total_mem_used(self) -> int:
        total = 0
        for vm in self._vms.values():
            if vm.state in ("running", "suspended"):
                total = self.bridge.add(total, vm.config.mem_quota)
                self._op()
        return total

    def boot_vm(self, vmid: int) -> dict:
        """Boot a guest VM (starts a NeuralKernel)."""
        vm = self._vms.get(vmid)
        if not vm:
            return {"error": f"VM {vmid} not found"}
        if vm.state == "running":
            return {"error": "Already running"}

        from bridge.neural_kernel import NeuralKernel

        # Boot kernel silently, capture log
        vm.uptime_start = time.time()
        vm.state = "running"

        kernel = NeuralKernel()
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            kernel.boot()

        vm.kernel = kernel
        vm.boot_log = buf.getvalue().split("\n")

        # Track CPU usage (neural ADD)
        boot_time_ms = int((time.time() - vm.uptime_start) * 1000)
        vm.cpu_used = self.bridge.add(vm.cpu_used, boot_time_ms)
        self._op()
        self._total_cpu = self.bridge.add(self._total_cpu, boot_time_ms)
        self._op()

        # Track memory (neural: count boot log lines as proxy)
        vm.mem_used = self.bridge.mul(len(vm.boot_log), 32)
        self._op()

        return {
            "vmid": vmid,
            "name": vm.config.name,
            "boot_time_ms": boot_time_ms,
            "status": "running",
        }

    def suspend_vm(self, vmid: int) -> dict:
        """Suspend a VM, preserving state."""
        vm = self._vms.get(vmid)
        if not vm or vm.state != "running":
            return {"error": f"VM {vmid} not running"}
        vm.state = "suspended"
        return {"vmid": vmid, "state": "suspended"}

    def resume_vm(self, vmid: int) -> dict:
        """Resume a suspended VM."""
        vm = self._vms.get(vmid)
        if not vm or vm.state != "suspended":
            return {"error": f"VM {vmid} not suspended"}
        vm.state = "running"
        return {"vmid": vmid, "state": "running"}

    def terminate_vm(self, vmid: int) -> dict:
        """Terminate a VM and free resources."""
        vm = self._vms.get(vmid)
        if not vm:
            return {"error": f"VM {vmid} not found"}
        vm.state = "terminated"
        vm.kernel = None
        return {"vmid": vmid, "state": "terminated"}

    def syscall_passthrough(self, vmid: int, call: str, *args) -> dict:
        """Pass a system call to a guest VM's kernel."""
        vm = self._vms.get(vmid)
        if not vm or vm.state != "running":
            return {"error": "VM not running"}

        # Neural: verify call is permitted (quota check)
        cpu_quota = self.bridge.mul(vm.config.cpu_shares, 10)
        self._op()
        zf, sf = self.bridge.cmp(vm.cpu_used, cpu_quota)
        self._op()
        if not sf and not zf:
            return {"error": "CPU quota exceeded"}

        result = vm.kernel.syscall(call, *args)

        # Neural: charge CPU time
        vm.cpu_used = self.bridge.add(vm.cpu_used, 1)
        self._op()

        return result

    def inter_vm_message(self, src_vmid: int, dst_vmid: int, msg: str) -> bool:
        """Send message between VMs via neural message queue."""
        src = self._vms.get(src_vmid)
        dst = self._vms.get(dst_vmid)

        if not src or not dst:
            return False
        if dst.state != "running":
            return False

        # Deliver via dst kernel's message queue
        mq = dst.kernel._subsystems.get("mq") if dst.kernel else None
        if mq:
            data = [ord(c) for c in msg[:32]]
            mq.send(priority=1, sender=src_vmid, data=data)
            return True

        return False

    def scheduler_tick(self) -> dict:
        """Neural scheduler: allocate CPU shares among VMs."""
        running = [vm for vm in self._vms.values() if vm.state == "running"]
        if not running:
            return {"scheduled": 0}

        total_shares = 0
        for vm in running:
            total_shares = self.bridge.add(total_shares, vm.config.cpu_shares)
            self._op()

        allocations = {}
        for vm in running:
            # Neural: each VM's share = cpu_shares / total_shares * 100
            share = self.bridge.div(self.bridge.mul(vm.config.cpu_shares, 100), total_shares)
            self._op()
            allocations[vm.id] = share

        return {"scheduled": len(running), "allocations": allocations}

    def status(self) -> dict:
        running = sum(1 for v in self._vms.values() if v.state == "running")
        suspended = sum(1 for v in self._vms.values() if v.state == "suspended")
        total_mem = self._total_mem_used()

        return {
            "total_vms": len(self._vms),
            "running": running,
            "suspended": suspended,
            "terminated": len(self._vms) - running - suspended,
            "host_mem_used": total_mem,
            "host_mem_total": self._total_mem,
            "total_cpu_ms": self._total_cpu,
            "neural_ops": self._ops,
        }


def demo():
    print("Neural Hypervisor")
    print("=" * 60)
    print("Multi-VM management + resource quotas → neural ALU\n")

    hv = NeuralHypervisor()

    # Create VMs
    print("── Create VMs ──")
    vm1 = hv.create_vm(VMConfig("sentinel-vm", mem_quota=512, cpu_shares=50))
    vm2 = hv.create_vm(VMConfig("monitor-vm", mem_quota=512, cpu_shares=30))
    vm3 = hv.create_vm(VMConfig("dev-vm", mem_quota=256, cpu_shares=20))

    for vm in [vm1, vm2, vm3]:
        print(f"  VM{vm.id}: {vm.config.name} mem={vm.config.mem_quota}B cpu={vm.config.cpu_shares}%")
    print()

    # Boot VMs
    print("── Boot VMs ──")
    for vmid in [vm1.id, vm2.id, vm3.id]:
        result = hv.boot_vm(vmid)
        print(f"  VM{vmid} ({hv._vms[vmid].config.name}): boot_time={result['boot_time_ms']}ms")
    print()

    # Scheduler tick
    print("── CPU Scheduler ──")
    tick = hv.scheduler_tick()
    print(f"  Scheduled: {tick['scheduled']} VMs")
    for vmid, share in tick["allocations"].items():
        name = hv._vms[vmid].config.name
        print(f"    VM{vmid} ({name}): {share}% CPU share (neural DIV)")
    print()

    # Syscall passthrough
    print("── Syscall Passthrough ──")
    for vmid in [vm1.id, vm2.id]:
        result = hv.syscall_passthrough(vmid, "hostname")
        name = hv._vms[vmid].config.name
        print(f"  VM{vmid} ({name}) hostname → {result.get('hostname', result)}")

    result = hv.syscall_passthrough(vm1.id, "version")
    print(f"  VM{vm1.id} version → {result.get('version', '?')}")
    print()

    # Inter-VM messaging
    print("── Inter-VM Messaging ──")
    sent = hv.inter_vm_message(vm1.id, vm2.id, "PING from sentinel")
    print(f"  VM1 → VM2: {'✅ delivered' if sent else '❌ failed'}")

    # Check monitor's message queue
    vm2_kernel = hv._vms[vm2.id].kernel
    mq = vm2_kernel._subsystems.get("mq")
    if mq and mq.size > 0:
        msg = mq.recv()
        text = "".join(chr(b) for b in msg.data)
        print(f"  VM2 received: '{text}' from VM{msg.sender}")
    print()

    # Suspend/resume
    print("── Suspend/Resume ──")
    hv.suspend_vm(vm3.id)
    s = hv.status()
    print(f"  After suspend VM3: {s['running']} running, {s['suspended']} suspended")

    hv.resume_vm(vm3.id)
    s = hv.status()
    print(f"  After resume VM3:  {s['running']} running, {s['suspended']} suspended")
    print()

    # Final status
    print("── Hypervisor Status ──")
    s = hv.status()
    mem_pct = s["host_mem_used"] / s["host_mem_total"] * 100
    print(f"  VMs: {s['total_vms']} total, {s['running']} running")
    print(f"  Host memory: {s['host_mem_used']}/{s['host_mem_total']} bytes ({mem_pct:.0f}%)")
    print(f"  Total CPU time: {s['total_cpu_ms']}ms across all VMs")
    print(f"  Neural ops: {s['neural_ops']}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_hypervisor [demo]")


if __name__ == "__main__":
    main()
