"""Neural Kernel — a complete neural operating system kernel.

Combines all modules into a bootable kernel that manages:
- Process management (VM)
- Memory management (heap + GC)
- Filesystem (neural FS)  
- Networking (HTTP + DNS)
- IPC (pipes + message queues)
- Device drivers (framebuffer, audio)
- System calls
- Boot sequence

Every kernel operation runs through trained neural networks.

Usage:
    python -m bridge.neural_kernel boot
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


class NeuralKernel:
    """The neural operating system kernel."""
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._ops = 0
        self._uptime_start = time.time()
        self._boot_log = []
        self._subsystems = {}
    
    def _op(self):
        self._ops += 1
    
    def _log(self, msg: str):
        elapsed = time.time() - self._uptime_start
        entry = f"[{elapsed:7.3f}] {msg}"
        self._boot_log.append(entry)
        print(f"  {entry}")
    
    def boot(self):
        """Boot the neural kernel."""
        print("╔══════════════════════════════════════════════════════════╗")
        print("║            nCPU Neural Operating System v1.0            ║")
        print("║         Every operation is a neural network             ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()
        
        self._uptime_start = time.time()
        
        # Phase 1: Hardware init
        self._log("BOOT: Neural ALU self-test...")
        self._self_test()
        
        # Phase 2: Memory subsystem
        self._log("MEM: Initializing neural heap (1024 bytes)...")
        from bridge.neural_vm import NeuralVM
        self._subsystems["vm"] = NeuralVM()
        self._log("MEM: Heap ready")
        
        # Phase 3: Garbage collector
        self._log("GC: Initializing mark-and-sweep collector...")
        from bridge.neural_gc import NeuralGC
        self._subsystems["gc"] = NeuralGC()
        self._log("GC: Ready (mark-and-sweep + refcount)")
        
        # Phase 4: Filesystem
        self._log("FS: Mounting neural filesystem...")
        from bridge.neural_fs import NeuralFilesystem
        fs = NeuralFilesystem()
        fs.mkdir("/bin")
        fs.mkdir("/etc")
        fs.mkdir("/var")
        fs.mkdir("/var/log")
        fs.mkdir("/tmp")
        fs.mkdir("/home")
        fs.create("/etc/hostname", "ncpu-kernel")
        fs.create("/etc/version", "1.0.0")
        self._subsystems["fs"] = fs
        self._log("FS: Mounted (6 dirs, 2 files)")
        
        # Phase 5: IPC
        self._log("IPC: Setting up pipes and message queues...")
        from bridge.neural_ipc import NeuralPipe, NeuralMessageQueue, NeuralSemaphore
        self._subsystems["mq"] = NeuralMessageQueue(self.bridge)
        self._subsystems["kernel_pipe"] = NeuralPipe(self.bridge, capacity=128)
        self._log("IPC: Ready (pipe + message queue)")
        
        # Phase 6: Networking
        self._log("NET: Starting neural network stack...")
        from bridge.neural_dns import NeuralDNS
        dns = NeuralDNS()
        self._subsystems["dns"] = dns
        stats = dns.cache_stats()
        self._log(f"NET: DNS resolver ready ({stats.get('zones', stats.get('zone_records', 0))} zones)")
        
        from bridge.neural_http import NeuralHTTPServer
        self._subsystems["http"] = NeuralHTTPServer()
        self._log("NET: HTTP server ready")
        
        # Phase 7: Scheduler
        self._log("SCHED: Initializing neural scheduler...")
        from bridge.neural_scheduler import NeuralScheduler
        self._subsystems["scheduler"] = NeuralScheduler()
        self._log("SCHED: Priority + EDF + load balance ready")
        
        # Phase 8: Devices
        self._log("DEV: Initializing framebuffer (60x20)...")
        from bridge.neural_gfx import NeuralFramebuffer
        fb = NeuralFramebuffer(60, 12)
        self._subsystems["fb"] = fb
        self._log("DEV: Framebuffer ready")
        
        # Phase 9: Crypto
        self._log("SEC: Loading crypto subsystem...")
        from bridge.neural_crypto import NeuralStreamCipher
        self._subsystems["crypto"] = NeuralStreamCipher()
        self._log("SEC: Stream cipher + KDF ready")
        
        # Phase 10: State machine
        self._log("STATE: Initializing obligation state machine...")
        from bridge.neural_state_machine import NeuralStateMachine
        self._subsystems["state"] = NeuralStateMachine()
        self._log("STATE: 6-state lifecycle ready")
        
        # Phase 11: Spawn init process
        self._log("INIT: Spawning PID 1 (init)...")
        vm = self._subsystems["vm"]
        init = vm.spawn("init")
        addr = vm.syscall_alloc(init.pid, 64)
        vm.syscall_write(init.pid, addr, [ord(c) for c in "nCPU kernel"])
        self._log(f"INIT: PID {init.pid} running (64 bytes allocated)")
        
        # Phase 12: Write boot log
        self._log("LOG: Writing boot log to /var/log/boot.log...")
        boot_log_text = "\n".join(self._boot_log)
        fs.create("/var/log/boot.log", boot_log_text)
        
        # Boot complete
        elapsed = time.time() - self._uptime_start
        self._log(f"BOOT: Kernel ready in {elapsed:.3f}s")
        
        print()
        self._show_status()
    
    def _self_test(self):
        """ALU self-test: verify basic operations."""
        tests = [
            ("ADD", lambda: self.bridge.add(42, 58), 100),
            ("SUB", lambda: self.bridge.sub(100, 37), 63),
            ("MUL", lambda: self.bridge.mul(7, 8), 56),
            ("DIV", lambda: self.bridge.div(144, 12), 12),
            ("CMP", lambda: self.bridge.cmp(42, 42), (True, False)),
        ]
        
        passed = 0
        for name, func, expected in tests:
            result = func()
            self._op()
            if result == expected:
                passed += 1
        
        self._log(f"BOOT: ALU self-test: {passed}/{len(tests)} passed")
    
    def _show_status(self):
        """Display kernel status."""
        print("╔══════════════════════════════════════════════════════════╗")
        print("║                    KERNEL STATUS                       ║")
        print("╠══════════════════════════════════════════════════════════╣")
        
        # Subsystems
        subsys = list(self._subsystems.keys())
        print(f"║  Subsystems: {len(subsys):3d} loaded                            ║")
        
        # Filesystem
        fs = self._subsystems.get("fs")
        if fs:
            stats = fs.stat()
            print(f"║  Filesystem: {stats['inodes_used']}/{stats['inodes_max']} inodes, "
                  f"{stats['used_blocks']}/{stats['total_blocks']} blocks        ║")
        
        # VM
        vm = self._subsystems.get("vm")
        if vm:
            procs = vm.ps()
            print(f"║  Processes:  {len(procs)} running                              ║")
        
        # Memory
        if vm:
            heap = vm.heap.stats()
            print(f"║  Heap:       {heap['allocated']}/{heap['heap_size']} bytes "
                  f"({heap['fragments']} fragments)          ║")
        
        # Network
        dns = self._subsystems.get("dns")
        if dns:
            ds = dns.cache_stats()
            zones = ds.get('zones', ds.get('zone_records', 0))
            cached = ds.get('cached', ds.get('entries', 0))
            print(f"║  DNS:        {zones} zones, {cached} cached                     ║")
        
        elapsed = time.time() - self._uptime_start
        print(f"║  Uptime:     {elapsed:.3f}s                                  ║")
        print(f"║  Neural ops: {self._ops}                                       ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print("║                                                        ║")
        print("║  Subsystem modules:                                    ║")
        
        module_names = {
            "vm": "Process Manager",
            "gc": "Garbage Collector",
            "fs": "Filesystem",
            "mq": "Message Queue",
            "kernel_pipe": "Kernel Pipe",
            "dns": "DNS Resolver",
            "http": "HTTP Server",
            "scheduler": "Task Scheduler",
            "fb": "Framebuffer",
            "crypto": "Crypto Engine",
            "state": "State Machine",
        }
        
        for key in subsys:
            name = module_names.get(key, key)
            print(f"║    ✅ {name:<48s}  ║")
        
        print("║                                                        ║")
        print("╚══════════════════════════════════════════════════════════╝")
        
        # Render something on the framebuffer
        fb = self._subsystems.get("fb")
        if fb:
            fb.clear()
            fb.rect(0, 0, 60, 12)
            fb.text(3, 2, "NCPU OK")
            fb.line(35, 2, 55, 10)
            fb.line(55, 2, 35, 10)
            fb.circle(20, 8, 3)
            
            print()
            print("  Framebuffer output:")
            print(fb.render())
    
    def syscall(self, call: str, *args) -> dict:
        """Handle system calls."""
        if call == "getpid":
            return {"pid": 1}
        elif call == "hostname":
            fs = self._subsystems.get("fs")
            if fs:
                result = fs.read("/etc/hostname")
                return {"hostname": result.get("data", "unknown")}
        elif call == "version":
            fs = self._subsystems.get("fs")
            if fs:
                result = fs.read("/etc/version")
                return {"version": result.get("data", "unknown")}
        elif call == "uptime":
            return {"uptime_seconds": time.time() - self._uptime_start}
        elif call == "resolve" and args:
            dns = self._subsystems.get("dns")
            if dns:
                records = dns.resolve(args[0])
                if records:
                    return {"address": records[0].value}
            return {"error": "NXDOMAIN"}
        
        return {"error": f"Unknown syscall: {call}"}


def main():
    kernel = NeuralKernel()
    
    if len(sys.argv) > 1 and sys.argv[1] == "boot":
        kernel.boot()
    else:
        kernel.boot()
        
        # Run some syscalls
        print()
        print("  System calls:")
        for call in ["hostname", "version", "uptime"]:
            result = kernel.syscall(call)
            print(f"    syscall({call}) → {result}")
        
        result = kernel.syscall("resolve", "ncpu.local")
        print(f"    syscall(resolve, ncpu.local) → {result}")


if __name__ == "__main__":
    main()
