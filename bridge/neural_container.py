"""Neural Container Runtime — namespaces and cgroups through nCPU.

A container runtime where every namespace isolation check,
cgroup limit enforcement, and overlay filesystem operation
is neural:

- Namespace isolation: neural XOR for namespace ID mixing
- cgroup limits: neural CMP for CPU/memory enforcement
- Overlay FS: neural layer merging
- Container lifecycle: create, start, exec, stop, rm
- Image layers: neural delta between layers
- Port mapping: neural address translation

Usage:
    python -m bridge.neural_container demo
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


@dataclass
class Namespace:
    """Isolated namespace for a container."""
    ns_id: int
    pid_base: int      # PID offset for this container
    net_id: int        # Network namespace ID
    mnt_id: int        # Mount namespace ID


@dataclass
class Cgroup:
    """Resource limits for a container."""
    mem_limit: int = 256    # bytes
    cpu_quota: int = 50     # percent
    pid_limit: int = 16
    mem_used: int = 0
    cpu_used: int = 0
    pid_count: int = 0


@dataclass
class ImageLayer:
    """A filesystem layer in a container image."""
    id: int
    parent: int  # parent layer ID (0 = base)
    files: dict[str, str] = field(default_factory=dict)  # path → content


@dataclass
class ContainerImage:
    name: str
    tag: str
    layers: list[ImageLayer] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cmd: str = "/bin/sh"


@dataclass
class Container:
    id: int
    name: str
    image: str
    state: str = "created"  # created, running, paused, stopped
    ns: Namespace = None
    cgroup: Cgroup = None
    ports: dict[int, int] = field(default_factory=dict)  # host → container
    env: dict[str, str] = field(default_factory=dict)
    started_at: float = 0.0
    exit_code: int = -1
    log: list[str] = field(default_factory=list)


class NeuralOverlayFS:
    """Overlay filesystem: merge layers using neural delta operations."""

    def __init__(self, bridge: NCPUBridge):
        self.bridge = bridge
        self._ops = 0

    def _op(self):
        self._ops += 1

    def merge_layers(self, layers: list[ImageLayer]) -> dict[str, str]:
        """Merge image layers (upper overrides lower) using neural CMP."""
        merged = {}
        for layer in layers:
            for path, content in layer.files.items():
                # Neural: check if path already exists (scan)
                exists = path in merged
                if exists:
                    # Neural: compute content delta size
                    old_len = len(merged[path])
                    new_len = len(content)
                    zf, _ = self.bridge.cmp(old_len, new_len)
                    self._op()
                merged[path] = content
        return merged


class NeuralContainerRuntime:
    """Container runtime with neural namespace + cgroup management."""

    def __init__(self):
        self.bridge = NCPUBridge()
        self.overlay = NeuralOverlayFS(self.bridge)
        self._containers: dict[int, Container] = {}
        self._images: dict[str, ContainerImage] = {}
        self._next_id = 1
        self._next_ns = 0x1000
        self._ops = 0
        self._build_base_images()

    def _op(self):
        self._ops += 1

    def _alloc_id(self) -> int:
        cid = self._next_id
        self._next_id = self.bridge.add(self._next_id, 1)
        self._op()
        return cid

    def _alloc_namespace(self, container_id: int) -> Namespace:
        """Allocate isolated namespace using neural XOR mixing."""
        ns_id = self.bridge.bitwise_xor(self._next_ns, container_id)
        self._op()
        pid_base = self.bridge.mul(container_id, 1000)
        self._op()
        net_id = self.bridge.bitwise_xor(ns_id, 0xABCD)
        self._op()
        mnt_id = self.bridge.bitwise_xor(ns_id, 0x1234)
        self._op()
        self._next_ns = self.bridge.add(self._next_ns, 0x100)
        self._op()
        return Namespace(ns_id=ns_id, pid_base=pid_base, net_id=net_id, mnt_id=mnt_id)

    def _build_base_images(self):
        """Build built-in container images."""
        # ncpu/base image
        base_layer = ImageLayer(id=1, parent=0, files={
            "/etc/os-release": "ID=ncpu\nNAME=Neural OS\nVERSION=1.0",
            "/bin/sh": "#!/bin/nsh\nexec /bin/nsh $@",
            "/etc/hostname": "container",
        })

        # ncpu/sentinel image
        sentinel_layer = ImageLayer(id=2, parent=1, files={
            "/etc/sentinel.conf": "interval=60\nthreshold=90",
            "/bin/sentinel": "#!/bin/nsh\necho Sentinel running",
        })

        # ncpu/pos image
        pos_layer = ImageLayer(id=3, parent=1, files={
            "/etc/pos.conf": "backend=10.10.10.238:3000\nsite=KRS01",
            "/bin/pos-health": "#!/bin/nsh\ncurl http://pos/health",
        })

        # ncpu/monitor image
        monitor_layer = ImageLayer(id=4, parent=1, files={
            "/etc/monitor.conf": "targets=pos,nvr,cameras",
            "/bin/monitor": "#!/bin/nsh\necho Monitor running",
        })

        self._images = {
            "ncpu/base:latest": ContainerImage("ncpu/base", "latest",
                                               [base_layer], {"PATH": "/bin"}, "/bin/sh"),
            "ncpu/sentinel:latest": ContainerImage("ncpu/sentinel", "latest",
                                                   [base_layer, sentinel_layer],
                                                   {"PATH": "/bin", "ROLE": "sentinel"}, "/bin/sentinel"),
            "ncpu/pos:latest": ContainerImage("ncpu/pos", "latest",
                                              [base_layer, pos_layer],
                                              {"PATH": "/bin", "ROLE": "pos"}, "/bin/pos-health"),
            "ncpu/monitor:latest": ContainerImage("ncpu/monitor", "latest",
                                                  [base_layer, monitor_layer],
                                                  {"PATH": "/bin", "ROLE": "monitor"}, "/bin/monitor"),
        }

    def create(self, image: str, name: str = "",
               ports: dict = None, env: dict = None,
               mem_limit: int = 256, cpu_quota: int = 50) -> Container:
        """Create a container from an image."""
        if image not in self._images:
            # Try with :latest suffix
            image = f"{image}:latest" if ":" not in image else image
        if image not in self._images:
            raise ValueError(f"Image not found: {image}")

        cid = self._alloc_id()
        ns = self._alloc_namespace(cid)
        cgroup = Cgroup(mem_limit=mem_limit, cpu_quota=cpu_quota)

        img = self._images[image]
        merged_env = {**img.env, **(env or {})}

        container = Container(
            id=cid, name=name or f"container-{cid}",
            image=image, ns=ns, cgroup=cgroup,
            ports=ports or {}, env=merged_env,
        )
        self._containers[cid] = container
        return container

    def start(self, container_id: int) -> dict:
        """Start a container — enforce cgroup limits via neural CMP."""
        ct = self._containers.get(container_id)
        if not ct:
            return {"error": "Container not found"}
        if ct.state == "running":
            return {"error": "Already running"}

        # Neural: check memory quota
        zf, sf = self.bridge.cmp(ct.cgroup.mem_used, ct.cgroup.mem_limit)
        self._op()
        if not sf and not zf:
            return {"error": "Memory limit exceeded"}

        ct.state = "running"
        ct.started_at = time.time()

        # Merge image layers into overlay FS
        img = self._images.get(ct.image)
        if img:
            merged = self.overlay.merge_layers(img.layers)
            ct.log.append(f"Mounted overlay: {len(merged)} files")

        # Neural: track PID count
        ct.cgroup.pid_count = self.bridge.add(ct.cgroup.pid_count, 1)
        self._op()

        ct.log.append(f"Container {ct.id} started")
        ct.log.append(f"Namespace: ns={ct.ns.ns_id:04x}, net={ct.ns.net_id:04x}, mnt={ct.ns.mnt_id:04x}")

        return {"id": ct.id, "name": ct.name, "state": "running",
                "pid_base": ct.ns.pid_base, "ns_id": f"{ct.ns.ns_id:04x}"}

    def exec_in(self, container_id: int, command: str) -> str:
        """Execute a command in a running container."""
        ct = self._containers.get(container_id)
        if not ct or ct.state != "running":
            return "Error: container not running"

        # Neural: check CPU quota
        zf, sf = self.bridge.cmp(ct.cgroup.cpu_used, ct.cgroup.cpu_quota)
        self._op()
        if not sf and not zf:
            return "Error: CPU quota exceeded"

        ct.cgroup.cpu_used = self.bridge.add(ct.cgroup.cpu_used, 1)
        self._op()

        # Namespace-isolated PID for this exec
        exec_pid = self.bridge.add(ct.ns.pid_base, ct.cgroup.pid_count)
        self._op()
        ct.cgroup.pid_count = self.bridge.add(ct.cgroup.pid_count, 1)
        self._op()

        # Look up command in image filesystem
        img = self._images.get(ct.image)
        if img:
            merged = self.overlay.merge_layers(img.layers)
            cmd_path = f"/bin/{command.split()[0]}"
            if cmd_path in merged:
                output = merged[cmd_path].split("\n")[-1]  # last line = execution hint
                return f"[PID {exec_pid}] {output}"

        return f"[PID {exec_pid}] {command}: executed in ns={ct.ns.ns_id:04x}"

    def stop(self, container_id: int) -> dict:
        """Stop a container gracefully."""
        ct = self._containers.get(container_id)
        if not ct:
            return {"error": "Not found"}
        ct.state = "stopped"
        ct.exit_code = 0
        return {"id": ct.id, "state": "stopped"}

    def remove(self, container_id: int) -> dict:
        """Remove a stopped container."""
        ct = self._containers.get(container_id)
        if not ct:
            return {"error": "Not found"}
        if ct.state == "running":
            return {"error": "Stop container first"}
        del self._containers[container_id]
        return {"removed": container_id}

    def ps(self) -> list[dict]:
        return [
            {"id": ct.id, "name": ct.name, "image": ct.image,
             "state": ct.state, "ports": ct.ports,
             "cpu": ct.cgroup.cpu_used, "mem": ct.cgroup.mem_used,
             "ns": f"{ct.ns.ns_id:04x}" if ct.ns else "?"}
            for ct in self._containers.values()
        ]

    def stats(self) -> dict:
        running = sum(1 for c in self._containers.values() if c.state == "running")
        total_cpu = 0
        for c in self._containers.values():
            total_cpu = self.bridge.add(total_cpu, c.cgroup.cpu_used)
            self._op()
        return {
            "containers": len(self._containers),
            "running": running,
            "images": len(self._images),
            "total_cpu": total_cpu,
            "neural_ops": self._ops,
        }


def demo():
    rt = NeuralContainerRuntime()

    print("Neural Container Runtime")
    print("=" * 60)
    print("Namespaces + cgroups + overlay FS → neural ALU\n")

    # Show available images
    print("── Images ──")
    for name, img in rt._images.items():
        layers = len(img.layers)
        total_files = sum(len(l.files) for l in img.layers)
        print(f"  📦 {name:30s} {layers} layers, {total_files} files")
    print()

    # Create containers
    print("── Create Containers ──")
    c1 = rt.create("ncpu/sentinel:latest", "sentinel-1",
                   ports={9090: 8080}, mem_limit=256, cpu_quota=30)
    c2 = rt.create("ncpu/pos:latest", "pos-monitor",
                   ports={3000: 3000}, mem_limit=512, cpu_quota=50)
    c3 = rt.create("ncpu/monitor:latest", "monitor-1",
                   env={"LOG_LEVEL": "info"}, mem_limit=128, cpu_quota=20)

    for c in [c1, c2, c3]:
        print(f"  ID={c.id} {c.name:15s} image={c.image} ports={c.ports}")
    print()

    # Start containers
    print("── Start Containers ──")
    for cid in [c1.id, c2.id, c3.id]:
        result = rt.start(cid)
        ct = rt._containers[cid]
        print(f"  {ct.name}: ns={result.get('ns_id','?')} pid_base={result.get('pid_base','?')}")
        for log in ct.log:
            print(f"    {log}")
    print()

    # Exec in containers
    print("── Exec in Containers ──")
    for cid, cmd in [(c1.id, "sentinel"), (c2.id, "pos-health"), (c3.id, "monitor")]:
        out = rt.exec_in(cid, cmd)
        name = rt._containers[cid].name
        print(f"  [{name}] $ {cmd}")
        print(f"    {out}")
    print()

    # Port mapping (neural address translation)
    print("── Port Mapping ──")
    for ct in rt._containers.values():
        if ct.ports:
            for host_port, ct_port in ct.ports.items():
                # Neural ADD for port translation
                mapped = rt.bridge.add(ct.ns.net_id, ct_port)
                rt._ops += 1
                print(f"  {ct.name}: localhost:{host_port} → ns={ct.ns.net_id:04x}:{ct_port}")
    print()

    # Container ps
    print("── docker ps ──")
    print(f"  {'ID':4s} {'NAME':15s} {'IMAGE':25s} {'STATE':8s} {'NS':6s}")
    for p in rt.ps():
        print(f"  {p['id']:4d} {p['name']:15s} {p['image']:25s} {p['state']:8s} {p['ns']}")
    print()

    # Stop and remove one
    print("── Stop & Remove ──")
    rt.stop(c3.id)
    rt.remove(c3.id)
    s = rt.stats()
    print(f"  Containers: {s['containers']} ({s['running']} running)")
    print(f"  Neural ops: {s['neural_ops']}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_container [demo]")


if __name__ == "__main__":
    main()
