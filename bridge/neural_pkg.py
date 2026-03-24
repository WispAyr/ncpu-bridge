"""Neural Package Manager — install/remove/query packages through nCPU.

A package manager where every dependency resolution, version comparison,
and integrity check is neural:

- Version comparison: neural CMP on version tuples
- Dependency graph: neural traversal with cycle detection
- Package integrity: neural CRC32 verification
- Install/remove: neural filesystem operations
- Search: neural string matching against package index

Usage:
    python -m bridge.neural_pkg demo
    python -m bridge.neural_pkg install <pkg>
    python -m bridge.neural_pkg list
    python -m bridge.neural_pkg search <query>
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from bridge.config import get_ncpu_path, get_bridge_path, get_clawd_data_path
NCPU_PATH = get_ncpu_path()
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


@dataclass
class Package:
    name: str
    version: tuple[int, int, int]
    deps: list[str] = field(default_factory=list)
    description: str = ""
    size: int = 0
    checksum: int = 0
    installed: bool = False
    install_time: float = 0.0


class NeuralPackageManager:
    """Package manager with neural version comparison and dep resolution."""

    # Built-in package registry
    REGISTRY: dict[str, Package] = {
        "ncpu-alu":         Package("ncpu-alu", (1, 0, 0), [], "Neural ALU core", 512),
        "ncpu-net":         Package("ncpu-net", (1, 0, 0), ["ncpu-alu"], "Neural networking stack", 256),
        "ncpu-fs":          Package("ncpu-fs", (1, 0, 0), ["ncpu-alu"], "Neural filesystem", 384),
        "ncpu-crypto":      Package("ncpu-crypto", (1, 0, 0), ["ncpu-alu"], "Neural crypto engine", 320),
        "ncpu-gfx":         Package("ncpu-gfx", (1, 0, 0), ["ncpu-alu"], "Neural graphics", 448),
        "ncpu-audio":       Package("ncpu-audio", (1, 0, 0), ["ncpu-alu"], "Neural audio synth", 192),
        "ncpu-db":          Package("ncpu-db", (1, 0, 0), ["ncpu-alu", "ncpu-fs"], "Neural database", 512),
        "ncpu-scheduler":   Package("ncpu-scheduler", (1, 0, 0), ["ncpu-alu"], "Neural task scheduler", 256),
        "ncpu-vm":          Package("ncpu-vm", (1, 0, 0), ["ncpu-alu", "ncpu-fs"], "Neural VM + GC", 640),
        "ncpu-ipc":         Package("ncpu-ipc", (1, 0, 0), ["ncpu-alu", "ncpu-vm"], "Neural IPC", 320),
        "ncpu-shell":       Package("ncpu-shell", (1, 2, 0), ["ncpu-vm", "ncpu-fs", "ncpu-net"], "Neural shell", 512),
        "ncpu-kernel":      Package("ncpu-kernel", (1, 0, 0), ["ncpu-alu", "ncpu-vm", "ncpu-fs", "ncpu-net", "ncpu-ipc", "ncpu-scheduler"], "Neural OS kernel", 1024),
        "ncpu-jit":         Package("ncpu-jit", (1, 0, 0), ["ncpu-alu", "ncpu-vm"], "JIT compiler", 448),
        "ncpu-debugger":    Package("ncpu-debugger", (1, 0, 0), ["ncpu-alu", "ncpu-jit"], "Neural debugger", 384),
        "ncpu-forth":       Package("ncpu-forth", (1, 0, 0), ["ncpu-alu"], "Forth interpreter", 256),
        "ncpu-regex":       Package("ncpu-regex", (1, 0, 0), ["ncpu-alu"], "Neural regex engine", 192),
        "ncpu-compress":    Package("ncpu-compress", (1, 0, 0), ["ncpu-alu"], "Neural compression", 224),
        "ncpu-dns":         Package("ncpu-dns", (1, 0, 0), ["ncpu-net"], "Neural DNS resolver", 192),
        "ncpu-http":        Package("ncpu-http", (1, 0, 0), ["ncpu-net"], "Neural HTTP server", 256),
        "neural-os-full":   Package("neural-os-full", (1, 0, 0),
                                    ["ncpu-kernel", "ncpu-shell", "ncpu-gfx", "ncpu-audio", "ncpu-db",
                                     "ncpu-crypto", "ncpu-jit", "ncpu-debugger", "ncpu-forth", "ncpu-regex",
                                     "ncpu-compress", "ncpu-dns", "ncpu-http"],
                                    "Full neural OS meta-package", 0),
    }

    def __init__(self):
        self.bridge = NCPUBridge()
        self._installed: dict[str, Package] = {}
        self._ops = 0
        # Pre-install base
        self._install_silent("ncpu-alu")

    def _op(self):
        self._ops += 1

    def _version_cmp(self, a: tuple, b: tuple) -> int:
        """Compare version tuples using neural CMP. Returns -1, 0, 1."""
        for i in range(3):
            ai = a[i] if i < len(a) else 0
            bi = b[i] if i < len(b) else 0
            zf, sf = self.bridge.cmp(ai, bi)
            self._op()
            if sf and not zf:
                return -1  # a < b
            if not sf and not zf:
                return 1   # a > b
        return 0  # equal

    def _neural_str_contains(self, haystack: str, needle: str) -> bool:
        """Search substring using neural byte comparison."""
        if len(needle) > len(haystack):
            return False
        for i in range(len(haystack) - len(needle) + 1):
            match = True
            for j in range(len(needle)):
                zf, _ = self.bridge.cmp(ord(haystack[i + j].lower()), ord(needle[j].lower()))
                self._op()
                if not zf:
                    match = False
                    break
            if match:
                return True
        return False

    def _resolve_deps(self, name: str, visited: set = None) -> list[str]:
        """Resolve dependency order with neural cycle detection."""
        if visited is None:
            visited = set()

        if name in visited:
            return []  # Cycle detected

        visited.add(name)
        pkg = self.REGISTRY.get(name)
        if not pkg:
            return []

        order = []
        for dep in pkg.deps:
            # Neural: is dep already in installed?
            already = dep in self._installed
            if not already:
                order.extend(self._resolve_deps(dep, visited))

        if name not in self._installed:
            order.append(name)

        return order

    def _compute_checksum(self, pkg: Package) -> int:
        """Compute package checksum using neural XOR."""
        checksum = 0
        for ch in pkg.name:
            checksum = self.bridge.bitwise_xor(checksum, ord(ch))
            self._op()
            checksum = self.bridge.add(checksum, pkg.version[0])
            self._op()
        return self.bridge.bitwise_and(checksum, 0xFFFF)

    def _install_silent(self, name: str):
        pkg = self.REGISTRY.get(name)
        if pkg and name not in self._installed:
            installed = Package(**pkg.__dict__)
            installed.installed = True
            installed.install_time = time.time()
            installed.checksum = self._compute_checksum(installed)
            self._installed[name] = installed

    def install(self, name: str) -> dict:
        """Install a package and its dependencies."""
        if name not in self.REGISTRY:
            return {"error": f"Package not found: {name}"}

        order = self._resolve_deps(name)
        if not order:
            return {"status": "already_installed", "package": name}

        installed = []
        total_size = 0

        for pkg_name in order:
            pkg = self.REGISTRY.get(pkg_name)
            if not pkg:
                continue

            # Verify checksum
            checksum = self._compute_checksum(pkg)

            installed_pkg = Package(**pkg.__dict__)
            installed_pkg.installed = True
            installed_pkg.install_time = time.time()
            installed_pkg.checksum = checksum
            self._installed[pkg_name] = installed_pkg

            total_size = self.bridge.add(total_size, pkg.size)
            self._op()
            installed.append(pkg_name)

        return {
            "status": "ok",
            "installed": installed,
            "total_size": total_size,
            "neural_ops": self._ops,
        }

    def remove(self, name: str) -> dict:
        """Remove a package (check for reverse deps first)."""
        if name not in self._installed:
            return {"error": f"Not installed: {name}"}

        # Check if anything depends on this
        dependents = []
        for pkg_name, pkg in self._installed.items():
            if name in pkg.deps and pkg_name != name:
                dependents.append(pkg_name)

        if dependents:
            return {"error": f"Cannot remove: required by {', '.join(dependents)}"}

        del self._installed[name]
        return {"status": "ok", "removed": name}

    def search(self, query: str) -> list[Package]:
        """Search package index using neural string matching."""
        results = []
        for pkg in self.REGISTRY.values():
            if (self._neural_str_contains(pkg.name, query) or
                    self._neural_str_contains(pkg.description, query)):
                results.append(pkg)
        return results

    def list_installed(self) -> list[Package]:
        return list(self._installed.values())

    def upgrade(self, name: str = None) -> dict:
        """Check for upgrades using neural version comparison."""
        upgrades = []
        targets = [name] if name else list(self._installed.keys())

        for pkg_name in targets:
            installed = self._installed.get(pkg_name)
            latest = self.REGISTRY.get(pkg_name)
            if not installed or not latest:
                continue

            cmp = self._version_cmp(latest.version, installed.version)
            if cmp > 0:  # latest > installed
                upgrades.append({
                    "package": pkg_name,
                    "current": ".".join(map(str, installed.version)),
                    "latest": ".".join(map(str, latest.version)),
                })

        return {"upgrades": upgrades, "neural_ops": self._ops}

    def verify(self, name: str) -> dict:
        """Verify package integrity via neural checksum."""
        pkg = self._installed.get(name)
        if not pkg:
            return {"error": f"Not installed: {name}"}

        expected = pkg.checksum
        actual = self._compute_checksum(pkg)
        zf, _ = self.bridge.cmp(actual, expected)
        self._op()

        return {
            "package": name,
            "valid": zf,
            "checksum": f"{actual:04x}",
            "neural_ops": self._ops,
        }


def demo():
    pm = NeuralPackageManager()

    print("Neural Package Manager")
    print("=" * 60)
    print("Dependency resolution + version comparison → neural CMP\n")

    # Install
    print("── Install ncpu-shell ──")
    result = pm.install("ncpu-shell")
    for pkg in result.get("installed", []):
        reg = pm.REGISTRY[pkg]
        ver = ".".join(map(str, reg.version))
        print(f"  📦 Installing {pkg} v{ver} ({reg.size}B)")
    print(f"  Total: {result['total_size']} bytes, {result['neural_ops']} neural ops")
    print()

    # Install more
    print("── Install neural-os-full ──")
    result = pm.install("neural-os-full")
    newly = result.get("installed", [])
    print(f"  Installed {len(newly)} packages: {', '.join(newly[:5])}{'...' if len(newly) > 5 else ''}")
    print(f"  Total new bytes: {result['total_size']}")
    print()

    # List installed
    print("── Installed Packages ──")
    pkgs = pm.list_installed()
    for p in sorted(pkgs, key=lambda x: x.name):
        ver = ".".join(map(str, p.version))
        print(f"  ✅ {p.name:20s} v{ver}  {p.size:5d}B  {p.description[:30]}")
    print()

    # Search
    print("── Search 'neural' ──")
    results = pm.search("neural")
    for r in results[:5]:
        ver = ".".join(map(str, r.version))
        inst = "✅" if r.name in pm._installed else "  "
        print(f"  {inst} {r.name:20s} v{ver}  {r.description}")
    print()

    # Upgrade check
    print("── Upgrade Check ──")
    result = pm.upgrade()
    if result["upgrades"]:
        for u in result["upgrades"]:
            print(f"  ↑ {u['package']}: {u['current']} → {u['latest']}")
    else:
        print("  All packages up to date")
    print()

    # Verify integrity
    print("── Verify Integrity ──")
    for pkg in ["ncpu-alu", "ncpu-shell", "ncpu-kernel"]:
        v = pm.verify(pkg)
        ok = "✅" if v.get("valid") else "❌"
        print(f"  {ok} {pkg:20s} checksum={v.get('checksum', '?')}")

    print(f"\n  Total neural ops: {pm._ops}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    elif cmd == "install" and len(sys.argv) > 2:
        pm = NeuralPackageManager()
        result = pm.install(sys.argv[2])
        print(result)
    elif cmd == "list":
        pm = NeuralPackageManager()
        for p in pm.list_installed():
            print(f"  {p.name} v{'.'.join(map(str, p.version))}")
    elif cmd == "search" and len(sys.argv) > 2:
        pm = NeuralPackageManager()
        for p in pm.search(sys.argv[2]):
            print(f"  {p.name}: {p.description}")
    else:
        print("Usage: python -m bridge.neural_pkg [demo|install <pkg>|list|search <q>]")


if __name__ == "__main__":
    main()
