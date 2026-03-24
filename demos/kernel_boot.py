#!/usr/bin/env python3
"""Kernel boot demo: boot the neural kernel end-to-end, show all 11 subsystems."""

import sys
from pathlib import Path

BRIDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BRIDGE_ROOT))

from bridge.neural_kernel import NeuralKernel


def run_demo():
    print("Booting Neural Kernel — all 11 subsystems\n")

    kernel = NeuralKernel()
    kernel.boot()

    # Exercise syscalls
    print()
    print("── Post-boot syscalls ──")
    for call in ["hostname", "version", "uptime"]:
        result = kernel.syscall(call)
        print(f"  syscall({call}) → {result}")

    result = kernel.syscall("resolve", "ncpu.local")
    print(f"  syscall(resolve, ncpu.local) → {result}")

    # Show filesystem tree
    fs = kernel._subsystems.get("fs")
    if fs:
        print()
        print("── Filesystem after boot ──")
        listing = fs.ls("/")
        for entry in listing.get("entries", []):
            kind = "dir " if entry.get("is_dir") else "file"
            print(f"  [{kind}] /{entry['name']}")

    print()
    print(f"Subsystems loaded: {len(kernel._subsystems)}")
    print("Done.")


if __name__ == "__main__":
    run_demo()
