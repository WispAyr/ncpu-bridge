"""
Phase 36 — Real Neural MMU Integration
========================================
Uses the actual trained NeuralMMU from nCPU's neurOS — a neural network
that learns virtual→physical address translation. VPN+ASID embeddings
fed through an MLP to predict physical frame numbers + permission bits.

This replaces our manual page table in neural_vm.py with the real deal:
a neural network doing address translation in a single forward pass.
"""

import sys
import torch
from pathlib import Path

from bridge.config import get_ncpu_path, get_bridge_path, get_clawd_data_path
NCPU_PATH = get_ncpu_path()
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from ncpu.os.neuros.mmu import NeuralMMU, PageFault, PAGE_SIZE
from bridge.compute import NCPUBridge

bridge = NCPUBridge()


class RealNeuralMMU:
    """Bridge wrapper around nCPU's actual NeuralMMU.
    
    The MMU contains a NeuralPageTable network:
      VPN → Embedding(4096, 64) + ASID → Embedding(256, 16)
      → MLP(80→256→256→4102) → PFN classification + permission bits
    
    After training on page table mappings, all translations are
    neural network forward passes — no page table walks.
    """

    def __init__(self, max_pages=64, max_frames=64):
        self.mmu = NeuralMMU(
            max_virtual_pages=max_pages,
            max_physical_frames=max_frames,
            device=torch.device("cpu")  # keep on CPU for bridge compat
        )
        self._bridge_ops = 0

    def map_region(self, base_vpn: int, num_pages: int, 
                   read=True, write=False, execute=False) -> list:
        """Map a contiguous region of virtual pages, return allocated PFNs."""
        pfns = []
        for i in range(num_pages):
            vpn = bridge.add(base_vpn, i)
            self._bridge_ops += 1
            pfn = self.mmu.alloc_and_map(vpn, asid=0, 
                                          read=read, write=write, execute=execute)
            pfns.append(pfn)
        return pfns

    def train(self, epochs=200) -> dict:
        """Train the neural page table on current mappings."""
        return self.mmu.train_from_table(epochs=epochs, lr=1e-3)

    def translate(self, virtual_addr: int, write=False, execute=False):
        """Translate virtual → physical using the neural network."""
        phys, fault = self.mmu.translate(virtual_addr, asid=0, 
                                          write=write, execute=execute)
        return phys, fault

    def translate_batch(self, addrs: list) -> list:
        """Batch translate using neural network (single forward pass)."""
        t = torch.tensor(addrs, dtype=torch.int64)
        results = self.mmu.translate_batch(t, asid=0)
        return results.tolist()

    def stats(self):
        s = self.mmu.stats()
        s['bridge_ops'] = self._bridge_ops
        return s


def demo():
    print("Real Neural MMU (from nCPU neurOS)")
    print("=" * 60)
    print("Neural network learns virtual→physical address translation")
    print(f"Architecture: VPN Embedding + ASID Embedding → MLP → PFN + Perms\n")

    mmu = RealNeuralMMU(max_pages=64, max_frames=64)

    # Map some virtual pages
    print("  Mapping virtual pages...")
    
    # Code segment: pages 0-3 (read + execute)
    code_pfns = mmu.map_region(0, 4, read=True, execute=True)
    print(f"    .text  VPN 0-3  → PFN {code_pfns}  (r-x)")

    # Data segment: pages 4-7 (read + write)
    data_pfns = mmu.map_region(4, 4, read=True, write=True)
    print(f"    .data  VPN 4-7  → PFN {data_pfns}  (rw-)")

    # Stack: pages 8-11 (read + write)
    stack_pfns = mmu.map_region(8, 4, read=True, write=True)
    print(f"    stack  VPN 8-11 → PFN {stack_pfns}  (rw-)")

    # Heap: pages 16-19 (read + write)
    heap_pfns = mmu.map_region(16, 4, read=True, write=True)
    print(f"    heap   VPN 16-19 → PFN {heap_pfns}  (rw-)")

    stats = mmu.stats()
    print(f"\n    Mapped: {stats['mapped_pages']} pages, {stats['free_frames']} frames free")

    # Train the neural page table
    print("\n  Training neural page table...")
    train_stats = mmu.train(epochs=300)
    print(f"    Epochs: {train_stats.get('epochs', '?')}")
    print(f"    Accuracy: {train_stats.get('final_accuracy', 0)*100:.1f}%")
    print(f"    Best: {train_stats.get('best_accuracy', 0)*100:.1f}%")
    print(f"    Loss: {train_stats.get('final_loss', '?'):.4f}" if isinstance(train_stats.get('final_loss'), float) else "")

    # Test translations
    print("\n  Neural translations:")
    test_addrs = [
        (0x0000, "code base"),      # VPN 0
        (0x1004, "code offset"),    # VPN 1, offset 4
        (0x4000, "data base"),      # VPN 4
        (0x5ABC, "data mid"),       # VPN 5, offset 0xABC
        (0x8000, "stack base"),     # VPN 8
    ]

    correct = 0
    for vaddr, desc in test_addrs:
        phys, fault = mmu.translate(vaddr)
        vpn = vaddr >> 12
        offset = vaddr & 0xFFF
        
        if fault:
            print(f"    0x{vaddr:05x} ({desc:15s}) → FAULT: {fault.fault_type}")
        else:
            # Verify against expected PFN
            expected_pfn = int(mmu.mmu.page_table_pfn[0, vpn].item())
            expected_phys = expected_pfn * PAGE_SIZE + offset
            ok = phys == expected_phys
            if ok:
                correct += 1
            print(f"    0x{vaddr:05x} ({desc:15s}) → 0x{phys:05x} "
                  f"(PFN={phys // PAGE_SIZE}) {'✅' if ok else '❌'}")

    # Test page fault on unmapped region
    phys, fault = mmu.translate(0xF000)  # VPN 15, not mapped
    fault_ok = fault is not None
    print(f"    0x0F000 (unmapped       ) → {'FAULT ✅' if fault_ok else 'NO FAULT ❌'}: {fault.fault_type if fault else ''}")

    # Permission check: write to code segment should fault
    phys, fault = mmu.translate(0x0000, write=True)
    perm_ok = fault is not None
    print(f"    0x00000 (write to .text ) → {'FAULT ✅' if perm_ok else 'NO FAULT ❌'}: {fault.fault_type if fault else ''}")

    # Batch translation
    print("\n  Batch translation (single forward pass):")
    batch_addrs = [0x0000, 0x1000, 0x4000, 0x8000]
    batch_results = mmu.translate_batch(batch_addrs)
    for va, pa in zip(batch_addrs, batch_results):
        print(f"    0x{va:05x} → 0x{int(pa):05x}" if pa >= 0 else f"    0x{va:05x} → FAULT")

    final = mmu.stats()
    print(f"\n  Stats: {final['translations']} translations, {final['page_faults']} faults")
    print(f"  Neural model: trained={final['trained']}")
    print(f"  Bridge neural ops: {final['bridge_ops']}")

    net_params = sum(p.numel() for p in mmu.mmu.net.parameters())
    print(f"  NeuralPageTable params: {net_params:,}")

    print(f"\n  Translation accuracy: {correct}/{len(test_addrs)}")
    print(f"\n✅ Real Neural MMU: trained page table, neural translations, permission checks")


if __name__ == "__main__":
    demo()
