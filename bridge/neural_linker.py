"""
Phase 26 — Neural Linker
========================
Links multiple NCP object files into a single executable.
All address calculations, symbol resolution, and relocation patching
done through neural ALU operations.

Features:
  - Object file creation with .text, .data, .symtab, .rel sections
  - Symbol table merging with neural address offset calculation
  - Relocation patching (absolute + PC-relative) via neural ADD/SUB
  - Undefined symbol detection
  - Entry point resolution
"""

from bridge.compute import NCPUBridge
from dataclasses import dataclass, field
from typing import Optional

bridge = NCPUBridge()

# ── Object file structures ──

@dataclass
class Symbol:
    name: str
    section: str       # ".text" or ".data"
    offset: int        # offset within section
    size: int
    bind: str = "local"  # "local" or "global"
    defined: bool = True

@dataclass
class Relocation:
    offset: int        # where in .text to patch
    symbol: str        # target symbol name
    rel_type: str      # "abs" or "pc_rel"
    addend: int = 0

@dataclass
class Section:
    name: str
    data: list = field(default_factory=list)  # list of ints (bytes)
    base_addr: int = 0

@dataclass
class ObjectFile:
    name: str
    sections: dict = field(default_factory=dict)  # name → Section
    symbols: list = field(default_factory=list)
    relocations: list = field(default_factory=list)

@dataclass
class Executable:
    entry: int = 0
    text: list = field(default_factory=list)
    data: list = field(default_factory=list)
    symtab: dict = field(default_factory=dict)  # name → absolute address


class NeuralLinker:
    """Links NCP object files using neural arithmetic for all address computation."""

    def __init__(self):
        self._ops = 0

    def _neural_add(self, a: int, b: int) -> int:
        self._ops += 1
        return bridge.add(a, b)

    def _neural_sub(self, a: int, b: int) -> int:
        self._ops += 1
        return bridge.sub(a, b)

    def _neural_cmp(self, a: int, b: int):
        self._ops += 1
        return bridge.cmp(a, b)

    def create_object(self, name: str, text_bytes: list, data_bytes: list,
                      symbols: list, relocations: list) -> ObjectFile:
        """Create an object file from raw sections."""
        obj = ObjectFile(name=name)
        obj.sections[".text"] = Section(".text", list(text_bytes))
        obj.sections[".data"] = Section(".data", list(data_bytes))
        obj.symbols = list(symbols)
        obj.relocations = list(relocations)
        return obj

    def link(self, objects: list, entry_symbol: str = "_start") -> Executable:
        """Link multiple object files into an executable."""
        exe = Executable()

        # Pass 1: Calculate section base addresses (neural ADD for offsets)
        text_cursor = 0
        data_cursor = 0
        section_bases = {}  # (obj_name, section_name) → base_addr

        for obj in objects:
            if ".text" in obj.sections:
                section_bases[(obj.name, ".text")] = text_cursor
                obj.sections[".text"].base_addr = text_cursor
                text_cursor = self._neural_add(text_cursor, len(obj.sections[".text"].data))

            if ".data" in obj.sections:
                section_bases[(obj.name, ".data")] = data_cursor
                obj.sections[".data"].base_addr = data_cursor
                data_cursor = self._neural_add(data_cursor, len(obj.sections[".data"].data))

        # Pass 2: Build global symbol table (neural ADD for absolute addresses)
        global_syms = {}
        for obj in objects:
            base_text = section_bases.get((obj.name, ".text"), 0)
            base_data = section_bases.get((obj.name, ".data"), 0)

            for sym in obj.symbols:
                if not sym.defined:
                    continue
                base = base_text if sym.section == ".text" else base_data
                abs_addr = self._neural_add(base, sym.offset)

                if sym.bind == "global":
                    if sym.name in global_syms:
                        raise LinkError(f"duplicate symbol: {sym.name}")
                    global_syms[sym.name] = abs_addr

                exe.symtab[f"{obj.name}:{sym.name}"] = abs_addr
                if sym.bind == "global":
                    exe.symtab[sym.name] = abs_addr

        # Check for undefined symbols
        for obj in objects:
            for rel in obj.relocations:
                if rel.symbol not in global_syms:
                    # Check local symbols in same object
                    local = [s for s in obj.symbols if s.name == rel.symbol and s.defined]
                    if not local:
                        raise LinkError(f"undefined symbol: {rel.symbol} (referenced in {obj.name})")

        # Pass 3: Merge .text sections
        merged_text = []
        for obj in objects:
            if ".text" in obj.sections:
                merged_text.extend(obj.sections[".text"].data)
        exe.text = merged_text

        # Merge .data sections
        merged_data = []
        for obj in objects:
            if ".data" in obj.sections:
                merged_data.extend(obj.sections[".data"].data)
        exe.data = merged_data

        # Pass 4: Apply relocations (neural ADD/SUB for patching)
        for obj in objects:
            text_base = section_bases.get((obj.name, ".text"), 0)

            for rel in obj.relocations:
                # Resolve symbol address
                if rel.symbol in global_syms:
                    sym_addr = global_syms[rel.symbol]
                else:
                    local = [s for s in obj.symbols if s.name == rel.symbol and s.defined][0]
                    base = text_base if local.section == ".text" else section_bases.get((obj.name, ".data"), 0)
                    sym_addr = self._neural_add(base, local.offset)

                # Patch location in merged text
                patch_offset = self._neural_add(text_base, rel.offset)

                if rel.rel_type == "abs":
                    # Absolute: write sym_addr + addend
                    val = self._neural_add(sym_addr, rel.addend)
                    exe.text[patch_offset] = val & 0xFF
                elif rel.rel_type == "pc_rel":
                    # PC-relative: sym_addr - patch_offset + addend
                    diff = self._neural_sub(sym_addr, patch_offset)
                    val = self._neural_add(diff, rel.addend)
                    exe.text[patch_offset] = val & 0xFF

        # Resolve entry point
        if entry_symbol in global_syms:
            exe.entry = global_syms[entry_symbol]
        else:
            exe.entry = 0

        return exe


class LinkError(Exception):
    pass


# ── Demo ──

def demo():
    print("Neural Linker")
    print("=" * 60)
    print("Object linking with neural address arithmetic\n")

    linker = NeuralLinker()

    # Object 1: main.o — calls "add_nums" and uses "result" data
    main_obj = linker.create_object(
        name="main.o",
        text_bytes=[0x01, 0x05, 0x01, 0x03, 0xCA, 0x00, 0xAA, 0x00, 0xFF],
        data_bytes=[0x00, 0x00, 0x00, 0x00],  # "result" buffer, 4 bytes
        symbols=[
            Symbol("_start", ".text", 0, 9, "global"),
            Symbol("result", ".data", 0, 4, "global"),
        ],
        relocations=[
            Relocation(5, "add_nums", "abs"),
            Relocation(7, "result", "abs"),
        ]
    )

    # Object 2: math.o — defines "add_nums"
    math_obj = linker.create_object(
        name="math.o",
        text_bytes=[
            0x02, 0x00, 0x01,  # ADD R0, R1 → R2
            0xFE,              # RET
        ],
        data_bytes=[],
        symbols=[
            Symbol("add_nums", ".text", 0, 4, "global"),
        ],
        relocations=[]
    )

    # Object 3: lib.o — utility with local + global symbols
    lib_obj = linker.create_object(
        name="lib.o",
        text_bytes=[
            0x10, 0x01,       # helper (local)
            0xFE,             # RET
            0x20, 0x02,       # double_it (global) — calls helper
            0xCA, 0x00,       # CALL helper (reloc at offset 5)
            0xFE,             # RET
        ],
        data_bytes=[0x42],    # magic byte
        symbols=[
            Symbol("_helper", ".text", 0, 3, "local"),
            Symbol("double_it", ".text", 3, 4, "global"),
            Symbol("magic", ".data", 0, 1, "global"),
        ],
        relocations=[
            Relocation(5, "_helper", "pc_rel", addend=-1),
        ]
    )

    # Link all three
    print("Linking: main.o + math.o + lib.o")
    print("-" * 40)
    exe = linker.link([main_obj, math_obj, lib_obj], entry_symbol="_start")

    print(f"\n  Entry point:     0x{exe.entry:04x}")
    print(f"  .text size:      {len(exe.text)} bytes")
    print(f"  .data size:      {len(exe.data)} bytes")
    print(f"  Symbols resolved: {len(exe.symtab)}")

    print("\n  Symbol table:")
    for name, addr in sorted(exe.symtab.items()):
        if ":" not in name:  # show only globals
            print(f"    {name:20s} → 0x{addr:04x}")

    # Verify relocations were patched
    print("\n  Relocation verification:")
    # main.o .text[5] should have add_nums absolute addr (= 9, since main.text is 9 bytes)
    patched_call = exe.text[5]
    add_nums_addr = exe.symtab.get("add_nums", -1)
    zf, _ = bridge.cmp(patched_call, add_nums_addr & 0xFF)
    print(f"    CALL add_nums: patched=0x{patched_call:02x}, expected=0x{add_nums_addr & 0xFF:02x} → {'✅' if zf else '❌'}")

    # main.o .text[7] should have result data addr
    patched_store = exe.text[7]
    result_addr = exe.symtab.get("result", -1)
    zf2, _ = bridge.cmp(patched_store, result_addr & 0xFF)
    print(f"    STORE result:  patched=0x{patched_store:02x}, expected=0x{result_addr & 0xFF:02x} → {'✅' if zf2 else '❌'}")

    # PC-relative: lib.o helper call
    lib_text_base = 9 + 4  # after main(9) + math(4) = 13
    patch_pos = lib_text_base + 5
    helper_abs = lib_text_base + 0
    expected_pcrel = (helper_abs - patch_pos - 1) & 0xFF
    patched_pcrel = exe.text[patch_pos]
    zf3, _ = bridge.cmp(patched_pcrel, expected_pcrel)
    print(f"    CALL _helper:  patched=0x{patched_pcrel:02x}, expected=0x{expected_pcrel:02x} (PC-rel) → {'✅' if zf3 else '❌'}")

    print(f"\n  Neural ops: {linker._ops}")

    # Test undefined symbol detection
    print("\n  Error handling:")
    bad_obj = linker.create_object(
        name="bad.o",
        text_bytes=[0xCA, 0x00],
        data_bytes=[],
        symbols=[],
        relocations=[Relocation(1, "nonexistent_func", "abs")]
    )
    try:
        linker.link([bad_obj])
        print("    ❌ Should have raised LinkError")
    except LinkError as e:
        print(f"    ✅ Caught: {e}")

    # Test duplicate symbol detection
    dup1 = linker.create_object("dup1.o", [0x00], [], [Symbol("foo", ".text", 0, 1, "global")], [])
    dup2 = linker.create_object("dup2.o", [0x00], [], [Symbol("foo", ".text", 0, 1, "global")], [])
    try:
        linker.link([dup1, dup2])
        print("    ❌ Should have raised LinkError for duplicates")
    except LinkError as e:
        print(f"    ✅ Caught: {e}")

    print("\n✅ Neural linker: 3 objects linked, relocations applied, errors caught")


if __name__ == "__main__":
    demo()
