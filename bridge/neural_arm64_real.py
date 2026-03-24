"""
Phase 42 — Real Neural ARM64 Decoder (Transformer + Attention)
================================================================
Uses nCPU's NeuralARM64Decoder — a Transformer that decodes 32-bit
ARM64 instructions. Each bit gets an embedding, self-attention runs
over all 32 bit positions, then cross-attention with 6 learned queries
extracts instruction fields.

Architecture:
  Input:  [batch, 32] binary (each bit = 0 or 1)
  Encoder: bit_embed(2,64) + pos_embed(32,64) → Linear(128,256) → [batch, 32, 256]
  FieldExtractor: self-attn(256,8h) → cross-attn(6 queries, 256, 8h) → [batch, 6, 256]
  Output: 6 field vectors, each 256-dim
"""

import sys
import torch
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from ncpu.model.architectures import NeuralARM64Decoder
from bridge.compute import NCPUBridge

bridge = NCPUBridge()

# ARM64 instruction encodings (real 32-bit values)
ARM64_EXAMPLES = {
    # MOV X0, #0         — MOVZ Xd, imm
    "MOV X0,#0":    0xD2800000,
    # MOV X1, #1
    "MOV X1,#1":    0xD2800021,
    # ADD X0, X1, X2
    "ADD X0,X1,X2": 0x8B020020,
    # SUB X0, X1, X2
    "SUB X0,X1,X2": 0xCB020020,
    # LDR X0, [X1]
    "LDR X0,[X1]":  0xF9400020,
    # STR X0, [X1]
    "STR X0,[X1]":  0xF9000020,
    # B #0 (branch to self)
    "B #0":         0x14000000,
    # BL #0 (call)
    "BL #0":        0x94000000,
    # RET
    "RET":          0xD65F03C0,
    # NOP
    "NOP":          0xD503201F,
}

def instr_to_bits(instr: int) -> torch.Tensor:
    """Convert 32-bit instruction to [32] binary tensor."""
    bits = []
    for i in range(31, -1, -1):
        bits.append((instr >> i) & 1)
    return torch.tensor(bits, dtype=torch.long)

def bits_to_hex(bits: torch.Tensor) -> str:
    val = 0
    for b in bits:
        val = (val << 1) | int(b.item())
    return f"0x{val:08X}"


class RealARM64Decoder:
    """Bridge to nCPU's Transformer ARM64 decoder."""

    def __init__(self):
        self.net = NeuralARM64Decoder()
        
        model_path = NCPU_PATH / "models" / "decoder" / "arm64_decoder.pt"
        if model_path.exists():
            self.net.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            self._pretrained = True
        else:
            self._pretrained = False
        self.net.eval()

        self._params = sum(p.numel() for p in self.net.parameters())

    def decode(self, instr: int) -> dict:
        """Decode a 32-bit ARM64 instruction through the Transformer."""
        bits = instr_to_bits(instr).unsqueeze(0)  # [1, 32]
        
        with torch.no_grad():
            fields = self.net(bits)  # [1, 6, 256]

        # fields[0] = [6, 256] — 6 extracted field vectors
        field_vecs = fields[0]

        # Extract useful info from field vectors
        # Field 0: instruction category (we can use argmax of projected vector)
        # Field 1-5: operand fields
        result = {
            "raw": f"0x{instr:08X}",
            "fields": field_vecs.shape,
            "field_norms": [field_vecs[i].norm().item() for i in range(6)],
            # Compute similarity between fields (same instruction → similar fields)
            "field0_mean": field_vecs[0].mean().item(),
            "field1_mean": field_vecs[1].mean().item(),
        }

        # Use field vectors to detect instruction type via simple heuristics on the bits
        bits_val = bits[0]
        op_bits = bits_val[:8]  # top 8 bits often encode opcode class

        # Classify using neural field vector norms
        norms = result["field_norms"]
        dom_field = max(range(6), key=lambda i: norms[i])
        result["dominant_field"] = dom_field

        return result

    def decode_batch(self, instrs: list[int]) -> torch.Tensor:
        """Batch decode instructions — single Transformer forward pass."""
        bits = torch.stack([instr_to_bits(i) for i in instrs])  # [N, 32]
        with torch.no_grad():
            fields = self.net(bits)  # [N, 6, 256]
        return fields


def demo():
    print("Real Neural ARM64 Decoder (Transformer + Attention)")
    print("=" * 60)
    print("Architecture: bit_embed → self-attn → cross-attn(6 queries) → fields")

    dec = RealARM64Decoder()
    print(f"Parameters: {dec._params:,}")
    print(f"Pretrained: {dec._pretrained}\n")

    # Decode individual instructions
    print("  Individual instruction decoding:")
    for name, enc in list(ARM64_EXAMPLES.items())[:6]:
        result = dec.decode(enc)
        norms_str = " ".join(f"{n:.2f}" for n in result["field_norms"])
        print(f"    {name:20s} {result['raw']}  field_norms=[{norms_str}]")

    # Batch decode — single forward pass for all instructions
    print(f"\n  Batch decode ({len(ARM64_EXAMPLES)} instructions, 1 forward pass):")
    all_enc = list(ARM64_EXAMPLES.values())
    all_names = list(ARM64_EXAMPLES.keys())
    batch_fields = dec.decode_batch(all_enc)  # [N, 6, 256]
    print(f"    Input:  {len(all_enc)} × 32-bit instructions")
    print(f"    Output: {list(batch_fields.shape)} field vectors")

    # Compute instruction similarity via field vectors
    print(f"\n  Instruction similarity (cosine of field[0] vectors):")
    field0 = batch_fields[:, 0, :]  # [N, 256] — take first field
    # Normalise
    norms = field0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    field0_norm = field0 / norms

    # Find most similar pairs
    sim = field0_norm @ field0_norm.T  # [N, N]
    print(f"    {'':20s}", end="")
    for n in all_names:
        print(f"  {n[:5]:5s}", end="")
    print()
    for i, ni in enumerate(all_names):
        print(f"    {ni:20s}", end="")
        for j in range(len(all_names)):
            s = sim[i, j].item()
            print(f"  {s:+.2f}", end="")
        print()

    # Show that similar instructions cluster together
    add_idx = all_names.index("ADD X0,X1,X2")
    sub_idx = all_names.index("SUB X0,X1,X2")
    ldr_idx = all_names.index("LDR X0,[X1]")
    str_idx = all_names.index("STR X0,[X1]")
    b_idx = all_names.index("B #0")
    bl_idx = all_names.index("BL #0")

    add_sub_sim = sim[add_idx, sub_idx].item()
    ldr_str_sim = sim[ldr_idx, str_idx].item()
    b_bl_sim = sim[b_idx, bl_idx].item()
    add_ldr_sim = sim[add_idx, ldr_idx].item()

    print(f"\n  Semantic grouping (higher similarity = same instruction family):")
    print(f"    ADD  ↔ SUB  (arithmetic):   {add_sub_sim:+.4f}")
    print(f"    LDR  ↔ STR  (memory):        {ldr_str_sim:+.4f}")
    print(f"    B    ↔ BL   (branches):      {b_bl_sim:+.4f}")
    print(f"    ADD  ↔ LDR  (different):     {add_ldr_sim:+.4f}")

    # Check grouping makes sense
    arith_closer_than_cross = add_sub_sim > add_ldr_sim
    mem_closer = ldr_str_sim > add_ldr_sim
    branch_closer = b_bl_sim > add_ldr_sim

    print(f"\n    Arithmetic cluster: {'✅' if arith_closer_than_cross else '❌'}")
    print(f"    Memory cluster:     {'✅' if mem_closer else '❌'}")
    print(f"    Branch cluster:     {'✅' if branch_closer else '❌'}")

    print(f"\n  Parameters: {dec._params:,}")
    print(f"\n✅ Real ARM64 Transformer decoder: bit embeddings + attention field extraction")


if __name__ == "__main__":
    demo()
