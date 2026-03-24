"""
Phase 43 — Real Neural Assembler Pipeline
==========================================
Uses nCPU's full neural assembler — tokenizer CNN + codegen MLP.

Pipeline:
  Source text → NeuralTokenizerNet (CNN) → token classes
  Token classes → NeuralCodeGenNet (MLP) → 32-bit encoding

Tokenizer: char → Embedding(128,32) → Conv1d(64,k=5) → Conv1d(32,k=3) → Linear(8 classes)
CodeGen:   [opcode(8)+rd(8)+rs1(8)+rs2(8)+imm(16)+fmt(8)] → MLP(56→128→128→128→32)
"""

import sys
import torch
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from ncpu.os.neuros.assembler import (
    NeuralTokenizerNet, NeuralCodeGenNet,
    ClassicalAssembler, NeuralAssembler,
    encode_instruction_features, Opcode, AsmToken
)
from bridge.compute import NCPUBridge

bridge = NCPUBridge()

TOKEN_CLASSES = ["OPCODE", "REGISTER", "IMMEDIATE", "LABEL", "COMMA", "COLON", "NEWLINE", "OTHER"]


class RealNeuralAssembler:
    """Bridge to nCPU's neural assembler pipeline."""

    def __init__(self):
        # Tokenizer CNN
        self.tokenizer = NeuralTokenizerNet(vocab_size=128, embed_dim=32, num_classes=8)
        tok_path = NCPU_PATH / "models" / "os" / "assembler_tokenizer.pt"
        if tok_path.exists():
            self.tokenizer.load_state_dict(
                torch.load(tok_path, map_location="cpu", weights_only=True)
            )
            self._tok_pretrained = True
        else:
            self._tok_pretrained = False
        self.tokenizer.eval()

        # CodeGen MLP — pretrained with hidden_dim=256
        self.codegen = NeuralCodeGenNet(hidden_dim=256)
        # NeuralAssembler internal also needs hidden_dim=256 to match checkpoint
        cg_path = NCPU_PATH / "models" / "os" / "assembler_codegen.pt"
        if cg_path.exists():
            self.codegen.load_state_dict(
                torch.load(cg_path, map_location="cpu", weights_only=True)
            )
            self._cg_pretrained = True
        else:
            self._cg_pretrained = False
        self.codegen.eval()

        # Classical assembler for ground truth
        self.classical = ClassicalAssembler()

        # Neural assembler wrapping both
        self.neural = NeuralAssembler()
        # Patch to correct hidden_dim before loading checkpoint
        self.neural.codegen_net = NeuralCodeGenNet(hidden_dim=256)
        if tok_path.exists():
            self.neural.tokenizer_net.load_state_dict(
                torch.load(tok_path, map_location="cpu", weights_only=True)
            )
        if cg_path.exists():
            self.neural.codegen_net.load_state_dict(
                torch.load(cg_path, map_location="cpu", weights_only=True)
            )
        self.neural.codegen_net.eval()

        self._tok_params = sum(p.numel() for p in self.tokenizer.parameters())
        self._cg_params = sum(p.numel() for p in self.codegen.parameters())

    def tokenize(self, source: str) -> list:
        """Run the CNN tokenizer on assembly source text."""
        # Encode as character indices
        chars = [ord(c) % 128 for c in source[:128]]
        if not chars:
            return []
        char_tensor = torch.tensor(chars).unsqueeze(0)  # [1, L]

        with torch.no_grad():
            logits = self.tokenizer(char_tensor)  # [1, L, 8]

        # Take argmax for each character
        classes = logits[0].argmax(dim=-1).tolist()
        tokens = []
        for ch, cls in zip(source[:128], classes):
            tokens.append((ch, TOKEN_CLASSES[cls]))
        return tokens

    def encode_instruction(self, opcode: int, rd: int = 0, rs1: int = 0,
                            rs2: int = 0, imm: int = 0, fmt: int = 0) -> int:
        """Encode one instruction via the neural codegen MLP."""
        features = encode_instruction_features(opcode, rd, rs1, rs2, imm, fmt)
        features = features.unsqueeze(0)  # [1, 56]

        with torch.no_grad():
            bit_logits = self.codegen(features)  # [1, 32]

        # Convert bit logits to integer
        bits = (bit_logits[0] > 0).int().tolist()
        result = 0
        for b in bits:
            result = (result << 1) | b
        return result

    def assemble_classical(self, source: str):
        """Assemble using classical (ground truth) assembler."""
        return self.classical.assemble(source)

    def assemble_neural(self, source: str):
        """Assemble using neural pipeline."""
        return self.neural.assemble_neural(source)


def demo():
    print("Real Neural Assembler Pipeline")
    print("=" * 60)
    print("CNN tokenizer + MLP code generator\n")

    asm = RealNeuralAssembler()
    print(f"Tokenizer:  {asm._tok_params:,} params (pretrained={asm._tok_pretrained})")
    print(f"CodeGen:    {asm._cg_params:,} params (pretrained={asm._cg_pretrained})\n")

    # Test program
    source = """
MOV R0, 10
MOV R1, 20
ADD R2, R0, R1
MUL R3, R2, R0
HALT
"""

    # Classical assembly
    print("  Classical assembly:")
    classical = asm.assemble_classical(source.strip())
    if classical.success:
        for i, (enc, instr) in enumerate(zip(classical.binary, classical.instructions)):
            print(f"    {instr.source or f'instr[{i}]':20s} → 0x{enc:08X}")
    else:
        print(f"    Errors: {classical.errors}")

    # Neural tokenization
    print(f"\n  Neural tokenizer (CNN, char-level):")
    tokens = asm.tokenize("MOV R0, 10")
    # Group consecutive same-class characters
    groups = []
    cur_text = ""
    cur_cls = None
    for ch, cls in tokens:
        if cls == cur_cls:
            cur_text += ch
        else:
            if cur_text and cur_cls != "OTHER":
                groups.append((cur_text.strip(), cur_cls))
            cur_text = ch
            cur_cls = cls
    if cur_text and cur_cls != "OTHER":
        groups.append((cur_text.strip(), cur_cls))

    for text, cls in groups:
        if text.strip():
            print(f"    '{text}' → {cls}")

    # Neural codegen: encode instructions
    print(f"\n  Neural codegen (MLP, 56→128→128→128→32 bits):")
    test_instrs = [
        (Opcode.MOV_IMM.value, 0, 0, 0, 10, 1, "MOV R0, 10"),
        (Opcode.MOV_IMM.value, 1, 0, 0, 20, 1, "MOV R1, 20"),
        (Opcode.ADD.value,     2, 0, 1, 0,  0, "ADD R2, R0, R1"),
        (Opcode.MUL.value,     3, 2, 0, 0,  0, "MUL R3, R2, R0"),
        (Opcode.HALT.value,    0, 0, 0, 0,  0, "HALT"),
    ]

    neural_correct = 0
    for opcode, rd, rs1, rs2, imm, fmt, desc in test_instrs:
        neural_enc = asm.encode_instruction(opcode, rd, rs1, rs2, imm, fmt)

        print(f"    {desc:20s} → neural=0x{neural_enc:08X}")

    # Full neural assembly
    print(f"\n  Full neural assembly pipeline:")
    try:
        neural_result = asm.assemble_neural(source.strip())
        if neural_result.success:
            print(f"    ✅ Assembled {neural_result.num_instructions} instructions")
            for enc, instr in zip(neural_result.binary, neural_result.instructions):
                print(f"       {instr.source or '?':20s} → 0x{enc:08X}")
        else:
            print(f"    Pipeline result: {neural_result.error or 'partial assembly'}")
    except Exception as e:
        print(f"    Neural pipeline: {e}")

    print(f"\n  Model sizes:")
    print(f"    Tokenizer CNN: {asm._tok_params:,} params")
    print(f"    CodeGen MLP:   {asm._cg_params:,} params")
    print(f"    Total:         {asm._tok_params + asm._cg_params:,} params")
    print(f"\n✅ Real neural assembler: CNN tokenizer + MLP code generator")


if __name__ == "__main__":
    demo()
