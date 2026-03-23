"""Neural Crypto — encryption and key derivation through nCPU.

Stream cipher and key operations using only neural ALU:
- XOR-based stream cipher (key XOR plaintext, bitwise through neural nets)
- Neural key derivation (repeated hash-like mixing via ADD/XOR/SHL)
- HMAC-like message authentication via neural hash
- Key exchange simulation (Diffie-Hellman-like with neural MUL/MOD)

NOT for actual security — this is a proof that the neural CPU can
execute cryptographic primitives correctly.

Usage:
    python -m bridge.neural_crypto demo           # Full crypto demo
    python -m bridge.neural_crypto encrypt <text>  # Encrypt with default key
    python -m bridge.neural_crypto decrypt <hex>   # Decrypt
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


class NeuralKeyDerivation:
    """Derive keys using neural ALU mixing operations.
    
    Takes a seed and produces a pseudorandom key stream by
    repeatedly mixing with neural XOR, ADD, and shift operations.
    
    Like a very simple KDF: hash(seed || counter) for each byte.
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
    
    def derive_key(self, seed: int, length: int) -> list[int]:
        """Generate a key stream of `length` bytes from a seed.
        
        Each byte: state = ((state XOR counter) + seed) SHL 3 XOR state
        All operations are neural.
        """
        state = seed
        key = []
        
        for i in range(length):
            # Mix: XOR with counter
            mixed = self.bridge.bitwise_xor(state, i)
            # Add seed back in
            mixed = self.bridge.add(mixed, seed)
            # Shift and fold
            shifted = self.bridge.shl(mixed, 3)
            state = self.bridge.bitwise_xor(shifted, mixed)
            # Extract byte (AND with 0xFF)
            byte = self.bridge.bitwise_and(state, 0xFF)
            key.append(byte)
        
        return key
    
    def derive_from_password(self, password: str) -> list[int]:
        """Derive a key from a password string.
        
        Folds all password bytes into a seed using neural XOR+ADD,
        then derives a 16-byte key.
        """
        # Fold password into seed
        seed = 0x5A5A  # Initial value
        for ch in password.encode():
            seed = self.bridge.bitwise_xor(seed, ch)
            seed = self.bridge.add(seed, ch)
            shifted = self.bridge.shl(seed, 2)
            seed = self.bridge.bitwise_xor(seed, shifted)
            seed = self.bridge.bitwise_and(seed, 0xFFFF)  # Keep 16-bit
        
        return self.derive_key(seed, 16)


class NeuralStreamCipher:
    """XOR stream cipher with neural operations.
    
    Encrypt: ciphertext[i] = plaintext[i] XOR keystream[i]
    Decrypt: plaintext[i] = ciphertext[i] XOR keystream[i]
    
    Symmetric — same operation encrypts and decrypts.
    Every XOR goes through a trained neural network.
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self.kdf = NeuralKeyDerivation()
        self._ops = 0
    
    def encrypt(self, plaintext: bytes, key_seed: int) -> list[int]:
        """Encrypt bytes using neural XOR stream cipher."""
        keystream = self.kdf.derive_key(key_seed, len(plaintext))
        ciphertext = []
        
        for i, byte in enumerate(plaintext):
            encrypted = self.bridge.bitwise_xor(byte, keystream[i])
            self._ops += 1
            ciphertext.append(encrypted)
        
        return ciphertext
    
    def decrypt(self, ciphertext: list[int], key_seed: int) -> bytes:
        """Decrypt — same as encrypt (XOR is its own inverse)."""
        keystream = self.kdf.derive_key(key_seed, len(ciphertext))
        plaintext = []
        
        for i, byte in enumerate(ciphertext):
            decrypted = self.bridge.bitwise_xor(byte, keystream[i])
            self._ops += 1
            plaintext.append(decrypted)
        
        return bytes(plaintext)
    
    def encrypt_string(self, text: str, password: str) -> list[int]:
        """Encrypt a string with a password."""
        key = self.kdf.derive_from_password(password)
        # Use first 2 bytes of derived key as seed for stream
        seed = self.bridge.add(self.bridge.shl(key[0], 8), key[1])
        return self.encrypt(text.encode(), seed)
    
    def decrypt_string(self, ciphertext: list[int], password: str) -> str:
        """Decrypt a ciphertext with a password."""
        key = self.kdf.derive_from_password(password)
        seed = self.bridge.add(self.bridge.shl(key[0], 8), key[1])
        return self.decrypt(ciphertext, seed).decode("utf-8", errors="replace")


class NeuralMAC:
    """Message Authentication Code using neural hashing.
    
    Computes: MAC = H(key || message)
    where H is a neural hash (folding XOR+ADD).
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
    
    def compute(self, message: bytes, key: list[int]) -> int:
        """Compute MAC of message with key."""
        # Prefix key to message
        data = key + list(message)
        
        # Neural hash: fold with XOR and ADD
        state = 0xDEAD
        for byte in data:
            state = self.bridge.bitwise_xor(state, byte)
            state = self.bridge.add(state, byte)
            shifted = self.bridge.shl(state, 5)
            state = self.bridge.bitwise_xor(state, shifted)
            state = self.bridge.bitwise_and(state, 0xFFFF)
        
        return state
    
    def verify(self, message: bytes, key: list[int], expected_mac: int) -> bool:
        """Verify a MAC using neural CMP."""
        actual = self.compute(message, key)
        zf, _ = self.bridge.cmp(actual, expected_mac)
        return zf


class NeuralDiffieHellman:
    """Simplified Diffie-Hellman-like key exchange using neural MUL.
    
    Uses small numbers (not cryptographically secure) to demonstrate
    the protocol flow with neural arithmetic.
    
    g=5, p=97 (small prime for demo)
    Alice: A = g^a mod p (neural)
    Bob:   B = g^b mod p (neural)  
    Shared: s = B^a mod p = A^b mod p
    """
    
    def __init__(self, g: int = 5, p: int = 97):
        self.bridge = NCPUBridge()
        self.g = g
        self.p = p
    
    def _neural_modpow(self, base: int, exp: int, mod: int) -> int:
        """Modular exponentiation using neural MUL and DIV.
        
        Simple repeated multiplication (not binary exponentiation
        because we want more neural ops for the demo).
        """
        result = 1
        for _ in range(exp):
            result = self.bridge.mul(result, base)
            # Neural modulo: result - (result / mod) * mod
            quotient = self.bridge.div(result, mod)
            product = self.bridge.mul(quotient, mod)
            result = self.bridge.sub(result, product)
        return result
    
    def generate_public(self, private_key: int) -> int:
        """Generate public key: g^private mod p."""
        return self._neural_modpow(self.g, private_key, self.p)
    
    def compute_shared(self, other_public: int, my_private: int) -> int:
        """Compute shared secret: other_public^my_private mod p."""
        return self._neural_modpow(other_public, my_private, self.p)


# ── CLI ──────────────────────────────────────────────────────

def demo():
    print("Neural Crypto Suite")
    print("=" * 60)
    print("Every XOR, shift, and multiply → trained neural network\n")
    
    # ── Key Derivation ──
    print("── Key Derivation ──")
    kdf = NeuralKeyDerivation()
    key = kdf.derive_from_password("parkwise2026")
    print(f"  Password: 'parkwise2026'")
    print(f"  Derived key (16 bytes): {' '.join(f'{b:02x}' for b in key)}")
    print()
    
    # ── Stream Cipher ──
    print("── Stream Cipher (encrypt/decrypt) ──")
    cipher = NeuralStreamCipher()
    
    plaintext = "nCPU"  # Keep short for speed
    password = "skynet"
    
    print(f"  Plaintext:  '{plaintext}'")
    print(f"  Password:   '{password}'")
    
    encrypted = cipher.encrypt_string(plaintext, password)
    print(f"  Ciphertext: {' '.join(f'{b:02x}' for b in encrypted)}")
    
    decrypted = cipher.decrypt_string(encrypted, password)
    print(f"  Decrypted:  '{decrypted}'")
    print(f"  Roundtrip:  {'✅' if decrypted == plaintext else '❌'}")
    print(f"  Neural XOR ops: {cipher._ops}")
    print()
    
    # ── MAC ──
    print("── Message Authentication ──")
    mac = NeuralMAC()
    message = b"POS is operational"
    mac_key = key[:8]
    
    tag = mac.compute(message, mac_key)
    verified = mac.verify(message, mac_key, tag)
    tampered = mac.verify(b"POS is down", mac_key, tag)
    
    print(f"  Message: '{message.decode()}'")
    print(f"  MAC: {tag:04x}")
    print(f"  Verify original: {'✅' if verified else '❌'}")
    print(f"  Verify tampered: {'❌ rejected' if not tampered else '⚠️ collision!'}")
    print()
    
    # ── Key Exchange ──
    print("── Diffie-Hellman Key Exchange ──")
    dh = NeuralDiffieHellman(g=5, p=97)
    
    alice_private = 7
    bob_private = 11
    
    alice_public = dh.generate_public(alice_private)
    bob_public = dh.generate_public(bob_private)
    
    alice_shared = dh.compute_shared(bob_public, alice_private)
    bob_shared = dh.compute_shared(alice_public, bob_private)
    
    print(f"  g={dh.g}, p={dh.p}")
    print(f"  Alice: private={alice_private}, public={alice_public}")
    print(f"  Bob:   private={bob_private}, public={bob_public}")
    print(f"  Alice's shared secret: {alice_shared}")
    print(f"  Bob's shared secret:   {bob_shared}")
    print(f"  Secrets match: {'✅' if alice_shared == bob_shared else '❌'}")
    print()
    
    print("All crypto operations computed through neural ALU ✅")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    if cmd == "demo":
        demo()
    elif cmd == "encrypt" and len(sys.argv) > 2:
        cipher = NeuralStreamCipher()
        text = " ".join(sys.argv[2:])
        encrypted = cipher.encrypt_string(text, "default")
        print(" ".join(f"{b:02x}" for b in encrypted))
    elif cmd == "decrypt" and len(sys.argv) > 2:
        cipher = NeuralStreamCipher()
        hex_bytes = sys.argv[2:]
        ciphertext = [int(h, 16) for h in hex_bytes]
        decrypted = cipher.decrypt_string(ciphertext, "default")
        print(decrypted)
    else:
        print("Usage: python -m bridge.neural_crypto [demo|encrypt <text>|decrypt <hex bytes>]")


if __name__ == "__main__":
    main()
