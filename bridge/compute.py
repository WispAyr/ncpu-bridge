"""Core computation interface to nCPU's verified neural ALU."""

import re
import sys
import time
from pathlib import Path


NCPU_PATH = Path("/Users/noc/projects/nCPU")


def _ensure_ncpu():
    """Add nCPU to sys.path if not already present."""
    s = str(NCPU_PATH)
    if s not in sys.path:
        sys.path.insert(0, s)


class NCPUBridge:
    """Bridge to nCPU's verified neural ALU and GPU compute."""

    def __init__(self, ncpu_path: str = str(NCPU_PATH)):
        self.ncpu_path = Path(ncpu_path)
        _ensure_ncpu()

        from ncpu.model.neural_ops import NeuralOps

        self.neural_ops = NeuralOps(models_dir=str(self.ncpu_path / "models"))
        self._available = self.neural_ops.load()

    # ── Direct neural ALU operations ──────────────────────────────

    def add(self, a: int, b: int) -> int:
        return self.neural_ops.neural_add(a, b)

    def sub(self, a: int, b: int) -> int:
        return self.neural_ops.neural_sub(a, b)

    def mul(self, a: int, b: int) -> int:
        return self.neural_ops.neural_mul(a, b)

    def div(self, a: int, b: int) -> int:
        return self.neural_ops.neural_div(a, b)

    def cmp(self, a: int, b: int) -> tuple[bool, bool]:
        """Compare a and b. Returns (zero_flag, sign_flag)."""
        return self.neural_ops.neural_cmp(a, b)

    def bitwise_and(self, a: int, b: int) -> int:
        return self.neural_ops.neural_and(a, b)

    def bitwise_or(self, a: int, b: int) -> int:
        return self.neural_ops.neural_or(a, b)

    def bitwise_xor(self, a: int, b: int) -> int:
        return self.neural_ops.neural_xor(a, b)

    def shl(self, value: int, amount: int) -> int:
        return self.neural_ops.neural_shl(value, amount)

    def shr(self, value: int, amount: int) -> int:
        return self.neural_ops.neural_shr(value, amount)

    # ── Expression evaluator ─────────────────────────────────────

    _OP_MAP = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
        "&": "bitwise_and",
        "|": "bitwise_or",
        "^": "bitwise_xor",
        "<<": "shl",
        ">>": "shr",
    }

    _EXPR_RE = re.compile(
        r"^\s*(-?\d+)\s*(<<|>>|[+\-*/&|^])\s*(-?\d+)\s*$"
    )

    def calculate(self, expression: str) -> int:
        """Evaluate a binary arithmetic expression through the neural ALU.

        Supports: +, -, *, /, &, |, ^, <<, >>
        Example: "48 * 365" → 17520
        """
        m = self._EXPR_RE.match(expression)
        if not m:
            raise ValueError(
                f"Unsupported expression format: {expression!r}. "
                "Use '<int> <op> <int>' (e.g. '48 * 365')."
            )
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        method_name = self._OP_MAP.get(op)
        if method_name is None:
            raise ValueError(f"Unknown operator: {op!r}")
        return getattr(self, method_name)(a, b)

    # ── Assembly program execution ───────────────────────────────

    def run_program(self, assembly: str) -> dict:
        """Run an assembly program on the neural CPU and return register state."""
        from ncpu.model import CPU

        cpu = CPU(
            mock_mode=True,
            neural_execution=True,
            models_dir=str(self.ncpu_path / "models"),
        )
        cpu.load_program(assembly)
        cpu.run()
        return cpu.get_summary()

    # ── GPU Metal compute ────────────────────────────────────────

    def run_program_gpu(self, assembly: str, max_cycles: int = 1_000_000) -> dict:
        """Run an assembly program on the Metal GPU compute kernel."""
        from kernels.mlx.ncpu_kernel import NCPUComputeKernel

        kernel = NCPUComputeKernel()
        kernel.load_program_from_asm(assembly)
        result = kernel.execute(max_cycles=max_cycles)
        return {
            "cycles": result.cycles,
            "elapsed_seconds": result.elapsed_seconds,
            "ips": result.ips,
            "stop_reason": result.stop_reason_name,
            "registers": kernel.get_registers_dict(),
            "flags": kernel.get_flags(),
        }

    # ── Verification ─────────────────────────────────────────────

    _VERIFY_OPS = {
        "add": "add",
        "sub": "sub",
        "mul": "mul",
        "div": "div",
        "and": "bitwise_and",
        "or": "bitwise_or",
        "xor": "bitwise_xor",
        "shl": "shl",
        "shr": "shr",
    }

    def verify(self, operation: str, a: int, b: int, expected: int) -> bool:
        """Verify a computation result through the neural ALU.

        Returns True if neural ALU agrees with expected result.
        """
        method_name = self._VERIFY_OPS.get(operation.lower())
        if method_name is None:
            raise ValueError(
                f"Unknown operation: {operation!r}. "
                f"Supported: {', '.join(sorted(self._VERIFY_OPS))}"
            )
        neural_result = getattr(self, method_name)(a, b)
        return neural_result == expected

    # ── Benchmark ────────────────────────────────────────────────

    def benchmark(self, a: int = 12345, b: int = 6789, iterations: int = 100) -> dict:
        """Benchmark neural ALU vs native Python arithmetic."""
        ops = {
            "add": (lambda x, y: x + y),
            "sub": (lambda x, y: x - y),
            "mul": (lambda x, y: x * y),
        }
        results = {}
        for name, native_fn in ops.items():
            neural_fn = getattr(self, name)

            t0 = time.perf_counter()
            for _ in range(iterations):
                neural_fn(a, b)
            neural_elapsed = time.perf_counter() - t0

            t0 = time.perf_counter()
            for _ in range(iterations):
                native_fn(a, b)
            native_elapsed = time.perf_counter() - t0

            results[name] = {
                "neural_us": round(neural_elapsed / iterations * 1e6, 1),
                "native_us": round(native_elapsed / iterations * 1e6, 3),
                "ratio": round(neural_elapsed / max(native_elapsed, 1e-12), 1),
            }
        return results
