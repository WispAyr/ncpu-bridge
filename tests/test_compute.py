"""Tests for NCPUBridge — proves computation goes through neural nets."""

import pytest
from bridge.compute import NCPUBridge


@pytest.fixture(scope="module")
def bridge():
    return NCPUBridge()


class TestNeuralALU:
    """Test direct neural ALU operations."""

    def test_add(self, bridge):
        assert bridge.add(100, 200) == 300

    def test_sub(self, bridge):
        assert bridge.sub(500, 123) == 377

    def test_mul(self, bridge):
        assert bridge.mul(48, 365) == 17520

    def test_div(self, bridge):
        assert bridge.div(100, 3) == 33

    def test_add_negative(self, bridge):
        assert bridge.add(-10, 25) == 15

    def test_sub_negative_result(self, bridge):
        assert bridge.sub(10, 30) == -20

    def test_mul_large(self, bridge):
        assert bridge.mul(1000, 1000) == 1_000_000

    def test_cmp_equal(self, bridge):
        zf, sf = bridge.cmp(42, 42)
        assert zf is True
        assert sf is False

    def test_cmp_less(self, bridge):
        zf, sf = bridge.cmp(10, 42)
        assert zf is False
        assert sf is True

    def test_cmp_greater(self, bridge):
        zf, sf = bridge.cmp(42, 10)
        assert zf is False
        assert sf is False

    def test_bitwise_and(self, bridge):
        assert bridge.bitwise_and(0xFF, 0x0F) == 0x0F

    def test_bitwise_or(self, bridge):
        assert bridge.bitwise_or(0xF0, 0x0F) == 0xFF

    def test_bitwise_xor(self, bridge):
        assert bridge.bitwise_xor(0xFF, 0xFF) == 0

    def test_shl(self, bridge):
        assert bridge.shl(1, 8) == 256

    def test_shr(self, bridge):
        assert bridge.shr(256, 8) == 1


class TestCalculate:
    """Test expression evaluation through neural ALU."""

    def test_add_expression(self, bridge):
        assert bridge.calculate("100 + 200") == 300

    def test_mul_expression(self, bridge):
        assert bridge.calculate("48 * 365") == 17520

    def test_sub_expression(self, bridge):
        assert bridge.calculate("1000 - 1") == 999

    def test_div_expression(self, bridge):
        assert bridge.calculate("100 / 4") == 25

    def test_shift_left(self, bridge):
        assert bridge.calculate("1 << 10") == 1024

    def test_shift_right(self, bridge):
        assert bridge.calculate("1024 >> 10") == 1

    def test_negative_operand(self, bridge):
        assert bridge.calculate("-5 + 15") == 10

    def test_invalid_expression(self, bridge):
        with pytest.raises(ValueError):
            bridge.calculate("not math")


class TestVerify:
    """Test verification of computation results."""

    def test_verify_correct(self, bridge):
        assert bridge.verify("add", 100, 200, 300) is True

    def test_verify_incorrect(self, bridge):
        assert bridge.verify("add", 100, 200, 999) is False

    def test_verify_mul(self, bridge):
        assert bridge.verify("mul", 48, 365, 17520) is True

    def test_verify_unknown_op(self, bridge):
        with pytest.raises(ValueError):
            bridge.verify("sqrt", 4, 0, 2)


class TestRunProgram:
    """Test assembly execution on neural CPU."""

    def test_simple_program(self, bridge):
        result = bridge.run_program("MOV R0, 42\nHALT")
        assert result["registers"]["R0"] == 42
        assert result["halted"] is True

    def test_addition_program(self, bridge):
        asm = """\
MOV R0, 100
MOV R1, 200
ADD R2, R0, R1
HALT
"""
        result = bridge.run_program(asm)
        assert result["registers"]["R2"] == 300

    def test_loop_program(self, bridge):
        asm = """\
MOV R0, 0
MOV R1, 10
MOV R2, 1
loop:
ADD R0, R0, R2
DEC R1
JNZ loop
HALT
"""
        result = bridge.run_program(asm)
        assert result["registers"]["R0"] == 10
        assert result["registers"]["R1"] == 0


class TestBenchmark:
    def test_benchmark_runs(self, bridge):
        results = bridge.benchmark(iterations=5)
        assert "add" in results
        assert "sub" in results
        assert "mul" in results
        for data in results.values():
            assert data["neural_us"] > 0
            assert data["ratio"] >= 1
