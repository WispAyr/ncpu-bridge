"""Tests for HealthComputer."""

import pytest
from bridge.health import HealthComputer


@pytest.fixture(scope="module")
def hc():
    return HealthComputer()


class TestCheckThreshold:
    def test_below_threshold(self, hc):
        result = hc.check_threshold(62, 90, "disk_pct")
        assert result["exceeded"] is False
        assert result["headroom"] == 28
        assert result["name"] == "disk_pct"

    def test_above_threshold(self, hc):
        result = hc.check_threshold(95, 90, "cpu_pct")
        assert result["exceeded"] is True

    def test_at_threshold(self, hc):
        result = hc.check_threshold(90, 90, "mem_pct")
        assert result["exceeded"] is False
        assert result["headroom"] == 0


class TestComputeStats:
    def test_basic_stats(self, hc):
        result = hc.compute_stats([10, 20, 30, 40, 50])
        assert result["sum"] == 150
        assert result["min"] == 10
        assert result["max"] == 50
        assert result["count"] == 5
        assert result["avg"] == 30

    def test_single_value(self, hc):
        result = hc.compute_stats([42])
        assert result["sum"] == 42
        assert result["min"] == 42
        assert result["max"] == 42
        assert result["avg"] == 42

    def test_empty(self, hc):
        result = hc.compute_stats([])
        assert result["sum"] == 0
        assert result["count"] == 0


class TestCheckThresholdASM:
    def test_below_threshold_asm(self, hc):
        result = hc.check_threshold_asm(62, 90)
        assert result["exceeded"] is False
        assert result["headroom"] == 28

    def test_above_threshold_asm(self, hc):
        result = hc.check_threshold_asm(95, 90)
        assert result["exceeded"] is True
