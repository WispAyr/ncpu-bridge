"""Tests for ObligationChecker."""

import pytest
from bridge.obligations import ObligationChecker


@pytest.fixture(scope="module")
def oc():
    return ObligationChecker()


class TestCheckInterval:
    def test_not_overdue(self, oc):
        result = oc.check_interval(
            last_run_epoch=1000, now_epoch=1500, interval_seconds=3600
        )
        assert result["overdue"] is False
        assert result["elapsed"] == 500
        assert result["seconds_until_due"] == 3100

    def test_overdue(self, oc):
        result = oc.check_interval(
            last_run_epoch=1000, now_epoch=5000, interval_seconds=3600
        )
        assert result["overdue"] is True
        assert result["elapsed"] == 4000

    def test_exactly_due(self, oc):
        result = oc.check_interval(
            last_run_epoch=1000, now_epoch=4600, interval_seconds=3600
        )
        assert result["overdue"] is True
        assert result["elapsed"] == 3600


class TestComputeTrend:
    def test_improving_trend(self, oc):
        result = oc.compute_trend(
            pass_counts=[5, 6, 8, 10],
            fail_counts=[5, 4, 2, 0],
        )
        assert result["trend"] == "improving"
        assert result["total_pass"] == 29
        assert result["total_fail"] == 11
        assert result["total_checks"] == 40

    def test_declining_trend(self, oc):
        result = oc.compute_trend(
            pass_counts=[10, 8, 5, 3],
            fail_counts=[0, 2, 5, 7],
        )
        assert result["trend"] == "declining"

    def test_stable_trend(self, oc):
        result = oc.compute_trend(
            pass_counts=[10, 10],
            fail_counts=[0, 0],
        )
        assert result["trend"] == "stable"

    def test_empty(self, oc):
        result = oc.compute_trend([], [])
        assert result["trend"] == "unknown"

    def test_pass_rate(self, oc):
        result = oc.compute_trend(
            pass_counts=[80],
            fail_counts=[20],
        )
        assert result["pass_rate_pct"] == 80


class TestCheckIntervalASM:
    def test_not_overdue_asm(self, oc):
        result = oc.check_interval_asm(
            last_run_epoch=1000, now_epoch=1500, interval_seconds=3600
        )
        assert result["overdue"] is False
        assert result["elapsed"] == 500

    def test_overdue_asm(self, oc):
        result = oc.check_interval_asm(
            last_run_epoch=1000, now_epoch=5000, interval_seconds=3600
        )
        assert result["overdue"] is True
