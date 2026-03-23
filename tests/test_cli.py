"""Tests for CLI interface."""

import json
import subprocess
import sys

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "bridge.cli", *args],
        capture_output=True,
        text=True,
        cwd="/Users/noc/projects/ncpu-bridge",
    )


class TestCLICalculate:
    def test_calculate(self):
        r = run_cli("calculate", "48 * 365")
        assert r.returncode == 0
        assert r.stdout.strip() == "17520"

    def test_calculate_add(self):
        r = run_cli("calculate", "100 + 200")
        assert r.returncode == 0
        assert r.stdout.strip() == "300"


class TestCLIVerify:
    def test_verify_correct(self):
        r = run_cli("verify", "add", "100", "200", "300")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["verified"] is True

    def test_verify_incorrect(self):
        r = run_cli("verify", "add", "100", "200", "999")
        assert r.returncode == 1
        data = json.loads(r.stdout)
        assert data["verified"] is False


class TestCLIHealthCheck:
    def test_health_ok(self):
        r = run_cli("health-check", "--value", "62", "--threshold", "90")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["exceeded"] is False

    def test_health_exceeded(self):
        r = run_cli("health-check", "--value", "95", "--threshold", "90")
        assert r.returncode == 1


class TestCLIObligation:
    def test_not_overdue(self):
        r = run_cli(
            "obligation-check",
            "--last-run", "1000",
            "--interval", "3600",
            "--now", "1500",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["overdue"] is False

    def test_overdue(self):
        r = run_cli(
            "obligation-check",
            "--last-run", "1000",
            "--interval", "3600",
            "--now", "5000",
        )
        assert r.returncode == 1
        data = json.loads(r.stdout)
        assert data["overdue"] is True
