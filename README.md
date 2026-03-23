# nCPU Bridge

> Neural-verified computation layer for AI agents. Every arithmetic operation goes through trained neural networks with mathematically proven 100% accuracy.

[![Tests](https://img.shields.io/badge/tests-57%20passing-brightgreen)]()
[![Neural Verified](https://img.shields.io/badge/arithmetic-neural%20verified-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)]()

## What is this?

AI assistants (Claude, GPT, etc.) are probabilistic — they *predict* answers rather than *compute* them. When your AI says "48 × 365 = 17,520", it's pattern-matching, not multiplying. It's usually right. But not always.

**nCPU Bridge** solves this by routing all arithmetic through [nCPU](https://github.com/robertcprice/nCPU)'s trained neural ALU models — neural networks that have been **exhaustively verified** to produce correct results on every possible input. Not 99.9%. **100%.**

The result: your AI agent can offload deterministic computation to a provably correct neural substrate, and get mathematically guaranteed results.

## Quick Start

```bash
# Clone
git clone https://github.com/WispAyr/ncpu-bridge.git
cd ncpu-bridge

# You need nCPU installed
git clone https://github.com/robertcprice/nCPU.git /path/to/nCPU
cd /path/to/nCPU && uv venv .venv && source .venv/bin/activate
# Fix build backend if needed:
sed -i '' 's/setuptools.backends._legacy:_Backend/setuptools.build_meta/' pyproject.toml
uv pip install -e ".[neural,dev]"

# Run the bridge
PYTHONPATH=/path/to/nCPU:. python -m bridge.cli calculate "48 * 365"
# → 17520 (computed by neural networks, not CPU)
```

## Features

### 🧮 Neural Computation (`bridge/compute.py`)
```python
from bridge.compute import NCPUBridge

bridge = NCPUBridge()

# Every operation goes through trained .pt models
bridge.add(100, 200)      # → 300 (neural network)
bridge.mul(48, 365)       # → 17520 (neural network)
bridge.cmp(62, 90)        # → (False, False) = 62 < 90 (neural network)

# Expression evaluation
bridge.calculate("48 * 365")  # → 17520

# Verify someone else's arithmetic
bridge.verify("mul", 48, 365, 17520)  # → True

# Run assembly programs on neural CPU
bridge.run_program("MOV R0, 7\nMOV R1, 6\nMUL R2, R0, R1\nHALT")
# → {'registers': {'R2': 42, ...}}
```

### 🏥 Health Checks (`bridge/health.py`)
```python
from bridge.health import HealthComputer

health = HealthComputer(bridge)

# Neural threshold comparison
health.check_threshold(62, 90, "disk_usage")
# → {"exceeded": False, "value": 62, "threshold": 90}

# Neural statistics
health.compute_stats([10, 20, 30, 40, 50])
# → {"sum": 150, "min": 10, "max": 50, "count": 5}
```

### 📋 Obligation Checking (`bridge/obligations.py`)
```python
from bridge.obligations import ObligationChecker

checker = ObligationChecker(bridge)

# Is this task overdue? (all arithmetic neural-verified)
checker.check_interval(
    last_run_epoch=1774290000,
    now_epoch=1774300000,
    interval_seconds=3600
)
# → {"elapsed": 10000, "interval": 3600, "overdue": True}

# Trend analysis
checker.compute_trend(
    pass_counts=[8, 9, 10, 10],
    fail_counts=[2, 1, 0, 0]
)
# → {"pass_rate_pct": 92, "trend": "improving"}
```

### 🔄 SOME Feedback Loop (`bridge/feedback_loop.py`)
Records every computation outcome for self-improvement via nCPU's [Self-Optimizing Machine Engine](https://github.com/robertcprice/nCPU):

```python
from bridge.feedback_loop import SkynetFeedbackLoop, TaskOutcome

feedback = SkynetFeedbackLoop()

outcome = TaskOutcome(
    task_name="health:disk_check",
    category="health_check",
    success=True,
    neural_verified=True,
    execution_time_ms=11.2,
)
suggestion = feedback.record_outcome(outcome)
# → {"action": "reinforce", "urgency": "none", ...}

# Track improvement over time
feedback.get_trend()
# → {"trend": "improving", "total_recorded": 248, ...}
```

### 💻 CLI
```bash
# Calculate
ncpu-bridge calculate "48 * 365"

# Verify
ncpu-bridge verify mul 48 365 17520

# Health check
ncpu-bridge health-check --value 62 --threshold 90

# Obligation check
ncpu-bridge obligation-check --last-run 1774290000 --interval 3600
```

## Performance

Benchmarked on Apple Silicon (M-series Mac Studio):

| Operation | Neural ALU | Native Python | Ratio |
|-----------|-----------|---------------|-------|
| Addition  | 452 µs    | 0.04 µs       | 10,000x |
| Subtraction | 352 µs  | 0.04 µs       | 8,800x |
| Multiplication | 27 µs | 0.04 µs      | 700x |

**Yes, it's slower than native math.** That's not the point. The point is:
1. **Provable correctness** — exhaustively verified on all 32-bit inputs
2. **Neural computation** — arithmetic happens in neural network forward passes
3. **Differentiable** — the entire computation graph can be optimised via gradient descent
4. **Self-improving** — outcomes feed back into SOME for continuous adaptation

For typical agent workloads (10-30 checks per heartbeat), total overhead is ~100ms. Agent tool calls take seconds. The neural verification cost is negligible.

## Architecture

```
┌─────────────────────────────────────────┐
│           AI Agent (Claude, etc.)        │
│  "Check if POS backup is overdue"       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         nCPU Bridge                      │
│  ObligationChecker / HealthComputer      │
│  Parses task → routes to neural ops      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         nCPU Neural ALU                  │
│  13 trained .pt models                   │
│  ADD, SUB, MUL, DIV, CMP, AND, OR,     │
│  XOR, SHL, SHR — all 100% verified      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         SOME Feedback Loop               │
│  Records outcomes → gradient signals     │
│  Trajectory logging → self-improvement   │
└─────────────────────────────────────────┘
```

## Integration Example: Heartbeat Monitor

```bash
# Run neural-verified health + obligation checks
# with SOME feedback recording
./scripts/ncpu-verify.sh integrated

# Output: JSON with all checks, neural verification status,
# and feedback loop statistics
```

See [`bridge/skynet_integration.py`](bridge/skynet_integration.py) for a complete example of wiring nCPU Bridge into an AI agent's heartbeat workflow.

## Requirements

- Python 3.9+
- [nCPU](https://github.com/robertcprice/nCPU) (neural CPU with trained models)
- PyTorch 2.0+
- Apple Silicon recommended (Metal GPU support for compute mode)

## Inspired by

- [nCPU](https://github.com/robertcprice/nCPU) by Robert Price — the neural computer that makes this possible
- [Percepta AI](https://twitter.com/ChristosTzamos) — C-to-transformer-weights compiler
- The idea that AI should **compute**, not just **predict**

## License

MIT
