"""
Phase 38 — Real Neural Watchdog (LSTM Anomaly Detector)
========================================================
Uses nCPU's actual WatchdogNet — an LSTM that learns temporal patterns
in system metrics and detects anomalies. Monitors 8 metrics through
a sliding window, scores health with a trained neural network.

Architecture:
  Input:  [1, window_size, 8] — sliding window of system metrics
  LSTM:   hidden_size=32, 1 layer
  Scorer: Linear(32, 16) → ReLU → Linear(16, 1) → Sigmoid
  Output: anomaly score 0-1 (higher = more anomalous)

Metrics: cpu_util, mem_pressure, interrupt_rate, cache_hit_rate,
         scheduler_fairness, ipc_queue_depth, fs_ops_rate, tlb_miss_rate
"""

import sys
import torch
import random
import math
from pathlib import Path

from bridge.config import get_ncpu_path, get_bridge_path, get_clawd_data_path
NCPU_PATH = get_ncpu_path()
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from ncpu.os.neuros.watchdog import WatchdogNet, NUM_METRICS
from bridge.compute import NCPUBridge

bridge = NCPUBridge()

METRIC_NAMES = [
    "cpu_util", "mem_pressure", "interrupt_rate", "cache_hit_rate",
    "scheduler_fairness", "ipc_queue_depth", "fs_ops_rate", "tlb_miss_rate"
]


class RealNeuralWatchdog:
    """Bridge to nCPU's LSTM-based watchdog."""

    def __init__(self, window_size: int = 32, threshold: float = 0.7):
        self.window_size = window_size
        self.threshold = threshold
        self.net = WatchdogNet(input_size=NUM_METRICS, hidden_size=32)

        # Start fresh — we'll train on our own data
        # (Pretrained weights are for neurOS-specific metric distributions)
        self._pretrained = False
        self.net.eval()

        # Ring buffer
        self.buffer = torch.zeros(window_size, NUM_METRICS)
        self._ptr = 0
        self._count = 0
        self._alerts = []
        self._params = sum(p.numel() for p in self.net.parameters())

    def record(self, **metrics):
        """Record a metrics snapshot into the ring buffer."""
        values = []
        for name in METRIC_NAMES:
            values.append(float(metrics.get(name, 0.0)))
        self.buffer[self._ptr % self.window_size] = torch.tensor(values)
        self._ptr += 1
        self._count += 1

    def check(self) -> dict:
        """Run the LSTM anomaly detector on the current window."""
        # Build window tensor
        if self._count < 4:
            return {"score": 0.0, "alert": False, "reason": "insufficient data"}

        window_len = min(self._count, self.window_size)
        start = max(0, self._ptr - window_len)
        indices = [(start + i) % self.window_size for i in range(window_len)]
        window = self.buffer[indices].unsqueeze(0)  # [1, seq_len, 8]

        with torch.no_grad():
            score = self.net(window).item()

        alert = score > self.threshold
        result = {
            "score": score,
            "alert": alert,
            "window_len": window_len,
        }

        if alert:
            # Use neural CMP to find which metric is most anomalous
            latest = self.buffer[(self._ptr - 1) % self.window_size]
            mean = self.buffer[indices].mean(dim=0)
            max_dev_idx = (latest - mean).abs().argmax().item()
            result["anomalous_metric"] = METRIC_NAMES[max_dev_idx]
            result["metric_value"] = latest[max_dev_idx].item()
            result["metric_mean"] = mean[max_dev_idx].item()
            self._alerts.append(result)

        return result

    def train_baseline(self, normal_data: list[dict], epochs=100) -> dict:
        """Train the LSTM on normal operation data (anomaly = deviation from this)."""
        # Build training windows from normal data
        if len(normal_data) < self.window_size:
            return {"error": "need more data"}

        # Convert to tensor
        rows = []
        for d in normal_data:
            rows.append([d.get(n, 0.0) for n in METRIC_NAMES])
        data = torch.tensor(rows)

        # Create windows
        windows = []
        for i in range(len(data) - self.window_size):
            windows.append(data[i:i + self.window_size])
        X = torch.stack(windows)  # [N, window_size, 8]

        # Generate anomalous windows by perturbing normal data
        anomalous_windows = []
        for w in windows[:len(windows)//2]:
            aw = w.clone()
            # Spike random metrics
            aw[:, random.randint(0, 7)] += random.uniform(0.4, 0.8)
            aw[:, random.randint(0, 7)] += random.uniform(0.3, 0.6)
            anomalous_windows.append(aw)
        X_anom = torch.stack(anomalous_windows)

        # Train: normal → 0, anomalous → 1
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCELoss()
        self.net.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Normal batch
            scores_normal = self.net(X)
            loss_n = loss_fn(scores_normal, torch.zeros_like(scores_normal))
            # Anomaly batch
            scores_anom = self.net(X_anom)
            loss_a = loss_fn(scores_anom, torch.ones_like(scores_anom))
            loss = loss_n + loss_a
            loss.backward()
            optimizer.step()

        self.net.eval()
        final_normal = self.net(X).detach()
        final_anom = self.net(X_anom).detach()
        return {
            "epochs": epochs,
            "windows": len(windows),
            "mean_normal_score": final_normal.mean().item(),
            "mean_anomaly_score": final_anom.mean().item(),
            "separation": final_anom.mean().item() - final_normal.mean().item(),
        }


def demo():
    print("Real Neural Watchdog (LSTM Anomaly Detector)")
    print("=" * 60)
    print(f"Architecture: LSTM(8→32) → Linear(32→16→1) → Sigmoid")

    wd = RealNeuralWatchdog(window_size=32)
    print(f"Parameters: {wd._params:,}")
    print(f"Pretrained: {wd._pretrained}\n")

    # Generate normal operation data
    print("  Phase 1: Recording normal operation baseline...")
    normal_data = []
    for t in range(100):
        metrics = {
            "cpu_util": 0.3 + 0.1 * math.sin(t / 10.0) + random.gauss(0, 0.02),
            "mem_pressure": 0.4 + random.gauss(0, 0.03),
            "interrupt_rate": 0.2 + random.gauss(0, 0.02),
            "cache_hit_rate": 0.85 + random.gauss(0, 0.02),
            "scheduler_fairness": 0.9 + random.gauss(0, 0.01),
            "ipc_queue_depth": 0.1 + random.gauss(0, 0.02),
            "fs_ops_rate": 0.15 + random.gauss(0, 0.02),
            "tlb_miss_rate": 0.05 + random.gauss(0, 0.01),
        }
        normal_data.append(metrics)
        wd.record(**metrics)

    # Check baseline — should be low anomaly
    baseline_check = wd.check()
    print(f"    Baseline anomaly score: {baseline_check['score']:.4f}")
    print(f"    Alert: {baseline_check['alert']}")

    # Train on normal data
    print("\n  Phase 2: Training LSTM on normal patterns...")
    train_stats = wd.train_baseline(normal_data, epochs=50)
    print(f"    Windows: {train_stats.get('windows', '?')}")
    print(f"    Normal score:  {train_stats.get('mean_normal_score', 0):.4f}")
    print(f"    Anomaly score: {train_stats.get('mean_anomaly_score', 0):.4f}")
    print(f"    Separation:    {train_stats.get('separation', 0):.4f}")

    # Re-check normal — should score very low now
    wd2 = RealNeuralWatchdog(window_size=32)
    wd2.net = wd.net  # share trained net
    for d in normal_data[-32:]:
        wd2.record(**d)
    normal_check = wd2.check()
    print(f"\n    Normal check after training: score={normal_check['score']:.4f} → {'✅ low' if normal_check['score'] < 0.5 else '⚠️ high'}")

    # Inject anomaly: CPU spike + memory pressure
    print("\n  Phase 3: Injecting anomaly (CPU spike + memory pressure)...")
    for t in range(10):
        anomaly = {
            "cpu_util": 0.95 + random.gauss(0, 0.02),       # WAY above normal
            "mem_pressure": 0.9 + random.gauss(0, 0.02),     # WAY above normal
            "interrupt_rate": 0.8 + random.gauss(0, 0.05),   # spike
            "cache_hit_rate": 0.3 + random.gauss(0, 0.05),   # thrashing
            "scheduler_fairness": 0.4 + random.gauss(0, 0.05),  # degraded
            "ipc_queue_depth": 0.7 + random.gauss(0, 0.05),  # backed up
            "fs_ops_rate": 0.8 + random.gauss(0, 0.05),      # heavy I/O
            "tlb_miss_rate": 0.4 + random.gauss(0, 0.05),    # thrashing
        }
        wd2.record(**anomaly)

    anomaly_check = wd2.check()
    print(f"    Anomaly score: {anomaly_check['score']:.4f}")
    print(f"    Alert: {anomaly_check['alert']}")
    if anomaly_check.get('anomalous_metric'):
        print(f"    Most anomalous: {anomaly_check['anomalous_metric']} "
              f"({anomaly_check['metric_value']:.3f} vs mean {anomaly_check['metric_mean']:.3f})")

    # Score comparison
    print("\n  Score comparison:")
    print(f"    Normal operation: {normal_check['score']:.4f}")
    print(f"    Under anomaly:   {anomaly_check['score']:.4f}")
    
    detected = anomaly_check['score'] > normal_check['score']
    print(f"    Anomaly detected: {'✅' if detected else '❌'} (score increased by {anomaly_check['score'] - normal_check['score']:.4f})")

    print(f"\n  Parameters: {wd._params:,}")
    print(f"\n✅ Real LSTM watchdog: temporal pattern learning + anomaly detection")


if __name__ == "__main__":
    demo()
