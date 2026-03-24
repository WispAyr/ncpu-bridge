#!/usr/bin/env python3
"""
Hailo-8 Deployment — compile ONNX→HEF and run inference on Hailo hardware.

This module provides:
1. compile_to_hef()  — ONNX → HEF via Hailo Dataflow Compiler (x86 Docker or native)
2. HailoInferenceEngine — run HEF models on Hailo-8 via HailoRT
3. benchmark_hailo()  — compare Hailo-8 vs ONNX Runtime on Pi
4. CLI entrypoint      — compile / deploy / benchmark / status

Architecture:
  Dev machine (x86): ONNX → [DFC in Docker] → HEF files
  Pi (ARM + Hailo-8): HEF files → [HailoRT] → inference

Usage:
    # On x86 dev machine — compile all ONNX models to HEF
    python -m bridge.hailo_deploy compile --docker

    # On Pi — check Hailo-8 status
    python -m bridge.hailo_deploy status

    # On Pi — run inference benchmark
    python -m bridge.hailo_deploy benchmark

    # On Pi — deploy (copy HEFs and test)
    python -m bridge.hailo_deploy deploy --pi 192.168.195.238
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BRIDGE_DIR = Path(__file__).parent
PROJECT_DIR = BRIDGE_DIR.parent
ONNX_DIR = PROJECT_DIR / "exported_models" / "onnx"
HEF_DIR = PROJECT_DIR / "exported_models" / "hef"

# Model metadata: name → (input_shape, output_shape)
MODEL_SPECS = {
    "arithmetic":    ((3,),  (2,)),
    "multiply":      ((16,), (8,)),
    "divide":        ((16,), (8,)),
    "compare":       ((4,),  (3,)),
    "logical":       ((2,),  (1,)),
    "carry_combine": ((4,),  (2,)),
    "sincos":        ((1,),  (2,)),
    "sqrt":          ((1,),  (1,)),
    "exp":           ((1,),  (1,)),
    "log":           ((1,),  (1,)),
    "atan2":         ((2,),  (1,)),
    "lsl":           ((16,), (8,)),
    "lsr":           ((16,), (8,)),
    "asr":           ((16,), (8,)),
    "rol":           ((16,), (8,)),
}

# ---------------------------------------------------------------------------
# 1. Compilation: ONNX → HEF
# ---------------------------------------------------------------------------

def compile_to_hef(
    onnx_dir: Path = ONNX_DIR,
    hef_dir: Path = HEF_DIR,
    use_docker: bool = True,
    hw_arch: str = "hailo8",
) -> dict[str, Any]:
    """Compile all ONNX models to HEF format.

    If use_docker=True, uses the Hailo Model Zoo Docker image (x86 required).
    Otherwise assumes hailo CLI is installed natively.

    Returns dict of model_name → {status, hef_path, error}.
    """
    hef_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, (in_shape, out_shape) in MODEL_SPECS.items():
        onnx_path = onnx_dir / f"{name}.onnx"
        hef_path = hef_dir / f"{name}.hef"

        if not onnx_path.exists():
            results[name] = {"status": "skip", "error": f"ONNX not found: {onnx_path}"}
            continue

        try:
            if use_docker:
                _compile_docker(name, onnx_path, hef_path, hw_arch, in_shape)
            else:
                _compile_native(name, onnx_path, hef_path, hw_arch, in_shape)

            if hef_path.exists():
                results[name] = {
                    "status": "ok",
                    "hef_path": str(hef_path),
                    "size_bytes": hef_path.stat().st_size,
                }
            else:
                results[name] = {"status": "error", "error": "HEF not produced"}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}

    return results


def _compile_docker(
    name: str, onnx_path: Path, hef_path: Path, hw_arch: str, in_shape: tuple
):
    """Compile via Hailo DFC Docker container."""
    # Mount the parent dirs so Docker can read ONNX and write HEF
    onnx_mount = onnx_path.parent.resolve()
    hef_mount = hef_path.parent.resolve()

    # Generate a minimal Hailo compilation script
    batch_size = 1
    input_dim = ",".join(str(d) for d in (batch_size, *in_shape))

    compile_script = f"""#!/bin/bash
set -e
cd /workspace

# Parse ONNX to Hailo Archive (.har)
hailo parser onnx /onnx/{name}.onnx \\
    --net-name {name} \\
    --start-node-names input \\
    --end-node-names output

# Optimize (quantize to int8 — using random calibration for these small MLPs)
hailo optimize {name}.har \\
    --use-random-calib-set \\
    --calib-set-size 64 \\
    --hw-arch {hw_arch}

# Compile to HEF
hailo compiler {name}.har \\
    --hw-arch {hw_arch} \\
    -o /hef/{name}.hef

echo "OK: {name}.hef"
"""
    script_path = onnx_mount / f"_compile_{name}.sh"
    script_path.write_text(compile_script)
    script_path.chmod(0o755)

    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{onnx_mount}:/onnx:ro",
                "-v", f"{hef_mount}:/hef",
                "-v", f"{onnx_mount}/_compile_{name}.sh:/workspace/compile.sh:ro",
                "hailo-ai/hailo_model_zoo:2.13.0",
                "bash", "/workspace/compile.sh",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Docker compilation failed:\n{result.stderr[-500:]}"
            )
    finally:
        script_path.unlink(missing_ok=True)


def _compile_native(
    name: str, onnx_path: Path, hef_path: Path, hw_arch: str, in_shape: tuple
):
    """Compile via locally installed Hailo DFC CLI."""
    work_dir = hef_path.parent / "_work"
    work_dir.mkdir(exist_ok=True)

    # Step 1: Parse
    subprocess.run(
        ["hailo", "parser", "onnx", str(onnx_path), "--net-name", name],
        cwd=work_dir, check=True, capture_output=True, text=True, timeout=120,
    )
    har_path = work_dir / f"{name}.har"

    # Step 2: Optimize (int8 quantization with random calibration)
    subprocess.run(
        [
            "hailo", "optimize", str(har_path),
            "--use-random-calib-set", "--calib-set-size", "64",
            "--hw-arch", hw_arch,
        ],
        cwd=work_dir, check=True, capture_output=True, text=True, timeout=120,
    )

    # Step 3: Compile
    subprocess.run(
        [
            "hailo", "compiler", str(har_path),
            "--hw-arch", hw_arch, "-o", str(hef_path),
        ],
        cwd=work_dir, check=True, capture_output=True, text=True, timeout=120,
    )


# ---------------------------------------------------------------------------
# 2. Hailo Inference Engine
# ---------------------------------------------------------------------------

@dataclass
class HailoInferenceEngine:
    """Run HEF models on Hailo-8 hardware via HailoRT.

    Usage:
        engine = HailoInferenceEngine(hef_dir="/path/to/hefs")
        engine.load("arithmetic")
        result = engine.infer("arithmetic", np.array([1.0, 0.0, 1.0], dtype=np.float32))
        engine.close()
    """

    hef_dir: str | Path = HEF_DIR
    _device: Any = field(default=None, repr=False)
    _loaded: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.hef_dir = Path(self.hef_dir)

    def _ensure_device(self):
        """Lazy-init Hailo virtual device."""
        if self._device is not None:
            return
        try:
            from hailo_platform import VDevice, HailoSchedulingAlgorithm
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            self._device = VDevice(params)
        except ImportError:
            raise RuntimeError(
                "hailo_platform not installed. Install HailoRT: "
                "pip install hailort (on Pi with Hailo-8)"
            )

    def load(self, model_name: str) -> None:
        """Load a HEF model onto the device."""
        if model_name in self._loaded:
            return
        self._ensure_device()

        from hailo_platform import HEF

        hef_path = self.hef_dir / f"{model_name}.hef"
        if not hef_path.exists():
            raise FileNotFoundError(f"HEF not found: {hef_path}")

        hef = HEF(str(hef_path))
        network_group = self._device.configure(hef)[0]
        network_group_params = network_group.create_params()

        # Get input/output vstream info
        input_vstreams_info = hef.get_input_vstream_infos()
        output_vstreams_info = hef.get_output_vstream_infos()

        self._loaded[model_name] = {
            "hef": hef,
            "network_group": network_group,
            "params": network_group_params,
            "input_info": input_vstreams_info,
            "output_info": output_vstreams_info,
        }

    def infer(self, model_name: str, input_data: np.ndarray) -> np.ndarray:
        """Run inference on a loaded model.

        Args:
            model_name: Name of the model (must be loaded first)
            input_data: Input array, shape (input_dim,) or (batch, input_dim)

        Returns:
            Output numpy array
        """
        if model_name not in self._loaded:
            self.load(model_name)

        from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams

        ctx = self._loaded[model_name]
        ng = ctx["network_group"]

        # Ensure batch dimension
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        input_data = input_data.astype(np.float32)

        # Create vstream params
        input_params = InputVStreamParams.make(ng)
        output_params = OutputVStreamParams.make(ng)

        # Build input dict
        input_name = ctx["input_info"][0].name
        input_dict = {input_name: input_data}

        # Run inference
        with InferVStreams(ng, input_params, output_params) as pipeline:
            results = pipeline.infer(input_dict)

        # Extract output
        output_name = ctx["output_info"][0].name
        return results[output_name]

    def infer_batch(
        self, model_name: str, inputs: np.ndarray
    ) -> np.ndarray:
        """Batch inference — inputs shape (batch_size, input_dim)."""
        return self.infer(model_name, inputs)

    def close(self):
        """Release device resources."""
        self._loaded.clear()
        if self._device is not None:
            try:
                self._device.release()
            except Exception:
                pass
            self._device = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# 3. Benchmark: Hailo-8 vs ONNX Runtime
# ---------------------------------------------------------------------------

def benchmark_hailo(
    hef_dir: Path = HEF_DIR,
    onnx_dir: Path = ONNX_DIR,
    iterations: int = 1000,
    warmup: int = 100,
) -> dict[str, dict]:
    """Benchmark Hailo-8 vs ONNX Runtime inference.

    Returns dict of model_name → {hailo_us, onnx_us, speedup, hailo_ops, onnx_ops}.
    """
    import onnxruntime as ort

    results = {}

    # Try Hailo
    hailo_available = False
    engine = None
    try:
        engine = HailoInferenceEngine(hef_dir=hef_dir)
        engine._ensure_device()
        hailo_available = True
    except Exception as e:
        print(f"⚠️  Hailo-8 not available: {e}")
        print("   Running ONNX Runtime only.\n")

    for name, (in_shape, out_shape) in MODEL_SPECS.items():
        result = {"model": name}

        # Generate random input
        test_input = np.random.randn(1, *in_shape).astype(np.float32)

        # ONNX Runtime benchmark
        onnx_path = onnx_dir / f"{name}.onnx"
        if onnx_path.exists():
            try:
                sess = ort.InferenceSession(str(onnx_path))
                input_name = sess.get_inputs()[0].name

                # Warmup
                for _ in range(warmup):
                    sess.run(None, {input_name: test_input})

                # Timed
                t0 = time.perf_counter()
                for _ in range(iterations):
                    sess.run(None, {input_name: test_input})
                elapsed = time.perf_counter() - t0

                onnx_us = (elapsed / iterations) * 1e6
                result["onnx_us"] = round(onnx_us, 1)
                result["onnx_ops"] = round(iterations / elapsed)
            except Exception as e:
                result["onnx_us"] = None
                result["onnx_error"] = str(e)
        else:
            result["onnx_us"] = None

        # Hailo benchmark
        if hailo_available:
            hef_path = hef_dir / f"{name}.hef"
            if hef_path.exists():
                try:
                    engine.load(name)

                    # Warmup
                    for _ in range(warmup):
                        engine.infer(name, test_input)

                    # Timed
                    t0 = time.perf_counter()
                    for _ in range(iterations):
                        engine.infer(name, test_input)
                    elapsed = time.perf_counter() - t0

                    hailo_us = (elapsed / iterations) * 1e6
                    result["hailo_us"] = round(hailo_us, 1)
                    result["hailo_ops"] = round(iterations / elapsed)
                except Exception as e:
                    result["hailo_us"] = None
                    result["hailo_error"] = str(e)
            else:
                result["hailo_us"] = None
                result["hailo_note"] = "HEF not compiled"

        # Speedup
        if result.get("hailo_us") and result.get("onnx_us"):
            result["speedup"] = round(result["onnx_us"] / result["hailo_us"], 1)

        results[name] = result

    if engine:
        engine.close()

    return results


# ---------------------------------------------------------------------------
# 4. Deployment helpers
# ---------------------------------------------------------------------------

def deploy_to_pi(
    pi_host: str = "192.168.195.238",
    pi_user: str = "pi",
    remote_dir: str = "/home/pi/ncpu-hailo",
):
    """SCP HEF models and this script to the Pi."""
    hef_files = list(HEF_DIR.glob("*.hef"))
    if not hef_files:
        print("❌ No HEF files found. Run 'compile' first.")
        return False

    remote_hef_dir = f"{remote_dir}/hef_models"

    print(f"Deploying {len(hef_files)} HEF models to {pi_user}@{pi_host}:{remote_hef_dir}")

    # Create remote dir
    subprocess.run(
        ["ssh", f"{pi_user}@{pi_host}", f"mkdir -p {remote_hef_dir}"],
        check=True,
    )

    # Copy HEFs
    for hef in hef_files:
        print(f"  → {hef.name}")
        subprocess.run(
            ["scp", str(hef), f"{pi_user}@{pi_host}:{remote_hef_dir}/"],
            check=True,
        )

    # Copy this script
    subprocess.run(
        ["scp", __file__, f"{pi_user}@{pi_host}:{remote_dir}/hailo_deploy.py"],
        check=True,
    )

    print(f"\n✅ Deployed. Run on Pi:")
    print(f"   ssh {pi_user}@{pi_host}")
    print(f"   cd {remote_dir}")
    print(f"   python3 hailo_deploy.py benchmark")
    return True


def check_status() -> dict:
    """Check Hailo-8 hardware and software status (run on Pi)."""
    status = {
        "hailo_device": False,
        "hailort_python": False,
        "hailort_version": None,
        "firmware_version": None,
        "hef_models": [],
        "onnx_models": [],
    }

    # Check device
    if Path("/dev/hailo0").exists():
        status["hailo_device"] = True

    # Check HailoRT Python
    try:
        import hailo_platform
        status["hailort_python"] = True
        status["hailort_version"] = getattr(hailo_platform, "__version__", "unknown")
    except ImportError:
        pass

    # Check firmware via CLI
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            status["firmware_version"] = result.stdout.strip()
    except Exception:
        pass

    # Check available models
    for hef in HEF_DIR.glob("*.hef"):
        status["hef_models"].append(hef.stem)
    for onnx in ONNX_DIR.glob("*.onnx"):
        status["onnx_models"].append(onnx.stem)

    return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "status":
        s = check_status()
        print("Hailo-8 Deployment Status")
        print("=" * 50)
        print(f"  Hailo device:    {'✅ /dev/hailo0' if s['hailo_device'] else '❌ not found'}")
        print(f"  HailoRT Python:  {'✅ ' + (s['hailort_version'] or '') if s['hailort_python'] else '❌ not installed'}")
        print(f"  Firmware:        {s['firmware_version'] or 'N/A'}")
        print(f"  HEF models:      {len(s['hef_models'])} ({', '.join(s['hef_models'][:5])}{'...' if len(s['hef_models']) > 5 else ''})")
        print(f"  ONNX models:     {len(s['onnx_models'])}")

        if not s["hef_models"]:
            print("\n  ⚠️  No HEF models. Compile with: python -m bridge.hailo_deploy compile")

    elif cmd == "compile":
        use_docker = "--docker" in sys.argv
        print(f"Compiling ONNX → HEF ({'Docker' if use_docker else 'native'})...")
        print("=" * 50)
        results = compile_to_hef(use_docker=use_docker)
        ok = sum(1 for r in results.values() if r["status"] == "ok")
        fail = sum(1 for r in results.values() if r["status"] == "error")
        skip = sum(1 for r in results.values() if r["status"] == "skip")
        for name, r in results.items():
            if r["status"] == "ok":
                print(f"  ✅ {name:20s} → {r['hef_path']} ({r['size_bytes']:,} bytes)")
            elif r["status"] == "skip":
                print(f"  ⏭️  {name:20s} — {r['error']}")
            else:
                print(f"  ❌ {name:20s} — {r['error']}")
        print(f"\n  Compiled: {ok}, Failed: {fail}, Skipped: {skip}")

    elif cmd == "benchmark":
        iters = 1000
        for i, arg in enumerate(sys.argv):
            if arg == "--iterations" and i + 1 < len(sys.argv):
                iters = int(sys.argv[i + 1])

        print(f"Benchmarking (iterations={iters})...")
        print("=" * 60)
        results = benchmark_hailo(iterations=iters)

        print(f"\n{'Model':20s} {'ONNX (µs)':>12s} {'Hailo (µs)':>12s} {'Speedup':>10s} {'Hailo ops/s':>14s}")
        print("-" * 70)
        total_onnx_ops = 0
        total_hailo_ops = 0
        for name, r in sorted(results.items(), key=lambda x: x[1].get("onnx_us", 9e9)):
            onnx_s = f"{r['onnx_us']}" if r.get("onnx_us") else "N/A"
            hailo_s = f"{r['hailo_us']}" if r.get("hailo_us") else r.get("hailo_note", "N/A")
            spd = f"{r['speedup']}x" if r.get("speedup") else "-"
            hops = f"{r.get('hailo_ops', 0):,}" if r.get("hailo_ops") else "-"
            print(f"  {name:20s} {onnx_s:>12s} {hailo_s:>12s} {spd:>10s} {hops:>14s}")
            total_onnx_ops += r.get("onnx_ops", 0)
            total_hailo_ops += r.get("hailo_ops", 0)

        print(f"\n  Total throughput: ONNX={total_onnx_ops:,} ops/s" +
              (f", Hailo={total_hailo_ops:,} ops/s" if total_hailo_ops else ""))

    elif cmd == "deploy":
        pi_host = "192.168.195.238"
        pi_user = "pi"
        for i, arg in enumerate(sys.argv):
            if arg == "--pi" and i + 1 < len(sys.argv):
                pi_host = sys.argv[i + 1]
            if arg == "--user" and i + 1 < len(sys.argv):
                pi_user = sys.argv[i + 1]
        deploy_to_pi(pi_host=pi_host, pi_user=pi_user)

    else:
        print("Usage: python -m bridge.hailo_deploy <command>")
        print()
        print("Commands:")
        print("  status                  Check Hailo-8 hardware/software status")
        print("  compile [--docker]      Compile ONNX models to HEF format")
        print("  benchmark [--iterations N]  Benchmark Hailo vs ONNX Runtime")
        print("  deploy [--pi HOST]      Deploy HEF models to Pi")


if __name__ == "__main__":
    main()
