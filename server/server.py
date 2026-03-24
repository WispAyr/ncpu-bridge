"""nCPU Bridge FastAPI RPC Server — thin wrapper over neural ALU ops."""

import os
import sys
import time
import random
import statistics
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# Configurable paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridge.config import get_ncpu_path, get_bridge_path
NCPU_PATH = get_ncpu_path()
BRIDGE_PATH = get_bridge_path()

# Ensure imports work
for p in [str(NCPU_PATH), str(BRIDGE_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from bridge.compute import NCPUBridge
from bridge.obligations import ObligationChecker

# ── Startup: load models once ──
_start_time = time.time()
_bridge = NCPUBridge(ncpu_path=str(NCPU_PATH))
_obligations = ObligationChecker(bridge=_bridge)
_op_count = 0
_error_count = 0

app = FastAPI(title="nCPU Bridge RPC", version="1.0.0")

# ── Models ──

class ComputeRequest(BaseModel):
    op: str  # add|sub|mul|div|cmp|and|or|xor|shl|shr
    a: int
    b: int

class BatchRequest(BaseModel):
    ops: List[ComputeRequest]

class ObligationCheckRequest(BaseModel):
    last_run_epoch: int
    now_epoch: int
    interval_seconds: int

class ObligationTrendRequest(BaseModel):
    pass_counts: List[int]
    fail_counts: List[int]

# ── Op dispatch ──

_OP_MAP = {
    "add": _bridge.add,
    "sub": _bridge.sub,
    "mul": _bridge.mul,
    "div": _bridge.div,
    "and": _bridge.bitwise_and,
    "or": _bridge.bitwise_or,
    "xor": _bridge.bitwise_xor,
    "shl": _bridge.shl,
    "shr": _bridge.shr,
}

def _execute_op(req: ComputeRequest) -> dict:
    global _op_count, _error_count
    t0 = time.perf_counter_ns()
    
    if req.op == "cmp":
        zf, sf = _bridge.cmp(req.a, req.b)
        result = {"zero_flag": zf, "sign_flag": sf}
    elif req.op in _OP_MAP:
        result = _OP_MAP[req.op](req.a, req.b)
    else:
        _error_count += 1
        return {"error": f"Unknown op: {req.op}", "valid_ops": list(_OP_MAP.keys()) + ["cmp"]}
    
    elapsed_us = (time.perf_counter_ns() - t0) / 1000
    _op_count += 1
    return {"result": result, "latency_us": round(elapsed_us, 2)}

# ── Endpoints ──

@app.post("/compute")
def compute(req: ComputeRequest):
    return _execute_op(req)

@app.post("/compute/batch")
def compute_batch(req: BatchRequest):
    t0 = time.perf_counter_ns()
    results = [_execute_op(op) for op in req.ops]
    total_us = (time.perf_counter_ns() - t0) / 1000
    return {"results": results, "total_latency_us": round(total_us, 2)}

@app.post("/obligation/check")
def obligation_check(req: ObligationCheckRequest):
    t0 = time.perf_counter_ns()
    result = _obligations.check_interval(req.last_run_epoch, req.now_epoch, req.interval_seconds)
    elapsed_us = (time.perf_counter_ns() - t0) / 1000
    return {**result, "latency_us": round(elapsed_us, 2)}

@app.post("/obligation/trend")
def obligation_trend(req: ObligationTrendRequest):
    t0 = time.perf_counter_ns()
    result = _obligations.compute_trend(req.pass_counts, req.fail_counts)
    elapsed_us = (time.perf_counter_ns() - t0) / 1000
    return {**result, "latency_us": round(elapsed_us, 2)}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": _bridge._available,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "op_count": _op_count,
        "error_count": _error_count,
    }

@app.get("/benchmark")
def benchmark():
    ops = ["add", "sub", "mul", "div"]
    latencies = []
    for _ in range(100):
        op = random.choice(ops)
        a, b = random.randint(1, 100), random.randint(1, 100)
        t0 = time.perf_counter_ns()
        _OP_MAP[op](a, b)
        latencies.append((time.perf_counter_ns() - t0) / 1000)
    
    latencies.sort()
    return {
        "ops": 100,
        "min_us": round(latencies[0], 2),
        "avg_us": round(statistics.mean(latencies), 2),
        "max_us": round(latencies[-1], 2),
        "p99_us": round(latencies[98], 2),
        "median_us": round(statistics.median(latencies), 2),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3950)
