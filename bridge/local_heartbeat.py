"""Local heartbeat — nCPU neural verification + local LLM interpretation.

Runs the full check pipeline locally without calling Claude:
1. nCPU neural ALU verifies all health/obligation computations
2. Local LLM (Qwen via Ollama) interprets results and decides action
3. Only escalates to Claude/Telegram if something needs human attention

Usage:
    python -m bridge.local_heartbeat          # Run and print result
    python -m bridge.local_heartbeat --quiet  # Only output if alert needed
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
BRIDGE_PATH = Path("/Users/noc/projects/ncpu-bridge")

if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))
if str(BRIDGE_PATH) not in sys.path:
    sys.path.insert(0, str(BRIDGE_PATH))

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:1.7b"

SYSTEM_PROMPT = """You are Skynet's local heartbeat monitor. You receive JSON health data where every computation was verified by a neural ALU (provably correct arithmetic).

Rules:
- If everything is OK (all obligations fresh, health checks passing): respond with exactly "HEARTBEAT_OK"
- If there are issues: respond with a brief 1-2 line alert describing what needs attention
- Never explain your reasoning. Just the verdict.
- Memory below 200MB is a warning, not critical.
- Stale obligations that last passed are lower priority than failing ones."""


def run_neural_checks() -> dict:
    """Run nCPU-verified checks and return structured results."""
    from bridge.skynet_integration import run_checks as _run
    
    # Capture stdout from run_checks
    import io
    old_stdout = sys.stdout
    sys.stdout = capture = io.StringIO()
    
    try:
        _run()
    except SystemExit:
        pass
    finally:
        sys.stdout = old_stdout
    
    output = capture.getvalue().strip()
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"error": "Failed to parse neural check output", "raw": output[:500]}


def ask_local_llm(check_results: dict) -> dict:
    """Ask local LLM to interpret the neural-verified results."""
    prompt = f"""/no_think
{SYSTEM_PROMPT}

Neural-verified check results:
{json.dumps(check_results, indent=2)}"""

    try:
        result = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", json.dumps({
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 300},
            })],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        response = data.get("response", "").strip()
        
        # Strip thinking tags if present
        if "<think>" in response:
            parts = response.split("</think>")
            response = parts[-1].strip() if len(parts) > 1 else response
        
        tokens = data.get("eval_count", 0)
        duration_ns = data.get("eval_duration", 1)
        tok_s = tokens / (duration_ns / 1e9) if duration_ns > 0 else 0
        
        return {
            "verdict": response,
            "is_ok": "HEARTBEAT_OK" in response.upper(),
            "tokens": tokens,
            "tok_s": round(tok_s),
            "latency_s": round(data.get("total_duration", 0) / 1e9, 1),
            "model": MODEL,
        }
    except Exception as e:
        return {
            "verdict": f"LLM unavailable: {e}",
            "is_ok": False,
            "error": str(e),
            "model": MODEL,
        }


def run_local_heartbeat(quiet: bool = False) -> dict:
    """Full local heartbeat: neural checks + local LLM interpretation."""
    t0 = time.perf_counter()
    
    # Step 1: Neural-verified checks
    check_results = run_neural_checks()
    neural_time = time.perf_counter() - t0
    
    # Step 2: Local LLM interpretation
    t1 = time.perf_counter()
    llm_result = ask_local_llm(check_results)
    llm_time = time.perf_counter() - t1
    
    total_time = time.perf_counter() - t0
    
    result = {
        "verdict": llm_result["verdict"],
        "is_ok": llm_result["is_ok"],
        "neural_checks": check_results.get("feedback", {}),
        "llm": {
            "model": llm_result.get("model"),
            "tokens": llm_result.get("tokens", 0),
            "tok_s": llm_result.get("tok_s", 0),
        },
        "timing": {
            "neural_ms": round(neural_time * 1000),
            "llm_ms": round(llm_time * 1000),
            "total_ms": round(total_time * 1000),
        },
        "cloud_calls": 0,
        "neural_verified": True,
    }
    
    if not quiet or not result["is_ok"]:
        print(json.dumps(result, indent=2))
    
    return result


def main():
    quiet = "--quiet" in sys.argv
    result = run_local_heartbeat(quiet=quiet)
    
    if not result["is_ok"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
