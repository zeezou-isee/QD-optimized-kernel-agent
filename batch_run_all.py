"""Batch-run the OperatorAgent over every operator in Mobilekernelbench.

- Discovers all <Op>.py models under dataset/Mobilekernelbench_subset/.
- Runs the full pipeline (kernel + graph + e2e + production) per op.
- Records a compact result row per op into batch_all_results.json.
- Resumable: ops already present (with a terminal status) in the results file
  are skipped on re-run.

LLM backend configuration:
- Set LLM_BACKEND=deepseek to use DeepSeek API directly (needs DEEPSEEK_API_KEY).
- Default (unset) uses OpenRouter (needs OPENROUTER_API_KEY).
- DEEPSEEK_MODEL overrides the model name for direct DeepSeek calls.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "Mobilekernelbench_subset"
CLI = ROOT / "opgen" / "cli" / "run_operator_agent.py"
RESULTS = ROOT / "batch_all_results_arm.json"
RUNS = ROOT / "opgen" / "runs"

# --- LLM backend ---
# Set LLM_BACKEND=deepseek (in env or below) to use direct DeepSeek API.
# Default: OpenRouter via OPENROUTER_API_KEY.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "openrouter")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
# Model name: bare for deepseek (e.g. deepseek-v4-pro); deepseek/ prefix for OpenRouter
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro") if LLM_BACKEND == "deepseek" \
        else "deepseek/deepseek-v4-pro"
# ---
MAX_ROUNDS = "15"
GRAPH_MAX_ROUNDS = "10"
PER_OP_TIMEOUT = 1800  # 30 min hard cap per op


def discover_ops() -> list[tuple[str, str]]:
    """Return sorted (category, op_name) for every <Op>.py model."""
    ops = []
    for py in sorted(DATASET.rglob("*.py")):
        if py.stem == "__init__":
            continue
        ops.append((py.parent.name, py.stem))
    return ops


def load_results() -> dict:
    if RESULTS.exists():
        try:
            return json.loads(RESULTS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_results(data: dict) -> None:
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(op: str) -> dict:
    """Read the operator summary.json the agent wrote, extract phase outcomes."""
    sj = RUNS / op / "operator" / "summary.json"
    if not sj.exists():
        return {}
    try:
        s = json.loads(sj.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    ph = s.get("phases", {})
    return {
        "status": s.get("status"),
        "kernel": (ph.get("kernel") or {}).get("status"),
        "kernel_arm": (ph.get("kernel_arm") or {}).get("status"),
        "graph": (ph.get("graph") or {}).get("status"),
        "already_in_ncnn": (ph.get("existence_check") or {}).get("already_in_ncnn"),
        "e2e": (ph.get("end_to_end_numeric") or {}).get("passed"),
        "production": (ph.get("production") or {}).get("_mandatory_ok"),
        "note": s.get("note"),
    }


def run_one(category: str, op: str) -> dict:
    cmd = [
        sys.executable, str(CLI),
        "--task", op,
        "--dataset-root", str(DATASET),
        "--model-name", MODEL,
        "--max-rounds", MAX_ROUNDS,
        "--graph-max-rounds", GRAPH_MAX_ROUNDS,
        "--backends", "base,arm",
        "--compile-mode", "build_lib",
    ]
    # Inherit parent env + inject deepseek config for the child process
    env = os.environ.copy()
    env["LLM_BACKEND"] = LLM_BACKEND
    env["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    if os.environ.get("DEEPSEEK_MODEL"):
        env["DEEPSEEK_MODEL"] = os.environ["DEEPSEEK_MODEL"]
    if os.environ.get("OPENROUTER_API_KEY"):
        env["OPENROUTER_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=PER_OP_TIMEOUT, cwd=str(ROOT), env=env)
        rc = proc.returncode
        tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
        timed_out = False
    except subprocess.TimeoutExpired:
        rc = -1
        tail = "TIMEOUT"
        timed_out = True
    dt = round(time.time() - t0, 1)

    row = {"category": category, "elapsed_s": dt, "returncode": rc,
           "timed_out": timed_out}
    row.update(summarize(op))
    if not row.get("status"):
        # agent crashed before writing summary
        row["status"] = "crash" if not timed_out else "timeout"
        row["tail"] = tail[-600:]
    return row


def main() -> None:
    ops = discover_ops()
    results = load_results()
    total = len(ops)
    print(f"[batch] {total} operators discovered; {len(results)} already done")

    for i, (cat, op) in enumerate(ops, 1):
        if op in results and results[op].get("status") not in (None, "crash", "timeout"):
            print(f"[{i}/{total}] {cat}/{op}: SKIP (already {results[op]['status']})")
            continue
        print(f"[{i}/{total}] {cat}/{op}: running...", flush=True)
        row = run_one(cat, op)
        results[op] = row
        save_results(results)
        print(f"[{i}/{total}] {cat}/{op}: {row['status']} "
              f"(kernel={row.get('kernel')} graph={row.get('graph')} "
              f"e2e={row.get('e2e')} prod={row.get('production')} {row['elapsed_s']}s)",
              flush=True)

    # final tally
    ok = sum(1 for r in results.values() if r.get("status") == "success")
    print(f"\n[batch] DONE: {ok}/{total} success")


if __name__ == "__main__":
    main()
