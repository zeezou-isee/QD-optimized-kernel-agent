"""Batch OptimizeAgent runner over the Mobilekernelbench_optimized 30-op set.

Runs run_optimize.py for each op with a REAL-PHONE latency objective (arm +
--device-verify auto), MAP-Elites policy, simpleperf OFF. Resumable: an op whose
optimize summary.json already exists is skipped (unless --force). Each op runs in
its OWN process group with a per-op timeout (SIGTERM the group -> ncnn-tree guard
restores -> SIGKILL if it lingers), mirroring batch/batch_runner.py.

Key safety: the LLM key is read from the ENVIRONMENT only (never written here).
    IDEALAB_API_KEY=<key> .venv/bin/python batch/run_optimize_batch.py
The idealab route needs claude-opus-4-8 (the script's default --model-name).

Usage:
    IDEALAB_API_KEY=... .venv/bin/python batch/run_optimize_batch.py
    ... --ops Softmax,Det                 # just a few
    ... --backend arm --policy map_elites --map-budget 80
    ... --force                           # re-run even if a summary exists
    ... --dry-run                         # print the plan, run nothing

Results -> batch/results/optimize_batch.json (written incrementally).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "batch" / "results"
RUNS        = ROOT / "opgen" / "runs"
VENV_BIN    = ROOT / ".venv" / "bin"
CLI_OPT     = ROOT / "opgen" / "optimize" / "run_optimize.py"
DATASET     = ROOT / "dataset" / "Mobilekernelbench_optimized"

sys.path.insert(0, str(ROOT / "opgen"))
import paths  # noqa: E402


def _ops_from_dataset() -> list[str]:
    """Authoritative op list from the selection manifest; fall back to *.py scan."""
    sel = DATASET / "selection.json"
    if sel.exists():
        return [r["op"] for r in json.loads(sel.read_text(encoding="utf-8"))]
    return sorted(p.stem for p in DATASET.rglob("*.py"))


def _mask(k: str) -> str:
    return f"{k[:6]}...{k[-4:]}" if k and len(k) > 12 else "(set)"


def _child_env() -> dict:
    env = dict(os.environ)
    if VENV_BIN.exists() and str(VENV_BIN) not in env.get("PATH", ""):
        env["PATH"] = f"{VENV_BIN}{os.pathsep}{env.get('PATH', '')}"
    return env


def _summary_path(op: str, backend: str) -> Path:
    return paths.backend_optimize_dir(RUNS, op, backend) / "summary.json"


def _digest(op: str, backend: str) -> dict:
    """Pull the headline numbers out of an optimize summary for the rollup."""
    sp = _summary_path(op, backend)
    if not sp.exists():
        return {}
    try:
        s = json.loads(sp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    extra = s.get("extra") or {}
    best = s.get("best_perf") or {}
    base_lat = extra.get("baseline_latency_ms")
    best_lat = best.get("min") if best.get("min") is not None else best.get("avg")
    speedup = (base_lat / best_lat) if (base_lat and best_lat) else None
    return {
        "best_round": s.get("best_round"),
        "improved": s.get("best_round") not in (None, -1),
        "baseline_latency_ms": base_lat,
        "best_latency_ms": best_lat,
        "self_speedup": round(speedup, 4) if speedup else None,
        "regime": extra.get("regime"),
        "coverage": extra.get("coverage"),
        "stopped_reason": s.get("stopped_reason"),
    }


def run_one(op: str, args, env: dict) -> dict:
    argv = [str(VENV_BIN / "python") if (VENV_BIN / "python").exists() else sys.executable,
            str(CLI_OPT), "--task", op, "--backend", args.backend,
            "--policy", args.policy, "--model-name", args.model_name,
            "--device-verify", args.device_verify,
            "--dataset-root", str(DATASET)]
    if args.map_budget is not None:
        argv += ["--map-budget", str(args.map_budget)]
    if args.inner_budget is not None:
        argv += ["--inner-budget", str(args.inner_budget)]
    if args.max_rounds is not None:
        argv += ["--max-rounds", str(args.max_rounds)]
    if args.record_trace:
        argv += ["--record-trace"]

    t0 = time.time()
    proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    status, tail = "success", ""
    try:
        out, _ = proc.communicate(timeout=args.timeout)
        tail = (out or "")[-1500:]
        if proc.returncode != 0:
            status = "crash"
    except subprocess.TimeoutExpired:
        status = "timeout"
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(5)
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, _ = proc.communicate(timeout=30)
            tail = (out or "")[-1500:]
        except Exception:  # noqa: BLE001
            pass
    elapsed = round(time.time() - t0, 1)
    rec = {"op": op, "status": status, "elapsed_s": elapsed, **_digest(op, args.backend)}
    if status != "success":
        rec["log_tail"] = tail
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch OptimizeAgent over Mobilekernelbench_optimized.")
    ap.add_argument("--ops", default=None, help="comma list (else all 30 from selection.json)")
    ap.add_argument("--backend", default="arm", choices=["base", "arm", "vulkan"])
    ap.add_argument("--policy", default="map_elites", choices=["linear", "map_elites"])
    ap.add_argument("--model-name", default="claude-opus-4-8")
    ap.add_argument("--device-verify", default="auto", choices=["off", "auto", "on"])
    ap.add_argument("--map-budget", type=int, default=None, help="map_elites outer budget (else run_optimize default)")
    ap.add_argument("--inner-budget", type=int, default=None, help="inner-search budget per template (else run_optimize default)")
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--record-trace", action="store_true",
                    help="persist full inner-search trace for paper viz (bloats summaries)")
    ap.add_argument("--timeout", type=int, default=5400, help="per-op wall-clock seconds (default 90 min)")
    ap.add_argument("--force", action="store_true", help="re-run even if an optimize summary exists")
    ap.add_argument("--out", default=str(RESULTS_DIR / "optimize_batch.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # --- key check (env only; never printed in full, never written) ---
    key = os.environ.get("IDEALAB_API_KEY", "")
    if not args.dry_run and args.model_name in ("claude-opus-4-8",) and not key:
        sys.exit("IDEALAB_API_KEY not set. Run: IDEALAB_API_KEY=<key> .venv/bin/python "
                 "batch/run_optimize_batch.py")

    ops = [o.strip() for o in args.ops.split(",")] if args.ops else _ops_from_dataset()
    env = _child_env()

    out_path = Path(args.out)
    results = {}
    if out_path.exists() and not args.force:
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results = {}

    todo, skip = [], []
    for op in ops:
        done = _summary_path(op, args.backend).exists() and not args.force
        (skip if done else todo).append(op)

    print("=" * 72)
    print(f"BATCH OPTIMIZE  backend={args.backend} policy={args.policy} "
          f"model={args.model_name} key={_mask(key)}")
    print(f"  dataset  : {DATASET.name}")
    print(f"  device   : --device-verify {args.device_verify}  (simpleperf OFF)")
    print(f"  budget   : map={args.map_budget or 'default'}  per-op timeout={args.timeout}s")
    print(f"  ops      : {len(ops)} total | {len(todo)} to run | {len(skip)} skipped (have summary)")
    if skip:
        print(f"  skipping : {', '.join(skip)}")
    print("=" * 72)
    if args.dry_run:
        for op in todo:
            print(f"  would run: {op}")
        return

    for i, op in enumerate(todo, 1):
        print(f"\n----- [{i}/{len(todo)}] {op} -----", flush=True)
        rec = run_one(op, args, env)
        results[op] = rec
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        sp = rec.get("self_speedup")
        print(f"  -> {rec['status']} in {rec['elapsed_s']}s  "
              f"improved={rec.get('improved')} self_speedup={sp} "
              f"base={rec.get('baseline_latency_ms')}ms best={rec.get('best_latency_ms')}ms",
              flush=True)

    ok = sum(1 for r in results.values() if r.get("status") == "success")
    imp = sum(1 for r in results.values() if r.get("improved"))
    print("\n" + "=" * 72)
    print(f"DONE  {ok}/{len(results)} success | {imp} improved over baseline")
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
