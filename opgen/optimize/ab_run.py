"""A/B driver for the OptimizeAgent wiki injection (v0).

Runs each task twice — once with KERNELGEN_WIKI=on and once with =off —
against the arm baseline in `opgen/runs_arm/`, writing per-run summaries to
`opgen/runs_arm_optmz/<task>/optimize_wiki_{on,off}/` and an aggregate
`ab_report.json` in the same tree root.

Signals collected (per the plan file, §端到端验证方式):
  1) first-round compile pass rate  (fraction with no E1_COMPILE in round 0)
  2) failure-code drift             (Counter over all rounds, per-mode)
  3) best-latency vs baseline       (best_perf.avg / baseline avg, per-mode)

The driver is sequential — Vulkan/arm harnesses use process-groups and
subprocess-timed measurements that don't tolerate two runs sharing the same
workdir. Wall time budget: ~10 tasks × 2 modes × ~5 min per run ≈ 90 min.

Env in:
  IDEALAB_API_KEY / IDEALAB_API_TOKEN — proxy creds
Env out (per run):
  KERNELGEN_WIKI={on|off}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_OPTIMIZE = REPO / "opgen" / "optimize" / "run_optimize.py"
VENV_PY = REPO / ".venv" / "bin" / "python"

# Bootstrap opgen path so `import paths` works from this script.
sys.path.insert(0, str(REPO / "opgen"))
import paths  # noqa: E402

# 10-task, 3-family A/B matrix. Aligned with the wiki playbook pages:
# elementwise_binary / reduction / conv.
DEFAULT_TASKS: list[tuple[str, str]] = [
    ("Add", "elementwise_binary"),
    ("Mul", "elementwise_binary"),
    ("Greater", "elementwise_binary"),
    ("ReduceMean", "reduction"),
    ("ReduceMax", "reduction"),
    ("ReduceSum", "reduction"),
    ("Conv", "conv"),
    ("Dense_Convolution_2D", "conv"),
    ("Winograd_Convolution_2D", "conv"),
    ("Group_Convolution_2D_kernel", "conv"),
]

MODES = ("on", "off")


def _extract_signals(summary: dict) -> dict:
    """Distill the OptimizeResult summary into the A/B signals we care about.
    Reads iterations for E-codes and round0/best latency."""
    iters = summary.get("iterations") or []
    ecodes = Counter()
    round0_e1 = False
    for it in iters:
        basin = it.get("basin") or {}
        for s in (basin.get("samples") or []):
            cat = (s.get("correctness") or {}).get("failure_category") or ""
            if cat:
                ecodes[cat] += 1
                if it.get("round_idx") == 0 and cat == "E1_COMPILE":
                    round0_e1 = True
    best_perf = summary.get("best_perf") or {}
    return {
        "best_perf_avg_ms": best_perf.get("avg"),
        "best_perf_min_ms": best_perf.get("min"),
        "best_round": summary.get("best_round"),
        "stopped_reason": summary.get("stopped_reason"),
        "n_iterations": len(iters),
        "round0_e1_compile": round0_e1,
        "ecode_counts": dict(ecodes),
    }


def _run_one(task: str, family: str, mode: str, args) -> dict:
    # New layout: A/B outputs live UNDER the target backend's optimize dir
    # (runs/<task>/backends/<backend>/optimize/wiki_{on,off}/). The old flat
    # `optimize_wiki_{mode}` name migrates to a `wiki_{mode}` subdir there.
    out_dir = paths.backend_optimize_dir(args.out_root, task, args.backend) / f"wiki_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["KERNELGEN_WIKI"] = mode
    # unify the two possible env-var names into the one llm_api.py wants
    tok = env.get("IDEALAB_API_KEY") or env.get("IDEALAB_API_TOKEN")
    if tok:
        env["IDEALAB_API_KEY"] = tok

    cmd = [
        str(VENV_PY), str(RUN_OPTIMIZE),
        "--task", task,
        "--backend", args.backend,
        "--model-name", args.model_name,
        "--runs-root", args.runs_root,
        "--dataset-root", args.dataset_root,
        "--out-dir", str(out_dir),
        "--max-rounds", str(args.max_rounds),
        "--inner-budget", str(args.inner_budget),
        "--runs", str(args.runs),
        "--warmup", str(args.warmup),
    ]
    log_path = out_dir / "run.log"
    t0 = time.time()
    print(f"[{task}/{family}/{mode}] START -> {out_dir}", flush=True)
    with log_path.open("w") as lf:
        proc = subprocess.run(cmd, env=env, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                              timeout=args.per_run_timeout)
    wall = time.time() - t0
    summary_path = out_dir / "summary.json"
    if proc.returncode != 0 or not summary_path.exists():
        print(f"[{task}/{family}/{mode}] FAIL rc={proc.returncode} wall={wall:.1f}s "
              f"(tail: {log_path})", flush=True)
        return {
            "task": task, "family": family, "mode": mode, "ok": False,
            "returncode": proc.returncode, "wall_s": round(wall, 1),
            "log_path": str(log_path), "signals": {},
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sig = _extract_signals(summary)
    kept_rounds = sum(1 for it in (summary.get("iterations") or []) if it.get("kept"))
    print(f"[{task}/{family}/{mode}] OK  wall={wall:.1f}s  best={sig['best_perf_avg_ms']} "
          f"kept={kept_rounds}/{sig['n_iterations']}  ecodes={sig['ecode_counts']}",
          flush=True)
    return {
        "task": task, "family": family, "mode": mode, "ok": True,
        "returncode": 0, "wall_s": round(wall, 1),
        "log_path": str(log_path), "signals": sig,
    }


def _aggregate(cells: list[dict]) -> dict:
    """Fold per-cell records into the 3 signals grouped by mode + family."""
    by_mode: dict[str, dict] = {m: {"cells": [], "ecodes": Counter(), "compile_first_ok": 0,
                                    "n_ok": 0} for m in MODES}
    per_family: dict[str, dict] = {}
    for c in cells:
        if not c["ok"]:
            continue
        m = c["mode"]; s = c["signals"]
        by_mode[m]["cells"].append({
            "task": c["task"], "family": c["family"],
            "best_ms": s.get("best_perf_avg_ms"),
            "compile_first_ok": not s.get("round0_e1_compile", True),
            "ecodes": s.get("ecode_counts", {}),
        })
        by_mode[m]["ecodes"].update(s.get("ecode_counts", {}))
        by_mode[m]["compile_first_ok"] += 0 if s.get("round0_e1_compile") else 1
        by_mode[m]["n_ok"] += 1
        pf = per_family.setdefault(c["family"], {m2: {"n": 0, "best_ms_sum": 0.0,
                                                       "compile_first_ok": 0}
                                                  for m2 in MODES})
        pf[m]["n"] += 1
        if s.get("best_perf_avg_ms") is not None:
            pf[m]["best_ms_sum"] += float(s["best_perf_avg_ms"])
        if not s.get("round0_e1_compile", True):
            pf[m]["compile_first_ok"] += 1
    # finalize
    for m in MODES:
        n = by_mode[m]["n_ok"]
        by_mode[m]["compile_first_rate"] = (by_mode[m]["compile_first_ok"] / n) if n else None
        by_mode[m]["ecodes"] = dict(by_mode[m]["ecodes"])
    for fam, mm in per_family.items():
        for m in MODES:
            if mm[m]["n"]:
                mm[m]["mean_best_ms"] = mm[m]["best_ms_sum"] / mm[m]["n"]
                mm[m]["compile_first_rate"] = mm[m]["compile_first_ok"] / mm[m]["n"]
    return {"by_mode": by_mode, "per_family": per_family}


def main() -> None:
    p = argparse.ArgumentParser(description="A/B driver for OptimizeAgent wiki (v0).")
    p.add_argument("--tasks-file", default=None,
                   help="path to a text file with one 'task,family' per line; default = 10 built-ins")
    p.add_argument("--backend", choices=["arm", "vulkan"], default="arm",
                   help="which backend the baselines live under & the optimize loop targets")
    p.add_argument("--out-root", default=None,
                   help="default: opgen/runs_<backend>_optmz")
    p.add_argument("--runs-root", default=None,
                   help="default: opgen/runs_<backend>")
    p.add_argument("--dataset-root", default=str(REPO / "dataset" / "Mobilekernelbench"))
    p.add_argument("--model-name", default="idealab/claude-opus-4-8")
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--inner-budget", type=int, default=8)
    p.add_argument("--runs", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--per-run-timeout", type=int, default=1800)
    p.add_argument("--modes", default="off,on", help="comma list, subset of {on,off}")
    p.add_argument("--only", default=None, help="only run these task names (comma-sep)")
    args = p.parse_args()

    # backend-scoped defaults so the same driver serves arm and vulkan runs
    if args.runs_root is None:
        args.runs_root = str(REPO / "opgen" / f"runs_{args.backend}")
    if args.out_root is None:
        args.out_root = str(REPO / "opgen" / f"runs_{args.backend}_optmz")

    tasks = DEFAULT_TASKS
    if args.tasks_file:
        tasks = []
        for line in Path(args.tasks_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            tasks.append((parts[0], parts[1] if len(parts) > 1 else "unknown"))
    if args.only:
        keep = {t.strip() for t in args.only.split(",")}
        tasks = [(t, f) for t, f in tasks if t in keep]

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip() in MODES)
    if not modes:
        print("no valid modes selected", file=sys.stderr); sys.exit(2)

    cells: list[dict] = []
    report_path = Path(args.out_root) / "ab_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(tasks) * len(modes)
    idx = 0
    for mode in modes:                             # outer loop = mode (batching cache warms)
        for task, family in tasks:
            idx += 1
            print(f"\n=== [{idx}/{total}] task={task} family={family} mode={mode} ===", flush=True)
            try:
                cell = _run_one(task, family, mode, args)
            except subprocess.TimeoutExpired:
                print(f"[{task}/{family}/{mode}] TIMEOUT after {args.per_run_timeout}s", flush=True)
                cell = {"task": task, "family": family, "mode": mode, "ok": False,
                        "returncode": -1, "wall_s": args.per_run_timeout,
                        "log_path": str(Path(args.out_root) / task / f"optimize_wiki_{mode}" / "run.log"),
                        "signals": {}}
            cells.append(cell)
            # incremental checkpoint every cell in case of interruption
            report_path.write_text(json.dumps({
                "cells": cells, "aggregate": _aggregate(cells)
            }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== A/B done. Report: {report_path} ===")


if __name__ == "__main__":
    main()
