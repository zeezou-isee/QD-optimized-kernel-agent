"""Batch: author + verify + optimize the ARM kernel for every dataset operator.

Per operator (3 stages, all on-machine, arm64 NEON):
  1) KernelAgent backend=base  -> base kernel, compile + numeric vs PyTorch
  2) KernelAgent backend=arm   -> NEON/NC4HW4 subclass, compile + numeric (packing=4)
  3) OptimizeAgent backend=arm -> baseline vs best latency (each candidate packed-对拍ed)

Records, per op: compile_ok / numeric_ok / max_diff (base & arm) + optimize
baseline_ms / best_ms / improvement. Robust (one op's failure never stops the
batch), resumable (skips ops already in results.json), incremental (writes after
each op). Results: runs/_arm_batch/results.json + report.md.

Usage:
  python agents/cli/run_arm_batch.py --model-name z-ai/glm-5.2          # all 183 ops
  python agents/cli/run_arm_batch.py --category Unary,Activation        # subset
  python agents/cli/run_arm_batch.py --ops Abs,Exp,Sqrt                 # explicit
  python agents/cli/run_arm_batch.py --limit 10                         # first N
  python agents/cli/run_arm_batch.py --skip-optimize                    # author+verify only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import agents as _agents; _agents.bootstrap_paths()

from config import GraphConfig, RUNS_ROOT
from kernel_agent import KernelAgent
from llm_api import query_llm
from optimize_agent import OptimizeAgent

OUT_DIR = RUNS_ROOT / "_arm_batch"


def _discover_ops(dataset_root: Path) -> list[tuple[str, str, str]]:
    """(category, op_name, py_path) for every dataset .py (excluding helpers)."""
    out = []
    for py in sorted(dataset_root.rglob("*.py")):
        if "_cache_" in py.name or py.stem.startswith("_"):
            continue
        out.append((py.parent.name, py.stem, str(py)))
    return out


def _kernel_row(summary: dict) -> dict:
    fr = summary.get("final_result") or {}
    return {
        "status": summary.get("status"),
        "rounds": summary.get("rounds"),
        "compile_ok": bool(fr.get("compile_ok")),
        "numeric_ok": bool(fr.get("numeric_ok")),
        "max_diff": fr.get("max_diff"),
    }


def _run_one(op: str, py_path: str, model: str, cfg_rounds: int,
             do_optimize: bool, map_budget: int, inner_budget: int) -> dict:
    row: dict = {"op": op, "model_py": py_path}

    # --- 1) base kernel ---
    cfg = GraphConfig(model=model, max_rounds=cfg_rounds, run_numeric=True)
    base_sum = KernelAgent(task_name=op, model_py=py_path, cfg=cfg,
                           llm_query=query_llm, backend="base").run()
    row["base"] = _kernel_row(base_sum)
    if base_sum.get("status") != "success":
        row["stage"] = "base_failed"
        return row
    base_code = (base_sum.get("final_result") or {}).get("response_code") or {}
    base_prof = base_sum.get("kernel_profile") or {}

    # --- 2) arm kernel ---
    arm_sum = KernelAgent(task_name=op, model_py=py_path, cfg=cfg, llm_query=query_llm,
                          backend="arm", base_kernel_code=base_code,
                          base_profile=base_prof).run()
    row["arm"] = _kernel_row(arm_sum)
    if arm_sum.get("status") != "success":
        row["stage"] = "arm_failed"
        return row
    arm_code = (arm_sum.get("final_result") or {}).get("response_code") or {}

    if not do_optimize:
        row["stage"] = "verified"
        return row

    # --- 3) optimize arm kernel ---
    weight_keys = list(base_prof.get("weight_keys", []) or [])
    params = {int(k): v for k, v in (base_prof.get("params") or {}).items()}
    res = OptimizeAgent(
        task_name=op, baseline_kernel_code=arm_code, model_py=py_path,
        ncnn_root=cfg.ncnn_root, llm_query=query_llm, model=model,
        weight_keys=weight_keys, params=params, backend="arm", base_files=base_code,
        policy="map_elites", map_budget=map_budget, inner_budget=inner_budget,
        coverage_target=2, op_class=base_prof.get("class_name", ""),
    ).run().to_dict()
    e = res.get("extra", {}) or {}
    base_ms = e.get("baseline_latency_ms")
    best_ms = (res.get("best_perf") or {}).get("avg")
    imp = None
    if base_ms and best_ms:
        imp = round((base_ms - best_ms) / base_ms * 100, 2)
    row["optimize"] = {
        "baseline_ms": base_ms, "best_ms": best_ms, "improvement_pct": imp,
        "best_round": res.get("best_round"), "rounds": e.get("rounds"),
        "regime": e.get("regime"), "argmin_cell": e.get("argmin_cell"),
        "stopped": res.get("stopped_reason"),
    }
    row["stage"] = "optimized"
    return row


def _write_report(results: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# ARM-backend batch: compile / correctness / performance", "",
             "| op | base compile | base numeric | arm compile | arm numeric | "
             "arm max_diff | baseline ms | best ms | Δ% | stage |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    n_base = n_arm = n_opt = n_improved = 0
    for r in results:
        b = r.get("base", {}); a = r.get("arm", {}); o = r.get("optimize", {})
        if b.get("status") == "success": n_base += 1
        if a.get("status") == "success": n_arm += 1
        if r.get("stage") == "optimized": n_opt += 1
        if (o.get("improvement_pct") or 0) > 0: n_improved += 1
        lines.append("| {op} | {bc} | {bn} | {ac} | {an} | {ad} | {bms} | {best} | {imp} | {st} |".format(
            op=r["op"],
            bc="✅" if b.get("compile_ok") else "—", bn="✅" if b.get("numeric_ok") else "—",
            ac="✅" if a.get("compile_ok") else "—", an="✅" if a.get("numeric_ok") else "—",
            ad=(f"{a.get('max_diff'):.2g}" if a.get("max_diff") is not None else "—"),
            bms=(f"{o.get('baseline_ms'):.3f}" if o.get("baseline_ms") else "—"),
            best=(f"{o.get('best_ms'):.3f}" if o.get("best_ms") else "—"),
            imp=(f"{o.get('improvement_pct')}" if o.get("improvement_pct") is not None else "—"),
            st=r.get("stage", "?")))
    total = len(results)
    lines += ["", f"**Totals**: {total} ops | base ok {n_base} | arm ok {n_arm} | "
                  f"optimized {n_opt} | improved {n_improved}"]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Batch arm-backend author+verify+optimize over the dataset.")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--model-name", default="z-ai/glm-5.2")
    p.add_argument("--category", default=None, help="comma list of categories to include")
    p.add_argument("--ops", default=None, help="comma list of explicit op names")
    p.add_argument("--limit", type=int, default=0, help="cap number of ops (0 = all)")
    p.add_argument("--max-rounds", type=int, default=3, help="KernelAgent rounds per backend")
    p.add_argument("--map-budget", type=int, default=10)
    p.add_argument("--inner-budget", type=int, default=4)
    p.add_argument("--skip-optimize", action="store_true", help="author+verify only (no perf)")
    p.add_argument("--no-resume", action="store_true", help="ignore existing results.json")
    args = p.parse_args()

    ds = Path(args.dataset_root) if args.dataset_root else (
        Path(__file__).resolve().parents[2] / "dataset" / "Mobilekernelbench")
    ops = _discover_ops(ds)
    if args.category:
        cats = {c.strip() for c in args.category.split(",")}
        ops = [o for o in ops if o[0] in cats]
    if args.ops:
        want = {o.strip() for o in args.ops.split(",")}
        ops = [o for o in ops if o[1] in want]
    if args.limit:
        ops = ops[:args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    done: set[str] = set()
    rj = OUT_DIR / "results.json"
    if rj.exists() and not args.no_resume:
        results = json.loads(rj.read_text(encoding="utf-8"))
        done = {r["op"] for r in results}

    todo = [o for o in ops if o[1] not in done]
    print(f"[batch] dataset={ds}\n[batch] {len(ops)} ops selected, {len(done)} already done, "
          f"{len(todo)} to run; optimize={not args.skip_optimize}")

    for i, (cat, op, py) in enumerate(todo, 1):
        t0 = time.time()
        print(f"\n[batch {i}/{len(todo)}] {cat}/{op} ...")
        try:
            row = _run_one(op, py, args.model_name, args.max_rounds,
                           not args.skip_optimize, args.map_budget, args.inner_budget)
        except Exception as exc:  # noqa: BLE001 — never let one op kill the batch
            import traceback
            row = {"op": op, "stage": "exception",
                   "error": f"{type(exc).__name__}: {exc}",
                   "trace": traceback.format_exc()[-1500:]}
        row["category"] = cat
        row["elapsed_s"] = round(time.time() - t0, 1)
        results.append(row)
        _write_report(results)
        o = row.get("optimize", {})
        print(f"[batch {i}/{len(todo)}] {op}: stage={row.get('stage')} "
              f"arm_numeric={row.get('arm', {}).get('numeric_ok')} "
              f"imp={o.get('improvement_pct')}% ({row['elapsed_s']}s)")

    print(f"\n[batch] DONE. results -> {rj}  report -> {OUT_DIR/'report.md'}")


if __name__ == "__main__":
    main()
