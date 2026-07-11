"""A/B: single-phase (control) vs two-phase illumination, on the real phone.

For each op runs run_optimize.py twice at a MATCHED eval budget:
  - single : --fill-budget 0  (genuine old single-phase loop, full budget)
  - two    : --fill-budget F  (Phase-1 cheap fill + Phase-2 bounded deep search)
and compares grid thickness (coverage) vs search burden (evals). Writes each run
to its own --out-dir (never clobbers the real runs/ summaries), then emits a
comparison table to /tmp/ab/ab_results.{json,md}.

    IDEALAB_API_KEY=... .venv/bin/python scripts/ab_two_phase.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "opgen" / "optimize" / "run_optimize.py"
DATASET = ROOT / "dataset" / "Mobilekernelbench_optimized"
PY = ROOT / ".venv" / "bin" / "python"
OUT = Path("/tmp/ab")

OPS = ["Dense_Convolution_2D", "Winograd_Convolution_2D", "Group_Convolution_2D", "Cos"]
MAP_BUDGET, INNER = 24, 4
FILL, TOPK = 12, 4           # two-phase: 12 cheap fills + up to 4 deep
PER_OP_TIMEOUT = 3600        # 60 min/run safety


def ensure_device(retries: int = 3) -> bool:
    """Make sure adb sees a live device before a run (the phone drops when idle).
    Resets the adb server + waits on failure. Returns True once responsive."""
    for _ in range(retries):
        try:
            out = subprocess.run(["adb", "shell", "echo", "alive"], capture_output=True,
                                 text=True, timeout=20).stdout
            if "alive" in out:
                return True
        except Exception:  # noqa: BLE001
            pass
        subprocess.run(["adb", "kill-server"], capture_output=True, timeout=20)
        time.sleep(2)
        subprocess.run(["adb", "start-server"], capture_output=True, timeout=20)
        subprocess.run(["adb", "wait-for-device"], capture_output=True, timeout=60)
        time.sleep(2)
    return False


def run(op: str, arm: str) -> dict:
    if not ensure_device():
        return {"status": "no_device", "elapsed_s": 0.0, "summary": False}
    out_dir = OUT / op / arm
    argv = [str(PY), str(CLI), "--task", op, "--backend", "arm", "--policy", "map_elites",
            "--model-name", "claude-opus-4-8", "--device-verify", "auto",
            "--dataset-root", str(DATASET), "--map-budget", str(MAP_BUDGET),
            "--inner-budget", str(INNER), "--out-dir", str(out_dir)]
    if arm == "single":
        argv += ["--fill-budget", "0"]
    else:
        argv += ["--fill-budget", str(FILL), "--optimize-topk", str(TOPK)]
    t0 = time.time()
    try:
        p = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True,
                           timeout=PER_OP_TIMEOUT)
        status = "ok" if p.returncode == 0 else "crash"
        tail = (p.stdout or "")[-800:] + (p.stderr or "")[-800:]
    except subprocess.TimeoutExpired:
        status, tail = "timeout", ""
    dt = round(time.time() - t0, 1)
    return {"status": status, "elapsed_s": dt, **digest(out_dir / "summary.json"), "tail": tail}


def digest(sp: Path) -> dict:
    if not sp.exists():
        return {"summary": False}
    try:
        s = json.loads(sp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"summary": False}
    e = s.get("extra") or {}
    bp = s.get("best_perf") or {}
    base, best = e.get("baseline_latency_ms"), bp.get("avg")
    dp = e.get("dispatch_prior") or {}
    return {"summary": True, "two_phase": e.get("two_phase"), "regime": e.get("regime"),
            "coverage": e.get("coverage"), "filled_p1": e.get("phase1_filled"),
            "p1_evals": e.get("phase1_evals"), "p2_evals": e.get("phase2_evals"),
            "rounds": e.get("rounds"), "baseline_ms": base, "best_ms": best,
            "speedup": round(base / best, 3) if (base and best) else None,
            "winner_cell": e.get("argmin_cell"),
            "dispatch": dp.get("preferred") if dp else None}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    resfile = OUT / "ab_results.json"
    if resfile.exists():
        res = json.loads(resfile.read_text(encoding="utf-8"))
    for op in OPS:
        for arm in ("single", "two"):
            key = f"{op}/{arm}"
            if res.get(key, {}).get("summary"):
                print(f"skip {key} (have summary)", flush=True)
                continue
            print(f"\n===== {key} =====", flush=True)
            r = run(op, arm)
            res[key] = r
            resfile.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {r['status']} {r['elapsed_s']}s cov={r.get('coverage')} "
                  f"filled={r.get('filled_p1')} p1={r.get('p1_evals')} p2={r.get('p2_evals')} "
                  f"rounds={r.get('rounds')} best={r.get('best_ms')} spd={r.get('speedup')}", flush=True)

    # comparison table
    md = ["# Two-phase A/B (arm, real phone)\n",
          f"budget: map={MAP_BUDGET} inner={INNER} | two-phase fill={FILL} topk={TOPK}\n",
          "\n| op | arm | regime | cov | filled | p1 | p2 | rounds | baseline_ms | best_ms | speedup | winner | dispatch |",
          "|----|-----|--------|----:|-------:|---:|---:|-------:|-----------:|--------:|--------:|--------|----------|"]
    for op in OPS:
        for arm in ("single", "two"):
            r = res.get(f"{op}/{arm}", {})
            if not r.get("summary"):
                md.append(f"| `{op}` | {arm} | — | — | — | — | — | — | — | — | — | {r.get('status','?')} | — |")
                continue
            wc = "/".join(map(str, r["winner_cell"])) if r.get("winner_cell") else "—"
            dp = ",".join(r["dispatch"]) if r.get("dispatch") else "—"
            md.append(f"| `{op}` | {arm} | {r.get('regime','—')} | {r.get('coverage','—')} | "
                      f"{r.get('filled_p1','—')} | {r.get('p1_evals','—')} | {r.get('p2_evals','—')} | "
                      f"{r.get('rounds','—')} | {r.get('baseline_ms','—')} | {r.get('best_ms','—')} | "
                      f"{r.get('speedup','—')} | {wc} | {dp} |")
    (OUT / "ab_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nwrote {OUT/'ab_results.json'} and {OUT/'ab_results.md'}")


if __name__ == "__main__":
    sys.exit(main())
