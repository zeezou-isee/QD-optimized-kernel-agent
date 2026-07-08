"""Measure each op's BASELINE latency on the phone in the optimize SANDBOX
(single-layer runner, same shape/harness the OptimizeAgent uses) — no LLM.

For each op it builds the same Evaluator run_optimize builds (device_measure on,
bench/warmup configurable) and evaluates the verified baseline once, recording the
on-device avg/min/max ms. Use it to decide which ops are genuinely ms-scale
(above the device noise floor) before curating the dataset.

Usage:
    .venv/bin/python scripts/device_sandbox_probe.py --ops Conv,Winograd_Convolution_2D
    .venv/bin/python scripts/device_sandbox_probe.py --ops-file /tmp/probe_ops.txt
    -> batch/results/sandbox_probe.json  (op -> {avg_ms,min_ms,max_ms,ok,error})
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # repo root, so `import opgen` (the package) works
import opgen; opgen.bootstrap_paths()  # noqa: E402  (adds opgen/ subdirs to path)
from config import RUNS_ROOT, GraphConfig   # noqa: E402
import paths                                 # noqa: E402
from schemas import ParameterizedTemplate    # noqa: E402
from evaluator import Evaluator              # noqa: E402

RESULTS = ROOT / "batch" / "results"
DATASET = ROOT / "MobileKernelBench_git"  # placeholder; model resolved from full dataset
FULL_DS = ROOT / "dataset" / "Mobilekernelbench"


def _kernel(op: str, be: str) -> dict:
    summ = paths.kernel_summary(RUNS_ROOT, op, be)
    if not summ.exists():
        return {}
    d = json.loads(summ.read_text(encoding="utf-8"))
    return (d.get("final_result") or {}).get("response_code") or {}


def _model_py(op: str) -> str | None:
    hits = sorted(FULL_DS.rglob(f"{op}.py"))
    return str(hits[0]) if hits else None


def probe(op: str, backend: str, bench: int, warmup: int) -> dict:
    base = _kernel(op, "base"); arm = _kernel(op, backend)
    mp = _model_py(op)
    if not (base and mp):
        return {"ok": False, "error": "missing baseline/model"}
    baseline = arm or base
    base_files = base if backend in ("arm", "vulkan") else {}
    # profile: params + weight_keys
    prof = {}
    for pp in (paths.kernel_profile_shared_json(RUNS_ROOT, op),
               RUNS_ROOT / op / "analyze" / "kernel_profile.json",
               RUNS_ROOT / op / "kernel" / "kernel_profile.json"):
        if pp.exists():
            prof = json.loads(pp.read_text(encoding="utf-8")); break
    params = {int(k): v for k, v in (prof.get("params") or {}).items()}
    weight_keys = list(prof.get("weight_keys") or [])
    ncnn_py = next(iter(sorted((RUNS_ROOT / op).rglob("*_ncnn.py"))), None)
    try:
        ev = Evaluator(baseline_kernel=baseline, model_py=mp,
                       ncnn_root=GraphConfig().ncnn_root, weight_keys=weight_keys,
                       params=params, backend=backend, base_files=base_files,
                       device_measure=True, device_bench=bench, device_warmup=warmup,
                       ncnn_py=str(ncnn_py) if ncnn_py else None)
        if ev._measurer is None or not ev._measurer.available():
            return {"ok": False, "error": "no device"}
        base_t = ParameterizedTemplate(kernel_files=dict(baseline), params={},
                                       class_name=ev.class_name, header=ev.header,
                                       file=ev.file, rationale="baseline", techniques=[])
        s = ev.evaluate(base_t, {})
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[-200:]}
    if not s.correct or s.latency_ms is None:
        return {"ok": False, "error": s.error or "no latency (host fallback / device fail)"}
    return {"ok": True, "avg_ms": s.latency_ms, "min_ms": s.latency_min_ms,
            "max_ms": s.latency_max_ms, "n": s.n_runs}


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe on-device sandbox baseline latency per op.")
    ap.add_argument("--ops", default=None)
    ap.add_argument("--ops-file", default=None)
    ap.add_argument("--backend", default="arm")
    ap.add_argument("--bench", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=str(RESULTS / "sandbox_probe.json"))
    args = ap.parse_args()

    if args.ops_file:
        ops = [l.strip() for l in Path(args.ops_file).read_text().splitlines() if l.strip()]
    else:
        ops = [o.strip() for o in (args.ops or "").split(",") if o.strip()]
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() else {}
    for i, op in enumerate(ops, 1):
        r = probe(op, args.backend, args.bench, args.warmup)
        results[op] = r
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        tag = (f"avg={r['avg_ms']:.4f} min={r['min_ms']:.4f} max={r['max_ms']:.4f}ms"
               if r.get("ok") else f"FAIL: {r.get('error')}")
        print(f"[{i}/{len(ops)}] {op:<44} {tag}", flush=True)


if __name__ == "__main__":
    main()
