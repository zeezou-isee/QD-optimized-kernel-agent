"""方案C batch driver: optimize ncnn's EXISTING (native) operators in place.

For each op that ncnn already supports AND that maps to a single native layer, the
OperatorAgent (native_override=True, the default) overwrites that native layer with
the agent's kernel, runs end-to-end + production [+ optimize] against it, then
RESTORES the native layer. This script additionally asserts, between ops, that the
ncnn source tree is back to its pristine native state — so one op's run cannot leak
into the next.

Run (needs torch/numpy/openai + built libncnn + pnnx + OPENROUTER_API_KEY):
  python eval/overwrite_native_batch.py --ops Softmax,Sigmoid,AbsVal
  python eval/overwrite_native_batch.py            # default single-layer op set
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = HERE.parent / "agents"
sys.path.insert(0, str(AGENTS.parent))  # repo root, for `import agents`
sys.path.insert(0, str(AGENTS))
import agents as _agents; _agents.bootstrap_paths()

from config import GraphConfig
from operator_agent import OperatorAgent


def _torch_dir():
    try:
        import torch, os
        return Path(os.path.dirname(torch.__file__))
    except Exception:
        return None


# Ops ncnn implements as a single native layer (good 方案C candidates).
DEFAULT_OPS = ["Softmax", "Sigmoid", "AbsVal", "Exp", "ReLU", "TanH"]


def _layer_files(ncnn_root: Path, op: str) -> list[Path]:
    name = op.lower()
    layer = ncnn_root / "src" / "layer"
    files = [layer / f"{name}.h", layer / f"{name}.cpp"]
    files += list(layer.glob(f"*/{name}_*"))
    return [f for f in files if f.exists()]


def _snapshot(ncnn_root: Path, op: str) -> dict[str, str]:
    return {str(f): f.read_text(encoding="utf-8", errors="replace")
            for f in _layer_files(ncnn_root, op)}


def run(ops: list[str], model: str, optimize: bool) -> None:
    cfg = GraphConfig()
    ncnn_root = cfg.ncnn_root
    torch_dir = _torch_dir()
    rows = []
    for op in ops:
        print(f"\n################ {op} (native-override) ################", flush=True)
        before = _snapshot(ncnn_root, op)
        row = {"op": op}
        try:
            s = OperatorAgent(
                task_name=op, model=model, torch_install_dir=torch_dir,
                end_to_end=True, install=False, optimize=optimize,
                native_override=True,
            ).run()
            ph = s.get("phases", {})
            nov = ph.get("native_override", {})
            row.update({
                "status": s.get("status"),
                "override_applied": nov.get("applies"),
                "native_class": nov.get("native_class"),
                "e2e": ph.get("end_to_end_numeric", {}).get("passed"),
                "opt": (ph.get("optimization", {}) or {}).get("production_optimized_ok"),
                "restore_errors": (ph.get("native_restore", {}) or {}).get("errors"),
            })
        except Exception as e:  # noqa: BLE001
            row.update({"status": "crash", "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()

        # SAFETY ASSERTION: tree must be pristine before the next op runs.
        after = _snapshot(ncnn_root, op)
        restored = (after == before)
        row["tree_restored"] = restored
        if not restored:
            changed = [k for k in before if k in after and before[k] != after[k]]
            missing = sorted(set(before) - set(after))
            extra = sorted(set(after) - set(before))
            row["restore_detail"] = {"changed": changed, "missing": missing, "extra": extra}
            print(f"[batch] !!! {op}: ncnn tree NOT restored — changed={changed} "
                  f"missing={missing} extra={extra}")
        else:
            print(f"[batch] {op}: ncnn tree verified pristine after run")
        rows.append(row)
        (HERE / "overwrite_native_results.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = sum(1 for r in rows if r.get("status") == "success")
    clean = sum(1 for r in rows if r.get("tree_restored"))
    print(f"\n[batch] done: {ok}/{len(rows)} success, {clean}/{len(rows)} tree-restored-clean")
    if clean != len(rows):
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="方案C: optimize native ops in place + assert restore")
    p.add_argument("--ops", default=None, help="comma list (default: single-layer native ops)")
    p.add_argument("--model", default="z-ai/glm-5.1")
    p.add_argument("--no-optimize", action="store_true", help="skip the [6] optimization stage")
    args = p.parse_args()
    ops = [o.strip() for o in args.ops.split(",")] if args.ops else DEFAULT_OPS
    run(ops, args.model, optimize=not args.no_optimize)


if __name__ == "__main__":
    main()
