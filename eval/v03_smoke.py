"""v0.3 smoke: real glm-5.1, 4 representative ops, exercise the new 7-stage flow.

- Erf       already-in-ncnn  (skip graph)
- Greater   not-in-ncnn      (simple)
- Mod       not-in-ncnn      (pnnx.Expression)
- CumSum    not-in-ncnn      (axis param)

No benchmark (no android device). Optimize=on with 2 rounds (placeholder).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = HERE.parent / "agents"
sys.path.insert(0, str(AGENTS.parent))  # repo root, for `import agents`
sys.path.insert(0, str(AGENTS))         # for top-level config/llm_api
import agents as _agents; _agents.bootstrap_paths()  # add subdirs to sys.path

from operator_agent import OperatorAgent

def _torch_dir():
    """Installed torch's dir (so pnnx links the right libtorch); None lets pnnx auto-probe."""
    try:
        import torch, os
        return Path(os.path.dirname(torch.__file__))
    except Exception:
        return None


TORCH = _torch_dir()

CASES = [
    ("Erf", "already-in-ncnn (skip graph)"),
    ("Greater", "not-in-ncnn (simple)"),
    ("Mod", "not-in-ncnn (Expression)"),
    ("CumSum", "not-in-ncnn (axis param)"),
]


def main() -> None:
    rows = []
    for op, kind in CASES:
        print(f"\n################ {op} — {kind} ################", flush=True)
        rec = {"op": op, "kind": kind}
        try:
            s = OperatorAgent(
                task_name=op, model="z-ai/glm-5.1",
                max_rounds=6, graph_max_rounds=15,
                torch_install_dir=TORCH,
                end_to_end=True, install=False,
                compile_mode="build_lib", benchmark=False,
                optimize=True, max_optimize_rounds=2, improve_tol=0.02,
            ).run()
            ph = s.get("phases", {})
            rec["status"] = s.get("status")
            rec["kernel"] = ph.get("kernel", {}).get("status")
            ec = ph.get("existence_check", {})
            rec["already_in_ncnn"] = ec.get("already_in_ncnn")
            rec["baseline_op_types"] = ec.get("baseline_op_types")
            rec["graph"] = ph.get("graph", {}).get("status")
            rec["e2e"] = ph.get("end_to_end_numeric", {}).get("passed")
            rec["e2e_diff"] = ph.get("end_to_end_numeric", {}).get("max_diff")
            prod = ph.get("production", {})
            rec["prod_compile"] = (prod.get("compile") or {}).get("ok")
            rec["prod_correctness"] = (prod.get("correctness") or {}).get("passed")
            opt = ph.get("optimization") or {}
            rec["opt_stopped"] = opt.get("stopped_reason")
            rec["opt_iters"] = len(opt.get("iterations", []))
        except Exception as e:  # noqa: BLE001
            rec["status"] = "crash"
            rec["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        rows.append(rec)
        (HERE / "v03_smoke_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"   -> status={rec.get('status')} already={rec.get('already_in_ncnn')} "
              f"graph={rec.get('graph')} e2e={rec.get('e2e')} prod_c={rec.get('prod_compile')} "
              f"prod_corr={rec.get('prod_correctness')} opt={rec.get('opt_stopped')}", flush=True)

    print("\n=== DONE ===")
    succ = [r for r in rows if r.get("status") == "success"]
    print(f"success: {len(succ)}/{len(rows)}")


if __name__ == "__main__":
    main()
