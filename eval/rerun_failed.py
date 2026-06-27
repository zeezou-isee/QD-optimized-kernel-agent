"""Re-run the previously-FAILED ops with the dtype-fixed harness, full end-to-end.

For each op: kernel(numeric) + install + graph(forced target) + end-to-end numeric
(temporary, restored). Records the new verdict + pnnx op_types so we can compare
against the pre-fix run and update the report.
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
RUNS = AGENTS / "runs"

# (op, previous failure mode)
FAILED = [
    ("And", "graph(dtype)"),
    ("Where", "graph(dtype)"),
    ("BitwiseAnd", "graph(dtype)"),
    ("Cast", "graph(identity-fold)"),
    ("CumSum", "graph(axis)"),
    ("Trilu_lower", "graph"),
    ("GatherElements", "graph"),
    ("ScatterElements", "kernel"),
    ("OneHot", "kernel"),
    ("Det", "kernel"),
    ("Unique", "kernel"),
]


def _pnnx_types(op: str):
    p = RUNS / op / "graph" / "pnnx_ir_probe.json"
    if p.exists():
        return json.loads(p.read_text()).get("op_types")
    return None


def main() -> None:
    rows = []
    for op, prev in FAILED:
        print(f"\n################ {op} (was: {prev}) ################", flush=True)
        rec = {"op": op, "prev_fail": prev}
        try:
            s = OperatorAgent(task_name=op, model="z-ai/glm-5.1", max_rounds=5,
                              torch_install_dir=TORCH, end_to_end=True, install=False).run()
            ph = s.get("phases", {})
            rec["status"] = s.get("status")
            rec["kernel"] = ph.get("kernel", {}).get("status")
            rec["graph"] = ph.get("graph", {}).get("status")
            rec["e2e"] = ph.get("end_to_end_numeric", {}).get("passed")
            rec["e2e_diff"] = ph.get("end_to_end_numeric", {}).get("max_diff")
            rec["pnnx_op_types"] = _pnnx_types(op)
        except Exception as e:  # noqa: BLE001
            rec["status"] = "crash"; rec["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        rows.append(rec)
        (HERE / "rerun_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"   -> status={rec.get('status')} kernel={rec.get('kernel')} graph={rec.get('graph')} "
              f"e2e={rec.get('e2e')} pnnx={rec.get('pnnx_op_types')}", flush=True)

    print("\n=== RERUN DONE ===")
    rec_ok = [r for r in rows if r.get("status") == "success"]
    print(f"now success: {len(rec_ok)}/{len(rows)} -> {[r['op'] for r in rec_ok]}")


if __name__ == "__main__":
    main()
