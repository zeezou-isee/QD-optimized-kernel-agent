"""Resume: run the 3 ops that did not finish in the previous rerun
(OneHot, Det, Unique). Appends to rerun_results.json.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPGEN = HERE.parent / "opgen"
sys.path.insert(0, str(OPGEN.parent))  # for `import opgen`
sys.path.insert(0, str(OPGEN))         # for top-level config/llm_api
import opgen as _opgen; _opgen.bootstrap_paths()  # add subdirs to sys.path

from operator_agent import OperatorAgent

TORCH = OPGEN.parent / ".venv" / "lib" / "python3.12" / "site-packages" / "torch"
RUNS = OPGEN / "runs"

REMAINING = [
    ("OneHot", "kernel"),
    ("Det", "kernel"),
    ("Unique", "kernel"),
]
RESULTS = HERE / "rerun_results.json"


def _pnnx_types(op: str):
    p = RUNS / op / "graph" / "pnnx_ir_probe.json"
    return json.loads(p.read_text()).get("op_types") if p.exists() else None


def main() -> None:
    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    done = {r["op"] for r in rows}
    for op, prev in REMAINING:
        if op in done:
            print(f"[skip] {op} already in results"); continue
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
        RESULTS.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"   -> status={rec.get('status')} kernel={rec.get('kernel')} graph={rec.get('graph')} "
              f"e2e={rec.get('e2e')} pnnx={rec.get('pnnx_op_types')}", flush=True)

    print("\n=== RESUME DONE ===")
    for r in rows[-len(REMAINING):]:
        print(f"{r['op']:18s} status={r.get('status')} k={r.get('kernel')} g={r.get('graph')} e2e={r.get('e2e')}")


if __name__ == "__main__":
    main()
