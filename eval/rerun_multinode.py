"""After P1+P2+P3 (imperative-pass support), re-run the 3 true multi-node ops."""

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

CASES = [("Cast", "B2 fold + still multi-node"),
         ("Trilu_lower", "Expression+aten::tril"),
         ("ScatterElements", "Expression+aten::scatter")]


def main() -> None:
    rows = []
    for op, kind in CASES:
        print(f"\n################ {op} ({kind}) ################", flush=True)
        rec = {"op": op, "kind": kind}
        try:
            s = OperatorAgent(task_name=op, model="z-ai/glm-5.1", max_rounds=6,
                              torch_install_dir=TORCH, end_to_end=True, install=False).run()
            ph = s.get("phases", {})
            rec["status"] = s.get("status")
            rec["kernel"] = ph.get("kernel", {}).get("status")
            rec["graph"] = ph.get("graph", {}).get("status")
            rec["e2e"] = ph.get("end_to_end_numeric", {}).get("passed")
            rec["e2e_diff"] = ph.get("end_to_end_numeric", {}).get("max_diff")
        except Exception as e:
            rec["status"] = "crash"; rec["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        rows.append(rec)
        (HERE / "rerun_multinode_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"   -> {rec.get('status')} k={rec.get('kernel')} g={rec.get('graph')} e2e={rec.get('e2e')}", flush=True)
    print("\nDONE")


if __name__ == "__main__":
    main()
