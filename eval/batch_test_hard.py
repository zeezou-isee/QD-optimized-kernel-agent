"""Stress-test the GraphAgent on 10 HARD genuinely-unsupported ops and capture
WHY the graph phase fails, to extract general patterns.

Runs kernel(numeric) + graph(structural, forced target) — no libncnn rebuild,
so it's fast and focuses on graph convergence. For each op it records:
  - kernel status, graph status
  - graph failing stage + a trimmed diagnostic
  - the pnnx-level op_types the op decomposes to (what a pass must match)
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

from config import GraphConfig
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

CASES = [
    ("GatherElements", "动态索引 gather"),
    ("ScatterElements", "动态索引 scatter"),
    ("Range", "常量/arange"),
    ("OneHot", "索引→onehot,变形"),
    ("Det", "矩阵行列式"),
    ("MaxPool_2d_dilations", "池化变体(dilation)"),
    ("InstanceNormalization_1d", "归一化+权重+轴"),
    ("DepthToSpace", "像素重排/变形"),
    ("BitwiseAnd", "位运算"),
    ("Unique", "动态输出尺寸(极难)"),
]


def _graph_diag(op: str) -> dict:
    """Mine runs/<op>/graph/summary.json + probe for the failure signal."""
    out = {"graph_stage": "?", "signal": "", "pnnx_op_types": None}
    gp = RUNS / op / "graph" / "summary.json"
    if gp.exists():
        d = json.loads(gp.read_text())
        fr = d.get("final_result") or {}
        # which stage failed first
        for stage, flag in [("inject", fr.get("inject_ok")), ("build", fr.get("build_ok")),
                            ("convert", fr.get("convert_ok")), ("structural", fr.get("structural_ok"))]:
            if not flag:
                out["graph_stage"] = stage
                break
        else:
            out["graph_stage"] = "ok"
        sig = (fr.get("build_error") or "") + "\n" + (fr.get("convert_log") or "") + "\n" + (fr.get("structural_log") or "")
        out["signal"] = sig.strip()[-600:]
    pr = RUNS / op / "graph" / "pnnx_ir_probe.json"
    if pr.exists():
        out["pnnx_op_types"] = json.loads(pr.read_text()).get("op_types")
    return out


def run() -> None:
    rows = []
    for op, kind in CASES:
        print(f"\n################ {op} ({kind}) ################", flush=True)
        rec = {"op": op, "kind": kind}
        try:
            s = OperatorAgent(task_name=op, model="z-ai/glm-5.1", max_rounds=4,
                              torch_install_dir=TORCH, end_to_end=False, install=False).run()
            ph = s.get("phases", {})
            rec["status"] = s.get("status")
            rec["kernel"] = ph.get("kernel", {}).get("status")
            rec["graph"] = ph.get("graph", {}).get("status")
            rec.update(_graph_diag(op))
        except Exception as e:  # noqa: BLE001
            rec["status"] = "crash"; rec["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        rows.append(rec)
        (HERE / "batch_hard_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"   -> kernel={rec.get('kernel')} graph={rec.get('graph')} "
              f"stage={rec.get('graph_stage')} pnnx={rec.get('pnnx_op_types')}", flush=True)

    print("\n=== DONE ===")
    for r in rows:
        print(f"{r['op']:28s} kernel={r.get('kernel')} graph={r.get('graph')} "
              f"stage={r.get('graph_stage')} pnnx={r.get('pnnx_op_types')}")


if __name__ == "__main__":
    run()
