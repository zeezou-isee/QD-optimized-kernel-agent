"""Run the OperatorAgent end-to-end on 10 diverse genuinely-unsupported ops.

Temporary (install=False): each op is kernel-numeric verified + conversion forced
to the new layer + end-to-end numeric (Net runner), then the ncnn tree is restored.
Writes BATCH_REPORT.md with a per-op verdict.
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

CASES = [
    ("Equal", "二元比较"),
    ("And", "二元逻辑"),
    ("Not", "一元逻辑"),
    ("Sinh", "一元数学(表达式)"),
    ("Where", "三输入选择"),
    ("Cast", "类型转换"),
    ("Mod", "二元取模"),
    ("CumSum", "带轴扫描"),
    ("Trilu_lower", "三角掩码"),
    ("TopK", "多输出(难)"),
]


def run() -> None:
    rows = []
    for op, kind in CASES:
        print(f"\n################ {op} ({kind}) ################", flush=True)
        try:
            s = OperatorAgent(task_name=op, model="z-ai/glm-5.1", max_rounds=5,
                              torch_install_dir=TORCH, end_to_end=True, install=False).run()
            ph = s.get("phases", {})
            rows.append({
                "op": op, "kind": kind, "status": s.get("status"),
                "kernel": ph.get("kernel", {}).get("status"),
                "kernel_diff": ph.get("kernel", {}).get("max_diff"),
                "graph": ph.get("graph", {}).get("status"),
                "e2e": ph.get("end_to_end_numeric", {}).get("passed"),
                "e2e_diff": ph.get("end_to_end_numeric", {}).get("max_diff"),
            })
        except Exception as e:  # noqa: BLE001
            rows.append({"op": op, "kind": kind, "status": "crash", "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()
        (HERE / "batch_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    # report
    lines = ["# 10 类未支持算子的 agent 端到端测试", "",
             "对每个真未支持算子跑:kernel 数值(vs PyTorch) + 计算图转换(强制目标新层) + "
             "端到端数值(Net 跑转换模型 vs PyTorch)。临时验证,跑完还原源码树。", "",
             "| 算子 | 种类 | kernel | graph | 端到端数值 | 总判定 |",
             "|---|---|---|---|---|---|"]
    ok = 0
    for r in rows:
        if r.get("status") == "success":
            ok += 1
        e2e = r.get("e2e")
        e2e_s = "✅" if e2e else ("❌" if e2e is False else "—")
        lines.append(f"| {r['op']} | {r['kind']} | {r.get('kernel','-')} | {r.get('graph','-')} | "
                     f"{e2e_s} {r.get('e2e_diff','') if e2e else ''} | "
                     f"{'✅ '+ r['status'] if r.get('status')=='success' else '❌ '+str(r.get('status'))} |")
    lines.append("")
    lines.append(f"**通过(完整端到端成功):{ok}/{len(rows)}**")
    (HERE / "BATCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
