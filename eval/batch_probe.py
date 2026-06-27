"""Cheap coverage classification of the 'unsupported' dataset ops (no LLM).

For each op, run native pnnx (current build) on its dataset model and check:
  - baseline_supported: pnnx converts it AND ncnn output matches PyTorch
  - else: which torch/aten op-types remain (what the agent would need to handle)

Splits the list into ALREADY-supported (false negatives) vs GENUINELY-unsupported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = HERE.parent / "agents"
sys.path.insert(0, str(AGENTS.parent))  # repo root, for `import agents`
sys.path.insert(0, str(AGENTS))         # for top-level config/llm_api
import agents as _agents; _agents.bootstrap_paths()  # add subdirs to sys.path

from config import GraphConfig
from graph_pipeline import probe_pnnx_ir

def _torch_dir():
    """Installed torch's dir (so pnnx links the right libtorch); None lets pnnx auto-probe."""
    try:
        import torch, os
        return Path(os.path.dirname(torch.__file__))
    except Exception:
        return None


TORCH = _torch_dir()


def main() -> None:
    cfg = GraphConfig(torch_install_dir=TORCH)
    ops = [l.strip() for l in Path("/tmp/unsupported_ops.txt").read_text().splitlines() if l.strip()]
    rows = []
    for i, op in enumerate(ops):
        matches = sorted(Path(cfg.dataset_root).rglob(f"{op}.py"))
        if not matches:
            rows.append({"op": op, "status": "no_model"}); continue
        try:
            g = probe_pnnx_ir(cfg, matches[0], HERE / "_probe" / op, op)
            rows.append({
                "op": op,
                "supported": bool(g.get("baseline_supported")),
                "numeric": g.get("baseline_numeric_ok"),
                "op_types": g.get("op_types"),
                "residual": g.get("residual_aten"),
            })
        except Exception as e:  # noqa: BLE001
            rows.append({"op": op, "status": f"probe_error: {type(e).__name__}"})
        print(f"[{i+1}/{len(ops)}] {op}: supported={rows[-1].get('supported')} "
              f"types={rows[-1].get('op_types')}", flush=True)

    (HERE / "probe_classify.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    sup = [r for r in rows if r.get("supported")]
    uns = [r for r in rows if r.get("supported") is False]
    err = [r for r in rows if "status" in r]
    print(f"\n=== SUMMARY: {len(sup)} already-supported, {len(uns)} genuinely-unsupported, {len(err)} error/no-model ===")
    print("already-supported:", [r["op"] for r in sup])
    print("genuinely-unsupported:", [r["op"] for r in uns])
    print("error/no-model:", [(r['op'], r.get('status')) for r in err])


if __name__ == "__main__":
    main()
