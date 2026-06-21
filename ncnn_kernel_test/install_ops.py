"""Permanently register agent-generated operators into ncnn + pnnx.

For each op it installs the VERIFIED artifacts (from opgen/runs/<task>/...):
  - kernel  -> ncnn/src/layer/<cand>.{h,cpp} + ncnn_add_layer(<Class>)   (rebuild libncnn)
  - pnnx pass -> ncnn/tools/pnnx/src/pass_ncnn|pass_level2/...  + CMake    (rebuild pnnx)

After this, native `pnnx model.pt` converts the op automatically and ncnn can run it.
This does NOT restore (the whole point is to register permanently).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OPGEN = Path(__file__).resolve().parent.parent / "opgen"
sys.path.insert(0, str(OPGEN.parent))  # for `import opgen`
sys.path.insert(0, str(OPGEN))         # for top-level config/llm_api
import opgen as _opgen; _opgen.bootstrap_paths()  # add subdirs to sys.path

from config import GraphConfig
from graph_pipeline import build_pnnx, inject_files
from graph_schemas import BackupHandle
from layer_oracle import NetOracle

RUNS = OPGEN / "runs"
OPS = ["Greater", "LessEqual"]
TORCH = OPGEN.parent / ".venv" / "lib" / "python3.12" / "site-packages" / "torch"


def _final_code(summary_path: Path) -> tuple[dict, dict]:
    d = json.loads(summary_path.read_text())
    fr = d.get("final_result") or {}
    return fr.get("response_code") or {}, d


def main() -> None:
    cfg = GraphConfig(torch_install_dir=TORCH)
    netoc = NetOracle(ncnn_root=cfg.ncnn_root, workdir=RUNS / "_net")
    session = BackupHandle()  # accumulates pnnx-tree edits (kept, not restored)

    for op in OPS:
        print(f"\n===== installing {op} =====")
        kcode, ksum = _final_code(RUNS / op / "kernel" / "summary.json")
        cls = (ksum.get("kernel_profile") or {}).get("class_name", "")
        gcode, _ = _final_code(RUNS / op / "graph" / "summary.json")
        pass_code = {k: v for k, v in gcode.items() if k.split("/")[0] in ("pass_ncnn", "pass_level1", "pass_level2")}

        # 1) kernel -> src/layer + ncnn_add_layer
        netoc.install_layer(kcode, cls)
        print(f"  kernel installed: {list(kcode)} -> ncnn_add_layer({cls})")
        # 2) pnnx pass -> pass_ncnn / pass_level2 + CMake
        ok, _, err = inject_files(cfg, pass_code, session)
        print(f"  pass installed: {list(pass_code)} ok={ok} {err}")

    print("\n===== rebuild libncnn =====")
    ok, log = netoc.rebuild_libncnn()
    (Path(__file__).resolve().parent / "libncnn_build.log").write_text(log, encoding="utf-8")
    print(f"  libncnn rebuilt: {ok}")

    print("===== rebuild pnnx =====")
    bok, blog = build_pnnx(cfg, Path(__file__).resolve().parent / "pnnx_build.log")
    print(f"  pnnx rebuilt: {bok}")

    print("\n[install] DONE. Registered:", OPS)


if __name__ == "__main__":
    main()
