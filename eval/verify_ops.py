"""Independent verification that the registered operators now work in ncnn.

For each op (no agent involved):
  1. trace the dataset PyTorch model -> .pt
  2. run the NATIVE (rebuilt) pnnx -> .ncnn.param/.bin
  3. confirm pnnx natively converts it to the new ncnn layer (no torch/aten residue)
  4. run the converted model via ncnn (libncnn now has the kernel) and allclose vs PyTorch
Writes REPORT.md.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AGENTS = HERE.parent / "agents"
sys.path.insert(0, str(AGENTS.parent))  # repo root, for `import agents`
sys.path.insert(0, str(AGENTS))         # for top-level config/llm_api
import agents as _agents; _agents.bootstrap_paths()  # add subdirs to sys.path

from config import GraphConfig
from graph_pipeline import make_pt, parse_pnnx_op_types, run_conversion
from layer_oracle import NetOracle, parse_ncnn_io, torch_to_ncnn_input

# op -> (expected new ncnn layer type, dataset file stem)
OPS = {
    "Greater": "Cand_Greater",
    "LessEqual": "Cand_LessEqual",
}


def _load_model(model_py: Path):
    spec = importlib.util.spec_from_file_location("ds_model", str(model_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    model = (mod.Model(*init) if init else mod.Model()).eval()
    return mod, model


def verify_one(cfg: GraphConfig, netoc: NetOracle, task: str, expect_layer: str) -> dict:
    import torch
    matches = sorted(Path(cfg.dataset_root).rglob(f"{task}.py"))
    if not matches:
        return {"op": task, "error": "dataset model not found"}
    model_py = matches[0]
    rd = HERE / "_work" / task
    rd.mkdir(parents=True, exist_ok=True)

    # 1) trace
    ok, pt, ishape, log = make_pt(cfg, model_py, rd)
    if not ok:
        return {"op": task, "convert": False, "error": "trace failed"}

    # 2) native pnnx conversion (no agent)
    cok, art, clog = run_conversion(cfg, pt, ishape, rd, task)
    if not cok:
        return {"op": task, "convert": False, "error": "pnnx produced no .ncnn.param"}
    param_txt = Path(art[".ncnn.param"]).read_text()
    ncnn_types = {ln.split()[0] for ln in param_txt.splitlines()[2:] if ln.split()}

    # 3) native support check — the FINAL .ncnn.param must contain the target
    # layer and NO unconverted op. (.pnnx.param always keeps the torch-domain op;
    # the torch->ncnn rewrite lives in the .ncnn graph, so check there.)
    residue = [t for t in ncnn_types if t.startswith(("aten::", "prim::", "torch."))]
    supported = (expect_layer in ncnn_types) and not residue

    # 4) numeric: ncnn run vs PyTorch
    mod, model = _load_model(model_py)
    inputs = mod.get_inputs()
    with torch.no_grad():
        ref = model(*inputs)
    if isinstance(ref, (tuple, list)):
        ref = ref[0]
    ref_np = ref.detach().numpy()
    reference = ref_np[0] if ref_np.ndim >= 2 else ref_np
    ncnn_inputs = [torch_to_ncnn_input(t.detach().numpy()) for t in inputs]
    in_names, out_name = parse_ncnn_io(param_txt)
    if len(in_names) != len(ncnn_inputs):
        in_names = [f"in{i}" for i in range(len(ncnn_inputs))]
    feed = {n: x for n, x in zip(in_names, ncnn_inputs)}
    out, runlog = netoc.run_net(art[".ncnn.param"], art[".ncnn.bin"], feed, out_name)
    if out is None:
        return {"op": task, "convert": True, "supported": supported, "numeric": False,
                "ncnn_layer_line": _layer_line(param_txt, expect_layer), "error": "net run failed"}
    out_r = out.reshape(reference.shape)
    diff = np.abs(out_r - np.asarray(reference, dtype=np.float32))
    passed = bool(np.allclose(out_r, reference, atol=2e-3, rtol=2e-3))
    return {
        "op": task, "convert": True, "supported": supported, "numeric": passed,
        "max_diff": float(diff.max()), "expect_layer": expect_layer,
        "ncnn_layer_line": _layer_line(param_txt, expect_layer),
        "residue": residue, "in_names": in_names, "out_name": out_name,
    }


def _layer_line(param_txt: str, layer: str) -> str:
    for ln in param_txt.splitlines():
        if ln.split() and ln.split()[0] == layer:
            return ln.strip()
    return "(target layer not found)"


def _torch_dir():
    """Installed torch's dir (so pnnx links the right libtorch); None lets pnnx auto-probe."""
    try:
        import torch, os
        return Path(os.path.dirname(torch.__file__))
    except Exception:
        return None


def main() -> None:
    cfg = GraphConfig(torch_install_dir=_torch_dir())
    netoc = NetOracle(ncnn_root=cfg.ncnn_root, workdir=HERE / "_net")
    results = [verify_one(cfg, netoc, t, lyr) for t, lyr in OPS.items()]
    (HERE / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    lines = ["# 注册算子的端到端验证(原生 pnnx + ncnn,无 agent)", ""]
    lines.append("数据来源:`MobileKernelBench_git/dataset/Mobilekernelbench`;"
                 "流程:PyTorch 模型 → 原生 `pnnx` 转换 → ncnn 运行 → 对 PyTorch `allclose(2e-3)`。")
    lines.append("")
    lines.append("| 算子 | pnnx 转换 | 转成的 ncnn 层 | 原生支持(无残留) | 数值正确 | max_diff |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        line = r.get("ncnn_layer_line", "")
        lines.append(f"| {r['op']} | {'✅' if r.get('convert') else '❌'} | `{line}` | "
                     f"{'✅' if r.get('supported') else '❌'} | {'✅' if r.get('numeric') else '❌'} | "
                     f"{r.get('max_diff', r.get('error',''))} |")
    lines.append("")
    allok = all(r.get("convert") and r.get("supported") and r.get("numeric") for r in results)
    lines.append(f"**总判定:{'全部通过 ✅' if allok else '存在未通过项 ❌'}**")
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
