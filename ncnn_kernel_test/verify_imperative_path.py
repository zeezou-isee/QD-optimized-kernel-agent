"""Manually validate the imperative-style pass injection path.

Goal (no LLM): hand-write a minimal command-style pnnx pass + kernel for Trilu_lower,
plug it in like ncnn's own convert_torch_cat.cpp does:
  1) install kernel `cand_trilu.{h,cpp}` + ncnn_add_layer(Cand_Trilu)
  2) write `pass_ncnn/convert_cand_trilu.{h,cpp}` + add to src/CMakeLists.txt
  3) patch pass_ncnn.cpp: add #include + call site
  4) rebuild libncnn + pnnx
  5) run pnnx on Trilu_lower model, check .ncnn.param has Cand_Trilu (no aten residue)
  6) Net runner vs PyTorch
  7) ALWAYS restore the ncnn tree on exit
"""

from __future__ import annotations

import sys
import re
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OPGEN = HERE.parent / "opgen"
sys.path.insert(0, str(OPGEN.parent))  # for `import opgen`
sys.path.insert(0, str(OPGEN))         # for top-level config/llm_api
import opgen as _opgen; _opgen.bootstrap_paths()  # add subdirs to sys.path

from config import GraphConfig
from graph_pipeline import make_pt, run_conversion
from layer_oracle import NetOracle, parse_ncnn_io, torch_to_ncnn_input

NCNN = Path("/Users/xingze/Documents/project/kernelgen/ncnn")
DS = Path("/Users/xingze/Documents/project/kernelgen/MobileKernelBench_git/dataset/Mobilekernelbench")
TORCH = OPGEN.parent / ".venv" / "lib" / "python3.12" / "site-packages" / "torch"

# ---- the hand-written kernel: row-broadcasted lower-triangular mask multiplied in ----
CAND_TRILU_H = r"""
#ifndef CAND_TRILU_H
#define CAND_TRILU_H
#include "layer.h"
namespace ncnn {
class Cand_Trilu : public Layer {
public:
    Cand_Trilu();
    virtual int load_param(const ParamDict& pd);
    virtual int forward(const Mat& bottom, Mat& top, const Option& opt) const;
public:
    int diagonal;
    int upper;  // 0=lower, 1=upper
};
}
#endif
"""

CAND_TRILU_CPP = r"""
#include "cand_trilu.h"
namespace ncnn {
Cand_Trilu::Cand_Trilu(){ one_blob_only=true; support_inplace=false; }
int Cand_Trilu::load_param(const ParamDict& pd){
    diagonal = pd.get(0, 0);
    upper    = pd.get(1, 0);
    return 0;
}
int Cand_Trilu::forward(const Mat& bottom, Mat& top, const Option& opt) const {
    // 2D input only (w = cols, h = rows). Treat each row as a vector.
    int w = bottom.w, h = bottom.h;
    size_t es = bottom.elemsize;
    top.create(w, h, es, opt.blob_allocator);
    if (top.empty()) return -100;
    // pnnx exports int64 input, but ncnn substitutes int32; treat as 4-byte int.
    const int32_t* in = (const int32_t*)bottom.data;
    int32_t* out = (int32_t*)top.data;
    for (int y=0; y<h; ++y){
        for (int x=0; x<w; ++x){
            bool keep = upper ? (x >= y + diagonal) : (x <= y + diagonal);
            out[y*w + x] = keep ? in[y*w + x] : 0;
        }
    }
    return 0;
}
}
"""

# ---- imperative pass: walk Graph, find aten::tril, swap to Cand_Trilu ----
PASS_H = r"""
#include "pass_ncnn.h"
namespace pnnx { namespace ncnn {
void convert_cand_trilu(Graph& graph);
} }
"""

PASS_CPP = r"""
#include "convert_cand_trilu.h"
namespace pnnx { namespace ncnn {
void convert_cand_trilu(Graph& graph){
    int idx = 0;
    while (true) {
        bool matched = false;
        for (size_t i = 0; i < graph.ops.size(); ++i) {
            Operator* op = graph.ops[i];
            if (op->type != "aten::tril") continue;
            matched = true;
            op->type = "Cand_Trilu";
            op->name = std::string("trilu_") + std::to_string(idx++);
            int diag = 0;
            Operator* d = nullptr;
            Operand* dop = nullptr;
            if (op->inputs.size() >= 2) {
                dop = op->inputs[1];
                d = dop->producer;
                if (d && d->type == "pnnx.Expression" && d->has_param("expr")) {
                    const std::string& e = d->params.at("expr").s;
                    diag = atoi(e.c_str());
                }
            }
            op->params["0"] = diag;
            op->params["1"] = 0; // lower
            // detach the diagonal scalar input and drop the orphan expr op + operand
            if (op->inputs.size() >= 2) op->inputs.resize(1);
            if (dop) {
                for (auto it = dop->consumers.begin(); it != dop->consumers.end(); ) {
                    if (*it == op) it = dop->consumers.erase(it); else ++it;
                }
                if (d && dop->consumers.empty()) {
                    auto oit = std::find(graph.operands.begin(), graph.operands.end(), dop);
                    if (oit != graph.operands.end()) graph.operands.erase(oit);
                    delete dop;
                    auto dit = std::find(graph.ops.begin(), graph.ops.end(), d);
                    if (dit != graph.ops.end()) graph.ops.erase(dit);
                    delete d;
                }
            }
            break; // restart loop after mutation
        }
        if (!matched) break;
    }
}
} }
"""


def install():
    """Install all files + patch pass_ncnn.cpp + rebuild."""
    pn = NCNN / "tools" / "pnnx" / "src" / "pass_ncnn"
    (NCNN / "src" / "layer" / "cand_trilu.h").write_text(CAND_TRILU_H, encoding="utf-8")
    (NCNN / "src" / "layer" / "cand_trilu.cpp").write_text(CAND_TRILU_CPP, encoding="utf-8")
    (pn / "convert_cand_trilu.h").write_text(PASS_H, encoding="utf-8")
    (pn / "convert_cand_trilu.cpp").write_text(PASS_CPP, encoding="utf-8")

    # 1) ncnn src/CMakeLists.txt: ncnn_add_layer(Cand_Trilu)
    cm = NCNN / "src" / "CMakeLists.txt"
    t = cm.read_text(encoding="utf-8")
    if "ncnn_add_layer(Cand_Trilu)" not in t:
        i = t.find("ncnn_add_layer(")
        cm.write_text(t[:i] + "ncnn_add_layer(Cand_Trilu)\n" + t[i:], encoding="utf-8")

    # 2) pnnx src CMakeLists: add convert_cand_trilu.cpp to pnnx_pass_ncnn_SRCS
    cm2 = NCNN / "tools" / "pnnx" / "src" / "CMakeLists.txt"
    t = cm2.read_text(encoding="utf-8")
    if "pass_ncnn/convert_cand_trilu.cpp" not in t:
        marker = "set(pnnx_pass_ncnn_SRCS"
        i = t.find(marker)
        i = t.find("\n", i) + 1
        cm2.write_text(t[:i] + "    pass_ncnn/convert_cand_trilu.cpp\n" + t[i:], encoding="utf-8")

    # 3) pass_ncnn.cpp: add include + call site
    pnnx_cpp = NCNN / "tools" / "pnnx" / "src" / "pass_ncnn.cpp"
    t = pnnx_cpp.read_text(encoding="utf-8")
    if "convert_cand_trilu.h" not in t:
        inc = '#include "pass_ncnn/convert_cand_trilu.h"\n'
        anchor = '#include "pass_ncnn/convert_Tensor_slice_copy.h"\n'
        t = t.replace(anchor, anchor + inc)
        # call right after the other convert_Tensor_* block
        call_anchor = "    ncnn::convert_Tensor_slice_copy(g);\n"
        t = t.replace(call_anchor, call_anchor + "\n    ncnn::convert_cand_trilu(g);\n")
        pnnx_cpp.write_text(t, encoding="utf-8")


def restore():
    NCNN_SRC = NCNN / "src"
    for f in (NCNN_SRC / "layer" / "cand_trilu.h",
              NCNN_SRC / "layer" / "cand_trilu.cpp",
              NCNN / "tools" / "pnnx" / "src" / "pass_ncnn" / "convert_cand_trilu.h",
              NCNN / "tools" / "pnnx" / "src" / "pass_ncnn" / "convert_cand_trilu.cpp"):
        f.unlink(missing_ok=True)
    # revert all tracked changes
    subprocess.run(["git", "-C", str(NCNN), "checkout", "--", "src/CMakeLists.txt",
                    "tools/pnnx/src/CMakeLists.txt", "tools/pnnx/src/pass_ncnn.cpp"], check=False)


def rebuild():
    subprocess.run(["cmake", "--build", str(NCNN / "build_lib"), "-j8"], check=True)
    subprocess.run(["cmake", "--build", str(NCNN / "tools" / "pnnx" / "build"), "-j8"], check=True)


def verify():
    cfg = GraphConfig(torch_install_dir=TORCH)
    rd = HERE / "_imperative_work"
    if rd.exists():
        import shutil; shutil.rmtree(rd)
    rd.mkdir(parents=True)
    mp = next(DS.rglob("Trilu_lower.py"))

    ok, pt, ish, log = make_pt(cfg, mp, rd)
    if not ok:
        return False, "trace failed"
    cok, art, _ = run_conversion(cfg, pt, ish, rd, "Trilu_lower")
    if not cok:
        return False, "pnnx no .ncnn.param"
    param_txt = Path(art[".ncnn.param"]).read_text()
    print("=== .ncnn.param ===\n" + param_txt)
    if "Cand_Trilu" not in param_txt:
        return False, "Cand_Trilu missing in .ncnn.param"
    if "aten::" in param_txt:
        return False, "aten:: residue in .ncnn.param"

    # numeric: ncnn vs PyTorch
    import importlib.util, torch
    spec = importlib.util.spec_from_file_location("m", str(mp))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    init = m.get_init_inputs() if hasattr(m, "get_init_inputs") else []
    model = (m.Model(*init) if init else m.Model()).eval()
    inputs = m.get_inputs()
    with torch.no_grad():
        ref = model(*inputs)
    ref_np = ref.detach().numpy()
    reference = ref_np[0] if ref_np.ndim >= 2 else ref_np
    ncnn_inputs = [torch_to_ncnn_input(t.detach().numpy().astype(np.int32)) for t in inputs]
    in_names, out_name = parse_ncnn_io(param_txt)
    feed = {n: x for n, x in zip(in_names, ncnn_inputs)}
    netoc = NetOracle(ncnn_root=NCNN, workdir=rd / "_net")
    out, runlog = netoc.run_net(art[".ncnn.param"], art[".ncnn.bin"], feed, out_name)
    if out is None:
        return False, "net run failed\n" + runlog[-400:]
    try:
        out_r = out.reshape(reference.shape).astype(np.int64)
    except ValueError:
        return False, f"shape mismatch {out.shape} vs {reference.shape}"
    diff = np.abs(out_r - reference)
    passed = bool(np.allclose(out_r, reference, atol=0, rtol=0))
    return passed, f"max_diff={int(diff.max())} mean={float(diff.mean()):.3f} out_shape={out_r.shape}"


def main():
    try:
        restore()  # clean baseline
        print("[1] install imperative pass + kernel"); install()
        print("[2] rebuild libncnn + pnnx"); rebuild()
        print("[3] verify"); ok, info = verify()
        print(f"\n========== IMPERATIVE PATH VERDICT: {'✅ PASS' if ok else '❌ FAIL'} ==========")
        print(info)
    finally:
        print("\n[cleanup] restore ncnn tree")
        restore()
        try:
            rebuild()
        except Exception:
            pass


if __name__ == "__main__":
    main()
