"""Self-check for 方案C native-override install/restore safety (no LLM, no ncnn build).

Two layers:
  1. Pure-helper checks (always run, even without numpy): rewrite_class_name,
     detect_native_layer, _infer_class_name — extracted from net_oracle.py source.
  2. Filesystem round-trip (when numpy importable): build a fake ncnn tree, run
     NetOracle.install_native_override + restore_native_override, and assert the
     tree is byte-for-byte identical to the pristine state afterwards.

Run: python eval/test_native_override.py
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = HERE.parent / "agents"
sys.path.insert(0, str(AGENTS.parent))  # repo root, for `import agents`
sys.path.insert(0, str(AGENTS))
import agents as _agents; _agents.bootstrap_paths()

NET_ORACLE_SRC = (AGENTS / "layer_oracle" / "net_oracle.py").read_text(encoding="utf-8")


def _load_pure_funcs() -> dict:
    """Exec just the pure helper functions out of net_oracle.py (avoids the
    module-level numpy import so this runs in a bare environment)."""
    ns = {"re": re, "Path": Path, "_NON_COMPUTE_LAYERS": {"Input", "Output", "Split", "Noop"}}
    for fn in ("_infer_class_name", "rewrite_class_name", "detect_native_layer"):
        m = re.search(rf"(?ms)^def {fn}\(.*?(?=\n\ndef |\n\n\n)", NET_ORACLE_SRC)
        assert m, f"could not extract {fn}"
        exec(m.group(0), ns)
    return ns


def check_pure_helpers() -> None:
    ns = _load_pure_funcs()
    code_h = ("#ifndef CAND_SOFTMAX_H\n#define CAND_SOFTMAX_H\n#include \"layer.h\"\n"
              "class Cand_Softmax : public Layer { int load_param(const ParamDict&); };\n"
              "#endif // CAND_SOFTMAX_H\n")
    code_cpp = ("#include \"cand_softmax.h\"\nnamespace ncnn {\n"
                "Cand_Softmax::Cand_Softmax() {}\nDEFINE_LAYER_CREATOR(Cand_Softmax)\n}\n")

    assert ns["_infer_class_name"]({"cand_softmax.h": code_h}) == "Cand_Softmax"

    rh = ns["rewrite_class_name"](code_h, "Cand_Softmax", "Softmax")
    assert "class Softmax :" in rh and "Cand_Softmax" not in rh, rh
    assert "SOFTMAX_H" in rh and "CAND_SOFTMAX_H" not in rh, rh
    rc = ns["rewrite_class_name"](code_cpp, "Cand_Softmax", "Softmax")
    assert '#include "softmax.h"' in rc and "Cand_Softmax" not in rc, rc

    # detect: single native layer with a real src/layer/<x>.cpp -> class; else None
    ncnn_root = AGENTS.parent / "frameworks" / "ncnn"
    single = "x\nx\nInput in 0 1 in0\nSoftmax s 1 1 in0 out0 0=1"
    multi = "x\nx\nInput in 0 1 in0\nSoftmax s 1 1 in0 1 0=1\nUnaryOp u 1 1 1 out0 0=8"
    if (ncnn_root / "src" / "layer" / "softmax.cpp").exists():
        assert ns["detect_native_layer"](single, ncnn_root) == "Softmax"
    assert ns["detect_native_layer"](multi, ncnn_root) is None
    fake = "x\nx\nInput in 0 1 in0\nFrobnicate f 1 1 in0 out0"
    assert ns["detect_native_layer"](fake, ncnn_root) is None
    print("[ok] pure helpers: rewrite_class_name / detect_native_layer / _infer_class_name")


def _snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text(encoding="utf-8")
            for p in sorted(root.rglob("*")) if p.is_file()}


def check_install_restore_roundtrip() -> None:
    try:
        import numpy  # noqa: F401
    except Exception:
        print("[skip] install/restore round-trip needs numpy (not installed); "
              "pure-helper checks already cover the rename/detect logic")
        return
    from layer_oracle import NetOracle, NativeOverrideHandle  # noqa: F401

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ncnn"
        layer = root / "src" / "layer"
        (layer / "x86").mkdir(parents=True)
        # pristine native tree
        (layer / "softmax.h").write_text("// native softmax header\nclass Softmax {};\n", encoding="utf-8")
        (layer / "softmax.cpp").write_text("// native softmax impl\n", encoding="utf-8")
        (layer / "x86" / "softmax_x86.h").write_text("// native x86 header\n", encoding="utf-8")
        (layer / "x86" / "softmax_x86.cpp").write_text("// native x86 impl\n", encoding="utf-8")
        (layer / "softmax_int8.cpp").write_text("// unrelated, must NOT be touched\n", encoding="utf-8")
        before = _snapshot(root)

        oc = NetOracle(ncnn_root=root, build_lib=root / "build")  # build dir unused here
        agent_code = {
            "cand_softmax.h": ("#ifndef CAND_SOFTMAX_H\n#define CAND_SOFTMAX_H\n"
                               "class Cand_Softmax : public Layer {};\n#endif\n"),
            "cand_softmax.cpp": ("#include \"cand_softmax.h\"\n"
                                 "DEFINE_LAYER_CREATOR(Cand_Softmax)\n"),
        }
        h = oc.install_native_override(agent_code, "Softmax", cand_class="Cand_Softmax")

        # base files overwritten with renamed agent code
        assert "Softmax" in (layer / "softmax.h").read_text() and \
               "Cand_Softmax" not in (layer / "softmax.h").read_text()
        # DEFINE_LAYER_CREATOR stripped from the overwritten .cpp
        assert "DEFINE_LAYER_CREATOR" not in (layer / "softmax.cpp").read_text()
        # x86 arch variant parked aside; base unrelated file untouched
        assert not (layer / "x86" / "softmax_x86.cpp").exists()
        assert (layer / "x86" / "softmax_x86.cpp.ka_parked").exists()
        assert (layer / "softmax_int8.cpp").read_text() == "// unrelated, must NOT be touched\n"
        assert len(h.parked_arch) == 2  # _x86.h and _x86.cpp

        errs = oc.restore_native_override(h)
        assert not errs, errs
        after = _snapshot(root)
        assert after == before, ("tree not restored:\n"
                                 f"  only-before={set(before)-set(after)}\n"
                                 f"  only-after={set(after)-set(before)}\n"
                                 f"  changed={[k for k in before if k in after and before[k]!=after[k]]}")
        print("[ok] install_native_override + restore_native_override: tree fully restored")


if __name__ == "__main__":
    check_pure_helpers()
    check_install_restore_roundtrip()
    print("\nALL NATIVE-OVERRIDE SELF-CHECKS PASSED")
