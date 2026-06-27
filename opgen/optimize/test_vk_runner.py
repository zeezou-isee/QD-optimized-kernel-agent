"""Minimal experiment for #3 (vulkan optimize backend): can the optimizer's vulkan
runner COMPILE + RUN + MEASURE a vulkan kernel on the GPU?

Uses the hand-written sample Cand_AbsVal_vulkan (layer_oracle/samples/) directly via
VkRunner — NO LLM, NO torch. Inputs are numpy; reference is np.abs. Proves the
optimizer's measurement path (compile_only -> run_once xN -> read_output) works on
a real Vulkan device (MoltenVK); SKIPs cleanly when none is present.

Run:  python opgen/optimize/test_vk_runner.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))                 # opgen/  -> layer_oracle, config
sys.path.insert(0, str(_THIS))                        # opgen/optimize/ -> schemas, evaluator, ...

from layer_oracle import VulkanLayerOracle             # noqa: E402
from evaluator import VkRunner                         # noqa: E402

_SAMPLES = _THIS.parent / "layer_oracle" / "samples"


def main() -> int:
    runner = VkRunner(VulkanLayerOracle())
    cpp = _SAMPLES / "cand_absval_vulkan.cpp"
    shader = _SAMPLES / "cand_absval.comp"
    x = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)  # 2D, c=1

    print("[1/2] compile vulkan candidate via VkRunner ...")
    try:
        art, _clog = runner.compile_only(
            candidate_cpp=cpp, class_name="Cand_AbsVal_vulkan",
            header="cand_absval_vulkan.h", inputs=[x], shader=shader)
    except Exception as exc:  # noqa: BLE001
        print("COMPILE FAILED:\n", exc)
        return 1
    print(f"  COMPILE OK -> {art.runner_path}")

    print("[2/2] run xN on GPU + check vs np.abs (skips if no device) ...")
    lat = []
    for _ in range(5):
        ok, ms, err = runner.run_once(art)
        if not ok and "no vulkan device" in err:
            print("  SKIPPED: no vulkan device (install MoltenVK to measure)")
            print("  (compile gate already proves the optimizer's vulkan runner builds)")
            return 0
        if not ok:
            print("  RUN FAILED:", err)
            return 2
        lat.append(ms)
    out = runner.read_output(art)
    if not np.allclose(out.reshape(x.shape), np.abs(x), atol=1e-4):
        print("  NUMERIC FAIL: max_diff =", float(np.abs(out.reshape(x.shape) - np.abs(x)).max()))
        return 3
    print(f"  PASS  correct vs np.abs; latency(min/median over {len(lat)})="
          f"{min(lat):.3f}/{sorted(lat)[len(lat)//2]:.3f} ms (wall-clock incl. process start)")
    print("  -> optimizer can compile + run + measure a vulkan kernel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
