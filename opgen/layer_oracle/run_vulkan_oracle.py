"""Self-check for the vulkan layer oracle (方案A, vulkan).

Verifies the hand-written sample `Cand_AbsVal_vulkan` (samples/) against a numpy
reference (np.abs — no torch needed). Two gates:

  1. COMPILE gate: build the vulkan runner + candidate against build_lib_vk via
     find_package(ncnn). Proves the vulkan compile/link path works (doable on any
     arm64 mac with simplevk; no Vulkan SDK required).
  2. RUN gate: actually execute on the GPU. Requires a Vulkan device (MoltenVK).
     Without one the runner exits 42 and we report SKIPPED (not a failure).

Usage:
    python opgen/layer_oracle/run_vulkan_oracle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent          # .../opgen/layer_oracle
sys.path.insert(0, str(_THIS.parent))            # put opgen/ on path -> `import layer_oracle`

from layer_oracle.vulkan_oracle import VulkanLayerOracle  # noqa: E402


def main() -> int:
    samples = _THIS / "samples"
    candidate_cpp = samples / "cand_absval_vulkan.cpp"
    header = "cand_absval_vulkan.h"
    shader = samples / "cand_absval.comp"
    class_name = "Cand_AbsVal_vulkan"

    oc = VulkanLayerOracle()

    # ---- compile gate ----
    print("[1/2] compile gate: building vulkan runner + candidate ...")
    try:
        runner, log = oc.compile(candidate_cpp=candidate_cpp, class_name=class_name,
                                 header=header, shader=shader)
    except Exception as exc:  # noqa: BLE001
        print("COMPILE FAILED:\n", exc)
        return 1
    print(f"  COMPILE OK -> {runner}")

    # ---- run gate ----
    print("[2/2] run gate: executing on GPU (skipped if no vulkan device) ...")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 16)).astype(np.float32)   # 2D, c=1, contiguous
    ref = np.abs(x)

    verdict = oc.verify(candidate_cpp=candidate_cpp, class_name=class_name, header=header,
                        shader=shader, params={}, inputs=[x], reference=ref, tol=1e-4)

    if verdict.skipped:
        print("  SKIPPED: no vulkan device (install MoltenVK to run the numeric check)")
        print("  (compile gate already proves the vulkan harness builds & links)")
        return 0
    if verdict.passed:
        print(f"  PASS  {verdict.detail}")
        return 0
    print(f"  FAIL  {verdict.detail}")
    print(verdict.run_log[-1500:])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
