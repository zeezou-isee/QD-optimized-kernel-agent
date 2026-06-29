"""Tier 2 — Vulkan hardware profile (microbench side, SPIKE skeleton).

What it does today (skeleton / spike):
  1. Load the device profile written by vulkan_query.py.
  2. Run a handful of placeholder microbenches that record their PLAN, not
     actual measurements. Each entry carries `status: "stub"` and the exact
     shader + dispatch shape it would run, so the next iteration can fill
     them in without re-deciding the schema.
  3. Append the results under `behavior_profile` of the same profile JSON.

Why a skeleton now:
  Real measurement needs either (a) a vulkan-enabled libncnn (the project's
  default build is NCNN_VULKAN=OFF) or (b) a Python ctypes binding to
  libvulkan + glslang. Both are bigger commits than this spike intends.
  Locking the SCHEMA and the call site here lets us swap in real numbers
  later without changing the consumer (the LLM proposer).

Initial microbench plan (defined in BENCHES, not yet measured):
  - wg_size_sweep:     occupancy-vs-workgroup-size on a trivial fp32 copy
                       (find the workgroup-size cliff for this device)
  - fp16_vs_fp32:      ratio of fp16 to fp32 throughput on a memory-bound copy
                       (only meaningful if features.shader_float16 = true)
  - shared_mem_bw:     bandwidth of shared-memory reads vs global reads
                       (locate the L1/SMEM crossover)

Run:
  python3 opgen/profile/vulkan_microbench.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

PROJ = Path(__file__).resolve().parents[2]
PROFILE_DIR = PROJ / "experience_pool" / "backend_vulkan" / "tier2_hardware_profiles"


# ---------------------------------------------------------------------------
@dataclass
class BenchPlan:
    """A microbench's intended measurement — what shader, what sweep axis,
    what each datapoint reports. Stays valid even before real numbers exist."""
    name: str
    purpose: str                     # what decision-table cell this informs
    shader_glsl_sketch: str          # short pseudo-GLSL of the kernel under test
    sweep_axis: str                  # which dispatch param varies
    sweep_values: list[Any]          # the values to try
    datapoint_fields: list[str]      # keys each entry in `data` will carry


# initial spike: 3 microbenches, all small. Edit here to add more.
BENCHES: list[BenchPlan] = [
    BenchPlan(
        name="wg_size_sweep",
        purpose="occupancy cliff: largest workgroup size before throughput drops",
        shader_glsl_sketch=(
            "layout(local_size_x=WG) in;\n"
            "layout(set=0,binding=0) buffer A { float a[]; };\n"
            "layout(set=0,binding=1) buffer B { float b[]; };\n"
            "void main(){ b[gl_GlobalInvocationID.x] = a[gl_GlobalInvocationID.x]; }"
        ),
        sweep_axis="local_size_x",
        sweep_values=[32, 64, 128, 256, 512, 1024],
        datapoint_fields=["wg_size", "gflops", "gbps", "ms_per_dispatch"],
    ),
    BenchPlan(
        name="fp16_vs_fp32",
        purpose="ratio of fp16:fp32 throughput on memory-bound copy",
        shader_glsl_sketch=(
            "// two variants: float vs float16_t (requires VK_KHR_shader_float16_int8)\n"
            "layout(local_size_x=64) in;\n"
            "// copy 4 elements per thread; one variant uses f32, the other f16."
        ),
        sweep_axis="dtype",
        sweep_values=["fp32", "fp16"],
        datapoint_fields=["dtype", "gbps", "ms_per_dispatch", "ratio_vs_fp32"],
    ),
    BenchPlan(
        name="shared_mem_bw",
        purpose="bandwidth ratio: shared-memory read vs global read",
        shader_glsl_sketch=(
            "shared float tile[TILE];\n"
            "// one variant reads from shared after a barrier; the other from\n"
            "// the storage buffer directly. ratio = SMEM_BW / GLOBAL_BW."
        ),
        sweep_axis="tile_size",
        sweep_values=[64, 256, 1024, 4096],
        datapoint_fields=["tile_size", "smem_gbps", "global_gbps", "ratio"],
    ),
]


# ---------------------------------------------------------------------------
def run_stub(plan: BenchPlan) -> dict[str, Any]:
    """Produce a stub result: schema-shaped but no real numbers yet."""
    data = []
    for v in plan.sweep_values:
        entry: dict[str, Any] = {plan.sweep_axis: v}
        # fill measurement fields with None so the JSON shape is final
        for f in plan.datapoint_fields:
            entry.setdefault(f, None)
        data.append(entry)
    return {
        "name": plan.name,
        "status": "stub",         # set to "ok" when real measurement lands
        "purpose": plan.purpose,
        "shader_glsl_sketch": plan.shader_glsl_sketch,
        "sweep_axis": plan.sweep_axis,
        "data": data,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile",
                   help="path to <device>.json (default: pick the lone file in "
                        "experience_pool/backend_vulkan/tier2_hardware_profiles/)")
    args = p.parse_args()

    if args.profile:
        prof_path = Path(args.profile)
    else:
        cands = [p for p in PROFILE_DIR.glob("*.json") if not p.name.endswith(".raw.json")]
        if len(cands) != 1:
            raise SystemExit(f"need --profile: found {len(cands)} candidate(s) in {PROFILE_DIR}")
        prof_path = cands[0]

    profile = json.loads(prof_path.read_text(encoding="utf-8"))
    print(f"[microbench] target = {profile['identity'].get('device_name')}")

    results = []
    for plan in BENCHES:
        # skip benches whose preconditions don't hold on this device
        if plan.name == "fp16_vs_fp32" and not profile["features"].get("shader_float16"):
            results.append({"name": plan.name, "status": "skipped",
                            "reason": "shader_float16 = false on this device"})
            print(f"[microbench] {plan.name:20s} SKIPPED (no fp16)")
            continue
        r = run_stub(plan)
        results.append(r)
        print(f"[microbench] {plan.name:20s} {r['status']:8s} "
              f"({len(r.get('data', []))} datapoints planned)")

    profile["behavior_profile"] = {
        "schema_version": 1,
        "measured": False,                  # flip to True when stubs become real
        "benches": results,
    }
    prof_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[microbench] updated {prof_path}")


if __name__ == "__main__":
    main()
