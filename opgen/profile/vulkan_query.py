"""Tier 2 — Vulkan hardware profile (query-only side).

What it does:
  1. Run `vulkaninfo --json` to dump the full VP_VULKANINFO_<device>.json
     (limits / features / extensions / queue families / formats).
  2. Project it down to ONLY the fields the optimization agent will actually
     read (defined in TIER2_KEYS), and write a clean profile JSON under
     experience_pool/backend_vulkan/tier2_hardware_profiles/<device>.json.

Why a projection (not the raw 180KB dump):
  - The raw dump is structured for spec validation, not for prompting an LLM.
  - The agent consumes 10-20 specific limits / extension flags; carrying the
    rest into a prompt is noise. The raw dump is kept alongside as
    "<device>.raw.json" for audit / debugging.

Run:
  python3 opgen/profile/vulkan_query.py [--gpu N]

A companion microbench script (vulkan_microbench.py) appends a `behavior_profile`
section to the same projection later — see plan
/Users/xingze/.claude/plans/glittery-jumping-sloth.md.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJ = Path(__file__).resolve().parents[2]
OUT_DIR = PROJ / "experience_pool" / "backend_vulkan" / "tier2_hardware_profiles"


# ---------------------------------------------------------------------------
# what we project out of the VP_VULKANINFO json. Keys live under
# capabilities.device.properties.<struct>.<field>; values land flat in the
# output. Order = decision-table-friendly grouping, not source order.
# ---------------------------------------------------------------------------
TIER2_KEYS: list[tuple[str, str, str]] = [
    # (output_key, source_struct, source_field)
    # ---- identity ----
    ("vendor_id",                        "VkPhysicalDeviceProperties", "vendorID"),
    ("device_id",                        "VkPhysicalDeviceProperties", "deviceID"),
    ("device_name",                      "VkPhysicalDeviceProperties", "deviceName"),
    ("device_type",                      "VkPhysicalDeviceProperties", "deviceType"),
    ("api_version",                      "VkPhysicalDeviceProperties", "apiVersion"),
    ("driver_version",                   "VkPhysicalDeviceProperties", "driverVersion"),
    # ---- compute limits (the most decision-relevant ones) ----
    ("max_compute_shared_memory_size",   "VkPhysicalDeviceProperties.limits", "maxComputeSharedMemorySize"),
    ("max_compute_workgroup_size",       "VkPhysicalDeviceProperties.limits", "maxComputeWorkGroupSize"),
    ("max_compute_workgroup_invocations","VkPhysicalDeviceProperties.limits", "maxComputeWorkGroupInvocations"),
    ("max_compute_workgroup_count",      "VkPhysicalDeviceProperties.limits", "maxComputeWorkGroupCount"),
    ("max_push_constants_size",          "VkPhysicalDeviceProperties.limits", "maxPushConstantsSize"),
    ("max_storage_buffer_range",         "VkPhysicalDeviceProperties.limits", "maxStorageBufferRange"),
    # storage-image alignment matters for image vs buffer decision
    ("min_storage_buffer_offset_alignment",
                                         "VkPhysicalDeviceProperties.limits", "minStorageBufferOffsetAlignment"),
    ("min_uniform_buffer_offset_alignment",
                                         "VkPhysicalDeviceProperties.limits", "minUniformBufferOffsetAlignment"),
    # ---- subgroup (Vulkan 1.1 core) ----
    ("subgroup_size",                    "VkPhysicalDeviceVulkan11Properties", "subgroupSize"),
    ("subgroup_supported_stages",        "VkPhysicalDeviceVulkan11Properties", "subgroupSupportedStages"),
    ("subgroup_supported_operations",    "VkPhysicalDeviceVulkan11Properties", "subgroupSupportedOperations"),
    # ---- image dims (for image-vs-buffer storage choice) ----
    ("max_image_dimension_2d",           "VkPhysicalDeviceProperties.limits", "maxImageDimension2D"),
    ("max_image_dimension_3d",           "VkPhysicalDeviceProperties.limits", "maxImageDimension3D"),
]


# ---- features we look up by name (boolean flags scattered across structs) --
TIER2_FEATURE_FLAGS: list[tuple[str, str, str]] = [
    # (output_key, source_struct, source_field)
    ("shader_int16",     "VkPhysicalDeviceFeatures",            "shaderInt16"),
    ("shader_int64",     "VkPhysicalDeviceFeatures",            "shaderInt64"),
    ("shader_float64",   "VkPhysicalDeviceFeatures",            "shaderFloat64"),
    ("storage_buffer_16bit_access",
                         "VkPhysicalDeviceVulkan11Features",    "storageBuffer16BitAccess"),
    ("uniform_and_storage_buffer_16bit_access",
                         "VkPhysicalDeviceVulkan11Features",    "uniformAndStorageBuffer16BitAccess"),
    ("shader_float16",   "VkPhysicalDeviceVulkan12Features",    "shaderFloat16"),
    ("shader_int8",      "VkPhysicalDeviceVulkan12Features",    "shaderInt8"),
    ("shader_buffer_int64_atomics",
                         "VkPhysicalDeviceVulkan12Features",    "shaderBufferInt64Atomics"),
    ("compute_full_subgroups",
                         "VkPhysicalDeviceVulkan13Features",    "computeFullSubgroups"),
    ("subgroup_size_control",
                         "VkPhysicalDeviceVulkan13Features",    "subgroupSizeControl"),
]


# ---- extensions we mark presence-of (perf-relevant only) ------------------
TIER2_EXTENSIONS = [
    "VK_KHR_cooperative_matrix",       # tensor-core-equivalent
    "VK_NV_cooperative_matrix",
    "VK_KHR_16bit_storage",
    "VK_KHR_shader_float16_int8",
    "VK_KHR_shader_subgroup_extended_types",
    "VK_KHR_shader_integer_dot_product",  # dp4a-equivalent
    "VK_EXT_subgroup_size_control",
    "VK_EXT_descriptor_indexing",
    "VK_KHR_push_descriptor",
]


# ---------------------------------------------------------------------------
def _lookup(props: dict[str, Any], struct_dotted: str, field: str) -> Any:
    """Resolve "VkPhysicalDeviceProperties.limits" -> props["VkPhysicalDeviceProperties"]["limits"][field]."""
    cur: Any = props
    for part in struct_dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, dict) and field in cur:
        return cur[field]
    return None


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return s or "unknown_device"


def run_vulkaninfo(gpu: int | None, work_dir: Path) -> Path:
    """Invoke vulkaninfo --json [=<gpu>], capture the VP_VULKANINFO_*.json it drops.

    vulkaninfo writes the JSON to its CWD, with a name we cannot pre-set —
    we run it from a temp working dir and grab whatever it produces.
    """
    if shutil.which("vulkaninfo") is None:
        raise RuntimeError("vulkaninfo not on PATH. Install vulkan-tools "
                           "(`brew install vulkan-tools` on macOS).")
    work_dir.mkdir(parents=True, exist_ok=True)
    arg = "--json" if gpu is None else f"--json={gpu}"
    proc = subprocess.run(["vulkaninfo", arg], cwd=str(work_dir),
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"vulkaninfo --json failed (rc={proc.returncode}):\n"
                           f"stderr={proc.stderr[-400:]}")
    drops = sorted(work_dir.glob("VP_VULKANINFO_*.json"))
    if not drops:
        raise RuntimeError(f"vulkaninfo --json produced no VP_VULKANINFO_*.json in {work_dir}")
    return drops[-1]


def project_tier2(vp_raw: dict[str, Any]) -> dict[str, Any]:
    """Project the VP dump down to the agent-facing fields defined in TIER2_*."""
    device = vp_raw.get("capabilities", {}).get("device", {})
    props = device.get("properties", {}) or {}
    feats = device.get("features", {}) or {}
    exts_raw = device.get("extensions", {}) or {}

    out: dict[str, Any] = {"identity": {}, "compute_limits": {}, "subgroup": {},
                           "image_limits": {}, "features": {}, "extensions_present": {}}

    # split TIER2_KEYS into sections by output_key prefix
    for out_key, struct_path, field in TIER2_KEYS:
        v = _lookup(props, struct_path, field)
        if out_key in ("vendor_id", "device_id", "device_name", "device_type",
                       "api_version", "driver_version"):
            out["identity"][out_key] = v
        elif out_key.startswith("subgroup_"):
            out["subgroup"][out_key] = v
        elif out_key.startswith("max_image_"):
            out["image_limits"][out_key] = v
        else:
            out["compute_limits"][out_key] = v

    for out_key, struct, field in TIER2_FEATURE_FLAGS:
        # features schema mirrors properties schema (one sub-dict per struct).
        v = _lookup(feats, struct, field)
        out["features"][out_key] = bool(v) if v is not None else None

    # extensions: vulkaninfo lists them as {name: revision} or as a list of dicts;
    # accept both shapes.
    if isinstance(exts_raw, dict):
        present = set(exts_raw.keys())
    elif isinstance(exts_raw, list):
        present = {e.get("extensionName") or e.get("name") for e in exts_raw if isinstance(e, dict)}
    else:
        present = set()
    for name in TIER2_EXTENSIONS:
        out["extensions_present"][name] = name in present

    # placeholder — vulkan_microbench.py appends in-place under this key
    out["behavior_profile"] = None

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index for multi-device hosts (default = device 0)")
    p.add_argument("--out-dir", default=str(OUT_DIR),
                   help="output dir for the projected profile JSON")
    p.add_argument("--keep-raw", action="store_true",
                   help="also write the raw VP_VULKANINFO_*.json next to the projection")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # vulkaninfo's working dir = a scratch under out_dir we then read from.
    raw_path = run_vulkaninfo(args.gpu, out_dir / "_raw")
    vp = json.loads(raw_path.read_text(encoding="utf-8"))
    projection = project_tier2(vp)

    device_name = projection["identity"].get("device_name") or "unknown_device"
    slug = _slugify(device_name)
    out_path = out_dir / f"{slug}.json"
    out_path.write_text(json.dumps(projection, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    if args.keep_raw:
        # raw stays alongside for audit; renamed to a stable per-device name
        raw_target = out_dir / f"{slug}.raw.json"
        raw_target.write_text(raw_path.read_text(encoding="utf-8"), encoding="utf-8")

    # report
    n_ext = sum(1 for v in projection["extensions_present"].values() if v)
    n_feat = sum(1 for v in projection["features"].values() if v)
    print(f"[vulkan_query] device = {device_name}")
    print(f"[vulkan_query] subgroup_size = {projection['subgroup'].get('subgroup_size')}")
    print(f"[vulkan_query] max_shared_mem = "
          f"{projection['compute_limits'].get('max_compute_shared_memory_size')} bytes")
    print(f"[vulkan_query] features ON: {n_feat}/{len(projection['features'])}; "
          f"perf-extensions present: {n_ext}/{len(projection['extensions_present'])}")
    print(f"[vulkan_query] wrote {out_path}")


if __name__ == "__main__":
    main()
