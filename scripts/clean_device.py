"""Clean up the files this project pushes to the phone (adb).

The device oracles / bench scripts push runner binaries, libc++_shared.so, input/
weight/output bins, and (for the sweep) benchncnn + models into a few fixed dirs
under /data/local/tmp. Over many runs these accumulate. This removes ONLY those
known kernelgen dirs — nothing else on the device is touched.

Usage:
    .venv/bin/python scripts/clean_device.py            # show sizes, then delete
    .venv/bin/python scripts/clean_device.py --dry-run  # show what's there, delete nothing
    .venv/bin/python scripts/clean_device.py -s <serial> # target a specific device
"""
from __future__ import annotations

import argparse
import subprocess
import sys

# Only these — populated by device_oracle.py / run_perf_compare.py / bench_*.py.
DEVICE_DIRS = [
    "/data/local/tmp/oracle",    # DeviceOracle (arm single-layer runner)
    "/data/local/tmp/vkoracle",  # VulkanDeviceOracle (vulkan single-layer runner)
    "/data/local/tmp/vkrun",     # vulkan bench
    "/data/local/tmp/ncnn",      # benchncnn sweep (binary + models)
]


def _devices() -> list[str]:
    try:
        out = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=15).stdout
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"adb not usable: {exc}")
    return [l.split("\t")[0] for l in out.splitlines()[1:] if l.strip().endswith("\tdevice")]


def _adb(serial: str | None, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    cmd = ["adb"] + (["-s", serial] if serial else []) + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove kernelgen's pushed files from the phone.")
    ap.add_argument("-s", "--serial", default=None, help="device serial (else the only connected one)")
    ap.add_argument("--dry-run", action="store_true", help="show sizes, delete nothing")
    args = ap.parse_args()

    devs = _devices()
    if not devs:
        sys.exit("no authorized android device (adb sees none).")
    serial = args.serial
    if serial is None:
        if len(devs) > 1:
            sys.exit(f"multiple devices {devs}; pass -s <serial>.")
        serial = devs[0]
    print(f"device: {serial}")

    total_present = 0
    for d in DEVICE_DIRS:
        # size if present, else mark absent
        r = _adb(serial, "shell", f"[ -d {d} ] && du -sh {d} 2>/dev/null || echo __ABSENT__")
        line = (r.stdout or "").strip()
        if "__ABSENT__" in line or not line:
            print(f"  absent   {d}")
            continue
        size = line.split()[0]
        total_present += 1
        if args.dry_run:
            print(f"  [dry]    {d}  ({size})")
        else:
            rm = _adb(serial, "shell", f"rm -rf {d}")
            ok = rm.returncode == 0
            print(f"  {'removed ' if ok else 'FAILED  '} {d}  ({size}){'' if ok else '  '+rm.stderr.strip()}")

    if args.dry_run:
        print(f"\ndry-run: {total_present} dir(s) present, nothing deleted.")
    else:
        print(f"\ncleaned {total_present} dir(s) on {serial}.")


if __name__ == "__main__":
    main()
