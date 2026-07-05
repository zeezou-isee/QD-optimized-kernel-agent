"""Merge legacy `opgen/runs_arm/`, `opgen/runs_vulkan/`, `opgen/runs_base/`,
`opgen/runs_arm_optmz/` into the canonical `opgen/runs/` (new 5-stage layout).

Historical context: earlier iterations used separate top-level run roots per
experiment (runs_arm was the ARM correctness batch, runs_vulkan the vulkan
audit, runs_arm_optmz the ARM optimize sweep, runs_base the plain-base
regeneration). After the 5-stage layout landed, all of these should live under
a single `opgen/runs/<task>/{analyze,base_kernel,backends/*,graph,...}/`.

For each legacy root and each task under it, this script:
  1. Extracts what backend / stage the legacy tree represents (see the
     `_SOURCE_MAP` table below).
  2. Compares mtime against the corresponding new-layout target dir.
  3. On no conflict → copies (verb "copy_tree") to the new location.
  4. On conflict → keeps the newer side by mtime; the losing side is logged.
  5. After the copy, hands the imported dirs off to `migrate_runs_layout.py`
     conceptually — but we replicate its logic here so this is a one-shot
     drop-in without needing two passes.

Non-destructive by default: NEVER deletes legacy sources unless
`--delete-legacy` is passed (out of scope for now — user asked dry-run only).

Usage:
    # dry-run all four legacy roots
    .venv/bin/python scripts/merge_legacy_runs.py

    # dry-run one root only
    .venv/bin/python scripts/merge_legacy_runs.py --source runs_arm

    # actually copy
    .venv/bin/python scripts/merge_legacy_runs.py --apply

    # actually copy + prune legacy sources after successful merge
    .venv/bin/python scripts/merge_legacy_runs.py --apply --delete-legacy
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "opgen"))

import paths  # noqa: E402
from config import RUNS_ROOT  # noqa: E402


# ---------------------------------------------------------------------------
# Which legacy subdir under `<root>/<task>/` maps to WHICH new-layout target.
# `dst_kind` values (resolved via paths.py):
#   "kernel"     → base_kernel (dst = base_kernel_dir(...))
#   "kernel_arm" → backends/arm/kernel
#   "kernel_vulkan" → backends/vulkan/kernel
#   "graph"      → graph  (already the canonical location, but included so
#                          runs_arm/<task>/graph moves)
#   "operator"   → operator (canonical)
#   "adapter"    → adapter (canonical)
#   "optimize"   → backends/base/optimize (legacy flat optimize was base)
#   "optimize_wiki_on" / "optimize_wiki_off" → backends/base/optimize/wiki_{on,off}
#
# Legacy roots and the subdirs we know can appear under each task:
_LEGACY_KNOWN_SUBDIRS = (
    "kernel", "kernel_arm", "kernel_vulkan",
    "graph", "operator", "adapter",
    "optimize", "optimize_wiki_on", "optimize_wiki_off",
    "analyze", "base_kernel", "backends",  # if a legacy root was already partially migrated
)


# Per-legacy-root defaults for ambiguous mappings. The `optimize/` subdir
# under `runs_arm_optmz/<task>/` really means arm optimize — reading the
# summary.json confirms `baseline_kernel keys: cand_*_arm.cpp`. Same story
# if we ever get a runs_vulkan_optmz/. Fall back to "base" for legacy roots
# that came from before optimize gained a backend column.
_OPTIMIZE_BACKEND_BY_LEGACY_ROOT = {
    "runs_arm_optmz": "arm",
    # future: "runs_vulkan_optmz": "vulkan",
}


def _new_dst_for(root: Path, task: str, legacy_sub: str,
                 legacy_root_name: str = "") -> Path | None:
    """Return the new-layout destination for a legacy subdir name.
    None if we don't know how to map it (caller reports as unmapped).

    `legacy_root_name` (e.g. "runs_arm_optmz") disambiguates optimize/*/
    subdirs whose target backend is encoded in the LEGACY ROOT name, not in
    the subdir name itself.
    """
    optimize_backend = _OPTIMIZE_BACKEND_BY_LEGACY_ROOT.get(legacy_root_name, "base")

    if legacy_sub == "kernel":
        return paths.base_kernel_dir(root, task)
    if legacy_sub.startswith("kernel_") and not legacy_sub.startswith("kernel_e2e_repair"):
        backend = legacy_sub[len("kernel_"):]
        return paths.backend_kernel_dir(root, task, backend)
    if legacy_sub == "graph":
        return paths.graph_dir(root, task)
    if legacy_sub in ("operator", "adapter"):
        return paths.task_root(root, task) / legacy_sub
    if legacy_sub == "optimize":
        return paths.backend_optimize_dir(root, task, optimize_backend)
    if legacy_sub.startswith("optimize_wiki_"):
        mode = legacy_sub[len("optimize_wiki_"):]
        return paths.backend_optimize_dir(root, task, optimize_backend) / f"wiki_{mode}"
    if legacy_sub == "analyze":
        return paths.analyze_dir(root, task)
    if legacy_sub == "base_kernel":
        return paths.base_kernel_dir(root, task)
    if legacy_sub == "backends":
        # a whole legacy-migrated backends/ tree — merge subtree at
        # runs/<task>/backends/  (dircmp merge)
        return paths.task_root(root, task) / "backends"
    return None


def _dir_mtime(d: Path) -> float:
    """Recursively find the newest mtime under `d`. Returns 0 if empty/missing."""
    if not d.exists():
        return 0.0
    best = d.stat().st_mtime
    for p in d.rglob("*"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m > best:
            best = m
    return best


def plan_merge(legacy_root: Path,
               legacy_root_name: str = "") -> list[tuple[str, Path, Path, str]]:
    """Return a list of (verb, src, dst, note) for merging one legacy root
    into RUNS_ROOT.

    Verbs:
      "copy_tree"  — copy `src` to `dst` (dst does not exist)
      "skip_older" — dst exists and is newer than src; nothing to do (logged)
      "conflict_replace" — dst exists but src is newer; replace dst with src
                           (backs up dst to .oss-merge-backup/ first)
      "unmapped"   — legacy subdir name isn't in _SOURCE_MAP (logged)
    """
    ops: list[tuple[str, Path, Path, str]] = []
    if not legacy_root.exists():
        return ops
    for task_dir in sorted(legacy_root.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue
        task = task_dir.name
        for sub in sorted(task_dir.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name.startswith("_"):        # _pnnx_probe / etc — handled by
                continue                          # the destination side's migrate script
            dst = _new_dst_for(RUNS_ROOT, task, sub.name, legacy_root_name)
            if dst is None:
                ops.append(("unmapped", sub, task_dir / f"???", sub.name))
                continue
            if not dst.exists():
                ops.append(("copy_tree", sub, dst, ""))
            else:
                src_m = _dir_mtime(sub)
                dst_m = _dir_mtime(dst)
                if src_m > dst_m:
                    ops.append(("conflict_replace", sub, dst,
                                f"src_mtime={int(src_m)} > dst_mtime={int(dst_m)}"))
                else:
                    ops.append(("skip_older", sub, dst,
                                f"dst_mtime={int(dst_m)} >= src_mtime={int(src_m)}"))
    return ops


def _backup_dst(dst: Path, dry: bool) -> Path:
    """Move dst to a sibling .oss-merge-backup/ TS dir before replacing."""
    import time
    ts = time.strftime("%Y%m%dT%H%M%S")
    backup_root = RUNS_ROOT / ".oss-merge-backup" / ts
    rel = dst.relative_to(RUNS_ROOT) if dst.is_relative_to(RUNS_ROOT) else dst.name
    backup_dst = backup_root / rel
    if dry:
        return backup_dst
    backup_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dst), str(backup_dst))
    return backup_dst


def apply_ops(ops: list[tuple[str, Path, Path, str]], *, dry: bool) -> dict:
    stats = {"copy_tree": 0, "conflict_replace": 0, "skip_older": 0, "unmapped": 0}
    for verb, src, dst, note in ops:
        stats[verb] = stats.get(verb, 0) + 1
        rel_src = src.relative_to(ROOT) if src.is_relative_to(ROOT) else src
        rel_dst = dst.relative_to(ROOT) if dst.is_relative_to(ROOT) else dst
        note_str = f"  ({note})" if note else ""
        marker = {"copy_tree": "+", "conflict_replace": "!", "skip_older": "-",
                  "unmapped": "?"}.get(verb, " ")
        print(f"  {marker} {verb:18s} {rel_src}  ->  {rel_dst}{note_str}")
        if dry or verb in ("skip_older", "unmapped"):
            continue
        if verb == "copy_tree":
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(src), str(dst))
        elif verb == "conflict_replace":
            _backup_dst(dst, dry=False)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(src), str(dst))
    return stats


LEGACY_ROOTS = ("runs_arm", "runs_vulkan", "runs_base", "runs_arm_optmz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually copy; without this, dry-run report only")
    ap.add_argument("--source", default=None, choices=LEGACY_ROOTS,
                    help="merge only this legacy root (default: all four)")
    ap.add_argument("--delete-legacy", action="store_true",
                    help="after a successful merge, rm -rf the legacy source dir")
    args = ap.parse_args()

    sources = [args.source] if args.source else list(LEGACY_ROOTS)

    grand_stats: dict[str, int] = {}
    empty_sources: list[str] = []
    per_source_ops: dict[str, list] = {}

    for src_name in sources:
        legacy_root = ROOT / "opgen" / src_name
        if not legacy_root.exists():
            empty_sources.append(src_name)
            continue
        print(f"\n=== legacy source: opgen/{src_name}/ ===")
        ops = plan_merge(legacy_root, legacy_root_name=src_name)
        per_source_ops[src_name] = ops
        if not ops:
            print(f"  (empty — no tasks with mappable subdirs)")
            continue
        stats = apply_ops(ops, dry=not args.apply)
        for k, v in stats.items():
            grand_stats[k] = grand_stats.get(k, 0) + v

    print(f"\n[merge] summary:")
    for verb in ("copy_tree", "conflict_replace", "skip_older", "unmapped"):
        print(f"  {verb:18s}: {grand_stats.get(verb, 0)}")
    if empty_sources:
        print(f"  (empty legacy roots skipped: {', '.join(empty_sources)})")

    if not args.apply:
        print("\n[merge] dry-run — pass --apply to actually copy")
        return

    if args.delete_legacy:
        print("\n[merge] --delete-legacy: pruning legacy sources...")
        for src_name in sources:
            legacy_root = ROOT / "opgen" / src_name
            if not legacy_root.exists():
                continue
            ops = per_source_ops.get(src_name, [])
            # Only delete if every op is copy_tree / conflict_replace / skip_older
            # (never on unmapped: user may want to inspect). And only after ops applied.
            if any(v == "unmapped" for v, *_ in ops):
                print(f"  [keep] opgen/{src_name}/ has unmapped subdirs — NOT deleting")
                continue
            print(f"  rm -rf opgen/{src_name}/")
            shutil.rmtree(str(legacy_root))


if __name__ == "__main__":
    main()
