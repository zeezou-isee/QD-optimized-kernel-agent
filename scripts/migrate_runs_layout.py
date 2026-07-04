"""One-shot migration: legacy `opgen/runs/<task>/{kernel,kernel_arm,kernel_vulkan,optimize}/`
layout → the new 5-stage layout (analyze/ + base_kernel/{artifacts} + backends/<b>/).

Usage:
    # dry-run: print the plan for every task
    .venv/bin/python scripts/migrate_runs_layout.py

    # actually move things for one task
    .venv/bin/python scripts/migrate_runs_layout.py --apply --task Abs

    # actually move everything
    .venv/bin/python scripts/migrate_runs_layout.py --apply

Idempotent: an already-migrated task is a no-op. Never touches `graph/`,
`operator/`, `adapter/`, `_oracle/`, `_net/`, `_arm_batch/`, `_vk_oracle/`.
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


# Never touch these subdirs — they are orchestrator-private / shared oracles.
PROTECTED = {"graph", "operator", "adapter", "_oracle", "_net", "_arm_batch",
             "_vk_oracle", "analyze", "base_kernel", "backends"}


def plan_for_task(task_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return a list of (verb, src, dst) tuples describing the migration.

    Verbs:
      - "move" : shutil.move src -> dst
      - "copy" : shutil.copytree src -> dst (used for probe hoist; the source
                  may still be read by legacy code, so keep both)
      - "write_artifacts" : extract .h/.cpp from summary.json into
                            base_kernel/artifacts/ (src=base_kernel dir)
      - "copy_file" : single file copy
    """
    ops: list[tuple[str, Path, Path]] = []
    task = task_dir.name
    root = RUNS_ROOT

    # 1. kernel/ -> base_kernel/
    old_base = task_dir / "kernel"
    new_base = paths.base_kernel_dir(root, task)
    if old_base.exists() and old_base.is_dir() and not new_base.exists():
        ops.append(("move", old_base, new_base))

    # 2. kernel_<backend>/ -> backends/<backend>/kernel/ (arm, vulkan, future)
    for legacy_sub in sorted(task_dir.glob("kernel_*")):
        if not legacy_sub.is_dir():
            continue
        backend = legacy_sub.name[len("kernel_"):]
        # skip any accidental "kernel_" subdirs that carry an e2e_repair suffix
        # (e.g. kernel_e2e_repair_1 — that's the base kernel with a suffix,
        # let it live where it lives; migration would break op_agent state)
        if backend.startswith("e2e_repair_"):
            continue
        new = paths.backend_kernel_dir(root, task, backend)
        if not new.exists():
            ops.append(("move", legacy_sub, new))

    # 3. optimize/ -> backends/base/optimize/  (assume the legacy flat optimize
    #    dir is base — that's how run_optimize.py historically defaulted)
    old_opt = task_dir / "optimize"
    new_opt = paths.backend_optimize_dir(root, task, "base")
    if old_opt.exists() and old_opt.is_dir() and not new_opt.exists():
        ops.append(("move", old_opt, new_opt))

    # After the moves above we can now reason about the NEW dirs when
    # collecting shared analyze/ artifacts.

    # 4. Hoist pnnx probe → analyze/pnnx_probe/  (COPY, not move — operator
    #    may still be reading its own _baseline_probe copy).
    #    Check BOTH pre-move locations (kernel/, kernel_*) and post-move
    #    locations (base_kernel/, backends/*/kernel/) so the plan is correct
    #    whether we run migrate step-by-step or all-at-once.
    probe_dst = paths.analyze_pnnx_probe_dir(root, task)
    if not probe_dst.exists():
        candidates: list[Path] = [
            task_dir / "operator" / "_baseline_probe",
            # post-move locations (in case moves in this same run haven't been applied yet
            # in dry mode — we still list them for the informational output)
            new_base / "_pnnx_probe",
            # legacy pre-move locations
            task_dir / "kernel" / "_pnnx_probe",
            task_dir / "kernel_vulkan" / "_pnnx_probe",
            task_dir / "kernel_arm" / "_pnnx_probe",
        ]
        if (task_dir / "backends").exists():
            candidates.extend(sorted((task_dir / "backends").glob("*/kernel/_pnnx_probe")))
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                ops.append(("copy", cand, probe_dst))
                break

    # 5. Hoist introspect.json → analyze/introspect.json (copy, keep per-backend).
    intro_dst = paths.introspect_json(root, task)
    if not intro_dst.exists():
        intro_candidates = [
            new_base / "introspect.json",           # post-move base
            task_dir / "kernel" / "introspect.json",       # legacy base
            task_dir / "kernel_vulkan" / "introspect.json",
            task_dir / "kernel_arm" / "introspect.json",
        ]
        if (task_dir / "backends").exists():
            intro_candidates.extend(
                sorted((task_dir / "backends").glob("*/kernel/introspect.json")))
        for cand in intro_candidates:
            if cand.exists():
                ops.append(("copy_file", cand, intro_dst))
                break

    # 6. Publish base_kernel/artifacts/ from the final round's response_code.
    #    IMPORTANT: this step runs AFTER moves (see _reorder_for_execution).
    #    If there's a pending move (old_base → new_base), point at new_base
    #    which is where summary.json will live at write_artifacts time.
    art_dst = paths.base_kernel_artifacts_dir(root, task)
    if not art_dst.exists():
        will_move_old_base = old_base.exists() and not new_base.exists()
        if will_move_old_base:
            ops.append(("write_artifacts", new_base, art_dst))
        elif new_base.exists() and (new_base / "summary.json").exists():
            ops.append(("write_artifacts", new_base, art_dst))
        elif old_base.exists() and (old_base / "summary.json").exists():
            # legacy-only edge case: no move planned yet artifacts missing
            ops.append(("write_artifacts", old_base, art_dst))

    return ops


def _apply_write_artifacts(base_dir: Path, art_dst: Path, *, dry: bool) -> None:
    """Extract .h/.cpp + profile from base_kernel/summary.json into artifacts/."""
    summ = base_dir / "summary.json"
    if not summ.exists():
        return
    try:
        data = json.loads(summ.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"    [warn] cannot parse {summ}: {exc}")
        return
    if data.get("status") != "success":
        print(f"    [skip] base_kernel status={data.get('status')}, not publishing artifacts/")
        return
    code = (data.get("final_result") or {}).get("response_code") or {}
    prof = data.get("kernel_profile") or {}
    if not code:
        print(f"    [skip] no response_code in {summ}")
        return
    if dry:
        for name in sorted(code):
            print(f"      would write artifacts/{name} ({len(code[name])} chars)")
        if prof:
            print("      would write artifacts/kernel_profile.json")
        return
    art_dst.mkdir(parents=True, exist_ok=True)
    for name, body in code.items():
        if name.endswith((".h", ".hpp", ".cpp", ".cc", ".cxx")):
            (art_dst / name).write_text(body, encoding="utf-8")
    if prof:
        (art_dst / "kernel_profile.json").write_text(
            json.dumps(prof, indent=2, ensure_ascii=False), encoding="utf-8")


def _reorder_for_execution(ops: list[tuple[str, Path, Path]]) -> list[tuple[str, Path, Path]]:
    """Execution order:
      1. copies/copy_files — reading legacy source dirs (like kernel_vulkan/
         _pnnx_probe) BEFORE the moves might delete them.
      2. moves — rename legacy → new layout. IMPORTANT: this must happen
         BEFORE write_artifacts, because write_artifacts would otherwise
         mkdir base_kernel/artifacts/ preemptively, and then `shutil.move
         kernel → base_kernel` would move the legacy dir INSIDE
         base_kernel/ (giving the buggy base_kernel/kernel/ nesting).
      3. write_artifacts — after moves, the new base_kernel/ (with summary
         and rounds) exists and we can safely mkdir artifacts/ inside it.
    """
    order = {"copy": 0, "copy_file": 0, "move": 1, "write_artifacts": 2}
    return sorted(ops, key=lambda t: order.get(t[0], 99))


def apply(ops: list[tuple[str, Path, Path]], *, dry: bool) -> None:
    ops = _reorder_for_execution(ops)
    for verb, src, dst in ops:
        rel_src = src.relative_to(RUNS_ROOT) if src.is_relative_to(RUNS_ROOT) else src
        rel_dst = dst.relative_to(RUNS_ROOT) if dst.is_relative_to(RUNS_ROOT) else dst
        print(f"    {verb:15s} {rel_src}  ->  {rel_dst}")
        if dry:
            if verb == "write_artifacts":
                _apply_write_artifacts(src, dst, dry=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if verb == "move":
            shutil.move(str(src), str(dst))
        elif verb == "copy":
            shutil.copytree(str(src), str(dst))
        elif verb == "copy_file":
            shutil.copy2(str(src), str(dst))
        elif verb == "write_artifacts":
            _apply_write_artifacts(src, dst, dry=False)
        else:
            raise ValueError(f"unknown verb: {verb}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Migrate opgen/runs/<task>/ to the 5-stage layout. Idempotent.")
    ap.add_argument("--apply", action="store_true", help="actually move; without this, dry-run")
    ap.add_argument("--task", default=None, help="only migrate this task")
    args = ap.parse_args()

    if not RUNS_ROOT.exists():
        print(f"[migrate] {RUNS_ROOT} does not exist — nothing to do")
        return

    if args.task:
        tasks = [RUNS_ROOT / args.task]
        if not tasks[0].exists():
            print(f"[migrate] task dir {tasks[0]} does not exist")
            return
    else:
        tasks = sorted(p for p in RUNS_ROOT.iterdir()
                       if p.is_dir() and not p.name.startswith("_"))

    total = 0
    changed = 0
    for t in tasks:
        total += 1
        ops = plan_for_task(t)
        if not ops:
            print(f"[{t.name}] up to date")
            continue
        changed += 1
        print(f"[{t.name}] {len(ops)} operations:")
        apply(ops, dry=not args.apply)

    action = "would migrate" if not args.apply else "migrated"
    print(f"\n[migrate] {action} {changed}/{total} tasks"
          f"{'  (dry-run; pass --apply to execute)' if not args.apply else ''}")


if __name__ == "__main__":
    main()
