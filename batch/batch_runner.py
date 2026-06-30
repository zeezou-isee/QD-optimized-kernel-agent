"""Single runner for all batch OperatorAgent jobs (miniset/subset/all/...).

Usage:
  DEEPSEEK_API_KEY=... .venv/bin/python batch/batch_runner.py --set miniset
  ... batch/batch_runner.py --set miniset --ops Add,Abs        # debug a few ops
  ... batch/batch_runner.py --set all                          # full bench

Per-set knobs live in batch/sets/<name>.py (DATASET / MODEL / TIMEOUT etc).
Results land at batch/results/<set>.json, resumable across re-runs (ops with a
non-{crash,timeout} status are skipped).

Key safety: each OperatorAgent runs in its OWN process group. On per-op timeout
we SIGTERM the whole group (lets the ncnn-tree guard restore) then SIGKILL if
it lingers — without this, grandchild compilers (cmake/make/g++) become
orphans and race against the next op's compile in the shared ncnn build dir.

Key convenience: the OperatorAgent subprocess gets `.venv/bin` prepended to its
PATH, so cmake (installed via pip into the venv but not on the system PATH)
is always findable. Removes the "FileNotFoundError: cmake" class of crashes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

ROOT          = Path(__file__).resolve().parents[1]
BATCH_DIR     = Path(__file__).resolve().parent
SETS_PKG      = "batch.sets"
RESULTS_DIR   = BATCH_DIR / "results"
CLI_OPERATOR  = ROOT / "opgen" / "cli" / "run_operator_agent.py"
CLI_KERNEL    = ROOT / "opgen" / "cli" / "run_kernel_agent.py"
RUNS          = ROOT / "opgen" / "runs"
VENV_BIN      = ROOT / ".venv" / "bin"

REQUIRED_FIELDS = ("DATASET", "MODEL", "MAX_ROUNDS", "GRAPH_MAX_ROUNDS",
                   "PER_OP_TIMEOUT", "BACKENDS", "COMPILE_MODE")


def preflight_ncnn_clean() -> None:
    """Refuse to start if the ncnn source tree is already dirty under the agent's
    guarded paths.

    This is the OUTER half of a two-layer safety: each OperatorAgent already
    arms ncnn_tree_guard with --auto-cleanup on entry, but that's silent.
    If a prior unrelated session (or a manual experiment) left the tree dirty,
    the user deserves to know BEFORE we plow through dozens of ops and
    silently auto-clean their work away.

    Reuses ncnn_tree_guard._dirty_summary / _GUARDED_PATHS so this check
    stays aligned with what the OperatorAgent-side guard actually watches.
    """
    sys.path.insert(0, str(ROOT / "opgen" / "orchestrator"))
    sys.path.insert(0, str(ROOT / "opgen"))
    from ncnn_tree_guard import _dirty_summary, _GUARDED_PATHS, _is_git_repo
    from config import GraphConfig
    ncnn_root = GraphConfig().ncnn_root
    if not ncnn_root.exists():
        raise SystemExit(f"[batch] preflight: ncnn root not found at {ncnn_root}")
    if not _is_git_repo(ncnn_root):
        print(f"[batch] preflight: {ncnn_root} is not a git repo — skipping clean check",
              flush=True)
        return
    dirty = _dirty_summary(ncnn_root)
    if not dirty:
        print(f"[batch] preflight: ncnn tree clean at {ncnn_root}", flush=True)
        return
    raise SystemExit(
        f"[batch] preflight: ncnn tree is DIRTY at {ncnn_root}\n"
        f"        under guarded paths: {list(_GUARDED_PATHS)}\n"
        f"{dirty}\n"
        f"[batch] refusing to start — your changes would be auto-cleaned by the\n"
        f"        per-op guard. To proceed:\n"
        f"  - commit/stash the changes if they matter, OR\n"
        f"  - clean manually:\n"
        f"      git -C {ncnn_root} checkout -- src/CMakeLists.txt "
        f"tools/pnnx/src/pass_ncnn.cpp tools/pnnx/src/CMakeLists.txt "
        f"tools/pnnx/tests/ncnn/CMakeLists.txt 2>/dev/null\n"
        f"      git -C {ncnn_root} clean -fd src/layer tools/pnnx/src/pass_ncnn "
        f"tools/pnnx/tests/ncnn"
    )


def load_set(name: str) -> ModuleType:
    """Import batch.sets.<name> and verify it exposes the required constants."""
    # ensure the project root is importable so `import batch.sets.<name>` works
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    mod = importlib.import_module(f"{SETS_PKG}.{name}")
    missing = [f for f in REQUIRED_FIELDS if not hasattr(mod, f)]
    if missing:
        raise SystemExit(f"set {name!r} is missing required fields: {missing}")
    if not Path(mod.DATASET).exists():
        raise SystemExit(f"set {name!r} DATASET does not exist: {mod.DATASET}")
    return mod


def discover_ops(dataset: Path) -> list[tuple[str, str]]:
    """Return sorted (category, op_name) for every <Op>.py model under DATASET."""
    return [(py.parent.name, py.stem)
            for py in sorted(Path(dataset).rglob("*.py"))
            if py.stem != "__init__"]


def load_results(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_results(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def summarize(op: str) -> dict:
    """Read the agent's summary.json for `op` and extract per-phase outcomes."""
    sj = RUNS / op / "operator" / "summary.json"
    if not sj.exists():
        return {}
    try:
        s = json.loads(sj.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    ph = s.get("phases", {})
    return {
        "status":           s.get("status"),
        "kernel":           (ph.get("kernel") or {}).get("status"),
        "kernel_arm":       (ph.get("kernel_arm") or {}).get("status"),
        "graph":            (ph.get("graph") or {}).get("status"),
        "already_in_ncnn":  (ph.get("existence_check") or {}).get("already_in_ncnn"),
        "e2e":              (ph.get("end_to_end_numeric") or {}).get("passed"),
        "production":       (ph.get("production") or {}).get("_mandatory_ok"),
        "note":             s.get("note"),
    }


def summarize_kernel(op: str, backend: str) -> dict:
    """Read the KernelAgent's standalone summary.json for `op` (base or arm)."""
    sub = "kernel" if backend == "base" else f"kernel_{backend}"
    sj = RUNS / op / sub / "summary.json"
    if not sj.exists():
        return {}
    try:
        s = json.loads(sj.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    fr = s.get("final_result") or {}
    if fr.get("numeric_skipped"):
        numeric = "skipped"
    elif fr.get("numeric_ok"):
        numeric = "passed"
    else:
        numeric = "failed"
    return {
        "status":     s.get("status"),
        "backend":    s.get("backend"),
        "rounds":     s.get("rounds"),
        "compile":    fr.get("compile_ok"),
        "numeric":    numeric,
        "max_diff":   fr.get("max_diff"),
        "category":   fr.get("failure_category"),
    }


def child_env() -> dict:
    """Inherit parent env, but ensure .venv/bin is on PATH so cmake is findable.

    Background: cmake is pip-installed into the venv (.venv/bin/cmake) but the
    system PATH typically lacks .venv/bin. When the agent shells out to `cmake`,
    that lookup fails — turning many ops into spurious crashes.
    """
    env = dict(os.environ)
    if VENV_BIN.exists() and str(VENV_BIN) not in env.get("PATH", ""):
        env["PATH"] = f"{VENV_BIN}{os.pathsep}{env.get('PATH', '')}"
    return env


def run_one(category: str, op: str, cfg: ModuleType,
            mode: str = "operator", backend: str = "base",
            model: str | None = None) -> dict:
    """Spawn one agent (operator | kernel) for `op` and capture its summary.

    mode="operator" — full OperatorAgent pipeline (kernel + graph + e2e)
    mode="kernel"   — just KernelAgent standalone (no ncnn tree mutation;
                       --backend base by default, or arm via `backend` arg)
    model           — override cfg.MODEL (e.g. "claude-opus-4-8"); None=use cfg
    """
    model_name = model or cfg.MODEL
    if mode == "kernel":
        cmd = [
            sys.executable, str(CLI_KERNEL),
            "--task", op,
            "--dataset-root", str(cfg.DATASET),
            "--model-name", model_name,
            "--max-rounds", cfg.MAX_ROUNDS,
            "--backend", backend,
        ]
    else:
        cmd = [
            sys.executable, str(CLI_OPERATOR),
            "--task", op,
            "--dataset-root", str(cfg.DATASET),
            "--model-name", model_name,
            "--max-rounds", cfg.MAX_ROUNDS,
            "--graph-max-rounds", cfg.GRAPH_MAX_ROUNDS,
            "--backends", cfg.BACKENDS,
            "--compile-mode", cfg.COMPILE_MODE,
            "--auto-cleanup",
            # arm is a perf-optimization backend; its LLM-written NEON kernel is
            # subject to per-run variance. If it can't converge in MAX_ROUNDS,
            # degrade to the (correctness-verified) base kernel and still run the
            # end-to-end + production checks, rather than failing an op whose base
            # e2e is green. The degradation is recorded in the op summary note.
            "--allow-backend-fallback",
        ]
    timed_out = False
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=str(ROOT), env=child_env(),
                            start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=cfg.PER_OP_TIMEOUT)
        rc = proc.returncode
        tail = (stdout or "")[-2000:] + (stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        timed_out = True
        try: os.killpg(proc.pid, signal.SIGTERM)        # 3s grace for ncnn-tree guard
        except ProcessLookupError: pass
        try:
            stdout, stderr = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            stdout, stderr = proc.communicate()
        rc = -1
        tail = "TIMEOUT\n" + (stdout or "")[-1500:] + (stderr or "")[-1500:]
    dt = round(time.time() - t0, 1)

    row = {"category": category, "elapsed_s": dt, "returncode": rc,
           "timed_out": timed_out}
    row.update(summarize_kernel(op, backend) if mode == "kernel" else summarize(op))
    if not row.get("status"):
        row["status"] = "crash" if not timed_out else "timeout"
        row["tail"] = tail[-600:]
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", dest="set_name", required=True,
                   help="set config name (file batch/sets/<name>.py)")
    p.add_argument("--ops", default=None,
                   help="comma-separated op names; only these will run (debug)")
    p.add_argument("--results",
                   help="override result JSON path (default batch/results/<set>.json "
                        "or batch/results/<set>_kernel_<backend>.json in kernel-only mode)")
    p.add_argument("--kernel-only", action="store_true",
                   help="run KernelAgent standalone (no ncnn tree mutation, no GraphAgent). "
                        "Use this to test kernel-prompt changes in isolation.")
    p.add_argument("--backend", choices=["base", "arm", "vulkan"], default="base",
                   help="--kernel-only: which backend to verify (default base)")
    p.add_argument("--model", default=None,
                   help="override cfg.MODEL (e.g. claude-opus-4-8 / deepseek-v4-pro). "
                        "When omitted, uses the MODEL declared in batch/sets/<set>.py.")
    args = p.parse_args()

    cfg = load_set(args.set_name)
    if args.results:
        results_path = Path(args.results)
    elif args.kernel_only:
        results_path = RESULTS_DIR / f"{args.set_name}_kernel_{args.backend}.json"
    else:
        results_path = RESULTS_DIR / f"{args.set_name}.json"
    only = {s.strip() for s in args.ops.split(",")} if args.ops else None

    # second layer of safety: refuse to run on a dirty ncnn tree
    # (kernel-only mode never touches the ncnn tree → skip)
    if not args.kernel_only:
        preflight_ncnn_clean()
    else:
        print(f"[batch:{args.set_name}] kernel-only mode (backend={args.backend}) — "
              f"ncnn tree preflight skipped (KernelAgent does not mutate the tree)",
              flush=True)

    ops = discover_ops(cfg.DATASET)
    if only:
        ops = [(c, o) for (c, o) in ops if o in only]
        if not ops:
            raise SystemExit(f"--ops {args.ops!r} matched nothing in {cfg.DATASET}")

    results = load_results(results_path)
    total = len(ops)
    effective_model = args.model or cfg.MODEL
    print(f"[batch:{args.set_name}] {total} operators selected; "
          f"{sum(1 for op in [o for _,o in ops] if op in results)} already in results; "
          f"model={effective_model}",
          flush=True)

    for i, (cat, op) in enumerate(ops, 1):
        prev = results.get(op, {}).get("status")
        if prev and prev not in ("crash", "timeout"):
            print(f"[{i}/{total}] {cat}/{op}: SKIP (already {prev})", flush=True)
            continue
        print(f"[{i}/{total}] {cat}/{op}: running...", flush=True)
        mode = "kernel" if args.kernel_only else "operator"
        row = run_one(cat, op, cfg, mode=mode, backend=args.backend, model=args.model)
        results[op] = row
        save_results(results_path, results)
        if args.kernel_only:
            print(f"[{i}/{total}] {cat}/{op}: {row['status']} "
                  f"(rounds={row.get('rounds')} compile={row.get('compile')} "
                  f"numeric={row.get('numeric')} max_diff={row.get('max_diff')} "
                  f"{row['elapsed_s']}s)",
                  flush=True)
        else:
            print(f"[{i}/{total}] {cat}/{op}: {row['status']} "
                  f"(kernel={row.get('kernel')} graph={row.get('graph')} "
                  f"e2e={row.get('e2e')} prod={row.get('production')} {row['elapsed_s']}s)",
                  flush=True)

    ok = sum(1 for r in results.values() if r.get("status") == "success")
    print(f"\n[batch:{args.set_name}] DONE: {ok}/{len(results)} success "
          f"(results at {results_path})")


if __name__ == "__main__":
    main()
