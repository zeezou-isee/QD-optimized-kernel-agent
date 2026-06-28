"""ncnn-tree guard: keep the ncnn source tree clean across agent runs.

The agent injects files into ncnn (`src/layer/`, `tools/pnnx/src/pass_ncnn/`) and
patches `pass_ncnn.cpp` / CMakeLists. The existing graph/net oracle restores those
mutations in its `finally`. But the finally cannot save us when:
  - the process is SIGKILL'd (OOM, sleep, user kill -9, batch driver timeout)
  - a previous run left the tree dirty for any reason
  - the orchestrator itself crashes before reaching its own teardown

This guard adds two complementary defenses:

  (A) Snapshot + force restore.  At construct time we record the ncnn HEAD SHA
      and the current `git status` of the agent-touched dirs. On `restore()` we
      `git reset --hard HEAD -- <dirs>` and `git clean -fd <dirs>`. Registered
      as an atexit + SIGTERM/SIGINT handler so a clean shutdown — and any
      crash — re-runs it. (SIGKILL still can't be intercepted; (C) covers that.)

  (B) Dirty-tree precheck.  On startup, if the tree is already dirty in the
      tracked dirs we either ABORT (default) or auto-clean (when policy=
      "auto"), so a leaked state from a prior killed run can't poison this one.

The guard only touches these directories:
    src/layer/                tools/pnnx/src/pass_ncnn/
    src/CMakeLists.txt        tools/pnnx/src/pass_ncnn.cpp
    tools/pnnx/src/CMakeLists.txt
"""

from __future__ import annotations

import atexit
import signal
import subprocess
import sys
from pathlib import Path
from typing import Sequence

# What the agents touch in ncnn — keep this list narrow on purpose.
_GUARDED_PATHS: tuple[str, ...] = (
    "src/layer",
    "src/CMakeLists.txt",
    "tools/pnnx/src/pass_ncnn",
    "tools/pnnx/src/pass_ncnn.cpp",
    "tools/pnnx/src/CMakeLists.txt",
    "tools/pnnx/tests/ncnn",
)


def _git(ncnn_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ncnn_root), *args],
                          capture_output=True, text=True, check=check)


def _is_git_repo(ncnn_root: Path) -> bool:
    try:
        r = _git(ncnn_root, "rev-parse", "--is-inside-work-tree", check=False)
        return r.returncode == 0 and r.stdout.strip() == "true"
    except Exception:  # noqa: BLE001
        return False


def _dirty_summary(ncnn_root: Path) -> str:
    r = _git(ncnn_root, "status", "--porcelain", "--", *_GUARDED_PATHS, check=False)
    return (r.stdout or "").strip()


class NcnnTreeGuard:
    """Snapshot + restore the ncnn tree under guarded paths."""

    # Policy values:
    #   "abort"  — startup precheck fails with an error if the tree is dirty
    #   "auto"   — startup precheck silently runs a restore using the SAME logic
    #              we use on teardown (so a leaked prior state can't poison this run)
    def __init__(self, ncnn_root: Path, *, dirty_policy: str = "abort",
                 verbose: bool = True) -> None:
        self.ncnn_root = Path(ncnn_root).resolve()
        self.verbose = verbose
        self.snapshot_sha: str | None = None
        self.enabled = _is_git_repo(self.ncnn_root)
        self._registered = False
        if not self.enabled:
            self._log(f"[ncnn-guard] {self.ncnn_root} is not a git repo — guard DISABLED")
            return
        self._precheck(dirty_policy)
        self.snapshot_sha = _git(self.ncnn_root, "rev-parse", "HEAD").stdout.strip()
        self._log(f"[ncnn-guard] armed at HEAD={self.snapshot_sha[:10]} "
                  f"(guards: {len(_GUARDED_PATHS)} paths)")
        self._install_handlers()

    # ------------------------------------------------------------------ (C)
    def _precheck(self, policy: str) -> None:
        dirty = _dirty_summary(self.ncnn_root)
        if not dirty:
            return
        msg = (f"[ncnn-guard] ncnn tree is DIRTY before this run "
               f"(under {list(_GUARDED_PATHS)}):\n{dirty}")
        if policy == "auto":
            self._log(msg)
            self._log("[ncnn-guard] policy=auto -> cleaning leaked state ...")
            self._do_restore("HEAD")            # restore to current HEAD (no prior tag yet)
            still = _dirty_summary(self.ncnn_root)
            if still:
                raise RuntimeError(f"[ncnn-guard] auto-clean did not fully clean tree:\n{still}")
            self._log("[ncnn-guard] auto-clean OK")
            return
        raise RuntimeError(
            msg + "\n[ncnn-guard] refusing to run on a dirty tree. Either:\n"
            "  - manually clean it (`git -C ncnn checkout -- src/CMakeLists.txt tools/pnnx/src/CMakeLists.txt "
            "tools/pnnx/src/pass_ncnn.cpp tools/pnnx/tests/ncnn/CMakeLists.txt && "
            "git -C ncnn clean -fd src/layer tools/pnnx/src/pass_ncnn tools/pnnx/tests/ncnn`)\n"
            "  - or pass --auto-cleanup so the guard cleans it for you next time.")

    # ------------------------------------------------------------------ (A)
    def _install_handlers(self) -> None:
        if self._registered: return
        atexit.register(self._safe_restore_on_exit)
        # only override SIGTERM/SIGINT; leave SIGKILL/SIGSEGV alone (untrappable).
        # signal handlers chain into the existing handler if any.
        for sig in (signal.SIGTERM, signal.SIGINT):
            prev = signal.getsignal(sig)
            def _h(signum, frame, _prev=prev, _sig=sig):
                self._safe_restore_on_exit()
                if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                    _prev(signum, frame)
                else:
                    signal.signal(_sig, signal.SIG_DFL)
                    signal.raise_signal(_sig)
            try: signal.signal(sig, _h)
            except (ValueError, OSError):
                pass                                    # not in main thread
        self._registered = True

    def _safe_restore_on_exit(self) -> None:
        if not self.enabled or self.snapshot_sha is None: return
        try: self.restore()
        except Exception as exc:                       # noqa: BLE001
            print(f"[ncnn-guard] restore on exit FAILED: {exc}", file=sys.stderr)

    def restore(self) -> None:
        """Force ncnn tree back to the snapshot SHA, deleting any agent injections."""
        if not self.enabled or self.snapshot_sha is None: return
        target = self.snapshot_sha
        self._do_restore(target)

    def _do_restore(self, target: str) -> None:
        # revert tracked changes under guarded paths
        _git(self.ncnn_root, "checkout", target, "--", *_GUARDED_PATHS, check=False)
        # delete untracked files/dirs under guarded paths (only the guarded subset)
        _git(self.ncnn_root, "clean", "-fdx" if False else "-fd", "--", *_GUARDED_PATHS, check=False)
        leftover = _dirty_summary(self.ncnn_root)
        if leftover:
            self._log(f"[ncnn-guard] restore leftover (not under guarded paths):\n{leftover}")
        else:
            self._log("[ncnn-guard] tree restored clean")

    def _log(self, msg: str) -> None:
        if self.verbose: print(msg, flush=True)


# convenience for OperatorAgent
def arm_guard(ncnn_root: Path, auto_cleanup: bool) -> NcnnTreeGuard:
    return NcnnTreeGuard(ncnn_root, dirty_policy="auto" if auto_cleanup else "abort")
