"""Reusable ncnn layer oracle (方案A).

Compile a generic runner together with ONE candidate layer .cpp (linked against a
prebuilt libncnn.a), feed it inputs/params/weights, run forward, and read the
output back as numpy — no per-operator C++ test file, no ncnn source-tree edits,
no libncnn rebuild. Optionally compare against a PyTorch reference (oracle).

Typical use:

    from layer_oracle import LayerOracle
    oc = LayerOracle()
    out = oc.run(
        candidate_cpp="ncnn/src/layer/convolution1d.cpp",
        class_name="Convolution1D", header="convolution1d.h",
        params={0:4, 1:3, 2:1, 3:1, 4:0, 5:1, 6:24},
        inputs=[x_ncnn],              # numpy in NCNN layout (batch already dropped)
        weights=[w_flat, bias],
    )
    # out["outputs"][0] is a numpy array

    verdict = oc.verify(... , reference=torch_out_np, tol=1e-3)
    # verdict["passed"] is True/False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import os
import re
import signal
import struct
import subprocess

import numpy as np

from .failure_taxonomy import classify_failure


# ---------------------------------------------------------------------------
# Shared helper: run a build command in its own process group with a timeout,
# so that (a) a SIGTERM from our parent (e.g. batch-driver timeout) propagates
# to make/g++ grandchildren via killpg, and (b) a runaway compile cannot pin
# the orchestrator forever. All cmake --build call sites should go through this.
# ---------------------------------------------------------------------------
def _run_cmake_bounded(cmd: list[str], timeout: int = 600,
                       cwd: str | Path | None = None,
                       env: dict | None = None) -> tuple[int, str]:
    """Spawn `cmd` in its own session, wait up to `timeout` seconds.

    On timeout: SIGTERM the process group (give cmake ~3s to flush), then SIGKILL.
    Returns (returncode_or_-1, captured_stdout+stderr).
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=str(cwd) if cwd else None, env=env,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, (out or "") + (err or "")
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try:
            out, err = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            out, err = proc.communicate()
        return -1, "TIMEOUT after %ds\n" % timeout + (out or "") + (err or "")

# The runner instantiates the candidate class directly, so DEFINE_LAYER_CREATOR is
# dead code — and it collides with the copy ncnn_add_layer bakes into libncnn.a when
# the same layer is already installed (orchestrator bridge path). Strip it.
_CREATOR_RE = re.compile(r"^\s*DEFINE_LAYER_CREATOR\s*\([^)]*\)\s*;?\s*$", re.MULTILINE)


def _strip_creator_inplace(p: Path) -> None:
    if p.suffix not in (".cpp", ".cc", ".cxx") or not p.exists():
        return
    txt = p.read_text(encoding="utf-8", errors="replace")
    stripped = _CREATOR_RE.sub("", txt)
    if stripped != txt:
        p.write_text(stripped, encoding="utf-8")

_THIS = Path(__file__).resolve().parent           # .../opgen/layer_oracle
_OPGEN = _THIS.parent                              # .../opgen


def _find_kernelgen(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "ncnn").is_dir():
            return p
    return start.parents[2]


_KERNELGEN = _find_kernelgen(_THIS)                # .../kernelgen


# ---------------------------------------------------------------------------
# bin protocol: [int32 ndim][int32 dims...][float32 data]  (matches the runner)
# ---------------------------------------------------------------------------
def write_bin(path: str | Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr.astype(np.float32))
    with open(path, "wb") as f:
        f.write(struct.pack("<i", arr.ndim))
        f.write(struct.pack(f"<{arr.ndim}i", *arr.shape))
        f.write(arr.tobytes())


def read_bin(path: str | Path) -> np.ndarray:
    with open(path, "rb") as f:
        ndim = struct.unpack("<i", f.read(4))[0]
        dims = struct.unpack(f"<{ndim}i", f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(dims)


def torch_to_ncnn_input(arr: np.ndarray) -> np.ndarray:
    """Drop the leading batch dim so the array is in NCNN per-sample layout.

    PyTorch (N,C,H,W)->(C,H,W); (N,C,L)->(C,L); (N,C)->(C,); (N,)->(,) scalar-ish.
    NCNN works on a single sample; batch is not represented in ncnn::Mat.

    NOTE: This is the OLD heuristic. It assumes every input has a leading batch
    dim — wrong when inputs include weights/biases (Conv has in1=[16,3,3,3] as
    a weight, not a batched tensor). Prefer pnnx_driven_ncnn_inputs() when the
    pnnx-emitted _ncnn.py is available; that mirrors pnnx's per-blob squeeze
    decision exactly.
    """
    if arr.ndim >= 2:
        return np.ascontiguousarray(arr[0])
    return np.ascontiguousarray(arr)


_EX_INPUT_RE = re.compile(
    r'ex\.input\(\s*"([^"]+)"\s*,\s*ncnn\.Mat\(\s*([a-zA-Z_]\w*)\s*'
    r'(?:\.squeeze\((\d+)\))?\s*\.numpy\(\)\s*\)\s*\.clone\(\)\s*\)'
)


def parse_pnnx_input_squeeze(ncnn_py_path: str | Path) -> dict[str, int | None]:
    """Return {blob_name: squeeze_axis or None} parsed from pnnx-emitted _ncnn.py.

    pnnx WRITES this file alongside .ncnn.param/.ncnn.bin — it is the
    authoritative source for how each input should be reshaped before being
    fed to ncnn (None = feed raw, int = squeeze that axis). Mirroring this
    eliminates all per-op driver heuristics (Conv weight vs batched data,
    Gemm constants, etc.).

    Returns an empty dict if the file is missing or contains no ex.input lines.
    """
    p = Path(ncnn_py_path)
    if not p.exists():
        return {}
    out: dict[str, int | None] = {}
    for m in _EX_INPUT_RE.finditer(p.read_text("utf-8", errors="replace")):
        blob, _var, ax = m.group(1), m.group(2), m.group(3)
        out[blob] = int(ax) if ax is not None else None
    return out


def pnnx_driven_ncnn_inputs(torch_inputs, in_names: list[str],
                            ncnn_py_path: str | Path | None) -> list[np.ndarray]:
    """Build the ncnn feed array list from torch inputs using pnnx's own policy.

    For each (blob_name, torch_tensor) pair, look up the squeeze axis in the
    pnnx-emitted _ncnn.py: None → feed as-is; int N → squeeze axis N (only if
    that axis is size 1, else leave raw). When no policy is available for a
    blob (rare: name missing from _ncnn.py, or the file absent entirely) we
    fall back to torch_to_ncnn_input's "drop axis 0 if rank>=2" heuristic —
    matches the old behavior for legacy cases.

    `torch_inputs` may be torch tensors or numpy arrays; either works.
    """
    policy = parse_pnnx_input_squeeze(ncnn_py_path) if ncnn_py_path else {}
    out: list[np.ndarray] = []
    for name, t in zip(in_names, torch_inputs):
        arr = t.detach().numpy() if hasattr(t, "detach") else np.asarray(t)
        if name in policy:
            ax = policy[name]
            if ax is not None and arr.ndim > ax and arr.shape[ax] == 1:
                arr = arr.squeeze(ax)
        else:
            arr = torch_to_ncnn_input(arr)
        out.append(np.ascontiguousarray(arr))
    return out


# ---------------------------------------------------------------------------
@dataclass
class OracleResult:
    ok: bool
    outputs: list[np.ndarray] = field(default_factory=list)
    return_code: int = 0
    compile_log: str = ""
    run_log: str = ""
    runner: str = ""
    error: str = ""
    skipped: bool = False   # vulkan: no GPU device available -> treat as skipped, not fail

    # filled by verify()
    passed: bool | None = None
    max_diff: float | None = None
    mean_diff: float | None = None
    detail: str = ""
    failure_category: str = ""   # diagnosis-conditioned feedback label (see failure_taxonomy)


class LayerOracle:
    def __init__(
        self,
        ncnn_root: str | Path | None = None,
        build_lib: str | Path | None = None,
        runner_src: str | Path | None = None,
        cxx: str = "g++",
        workdir: str | Path | None = None,
    ) -> None:
        self.ncnn_root = Path(ncnn_root) if ncnn_root else (_KERNELGEN / "ncnn")
        self.build_lib = Path(build_lib) if build_lib else (self.ncnn_root / "build_lib")
        self.runner_src = Path(runner_src) if runner_src else (_THIS / "layer_oracle_runner.cpp")
        self.cxx = cxx
        self.workdir = Path(workdir) if workdir else (_OPGEN / "runs" / "_oracle")
        self.workdir.mkdir(parents=True, exist_ok=True)

    # --- prerequisites -----------------------------------------------------
    @property
    def libncnn(self) -> Path:
        return self.build_lib / "src" / "libncnn.a"

    def _ensure_libncnn(self) -> None:
        if self.libncnn.exists() and (self.build_lib / "src" / "platform.h").exists():
            return
        raise FileNotFoundError(
            f"libncnn.a not found at {self.libncnn}. Build it once with:\n"
            f"  cmake -S {self.ncnn_root} -B {self.build_lib} "
            f"-DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=OFF "
            f"-DNCNN_BUILD_BENCHMARK=OFF -DNCNN_VULKAN=OFF -DNCNN_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release\n"
            f"  cmake --build {self.build_lib} -j"
        )

    # --- compile (cached by candidate mtime + class) -----------------------
    def compile(self, candidate_cpp: str | Path, class_name: str, header: str,
                extra_sources: Sequence[str | Path] = (),
                extra_includes: Sequence[str | Path] = ()) -> tuple[Path, str]:
        """Compile runner + candidate.cpp [+ extra_sources] against libncnn.a.

        For arm backend kernels (which subclass the base layer), pass the verified
        base .cpp via `extra_sources` (its symbols aren't in the standalone
        libncnn) and `src/layer/arm` via `extra_includes` (for neon_mathfun.h etc).
        NEON is baseline on arm64 — no `-march` needed.
        """
        self._ensure_libncnn()
        candidate_cpp = Path(candidate_cpp).resolve()
        extra_src = [Path(s).resolve() for s in extra_sources]
        # strip dead-code creators (avoid duplicate symbol vs an installed libncnn)
        for _p in [candidate_cpp, *extra_src]:
            _strip_creator_inplace(_p)
        runner = self.workdir / f"runner_{class_name}"
        # rebuild if exe missing or any input newer than exe
        inputs_mtime = [candidate_cpp.stat().st_mtime, self.runner_src.stat().st_mtime]
        inputs_mtime += [s.stat().st_mtime for s in extra_src if s.exists()]
        newest = max(inputs_mtime)
        if runner.exists() and runner.stat().st_mtime >= newest:
            return runner, "(cached)"

        cmd = [
            self.cxx, "-std=c++11", "-O2",
            "-I", str(candidate_cpp.parent),  # so the runner finds the candidate's own header
            "-I", str(self.ncnn_root / "src"),
            "-I", str(self.ncnn_root / "src" / "layer"),
            "-I", str(self.build_lib / "src"),
        ]
        for inc in extra_includes:
            cmd += ["-I", str(inc)]
        cmd += [str(self.runner_src), str(candidate_cpp)]
        cmd += [str(s) for s in extra_src]
        cmd += [
            str(self.libncnn),
            f'-DCANDIDATE_HEADER="{header}"',
            f"-DCANDIDATE_CLASS={class_name}",
            "-o", str(runner),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = " ".join(cmd) + "\n" + proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"runner compile failed:\n{log}")
        return runner, log

    # --- run ---------------------------------------------------------------
    def run(
        self,
        *,
        candidate_cpp: str | Path,
        class_name: str,
        header: str,
        params: dict[int, Any] | None = None,
        inputs: Sequence[np.ndarray],
        weights: Sequence[np.ndarray] = (),
        extra_sources: Sequence[str | Path] = (),
        extra_includes: Sequence[str | Path] = (),
        packing: int = 0,
    ) -> OracleResult:
        try:
            runner, clog = self.compile(candidate_cpp, class_name, header,
                                        extra_sources=extra_sources, extra_includes=extra_includes)
        except Exception as exc:  # noqa: BLE001
            return OracleResult(ok=False, error=str(exc), compile_log=str(exc))

        wd = self.workdir / class_name
        wd.mkdir(parents=True, exist_ok=True)

        argv = [str(runner)]
        if params:
            argv += ["--param", ",".join(self._fmt_param(k, v) for k, v in params.items())]
        for i, x in enumerate(inputs):
            p = wd / f"in{i}.bin"
            write_bin(p, np.asarray(x))
            argv += ["--input", str(p)]
        for i, w in enumerate(weights):
            p = wd / f"w{i}.bin"
            write_bin(p, np.asarray(w).reshape(-1))
            argv += ["--weight", str(p)]
        out_path = wd / "out.bin"
        argv += ["--out", str(out_path)]
        if packing > 0:
            argv += ["--packing", str(packing)]

        proc = subprocess.run(argv, capture_output=True, text=True)
        run_log = " ".join(argv) + "\n" + proc.stdout + proc.stderr
        if proc.returncode != 0 or not out_path.exists():
            return OracleResult(ok=False, return_code=proc.returncode, compile_log=clog,
                                run_log=run_log, runner=str(runner),
                                error=f"runner failed (rc={proc.returncode})")
        out = read_bin(out_path)
        return OracleResult(ok=True, outputs=[out], return_code=0, compile_log=clog,
                            run_log=run_log, runner=str(runner))

    # --- verify vs reference (oracle) --------------------------------------
    def verify(
        self,
        *,
        candidate_cpp: str | Path,
        class_name: str,
        header: str,
        params: dict[int, Any] | None,
        inputs: Sequence[np.ndarray],
        reference: np.ndarray,
        weights: Sequence[np.ndarray] = (),
        tol: float = 1e-3,
        extra_sources: Sequence[str | Path] = (),
        extra_includes: Sequence[str | Path] = (),
        packing: int = 0,
        backend: str = "base",
    ) -> OracleResult:
        res = self.run(candidate_cpp=candidate_cpp, class_name=class_name, header=header,
                       params=params, inputs=inputs, weights=weights,
                       extra_sources=extra_sources, extra_includes=extra_includes, packing=packing)
        if not res.ok:
            res.passed = False
            res.detail = "runner did not produce output"
            return res
        out = res.outputs[0]
        ref = np.asarray(reference, dtype=np.float32)
        try:
            out_r = out.reshape(ref.shape)
        except ValueError:
            # shape/element-count mismatch -> taxonomy (E3 wrong-count / E4 permuted)
            cat, det = classify_failure(out, ref, tol,
                                        input=(inputs[0] if len(inputs) else None), backend=backend)
            res.passed = False
            res.failure_category = cat
            res.detail = f"[{cat}] {det}"
            return res
        diff = np.abs(out_r - ref)
        res.max_diff = float(diff.max())
        res.mean_diff = float(diff.mean())
        res.passed = bool(np.allclose(out_r, ref, atol=tol, rtol=tol))
        if res.passed:
            res.detail = f"max_diff={res.max_diff:.6f} mean_diff={res.mean_diff:.6f} tol={tol}"
        else:
            # diagnosis-conditioned feedback (E4/E5/E6/E8/instability; backend-aware)
            cat, det = classify_failure(out, ref, tol,
                                        input=(inputs[0] if len(inputs) else None), backend=backend)
            res.failure_category = cat
            res.detail = f"[{cat}] {det}"
        return res

    @staticmethod
    def _fmt_param(key: int, value: Any) -> str:
        # ncnn array params use the "negative key" trick: -23310=N,v1,v2,...
        # where 23300 is added to the real key id and the count comes first.
        # We pass arrays through verbatim so layers like Convolution `10=pads_array`
        # work; layers expecting a SCALAR but handed a list would crash here, so
        # the typical use is when the dictionary's default for that param is itself
        # an array. Caller (KernelAgent) is responsible for matching the layer.
        #
        # CRITICAL: floats MUST keep a decimal point in their string form. The
        # runner's parse_params() decides int-vs-float by looking for '.'/'e'/'E';
        # a Python `1.0` formatted as "1" would parse as int, get stored as int
        # in ParamDict, and any kernel that reads it via `pd.get(id, 1.f)` would
        # bit-cast 0x00000001 to float -> denormal ~1.4e-45 silently. So we use
        # `:.8g` for the digits but force a trailing ".0" when needed.
        def _float_str(v: float) -> str:
            if v != v:  # NaN
                return "0"
            s = f"{v:.8g}"
            # Anything :.8g rendered without '.', 'e', or 'E' is integer-looking
            # and will be misread by the runner; pin a ".0" so it parses as float.
            if "." not in s and "e" not in s and "E" not in s:
                s += ".0"
            return s
        if isinstance(value, (list, tuple)):
            arr_key = -(23300 + int(key))
            elems = ",".join(f"{int(v)}" if not isinstance(v, float)
                             else _float_str(v) for v in value)
            return f"{arr_key}={len(value)},{elems}"
        if isinstance(value, float):
            return f"{key}={_float_str(value)}"
        return f"{key}={int(value)}"
