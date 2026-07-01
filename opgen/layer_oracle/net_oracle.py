"""NetOracle — end-to-end numeric verification of a graph conversion.

For a genuinely-new operator, the converted .ncnn.param references a custom layer
(e.g. Cand_Greater) that pip's pyncnn does not have. So we:
  1. install the verified kernel into ncnn/src/layer + ncnn_add_layer(Class),
  2. rebuild libncnn.a (build_lib),
  3. run the converted model via a generic Net runner linked to that libncnn.a,
  4. compare to PyTorch (allclose).

install_layer/restore track the source-tree mutation so the tree is left clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess

import numpy as np

from .oracle import read_bin, write_bin, _find_kernelgen, _run_cmake_bounded

_THIS = Path(__file__).resolve().parent           # .../opgen/layer_oracle
_OPGEN = _THIS.parent                              # .../opgen
_KERNELGEN = _find_kernelgen(_THIS)               # .../kernelgen


@dataclass
class InstallHandle:
    created_files: list[str] = field(default_factory=list)
    cmake_original: str = ""
    cmake_path: str = ""


def parse_ncnn_io(param_text: str) -> tuple[list[str], str]:
    """Return (input_blob_names, output_blob_name) of a pnnx .ncnn.param.

    Input layers ('Input') produce the input blobs; the graph output is the
    output blob of the last layer line.
    """
    lines = [ln for ln in param_text.splitlines() if ln.strip()]
    inputs: list[str] = []
    output = "out0"
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        ltype, _name, nin, nout = parts[0], parts[1], int(parts[2]), int(parts[3])
        blobs = parts[4:]
        out_blobs = blobs[nin:nin + nout]
        if ltype == "Input" and out_blobs:
            inputs.append(out_blobs[0])
        elif out_blobs:
            output = out_blobs[0]
    return inputs, output


def retarget_param_layer(param_text: str, src_type: str, dst_type: str,
                         layer_name: str | None = None) -> tuple[str, int]:
    """Re-point .ncnn.param layer lines of type `src_type` to `dst_type`.

    Why: for an op ncnn already supports (e.g. Sigmoid), the PNNX conversion emits
    the native layer type, so a Net/benchncnn run uses ncnn's BUILT-IN impl — not
    the candidate we installed under a distinct name (Cand_Sigmoid). Rewriting the
    layer TYPE token to `Cand_Sigmoid` makes the very same model resolve to OUR
    implementation, while ncnn's built-in stays untouched (no source deletion).

    Only the first whitespace-token (the layer TYPE) of matching layer lines is
    replaced; blob names, the layer name, params, and all whitespace are preserved.
    The two header lines (magic, "<layers> <blobs>") are never touched. If
    `layer_name` is given, only the layer with that NAME (2nd token) is retargeted.

    The new type MUST accept the same param-ids / inputs / outputs as the old one —
    which holds because Cand_<Op> is a re-implementation of the same op.

    Returns (new_param_text, n_replaced).
    """
    lines = param_text.splitlines()
    if len(lines) < 2:
        return param_text, 0
    out = lines[:2]                       # preserve magic + "<layers> <blobs>"
    n = 0
    for ln in lines[2:]:
        if not ln.strip():
            out.append(ln)
            continue
        # split into: leading ws | type | sep-ws | name | rest
        m = re.match(r"(\s*)(\S+)(\s+)(\S+)(.*)", ln)
        if m and m.group(2) == src_type and (layer_name is None or m.group(4) == layer_name):
            out.append(m.group(1) + dst_type + m.group(3) + m.group(4) + m.group(5))
            n += 1
        else:
            out.append(ln)
    new_text = "\n".join(out)
    if param_text.endswith("\n"):
        new_text += "\n"
    return new_text, n


def retarget_param_output_layer(param_text: str, dst_type: str,
                                expected_src_type: str | None = None) -> tuple[str, int]:
    """Rewrite the TYPE of the layer that produces the graph's final output blob.

    For a single-op reference model (the dataset ops) this layer IS the op under
    test, so this retargets it to our `dst_type` (Cand_<Op>) regardless of its
    native ncnn type — robust where the native name is NOT derivable from the task
    name (e.g. torch.exp converts to ncnn 'UnaryOp', not 'Exp'; torch.gt -> a
    'BinaryOp'/custom). Only the TYPE token is changed; everything else preserved.

    Idempotent: if the output layer is already `dst_type` (the new-op path, where
    GraphAgent forced the target), this rewrites cls->cls (text unchanged).

    DECOMPOSED-OP GUARD (`expected_src_type`): some ops do NOT map to a single
    ncnn layer — pnnx decomposes them into a chain of native layers, all correct
    on their own (e.g. `alpha*(A@B)+beta*C` -> Gemm + BinaryOp(mul) + BinaryOp(add),
    where the OUTPUT layer is the final Add, not our Gemm). Blindly retargeting the
    output layer to a monolithic `Cand_<Op>` mis-wires it (wrong input count) and
    produces garbage/NaN. When `expected_src_type` is given, we only retarget if the
    output layer's current type matches it (case-insensitive) OR is already
    `dst_type`; otherwise the op is a native multi-layer decomposition and we SKIP
    (return the text unchanged, n=0) so the caller runs the correct baseline graph.

    Returns (new_param_text, n) where n is 1 if a layer was rewritten else 0.
    """
    _, out_name = parse_ncnn_io(param_text)
    lines = param_text.splitlines()
    if len(lines) < 3:
        return param_text, 0
    target = None
    for i in range(2, len(lines)):
        parts = lines[i].split()
        if len(parts) < 4 or parts[0] == "Input":
            continue
        nin, nout = int(parts[2]), int(parts[3])
        out_blobs = parts[4:][nin:nin + nout]
        if out_name in out_blobs:
            target = i           # keep the LAST layer producing the output blob
    if target is None:
        return param_text, 0
    m = re.match(r"(\s*)(\S+)(\s+)(\S+)(.*)", lines[target])
    if not m:
        return param_text, 0
    cur_type = m.group(2)
    # Decide whether this output layer is actually the op we built a Cand for.
    # Retarget only when: (a) it is already our Cand (new-op path, idempotent), or
    # (b) its native type matches the detected principal layer (single-native-op).
    # Otherwise the op is a native multi-layer DECOMPOSITION (e.g. Gemm_alpha ->
    # Gemm + BinaryOp*2, output layer = Add) OR the baseline probe found no single
    # principal layer at all (expected_src_type is None for exactly that reason) —
    # substituting a monolithic Cand there mis-wires the layer (wrong input count
    # -> NaN). In both cases, skip and let the correct baseline native graph run.
    if cur_type != dst_type:
        if expected_src_type:
            if cur_type.lower() != expected_src_type.strip().lower():
                return param_text, 0        # different native type -> decomposed
        else:
            return param_text, 0            # no principal layer detected -> decomposed
    lines[target] = m.group(1) + dst_type + m.group(3) + m.group(4) + m.group(5)
    new_text = "\n".join(lines)
    if param_text.endswith("\n"):
        new_text += "\n"
    return new_text, 1


def retarget_param_output_file(src_path: str | Path, dst_path: str | Path,
                               dst_type: str,
                               expected_src_type: str | None = None) -> int:
    """Read a .ncnn.param, retarget its output-producing layer to `dst_type`,
    write to dst_path. Returns 1 if a layer was rewritten else 0. Idempotent.

    `expected_src_type` gates the retarget for decomposed ops — see
    retarget_param_output_layer. When it doesn't match, the file is copied
    through unchanged (n=0) so the baseline native graph runs."""
    text = Path(src_path).read_text(encoding="utf-8")
    new_text, n = retarget_param_output_layer(text, dst_type, expected_src_type)
    Path(dst_path).write_text(new_text, encoding="utf-8")
    return n


def retarget_param_file(src_path: str | Path, dst_path: str | Path,
                        src_type: str, dst_type: str,
                        layer_name: str | None = None) -> int:
    """Read a .ncnn.param, retarget `src_type`->`dst_type`, write to dst_path.

    Returns the number of layer lines rewritten (0 means nothing matched).
    """
    text = Path(src_path).read_text(encoding="utf-8")
    new_text, n = retarget_param_layer(text, src_type, dst_type, layer_name=layer_name)
    Path(dst_path).write_text(new_text, encoding="utf-8")
    return n


class NetOracle:
    def __init__(self, ncnn_root: str | Path | None = None, build_lib: str | Path | None = None,
                 runner_src: str | Path | None = None, cxx: str = "g++",
                 workdir: str | Path | None = None) -> None:
        self.ncnn_root = Path(ncnn_root) if ncnn_root else (_KERNELGEN / "ncnn")
        self.build_lib = Path(build_lib) if build_lib else (self.ncnn_root / "build_lib")
        self.runner_src = Path(runner_src) if runner_src else (_THIS / "net_oracle_runner.cpp")
        self.cxx = cxx
        self.workdir = Path(workdir) if workdir else (_OPGEN / "runs" / "_net")
        self.workdir.mkdir(parents=True, exist_ok=True)

    @property
    def libncnn(self) -> Path:
        return self.build_lib / "src" / "libncnn.a"

    # --- install / restore -------------------------------------------------
    def install_layer(self, code_book: dict[str, str], class_name: str,
                      subdir: str = "", add_cmake: bool = True) -> InstallHandle:
        """Install layer files into src/layer[/subdir] and (optionally) patch CMake.

        base   -> subdir="",    add_cmake=True  (ncnn_add_layer(<BaseClass>))
        arm    -> subdir="arm", add_cmake=False (the macro auto-discovers
                  src/layer/arm/<name>_arm.cpp from the base ncnn_add_layer call)
        """
        layer_dir = self.ncnn_root / "src" / "layer"
        if subdir:
            layer_dir = layer_dir / subdir
        layer_dir.mkdir(parents=True, exist_ok=True)
        h = InstallHandle()
        for name, content in code_book.items():
            # Built-in layers must NOT define their own creator — ncnn_add_layer
            # generates it. Strip DEFINE_LAYER_CREATOR(...) to avoid duplicate symbol.
            if name.endswith((".cpp", ".cc", ".cxx")):
                content = re.sub(r"^\s*DEFINE_LAYER_CREATOR\s*\([^)]*\)\s*;?\s*$", "",
                                 content, flags=re.MULTILINE)
            dst = layer_dir / name
            h.created_files.append(str(dst))
            dst.write_text(content, encoding="utf-8")
        if add_cmake:
            cmake = self.ncnn_root / "src" / "CMakeLists.txt"
            text = cmake.read_text(encoding="utf-8")
            h.cmake_path = str(cmake)
            h.cmake_original = text
            call = f"ncnn_add_layer({class_name})"
            if call not in text:
                idx = text.find("ncnn_add_layer(")
                text = text[:idx] + call + "\n" + text[idx:]
                cmake.write_text(text, encoding="utf-8")
        return h

    def restore(self, h: InstallHandle) -> None:
        if h.cmake_path and h.cmake_original:
            Path(h.cmake_path).write_text(h.cmake_original, encoding="utf-8")
        for f in h.created_files:
            try:
                Path(f).unlink(missing_ok=True)
            except OSError:
                pass

    def rebuild_libncnn(self, jobs: int = 8, timeout: int = 600) -> tuple[bool, str]:
        # start_new_session: if our parent gets SIGTERM (e.g. batch timeout), the
        # ncnn-tree guard signal handler propagates here; without an own session,
        # cmake/make/g++ grandchildren survive and corrupt the shared build dir
        # while the next op starts. Own session + bounded timeout = safe teardown.
        rc, out = _run_cmake_bounded(
            ["cmake", "--build", str(self.build_lib), "-j", str(jobs)],
            timeout=timeout,
        )
        return rc == 0 and self.libncnn.exists(), out

    # --- compile + run -----------------------------------------------------
    def compile_runner(self) -> tuple[Path, str]:
        runner = self.workdir / "net_runner"
        cmd = [self.cxx, "-std=c++11", "-O2",
               "-I", str(self.ncnn_root / "src"), "-I", str(self.build_lib / "src"),
               str(self.runner_src), str(self.libncnn), "-o", str(runner)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("net_runner compile failed:\n" + proc.stdout + proc.stderr)
        return runner, "ok"

    def run_net(self, param: str | Path, binf: str | Path,
                inputs: dict[str, np.ndarray], out_name: str) -> tuple[np.ndarray | None, str]:
        runner, _ = self.compile_runner()
        wd = self.workdir / "io"
        wd.mkdir(parents=True, exist_ok=True)
        argv = [str(runner), "--param", str(param), "--bin", str(binf), "--out", out_name,
                "--outfile", str(wd / "out.bin")]
        for name, arr in inputs.items():
            p = wd / f"{name}.bin"
            write_bin(p, np.asarray(arr))
            argv += ["--in", f"{name}={p}"]
        proc = subprocess.run(argv, capture_output=True, text=True)
        log = " ".join(argv) + "\n" + proc.stdout + proc.stderr
        outp = wd / "out.bin"
        if proc.returncode != 0 or not outp.exists():
            return None, log
        return read_bin(outp), log
