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

from .oracle import read_bin, write_bin, _find_kernelgen

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

    def rebuild_libncnn(self, jobs: int = 8) -> tuple[bool, str]:
        proc = subprocess.run(["cmake", "--build", str(self.build_lib), "-j", str(jobs)],
                              capture_output=True, text=True)
        return proc.returncode == 0 and self.libncnn.exists(), proc.stdout + proc.stderr

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
