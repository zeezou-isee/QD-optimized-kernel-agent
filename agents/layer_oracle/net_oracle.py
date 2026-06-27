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

from .oracle import read_bin, write_bin, _find_repo_root, _default_ncnn_root

_THIS = Path(__file__).resolve().parent           # .../agents/layer_oracle
_OPGEN = _THIS.parent                              # .../agents
_REPO_ROOT = _find_repo_root(_THIS)                # .../KernelAgent
_NCNN_ROOT = _default_ncnn_root(_REPO_ROOT)        # .../frameworks/ncnn


@dataclass
class InstallHandle:
    created_files: list[str] = field(default_factory=list)
    cmake_original: str = ""
    cmake_path: str = ""


@dataclass
class NativeOverrideHandle:
    """Records every source-tree mutation made by install_native_override so the
    tree can be returned to its pristine native state.

    overwritten : path -> original text   (rewrite on restore)
    created     : paths that did not exist before (delete on restore)
    parked_arch : (original_path, parked_path)  arch-opt variants moved aside
    reconfigured: whether a cmake configure (not just build) is needed on restore
    """
    overwritten: dict[str, str] = field(default_factory=dict)
    created: list[str] = field(default_factory=list)
    parked_arch: list[tuple[str, str]] = field(default_factory=list)
    reconfigured: bool = False

    def to_dict(self) -> dict:
        return {"overwritten": list(self.overwritten), "created": self.created,
                "parked_arch": self.parked_arch, "reconfigured": self.reconfigured}


# Layer types that are structural/IO and never represent "the op's compute".
_NON_COMPUTE_LAYERS = {"Input", "Output", "Split", "Noop"}


def detect_native_layer(ncnn_param_text: str, ncnn_root: str | Path) -> str | None:
    """If the converted graph maps the op to exactly ONE native ncnn layer that
    has a base src/layer/<name>.cpp, return that layer's class name; else None.

    Multi-layer decompositions (e.g. LogSoftmax -> Softmax + UnaryOp) return None
    — there is no single .cpp to overwrite, so native-override does not apply.
    """
    types: list[str] = []
    for ln in ncnn_param_text.splitlines()[2:]:
        parts = ln.split()
        if parts and parts[0] not in _NON_COMPUTE_LAYERS:
            types.append(parts[0])
    uniq = set(types)
    if len(uniq) != 1:
        return None
    cls = next(iter(uniq))
    base_cpp = Path(ncnn_root) / "src" / "layer" / f"{cls.lower()}.cpp"
    return cls if base_cpp.exists() else None


def _infer_class_name(code_book: dict[str, str]) -> str:
    """Best-effort: find `class Xxx : public ...Layer...` in the code book."""
    for content in code_book.values():
        m = re.search(r"class\s+(\w+)\s*:\s*public\s+\w*Layer", content or "")
        if m:
            return m.group(1)
    return ""


def rewrite_class_name(code: str, old_class: str, new_class: str) -> str:
    """Rename a C++ class identifier (whole-token only) from old_class to new_class.

    Also rewrites the conventional include-guard macro (CAND_X_H -> X_H style) and
    the lower-cased header filename references, so the overwritten native files
    compile under the native class name.
    """
    out = re.sub(rf"\b{re.escape(old_class)}\b", new_class, code)
    # include guards: derive from class names (e.g. Cand_Softmax -> CAND_SOFTMAX...)
    old_guard = re.sub(r"[^A-Za-z0-9]", "_", old_class).upper()
    new_guard = re.sub(r"[^A-Za-z0-9]", "_", new_class).upper()
    out = re.sub(rf"\b{re.escape(old_guard)}_H\b", f"{new_guard}_H", out)
    # header includes: cand_softmax.h -> softmax.h
    out = out.replace(f"{old_class.lower()}.h", f"{new_class.lower()}.h")
    return out



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
        self.ncnn_root = Path(ncnn_root) if ncnn_root else _NCNN_ROOT
        if build_lib:
            self.build_lib = Path(build_lib)
        else:
            # Prefer a dedicated build_lib/, fall back to an existing build/.
            cand = self.ncnn_root / "build_lib"
            self.build_lib = cand if cand.exists() else (self.ncnn_root / "build")
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

    def reconfigure_and_rebuild(self, jobs: int = 8) -> tuple[bool, str]:
        """Configure (re-run cmake -S -B) THEN build. Required after adding/removing
        layer source files (e.g. parking an arch-opt variant), because ncnn_add_layer
        decides which sources/registry slots exist at configure time."""
        cfg = subprocess.run(
            ["cmake", "-S", str(self.ncnn_root), "-B", str(self.build_lib)],
            capture_output=True, text=True)
        log = "$ cmake -S -B\n" + cfg.stdout + cfg.stderr
        if cfg.returncode != 0:
            return False, log
        ok, blog = self.rebuild_libncnn(jobs)
        return ok, log + "\n$ cmake --build\n" + blog

    # --- native-override (方案C): overwrite a native layer in place ----------
    def install_native_override(self, code_book: dict[str, str], native_class: str,
                                cand_class: str = "") -> NativeOverrideHandle:
        """Overwrite the native src/layer/<name>.{h,cpp} with the agent's kernel,
        renaming the agent class to the native class so the existing
        ncnn_add_layer(<native_class>) registers OUR code under the native type.

        Also parks every arch-opt variant (layer/<arch>/<name>_<arch>.*) aside, so
        the registry's arch slot falls back to our overwritten base class at runtime
        (create_layer_cpu: arch creator == 0 -> base creator).

        cand_class: the agent's original class (e.g. Cand_Softmax). If empty it is
        inferred from the code so we can rewrite it to native_class.
        """
        name = native_class.lower()
        layer_dir = self.ncnn_root / "src" / "layer"
        h = NativeOverrideHandle()

        if not cand_class:
            cand_class = _infer_class_name(code_book) or f"Cand_{native_class}"

        # 1) overwrite base .h/.cpp (rename class -> native), backing up originals
        for fname, content in code_book.items():
            if not fname.endswith((".h", ".hpp", ".cpp", ".cc", ".cxx")):
                continue
            # normalize destination name to the native stem (softmax.h / softmax.cpp)
            suffix = ".h" if fname.endswith((".h", ".hpp")) else ".cpp"
            dst = layer_dir / f"{name}{suffix}"
            rewritten = rewrite_class_name(content, cand_class, native_class)
            # built-in layers must not define their own creator (ncnn_add_layer does)
            if suffix == ".cpp":
                rewritten = re.sub(r"^\s*DEFINE_LAYER_CREATOR\s*\([^)]*\)\s*;?\s*$", "",
                                   rewritten, flags=re.MULTILINE)
            if dst.exists():
                h.overwritten[str(dst)] = dst.read_text(encoding="utf-8")
            else:
                h.created.append(str(dst))
            dst.write_text(rewritten, encoding="utf-8")

        # 2) park arch-opt variants so the base (our) class is what runs
        for arch_file in sorted(layer_dir.glob(f"*/{name}_*")):
            if not arch_file.is_file():
                continue
            parked = arch_file.with_suffix(arch_file.suffix + ".ka_parked")
            arch_file.rename(parked)
            h.parked_arch.append((str(arch_file), str(parked)))
        h.reconfigured = bool(h.parked_arch) or bool(h.created)
        return h

    def restore_native_override(self, h: NativeOverrideHandle) -> list[str]:
        """Undo install_native_override: move parked arch files back, restore/delete
        overwritten base files. Best-effort: every step runs even if a prior one
        raised; returns a list of error strings (empty == clean)."""
        errors: list[str] = []
        for orig, parked in h.parked_arch:
            try:
                if Path(parked).exists():
                    Path(parked).rename(orig)
            except OSError as e:
                errors.append(f"arch restore {orig}: {e}")
        for path, original in h.overwritten.items():
            try:
                Path(path).write_text(original, encoding="utf-8")
            except OSError as e:
                errors.append(f"overwrite restore {path}: {e}")
        for path in h.created:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as e:
                errors.append(f"created delete {path}: {e}")
        return errors

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
