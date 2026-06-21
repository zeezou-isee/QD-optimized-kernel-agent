"""Functional steps of the ncnn graph-conversion pipeline.

Each function is small and independently callable so it can be unit-tested
without the full agent loop:

    extract_code_blocks   parse LLM response into {repo_path: code}
    retrieve_examples     pull similar existing pass files for the coder prompt
    inject_files          write new pass files + patch the two CMakeLists (with backup)
    restore_files         undo an injection (delete new files, revert CMake)
    build_pnnx            cmake configure + build (incremental)
    make_pt               trace a PyTorch reference model -> .pt + inputshape
    run_conversion        run the pnnx binary -> .pnnx.param / .ncnn.param / .bin
    verify_structural     parse params: op matched? target ncnn layer present?
    verify_numeric        ctest the end-to-end allclose test

The agent loop (graph_agent.py) just orchestrates these.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import re
import shutil
import sys

from config import GraphConfig
from graph_schemas import BackupHandle, OpProfile

# Reuse the project's shell tool (honours EndtoEndMobilekernelAgent/tools).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.bash_exec import bash_exec  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Code extraction
# ---------------------------------------------------------------------------
_PASS_DIRS = ("pass_ncnn", "pass_level1", "pass_level2")

# matches a path line like "pass_ncnn/F_myop.cpp" or "tests/ncnn/test_F_myop.py"
_PATH_RE = re.compile(r"((?:pass_ncnn|pass_level1|pass_level2|tests/ncnn)/[A-Za-z0-9_./]+\.(?:cpp|h|py))")
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+]*)\s*\n(.*?)```", re.DOTALL)


def extract_code_blocks(response: str) -> dict[str, str]:
    """Parse fenced code blocks into {repo_relative_path: code}.

    The coder is instructed to start each block with the destination repo path,
    either on the line *before* the fence or as the first line *inside* it.
    """
    code: dict[str, str] = {}

    # form A: path line immediately before a fenced block
    for m in re.finditer(r"(?P<path>" + _PATH_RE.pattern + r")\s*\n```(?:[a-zA-Z0-9_+]*)\s*\n(?P<body>.*?)```", response, re.DOTALL):
        code[_norm_path(m.group("path"))] = m.group("body").strip() + "\n"

    # form B: path as first line inside the fence
    for m in _FENCE_RE.finditer(response):
        body = m.group(1)
        lines = body.splitlines()
        if not lines:
            continue
        first = lines[0].strip().lstrip("/* ").rstrip(" */")
        hit = _PATH_RE.search(first)
        if hit:
            rel = _norm_path(hit.group(1))
            if rel not in code:
                code[rel] = "\n".join(lines[1:]).strip() + "\n"

    return code


def _norm_path(p: str) -> str:
    return p.strip().lstrip("./")


# ---------------------------------------------------------------------------
# 2. Example retrieval (feed the coder analog passes to imitate)
# ---------------------------------------------------------------------------
def parse_pnnx_op_types(pnnx_param_text: str) -> list[str]:
    """Return the high-level op types in a .pnnx.param (excluding I/O nodes).

    e.g. ['nn.LayerNorm'] or ['F.layer_norm'] or ['aten::layer_norm'] — these are
    exactly what a pass_ncnn/pass_level2 match_pattern_graph must match.
    """
    types: list[str] = []
    for line in pnnx_param_text.splitlines()[2:]:
        parts = line.split()
        if not parts:
            continue
        t = parts[0]
        if t in ("pnnx.Input", "pnnx.Output"):
            continue
        if t not in types:
            types.append(t)
    return types


def probe_pnnx_ir(cfg: GraphConfig, model_py: str | Path, run_dir: Path, task_name: str) -> dict:
    """Run the CURRENT (unmodified) pnnx on the model to capture ground-truth IR.

    Gives the coder the real op signature it must match instead of guessing.
    Returns {pnnx_param, ncnn_param, op_types, residual_aten}.
    """
    probe_dir = Path(run_dir) / "_probe"
    out: dict = {"pnnx_param": "", "ncnn_param": "", "op_types": [], "residual_aten": []}
    ok, pt, ishape, log = make_pt(cfg, model_py, probe_dir)
    if not ok:
        out["error"] = "trace failed:\n" + log
        return out
    if not cfg.pnnx_bin.exists():
        out["error"] = "pnnx binary missing; build it first."
        return out
    _cok, artifacts, _clog = run_conversion(cfg, pt, ishape, probe_dir, task_name)
    if ".pnnx.param" in artifacts and Path(artifacts[".pnnx.param"]).exists():
        out["pnnx_param"] = Path(artifacts[".pnnx.param"]).read_text(encoding="utf-8", errors="replace")
    if ".ncnn.param" in artifacts and Path(artifacts[".ncnn.param"]).exists():
        out["ncnn_param"] = Path(artifacts[".ncnn.param"]).read_text(encoding="utf-8", errors="replace")
    out["op_types"] = parse_pnnx_op_types(out["pnnx_param"])
    out["residual_aten"] = sorted({t for t in re.findall(r"\b(aten::[A-Za-z0-9_]+|prim::[A-Za-z0-9_]+)", out["pnnx_param"])})
    # Does the CURRENT pnnx already convert this op correctly (structurally +
    # numerically)? If so, the op is already supported and the agent should not
    # author new passes (it would only risk breaking a working baseline).
    out["baseline_structural_ok"] = bool(out["ncnn_param"]) and not out["residual_aten"] \
        and len(_ncnn_layer_types(out["ncnn_param"]) - {"Input", "Output", "Split"}) > 0
    if out["baseline_structural_ok"]:
        out["baseline_numeric_ok"] = baseline_numeric(cfg, model_py, probe_dir, Path(pt).stem)
    else:
        out["baseline_numeric_ok"] = None
    out["baseline_supported"] = out["baseline_structural_ok"] and out["baseline_numeric_ok"] is True
    return out


_BASELINE_NUM_DRIVER = '''
import importlib.util, numpy as np, torch, ncnn

spec = importlib.util.spec_from_file_location("ref_model", r"{model_py}")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
net = (mod.Model(*init) if init else mod.Model()).eval()
inputs = mod.get_inputs()
if len(inputs) != 1:
    print("RESULT=SKIP multi-input"); raise SystemExit
inp = inputs[0]
with torch.no_grad():
    ref = net(inp)
if isinstance(ref, (tuple, list)):
    ref = ref[0]
ref = ref.detach().numpy()

a = inp.detach().numpy()
nd = a.ndim
mat_in = a[0] if nd >= 2 else a   # drop batch -> (C,H,W)/(H,W)/(C,)
with ncnn.Net() as n:
    n.load_param(r"{param}"); n.load_model(r"{bin}")
    with n.create_extractor() as ex:
        ex.input("in0", ncnn.Mat(np.ascontiguousarray(mat_in)).clone())
        ret, o = ex.extract("out0")
out = np.array(o).reshape(ref.shape)
ok = np.allclose(out, ref, atol=1e-3, rtol=1e-3)
print("RESULT=" + ("PASS" if ok else "FAIL"),
      "maxdiff=%.5f" % float(np.abs(out - ref).max()))
'''


def baseline_numeric(cfg: GraphConfig, model_py: str | Path, probe_dir: Path, stem: str) -> bool | None:
    """Best-effort allclose of baseline ncnn vs torch (single-input models).

    Returns True/False, or None if it couldn't run (multi-IO / error).
    """
    param = probe_dir / f"{stem}.ncnn.param"
    binf = probe_dir / f"{stem}.ncnn.bin"
    if not param.exists() or not binf.exists():
        return None
    driver = probe_dir / "_baseline_num.py"
    driver.write_text(_BASELINE_NUM_DRIVER.format(
        model_py=str(Path(model_py).resolve()), param=str(param), bin=str(binf)), encoding="utf-8")
    res = bash_exec(f"{sys.executable} {driver}", timeout=120000, cwd=str(probe_dir))
    out = res.get("stdout", "")
    if "RESULT=PASS" in out:
        return True
    if "RESULT=FAIL" in out:
        return False
    return None


def retrieve_examples(cfg: GraphConfig, profile: OpProfile, op_types: list[str] | None = None,
                      residual_aten: list[str] | None = None, max_each: int = 2) -> dict[str, str]:
    """Return {repo_path: file_text} of similar existing passes to imitate.

    Branches by complexity (see report 3.6):
      - residual aten:: / Expression / multi-node => seed with IMPERATIVE convert_*
        examples (convert_torch_cat etc.), so the coder learns the procedural style.
      - otherwise => pattern-style nearest neighbours.
    """
    out: dict[str, str] = {}

    needs_imperative = bool(residual_aten) or any(
        (t or "").startswith(("aten::", "prim::")) or t == "pnnx.Expression"
        for t in (op_types or [])
    )
    if needs_imperative:
        for stem in ("convert_torch_cat", "convert_Tensor_slice", "convert_torch_split"):
            for ext in (".cpp", ".h"):
                f = cfg.pass_ncnn_dir / f"{stem}{ext}"
                rel = f"pass_ncnn/{stem}{ext}"
                if f.exists() and rel not in out:
                    out[rel] = f.read_text(encoding="utf-8", errors="replace")

    # nearest-neighbour: existing passes that already match the real op type(s)
    if op_types:
        for base in (cfg.pass_ncnn_dir, cfg.pass_level2_dir, cfg.pass_level1_dir):
            if not base.exists():
                continue
            for f in sorted(base.glob("*.cpp")):
                rel = f"{base.name}/{f.name}"
                if rel in out:
                    continue
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if any(t in text for t in op_types):
                    out[rel] = text

    names = list(profile.analog_ops)
    # always include the canonical unary example as a fallback
    names += ["F_hardsigmoid", "F_relu6", "F_elu"]

    dirs = {
        "pass_ncnn": cfg.pass_ncnn_dir,
        "pass_level2": cfg.pass_level2_dir,
        "pass_level1": cfg.pass_level1_dir,
    }
    for kind, base in dirs.items():
        # pass_level1 modules are named nn_* (FuseModulePass); give a couple of
        # canonical examples so the coder uses the right header + base class.
        candidates = list(names)
        if kind == "pass_level1":
            candidates += ["nn_LayerNorm", "nn_Hardsigmoid", "nn_GroupNorm"]
        picked = 0
        for name in candidates:
            f = base / f"{name}.cpp"
            rel = f"{kind}/{name}.cpp"
            if f.exists() and rel not in out:
                out[rel] = f.read_text(encoding="utf-8", errors="replace")
                picked += 1
            if picked >= max_each:
                break
    return out


# ---------------------------------------------------------------------------
# 3. Inject files + patch CMakeLists  (with backup for clean restore)
# ---------------------------------------------------------------------------
def inject_files(cfg: GraphConfig, code_book: dict[str, str], session: BackupHandle | None = None) -> tuple[bool, BackupHandle, str]:
    """Write the new pass/test files and patch the CMakeLists.

    ``session`` accumulates the original tree state across rounds so the agent
    can restore once at the end. Cleanup is the caller's responsibility (this
    function does not auto-restore), so an incremental build can keep the tree
    between rounds.
    """
    backup = session if session is not None else BackupHandle()
    errors: list[str] = []

    if not code_book:
        return False, backup, "No code blocks were extracted from the response."

    for rel, content in code_book.items():
        kind = rel.split("/", 1)[0]
        if kind in _PASS_DIRS:
            ok, err = _write_pass_file(cfg, rel, content, backup)
        elif rel.startswith("tests/ncnn/"):
            ok, err = _write_test_file(cfg, rel, content, backup)
        else:
            ok, err = False, f"Unsupported destination path: {rel}"
        if not ok:
            errors.append(err)

    if errors:
        return False, backup, "\n".join(errors)
    return True, backup, ""


def _track_original(path: Path, backup: BackupHandle) -> None:
    """Record a file's pre-mutation state exactly once into the session backup."""
    key = str(path)
    if key in backup.created_files or key in backup.modified_files:
        return
    if path.exists():
        backup.modified_files[key] = path.read_text(encoding="utf-8", errors="replace")
    else:
        backup.created_files.append(key)


def _write_pass_file(cfg: GraphConfig, rel: str, content: str, backup: BackupHandle) -> tuple[bool, str]:
    dst = cfg.pnnx_src / rel
    _track_original(dst, backup)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")

    if dst.suffix == ".cpp":
        ok, err = _patch_src_cmake(cfg, rel, backup)
        if not ok:
            return False, err
        # imperative pass_ncnn convert_*.cpp also needs include + call in pass_ncnn.cpp
        if rel.startswith("pass_ncnn/convert_") and _is_imperative_pass(content):
            return _patch_pass_ncnn_dispatcher(cfg, rel, content, backup)
    # headers under pass_ncnn/convert_* are required for imperative passes; just write them
    return True, ""


def _is_imperative_pass(content: str) -> bool:
    """Detect 'void convert_X(Graph& g)' style (vs class GraphRewriterPass)."""
    return bool(re.search(r"\bvoid\s+convert_\w+\s*\(\s*Graph\s*&", content))


def _extract_imperative_fn(content: str) -> str | None:
    m = re.search(r"\bvoid\s+(convert_\w+)\s*\(\s*Graph\s*&", content)
    return m.group(1) if m else None


def _patch_pass_ncnn_dispatcher(cfg: GraphConfig, rel: str, content: str, backup: BackupHandle) -> tuple[bool, str]:
    """Add #include + call site in pass_ncnn.cpp so the imperative pass actually runs.

    Anchors:
      - include block:  right after `#include "pass_ncnn/convert_Tensor_slice_copy.h"`
      - call site:      right after `ncnn::convert_Tensor_slice_copy(g);` in pass_ncnn(Graph&...)
    Both are stable anchors that ship with current pnnx.
    """
    fn = _extract_imperative_fn(content)
    if not fn:
        return True, ""  # not imperative; nothing to do
    header_rel = rel.replace(".cpp", ".h")
    dispatcher = cfg.pnnx_src / "pass_ncnn.cpp"
    if not dispatcher.exists():
        return False, f"pass_ncnn.cpp not found at {dispatcher}"
    _track_original(dispatcher, backup)  # snapshot BEFORE we patch
    text = dispatcher.read_text(encoding="utf-8")
    include_line = f'#include "{header_rel}"\n'
    call_line = f"    ncnn::{fn}(g);\n"
    changed = False

    if include_line not in text:
        anchor = '#include "pass_ncnn/convert_Tensor_slice_copy.h"\n'
        if anchor not in text:
            return False, "include anchor not found in pass_ncnn.cpp"
        text = text.replace(anchor, anchor + include_line, 1); changed = True

    if call_line not in text:
        anchor = "    ncnn::convert_Tensor_slice_copy(g);\n"
        if anchor not in text:
            return False, "call-site anchor not found in pass_ncnn.cpp"
        text = text.replace(anchor, anchor + call_line, 1); changed = True

    if changed:
        dispatcher.write_text(text, encoding="utf-8")
    return True, ""


def _patch_src_cmake(cfg: GraphConfig, rel: str, backup: BackupHandle) -> tuple[bool, str]:
    """Insert ``    pass_xxx/File.cpp`` into the matching set(...) list."""
    kind = rel.split("/", 1)[0]
    var = cfg.CMAKE_SRC_VAR.get(kind)
    if not var:
        return False, f"No CMake SRCS variable for {kind}"

    cmake = cfg.src_cmake
    text = cmake.read_text(encoding="utf-8")
    if rel in text:
        return True, ""  # already registered

    set_line = f"set({var}"
    idx = text.find(set_line)
    if idx < 0:
        return False, f"Could not find '{set_line}' in {cmake}"
    insert_at = text.find("\n", idx) + 1  # right after the set(... line

    _track_original(cmake, backup)
    new_text = text[:insert_at] + f"    {rel}\n" + text[insert_at:]
    cmake.write_text(new_text, encoding="utf-8")
    return True, ""


def _write_test_file(cfg: GraphConfig, rel: str, content: str, backup: BackupHandle) -> tuple[bool, str]:
    name = Path(rel).stem  # test_F_myop
    dst = cfg.tests_ncnn_dir / Path(rel).name
    _track_original(dst, backup)
    dst.write_text(content, encoding="utf-8")

    # register pnnx_ncnn_add_test(<short>) where short = name without leading test_
    short = name[len("test_"):] if name.startswith("test_") else name
    cmake = cfg.tests_ncnn_cmake
    text = cmake.read_text(encoding="utf-8")
    call = f"pnnx_ncnn_add_test({short})"
    if call in text:
        return True, ""
    _track_original(cmake, backup)
    # append after the last existing add_test call
    last = text.rfind("pnnx_ncnn_add_test(")
    if last < 0:
        new_text = text.rstrip() + f"\n{call}\n"
    else:
        eol = text.find("\n", last) + 1
        new_text = text[:eol] + f"{call}\n" + text[eol:]
    cmake.write_text(new_text, encoding="utf-8")
    return True, ""


def restore_files(cfg: GraphConfig, backup: BackupHandle) -> None:
    """Undo an injection: delete created files, revert modified ones."""
    for path, original in backup.modified_files.items():
        try:
            Path(path).write_text(original, encoding="utf-8")
        except OSError:
            pass
    for path in backup.created_files:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 4. Build pnnx
# ---------------------------------------------------------------------------
def build_pnnx(cfg: GraphConfig, log_path: Path | None = None) -> tuple[bool, str]:
    build = cfg.pnnx_build
    build.mkdir(parents=True, exist_ok=True)

    configure = f"cmake -S {cfg.pnnx_src.parent} -B {build}"
    if cfg.torch_install_dir:
        configure += f" -DTorch_INSTALL_DIR={cfg.torch_install_dir}"
    cfg_res = bash_exec(configure, timeout=600000, cwd=str(cfg.pnnx_dir))
    build_res = bash_exec(
        f"cmake --build {build} -j{cfg.build_jobs}",
        timeout=600000,
        cwd=str(cfg.pnnx_dir),
    )
    warm = {}
    if build_res.get("success") and cfg.pnnx_bin.exists():
        # macOS kills a freshly linked binary on its first run (code-signature
        # verification) with SIGKILL/137. Absorb that one-time kill here so the
        # real conversion call succeeds.
        warm = bash_exec(f"{cfg.pnnx_bin} >/dev/null 2>&1 || true", timeout=60000, cwd=str(cfg.pnnx_dir))

    log = _join_logs(["[configure]", cfg_res], ["[build]", build_res], ["[warmup]", warm])
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(log, encoding="utf-8")

    ok = build_res.get("success", False) and cfg.pnnx_bin.exists()
    return ok, log


_KERNELGEN_ROOT = str(Path(__file__).resolve().parent.parent)
_FUNC_DIR = str(Path(_KERNELGEN_ROOT) / "func")


def _load_error_locator():
    """Return func/Repo_error_location.extract_compilation_errors, or None.

    That script gives tree-sitter-based, per-file error grouping with the
    enclosing code snippet (line-numbered) — far richer than a raw log tail.
    """
    try:
        for p in (_KERNELGEN_ROOT, _FUNC_DIR):
            if p not in sys.path:
                sys.path.insert(0, p)
        from Repo_error_location import extract_compilation_errors  # type: ignore
        return extract_compilation_errors
    except Exception:
        return None


def locate_build_errors(log: str, opname: str) -> str:
    """Rich compile-error feedback via the error-location script (with fallback)."""
    fn = _load_error_locator()
    if fn is None:
        return extract_build_errors(log, opname)
    try:
        d = fn(log, opname)
    except Exception:
        return extract_build_errors(log, opname)

    blocks: list[str] = []

    def _fmt_group(label: str, group: dict) -> None:
        for _key, e in group.items():
            head = f"[{label}]"
            if e.get("error_file"):
                head += f" file: {e['error_file']}"
            msgs = [f"  line {e[k.replace('message', 'line')]}: {e[k]}"
                    for k in e if k.startswith("error_message_")]
            ctx = e.get("error_context", "")
            blocks.append(head + "\n" + "\n".join(msgs) + (f"\n--- code ---\n{ctx}" if ctx else ""))

    _fmt_group("error in generated file", d.get("inplace_error", {}))
    _fmt_group("error in another file", d.get("crossfile_error", {}))
    others = d.get("other_error", [])
    if others:
        blocks.append("[linker/make/cmake]\n" + "\n".join(f"  {o['type']}: {o['message']}" for o in others))

    text = "\n\n".join(b for b in blocks if b.strip())
    # If the locator found nothing parseable, fall back to the raw tail.
    return text if text.strip() else extract_build_errors(log, opname)


def extract_build_errors(log: str, focus: str | None = None) -> str:
    """Pull compiler-error lines (+ context), optionally focused on a file."""
    lines = log.splitlines()
    markers = ("error:", "fatal error:", "undefined reference", "ld:", "FAILED:", "Error ")
    keep: list[str] = []
    for i, line in enumerate(lines):
        low = line.lower()
        if any(m.lower() in low for m in markers) or (focus and focus.lower() in low):
            keep.extend(lines[max(0, i - 2): i + 4])
    if not keep:
        return "\n".join(lines[-80:])
    # dedup consecutive
    out, last = [], None
    for ln in keep:
        if ln != last:
            out.append(ln)
        last = ln
    return "\n".join(out[-160:])


# ---------------------------------------------------------------------------
# 5. Trace the reference model + run conversion
# ---------------------------------------------------------------------------
_TRACE_DRIVER = '''
import importlib.util, sys, json
import torch

spec = importlib.util.spec_from_file_location("ref_model", r"{model_py}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

init_inputs = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
net = mod.Model(*init_inputs) if init_inputs else mod.Model()
net = net.eval()

inputs = mod.get_inputs()
with torch.no_grad():
    traced = torch.jit.trace(net, tuple(inputs))
traced.save(r"{pt_out}")

# pnnx inputshape needs the per-input dtype suffix; defaulting to f32 makes pnnx
# abort on bool/int ops (where/logical_*/bitwise_*/comparisons consuming bool).
_DT = {{"torch.float32":"f32","torch.float":"f32","torch.float16":"f16","torch.float64":"f64",
        "torch.bool":"bool","torch.uint8":"u8","torch.int8":"i8","torch.int16":"i16",
        "torch.int32":"i32","torch.int":"i32","torch.int64":"i64","torch.long":"i64"}}
shapes = []
for t in inputs:
    suffix = _DT.get(str(t.dtype), "f32")
    shapes.append("[" + ",".join(str(int(d)) for d in t.shape) + "]" + suffix)
print("INPUTSHAPE=" + ",".join(shapes))
'''


def make_pt(cfg: GraphConfig, model_py: str | Path, run_dir: Path) -> tuple[bool, str, str, str]:
    """Trace the dataset reference model to TorchScript.

    Returns (ok, pt_path, inputshape, log).
    """
    run_dir = Path(run_dir).resolve()  # absolute so cwd=run_dir doesn't double the path
    run_dir.mkdir(parents=True, exist_ok=True)
    pt_out = run_dir / f"{Path(model_py).stem}.pt"
    driver = run_dir / "_trace.py"
    driver.write_text(_TRACE_DRIVER.format(model_py=str(Path(model_py).resolve()), pt_out=str(pt_out)), encoding="utf-8")

    res = bash_exec(f"{sys.executable} {driver}", timeout=300000, cwd=str(run_dir))
    log = res.get("stdout", "") + "\n" + res.get("stderr", "")
    if not res.get("success") or not pt_out.exists():
        return False, "", "", log
    m = re.search(r"INPUTSHAPE=(.+)", res.get("stdout", ""))
    if not m:
        return False, str(pt_out), "", log
    return True, str(pt_out), m.group(1).strip(), log


def annotate_convert_log(log: str) -> str:
    """Append actionable hints for common pnnx convert-time runtime crashes.

    The conversion stage fails as a C++ exception (no file/line), so we pattern
    match the message and tell the coder the likely cause + fix.
    """
    hints: list[str] = []
    low = log.lower()
    if "map::at" in low or "out_of_range" in low:
        # find which pass stage crashed (last "#### pass_xxx" before the crash)
        stage = ""
        for line in log.splitlines():
            if line.strip().startswith("#####") and "pass_" in line:
                stage = line.strip().strip("# ").strip()
        hints.append(
            "HINT: 'map::at: key not found' means a captured_params.at(K) / "
            "captured_attrs.at(K) in your pass used a key K that is NOT declared "
            "in match_pattern_graph()."
            + (f" It crashed during {stage}." if stage else "")
            + " Fix: every params key must appear as `name=%K` in the pattern; "
            "every attr key must be captured (e.g. `@weight` -> captured_attrs.at(\"op_0.weight\")). "
            "Check exact capture names match the .at() keys."
        )
    if "find_node_by_kind" in low or "no member named" in low:
        hints.append("HINT: use the GraphRewriterPass match_pattern_graph/%capture API, "
                     "not graph traversal helpers.")
    if not hints:
        return log
    return log + "\n\n" + "\n".join(hints)


def run_conversion(cfg: GraphConfig, pt_path: str, inputshape: str, run_dir: Path, task_name: str) -> tuple[bool, dict[str, str], str]:
    """Run the pnnx binary; collect .pnnx.param / .ncnn.param / .ncnn.bin."""
    if not cfg.pnnx_bin.exists():
        return False, {}, f"pnnx binary not found: {cfg.pnnx_bin}"

    run_dir = Path(run_dir).resolve()
    pt_path = str(Path(pt_path).resolve())
    stem = Path(pt_path).stem
    cmd = (
        f"{cfg.pnnx_bin} {pt_path} inputshape={inputshape} "
        f"pnnxparam={stem}.pnnx.param pnnxbin={stem}.pnnx.bin "
        f"ncnnparam={stem}.ncnn.param ncnnbin={stem}.ncnn.bin "
        f"pnnxpy={stem}_pnnx.py ncnnpy={stem}_ncnn.py"
    )
    res = bash_exec(cmd, timeout=300000, cwd=str(run_dir))
    # retry once if macOS SIGKILL'd a freshly built binary (137)
    if res.get("exit_code") == 137:
        res = bash_exec(cmd, timeout=300000, cwd=str(run_dir))
    log = res.get("stdout", "") + "\n" + res.get("stderr", "")

    artifacts = {}
    for suffix in (".pnnx.param", ".ncnn.param", ".ncnn.bin"):
        p = run_dir / f"{stem}{suffix}"
        if p.exists():
            artifacts[suffix] = str(p)
    ok = ".ncnn.param" in artifacts
    if not ok:
        log = annotate_convert_log(log)
    return ok, artifacts, log


# ---------------------------------------------------------------------------
# 6. Verify structural (param inspection) — independent of the ncnn kernel
# ---------------------------------------------------------------------------
def verify_structural(cfg: GraphConfig, profile: OpProfile, artifacts: dict[str, str], convert_log: str = "") -> tuple[bool, str]:
    notes: list[str] = []

    pnnx_param = artifacts.get(".pnnx.param")
    ncnn_param = artifacts.get(".ncnn.param")
    if not ncnn_param:
        return False, "No .ncnn.param produced (conversion did not reach ncnn save stage)."

    pnnx_text = Path(pnnx_param).read_text(encoding="utf-8", errors="replace") if pnnx_param and Path(pnnx_param).exists() else ""
    ncnn_text = Path(ncnn_param).read_text(encoding="utf-8", errors="replace")

    # (a) no torch-domain residue for our op in the pnnx IR
    residue = sorted({tok for tok in re.findall(r"\b(aten::[A-Za-z0-9_]+|prim::[A-Za-z0-9_]+)", pnnx_text)})
    if residue:
        notes.append(f"PNNX IR still contains unconverted torch ops: {residue} "
                     f"(=> pass_level1/2 did not capture them).")

    # (b) the target ncnn layer type appears in the final ncnn graph
    ncnn_types = _ncnn_layer_types(ncnn_text)
    target = profile.target_ncnn_layer.strip()
    target_ok = (not target) or (target in ncnn_types)
    if target and not target_ok:
        notes.append(f"Target ncnn layer '{target}' NOT found in .ncnn.param. "
                     f"Present layer types: {sorted(ncnn_types)} (=> pass_ncnn type_str/match wrong).")

    # surface pnnx warnings that hint at unsupported ops
    for ln in convert_log.splitlines():
        low = ln.lower()
        if "unsupported" in low or "no rewrite" in low or "fallback" in low:
            notes.append(f"pnnx warning: {ln.strip()}")

    ok = (not residue) and target_ok
    if ok and not notes:
        notes.append(f"Structural check passed: target layer '{target or '(any)'}' present, no torch residue.")
    return ok, "\n".join(notes)


def _ncnn_layer_types(ncnn_param_text: str) -> set[str]:
    """ncnn .param: line1 magic, line2 'layer_count blob_count', then 'Type name ...'."""
    types: set[str] = set()
    lines = ncnn_param_text.splitlines()
    for ln in lines[2:]:
        parts = ln.split()
        if parts:
            types.add(parts[0])
    return types


# ---------------------------------------------------------------------------
# 7. Verify numeric (end-to-end allclose via ctest) — needs the ncnn kernel
# ---------------------------------------------------------------------------
def verify_numeric(cfg: GraphConfig, test_short_name: str, log_path: Path | None = None) -> tuple[bool, str]:
    """Run ctest for the injected end-to-end test (PyTorch vs ncnn allclose)."""
    res = bash_exec(
        f"ctest -R test_ncnn_{test_short_name} -V",
        timeout=600000,
        cwd=str(cfg.pnnx_build),
    )
    log = res.get("stdout", "") + "\n" + res.get("stderr", "")
    if log_path:
        log_path.write_text(log, encoding="utf-8")
    return res.get("success", False), log


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _join_logs(*sections: list) -> str:
    out: list[str] = []
    for sec in sections:
        title, res = sec[0], sec[1]
        out.append(title)
        out.append(f"  cmd: {res.get('command', '')}")
        out.append(f"  exit: {res.get('exit_code')}")
        if res.get("stdout"):
            out.append(res["stdout"])
        if res.get("stderr"):
            out.append(res["stderr"])
    return "\n".join(out)
