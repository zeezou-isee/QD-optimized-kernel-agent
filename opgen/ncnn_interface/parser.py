"""Parse a single ncnn layer pair (<op>.h + <op>.cpp) into a structured record.

Source of truth = the .h declares the class + forward overloads; the .cpp's
`load_param` defines the param-ID → variable name → default mapping; `load_model`
defines the weight load order. Plus two boolean flags (one_blob_only,
support_inplace) set in the constructor or load_param.

This module is REGEX-based and intentionally simple. It is meant to be
correct for ~95% of the 111 ncnn layers; anything it cannot understand goes
into `parse_warnings` rather than throwing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---- regex -----------------------------------------------------------------

# class XXX : public Layer    (or : public InnerProduct etc. — we want the first)
_RE_CLASS_DECL = re.compile(
    r"\bclass\s+(\w+)\s*:\s*public\s+(\w+)\b"
)

# virtual int forward(...) const;   (or forward_inplace)
_RE_FORWARD = re.compile(
    r"^\s*virtual\s+int\s+(forward(?:_inplace)?)\s*\(([^)]*)\)\s*const\s*;",
    re.MULTILINE,
)

# int <ClassName>::load_param(const ParamDict& pd) {  ...  }
# Also handles ctor (no return type): ClassName::ClassName().
def _slice_function(text: str, fn_name: str, class_name: str) -> str | None:
    """Return the body of `[int ] ClassName::fn_name(...)`, or None if not found.

    Constructor has no return type (`ClassName::ClassName()`) so we make the
    `int ` prefix optional.
    """
    sig = re.compile(
        rf"^(?:int\s+)?{re.escape(class_name)}::{re.escape(fn_name)}\s*\([^)]*\)\s*$",
        re.MULTILINE,
    )
    m = sig.search(text)
    if not m:
        return None
    # the function body is the brace-matched block starting at the next `{`
    open_idx = text.find("{", m.end())
    if open_idx < 0:
        return None
    depth, i = 0, open_idx
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx:i + 1]
        i += 1
    return None


# var = pd.get(<id>, <default>)
# default may be: 0, 1, 0.f, -FLT_MAX, Mat(), kernel_w, ...
_RE_PD_GET = re.compile(
    r"(\w+)\s*=\s*pd\.get\(\s*(\d+)\s*,\s*(.+?)\s*\)\s*;",
)

# var = mb.load(<size_expr>, <flag>)
_RE_MB_LOAD = re.compile(
    r"(\w+)\s*=\s*mb\.load\(\s*([^,]+?)\s*,\s*(-?\d+)\s*\)\s*;",
)

# one_blob_only = true | false
_RE_FLAG = re.compile(
    r"(one_blob_only|support_inplace)\s*=\s*(true|false)\b"
)

# heuristic: a default expression is "a variable" if it matches a bare C identifier
# AND is not a known C++ literal/expression.
_LITERAL_HINTS = ("FLT_MAX", "INT_MAX", "Mat(", "0.f", "1.f", "-1.f")
_RE_BARE_IDENT = re.compile(r"^[A-Za-z_]\w*$")


def _default_is_variable(default_expr: str) -> bool:
    s = default_expr.strip()
    if not s:
        return False
    if any(h in s for h in _LITERAL_HINTS):
        return False
    if s.startswith("-"):
        s = s[1:].lstrip()
    # pure integer/float?
    try:
        int(s, 0)        # also handles 0x...
        return False
    except ValueError:
        pass
    try:
        float(s)
        return False
    except ValueError:
        pass
    # what's left: bare identifier == probably another param var
    return bool(_RE_BARE_IDENT.match(s))


# ---- conditional detection (mb.load wrapped in `if (xxx)`) -----------------

# Detect the simplest, most common pattern:
#
#     if (bias_term)
#     {
#         bias_data = mb.load(num_output, 1);
#         ...
#     }
#
# i.e. the mb.load's containing block opens immediately after a single-line
# `if (<cond>)`. Anything more complex (nested if, #ifdef wrapped, mb.load on
# the same line as if) → return None, and the human reviewer reads the source.
_RE_IF_ON_LINE = re.compile(r"^\s*if\s*\(([^)]+)\)\s*$")


def _find_conditional_for_mb_load(body: str, match_start: int) -> str | None:
    """Return the condition of the nearest `if (...) {` block whose `{` opens
    directly before the mb.load and has NOT been closed yet at this position.

    Algorithm: split body into lines once; find the line containing match_start;
    walk backward through the preceding lines, tracking brace balance. The first
    time the running balance goes from 0 to -1 (we exited a `{` while walking
    back), check whether the two lines above that `{` are exactly an
    `if (<cond>)` — if so, that's our condition.
    """
    lines = body.splitlines()
    # locate which line index contains match_start
    cur = 0
    pos = 0
    for cur, ln in enumerate(lines):
        if pos + len(ln) + 1 > match_start:
            break
        pos += len(ln) + 1   # +1 for the '\n'
    # walk backward from the line BEFORE the mb.load
    depth = 0
    for j in range(cur - 1, -1, -1):
        ln = lines[j]
        # update balance: opens minus closes (we're walking backward so this is "uncount")
        depth += ln.count("}") - ln.count("{")
        if depth < 0:
            # we just walked out of an enclosing `{` block; the line that had
            # the `{` is `lines[j]` (or one of the preceding ones if `{` is
            # on its own line). Look at the line immediately before `lines[j]`.
            # Common ncnn style is:
            #     if (bias_term)
            #     {
            #         bias_data = mb.load(...);
            #
            # so we want lines[j-1] (the `if (...)` line). But if `{` is on the
            # same line as `if`, lines[j] itself contains the if.
            for k in (j, j - 1):
                if 0 <= k < len(lines):
                    m = _RE_IF_ON_LINE.match(lines[k])
                    if m:
                        return m.group(1).strip()
            return None
    return None


# ---- main dataclass --------------------------------------------------------

@dataclass
class ParseResult:
    name: str = ""
    header: str = ""
    source: str = ""
    base_class: str = ""
    forward_signatures: list[str] = field(default_factory=list)
    one_blob_only_default: bool = False    # ncnn::Layer base default
    support_inplace_default: bool = False
    params: list[dict[str, Any]] = field(default_factory=list)
    weights_load_order: list[dict[str, Any]] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- the entry point -------------------------------------------------------

def parse_layer(header_path: Path, source_path: Path | None = None) -> ParseResult:
    """Parse a (.h, .cpp) pair into a ParseResult. .cpp is optional (some layers
    are header-only) — when absent, params/weights stay empty.

    Heuristics, not a real C++ parser. We optimise for the common cases ncnn
    uses (a few dozen lines of pd.get and mb.load); exotic patterns get noted
    in `parse_warnings` so the operator pops up in the human-review pass."""
    r = ParseResult()
    r.header = header_path.name
    r.source = source_path.name if source_path and source_path.exists() else ""

    header_txt = header_path.read_text(encoding="utf-8", errors="replace")

    # class name + base class
    m = _RE_CLASS_DECL.search(header_txt)
    if not m:
        r.parse_warnings.append("could not find `class X : public Y` declaration")
        return r
    r.name, r.base_class = m.group(1), m.group(2)

    # forward signatures (collapse to single-line form)
    for fm in _RE_FORWARD.finditer(header_txt):
        fn, args = fm.group(1), " ".join(fm.group(2).split())
        r.forward_signatures.append(f"int {fn}({args}) const")

    if not source_path or not source_path.exists():
        r.parse_warnings.append("no .cpp file; load_param/load_model unparseable")
        return r

    src_txt = source_path.read_text(encoding="utf-8", errors="replace")

    # ctor (set defaults BEFORE load_param) — easiest to find via class name + '::'
    ctor_body = _slice_function(src_txt, r.name, r.name)
    if ctor_body:
        for fm in _RE_FLAG.finditer(ctor_body):
            setattr(r, f"{fm.group(1)}_default", fm.group(2) == "true")

    # load_param → params list
    lp_body = _slice_function(src_txt, "load_param", r.name)
    if lp_body:
        seen_ids = set()
        for pm in _RE_PD_GET.finditer(lp_body):
            var, pid_s, default = pm.group(1), pm.group(2), pm.group(3).strip()
            pid = int(pid_s)
            if pid in seen_ids:
                r.parse_warnings.append(f"duplicate pd.get id={pid} (var={var}); kept first")
                continue
            seen_ids.add(pid)
            r.params.append({
                "id": pid,
                "name": var,
                "default": default,
                "default_is_var": _default_is_variable(default),
            })
        # also scan for additional one_blob_only/support_inplace assignments
        # inside load_param (BinaryOp does this conditionally)
        for fm in _RE_FLAG.finditer(lp_body):
            # don't override the ctor's value; just note conditional behavior
            r.parse_warnings.append(
                f"load_param may toggle {fm.group(1)} to {fm.group(2)} "
                f"depending on params (see source)"
            )
    # else: no own load_param — common case for parameterless ops that inherit
    # the base class's empty default (AbsVal/Sigmoid/Mish/...). Silent.

    # load_model → weights_load_order
    lm_body = _slice_function(src_txt, "load_model", r.name)
    if lm_body:
        for idx, mm in enumerate(_RE_MB_LOAD.finditer(lm_body)):
            var, size_expr, flag = mm.group(1), mm.group(2).strip(), int(mm.group(3))
            cond = _find_conditional_for_mb_load(lm_body, mm.start())
            entry = {
                "index": idx,
                "var": var,
                "size_expr": size_expr,
                "flag": flag,
            }
            if cond:
                entry["conditional"] = cond
            r.weights_load_order.append(entry)
    elif source_path.exists():
        # not all layers need load_model (no weights) — only warn if .cpp existed
        # but didn't define it; an absent function is "no weights"
        pass

    # params sort by id for stable output
    r.params.sort(key=lambda p: p["id"])
    return r
