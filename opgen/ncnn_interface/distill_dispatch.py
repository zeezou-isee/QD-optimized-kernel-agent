"""Distill ncnn's convolution DISPATCH heuristics into machine-readable rules.

The "皇冠明珠" of the design doc (§8.4): ncnn's expert if-else that picks which
algorithm family (winograd / im2col-gemm / direct / depthwise) wins for a given
(kernel, stride, dilation, channel) shape. We do NOT re-derive it — we extract
ncnn's ACTUAL gate conditions verbatim (with repo:file:line cites) plus the key
numeric thresholds, so the QD niche prior can rank/prune algo_family cells
faithfully to the source (no guessing; `--diff` flags drift on ncnn upgrades).

Scope: only the family-SELECTION gates (which algo_family), NOT the dozens of
direct micro-variants (conv3x3s2 vs conv5x5s1 …) — those all map to `direct` and
add noise to an algo_family-granularity prior. Rich dispatch lives in three arm
files; everything else has a trivial/absent prior (loader falls back to no-op).

Reads (per --ncnn-root):
  src/layer/arm/convolution_arm.cpp            (fp32 gates)
  src/layer/arm/convolution_arm_asimdhp.cpp    (fp16 gates — different thresholds)
  src/layer/arm/convolutiondepthwise_arm.cpp   (→ dw family)
  src/layer/arm/deconvolution_arm.cpp          (gemm vs direct)

Writes under experience_pool/backend_ncnn/:
  dispatch_heuristics.json  — machine-readable; policy/dispatch.py consumes this
  dispatch_heuristics.md    — human-readable review surface

Usage:
  python -m opgen.ncnn_interface.distill_dispatch \
    --ncnn-root /Users/xingze/Documents/project/kernelgen/ncnn \
    [--out-dir experience_pool/backend_ncnn] \
    [--diff old_dispatch_heuristics.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJ / "experience_pool" / "backend_ncnn"


# ---------------------------------------------------------------------------
def _lineno(text: str, pos: int) -> int:
    """1-based line number of character offset `pos`."""
    return text.count("\n", 0, pos) + 1


def _cite(rel: str, text: str, pos: int) -> str:
    return f"ncnn/{rel}:{_lineno(text, pos)}"


# --- convolution (fp32 + fp16 share structure, differ in thresholds) --------
def _parse_convolution(text: str, rel: str, dtype: str) -> tuple[list[dict], list[str]]:
    """Extract the four family-selection gates from a convolution_arm*.cpp body."""
    rules: list[dict] = []
    warn: list[str] = []

    # 1) WINOGRAD gate: prefer_winograd channel threshold + the 3x3/s1/d1 gate.
    m_pref = re.search(
        r"bool prefer_winograd = .*?\(\s*num_input >= (\d+)\s*\|\|\s*num_output >= (\d+)\s*\)",
        text)
    m_gate = re.search(
        r"if \([^\n]*use_winograd_convolution && prefer_winograd"
        r" && kernel_w == (\d+) && kernel_h == (\d+)"
        r" && dilation_w == (\d+) && dilation_h == (\d+)"
        r" && stride_w == (\d+) && stride_h == (\d+)\)",
        text)
    if m_pref and m_gate:
        ch = int(m_pref.group(1))
        k, d, s = int(m_gate.group(1)), int(m_gate.group(3)), int(m_gate.group(5))
        # winograd tile caps (63 upper cap, 43 lower min) — audit only
        m63 = re.search(r"use_winograd63_convolution && \(num_input <= (\d+) && num_output <= (\d+)\)", text)
        m43 = re.search(r"use_winograd43_convolution && \(num_input >= (\d+) && num_output >= (\d+)\)", text)
        tiles = {}
        if m63:
            tiles["winograd63_channel_max"] = int(m63.group(1))
        if m43:
            tiles["winograd43_channel_min"] = int(m43.group(1))
        rules.append({
            "algo_family": "winograd", "priority": 1, "dtype": dtype,
            "when": f"kernel=={k}x{k} && stride=={s} && dilation=={d} && "
                    f"(num_input>={ch} || num_output>={ch})",
            "verbatim": m_gate.group(0)[:240],
            "predicate": {"kernel": k, "stride": s, "dilation": d, "channel_any_ge": ch},
            "tiles": tiles or None,
            "ncnn_fn": "conv3x3s1_winograd{23,43,63}",
            "cite": _cite(rel, text, m_gate.start()),
        })
    else:
        warn.append(f"[{rel}] winograd gate anchor not found (ncnn layout changed?)")

    # 2) 1x1 pointwise → ALWAYS im2col-gemm (part of the sgemm-or-1x1 if).
    m_1x1 = re.search(r"use_sgemm_convolution && prefer_sgemm\) \|\| \(kernel_w == 1 && kernel_h == 1\)", text)
    if m_1x1:
        rules.append({
            "algo_family": "gemm", "priority": 2, "dtype": dtype,
            "when": "kernel==1x1 (pointwise degenerates to GEMM)",
            "verbatim": m_1x1.group(0)[:240],
            "predicate": {"kernel": 1},
            "ncnn_fn": "convolution_im2col_gemm",
            "cite": _cite(rel, text, m_1x1.start()),
        })
    else:
        warn.append(f"[{rel}] 1x1 gemm anchor not found")

    # 3) SGEMM (im2col-gemm) gate: prefer_sgemm channel threshold + L2 formula.
    m_sg = re.search(
        r"bool prefer_sgemm = (.+?) \|\| \(num_input > (\d+) \|\| num_output > (\d+)\)",
        text)
    if m_sg:
        ch = int(m_sg.group(2))
        rules.append({
            "algo_family": "gemm", "priority": 3, "dtype": dtype,
            "when": f"prefer_sgemm: work > L2_cache OR (num_input>{ch} || num_output>{ch})",
            "verbatim": ("bool prefer_sgemm = " + m_sg.group(1).strip())[:240],
            "predicate": {"channel_any_gt": ch},
            "ncnn_fn": "convolution_im2col_gemm",
            "cite": _cite(rel, text, m_sg.start()),
        })
    else:
        warn.append(f"[{rel}] prefer_sgemm anchor not found")

    # 4) DIRECT fallback (small channels / stride>2 / other kernels): the else-cascade.
    rules.append({
        "algo_family": "direct", "priority": 9, "dtype": dtype,
        "when": "fallback: small channels, stride>2, or kernels without a gemm/winograd path",
        "verbatim": "(else branch of the dispatch cascade)",
        "predicate": {"default": True},
        "ncnn_fn": "conv{1x1,3x3,5x5,7x7}s{1,2}*_neon (direct cascade)",
        "cite": f"ncnn/{rel} (direct-kernel cascade)",
    })
    return rules, warn


def _parse_depthwise(text: str, rel: str) -> tuple[list[dict], list[str]]:
    """ConvolutionDepthWise → the whole family is `dw` (specialized 3x3/5x5 kernels)."""
    m = re.search(r"kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1"
                  r" && stride_w == 1 && stride_h == 1", text)
    cite = _cite(rel, text, m.start()) if m else f"ncnn/{rel}"
    warn = [] if m else [f"[{rel}] depthwise 3x3s1 anchor not found"]
    return [{
        "algo_family": "dw", "priority": 1, "dtype": "any",
        "when": "depthwise conv → specialized depthwise kernels (3x3/5x5 × s1/s2 × pack)",
        "verbatim": (m.group(0)[:240] if m else "convdw{3x3,5x5}s{1,2}*_neon"),
        "predicate": {"default": True},
        "ncnn_fn": "convdw{3x3,5x5}s{1,2}[_pack4/_pack8]_neon",
        "cite": cite,
    }], warn


def _parse_deconv(text: str, rel: str) -> tuple[list[dict], list[str]]:
    """Deconvolution → im2col-gemm when use_sgemm_convolution, else direct."""
    m = re.search(r"if \(opt\.use_sgemm_convolution\)", text)
    cite = _cite(rel, text, m.start()) if m else f"ncnn/{rel}"
    warn = [] if m else [f"[{rel}] deconv use_sgemm anchor not found"]
    return [
        {"algo_family": "gemm", "priority": 1, "dtype": "any",
         "when": "use_sgemm_convolution → gemm (via child Gemm layer)",
         "verbatim": "if (opt.use_sgemm_convolution)", "predicate": {"channel_any_gt": 0},
         "ncnn_fn": "gemm", "cite": cite},
        {"algo_family": "direct", "priority": 9, "dtype": "any",
         "when": "fallback: direct deconv (3x3/4x4 × s1/s2, pack1)",
         "verbatim": "(else branch)", "predicate": {"default": True},
         "ncnn_fn": "deconv{3x3,4x4}s{1,2}_neon", "cite": f"ncnn/{rel}"},
    ], warn


# ---------------------------------------------------------------------------
# family key -> (relative source path, ncnn layer, parser)
_FAMILIES = {
    "conv":    ("src/layer/arm/convolution_arm.cpp",           "Convolution",           _parse_convolution),
    "conv_dw": ("src/layer/arm/convolutiondepthwise_arm.cpp",  "ConvolutionDepthWise",  _parse_depthwise),
    "deconv":  ("src/layer/arm/deconvolution_arm.cpp",         "Deconvolution",         _parse_deconv),
}
# fp16 overlay: same family, extra threshold set (merged into conv rules as dtype=fp16)
_FP16_CONV = ("src/layer/arm/convolution_arm_asimdhp.cpp", "Convolution")


def distill(ncnn_root: Path, only: set[str] | None = None) -> dict:
    families: dict[str, dict] = {}
    all_warn: list[str] = []
    for key, (rel, layer, parser) in _FAMILIES.items():
        if only and key not in only:
            continue
        p = ncnn_root / rel
        if not p.exists():
            all_warn.append(f"[{key}] source missing: {rel}")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if parser is _parse_convolution:
            rules, warn = parser(text, rel, "fp32")
            # merge fp16 thresholds as additional dtype-tagged winograd/gemm rules
            fp16p = ncnn_root / _FP16_CONV[0]
            if fp16p.exists():
                fp16_rules, fp16_warn = _parse_convolution(
                    fp16p.read_text(encoding="utf-8", errors="replace"), _FP16_CONV[0], "fp16")
                # keep only the threshold-bearing fp16 rules (winograd + sgemm), for audit —
                # they mirror fp32 structure but with different channel thresholds.
                rules += [r for r in fp16_rules
                          if ("channel_any_ge" in r["predicate"]
                              or "channel_any_gt" in r["predicate"])]
                warn += fp16_warn
            else:
                all_warn.append(f"[conv] fp16 overlay missing: {_FP16_CONV[0]}")
        else:
            rules, warn = parser(text, rel)
        families[key] = {"layer": layer, "cite_file": f"ncnn/{rel}",
                         "rules": rules, "warnings": warn}
        all_warn += warn
    return {"backend": "arm", "version": 1,
            "algo_family_axis": ["direct", "gemm", "winograd", "fft", "dw"],
            "note": "family-selection gates only; direct micro-variants omitted. "
                    "fft is never selected by ncnn (kept as an explore-only niche).",
            "families": families, "warnings": all_warn}


# ---------------------------------------------------------------------------
def to_markdown(doc: dict) -> str:
    out = ["# ncnn convolution dispatch heuristics (auto-distilled)\n",
           f"- backend: **{doc['backend']}**  |  algo_family axis: "
           f"`{', '.join(doc['algo_family_axis'])}`",
           f"- families: **{len(doc['families'])}**  |  warnings: "
           f"**{len(doc['warnings'])}**\n",
           f"> {doc['note']}\n"]
    if doc["warnings"]:
        out.append("## ⚠ warnings\n")
        out += [f"- {w}" for w in doc["warnings"]]
        out.append("")
    for key, fam in doc["families"].items():
        out.append(f"## {key}  (`{fam['layer']}`, {fam['cite_file']})\n")
        for r in fam["rules"]:
            out.append(f"- **{r['algo_family']}** (prio {r['priority']}, {r['dtype']}) — "
                       f"{r['when']}")
            out.append(f"  - `{r['cite']}`  ncnn_fn=`{r['ncnn_fn']}`")
            if r.get("tiles"):
                out.append(f"  - tiles: `{r['tiles']}`")
        out.append("")
    return "\n".join(out) + "\n"


def diff_against(old_json: Path, new_doc: dict) -> str:
    if not old_json.exists():
        return f"(no prior json at {old_json})"
    old = json.loads(old_json.read_text(encoding="utf-8")).get("families", {})
    new = new_doc["families"]
    lines = []
    for key in sorted(set(old) | set(new)):
        o = {r["algo_family"] + "/" + r["dtype"]: r.get("verbatim") for r in old.get(key, {}).get("rules", [])}
        n = {r["algo_family"] + "/" + r["dtype"]: r.get("verbatim") for r in new.get(key, {}).get("rules", [])}
        changed = [k for k in set(o) & set(n) if o[k] != n[k]]
        if key not in old:
            lines.append(f"+ family added: {key}")
        elif key not in new:
            lines.append(f"- family removed: {key}")
        elif changed:
            lines.append(f"~ {key}: condition drift in {changed}")
    return "\n".join(lines) or "(no structural changes)"


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ncnn-root", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--only", default=None, help="comma list of family keys (conv,conv_dw,deconv)")
    p.add_argument("--diff", type=Path, default=None,
                   help="path to a previous dispatch_heuristics.json; print condition drift")
    args = p.parse_args()

    if not args.ncnn_root.exists():
        raise SystemExit(f"--ncnn-root not found: {args.ncnn_root}")

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    doc = distill(args.ncnn_root, only)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "dispatch_heuristics.json"
    md_path = args.out_dir / "dispatch_heuristics.md"
    json_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(to_markdown(doc), encoding="utf-8")

    n_rules = sum(len(f["rules"]) for f in doc["families"].values())
    print(f"[dispatch] wrote {json_path}")
    print(f"[dispatch] wrote {md_path}")
    print(f"[dispatch] {len(doc['families'])} families | {n_rules} rules | "
          f"{len(doc['warnings'])} warnings")
    for w in doc["warnings"]:
        print(f"  ⚠ {w}")

    if args.diff:
        print("\n[dispatch] diff vs", args.diff)
        print(diff_against(args.diff, doc))


if __name__ == "__main__":
    main()
