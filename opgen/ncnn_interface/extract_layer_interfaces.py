"""Extract the ncnn built-in layer interface dictionary.

Scans every (.h, .cpp) pair under ncnn/src/layer/ (NOT the per-arch
subdirs like vulkan/, arm/, x86/ — those are optimized backends of the
same layer and share the base interface).

Cross-checks every parsed record against the official
operation-param-weight-table.md and annotates mismatches.

Outputs two artifacts under experience_pool/backend_ncnn/:
  - layer_interfaces.json  — machine-readable; LLM/Proposer consumes this
  - layer_interfaces.md    — human-readable review surface

Usage:
  python -m opgen.ncnn_interface.extract_layer_interfaces \
    --ncnn-root /Users/xingze/Documents/project/kernelgen/ncnn \
    [--out-dir experience_pool/backend_ncnn] \
    [--only Convolution,BinaryOp] \
    [--diff old_layer_interfaces.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .parser import parse_layer, ParseResult
from .md_doc_loader import load_doc_table


PROJ = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJ / "experience_pool" / "backend_ncnn"


# ---------------------------------------------------------------------------
def cross_check_doc(rec: dict, doc_entry: dict | None) -> tuple[bool, list[dict]]:
    """Return (doc_present, mismatches[])."""
    if doc_entry is None:
        return False, []

    by_id_src = {p["id"]: p["name"] for p in rec["params"]}
    by_id_doc = {p["id"]: p["name"] for p in doc_entry.get("params", [])}

    out = []
    for pid in sorted(set(by_id_src) | set(by_id_doc)):
        s, d = by_id_src.get(pid), by_id_doc.get(pid)
        if s is None:
            out.append({"type": "doc_only", "id": pid, "name": d})
        elif d is None:
            out.append({"type": "src_only", "id": pid, "name": s})
        elif s != d:
            out.append({"type": "name_diff", "id": pid, "doc": d, "src": s})
    return True, out


# ---------------------------------------------------------------------------
def discover_layers(ncnn_root: Path) -> list[tuple[Path, Path | None]]:
    """Return (header, cpp_or_None) for every layer in src/layer/*.h.

    Skips utility headers that aren't real layers (no `class X : public Y` —
    e.g. `fused_activation.h` is a static-inline helper set).
    """
    out = []
    for h in sorted((ncnn_root / "src" / "layer").glob("*.h")):
        # quick filter: a real layer header has `class <Name> : public <Base>`
        try:
            txt = h.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not re.search(r"\bclass\s+\w+\s*:\s*public\s+\w+", txt):
            continue
        c = h.with_suffix(".cpp")
        out.append((h, c if c.exists() else None))
    return out


# ---------------------------------------------------------------------------
def to_markdown(records: list[dict]) -> str:
    """Render a human-readable summary highlighting MISMATCH ops at the top."""
    has_mm = [r for r in records if r.get("mismatches")]
    no_mm  = [r for r in records if not r.get("mismatches")]

    lines: list[str] = ["# ncnn built-in layer interfaces (auto-generated)\n"]
    lines.append(f"- Layers parsed: **{len(records)}**")
    lines.append(f"- With doc-table cross-check **MISMATCH**: **{len(has_mm)}**")
    n_missing_doc = sum(1 for r in records if not r.get("doc_table_present"))
    lines.append(f"- Not present in operation-param-weight-table.md: **{n_missing_doc}**")
    n_warn = sum(1 for r in records if r.get("parse_warnings"))
    lines.append(f"- With parse warnings: **{n_warn}**\n")

    if has_mm:
        lines.append("## ⚠ MISMATCH ops (review these first)\n")
        for r in has_mm:
            lines.extend(_fmt_one(r, with_mm=True))

    lines.append("\n## All layers\n")
    for r in records:
        if r.get("mismatches"):
            continue
        lines.extend(_fmt_one(r, with_mm=False))

    return "\n".join(lines) + "\n"


def _fmt_one(r: dict, *, with_mm: bool) -> list[str]:
    out = [f"### {r['name']}  ({r['header']})"]
    out.append(f"- base class: `{r['base_class']}`")
    out.append(f"- forward: {len(r['forward_signatures'])} overload(s)")
    out.append(
        f"- flags (default): one_blob_only={r['one_blob_only_default']} "
        f"support_inplace={r['support_inplace_default']}"
    )
    if r["params"]:
        out.append("- params:")
        for p in r["params"]:
            tag = " *(var default)*" if p["default_is_var"] else ""
            out.append(f"  - `id={p['id']}` **{p['name']}** = `{p['default']}`{tag}")
    if r["weights_load_order"]:
        out.append("- weights (load order):")
        for w in r["weights_load_order"]:
            cond = f" *(if {w['conditional']})*" if w.get("conditional") else ""
            out.append(
                f"  - `[{w['index']}]` **{w['var']}** size=`{w['size_expr']}` "
                f"flag={w['flag']}{cond}"
            )
    if with_mm:
        out.append("- ⚠ doc-mismatch:")
        for m in r["mismatches"]:
            out.append(f"  - `{m}`")
    if r.get("parse_warnings"):
        out.append("- parse warnings:")
        for w in r["parse_warnings"]:
            out.append(f"  - {w}")
    out.append("")
    return out


# ---------------------------------------------------------------------------
def diff_against(old_json: Path, new_records: list[dict]) -> str:
    """Report ops whose interface changed since `old_json` (for ncnn upgrades)."""
    if not old_json.exists():
        return f"(no prior json at {old_json})"
    old = {r["name"]: r for r in json.loads(old_json.read_text(encoding="utf-8"))}
    new = {r["name"]: r for r in new_records}
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = []
    for name in sorted(set(new) & set(old)):
        # compare just the structural fields (skip mismatches/warnings)
        keys = ("params", "weights_load_order", "forward_signatures",
                "one_blob_only_default", "support_inplace_default", "base_class")
        if any(new[name].get(k) != old[name].get(k) for k in keys):
            changed.append(name)
    return (f"added ({len(added)}): {added}\n"
            f"removed ({len(removed)}): {removed}\n"
            f"changed ({len(changed)}): {changed}")


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ncnn-root", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--only", default=None,
                   help="comma-separated layer header stems (e.g. convolution,binaryop) — debug")
    p.add_argument("--diff", type=Path, default=None,
                   help="path to a previous layer_interfaces.json; print structural diff")
    args = p.parse_args()

    if not args.ncnn_root.exists():
        raise SystemExit(f"--ncnn-root not found: {args.ncnn_root}")

    doc_path = (args.ncnn_root / "docs" / "developer-guide"
                / "operation-param-weight-table.md")
    doc = load_doc_table(doc_path)
    print(f"[ncnn-iface] doc table: {len(doc)} ops loaded from {doc_path.name}")

    pairs = discover_layers(args.ncnn_root)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        pairs = [(h, c) for (h, c) in pairs if h.stem in wanted]
    print(f"[ncnn-iface] parsing {len(pairs)} layer pair(s) under src/layer/")

    records = []
    for h, c in pairs:
        r: ParseResult = parse_layer(h, c)
        rec = r.to_dict()
        doc_entry = doc.get(rec["name"])
        present, mm = cross_check_doc(rec, doc_entry)
        rec["doc_table_present"] = present
        rec["mismatches"] = mm
        records.append(rec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "layer_interfaces.json"
    md_path = args.out_dir / "layer_interfaces.md"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    md_path.write_text(to_markdown(records), encoding="utf-8")

    # quick stdout summary
    n_mm = sum(1 for r in records if r["mismatches"])
    n_no_doc = sum(1 for r in records if not r["doc_table_present"])
    n_warn = sum(1 for r in records if r["parse_warnings"])
    print(f"[ncnn-iface] wrote {json_path}")
    print(f"[ncnn-iface] wrote {md_path}")
    print(f"[ncnn-iface] summary: {len(records)} parsed | "
          f"{n_mm} doc-mismatch | {n_no_doc} not in doc | {n_warn} parse-warned")

    if args.diff:
        print("\n[ncnn-iface] diff vs", args.diff)
        print(diff_against(args.diff, records))


if __name__ == "__main__":
    main()
