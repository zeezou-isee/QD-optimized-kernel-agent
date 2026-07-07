"""Audit which dataset ops are *decomposed* — pnnx converts them into a CHAIN of
≥2 distinct native ncnn layers, NOT a single op.

Why this matters (see 多层分解算子与QD性能测量分析.md):
    A decomposed op (e.g. LogSoftmax -> Softmax + UnaryOp(log)) is marked
    already_in_ncnn=True, KernelAgent still authors a monolithic Cand_<Op> and
    OptimizeAgent finds a QD winner FOR IT — but at runtime ncnn::Net runs the
    native chain, because retarget_param_output_layer's DECOMPOSED-OP GUARD
    refuses to point the output layer at the monolithic Cand. So the QD winner
    is NEVER called: its speedup does NOT land. Any reported QD speedup for such
    an op reflects an isolated LayerOracle island, not deployed runtime.

This script is the single source of truth for "which ops are island-only". It
reads the cached pnnx baseline params already sitting under
    runs*/<op>/operator/_baseline_probe/_probe/<op>.ncnn.param
so it needs NO pnnx re-run. Ops without a cached probe are reported as unknown.

Output:
    batch/results/decomposed_ops.json   — {op: {op_types, distinct, decomposed}}
    stdout table

Usage:
    python scripts/audit_decomposed_ops.py                     # scan all runs*/
    python scripts/audit_decomposed_ops.py --ops LogSoftmax,Gemm_alpha
    python scripts/audit_decomposed_ops.py --out batch/results/decomposed_ops.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "batch" / "results"

# pnnx-emitted structural layers that are not "the op" itself.
_STRUCTURAL = {"Input", "Output", "Split"}


def op_layer_types(param_path: str | Path) -> list[str]:
    """The ncnn layer types in a .ncnn.param, in file order, EXCLUDING structural
    layers (Input/Output/Split) and pnnx-only dotted names (torch.*, F.*)."""
    types: list[str] = []
    with open(param_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 2:                       # magic + "layer_count blob_count"
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            t = parts[0]
            if t in _STRUCTURAL or "." in t:
                continue
            types.append(t)
    return types


def _cached_param(op: str, runs_roots: list[Path]) -> Path | None:
    for root in runs_roots:
        for sub in ("operator/_baseline_probe/_probe", "operator/_baseline_probe",
                    "graph/_probe", "graph/round_00"):
            hits = sorted((root / op / sub).glob("*.ncnn.param")) if (root / op).exists() else []
            if hits:
                return hits[0]
    return None


def audit(ops: list[str] | None, runs_roots: list[Path]) -> dict[str, dict]:
    """{op: {op_types, distinct, decomposed(bool|None)}}. decomposed=None => no
    cached probe found (unknown)."""
    if ops is None:
        seen: set[str] = set()
        for root in runs_roots:
            if not root.exists():
                continue
            for p in root.glob("*/operator/_baseline_probe/_probe/*.ncnn.param"):
                seen.add(p.parents[3].name)
        ops = sorted(seen)

    out: dict[str, dict] = {}
    for op in ops:
        pp = _cached_param(op, runs_roots)
        if pp is None:
            out[op] = {"op_types": None, "distinct": None, "decomposed": None,
                       "note": "no cached pnnx probe (run the op or `run_operator_agent` first)"}
            continue
        types = op_layer_types(pp)
        distinct = sorted(set(types))
        out[op] = {"op_types": types, "distinct": distinct,
                   "decomposed": len(distinct) > 1, "param": str(pp)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit decomposed (multi-layer) ops.")
    ap.add_argument("--ops", default=None, help="explicit comma list (else scan runs*/)")
    ap.add_argument("--runs-roots", default=None,
                    help="comma list of run roots (default: opgen/runs,opgen/runs_arm,opgen/runs_vulkan)")
    ap.add_argument("--out", default=str(RESULTS / "decomposed_ops.json"))
    args = ap.parse_args()

    if args.runs_roots:
        roots = [Path(r.strip()) for r in args.runs_roots.split(",") if r.strip()]
    else:
        roots = [REPO / "opgen" / "runs", REPO / "opgen" / "runs_arm",
                 REPO / "opgen" / "runs_vulkan"]
    ops = [o.strip() for o in args.ops.split(",")] if args.ops else None

    res = audit(ops, roots)
    decomp = {k: v for k, v in res.items() if v.get("decomposed") is True}
    unknown = {k: v for k, v in res.items() if v.get("decomposed") is None}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"DECOMPOSED-OP AUDIT  ({len(res)} ops scanned, "
          f"{len(decomp)} decomposed, {len(unknown)} unknown/no-probe)")
    print("-" * 78)
    print("These ops run as a NATIVE CHAIN at runtime; any QD winner for the")
    print("monolithic Cand_<Op> is NOT called -> its speedup does NOT land:")
    print("-" * 78)
    for op in sorted(decomp):
        print(f"  {op:<48} {' + '.join(decomp[op]['distinct'])}")
    if unknown:
        print("-" * 78)
        print(f"  unknown (no cached pnnx probe): {', '.join(sorted(unknown))}")
    print("=" * 78)
    print(f"manifest -> {args.out}")


if __name__ == "__main__":
    main()
