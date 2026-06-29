"""Convert every model in dataset/Mobilekernelbench to ncnn — existence check +
graph conversion ONLY (no kernel generation).

For each <Op>.py reference model this does exactly two of the operator pipeline's
stages, reusing opgen's own code (no reimplementation):

  [existence check + conversion]  probe_pnnx_ir(cfg, model_py, ...)
      -> traces the model to TorchScript (make_pt)
      -> runs the prebuilt pnnx binary (run_conversion): PyTorch -> .ncnn.param/.bin
      -> reports whether ncnn's baseline pnnx already supports the op
         (baseline_structural_ok / baseline_numeric_ok / baseline_supported)

The converted .ncnn.param/.ncnn.bin (plus the .pnnx.param IR for reference) are
copied to dataset/converted/<category>/<Op>/. A per-op row and a roll-up are
written to dataset/converted/convert_results.json.

NOTE: kernel authoring, libncnn rebuild, GraphAgent (writing new passes), and
benchmarking are all intentionally skipped. This only exercises the *baseline*
pnnx converter against the unmodified ncnn source tree.

Run from anywhere with the repo's venv:
    python convert_dataset.py
    python convert_dataset.py --dataset dataset/Mobilekernelbench_subset
    python convert_dataset.py --only Neg,Sigmoid        # subset by op name
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# bootstrap opgen flat-import paths (same convention as the CLI entry points)
sys.path.insert(0, str(ROOT))            # so `import opgen` resolves
sys.path.insert(0, str(ROOT / "opgen"))  # so flat imports (config, graph_pipeline) resolve
import opgen as _opgen; _opgen.bootstrap_paths()

from config import GraphConfig            # noqa: E402
from graph_pipeline import probe_pnnx_ir  # noqa: E402


def discover_models(dataset_root: Path, only: set[str] | None) -> list[tuple[str, Path]]:
    """Return sorted (category, model_path) for every <Op>.py under dataset_root."""
    out = []
    for py in sorted(dataset_root.rglob("*.py")):
        if py.stem == "__init__":
            continue
        if only and py.stem not in only:
            continue
        out.append((py.parent.name, py))
    return out


def convert_one(cfg: GraphConfig, category: str, model_py: Path,
                work_root: Path, dest_root: Path) -> dict:
    """Existence-check + convert a single model; copy artifacts to dest_root."""
    op = model_py.stem
    work_dir = work_root / category / op
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        # probe_pnnx_ir = trace (make_pt) + pnnx convert (run_conversion) +
        # structural/numeric baseline checks, all in one. The conversion
        # artifacts land under work_dir/_probe/.
        g = probe_pnnx_ir(cfg, model_py, work_dir, op)
    except Exception as exc:  # noqa: BLE001 — one bad model shouldn't kill the batch
        return {"op": op, "category": category, "converted": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.time() - t0, 2)}
    dt = round(time.time() - t0, 2)

    probe_dir = work_dir / "_probe"
    converted = bool(g.get("ncnn_param"))
    row = {
        "op": op,
        "category": category,
        "converted": converted,
        "already_in_ncnn": bool(g.get("baseline_supported")),
        "baseline_structural_ok": g.get("baseline_structural_ok"),
        "baseline_numeric_ok": g.get("baseline_numeric_ok"),
        "op_types": g.get("op_types"),
        "residual_aten": g.get("residual_aten"),
        "elapsed_s": dt,
    }
    if g.get("error"):
        row["error"] = g["error"]
    if not converted:
        return row

    # copy artifacts (.ncnn.param/.bin = the deliverable; .pnnx.param = IR ref)
    dest = dest_root / category / op
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(probe_dir.glob(f"{op}.*")):
        if src.name.endswith((".ncnn.param", ".ncnn.bin", ".pnnx.param")):
            shutil.copy2(src, dest / src.name)
            copied.append(src.name)
    row["artifacts"] = copied
    # report dest relative to repo root when possible, else absolute (--dest may
    # point outside the repo).
    row["dest"] = str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset/Mobilekernelbench",
                    help="dataset dir to scan (relative to repo root or absolute)")
    ap.add_argument("--dest", default="dataset/converted",
                    help="output dir for converted models")
    ap.add_argument("--only", default=None,
                    help="comma list of op names to convert (default: all)")
    ap.add_argument("--ncnn-root", default=None, help="override ncnn source tree")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the _work/ scratch dir (traced .pt, pnnx intermediates, "
                         "drivers). Default: removed on success, leaving only deliverables.")
    args = ap.parse_args()

    dataset_root = (ROOT / args.dataset) if not Path(args.dataset).is_absolute() else Path(args.dataset)
    dest_root = (ROOT / args.dest) if not Path(args.dest).is_absolute() else Path(args.dest)
    only = {s.strip() for s in args.only.split(",")} if args.only else None

    if not dataset_root.is_dir():
        sys.exit(f"dataset dir not found: {dataset_root}")

    # GraphConfig's __post_init__ does Path(ncnn_root); passing None would crash,
    # so only override when --ncnn-root is given (else use its default_factory).
    cfg_kwargs = {"run_numeric": False}
    if args.ncnn_root:
        cfg_kwargs["ncnn_root"] = args.ncnn_root
    cfg = GraphConfig(**cfg_kwargs)
    if not cfg.pnnx_bin.exists():
        sys.exit(f"pnnx binary not found: {cfg.pnnx_bin}\n"
                 f"Build it first (see opgen/README.md '一次性前提').")

    work_root = dest_root / "_work"
    models = discover_models(dataset_root, only)
    print(f"[convert] {len(models)} models under {dataset_root}")
    print(f"[convert] pnnx = {cfg.pnnx_bin}")
    print(f"[convert] dest = {dest_root}\n")

    results: list[dict] = []
    for i, (cat, mp) in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {cat}/{mp.stem} ...", flush=True)
        row = convert_one(cfg, cat, mp, work_root, dest_root)
        results.append(row)
        flag = ("OK" if row["converted"] else "FAIL")
        native = " native" if row.get("already_in_ncnn") else ""
        print(f"        {flag}{native} "
              f"(types={row.get('op_types')} {row.get('elapsed_s')}s)"
              + (f" err={row['error']}" if row.get("error") else ""),
              flush=True)

    n_conv = sum(1 for r in results if r["converted"])
    n_native = sum(1 for r in results if r.get("already_in_ncnn"))
    summary = {
        "dataset": str(dataset_root.relative_to(ROOT) if dataset_root.is_relative_to(ROOT) else dataset_root),
        "total": len(results),
        "converted": n_conv,
        "already_in_ncnn": n_native,
        "failed": len(results) - n_conv,
        "results": results,
    }
    out_json = dest_root / "convert_results.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.keep_work and work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)

    print(f"\n[convert] DONE: {n_conv}/{len(results)} converted "
          f"({n_native} already native). Summary -> {out_json}")


if __name__ == "__main__":
    main()
