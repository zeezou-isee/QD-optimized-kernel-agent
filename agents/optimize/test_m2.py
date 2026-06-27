"""M2 self-tests — roofline + BD + MAP-Elites archive + experience pool.

No LLM, no ncnn build required (fake evaluator + stub vary_fn).
Run: .venv/bin/python agents/optimize/test_m2.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import agents as _agents; _agents.bootstrap_paths()

from schemas import CorrectnessReport, MeasureSample, ParamSpec, ParameterizedTemplate
from inner import ConstraintEngine, detect
from policy import (Archive, Elite, ExperiencePool, OperatorProfile, DeviceRoofline,
                    COMPUTE_BOUND, MEMORY_BOUND, classify, diagnose, grid_size,
                    run_map_elites)

_PASS = _FAIL = 0


def check(name, cond, extra=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1; print(f"  PASS  {name}")
    else:
        _FAIL += 1; print(f"  FAIL  {name}  {extra}")


def test_roofline():
    print("[test] roofline.diagnose")
    # elementwise: AI ~ 1/8 -> memory bound
    mem = diagnose(OperatorProfile(flops=1000, bytes=8000))
    check("elementwise memory_bound", mem.regime == MEMORY_BOUND, mem.regime)
    # high arithmetic intensity -> compute bound
    comp = diagnose(OperatorProfile(flops=1e9, bytes=1e6))
    check("high-AI compute_bound", comp.regime == COMPUTE_BOUND, comp.regime)
    # early stop fires only with peaks + within eps of floor
    r = diagnose(OperatorProfile(flops=1e9, bytes=1e8),
                 DeviceRoofline(peak_flops=1e11, peak_bw_bytes_s=1e10))
    check("min_latency computed", r.min_latency_ms is not None)
    check("early stop at floor", r.early_stop_ok(r.min_latency_ms * 1.01))
    check("no early stop far away", not r.early_stop_ok(r.min_latency_ms * 5))


def test_bd():
    print("[test] bd.classify")
    check("compute vec", classify(["vectorize"], COMPUTE_BOUND) == ("direct", "vec"))
    check("compute dotprod", classify(["winograd", "dotprod"], COMPUTE_BOUND) == ("winograd", "dotprod"))
    check("compute scalar default", classify([], COMPUTE_BOUND) == ("direct", "scalar"))
    check("memory tiling", classify(["tiling"], MEMORY_BOUND) == ("nchw", "single"))
    check("memory packed double", classify(["pack", "double"], MEMORY_BOUND) == ("packed", "double"))
    check("grid sizes", grid_size(COMPUTE_BOUND) == 15 and grid_size(MEMORY_BOUND) == 9)


def _elite(cell, lat, src="search"):
    return Elite(cell=cell, latency_ms=lat, kernel_code={"a.cpp": f"// {lat}"}, source=src)


def test_archive():
    print("[test] archive cell competition + persistence")
    a = Archive()
    check("place into empty", a.place(_elite(("direct", "vec"), 10.0)))
    check("slower same cell rejected", not a.place(_elite(("direct", "vec"), 12.0)))
    check("faster same cell wins", a.place(_elite(("direct", "vec"), 8.0)))
    check("new cell survives even if slower",
          a.place(_elite(("direct", "scalar"), 99.0)))     # ★局部竞争★
    check("coverage 2", a.coverage() == 2, str(a.coverage()))
    check("argmin is fastest", a.argmin().latency_ms == 8.0)
    parents = a.select_parents(2)
    check("select returns elites", len(parents) == 2)
    # persistence round-trip
    b = Archive.from_dict(a.to_dict())
    check("persist round-trip", b.coverage() == 2 and b.argmin().latency_ms == 8.0)


def test_experience_pool():
    print("[test] experience_pool seed/persist (兵器谱)")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pool.json"
        a = Archive()
        a.place(_elite(("direct", "dotprod"), 5.0))
        a.place(_elite(("direct", "vec"), 9.0))
        pool = ExperiencePool(path)
        n = pool.add_archive(a, regime=COMPUTE_BOUND, op_class="Abs", hardware="arm64")
        pool.save()
        check("added 2 records", n == 2)
        # reload + seed a same-regime task (floor, 不过滤)
        pool2 = ExperiencePool(path)
        seeds = pool2.seeds_for(COMPUTE_BOUND, hardware="arm64")
        check("reload + same-regime seeds", len(seeds) == 2, str(len(seeds)))
        check("seeds marked source=seed", all(s.source == "seed" for s in seeds))
        check("other regime no seeds", pool2.seeds_for(MEMORY_BOUND) == [])


class _FakeComputeEvaluator:
    """latency = base(by compute_mapping) + (UNROLL-4)^2.
       base: dotprod=5, vec=10, scalar=20 → global best at dotprod/UNROLL=4 = 5."""
    class_name = "Cand_X"; header = "cand_x.h"; file = "cand_x.cpp"

    def evaluate(self, template, point):
        tags = " ".join(template.techniques).lower()
        base = 5 if "dotprod" in tags else (10 if ("vec" in tags or "neon" in tags) else 20)
        u = point.get("UNROLL", 4)
        lat = float(base + (u - 4) ** 2)
        return MeasureSample(point=dict(point), correct=True, latency_ms=lat,
                             latency_min_ms=lat, latency_median_ms=lat,
                             latency_std_ms=0.0, n_runs=1,
                             correctness=CorrectnessReport(True))


def _tmpl(techs):
    return ParameterizedTemplate(
        kernel_files={"cand_x.cpp": "class Cand_X : public Layer {};\n"},
        params={"UNROLL": ParamSpec("UNROLL", [1, 2, 4, 8])},
        class_name="Cand_X", header="cand_x.h", file="cand_x.cpp", techniques=techs)


def test_map_elites():
    print("[test] map_elites outer loop (fake evaluator + stub vary)")
    eng = ConstraintEngine(detect())
    rotation = [["vectorize"], ["dotprod"], ["scalar", "tiling"]]

    def vary_fn(parent, directive, history):
        return _tmpl(rotation[len(history) % len(rotation)])

    baseline = ParameterizedTemplate(
        kernel_files={"cand_x.cpp": "class Cand_X : public Layer {};\n"}, params={},
        class_name="Cand_X", header="cand_x.h", file="cand_x.cpp", techniques=[])

    res = run_map_elites(
        baseline_template=baseline, baseline_latency=20.0,
        evaluator=_FakeComputeEvaluator(), engine=eng, vary_fn=vary_fn,
        regime=COMPUTE_BOUND, budget=80, inner_budget=8,
        coverage_target=3, patience=4)

    check("found global argmin 5.0", res["best_latency_ms"] == 5.0, str(res["best_latency_ms"]))
    check("argmin is dotprod cell", res["grid_argmin_cell"] == ["direct", "dotprod"],
          str(res["grid_argmin_cell"]))
    check("covered multiple niches", res["coverage"] >= 3, str(res["coverage"]))
    check("respected budget", res["rounds"] <= 80, str(res["rounds"]))
    check("terminated", bool(res["stopped_reason"]))
    # seed floor: a pre-known elite occupies its niche even though slower
    res2 = run_map_elites(
        baseline_template=baseline, baseline_latency=20.0,
        evaluator=_FakeComputeEvaluator(), engine=eng, vary_fn=vary_fn,
        regime=COMPUTE_BOUND, seeds=[_elite(("winograd", "vec"), 50.0, src="seed")],
        budget=40, inner_budget=8, coverage_target=3, patience=4)
    has_seed = any(tuple(c["cell"] if isinstance(c, dict) else c) == ("winograd", "vec")
                   for c in [e for e in [res2["best"]]] ) or \
        ("winograd" in str(res2["archive"]))
    check("seed niche present in archive", "winograd" in str(res2["archive"]))


if __name__ == "__main__":
    test_roofline()
    test_bd()
    test_archive()
    test_experience_pool()
    test_map_elites()
    print(f"\n==== {_PASS} passed, {_FAIL} failed ====")
    sys.exit(1 if _FAIL else 0)
