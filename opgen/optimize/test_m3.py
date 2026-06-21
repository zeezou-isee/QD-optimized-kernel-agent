"""M3 self-tests — best-first control arm, compare verdict, cross-task warm start.

No LLM, no ncnn build required (fake evaluator + stub proposer with .vary).
Run: .venv/bin/python opgen/optimize/test_m3.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import opgen as _opgen; _opgen.bootstrap_paths()

from schemas import CorrectnessReport, MeasureSample, ParamSpec, ParameterizedTemplate
from inner import ConstraintEngine, detect
from policy import COMPUTE_BOUND, run_best_first, compare, ExperiencePool
from optimize_agent import OptimizeAgent

_PASS = _FAIL = 0


def check(name, cond, extra=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1; print(f"  PASS  {name}")
    else:
        _FAIL += 1; print(f"  FAIL  {name}  {extra}")


class _FakeComputeEvaluator:
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


_BASELINE = ParameterizedTemplate(
    kernel_files={"cand_x.cpp": "class Cand_X : public Layer {};\n"}, params={},
    class_name="Cand_X", header="cand_x.h", file="cand_x.cpp", techniques=[])

_ROT = [["vectorize"], ["dotprod"], ["scalar", "tiling"]]


class _StubProposer:
    """vary() ignores parent; rotates structural tags by history length."""
    def vary(self, parent, directive, history):
        return _tmpl(_ROT[len(history) % len(_ROT)])


def test_best_first():
    print("[test] best_first (direct argmin, no archive)")
    eng = ConstraintEngine(detect())
    sp = _StubProposer()
    res = run_best_first(
        baseline_template=_BASELINE, baseline_latency=20.0,
        evaluator=_FakeComputeEvaluator(), engine=eng,
        vary_fn=lambda t, h: sp.vary(t, "optimize", h),
        budget=60, inner_budget=8, patience=4)
    check("best_first found 5.0", res["best_latency_ms"] == 5.0, str(res["best_latency_ms"]))
    check("respected budget", res["rounds"] <= 60)
    check("terminated", bool(res["stopped_reason"]))


def test_compare():
    print("[test] compare() verdicts (§7.5)")
    # QD faster AND from a non-mainstream niche -> diversity paid off
    qd = {"best_latency_ms": 5.0, "rounds": 40, "grid_argmin_cell": ["direct", "dotprod"]}
    bf = {"best_latency_ms": 8.0, "rounds": 40}
    c1 = compare(qd, bf, baseline_cell=("direct", "scalar"))
    check("qd wins", c1["verdict"] == "qd", c1["verdict"])
    check("nonmainstream flagged", c1["argmin_from_nonmainstream"] is True)
    # tie on latency, best-first used fewer rounds -> baseline suffices
    c2 = compare({"best_latency_ms": 5.0, "rounds": 60, "grid_argmin_cell": ["direct", "vec"]},
                 {"best_latency_ms": 5.0, "rounds": 20},
                 baseline_cell=("direct", "scalar"))
    check("best_first wins on rounds", c2["verdict"] == "best_first", c2["verdict"])


def test_optimize_agent_map_elites():
    print("[test] OptimizeAgent policy=map_elites (+ baseline compare)")
    agent = OptimizeAgent(
        task_name="Fake", baseline_kernel_code={"cand_x.cpp": "x", "cand_x.h": "y"},
        evaluator_obj=_FakeComputeEvaluator(), proposer=_StubProposer(),
        policy="map_elites", regime=COMPUTE_BOUND, map_budget=80,
        coverage_target=3, patience=4, run_baseline_comparison=True)
    res = agent.run()
    d = res.to_dict()
    check("policy tagged", d["policy"] == "map_elites")
    check("improved over baseline", d["best_round"] == 0, str(d["best_round"]))
    check("best perf is 5.0", d["best_perf"].get("avg") == 5.0, str(d["best_perf"]))
    check("argmin cell dotprod", d["extra"]["argmin_cell"] == ["direct", "dotprod"],
          str(d["extra"].get("argmin_cell")))
    check("coverage >=3", d["extra"]["coverage"] >= 3, str(d["extra"].get("coverage")))
    check("archive persisted in extra", "cells" in (d["extra"].get("archive") or {}))
    cmp = d["extra"].get("baseline_comparison")
    check("baseline comparison ran", cmp is not None and "verdict" in cmp, str(cmp))
    check("nonmainstream argmin", cmp and cmp["argmin_from_nonmainstream"] is True)


def test_warm_start_persistence():
    print("[test] cross-task warm start via experience pool (兵器谱)")
    with tempfile.TemporaryDirectory() as td:
        pool_path = Path(td) / "weapon.json"

        def make_agent():
            return OptimizeAgent(
                task_name="Fake", baseline_kernel_code={"cand_x.cpp": "x", "cand_x.h": "y"},
                evaluator_obj=_FakeComputeEvaluator(), proposer=_StubProposer(),
                policy="map_elites", regime=COMPUTE_BOUND, map_budget=60,
                coverage_target=3, patience=4, experience_pool_path=str(pool_path))

        make_agent().run()
        check("pool file written", pool_path.exists())
        n1 = len(ExperiencePool(pool_path).records)
        check("first run persisted records", n1 > 0, str(n1))

        # second run: should load same-regime seeds AND persist again (pool grows)
        res2 = make_agent().run()
        n2 = len(ExperiencePool(pool_path).records)
        check("pool grew after 2nd run (累积)", n2 > n1, f"{n1}->{n2}")
        check("second run still optimal", res2.to_dict()["best_perf"]["avg"] == 5.0)


if __name__ == "__main__":
    test_best_first()
    test_compare()
    test_optimize_agent_map_elites()
    test_warm_start_persistence()
    print(f"\n==== {_PASS} passed, {_FAIL} failed ====")
    sys.exit(1 if _FAIL else 0)
