"""M1 self-tests for the optimize agent — no LLM, no ncnn build required.

Covers (plan §验证 1-5):
  - constraint_engine: safe_eval + feasibility pruning
  - inner_search: fake gradient evaluator -> hill-climbs past the coarse grid
  - proposer.parse_template: canned LLM response -> ParameterizedTemplate
  - OptimizeAgent rich loop: stub proposer + fake evaluator -> best across rounds

Run: .venv/bin/python opgen/optimize/test_m1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import opgen as _opgen; _opgen.bootstrap_paths()

from schemas import (CorrectnessReport, MeasureSample, ParamSpec,
                     ParameterizedTemplate, materialize)
from inner import ConstraintEngine, detect, inner_search, safe_eval, coarse_points
from proposer import parse_template
from optimize_agent import OptimizeAgent

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {extra}")


# ---------------------------------------------------------------------------
def test_safe_eval():
    print("[test] safe_eval")
    check("arith le true", safe_eval("A*B <= 100", {"A": 10, "B": 5}) is True)
    check("arith le false", safe_eval("A*B <= 100", {"A": 10, "B": 20}) is False)
    check("chained", safe_eval("1 <= X < 8", {"X": 4}) is True)
    check("bool and", safe_eval("X <= 8 and Y <= 4", {"X": 8, "Y": 4}) is True)
    # no attribute / call access allowed
    raised = False
    try:
        safe_eval("__import__('os')", {})
    except Exception:
        raised = True
    check("rejects unsafe", raised)


def test_constraint_engine():
    print("[test] constraint_engine")
    hw = detect()
    eng = ConstraintEngine(hw)
    # UNROLL beyond register budget is pruned by built-in heuristic
    r = eng.feasible({"UNROLL": hw.vector_regs * 2}, [])
    check("unroll spill pruned", not r.ok, r.reason)
    # LLM equation referencing L1
    r2 = eng.feasible({"TILE": 4}, ["TILE*TILE*4 <= L1"])
    check("legal tile feasible", r2.ok, r2.reason)
    r3 = eng.feasible({"TILE": 100000}, ["TILE*TILE*4 <= L1"])
    check("huge tile pruned", not r3.ok, r3.reason)
    # bad vec width pruned
    r4 = eng.feasible({"VEC_WIDTH": 3}, [])
    check("vec width 3 pruned", not r4.ok, r4.reason)


class _FakeGradientEvaluator:
    """latency = (TILE_M-8)^2 + (UNROLL-2)^2 ; global min at (8,2)=0.

    Min is NOT on the coarse grid (reps TILE_M=[4,16,32], UNROLL=[1,4,8]), so a
    correct result requires the hill-climb stage to improve past the grid.
    """
    class_name = "Cand_Fake"
    header = "cand_fake.h"
    file = "cand_fake.cpp"

    def evaluate(self, template, point):
        if not point:                       # baseline measurement
            lat = 1000.0
        else:
            lat = float((point["TILE_M"] - 8) ** 2 + (point["UNROLL"] - 2) ** 2)
        return MeasureSample(point=dict(point), correct=True, latency_ms=lat,
                             latency_min_ms=lat, latency_median_ms=lat,
                             latency_std_ms=0.0, n_runs=1,
                             correctness=CorrectnessReport(True))


def _grad_template():
    return ParameterizedTemplate(
        kernel_files={"cand_fake.cpp": "class Cand_Fake : public Layer {};\n"},
        params={"TILE_M": ParamSpec("TILE_M", [4, 8, 16, 32]),
                "UNROLL": ParamSpec("UNROLL", [1, 2, 4, 8])},
        class_name="Cand_Fake", header="cand_fake.h", file="cand_fake.cpp",
    )


def test_inner_search():
    print("[test] inner_search (coarse + hill climb)")
    hw = detect()
    eng = ConstraintEngine(hw)
    basin = inner_search(_grad_template(), _FakeGradientEvaluator(), eng, budget=30)
    check("found global min params", basin.best_params == {"TILE_M": 8, "UNROLL": 2},
          str(basin.best_params))
    check("min latency 0", basin.best_latency_ms == 0.0, str(basin.best_latency_ms))
    check("counted evaluations", basin.n_evaluated > 0, str(basin.n_evaluated))


def test_coarse_points():
    print("[test] coarse_points")
    pts = coarse_points({"A": ParamSpec("A", [1, 2, 3, 4, 5])}, per_axis=3)
    check("3 reps from 5 values", pts == [{"A": 1}, {"A": 3}, {"A": 5}], str(pts))
    check("empty params -> one empty point", coarse_points({}) == [{}])


def test_parse_template():
    print("[test] proposer.parse_template")
    response = """Here is the optimized kernel.

```cpp
cand_abs.cpp
#include "cand_abs.h"
class Cand_Abs : public Layer {
  int forward() { for (int i=0;i<n;i+=<UNROLL>) {} return 0; }
};
```

```json
{
  "params": {"UNROLL": {"values": [1,2,4,8], "dtype": "int", "desc": "unroll"}},
  "constraints": ["UNROLL <= VECTOR_REGS"],
  "techniques": ["unroll"],
  "rationale": "unrolling hides latency"
}
```
"""
    t = parse_template(response, baseline_kernel={"cand_abs.cpp": "x", "cand_abs.h": "y"})
    check("class detected", t.class_name == "Cand_Abs", t.class_name)
    check("param parsed", "UNROLL" in t.params and t.params["UNROLL"].values == [1, 2, 4, 8])
    check("constraint parsed", t.constraints == ["UNROLL <= VECTOR_REGS"])
    check("technique parsed", t.techniques == ["unroll"])
    # materialize replaces the placeholder
    code = materialize(t, {"UNROLL": 4})
    check("materialize replaces", "<UNROLL>" not in code["cand_abs.cpp"]
          and "i+=4" in code["cand_abs.cpp"])


class _StubProposer:
    """Round 0: a knobbed template (best at TILE_M=8,UNROLL=2). Round 1: a slower
    template (best 100). Verifies the agent keeps round 0 as best."""
    def __init__(self):
        self.calls = 0

    def propose(self, history):
        self.calls += 1
        if self.calls == 1:
            return _grad_template()
        # a template that can only reach latency 100 (worse than round 0's 0)
        t = _grad_template()
        return ParameterizedTemplate(
            kernel_files=t.kernel_files,
            params={"TILE_M": ParamSpec("TILE_M", [16, 32]),  # min (16-8)^2=64
                    "UNROLL": ParamSpec("UNROLL", [4, 8])},    # +(4-2)^2=4 => 68
            class_name=t.class_name, header=t.header, file=t.file,
            techniques=["tiling"])


def test_optimize_agent_loop():
    print("[test] OptimizeAgent rich loop (stub proposer + fake evaluator)")
    agent = OptimizeAgent(
        task_name="Fake", baseline_kernel_code={"cand_fake.cpp": "x", "cand_fake.h": "y"},
        proposer=_StubProposer(), evaluator_obj=_FakeGradientEvaluator(),
        max_rounds=2, inner_budget=30, improve_tol=0.0,  # tol 0 => never early-converge
    )
    res = agent.run()
    d = res.to_dict()
    check("rich mode ran", agent._rich)
    check("has iterations", len(d["iterations"]) >= 1, str(len(d["iterations"])))
    check("best beats baseline (round 0)", d["best_round"] == 0, str(d["best_round"]))
    check("best perf is 0", d["best_perf"].get("avg") == 0.0, str(d["best_perf"]))
    check("best_kernel materialized", "<TILE_M>" not in str(d["best_kernel"]))
    check("round0 kept", d["iterations"][0]["kept"] is True)


def test_legacy_mode():
    print("[test] OptimizeAgent legacy mode (operator_agent compat)")
    seen = {}

    def ext_eval(code):
        seen["called"] = True
        return {"functional_ok": True, "perf": {"avg": 1.23}}

    agent = OptimizeAgent(task_name="X", baseline_kernel_code={"a.cpp": "x"},
                          evaluator=ext_eval, baseline_perf={"avg": 9.9})
    res = agent.run()
    d = res.to_dict()
    check("not rich", not agent._rich)
    check("baseline kept (round -1)", d["best_round"] == -1)
    check("perf from external eval", d["best_perf"].get("avg") == 1.23, str(d["best_perf"]))
    check("external evaluator called", seen.get("called") is True)


if __name__ == "__main__":
    test_safe_eval()
    test_constraint_engine()
    test_coarse_points()
    test_inner_search()
    test_parse_template()
    test_optimize_agent_loop()
    test_legacy_mode()
    print(f"\n==== {_PASS} passed, {_FAIL} failed ====")
    sys.exit(1 if _FAIL else 0)
