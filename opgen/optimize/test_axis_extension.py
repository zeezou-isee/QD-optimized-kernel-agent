"""Test the axis-extension closed loop (Method M2.5.2) — the paper's central
novelty made real: a novel structural label proposed by the LLM opens a new
niche, and after winning across N_promote DISTINCT tasks it is PROMOTED into the
machine-readable Σ registry (experience_pool/wiki/sigma/<backend>.json), which
literally grows.

Run: .venv/bin/python opgen/optimize/test_axis_extension.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # opgen/
sys.path.insert(0, str(Path(__file__).resolve().parent))              # opgen/optimize/

from schemas import CorrectnessReport, MeasureSample, ParamSpec, ParameterizedTemplate
from inner import ConstraintEngine, detect
from policy import COMPUTE_BOUND, run_map_elites, sigma as S

_PASS = _FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {extra}")


class _FakeEval:
    """Always-correct; latency favors the novel 'strassen' algo so it wins its
    (brand-new) niche — a novel label that's fast enough to keep."""
    class_name = "Cand_X"; header = "cand_x.h"; file = "cand_x.cpp"

    def evaluate(self, template, point):
        tags = " ".join(template.techniques).lower()
        labels = " ".join((template.bd_labels or {}).values()).lower()
        blob = tags + " " + labels
        lat = 3.0 if "strassen" in blob else (10.0 if "gemm" in blob else 20.0)
        return MeasureSample(point=dict(point), correct=True, latency_ms=lat,
                             latency_min_ms=lat, latency_median_ms=lat,
                             latency_std_ms=0.0, n_runs=1,
                             correctness=CorrectnessReport(True))


def _novel_tmpl():
    """A proposal declaring an OUT-OF-Σ structural label (algo_family=strassen)."""
    return ParameterizedTemplate(
        kernel_files={"cand_x.cpp": "class Cand_X : public Layer {};\n"},
        params={"UNROLL": ParamSpec("UNROLL", [1, 2, 4])},
        class_name="Cand_X", header="cand_x.h", file="cand_x.cpp",
        techniques=["strassen"],
        bd_labels={"algo_family": "strassen", "compute_mapping": "vec"})


def _seed_sigma(wiki_root: Path, backend: str = "base") -> None:
    sg = S.Sigma(backend=backend, path=S.sigma_path(wiki_root, backend),
                 data=S.fallback_data(backend))
    sg.save()


def _baseline():
    return ParameterizedTemplate(
        kernel_files={"cand_x.cpp": "class Cand_X : public Layer {};\n"}, params={},
        class_name="Cand_X", header="cand_x.h", file="cand_x.cpp", techniques=[])


def _run(wiki_root, task, n_promote=3):
    return run_map_elites(
        baseline_template=_baseline(), baseline_latency=20.0,
        evaluator=_FakeEval(), engine=ConstraintEngine(detect()),
        vary_fn=lambda parent, directive, history: _novel_tmpl(),
        regime=COMPUTE_BOUND, budget=6, inner_budget=3,
        coverage_target=2, patience=3,
        backend="base", wiki_root=wiki_root, task_name=task, n_promote=n_promote)


def test_novel_opens_niche_then_promotes():
    print("[test] axis-extension: novel label opens niche → cross-task promote → Σ grows")
    with tempfile.TemporaryDirectory() as td:
        wiki_root = Path(td) / "wiki"
        _seed_sigma(wiki_root)
        size0 = S.load(wiki_root, "base").size()

        # strassen must NOT be in Σ initially
        check("strassen unknown at start",
              not S.load(wiki_root, "base").is_known("compute_bound", "axis1", "strassen"))

        # run 1 & 2 on DISTINCT tasks — accumulate, no promotion yet
        r1 = _run(wiki_root, "MatMul")
        r2 = _run(wiki_root, "MatMul_3d")
        promos_12 = r1["axis_extension"]["promotions"] + r2["axis_extension"]["promotions"]
        check("no promotion after 2 tasks", promos_12 == [], str(promos_12))
        check("novel niche won each run",
              bool(r1["axis_extension"]["novel_niche_wins"]) and
              bool(r2["axis_extension"]["novel_niche_wins"]))
        check("strassen still not in Σ after 2",
              not S.load(wiki_root, "base").is_known("compute_bound", "axis1", "strassen"))

        # run 3 on a THIRD distinct task — triggers promotion (N_promote=3)
        r3 = _run(wiki_root, "Gemm")
        check("promotion fired on 3rd distinct task",
              any(e["value"] == "strassen" for e in r3["axis_extension"]["promotions"]),
              str(r3["axis_extension"]["promotions"]))

        # Σ ON DISK has grown
        sg = S.load(wiki_root, "base")
        check("strassen now IN Σ (persisted)",
              sg.is_known("compute_bound", "axis1", "strassen"))
        check("|Σ| grew by exactly 1", sg.size() == size0 + 1,
              f"{sg.size()} vs {size0}+1")
        check("promoted audit log records tasks",
              any(p["value"] == "strassen" and set(p["tasks"]) >= {"MatMul", "MatMul_3d", "Gemm"}
                  for p in sg.data.get("promoted", [])),
              str(sg.data.get("promoted")))


def test_same_task_does_not_promote():
    print("[test] axis-extension: repeating the SAME task must NOT promote (cross-task only)")
    with tempfile.TemporaryDirectory() as td:
        wiki_root = Path(td) / "wiki"
        _seed_sigma(wiki_root)
        for _ in range(5):
            r = _run(wiki_root, "MatMul")          # same task 5×
        sg = S.load(wiki_root, "base")
        check("strassen NOT promoted from one task",
              not sg.is_known("compute_bound", "axis1", "strassen"))
        pend = sg.data.get("pending", {})
        key = "compute_bound|algo_family|strassen"
        check("pending counter capped at 1 distinct task",
              key in pend and pend[key]["wins"] == 1, str(pend.get(key)))


def test_readonly_ablation_no_writeback():
    print("[test] axis-extension: wiki_root=None (ablation) opens niche in-run, no write-back")
    r = _run(None, "MatMul")
    check("novel niche still detected in-run",
          bool(r["axis_extension"]["novel_niche_wins"]))
    check("no promotions when Σ read-only",
          r["axis_extension"]["promotions"] == [])
    check("sigma_size None when read-only",
          r["axis_extension"]["sigma_size"] is None)


if __name__ == "__main__":
    test_novel_opens_niche_then_promotes()
    test_same_task_does_not_promote()
    test_readonly_ablation_no_writeback()
    print(f"\n==== {_PASS} passed, {_FAIL} failed ====")
    sys.exit(1 if _FAIL else 0)
