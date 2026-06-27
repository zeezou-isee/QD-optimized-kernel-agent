"""GraphAgent: the agent loop that drives ncnn graph-conversion generation.

Architecture = agent loop (state machine) + functional pipeline calls + 3 roles
(analyzer / coder / debugger). Designed as a single class so it can be invoked
and verified standalone.

    agent = GraphAgent(task_name="HardSigmoid", model_py=".../HardSigmoid.py")
    summary = agent.run()

The loop is driven by the pipeline result: each round repairs the FIRST failing
stage (inject -> build -> convert/structural -> numeric).
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GraphConfig
from graph_pipeline import (
    build_pnnx,
    locate_build_errors,
    extract_code_blocks,
    inject_files,
    make_pt,
    probe_pnnx_ir,
    restore_files,
    retrieve_examples,
    run_conversion,
    verify_numeric,
    verify_structural,
)
from graph_prompts import analyzer_prompt, coder_prompt, debugger_prompt, parse_profile_json
from graph_schemas import BackupHandle, GraphResult, GraphRound, OpProfile, write_json
from llm_api import query_llm


class _AlreadySupported(Exception):
    """Raised to stop the loop early when the op already converts correctly."""


class GraphAgent:
    def __init__(
        self,
        *,
        task_name: str,
        model_py: str | Path | None = None,
        model_code: str | None = None,
        cfg: GraphConfig | None = None,
        llm_query: Callable[[str, str], str] | None = None,
        force_target_layer: str | None = None,
    ) -> None:
        self.task_name = task_name
        self.cfg = cfg or GraphConfig()
        self.llm = llm_query or query_llm
        # When set, the conversion MUST map to this exact ncnn layer type (the
        # newly-written kernel), instead of reusing arbitrary existing ops.
        self.force_target_layer = force_target_layer
        self.session = BackupHandle()
        self.profile: OpProfile | None = None
        self.history: list[GraphRound] = []
        self.memory: list[dict] = []
        self._already_supported = False

        self.model_py, self.model_code = self._resolve_model(model_py, model_code)

    # ------------------------------------------------------------------ setup
    def _resolve_model(self, model_py: str | Path | None, model_code: str | None) -> tuple[str, str]:
        if model_py:
            p = Path(model_py)
            return str(p), p.read_text(encoding="utf-8")
        if model_code:
            # materialise so make_pt has a file to import
            self.run_dir.mkdir(parents=True, exist_ok=True)
            p = self.run_dir / f"{self.task_name}.py"
            p.write_text(model_code, encoding="utf-8")
            return str(p), model_code
        # search the dataset for <task>.py
        root = self.cfg.dataset_root
        if root:
            matches = sorted(Path(root).rglob(f"{self.task_name}.py"))
            if matches:
                return str(matches[0]), matches[0].read_text(encoding="utf-8")
        raise FileNotFoundError(
            f"No model provided and {self.task_name}.py not found under {root}."
        )

    @property
    def run_dir(self) -> Path:
        return self.cfg.run_dir(self.task_name)

    # ------------------------------------------------------------------ roles
    def analyze(self, grounding: dict | None = None) -> OpProfile:
        prompt = analyzer_prompt(self.task_name, self.model_code, grounding=grounding)
        text = self.llm(prompt, self.cfg.model)
        (self.run_dir / "analyzer.md").write_text(text, encoding="utf-8")
        profile = parse_profile_json(self.task_name, text)
        self._ground_target(profile, grounding)
        write_json(self.run_dir / "op_profile.json", profile.to_dict())
        return profile

    @staticmethod
    def _ground_target(profile: OpProfile, grounding: dict | None) -> None:
        """Hard guard against the analyzer inventing a non-existent target layer.

        If the probe shows the op is FULLY converted natively (no unconverted
        aten/prim residuals) but the LLM's target_ncnn_layer is not among the ncnn
        layer types the conversion actually produces, the LLM hallucinated a layer
        (e.g. "Log" when torch.log folds into "UnaryOp"). Correct it to the real
        compute layer from the probe. Only fires for the fully-native case so a
        genuinely-new op (residuals present -> custom Cand_<Op>) is never touched.
        """
        if not grounding:
            return
        op_types = [t for t in (grounding.get("op_types") or []) if t != "Input"]
        residual = grounding.get("residual_aten") or []
        if residual or not op_types:
            return  # genuinely new op, or no probe info — leave the LLM's choice
        if profile.target_ncnn_layer in op_types:
            profile.already_supported = True
            return  # already correct
        # invented name on a fully-native op -> correct to the real compute layer
        corrected = op_types[-1]  # the output-producing compute layer for single-op models
        print(f"[analyze] target corrected (grounded): '{profile.target_ncnn_layer}' "
              f"-> '{corrected}' (op natively converts to {op_types})")
        profile.target_ncnn_layer = corrected
        profile.already_supported = True

    # ------------------------------------------------------------------ memory
    def _format_memory(self) -> str:
        if not self.memory:
            return "(none)"
        lines = []
        for m in self.memory[-4:]:
            lines.append(f"round {m['round']} phase={m['phase']} -> stages={m['stages']}")
            if m.get("feedback"):
                lines.append(f"  feedback: {m['feedback'][:600]}")
        return "\n".join(lines)

    def _update_memory(self, round_idx: int, phase: str, result: GraphResult) -> None:
        self.memory.append({
            "round": round_idx,
            "phase": phase,
            "stages": {
                "inject": result.inject_ok,
                "build": result.build_ok,
                "convert": result.convert_ok,
                "structural": result.structural_ok,
                "numeric": result.numeric_status,
            },
            "feedback": result.feedback(result.first_failure() or ""),
        })
        write_json(self.run_dir / "memory.json", {"memory": self.memory})

    # ------------------------------------------------------------- pipeline
    def _run_pipeline(self, code_book: dict[str, str], round_idx: int) -> GraphResult:
        assert self.profile is not None
        rd = self.run_dir / f"round_{round_idx:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        res = GraphResult(
            task_name=self.task_name,
            op_profile=self.profile.to_dict(),
            response_code=code_book,
            identify_ok=True,
        )

        # 1) inject
        ok, _, err = inject_files(self.cfg, code_book, self.session)
        res.inject_ok, res.inject_error = ok, err
        if not ok:
            res.messages.append("inject failed")
            return res

        # 2) build pnnx
        focus = self._focus_file(code_book)
        bok, blog = build_pnnx(self.cfg, rd / "build.log")
        res.build_ok = bok
        if not bok:
            res.build_error = locate_build_errors(blog, opname=focus or self.task_name)
            res.messages.append("build failed")
            return res

        # 3) trace reference model
        pok, pt, ishape, plog = make_pt(self.cfg, self.model_py, rd)
        if not pok:
            res.convert_log = "TorchScript trace failed:\n" + plog
            res.messages.append("trace failed")
            return res

        # 4) run conversion
        cok, artifacts, clog = run_conversion(self.cfg, pt, ishape, rd, self.task_name)
        res.convert_ok = cok
        res.convert_log = clog
        res.artifacts.update(artifacts)
        if not cok:
            res.messages.append("conversion produced no .ncnn.param")
            return res

        # 5) structural verification (independent of the ncnn kernel)
        sok, slog = verify_structural(self.cfg, self.profile, artifacts, clog)
        res.structural_ok = sok
        res.structural_log = slog
        if not sok:
            res.messages.append("structural check failed")
            return res

        # 6) numeric verification (needs the ncnn kernel to exist & run)
        if self.cfg.run_numeric:
            short = self._test_short_name(code_book)
            nok, nlog = verify_numeric(self.cfg, short, rd / "numeric.log")
            res.numeric_ok = nok
            res.numeric_log = nlog
            res.messages.append("numeric passed" if nok else "numeric (allclose) failed")
        else:
            res.numeric_skipped = True
            res.messages.append("numeric skipped (run_numeric=False)")
        return res

    def _focus_file(self, code_book: dict[str, str]) -> str | None:
        for rel in code_book:
            if rel.startswith("pass_ncnn/") and rel.endswith(".cpp"):
                return Path(rel).stem
        return None

    def _test_short_name(self, code_book: dict[str, str]) -> str:
        for rel in code_book:
            if rel.startswith("tests/ncnn/") and rel.endswith(".py"):
                stem = Path(rel).stem
                return stem[len("test_"):] if stem.startswith("test_") else stem
        return f"F_{self.task_name.lower()}"

    # ------------------------------------------------------------- persistence
    def _save_round(self, round_idx: int, phase: str, prompt: str, response: str, result: GraphResult) -> None:
        rd = self.run_dir / f"round_{round_idx:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "prompt.md").write_text(prompt, encoding="utf-8")
        (rd / "response.md").write_text(response, encoding="utf-8")
        write_json(rd / "result.json", result.to_dict())
        self.history.append(GraphRound(
            round_idx=round_idx,
            phase=phase,
            prompt_path=str(rd / "prompt.md"),
            response_path=str(rd / "response.md"),
            result_path=str(rd / "result.json"),
            ok=result.ok,
            stages={
                "inject": result.inject_ok,
                "build": result.build_ok,
                "convert": result.convert_ok,
                "structural": result.structural_ok,
                "numeric": result.numeric_status,
            },
        ))
        write_json(self.run_dir / "history.json", {"history": [h.to_dict() for h in self.history]})

    # ------------------------------------------------------------------- loop
    def run(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.run_dir / "config.json", {
            "task_name": self.task_name,
            "model": self.cfg.model,
            "max_rounds": self.cfg.max_rounds,
            "model_py": self.model_py,
            "run_numeric": self.cfg.run_numeric,
            "ncnn_root": str(self.cfg.ncnn_root),
        })

        # Restore the ncnn source tree on ANY exit path (normal, exception, or
        # SIGTERM/SIGINT) so an interrupted run never leaves it dirty.
        def _on_signal(signum, _frame):
            raise KeyboardInterrupt(f"received signal {signum}")
        old_term = signal.signal(signal.SIGTERM, _on_signal)
        old_int = signal.signal(signal.SIGINT, _on_signal)

        result: GraphResult | None = None
        interrupted: str = ""
        grounding: dict = {}
        try:
            # GROUNDING ONLY: dump the op's real pnnx IR so the coder knows what to
            # match. pnnx emits the IR (incl. raw aten:: ops) even when it has NO
            # conversion pass for the op, so this works for a from-scratch op too.
            # NOTE: verification does NOT use any baseline — PyTorch is the oracle
            # (verify_numeric = allclose(torch_out, ncnn_out)). The baseline_supported
            # flag below is INFORMATIONAL ONLY and never gates authoring.
            grounding = probe_pnnx_ir(self.cfg, self.model_py, self.run_dir, self.task_name)
            write_json(self.run_dir / "pnnx_ir_probe.json", grounding)
            print(f"[agent] pnnx IR probe (grounding): op_types={grounding.get('op_types')} "
                  f"residual_aten={grounding.get('residual_aten')}")
            if grounding.get("baseline_supported") and not self.cfg.skip_if_supported:
                print("[agent] NOTE: current pnnx already converts this op correctly — "
                      "this is a poor from-scratch test case; authoring anyway.")
            if grounding.get("baseline_supported") and self.cfg.skip_if_supported:
                self._already_supported = True
                print("[agent] operator already supported; skip_if_supported=True -> stop.")
                raise _AlreadySupported()

            # identify (analyzer) then author the conversion from scratch.
            # Pass the pnnx probe so the analyzer grounds target_ncnn_layer on the
            # REAL ncnn layer types (avoids inventing e.g. "Log" for torch.log,
            # which actually folds into "UnaryOp").
            self.profile = self.analyze(grounding)
            if self.force_target_layer:
                self.profile.target_ncnn_layer = self.force_target_layer
                print(f"[agent] forced target ncnn layer = {self.force_target_layer}")
            code_book: dict[str, str] = {}

            for round_idx in range(self.cfg.max_rounds):
                if round_idx == 0:
                    phase = "identify_and_generate"
                    examples = retrieve_examples(self.cfg, self.profile,
                                                 op_types=grounding.get("op_types"),
                                                 residual_aten=grounding.get("residual_aten"))
                    prompt = coder_prompt(self.profile, examples, self.model_code, grounding=grounding,
                                          force_target=self.force_target_layer)
                else:
                    assert result is not None
                    phase = result.first_failure() or "identify_and_generate"
                    feedback = result.feedback(phase)
                    prompt = debugger_prompt(phase, self.profile, code_book, feedback, self._format_memory(),
                                             grounding=grounding, force_target=self.force_target_layer)

                response = self.llm(prompt, self.cfg.model)
                new_code = extract_code_blocks(response)
                if new_code:
                    code_book = {**code_book, **new_code}

                result = self._run_pipeline(code_book, round_idx)
                self._save_round(round_idx, phase, prompt, response, result)
                self._update_memory(round_idx, phase, result)

                print(f"[round {round_idx}] phase={phase} ok={result.ok} "
                      f"inject={result.inject_ok} build={result.build_ok} "
                      f"convert={result.convert_ok} structural={result.structural_ok} "
                      f"numeric={result.numeric_status}")
                if result.ok:
                    break
        except _AlreadySupported:
            pass  # handled via self._already_supported in the summary
        except (KeyboardInterrupt, Exception) as exc:  # noqa: BLE001
            interrupted = f"{type(exc).__name__}: {exc}"
            print(f"[agent] interrupted: {interrupted}")
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
            keep = bool(result and result.ok and self.cfg.keep_changes_on_success)
            if not keep:
                mutated = bool(self.session.created_files or self.session.modified_files)
                restore_files(self.cfg, self.session)
                print("[agent] source tree restored")
                if mutated:
                    # restore reverts SOURCE only; rebuild so the pnnx BINARY also
                    # matches the clean source (else a stale/broken binary leaks
                    # into the next run's IR probe).
                    build_pnnx(self.cfg, self.run_dir / "_teardown_build.log")
                    print("[agent] pnnx rebuilt from clean source")

        if self._already_supported:
            status = "already_supported"
        elif result and result.ok:
            status = "success"
        elif interrupted:
            status = "interrupted"
        else:
            status = "fail"
        summary = {
            "status": status,
            "task_name": self.task_name,
            "already_supported": self._already_supported,
            "baseline_ir": grounding if self._already_supported else {},
            "interrupted": interrupted,
            "rounds": len(self.history),
            "kept_changes": bool(result and result.ok and self.cfg.keep_changes_on_success),
            "op_profile": self.profile.to_dict() if self.profile else {},
            "history": [h.to_dict() for h in self.history],
            "final_result": result.to_dict() if result else {},
        }
        write_json(self.run_dir / "summary.json", summary)
        return summary
