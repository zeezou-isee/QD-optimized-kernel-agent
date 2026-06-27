"""KernelAgent: write an ncnn base (non-optimized) layer kernel from scratch and
verify it against PyTorch via LayerOracle.

Architecture mirrors GraphAgent: agent loop (state machine) + functional pipeline
+ 3 roles (analyzer / coder / debugger). Standalone — it never edits the ncnn
source tree (the kernel lives in the run dir and is compiled by LayerOracle).

    agent = KernelAgent(task_name="Abs")
    summary = agent.run()
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RUNS_ROOT, GraphConfig
from graph_schemas import write_json
from kernel_pipeline import (
    extract_kernel_code,
    introspect_model,
    retrieve_layer_example,
    verify_kernel,
)
from kernel_prompts import analyzer_prompt, coder_prompt, debugger_prompt, parse_profile_json
from kernel_schemas import KernelProfile, KernelResult, KernelRound
from layer_oracle import LayerOracle
from llm_api import query_llm


class KernelAgent:
    def __init__(
        self,
        *,
        task_name: str,
        model_py: str | Path | None = None,
        model_code: str | None = None,
        cfg: GraphConfig | None = None,
        llm_query: Callable[[str, str], str] | None = None,
        backend: str = "base",                      # "base" | "arm"
        base_kernel_code: dict[str, str] | None = None,   # arm: verified base files
        base_profile: dict | None = None,           # arm: the base KernelProfile dict
        native_class: str = "",                      # 方案C: overwrite this native ncnn layer
    ) -> None:
        self.task_name = task_name
        self.cfg = cfg or GraphConfig()
        self.llm = llm_query or query_llm
        self.backend = backend
        self.base_kernel_code = base_kernel_code or {}
        self.base_profile_dict = base_profile or {}
        self.native_class = native_class
        # In native-override mode, load the native layer's source so the coder can
        # match its param ids + semantics (the production model feeds native params).
        self.native_src = self._load_native_src(native_class) if native_class else {}
        self.oracle = LayerOracle(ncnn_root=self.cfg.ncnn_root, workdir=RUNS_ROOT / "_oracle")
        # arm: compile the candidate against src/layer/arm helpers + NC4HW4 packing.
        self._extra_includes = ([str(self.cfg.ncnn_root / "src" / "layer" / "arm")]
                                if backend == "arm" else [])
        self._packing = 4 if backend == "arm" else 0
        self.profile: KernelProfile | None = None
        self.intro: dict | None = None
        self.history: list[KernelRound] = []
        self.memory: list[dict] = []
        self.model_py, self.model_code = self._resolve_model(model_py, model_code)

    # ------------------------------------------------------------------ setup
    @property
    def run_dir(self) -> Path:
        sub = "kernel" if self.backend == "base" else f"kernel_{self.backend}"
        return RUNS_ROOT / self.task_name / sub

    def _resolve_model(self, model_py, model_code) -> tuple[str, str]:
        if model_py:
            p = Path(model_py)
            return str(p), p.read_text(encoding="utf-8")
        if model_code:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            p = self.run_dir / f"{self.task_name}.py"
            p.write_text(model_code, encoding="utf-8")
            return str(p), model_code
        root = self.cfg.dataset_root
        if root:
            matches = sorted(Path(root).rglob(f"{self.task_name}.py"))
            if matches:
                return str(matches[0]), matches[0].read_text(encoding="utf-8")
        raise FileNotFoundError(f"No model provided and {self.task_name}.py not found under {root}.")

    def _load_native_src(self, native_class: str) -> dict[str, str]:
        """Read the native ncnn layer's base .h/.cpp (src/layer/<name>.{h,cpp}) so the
        coder can match its param ids and semantics for native-override mode."""
        name = native_class.lower()
        layer_dir = self.cfg.ncnn_root / "src" / "layer"
        out: dict[str, str] = {}
        for suffix in (".h", ".cpp"):
            p = layer_dir / f"{name}{suffix}"
            if p.exists():
                out[p.name] = p.read_text(encoding="utf-8", errors="replace")
        return out

    # ------------------------------------------------------------------ roles
    def analyze(self) -> KernelProfile:
        prompt = analyzer_prompt(self.task_name, self.model_code, self.intro)
        text = self.llm(prompt, self.cfg.model)
        (self.run_dir / "analyzer.md").write_text(text, encoding="utf-8")
        profile = parse_profile_json(self.task_name, text)
        write_json(self.run_dir / "kernel_profile.json", profile.to_dict())
        return profile

    # ------------------------------------------------------------------ memory
    def _format_memory(self) -> str:
        if not self.memory:
            return "(none)"
        out = []
        for m in self.memory[-4:]:
            out.append(f"round {m['round']} phase={m['phase']} -> {m['stages']}")
            if m.get("feedback"):
                out.append(f"  feedback: {m['feedback'][:500]}")
        return "\n".join(out)

    def _update_memory(self, idx: int, phase: str, result: KernelResult) -> None:
        self.memory.append({
            "round": idx, "phase": phase,
            "stages": {"compile": result.compile_ok, "numeric": result.numeric_status},
            "feedback": result.feedback(result.first_failure() or ""),
        })
        write_json(self.run_dir / "memory.json", {"memory": self.memory})

    # ------------------------------------------------------------- pipeline
    def _run_pipeline(self, code_book: dict[str, str], idx: int) -> KernelResult:
        assert self.profile is not None
        rd = self.run_dir / f"round_{idx:02d}"
        return verify_kernel(self.oracle, self.profile, code_book, self.model_py, rd,
                             run_numeric=self.cfg.run_numeric,
                             base_files=(self.base_kernel_code if self.backend == "arm" else None),
                             extra_includes=tuple(self._extra_includes),
                             packing=self._packing)

    def _save_round(self, idx: int, phase: str, prompt: str, response: str, result: KernelResult) -> None:
        rd = self.run_dir / f"round_{idx:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "prompt.md").write_text(prompt, encoding="utf-8")
        (rd / "response.md").write_text(response, encoding="utf-8")
        write_json(rd / "result.json", result.to_dict())
        self.history.append(KernelRound(
            round_idx=idx, phase=phase,
            prompt_path=str(rd / "prompt.md"), response_path=str(rd / "response.md"),
            result_path=str(rd / "result.json"), ok=result.ok,
            stages={"compile": result.compile_ok, "numeric": result.numeric_status},
        ))
        write_json(self.run_dir / "history.json", {"history": [h.to_dict() for h in self.history]})

    # ------------------------------------------------------------------- loop
    def run(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.intro = introspect_model(self.model_py)
        write_json(self.run_dir / "introspect.json", self.intro)
        write_json(self.run_dir / "config.json", {
            "task_name": self.task_name, "model": self.cfg.model,
            "max_rounds": self.cfg.max_rounds, "model_py": self.model_py,
            "run_numeric": self.cfg.run_numeric,
        })

        if self.backend == "arm":
            # derive the arm profile from the verified base (no second analyzer call)
            base_prof = KernelProfile.from_llm(self.task_name, self.base_profile_dict, backend="base")
            self.profile = base_prof.as_backend("arm")
            write_json(self.run_dir / "kernel_profile.json", self.profile.to_dict())
        else:
            self.profile = self.analyze()
        example = retrieve_layer_example(self.cfg.ncnn_root, self.profile.analog_layer,
                                         backend=self.backend)
        print(f"[kernel] backend={self.backend} profile: class={self.profile.class_name} "
              f"analog={self.profile.analog_layer} weight_keys={self.profile.weight_keys} "
              f"params={self.profile.params}")

        result: KernelResult | None = None
        code_book: dict[str, str] = {}
        for idx in range(self.cfg.max_rounds):
            if idx == 0:
                phase = "identify_and_generate"
                prompt = coder_prompt(self.profile, example, self.model_code, self.intro,
                                      native_class=self.native_class, native_src=self.native_src)
            else:
                assert result is not None
                phase = result.first_failure() or "identify_and_generate"
                feedback = result.feedback(phase)
                prompt = debugger_prompt(phase, self.profile, code_book, feedback,
                                         self._format_memory(), self.intro,
                                         native_class=self.native_class, native_src=self.native_src)

            response = self.llm(prompt, self.cfg.model)
            new_code = extract_kernel_code(response)
            if new_code:
                code_book = {**code_book, **new_code}

            result = self._run_pipeline(code_book, idx)
            self._save_round(idx, phase, prompt, response, result)
            self._update_memory(idx, phase, result)
            print(f"[round {idx}] phase={phase} ok={result.ok} compile={result.compile_ok} "
                  f"numeric={result.numeric_status} max_diff={result.max_diff}")
            if result.ok:
                break

        summary = {
            "status": "success" if (result and result.ok) else "fail",
            "task_name": self.task_name,
            "backend": self.backend,
            "rounds": len(self.history),
            "kernel_profile": self.profile.to_dict() if self.profile else {},
            "history": [h.to_dict() for h in self.history],
            "final_result": result.to_dict() if result else {},
        }
        write_json(self.run_dir / "summary.json", summary)
        return summary
