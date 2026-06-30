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
from typing import Any, Callable

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
from layer_oracle import LayerOracle, VulkanLayerOracle
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
        backend: str = "base",                      # "base" | "arm" | "vulkan"
        base_kernel_code: dict[str, str] | None = None,   # arm/vulkan: verified base files
        base_profile: dict | None = None,           # arm/vulkan: the base KernelProfile dict
    ) -> None:
        self.task_name = task_name
        self.cfg = cfg or GraphConfig()
        self.llm = llm_query or query_llm
        self.backend = backend
        self.base_kernel_code = base_kernel_code or {}
        self.base_profile_dict = base_profile or {}
        # vulkan verifies on the GPU via VulkanLayerOracle (isolated instantiation,
        # runtime-compiled shader); base/arm use the CPU LayerOracle.
        if backend == "vulkan":
            self.oracle = VulkanLayerOracle()
        else:
            self.oracle = LayerOracle(ncnn_root=self.cfg.ncnn_root, workdir=RUNS_ROOT / "_oracle")
        # arm: compile the candidate against src/layer/arm helpers + NC4HW4 packing.
        # `_packing` defaults to elempack=4 on arm, BUT for functional ops (weights
        # arriving as bottom_blobs) we keep elempack=1 — the runner's
        # convert_packing() would otherwise repack the weight tensor too and
        # destroy its PyTorch layout. Finalised in run() once the profile is known.
        self._extra_includes = ([str(self.cfg.ncnn_root / "src" / "layer" / "arm")]
                                if backend == "arm" else [])
        self._packing = 4 if backend == "arm" else 0
        # arm and vulkan both subclass the verified base layer
        self._subclasses_base = backend in ("arm", "vulkan")
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

    # ------------------------------------------------------------------ roles
    def analyze(self) -> KernelProfile:
        prompt = analyzer_prompt(self.task_name, self.model_code, self.intro)
        text = self.llm(prompt, self.cfg.model)
        (self.run_dir / "analyzer.md").write_text(text, encoding="utf-8")
        profile = parse_profile_json(self.task_name, text)
        self._validate_weight_keys(profile)
        self._infer_params(profile)
        write_json(self.run_dir / "kernel_profile.json", profile.to_dict())
        return profile

    def _validate_weight_keys(self, profile: KernelProfile) -> None:
        """Auto-correct weight_keys against the model's actual state_dict.

        LLMs occasionally hallucinate truncated weight key names
        (e.g. ``weight`` instead of ``linear.weight``). This method matches
        each profile weight_key against the real state_dict and corrects
        mismatches before the compile step tries to load them.
        """
        if not profile.weight_keys:
            return  # weightless op — nothing to validate

        sd = (self.intro or {}).get("state_dict") or {}
        if not sd:
            return  # no state_dict in introspect (shouldn't happen if weight_keys is set)

        corrected: list[str] = []
        for wk in profile.weight_keys:
            if wk in sd:
                corrected.append(wk)
                continue
            # exact match failed — try suffix / fuzzy match
            candidates = [k for k in sd if k.endswith("." + wk) or k == wk]
            if len(candidates) == 1:
                print(f"[analyze] corrected weight_key: '{wk}' -> '{candidates[0]}'")
                corrected.append(candidates[0])
            elif len(candidates) > 1:
                # pick the shortest match (most direct)
                best = min(candidates, key=len)
                print(f"[analyze] corrected weight_key (ambiguous): '{wk}' -> '{best}' "
                      f"(from {candidates})")
                corrected.append(best)
            else:
                # no match at all — keep original (let compile step report error)
                print(f"[analyze] WARNING: weight_key '{wk}' not found in state_dict "
                      f"{list(sd)}; keeping as-is")
                corrected.append(wk)

        if corrected != profile.weight_keys:
            profile.weight_keys = corrected
            # persist the corrected profile immediately
            write_json(self.run_dir / "kernel_profile.json", profile.to_dict())

    def _infer_params(self, profile: KernelProfile) -> None:
        """Infer ncnn param values from the model state_dict, driven by the
        ncnn layer interface dictionary.

        LLMs cannot reliably map PyTorch semantics to ncnn param IDs. Rather
        than hand-coding the mapping per op family (the prior approach, which
        covered only ~5 of ~110 layers and mis-mapped LayerNorm), we:

          1. look up the analog ncnn layer in the interface dictionary
             (produced by `opgen/ncnn_interface/extract_layer_interfaces.py`),
          2. for each param the layer declares, resolve a value from the
             state_dict using a small set of well-known var-name patterns
             (`num_output`, `weight_data_size`, `bias_term`, `affine_size`,
             `channels`, `num_slope`, `hidden_size`, ...).

        Resolved values override LLM guesses (same policy as the old code).
        When the dictionary is missing the layer, or a pattern isn't
        registered for a var name, we leave that param alone — the LLM's
        value (or the ncnn default) is used.
        """
        # late-import keeps this independent of test setups that may not have
        # bootstrap_paths called
        try:
            from lookup import derive_params_from_dict, get_interface
        except ImportError:
            return

        sd = (self.intro or {}).get("state_dict") or {}
        analog = profile.analog_layer or ""

        iface = get_interface(analog)
        if not iface:
            return                                # unknown analog → free pass

        # Functional ops (F.conv2d, F.linear, ...) have an empty state_dict but
        # their weights ride in on forward inputs. Synthesize a state_dict from
        # the weights-from-inputs shapes so the dict-driven param resolver can
        # still compute num_output / weight_data_size / bias_term / etc.
        # Convention: the first weight-input is "weight", the second is "bias".
        wfi = list(profile.weights_from_inputs or [])
        if not sd and wfi:
            shapes = (self.intro or {}).get("input_shapes") or []
            tags = ("weight", "bias", "running_mean", "running_var")
            sd = {}
            for k, src_idx in enumerate(wfi):
                if src_idx < len(shapes) and k < len(tags):
                    sd[tags[k]] = list(shapes[src_idx])
            if sd:
                print(f"[analyze] functional op detected — synthesized state_dict "
                      f"from input shapes: {sd}")

        computed = derive_params_from_dict(analog, sd)
        if not computed:
            return

        # log corrections that disagree with the LLM
        for pid, val in computed.items():
            llm_val = profile.params.get(pid)
            if llm_val is not None and str(llm_val) != str(val):
                print(f"[analyze] param {pid} corrected: LLM={llm_val} -> "
                      f"dict-derived={val}")
        profile.params = {**profile.params, **computed}
        print(f"[analyze] dict-driven params for analog={iface['name']}: {computed}")
        write_json(self.run_dir / "kernel_profile.json", profile.to_dict())

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
                             base_files=(self.base_kernel_code if self._subclasses_base else None),
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

        if self._subclasses_base:
            # derive the arm/vulkan profile from the verified base (no 2nd analyzer call)
            base_prof = KernelProfile.from_llm(self.task_name, self.base_profile_dict, backend="base")
            self.profile = base_prof.as_backend(self.backend)
            write_json(self.run_dir / "kernel_profile.json", self.profile.to_dict())
        else:
            self.profile = self.analyze()
        example = retrieve_layer_example(self.cfg.ncnn_root, self.profile.analog_layer,
                                         backend=self.backend)
        # Functional ops + arm: keep elempack=1 (the runner's convert_packing()
        # would otherwise pack the weight bottom_blob too and break its PyTorch
        # layout). LLM still uses NEON intrinsics inside forward — just over
        # unpacked inputs. See _functional_routing_note for the kernel-side rules.
        if self.backend == "arm" and self.profile.is_functional:
            print(f"[kernel] arm + functional op detected → forcing packing=0 "
                  f"(elempack=1) to keep weight bottom_blobs in PyTorch layout")
            self._packing = 0
        print(f"[kernel] backend={self.backend} profile: class={self.profile.class_name} "
              f"analog={self.profile.analog_layer} weight_keys={self.profile.weight_keys} "
              f"params={self.profile.params}")

        result: KernelResult | None = None
        code_book: dict[str, str] = {}
        for idx in range(self.cfg.max_rounds):
            if idx == 0:
                phase = "identify_and_generate"
                prompt = coder_prompt(self.profile, example, self.model_code, self.intro)
            else:
                assert result is not None
                phase = result.first_failure() or "identify_and_generate"
                feedback = result.feedback(phase)
                prompt = debugger_prompt(phase, self.profile, code_book, feedback,
                                         self._format_memory(), self.intro)

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
