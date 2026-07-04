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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # opgen/ for `import paths`

import json  # noqa: E402
import paths  # noqa: E402
from config import RUNS_ROOT, GraphConfig  # noqa: E402
from graph_schemas import write_json  # noqa: E402
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
        seed_code: dict[str, str] | None = None,    # e2e_repair: previously-LayerOracle-passed code
        seed_feedback: str | None = None,           # e2e_repair: failure detail from end-to-end stage
        seed_profile: dict | None = None,           # e2e_repair: re-use the analyzer's profile
        run_subdir_suffix: str = "",                # e2e_repair: write to runs/<task>/kernel<suffix>/
        force_analog_layer: str | None = None,      # baseline-probe-driven hard constraint:
                                                    # the actual ncnn layer type pnnx emits
    ) -> None:
        self.task_name = task_name
        self.cfg = cfg or GraphConfig()
        self.llm = llm_query or query_llm
        self.backend = backend
        self.base_kernel_code = base_kernel_code or {}
        self.base_profile_dict = base_profile or {}
        # e2e_repair seeding: when set, KernelAgent skips analyzer + round-0
        # coder and starts at round 0 with a debugger prompt that carries the
        # provided seed_code + seed_feedback. LayerOracle still validates the
        # repaired code, so the loop converges on something that BOTH oracles
        # accept (per-op + end-to-end).
        self.seed_code = dict(seed_code) if seed_code else None
        self.seed_feedback = seed_feedback
        self.seed_profile = dict(seed_profile) if seed_profile else None
        self.run_subdir_suffix = run_subdir_suffix
        self.force_analog_layer = (force_analog_layer or "").strip() or None
        # vulkan verifies on the GPU via VulkanLayerOracle (isolated instantiation,
        # runtime-compiled shader); base/arm use the CPU LayerOracle.
        if backend == "vulkan":
            self.oracle = VulkanLayerOracle()
        else:
            self.oracle = LayerOracle(ncnn_root=self.cfg.ncnn_root, workdir=RUNS_ROOT / "_oracle")
        # arm: compile the candidate against src/layer/arm helpers.
        # PACKING is kept OFF (elempack=1) for the arm LayerOracle so it matches
        # what NetOracle / production actually run: net_oracle_runner sets
        # opt.use_packing_layout=false, so inside a real ncnn::Net the arm layer
        # receives elempack=1 mats. Validating arm at elempack=4 (--packing 4)
        # forced the LLM to get the packed NC4HW4 broadcast path right — a path
        # that is never exercised downstream — which was the dominant source of
        # arm false-negatives (packed-broadcast port bugs). Aligning LayerOracle
        # with NetOracle (elempack=1 NEON, 4-wide over the contiguous axis) keeps
        # the arm kernel a real NEON override while removing that variance. A
        # future fp16+packing pass can validate the packed path in BOTH oracles.
        self._extra_includes = ([str(self.cfg.ncnn_root / "src" / "layer" / "arm")]
                                if backend == "arm" else [])
        self._packing = 0
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
        # New layout: base -> runs/<task>/base_kernel;
        #             arm/vulkan/... -> runs/<task>/backends/<backend>/kernel
        # The run_subdir_suffix (e.g. "_e2e_repair_1") is appended to the LAST
        # segment, giving "base_kernel_e2e_repair_1" or "kernel_e2e_repair_1"
        # placed as a sibling — preserving OperatorAgent's e2e repair semantics.
        if self.backend == "base":
            base = paths.base_kernel_dir(RUNS_ROOT, self.task_name)
        else:
            base = paths.backend_kernel_dir(RUNS_ROOT, self.task_name, self.backend)
        return base.parent / f"{base.name}{self.run_subdir_suffix}"

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
        prompt = analyzer_prompt(self.task_name, self.model_code, self.intro,
                                 force_analog_layer=self.force_analog_layer)
        text = self.llm(prompt, self.cfg.model)
        (self.run_dir / "analyzer.md").write_text(text, encoding="utf-8")
        profile = parse_profile_json(self.task_name, text)
        # Hard constraint: if the orchestrator told us what ncnn layer pnnx
        # actually emits for this op, overwrite the LLM's analog_layer guess.
        # The LLM often picks a "semantically nearer" layer (e.g. nn.Linear →
        # InnerProduct) while pnnx may emit a different one (Gemm). Using the
        # LLM pick makes LayerOracle pass but NetOracle fail because the
        # .ncnn.param schemas differ.
        if self.force_analog_layer and profile.analog_layer != self.force_analog_layer:
            print(f"[analyze] forcing analog_layer: LLM said {profile.analog_layer!r} "
                  f"-> baseline-probe truth {self.force_analog_layer!r}")
            profile.analog_layer = self.force_analog_layer
            profile.params = {}   # LLM-supplied params follow the wrong schema; reset
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
        # numeric_repair convergence trend: a flat or oscillating max_diff over
        # multiple rounds means the LLM is chasing the wrong cause. Give it the
        # numbers up front so it can recognise a plateau and switch strategy.
        diffs = [m.get("max_diff") for m in self.memory[-4:] if m.get("max_diff") is not None]
        if len(diffs) >= 2:
            arrow = " → ".join(f"{d:.4g}" for d in diffs)
            trend = "decreasing" if diffs[-1] < diffs[0] * 0.5 else (
                    "plateau"    if abs(diffs[-1] - diffs[0]) < 0.1 * diffs[0] else
                    "oscillating")
            out.append(f"numeric max_diff trend ({len(diffs)} recent rounds): "
                       f"{arrow}  [{trend}]")
            if trend in ("plateau", "oscillating"):
                out.append("  → your current line of fixes is NOT converging. "
                           "Consider a different root-cause hypothesis instead of "
                           "iterating on the same change.")
        for m in self.memory[-4:]:
            md = m.get("max_diff")
            md_str = f" max_diff={md:.4g}" if isinstance(md, (int, float)) else ""
            out.append(f"round {m['round']} phase={m['phase']} -> {m['stages']}{md_str}")
            if m.get("feedback"):
                out.append(f"  feedback: {m['feedback'][:500]}")
        return "\n".join(out)

    def _update_memory(self, idx: int, phase: str, result: KernelResult) -> None:
        self.memory.append({
            "round": idx, "phase": phase,
            "stages": {"compile": result.compile_ok, "numeric": result.numeric_status},
            "max_diff": result.max_diff,
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

    # --------------------------------------------------------- native vulkan
    def _ensure_baseline_probe(self) -> None:
        """Run pnnx once so a `.ncnn.param` exists under this task's run tree.

        Needed only when nobody else (GraphAgent / OperatorAgent baseline_probe)
        has already produced one — e.g. kernel-only mode. Written to
        `analyze/pnnx_probe/_probe/` under the task root, so it is SHARED by
        every backend (base/arm/vulkan/...) and only runs once per task. Also
        matches any legacy probe under `kernel*/_pnnx_probe/` via the check
        below (avoids re-probing when a legacy run already produced one).
        Idempotent: skips if any `.ncnn.param` exists anywhere under the task
        root. Silent on failure (caller keeps its pre-probe fallback).
        """
        # Fast path: any existing .ncnn.param anywhere under the task root
        # (analyze/pnnx_probe/, operator/_baseline_probe/, legacy kernel*/,
        # graph/, ...) is good enough — pnnx output is backend-invariant.
        task_root = paths.task_root(RUNS_ROOT, self.task_name)
        if task_root.exists() and list(task_root.rglob("*.ncnn.param")):
            return
        probe_root = paths.analyze_pnnx_probe_dir(RUNS_ROOT, self.task_name)
        # Lazy import: graph_pipeline pulls tree-sitter / cmake helpers that
        # the base/arm KernelAgent paths don't need at load time.
        try:
            from graph_pipeline import probe_pnnx_ir
        except Exception as exc:                # graph_pipeline import broken
            print(f"[kernel] pnnx probe unavailable ({exc}); native-vulkan may miss ncnn-schema params")
            return
        try:
            probe_root.mkdir(parents=True, exist_ok=True)
            probe_pnnx_ir(self.cfg, self.model_py, probe_root, self.task_name)
        except Exception as exc:                # pnnx binary / trace / convert failed
            print(f"[kernel] pnnx probe failed: {exc}; native-vulkan may miss ncnn-schema params")

    def _detect_pnnx_layer_type(self) -> str | None:
        """Scan the pnnx baseline .ncnn.param for the real layer type pnnx
        picked for THIS task's op.

        Returns the first non-Input/Output/Split layer name (e.g. "BinaryOp",
        "ConvolutionDepthWise", "Reduction"), or None if no baseline exists or
        the model has no real layer (only Input/Output). Triggers a lazy pnnx
        probe if needed (via _parse_baseline_params' machinery — cheap re-use).
        """
        # First try existing baseline files
        for p in (RUNS_ROOT / self.task_name).rglob("*.ncnn.param"):
            layer = self._first_real_layer(p)
            if layer:
                return layer
        # Nothing yet — trigger the same probe path _parse_baseline_params uses
        self._ensure_baseline_probe()
        for p in (RUNS_ROOT / self.task_name).rglob("*.ncnn.param"):
            layer = self._first_real_layer(p)
            if layer:
                return layer
        return None

    @staticmethod
    def _first_real_layer(param_path: Path) -> str | None:
        _SKIP = {"Input", "Output", "Split"}
        try:
            txt = param_path.read_text(encoding="utf-8")
        except OSError:
            return None
        for ln in txt.splitlines()[2:]:
            parts = ln.split()
            if parts and parts[0] not in _SKIP:
                return parts[0]
        return None

    def _parse_baseline_params(self, analog: str) -> dict | None:
        """Correct ncnn params for `analog` from the pnnx baseline .ncnn.param.

        The from-scratch base kernel's params are bespoke and may not match ncnn's
        real schema (e.g. Mul needs BinaryOp op_type=2, Reduction needs an axes
        array via -23303 + fixbug0=1) — but a native <analog>_vulkan subclass
        reads params via ncnn's own load_param, so they MUST be the ncnn schema.
        pnnx knows this mapping; we invoke it once (via _ensure_baseline_probe)
        and parse the analog layer line. Returns {id_str: val} (arrays as
        {positive_id: [values]}), or None if no baseline param found.
        """
        analog_lc = (analog or "").lower()
        roots = list((RUNS_ROOT / self.task_name).rglob("*.ncnn.param"))
        if not roots:
            # kernel-only / standalone: no GraphAgent has run for this task —
            # trigger a KernelAgent-local pnnx probe and re-scan.
            self._ensure_baseline_probe()
            roots = list((RUNS_ROOT / self.task_name).rglob("*.ncnn.param"))
        for p in roots:
            try:
                txt = p.read_text(encoding="utf-8")
            except OSError:
                continue
            for ln in txt.splitlines()[2:]:
                parts = ln.split()
                # Case-insensitive match: profile.analog_layer is often the
                # lowercase file stem (`binaryop`, `reduction`, `pooling`) while
                # pnnx-emitted layer names are PascalCase (`BinaryOp`, ...).
                if len(parts) >= 4 and parts[0].lower() == analog_lc:
                    try:
                        nin, nout = int(parts[2]), int(parts[3])
                    except ValueError:
                        continue
                    toks = parts[4 + nin + nout:]
                    params: dict[str, Any] = {}
                    for t in toks:
                        if "=" not in t:
                            continue
                        k, v = t.split("=", 1)
                        try:
                            ki = int(k)
                        except ValueError:
                            continue
                        if ki <= -23300:                    # array param (neg-key trick)
                            real_id = -ki - 23300
                            vals = v.split(",")[1:]         # drop the leading count
                            params[str(real_id)] = [
                                float(x) if ("." in x or "e" in x.lower()) else int(x)
                                for x in vals]
                        elif "." in v or "e" in v.lower():
                            params[k] = float(v)
                        else:
                            params[k] = int(v)
                    return params
        return None

    def _prime_profile_from_pnnx_baseline(self) -> None:
        """Vulkan scratch mode: pull pnnx's real analog + params into the profile
        WITHOUT generating a native subclass.

        Without this, the base-derived vulkan profile carries the analyzer's
        guess for `analog_layer` (e.g. "innerproduct" for nn.Linear) and empty
        params — that's fine for the base kernel path (from-scratch LLM math
        doesn't need pnnx's schema), but for VULKAN scratch we still want:
          (a) the correct interface-dict entry to inject into the prompt (so
              the LLM sees the RIGHT layer's param IDs)
          (b) concrete params for the model at hand (so the LLM doesn't have
              to guess numeric values it will read via pd.get)
          (c) the right shader/header/file basenames (KernelProfile derives
              these from class_name; class_name is stable, no change needed)
        This mirrors _native_vulkan_run's initial "detect + parse baseline"
        block but stops BEFORE synthesising the subclass header/cpp.
        """
        pnnx_layer = self._detect_pnnx_layer_type()
        if pnnx_layer and pnnx_layer.lower() != (self.profile.analog_layer or "").lower():
            print(f"[kernel] vulkan scratch: pnnx emitted `{pnnx_layer}` "
                  f"(analog was `{self.profile.analog_layer}`) — using pnnx's choice")
            self.profile.analog_layer = pnnx_layer
        parsed = self._parse_baseline_params(self.profile.analog_layer or "")
        if parsed is not None:
            self.profile.params = parsed
            print(f"[kernel] vulkan scratch: pnnx baseline params for "
                  f"`{self.profile.analog_layer}` = {parsed}")
        # NEVER set native_vulkan=True here — this is the from-scratch prime,
        # and the shader-existence gate in kernel_pipeline.py keys on
        # profile.native_vulkan to decide whether a .comp is required.
        self.profile.native_vulkan = False
        write_json(self.run_dir / "kernel_profile.json", self.profile.to_dict())

    def _native_vulkan_run(self) -> dict | None:
        """Native-vulkan-subclass path: `Cand_X_vulkan : public ncnn::<analog>_vulkan`.

        Returns a success summary if the thin subclass verifies on the GPU, else
        None to fall back to the from-scratch LLM shader loop.
        """
        analog = (self.profile.analog_layer or "").strip()
        if not analog or "." in analog:            # pnnx-only name -> no native vk
            return None
        # pnnx may lower a torch op to a MORE SPECIFIC ncnn layer than our base
        # profile's `analog` guessed — the poster child is grouped Conv2d, which
        # goes to `ConvolutionDepthWise` (not plain `Convolution`) once groups>1.
        # Whenever the pnnx baseline exists (or can be produced), prefer the
        # actual pnnx-emitted layer type: it has the exact params + weight
        # layout ncnn's <Op>_vulkan will read.
        pnnx_layer = self._detect_pnnx_layer_type()
        if pnnx_layer and pnnx_layer.lower() != analog.lower():
            print(f"[kernel] native-vulkan: pnnx emitted `{pnnx_layer}` for this op "
                  f"(analog was `{analog}`) — using pnnx's choice")
            analog = pnnx_layer
            # verify_kernel looks up weight_flags via `profile.analog_layer` in
            # the ncnn interface dict. If we override analog here but leave the
            # profile field on the base's guess (e.g. `innerproduct` when pnnx
            # actually chose `Gemm`), the runner sends flags from the WRONG
            # layer's schema — bias flag for InnerProduct is 1 (raw) but Gemm's
            # C_data reads with a tag, so the tag byte gets pulled from bias data
            # and the whole model bin is off by 4 bytes → silent numerical drift.
            self.profile.analog_layer = analog
        stem = analog.lower()
        hdr = self.cfg.ncnn_root / "src" / "layer" / "vulkan" / f"{stem}_vulkan.h"
        if not hdr.exists():
            return None                            # ncnn has no vulkan for this op
        # Functional ops (F.conv2d / F.linear) feed weights as forward INPUTS,
        # but ncnn's native <Layer>_vulkan reads them via load_model into an
        # opaque VkTensor + upload path — there is no vulkan interface for
        # runtime weight bottom_blobs (Convolution_vulkan explicitly does not
        # support dynamic_weight). Native subclass would crash on the extra
        # bottom_blobs. Fall back to from-scratch.
        if self.profile.is_functional:
            print(f"[kernel] native-vulkan: functional op (weights as inputs) — "
                  f"ncnn {analog}_vulkan does not accept runtime weights, "
                  f"skipping native subclass")
            return None
        # ncnn's `<Op>_vulkan` class name is PascalCase (BinaryOp_vulkan,
        # BatchNorm_vulkan, InnerProduct_vulkan, ...) — NOT what you get by
        # lower-casing the file stem. Grep the authoritative class name from
        # the header instead of guessing the casing. `analog_layer` in the
        # profile may be either case ("binaryop" from base as_backend, or
        # "BinaryOp" from the pnnx-driven analyzer); the file stem is always
        # lowercase but the class isn't.
        import re as _re
        try:
            hdr_txt = hdr.read_text(encoding="utf-8")
            m = _re.search(r"class\s+(\w+_vulkan)\s*:\s*public\s+\w+", hdr_txt)
            native_class = m.group(1) if m else f"{analog}_vulkan"
        except OSError:
            native_class = f"{analog}_vulkan"
        native_header = f"vulkan/{stem}_vulkan.h"

        # Params come from either (best) the pnnx baseline .ncnn.param (which
        # ONLY exists after the full operator pipeline ran pnnx — kernel-only
        # mode has none) or (fallback) the base-kernel profile that this vulkan
        # subclass was derived from (as_backend copies params over). If BOTH
        # are absent we can't safely instantiate the native layer: BinaryOp
        # defaults op_type=0 (ADD) so a Mul/Greater/AND that hits this path
        # silently computes ADD.
        parsed = self._parse_baseline_params(analog)
        if parsed is not None:
            # pnnx is INTERNALLY CONSISTENT: the axes it writes to .ncnn.param
            # and the input feed policy it writes to <task>_ncnn.py agree on
            # the same rank. For batch_index=0 white-listed ops (Conv/BN/...)
            # pnnx rebases axes down and squeezes(0) the input; for
            # batch_index=233 isolated ops (torch.sum / F.softmax alone) it
            # keeps axes at the pytorch rank and keeps the input as-is (4D).
            # verify_kernel's pnnx_driven_ncnn_inputs mirrors that squeeze
            # decision (see kernel_pipeline.py:_PROBE_SUBS), so ncnn receives
            # a rank consistent with the axes pnnx wrote — no per-op rebase
            # needed here. A blanket `axes[i]-1` used to live here to patch
            # the 4D-vs-3D mismatch caused by verify_kernel not finding our
            # lazy pnnx probe; that mismatch is now fixed at the source.
            self.profile.params = parsed
        elif not self.profile.params:
            # No baseline + no inherited params → refuse. Matches the And case
            # (pnnx keeps torch.logical_and unlowered → no `binaryop` line +
            # base profile has params={} → we correctly bail out).
            print(f"[kernel] native-vulkan: no baseline .ncnn.param for `{analog}` "
                  f"and inherited params are empty — refusing native subclass to "
                  f"avoid default-op-type semantic mismatch")
            return None
        else:
            print(f"[kernel] native-vulkan: no pnnx baseline for `{analog}`, "
                  f"reusing base-profile params {self.profile.params}")
        self.profile.native_vulkan = True
        self.profile.native_vulkan_class = native_class
        self.profile.native_vulkan_header = native_header
        self.profile.shader = ""

        cls, hname, fname = self.profile.class_name, self.profile.header, self.profile.file
        guard = cls.upper().replace(".", "_") + "_H"
        code_book = {
            hname: (f"#ifndef {guard}\n#define {guard}\n"
                    f'#include "{native_header}"\n'
                    f"namespace ncnn {{ class {cls} : public {native_class} {{}}; }}\n"
                    f"#endif\n"),
            fname: (f'#include "{hname}"\n'
                    f"namespace ncnn {{ DEFINE_LAYER_CREATOR({cls}) }}\n"),
        }
        print(f"[kernel] vulkan NATIVE-SUBCLASS: {cls} : public {native_class} "
              f"(inherits ncnn baked shader; no from-scratch .comp). "
              f"params={self.profile.params}")
        write_json(self.run_dir / "kernel_profile.json", self.profile.to_dict())
        # Write the native round to its OWN dir (not round_00) so that fallback
        # from-scratch rounds don't overwrite the native runner's compile/numeric
        # log — otherwise every native-fail case looks like "kernel crashed at
        # runtime:\n" (the from-scratch round_00 result), hiding the real error.
        native_rd = self.run_dir / "round_native"
        native_rd.mkdir(parents=True, exist_ok=True)
        result = verify_kernel(self.oracle, self.profile, code_book, self.model_py,
                               native_rd, run_numeric=self.cfg.run_numeric,
                               base_files=(self.base_kernel_code if self._subclasses_base else None),
                               extra_includes=tuple(self._extra_includes),
                               packing=self._packing)
        (native_rd / "prompt.md").write_text("(native subclass, no LLM)", encoding="utf-8")
        (native_rd / "response.md").write_text("", encoding="utf-8")
        write_json(native_rd / "result.json", result.to_dict())
        self.history.append(KernelRound(
            round_idx=-1, phase="native_vulkan",
            prompt_path=str(native_rd / "prompt.md"),
            response_path=str(native_rd / "response.md"),
            result_path=str(native_rd / "result.json"),
            ok=result.ok,
            stages={"compile": result.compile_ok, "numeric": result.numeric_status},
        ))
        write_json(self.run_dir / "history.json", {"history": [h.to_dict() for h in self.history]})
        print(f"[kernel] native-vulkan: ok={result.ok} compile={result.compile_ok} "
              f"numeric={result.numeric_status} max_diff={result.max_diff}")
        if not result.ok:
            print(f"[kernel] native-vulkan did not verify (see {native_rd}/result.json); "
                  f"falling back to from-scratch shader")
            return None
        summary = {
            "status": "success", "task_name": self.task_name, "backend": self.backend,
            "rounds": len(self.history),
            "kernel_profile": self.profile.to_dict(),
            "history": [h.to_dict() for h in self.history],
            "final_result": result.to_dict(),
        }
        write_json(self.run_dir / "summary.json", summary)
        return summary

    # ------------------------------------------------------------------- loop
    def run(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # SHARED introspect cache: introspect_model is a pure function of the
        # PyTorch model file (shape / state_dict / init_inputs) — safe to cache
        # under analyze/ and reuse across all backends. First backend for this
        # task populates the cache; the rest read.
        intro_cache = paths.introspect_json(RUNS_ROOT, self.task_name)
        if intro_cache.exists():
            try:
                self.intro = json.loads(intro_cache.read_text(encoding="utf-8"))
            except Exception:
                self.intro = introspect_model(self.model_py)
        else:
            self.intro = introspect_model(self.model_py)
            intro_cache.parent.mkdir(parents=True, exist_ok=True)
            write_json(intro_cache, self.intro)
        # Keep a per-backend copy inside run_dir for post-mortem visibility
        # (small file; harmless duplication).
        write_json(self.run_dir / "introspect.json", self.intro)
        write_json(self.run_dir / "config.json", {
            "task_name": self.task_name, "model": self.cfg.model,
            "max_rounds": self.cfg.max_rounds, "model_py": self.model_py,
            "run_numeric": self.cfg.run_numeric,
        })

        if self.seed_profile is not None:
            # e2e_repair: re-use the profile that LayerOracle already accepted —
            # we are not re-analyzing, we are repairing the code on top of it.
            self.profile = KernelProfile.from_llm(self.task_name, self.seed_profile,
                                                  backend=self.backend)
            # Apply force_analog here too (analyze() isn't called on the seed
            # path). The seed profile may have used the wrong analog — likely
            # WHY e2e failed — so let baseline-probe truth override and reset
            # params to be re-derived from the (correct) dict entry.
            if self.force_analog_layer and self.profile.analog_layer != self.force_analog_layer:
                print(f"[e2e_repair] forcing analog_layer: seed said "
                      f"{self.profile.analog_layer!r} -> {self.force_analog_layer!r}")
                self.profile.analog_layer = self.force_analog_layer
                self.profile.params = {}
                self._infer_params(self.profile)
            write_json(self.run_dir / "kernel_profile.json", self.profile.to_dict())
        elif self._subclasses_base:
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

        # VULKAN dispatch modes (cfg.vulkan_mode):
        #   scratch       → agent authors .h+.cpp+.comp from scratch. The pnnx
        #                   baseline is still consulted by `_parse_baseline_params`
        #                   so the profile carries pnnx-emitted analog + params
        #                   (avoids default-op-type semantic drift), but we DO NOT
        #                   subclass ncnn's built-in <Op>_vulkan. This is the goal
        #                   configuration — every completed run means the LLM
        #                   actually wrote a working GLSL shader.
        #   native_first  → try native-subclass first; on non-verify fall through
        #                   to from-scratch. Useful for validation-vs-agent A/B.
        #   native_only   → legacy: only run the native path (miniset/subset audit
        #                   used this; agent shader authoring rate is 0).
        if self.backend == "vulkan":
            mode = getattr(self.cfg, "vulkan_mode", "scratch")
            if mode not in ("scratch", "native_first", "native_only"):
                raise ValueError(f"unknown cfg.vulkan_mode={mode!r}")
            if mode == "scratch":
                # Consult the pnnx baseline once so the profile has the right
                # analog + params, then throw the native subclass away and let
                # the LLM loop take over below.
                self._prime_profile_from_pnnx_baseline()
                print(f"[kernel] vulkan mode=scratch → agent will author "
                      f"{self.profile.header} + {self.profile.file} + "
                      f"{self.profile.shader} (no native subclass)")
            elif mode in ("native_first", "native_only"):
                native = self._native_vulkan_run()
                if native is not None:
                    return native
                if mode == "native_only":
                    print(f"[kernel] vulkan mode=native_only → refusing to fall "
                          f"back to from-scratch; returning failure summary")
                    summary = {
                        "status": "fail", "task_name": self.task_name,
                        "backend": self.backend, "rounds": len(self.history),
                        "kernel_profile": self.profile.to_dict(),
                        "history": [h.to_dict() for h in self.history],
                        "final_result": {},
                        "error": "native_only mode: no native vulkan subclass verified",
                    }
                    write_json(self.run_dir / "summary.json", summary)
                    return summary
                # native_first fell through — LLM loop below authors from scratch

        result: KernelResult | None = None
        # e2e_repair: prefill code_book with the seeded code so the first round's
        # debugger prompt shows the LLM what it previously wrote (and what e2e
        # then rejected). The loop's first iteration emits a debugger prompt,
        # not the cold-start coder prompt.
        code_book: dict[str, str] = dict(self.seed_code) if self.seed_code else {}
        for idx in range(self.cfg.max_rounds):
            if idx == 0 and self.seed_code is None:
                phase = "identify_and_generate"
                prompt = coder_prompt(self.profile, example, self.model_code, self.intro)
            elif idx == 0 and self.seed_code is not None:
                # e2e_repair entry: jump straight to numeric_repair with the
                # provided seed_feedback. Label it so the LLM and the saved
                # round/memory clearly show this is end-to-end driven.
                phase = "numeric_repair"
                feedback = ("[E2E_REPAIR] The kernel below passes the per-op "
                            "LayerOracle (numeric == PyTorch in isolation), but "
                            "the end-to-end NetOracle run reports:\n"
                            + (self.seed_feedback or "(no detail provided)"))
                prompt = debugger_prompt(phase, self.profile, code_book, feedback,
                                         self._format_memory(), self.intro)
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

        # BASE-KERNEL SoT CONTRACT: on a successful base run (not an e2e_repair
        # variant), publish the final .h/.cpp + shared kernel_profile.json into
        # `runs/<task>/base_kernel/artifacts/`. arm/vulkan/optimize readers pull
        # from THIS dir instead of digging into round_XX/ or summary.json.
        # Also mirror the backend-invariant subset of the profile into
        # `analyze/kernel_profile.json` so future backends can seed their own
        # profile without re-running the analyzer.
        if (self.backend == "base" and self.run_subdir_suffix == ""
                and result and result.ok and self.profile):
            self._publish_base_artifacts(result)
        return summary

    def _publish_base_artifacts(self, result: KernelResult) -> None:
        """Copy the final base .h/.cpp + profile into the shared artifacts dir.
        Also strip backend-specific fields from the profile and write it to
        analyze/kernel_profile.json for cross-backend reuse."""
        art = paths.base_kernel_artifacts_dir(RUNS_ROOT, self.task_name)
        art.mkdir(parents=True, exist_ok=True)
        code = (result.response_code or {})
        for name, body in code.items():
            if name.endswith((".h", ".hpp", ".cpp", ".cc", ".cxx")):
                (art / name).write_text(body, encoding="utf-8")
        prof_dict = self.profile.to_dict()
        write_json(art / "kernel_profile.json", prof_dict)
        # Shared analyze/kernel_profile.json — backend-invariant fields only
        shared = paths.strip_backend_fields(prof_dict)
        write_json(paths.kernel_profile_shared_json(RUNS_ROOT, self.task_name), shared)
