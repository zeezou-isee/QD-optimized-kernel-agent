"""AdapterAgent: make an algorithm-correct ncnn custom layer satisfy the
ncnn Layer-Net contract so it runs correctly inside ``ncnn::Net``.

Motivation
----------
KernelAgent produces a layer that is *mathematically* correct (its per-op
LayerOracle is green). But a layer can be numerically right in the sandbox yet
violate the ncnn Layer-Net contract — wrong ``mb.load`` type for a weight,
wrong forward overload for its flags, a forward path that assumes a squeezed
1-D input, param IDs that don't match what pnnx emitted, etc. Those bugs only
surface at end-to-end (NetOracle) time. Historically we fed the e2e failure
back to a *fresh KernelAgent*, which re-derived the whole kernel from a generic
"shape/value mismatch" hint and kept guessing.

The AdapterAgent replaces that guess-driven repair with a contract-driven one.
It is handed, in-context:
  * the authoritative ncnn Layer-Net contract spec (ncnn_contract.md),
  * the target layer's interface (param IDs / weight load order / flags) from
    the interface dictionary,
  * the *actual* built-in ncnn implementation of that layer as a reference,
  * the *actual* ``.ncnn.param`` line the graph will feed the layer at runtime
    (so it sees the exact param IDs/values, e.g. constantA/constantB/transB),
  * the current candidate code and the concrete e2e failure.

It is instructed to identify *which* contract rule (C1–C6) or *which* reference
line was violated and fix exactly that — not to redesign the algorithm and not
to guess. The output is a ``{filename: code}`` book in the same format as
KernelAgent's, so the orchestrator can install it unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from config import RUNS_ROOT
from kernel_pipeline import extract_kernel_code, retrieve_layer_example
from llm_api import query_llm
from lookup import get_interface, render_for_prompt


# The contract spec lives next to the interface dict (ncnn_interface/).
_CONTRACT_PATH = (Path(__file__).resolve().parents[1]
                  / "ncnn_interface" / "ncnn_contract.md")


def _load_contract() -> str:
    try:
        return _CONTRACT_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


_SYSTEM = (
    "You are an ncnn integration adapter. You take a custom ncnn layer whose "
    "math is ALREADY CORRECT and rewrite it so it satisfies the ncnn Layer-Net "
    "contract exactly, so it runs correctly inside ncnn::Net. You never redesign "
    "the algorithm. When something is wrong you cite the specific contract rule "
    "(C1-C6) or the specific line of the reference built-in implementation that "
    "the candidate violates, then fix exactly that. You do not guess."
)


class AdapterAgent:
    def __init__(
        self,
        *,
        task_name: str,
        target_layer: str,            # ncnn layer the graph emits (e.g. "Gemm")
        class_name: str,              # the candidate class (e.g. "Cand_Gemm")
        ncnn_root: str | Path,
        llm_query: Callable[[str, str], str] | None = None,
        model: str = "deepseek-v4-pro",
        run_dir: str | Path | None = None,
    ) -> None:
        self.task_name = task_name
        self.target_layer = target_layer or ""
        self.class_name = class_name
        self.ncnn_root = Path(ncnn_root)
        self.llm = llm_query or query_llm
        self.model = model
        self.run_dir = Path(run_dir) if run_dir else (RUNS_ROOT / task_name / "adapter")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._contract = _load_contract()

    # ------------------------------------------------------------------
    def _reference_impl(self) -> str:
        """The built-in ncnn implementation of the target layer, as a reference
        for the exact load_param/load_model/forward contract. Empty if unknown."""
        if not self.target_layer:
            return ""
        try:
            files = retrieve_layer_example(self.ncnn_root, self.target_layer,
                                           max_files=1, backend="base")
        except Exception:  # noqa: BLE001
            return ""
        if not files:
            return ""
        blocks = []
        for name, code in files.items():
            blocks.append(f"// ncnn/src/layer/{name}\n{code}")
        return "\n\n".join(blocks)

    def _interface_block(self) -> str:
        # exact-name first, then fall through to the dict's case-insensitive match
        block = render_for_prompt(self.target_layer, role="kernel")
        return block

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        code_book: dict[str, str],
        *,
        ncnn_param_text: str,
        e2e_detail: str,
        input_shapes: Any,
        expected_out_shape: Any,
        attempt: int,
    ) -> str:
        cur = "\n\n".join(
            f"{name}\n```cpp\n{code}```" for name, code in code_book.items()
        )
        ref = self._reference_impl()
        iface = self._interface_block()
        contract = self._contract

        parts: list[str] = []
        parts.append(_SYSTEM)
        parts.append(
            f"\n## Task\nAdapt the candidate layer `{self.class_name}` "
            f"(operator `{self.task_name}`) so it satisfies the ncnn Layer-Net "
            f"contract for an ncnn **{self.target_layer}**-shaped layer. The "
            f"algorithm is already numerically correct in the per-op sandbox; "
            f"the bug is a contract violation that only shows up inside "
            f"ncnn::Net (end-to-end)."
        )
        parts.append(
            "\n## End-to-end failure to fix\n"
            f"attempt {attempt}\n"
            f"input shapes (torch): {input_shapes}\n"
            f"expected ncnn output shape (batch dropped): {expected_out_shape}\n"
            f"NetOracle detail: {e2e_detail}"
        )
        if ncnn_param_text.strip():
            parts.append(
                "\n## The ACTUAL .ncnn.param the graph feeds this layer\n"
                "These are the EXACT param IDs/values your load_param will "
                "receive at runtime, and the input/output blob wiring. Your "
                "load_param MUST consume exactly these IDs (contract C2); your "
                "forward MUST honor the input/output counts (contract C1).\n"
                f"```\n{ncnn_param_text.strip()}\n```"
            )
        if iface:
            parts.append("\n## Target layer interface (from ncnn source)\n" + iface)
        if ref:
            parts.append(
                "\n## Reference: ncnn built-in implementation (ground truth for "
                "load_param/load_model/forward order & types)\n" + ref
            )
        if contract:
            parts.append("\n## ncnn Layer-Net contract (authoritative)\n" + contract)
        parts.append("\n## Current candidate code (algorithm correct; fix the contract only)\n" + cur)
        parts.append(
            "\n## Output\n"
            "First, in 3-6 bullet points, state which contract rule(s) C1-C6 or "
            "which reference line(s) the candidate violates and how you fix each "
            "(cite them). Then output the COMPLETE corrected files. Each file as "
            "a fenced block whose FIRST line inside the fence is the bare "
            "filename, e.g.:\n"
            "```cpp\n" + (next(iter(code_book), "cand_x.cpp")) + "\n// ... full file ...\n```\n"
            "Output every file (.h and .cpp). Do not omit unchanged files. Keep "
            "the class name `" + self.class_name + "` and the same filenames."
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def adapt(
        self,
        code_book: dict[str, str],
        *,
        ncnn_param_text: str = "",
        e2e_detail: str = "",
        input_shapes: Any = None,
        expected_out_shape: Any = None,
        attempt: int = 1,
    ) -> dict[str, str]:
        """Return a corrected ``{filename: code}`` book. On any failure to
        produce code, returns the original ``code_book`` unchanged (caller can
        treat that as "adapter made no progress")."""
        prompt = self._build_prompt(
            code_book,
            ncnn_param_text=ncnn_param_text,
            e2e_detail=e2e_detail,
            input_shapes=input_shapes,
            expected_out_shape=expected_out_shape,
            attempt=attempt,
        )
        sub = self.run_dir / f"attempt_{attempt}"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "prompt.txt").write_text(prompt, encoding="utf-8")

        try:
            response = self.llm(prompt, self.model)
        except Exception as e:  # noqa: BLE001
            (sub / "error.txt").write_text(str(e), encoding="utf-8")
            return dict(code_book)
        (sub / "response.md").write_text(response or "", encoding="utf-8")

        new_code = extract_kernel_code(response or "")
        if not new_code:
            return dict(code_book)
        # merge: adapter output wins for the files it returns, keep any others
        merged = {**code_book, **new_code}
        for name, code in merged.items():
            (sub / name).write_text(code, encoding="utf-8")
        return merged
