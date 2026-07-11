"""Prompts for the LLM Proposer (Workflow §3 Proposer / 微观参数优化 §一).

The Proposer's job is NOT to guess fast numbers — it is to (a) refactor the
baseline kernel into a **parameterized template** with `<PARAM>` placeholders,
(b) hand out a small set of discrete candidate values per knob, (c) derive the
**physical constraint equations** that bound the feasible region, and (d) state
*why* the change should help. The inner search (analytic prune + grid + climb)
does the actual number-picking.
"""

from __future__ import annotations

from typing import Any

_OUTPUT_CONTRACT = r"""
## Output format (STRICT)

Return exactly TWO parts:

1) The parameterized kernel source as fenced code blocks. The FIRST line inside
   each fence MUST be the filename, e.g.:

   ```cpp
   cand_abs.cpp
   #include "cand_abs.h"
   ...
   // use placeholders like <UNROLL> / <VEC_WIDTH> wherever a knob appears,
   // e.g.:  #pragma unroll <UNROLL>
   ...
   ```

   Keep the SAME class name, header filename and .cpp filename as the baseline
   (the harness compiles them under those exact names). Placeholders are written
   literally as <NAME> and will be textually replaced by integers before compile.

2) A single fenced ```json block describing the knobs, constraints, BD labels and rationale:

   ```json
   {
     "params": {
       "UNROLL":    {"values": [1, 2, 4, 8], "dtype": "int", "desc": "loop unroll factor"},
       "VEC_WIDTH": {"values": [1, 4],       "dtype": "int", "desc": "elements per SIMD step"}
     },
     "constraints": [
       "UNROLL <= VECTOR_REGS",
       "VEC_WIDTH <= FP32_PER_VEC"
     ],
     "techniques": ["unroll", "vectorize"],
     "bd_labels": {"<axis1_name>": "<value>", "<axis2_name>": "<value>"},
     "rationale": "why this should reduce latency on the target CPU"
   }
   ```

Rules:
- Every <PLACEHOLDER> used in the code MUST appear as a key in "params".
- "values" are a SMALL discrete set (2–4 each); the search explores them.
- Constraints are arithmetic/comparison expressions over the param names and the
  hardware symbols listed in the "Target hardware" section. No function calls.
- **"bd_labels" places this proposal in the search space.** Use the EXACT axis
  names and pick from the value menus shown in the "Search-space axes (Σ)"
  section below. If you believe a genuinely NEW structural family is warranted
  (one not in the menu), you MAY declare a new value there — it will open a new
  niche and, if it wins across tasks, be promoted into the space. Prefer the
  existing menu unless a new family is clearly justified in "rationale".
- The materialized kernel (any legal combination) MUST compile and be numerically
  equivalent to the baseline — correctness is gated by an oracle before timing.
"""


def _hw_block(hw: dict[str, Any], backend: str = "base") -> str:
    """Render hardware facts for the prompt. Vulkan gets GPU fields; arm/base
    get the CPU fields. Symbols emitted here become the vocabulary the LLM is
    expected to reference in its constraint equations — keep them in lockstep
    with ConstraintEngine.hw_ns + WikiLoader.hardware_extras().
    """
    if backend == "vulkan":
        lines = [
            f"- arch: {hw.get('arch')}",
            f"- SUBGROUP_SIZE: {hw.get('SUBGROUP_SIZE', '?')}",
            f"- MAX_WG_INVOCATIONS: {hw.get('MAX_WG_INVOCATIONS', '?')}",
            f"- MAX_SHARED_MEM_BYTES: {hw.get('MAX_SHARED_MEM_BYTES', '?')}",
            f"- MAX_PUSH_CONSTANTS_BYTES: {hw.get('MAX_PUSH_CONSTANTS_BYTES', '?')}",
            f"- HAS_FP16: {hw.get('HAS_FP16', 0)}",
            f"- HAS_INT8: {hw.get('HAS_INT8', 0)}",
            f"- HAS_COOPMAT: {hw.get('HAS_COOPMAT', 0)}",
            f"- HAS_SUBGROUP_ARITHMETIC: {hw.get('HAS_SUBGROUP_ARITHMETIC', 0)}",
            f"- HAS_SUBGROUP_SHUFFLE: {hw.get('HAS_SUBGROUP_SHUFFLE', 0)}",
            f"- HAS_SUBGROUP_BALLOT: {hw.get('HAS_SUBGROUP_BALLOT', 0)}",
        ]
        return "\n".join(lines) + "\n"
    # arm / base: CPU-facing fields
    lines = [
        f"- arch: {hw.get('arch')}",
        f"- L1 data cache: {hw.get('l1d_bytes')} bytes (symbol: L1 / L1D)",
        f"- L2 cache: {hw.get('l2_bytes')} bytes (symbol: L2)",
    ]
    if "L3" in hw and hw["L3"]:
        lines.append(f"- L3 cache: {hw['L3']} bytes (symbol: L3)")
    if "CACHE_LINE" in hw:
        lines.append(f"- cache line: {hw['CACHE_LINE']} bytes (symbol: CACHE_LINE)")
    lines += [
        f"- SIMD width: {hw.get('vector_bits')} bits "
        f"({hw.get('fp32_per_vector')} fp32/vector) (symbols: VEC_BITS, FP32_PER_VEC)",
        f"- vector registers: {hw.get('vector_regs')} (symbol: VECTOR_REGS)",
        f"- physical cores: {hw.get('n_cores')}",
    ]
    for k, sym in (
        ("HAS_DOTPROD", "HAS_DOTPROD"),
        ("HAS_ASIMDHP", "HAS_ASIMDHP"),
        ("HAS_BF16", "HAS_BF16"),
        ("HAS_I8MM", "HAS_I8MM"),
    ):
        if k in hw:
            lines.append(f"- {sym}: {int(hw[k])}")
    return "\n".join(lines) + "\n"


def _persona(backend: str) -> str:
    if backend == "vulkan":
        return (
            "You are a senior GPU compute-shader kernel optimization engineer. "
            "You optimize an ncnn Vulkan layer (GLSL compute shader + C++ pipeline "
            "wrapper) for a SINGLE operator on a SINGLE mobile GPU, targeting "
            "lower latency while keeping the output numerically identical to the "
            "baseline."
        )
    return (
        "You are a senior mobile-CPU kernel optimization engineer. You optimize "
        "an ncnn layer kernel for a SINGLE operator on a SINGLE CPU, targeting "
        "lower latency while keeping the output numerically identical to the "
        "baseline."
    )


def _precision_tier_block(backend: str, hw: dict[str, Any]) -> str:
    """Precision-tier knob (P2): tell the LLM that fp16-storage and fp16-arith
    are available on arm and how to opt in. This is the single highest-leverage
    lever vs ncnn's default fp16+packed baseline — surfacing it explicitly in
    the prompt (rather than hoping the wiki catches it) lands the tier on the
    LLM's shortlist. Empty on non-arm backends."""
    if backend != "arm":
        return ""
    asimdhp = int(hw.get("HAS_ASIMDHP", 0))
    arith_line = (
        "- fp16-arith IS available on this CPU (HAS_ASIMDHP=1). Declare "
        "`\"fp16-arith\"` in techniques (implies fp16-storage) to use half-precision "
        "compute + halve memory bytes."
        if asimdhp else
        "- fp16-arith is NOT available on this CPU (HAS_ASIMDHP=0). Only fp16-storage "
        "is safe here — compute stays fp32 with an implicit fp16<->fp32 conversion."
    )
    return (
        "\n# Precision tier (P2 opt-in) — arm\n"
        "The runner exposes two half-precision knobs. Declare them in the "
        "`techniques` list of your json metadata to enable ncnn's fp16 code path:\n"
        "- `\"fp16-storage\"` sets `opt.use_fp16_storage=true` (weights + blobs in fp16, "
        "halves bytes moved — a memory-bound win regardless of compute).\n"
        f"{arith_line}\n"
        "Always keep the ACCUMULATOR in fp32 (matmul/reduction/softmax) to avoid "
        "`E6_NUMERICAL_INSTABILITY`. Kernels without `support_fp16_storage=true` in "
        "`create_pipeline` will silently stay fp32 even when the tag is set — set the "
        "support flag in the pipeline when your kernel handles half-precision blobs.\n"
    )


def _persona_vary(backend: str) -> str:
    if backend == "vulkan":
        return (
            "You are a senior GPU compute-shader kernel optimization engineer "
            "running one step of a MAP-Elites search. You mutate a PARENT "
            "Vulkan kernel into a new parameterized template, keeping the "
            "output numerically identical to the parent."
        )
    return (
        "You are a senior mobile-CPU kernel optimization engineer running one "
        "step of a MAP-Elites search. You mutate a PARENT kernel into a new "
        "parameterized template, keeping the output numerically identical to "
        "the parent."
    )


_DIRECTIVE_TEXT = {
    "diversify": (
        "GOAL = DIVERSIFY (fill a NEW niche, NOT necessarily faster). Change the "
        "fundamental strategy vs the parent — a different algorithm family / data "
        "layout / compute mapping — so the result lands in a different behavior "
        "cell. Coverage matters more than speed this round (Workflow §7.1)."),
    "optimize": (
        "GOAL = OPTIMIZE (push the SAME strategy faster). Keep the parent's "
        "algorithm family / layout / mapping; refine it (better tiling, unroll, "
        "vectorization, instruction scheduling) to lower latency in its niche."),
}


def _context_section(context: str) -> str:
    """Optional wiki context — a 3-section markdown block (dialect + playbook +
    failure codes) built by WikiLoader.context_block(). Empty string when no
    wiki content is available for this backend/family; the section is omitted
    so the prompt stays lean."""
    context = (context or "").strip()
    if not context:
        return ""
    return f"\n# Backend & operator playbook\n{context}\n"


def _sigma_section(sigma_block: str) -> str:
    """The machine-readable Σ axis vocabulary for this (backend, regime), so the
    LLM projects its proposal onto known axes (Method M2.4). Empty when Σ is
    unavailable — the LLM then just uses free-form techniques (keyword fallback).
    """
    sigma_block = (sigma_block or "").strip()
    if not sigma_block:
        return ""
    return f"\n# Search-space axes (Σ) — declare bd_labels from these\n{sigma_block}\n"


def _dispatch_section(dispatch_block: str) -> str:
    """ncnn's distilled algo_family dispatch prior for THIS op's shape (design §8.4).
    A SOFT prior: bias bd_labels toward the PREFERRED cells, still allowed to
    explore others for coverage. Empty for non-conv / memory-bound ops."""
    dispatch_block = (dispatch_block or "").strip()
    if not dispatch_block:
        return ""
    return ("\n# ncnn dispatch prior (which algorithm family ncnn picks here — "
            f"bias toward these)\n{dispatch_block}\n")


def vary_prompt(
    task_name: str,
    parent_kernel: dict[str, str],
    hardware: dict[str, Any],
    directive: str,
    tried: list[str],
    recent_failures: list[str] | None = None,
    context: str = "",
    backend: str = "base",
    sigma_block: str = "",
    coverage_hint: str = "",
    dispatch_block: str = "",
) -> str:
    """Prompt for MAP-Elites variation: mutate a PARENT elite per a directive."""
    files = "\n\n".join(
        f"### {name}\n```cpp\n{code}\n```" for name, code in parent_kernel.items()
    )
    tried_block = ("\n".join(f"- {t}" for t in tried)) if tried else "(none yet)"
    fail_block = ("\n".join(f"- {f}" for f in recent_failures)) if recent_failures else "(none)"
    goal = _DIRECTIVE_TEXT.get(directive, _DIRECTIVE_TEXT["optimize"])
    return f"""{_persona_vary(backend)}

# Operator
{task_name}

# Target hardware
{_hw_block(hardware, backend)}
{_precision_tier_block(backend, hardware)}{_context_section(context)}{_sigma_section(sigma_block)}{_dispatch_section(dispatch_block)}
# Parent kernel (the elite you are mutating)
{files}

# Directives already explored
{tried_block}

# Recent candidate failures (diagnosis — fix the root cause, don't repeat these)
{fail_block}

# This round's directive
{goal}
{("- COVERAGE: " + coverage_hint) if coverage_hint else ""}

Emit a PARAMETERIZED template (knobs as <PLACEHOLDER>s) plus the json metadata.
List in "techniques" the structural tags of THIS variant (e.g. ["vectorize"],
["winograd","dotprod"], ["tiling","double"]) — these decide its niche, so be
accurate. ALSO set "bd_labels" from the Σ axes above (this is what actually
places the niche; techniques are a secondary hint).
{_OUTPUT_CONTRACT}
"""


def crossover_prompt(
    task_name: str,
    parent_a: dict[str, str],
    parent_b: dict[str, str],
    hardware: dict[str, Any],
    cell_a: str = "",
    cell_b: str = "",
    recent_failures: list[str] | None = None,
    context: str = "",
    backend: str = "base",
    sigma_block: str = "",
    dispatch_block: str = "",
) -> str:
    """Prompt for MAP-Elites CROSSOVER: recombine two winning elites from
    different niches into a child that inherits the best of both."""
    def _files(k):
        return "\n\n".join(f"#### {n}\n```cpp\n{c}\n```" for n, c in k.items())
    fail_block = ("\n".join(f"- {f}" for f in recent_failures)) if recent_failures else "(none)"
    return f"""{_persona_vary(backend)}

# Operator
{task_name}

# Target hardware
{_hw_block(hardware, backend)}
{_precision_tier_block(backend, hardware)}{_context_section(context)}{_sigma_section(sigma_block)}{_dispatch_section(dispatch_block)}
# Parent A (winner of niche: {cell_a or "?"})
{_files(parent_a)}

# Parent B (winner of niche: {cell_b or "?"})
{_files(parent_b)}

# Recent candidate failures (fix root cause, don't repeat)
{fail_block}

# This round: CROSSOVER
Synthesize ONE new kernel that RECOMBINES the strongest ideas of Parent A and
Parent B — e.g. take A's algorithmic structure with B's vectorization/memory
strategy, or graft B's tiling into A. Do NOT just copy one parent; the child must
inherit distinct traits from BOTH and be a coherent, compilable, correct kernel.

Emit a PARAMETERIZED template (knobs as <PLACEHOLDER>s) plus the json metadata.
In "techniques" list the structural tags of the CHILD (which decide its niche);
set "bd_labels" from the Σ axes above (this is what places the niche).
{_OUTPUT_CONTRACT}
"""


def proposer_prompt(
    task_name: str,
    baseline_kernel: dict[str, str],
    hardware: dict[str, Any],
    tried: list[str],
    context: str = "",
    backend: str = "base",
    sigma_block: str = "",
    dispatch_block: str = "",
) -> str:
    """Build the proposer prompt from the baseline kernel + hardware + history."""
    files = "\n\n".join(
        f"### {name}\n```cpp\n{code}\n```" for name, code in baseline_kernel.items()
    )
    tried_block = ("\n".join(f"- {t}" for t in tried)) if tried else "(none yet)"
    return f"""{_persona(backend)}

# Operator
{task_name}

# Target hardware
{_hw_block(hardware, backend)}
{_precision_tier_block(backend, hardware)}{_context_section(context)}{_sigma_section(sigma_block)}{_dispatch_section(dispatch_block)}
# Baseline kernel (already correct; this is your starting point)
{files}

# Optimization techniques already tried in previous rounds
{tried_block}

# Your task
Refactor the baseline into a PARAMETERIZED template: pick ONE coherent
optimization direction (that differs from what was already tried), expose its
tunable knobs as <PLACEHOLDER>s, give a few discrete candidate values per knob,
and derive the physical constraint equations that keep every combination legal
on the target hardware. Declare "bd_labels" from the Σ axes above. Do NOT pick
final numbers — the search does that.
{_OUTPUT_CONTRACT}
"""


_BATCH_OUTPUT_CONTRACT = r"""
## Output format (STRICT — MULTIPLE variants in ONE reply)

Return EXACTLY the requested number of variants. Separate each with a marker line:

=== VARIANT 1 ===
    (variant 1 body: code fences + one json block, format below)
=== VARIANT 2 ===
    (variant 2 body)
... and so on.

Each variant's body has the SAME two parts as a single proposal:
1) parameterized kernel source as fenced code blocks — the FIRST line inside each
   fence MUST be the filename (keep the baseline's class/header/.cpp names). Use
   <PLACEHOLDER> knobs. 2) a single ```json block with
   params/constraints/techniques/bd_labels/rationale.

CRITICAL: each variant MUST land in a DIFFERENT niche — pick a DIFFERENT
`bd_labels` combination (different algo_family / layout / compute_mapping) for
each. They must be STRUCTURALLY distinct strategies, NOT parameter tweaks of one
kernel. Every materialized variant must compile and stay numerically equivalent
to the baseline. Rules for each variant's params/constraints/bd_labels are the
same as the single-proposal contract.
"""


def batch_proposer_prompt(
    task_name: str,
    baseline_kernel: dict[str, str],
    hardware: dict[str, Any],
    n: int,
    tried: list[str],
    context: str = "",
    backend: str = "base",
    sigma_block: str = "",
    coverage_hint: str = "",
    dispatch_block: str = "",
) -> str:
    """Batch ILLUMINATION prompt (design §7.3①): one call → N structurally distinct
    variants, each targeting a different (ideally uncovered) niche. The inner
    search fills each at 1 eval; the diverse set thickens the grid cheaply."""
    files = "\n\n".join(
        f"### {name}\n```cpp\n{code}\n```" for name, code in baseline_kernel.items()
    )
    tried_block = ("\n".join(f"- {t}" for t in tried)) if tried else "(none yet)"
    return f"""{_persona(backend)}

# Operator
{task_name}

# Target hardware
{_hw_block(hardware, backend)}
{_precision_tier_block(backend, hardware)}{_context_section(context)}{_sigma_section(sigma_block)}{_dispatch_section(dispatch_block)}
# Baseline kernel (already correct; this is your starting point)
{files}

# Techniques already tried
{tried_block}

# Your task — ILLUMINATE the search space
Produce **{n} structurally DISTINCT** parameterized templates, each targeting a
DIFFERENT behavior niche (a different `bd_labels` combination from the Σ axes).
Goal is COVERAGE, not raw speed this round: spread across algorithm families /
layouts / compute mappings so the archive fills many cells. Prefer the ncnn
dispatch prior's PREFERRED families first, then cover others for diversity.
{("- COVERAGE: " + coverage_hint) if coverage_hint else ""}
{_BATCH_OUTPUT_CONTRACT}
"""
