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

2) A single fenced ```json block describing the knobs, constraints and rationale:

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
     "rationale": "why this should reduce latency on the target CPU"
   }
   ```

Rules:
- Every <PLACEHOLDER> used in the code MUST appear as a key in "params".
- "values" are a SMALL discrete set (2–4 each); the search explores them.
- Constraints are arithmetic/comparison expressions over the param names and the
  hardware symbols {L1, L2, VEC_BITS, FP32_PER_VEC, VECTOR_REGS}. No function calls.
- The materialized kernel (any legal combination) MUST compile and be numerically
  equivalent to the baseline — correctness is gated by an oracle before timing.
"""


def _hw_block(hw: dict[str, Any]) -> str:
    return (
        f"- arch: {hw.get('arch')}\n"
        f"- L1 data cache: {hw.get('l1d_bytes')} bytes\n"
        f"- L2 cache: {hw.get('l2_bytes')} bytes\n"
        f"- SIMD width: {hw.get('vector_bits')} bits ({hw.get('fp32_per_vector')} fp32/vector)\n"
        f"- vector registers: {hw.get('vector_regs')}\n"
        f"- physical cores: {hw.get('n_cores')}\n"
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


def vary_prompt(
    task_name: str,
    parent_kernel: dict[str, str],
    hardware: dict[str, Any],
    directive: str,
    tried: list[str],
    recent_failures: list[str] | None = None,
) -> str:
    """Prompt for MAP-Elites variation: mutate a PARENT elite per a directive."""
    files = "\n\n".join(
        f"### {name}\n```cpp\n{code}\n```" for name, code in parent_kernel.items()
    )
    tried_block = ("\n".join(f"- {t}" for t in tried)) if tried else "(none yet)"
    fail_block = ("\n".join(f"- {f}" for f in recent_failures)) if recent_failures else "(none)"
    goal = _DIRECTIVE_TEXT.get(directive, _DIRECTIVE_TEXT["optimize"])
    return f"""You are a senior mobile-CPU kernel optimization engineer running one step
of a MAP-Elites search. You mutate a PARENT kernel into a new parameterized
template, keeping the output numerically identical to the parent.

# Operator
{task_name}

# Target hardware
{_hw_block(hardware)}

# Parent kernel (the elite you are mutating)
{files}

# Directives already explored
{tried_block}

# Recent candidate failures (diagnosis — fix the root cause, don't repeat these)
{fail_block}

# This round's directive
{goal}

Emit a PARAMETERIZED template (knobs as <PLACEHOLDER>s) plus the json metadata.
List in "techniques" the structural tags of THIS variant (e.g. ["vectorize"],
["winograd","dotprod"], ["tiling","double"]) — these decide its niche, so be
accurate.
{_OUTPUT_CONTRACT}
"""


def proposer_prompt(
    task_name: str,
    baseline_kernel: dict[str, str],
    hardware: dict[str, Any],
    tried: list[str],
) -> str:
    """Build the proposer prompt from the baseline kernel + hardware + history."""
    files = "\n\n".join(
        f"### {name}\n```cpp\n{code}\n```" for name, code in baseline_kernel.items()
    )
    tried_block = ("\n".join(f"- {t}" for t in tried)) if tried else "(none yet)"
    return f"""You are a senior mobile-CPU kernel optimization engineer. You optimize an
ncnn layer kernel for a SINGLE operator on a SINGLE CPU, targeting lower latency
while keeping the output numerically identical to the baseline.

# Operator
{task_name}

# Target hardware
{_hw_block(hardware)}

# Baseline kernel (already correct; this is your starting point)
{files}

# Optimization techniques already tried in previous rounds
{tried_block}

# Your task
Refactor the baseline into a PARAMETERIZED template: pick ONE coherent
optimization direction (that differs from what was already tried), expose its
tunable knobs as <PLACEHOLDER>s, give a few discrete candidate values per knob,
and derive the physical constraint equations that keep every combination legal
on the target hardware. Do NOT pick final numbers — the search does that.
{_OUTPUT_CONTRACT}
"""
