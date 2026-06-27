"""Role prompts for graph_agent: analyzer / coder / debugger.

The same LLM plays three roles depending on the pipeline state, mirroring the
MoKA coder/debugger split but specialised for PNNX graph conversion.
"""

from __future__ import annotations

import json
import re
from typing import Any

from graph_schemas import OpProfile


# ---------------------------------------------------------------------------
# Shared domain knowledge injected into every coding/debugging prompt
# ---------------------------------------------------------------------------
PNNX_BACKGROUND = """\
You are adding a NEW operator's GRAPH CONVERSION to ncnn via PNNX
(tools/pnnx). A conversion needs passes in two stages:

1. torch -> pnnx
   - pass_level1/nn_Xxx.cpp  (FuseModulePass) only if the op is an nn.Module.
   - pass_level2/F_xxx.cpp    (GraphRewriterPass) maps aten::*/prim:: subgraphs
     to a PNNX op like `F.xxx`. ONE torch op may expand into several subgraphs,
     so you often need multiple match-pattern variants.
2. pnnx -> ncnn
   - pass_ncnn/F_xxx.cpp (GraphRewriterPass, namespace pnnx::ncnn) rewrites the
     PNNX op into an ncnn layer and fills params/weights.

PNNX-IR pattern format (used in match_pattern_graph):
    7767517                      magic
    <op_count> <operand_count>
    <type> <name> <nin> <nout> <in...> <out...> [key=value ...] [@weight @bias]
  - `%name` captures a param into captured_params.
  - `@weight @bias` capture weights into captured_attrs.

Registration macros:
  - pass_level1:  REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(CLASS)
  - pass_level2:  REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(CLASS, PRIORITY)
  - pass_ncnn:    REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(CLASS, PRIORITY)

REQUIRED include + base class per file kind (use EXACTLY these, no other header):
  - pass_level1/nn_*.cpp : #include "fuse_module_pass.h" ; class X : public FuseModulePass
                           (FuseModulePass uses match_type_str() returning the
                           python module path, e.g. "__torch__.torch.nn.modules.normalization.LayerNorm";
                           it does NOT use match_pattern_graph.)
  - pass_level2/F_*.cpp  : #include "pass_level2.h"      ; class X : public GraphRewriterPass
  - pass_ncnn/F_*.cpp    : #include "pass_ncnn.h"        ; class X : public GraphRewriterPass
Never invent header names (there is no pass_level1.h-for-modules, no pass_level1_torch.h).
Often you do NOT need a pass_level1 file at all: nn.* modules are already captured;
prefer writing only pass_level2 (aten->F.x) and pass_ncnn (F.x->ncnn layer).

In pass_ncnn write():
  - type_str() returns the ncnn layer type (must match an existing ncnn layer).
  - name_str() returns the instance name prefix.
  - op->params["N"] = ... uses ncnn's param-id convention for that layer
    (see ncnn/docs/developer-guide/operation-param-weight-table.md).

=== TWO WAYS TO WRITE A pass_ncnn PASS — pick the right one ===
A) PATTERN style (declarative, what we default to). Subclass GraphRewriterPass with
   match_pattern_graph(). Works when the source IR is a SINGLE high-level op
   (`F.x` / `nn.X` / `torch.x`) and the rewrite is a clean 1:1 to an ncnn layer.

B) IMPERATIVE style (procedural). Write `void convert_<name>(Graph& g)` and walk
   the graph yourself. **Use this when:**
   - the source op decomposes into multi-node subgraph (e.g. `pnnx.Expression + aten::xxx`),
   - you need to compute params from MULTIPLE upstream nodes (constants, shapes),
   - you must normalize negative axes / batch dims, or split one op into many,
   - the pattern DSL cannot express the rewrite cleanly.
   Real examples shipped with pnnx: `pass_ncnn/convert_torch_cat.cpp`,
   `pass_ncnn/convert_Tensor_slice.cpp`, `pass_ncnn/expand_expression.cpp`.

Imperative pass file shape (BOTH .h and .cpp REQUIRED, namespaces matter):
```cpp
// pass_ncnn/convert_<class_lower>.h
#include "pass_ncnn.h"
namespace pnnx { namespace ncnn {
void convert_<class_lower>(Graph& graph);
} }

// pass_ncnn/convert_<class_lower>.cpp
#include "convert_<class_lower>.h"
namespace pnnx { namespace ncnn {
void convert_<class_lower>(Graph& graph) {
    int idx = 0;
    while (true) {            // restart loop after mutating graph
        bool matched = false;
        for (size_t i = 0; i < graph.ops.size(); ++i) {
            Operator* op = graph.ops[i];
            if (op->type != "<TARGET_PNNX_OP>") continue;   // e.g. "aten::tril"
            matched = true;
            op->type = "<TARGET_NCNN_LAYER>";               // your new Cand_X
            op->name = std::string("<prefix>_") + std::to_string(idx++);
            // ... compute params from op->params / upstream constants ...
            op->params["0"] = <value>;
            // detach scalar/constant inputs you've absorbed into params:
            if (op->inputs.size() > 1) op->inputs.resize(1);
            break;             // restart the outer loop
        }
        if (!matched) break;
    }
}
} }
```
Imperative gotchas (these have bitten us):
  - You MUST also write the matching `.h`; without it `pass_ncnn.cpp` cannot include it.
  - You do NOT register with REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS. The
    harness wires your function into the dispatcher automatically — JUST WRITE the
    `void convert_X(Graph&)` and matching header.
  - After absorbing a constant scalar from an upstream `pnnx.Expression` into
    your op's params, REMOVE that orphan operand+operator from `graph.operands`
    and `graph.ops` (and erase it from its operand's consumers list). Otherwise
    the leftover `pnnx.Expression` node leaks into `.ncnn.param` and ncnn fails
    to load (`layer pnnx.Expression not exists or registered`).
  - The loop pattern `while(true){match-one; break; if(!matched) break;}` is
    necessary because mutating `graph.ops` invalidates the for-iterator.
"""

OUTPUT_CONTRACT = """\
OUTPUT CONTRACT — return ONLY fenced code blocks, nothing else.
The FIRST LINE INSIDE each fence MUST be the repo-relative destination path.
Allowed destinations:
  - pass_ncnn/F_<op>.cpp                       (pattern style, required if A)
  - pass_ncnn/convert_<class_lower>.cpp + .h   (imperative style, required if B)
  - pass_level2/F_<op>.cpp        (if torch op needs aten/prim capture)
  - pass_level1/nn_<Op>.cpp       (only if the op is an nn.Module)
  - tests/ncnn/test_F_<op>.py     (required, end-to-end test)
Example:
```cpp
pass_ncnn/F_myop.cpp
// code...
```
```python
tests/ncnn/test_F_myop.py
# code...
```
Do not include prose, explanations, or comments outside the code blocks.
"""

TEST_TEMPLATE_HINT = """\
The test file MUST follow the tools/pnnx/tests/ncnn convention so ctest can run
it (it traces a model, runs ../../src/pnnx, imports the generated *_ncnn module,
and compares with torch.allclose). Skeleton:

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Model(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x, y, z, w):
        # exercise multiple ranks and equivalent spellings of the op
        return F.myop(x), F.myop(y), F.myop(z), F.myop(w)

def test():
    net = Model().eval()
    torch.manual_seed(0)
    x = torch.rand(16); y = torch.rand(2, 16); z = torch.rand(3, 12, 16); w = torch.rand(5, 7, 9, 11)
    a = net(x, y, z, w)
    mod = torch.jit.trace(net, (x, y, z, w)); mod.save("test_F_myop.pt")
    import os
    os.system("../../src/pnnx test_F_myop.pt inputshape=[16],[2,16],[3,12,16],[5,7,9,11]")
    import test_F_myop_ncnn
    b = test_F_myop_ncnn.test_inference()
    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    exit(0 if test() else 1)
```
"""


# ---------------------------------------------------------------------------
# analyzer
# ---------------------------------------------------------------------------
_TARGET_JUDGMENT = """\
=== target_ncnn_layer — DECIDE FROM THE REAL IR ABOVE, DO NOT GUESS ===
The probe above is ground truth. Use it to set target_ncnn_layer correctly:
- "Baseline ncnn graph ... current pnnx already produces" lists the ACTUAL ncnn
  layer types this op converts to today. If the op already appears there as a
  concrete ncnn layer — very often a GENERIC one (UnaryOp / BinaryOp / Reduction /
  Slice / Crop / Permute), NOT a same-name dedicated layer — then THAT generic
  type is target_ncnn_layer and the op is ALREADY natively supported.
  Example: torch.log / torch.exp / torch.sqrt fold into "UnaryOp" (there is NO
  "Log"/"Exp" layer); torch.gt / torch.add fold into "BinaryOp"; torch.sum into
  "Reduction". Set target_ncnn_layer to the generic type shown, set
  "already_supported": true, and say so in notes.
- A dedicated / NEW layer is needed ONLY when "Unconverted aten/prim ops" is
  non-empty (no ncnn layer exists for it). THEN target_ncnn_layer is your new
  Cand_<Op> layer and "already_supported": false.
- NEVER invent a layer name that is neither in the baseline ncnn graph above nor a
  layer you are explicitly adding. When unsure, prefer the generic op in the IR."""


def analyzer_prompt(task_name: str, model_code: str, grounding: dict | None = None) -> str:
    return f"""You analyze a PyTorch operator and plan its ncnn graph conversion.

Operator task name: {task_name}
PyTorch reference model:
```python
{model_code}
```

{PNNX_BACKGROUND}

REAL pnnx/ncnn IR for THIS model (ground truth — base target_ncnn_layer on it):
{format_grounding(grounding)}

{_TARGET_JUDGMENT}

Classify the operator and decide which passes are needed. Return ONLY a JSON
object (no prose) with these fields:
{{
  "source_form": "nn_module | functional | aten | composite",
  "category": "unary | binary | weighted | tensor_manip | composite",
  "target_ncnn_layer": "<ncnn type from the IR above (e.g. UnaryOp), or your new Cand_<Op>>",
  "already_supported": false,
  "needs_weight": false,
  "torch_op": "<e.g. F.hardsigmoid or aten::hardsigmoid>",
  "rank_coverage": [1, 2, 3, 4],
  "files_to_write": ["pass_ncnn/F_<op>.cpp", "pass_level2/F_<op>.cpp", "tests/ncnn/test_F_<op>.py"],
  "analog_ops": ["F_hardsigmoid", "F_relu6"],
  "notes": "<short reasoning: which ncnn layer (generic or new) and why; expected param ids>"
}}
NOTE: this first version targets weightless unary/functional operators."""


def parse_profile_json(task_name: str, text: str) -> OpProfile:
    """Extract the JSON object from the analyzer response into an OpProfile."""
    payload: dict[str, Any] = {}
    # take the last {...} blob
    blocks = re.findall(r"\{.*\}", text, re.DOTALL)
    for blk in reversed(blocks):
        try:
            payload = json.loads(blk)
            break
        except json.JSONDecodeError:
            continue
    return OpProfile.from_llm(task_name, payload)


# ---------------------------------------------------------------------------
# coder
# ---------------------------------------------------------------------------
def _format_examples(examples: dict[str, str]) -> str:
    if not examples:
        return "(no examples retrieved)"
    parts = []
    for path, code in examples.items():
        parts.append(f"----- {path} -----\n{code}")
    return "\n\n".join(parts)


def format_grounding(grounding: dict | None) -> str:
    """Render the real pnnx IR (ground truth) the pass must match."""
    if not grounding:
        return "(pnnx IR probe unavailable)"
    parts = []
    if grounding.get("pnnx_param"):
        parts.append("REAL PNNX IR (.pnnx.param) — your pass_ncnn match_pattern_graph "
                     "MUST match these op types/operands/params EXACTLY:\n```\n"
                     + grounding["pnnx_param"].strip() + "\n```")
    if grounding.get("op_types"):
        parts.append(f"High-level op types present: {grounding['op_types']}")
    if grounding.get("residual_aten"):
        parts.append(f"Unconverted aten/prim ops (need pass_level2): {grounding['residual_aten']}")
    if grounding.get("ncnn_param"):
        parts.append("Baseline ncnn graph the current pnnx already produces (target to "
                     "reproduce):\n```\n" + grounding["ncnn_param"].strip() + "\n```")
    return "\n\n".join(parts) if parts else "(pnnx IR probe returned nothing)"


_MINIMAL_CHANGE = """\
IMPORTANT — write the MINIMUM set of files:
- The REAL PNNX IR above already shows the op at the PNNX level. If the op
  appears as nn.X or F.x there, its torch->pnnx capture (pass_level1/pass_level2)
  ALREADY EXISTS — do NOT rewrite those; you usually only need pass_ncnn/F_<op>.cpp.
- Only add a pass_level2/F_<op>.cpp if the IR still shows raw aten::/prim:: ops.
- Copy operand/param names in your match_pattern_graph VERBATIM from the IR above
  (e.g. exact op type, exact param keys); every captured_params.at("K") key must
  appear as `...=%K` in the pattern."""


def _force_target_block(force_target: str | None) -> str:
    if not force_target:
        return ""
    return (f"\n=== HARD CONSTRAINT ===\nYour pass_ncnn MUST map the operator to ncnn layer "
            f"type EXACTLY '{force_target}' — this is a NEWLY ADDED custom layer. "
            f"`type_str()` must return \"{force_target}\". Do NOT reuse any existing ncnn op "
            f"(e.g. BinaryOp/UnaryOp/Eltwise) — that would be semantically wrong. Emit only the "
            f"params/inputs that '{force_target}' needs (often none for a simple elementwise op).\n")


def coder_prompt(profile: OpProfile, examples: dict[str, str], model_code: str,
                 grounding: dict | None = None, force_target: str | None = None) -> str:
    return f"""You implement the ncnn graph-conversion passes for an operator.

{PNNX_BACKGROUND}
{_force_target_block(force_target)}

Operator profile (from analysis):
{json.dumps(profile.to_dict(), ensure_ascii=False, indent=2)}

=== GROUND TRUTH (from running the current pnnx on this model) ===
{format_grounding(grounding)}

PyTorch reference model:
```python
{model_code}
```

Similar EXISTING passes to imitate (study their structure and param-id usage):
{_format_examples(examples)}

{TEST_TEMPLATE_HINT}

{_MINIMAL_CHANGE}

{OUTPUT_CONTRACT}"""


# ---------------------------------------------------------------------------
# debugger (3 modes)
# ---------------------------------------------------------------------------
_PHASE_FRAMING = {
    "inject_repair": "The generated files could not be injected (bad path or CMake patch).",
    "build_repair": "pnnx failed to COMPILE after adding the new pass.",
    "convert_repair": "pnnx built but the CONVERSION is wrong: the torch op was not "
                      "captured (aten/prim residue) and/or the target ncnn layer is "
                      "missing from the .ncnn.param.",
    "numeric_repair": "Conversion produced an ncnn graph but the end-to-end output "
                      "does NOT match PyTorch (allclose failed).",
}


def debugger_prompt(phase: str, profile: OpProfile, code_book: dict[str, str], feedback: str, memory: str, grounding: dict | None = None, force_target: str | None = None) -> str:
    framing = _PHASE_FRAMING.get(phase, "The conversion failed.")
    current = "\n\n".join(f"----- {p} -----\n{c}" for p, c in code_book.items()) or "(none)"
    return f"""You are repairing an ncnn graph-conversion implementation.

Situation: {framing}

{PNNX_BACKGROUND}
{_force_target_block(force_target)}

Operator profile:
{json.dumps(profile.to_dict(), ensure_ascii=False, indent=2)}

=== GROUND TRUTH (from running the current pnnx on this model) ===
{format_grounding(grounding)}

{_MINIMAL_CHANGE}

Current files:
{current}

Diagnostic feedback (use this to localise the bug):
```
{feedback}
```

History / what was tried before:
{memory or "(none)"}

First, in 2-4 sentences, state the ROOT CAUSE and the precise fix.
Then return the COMPLETE corrected files.

{OUTPUT_CONTRACT}"""
