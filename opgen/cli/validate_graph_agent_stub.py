"""Harness validation for GraphAgent WITHOUT a live LLM.

OPENROUTER_API_KEY is not available in this environment, so we drive the real
GraphAgent.run() with a deterministic stub LLM whose canned responses are
modeled on ncnn's existing nn_LayerNorm pass (renamed to avoid symbol clash).

This validates EVERYTHING in the agent except LLM authoring quality:
  analyzer -> coder -> inject (new .cpp + CMake patch) -> rebuild pnnx ->
  trace -> convert -> verify_structural -> verify_numeric (ctest allclose) ->
  restore.

Run with the venv python:
    .venv/bin/python validate_graph_agent_stub.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# bootstrap opgen flat-import paths (we live in opgen/cli/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # need EndtoEnd... so `import opgen` works
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # opgen/ for flat imports
import opgen as _opgen; _opgen.bootstrap_paths()

from config import GraphConfig
from graph_agent import GraphAgent

VENV_TORCH = AGENT / ".venv" / "lib" / "python3.12" / "site-packages" / "torch"

# --- canned analyzer output (OpProfile JSON) -------------------------------
ANALYZER_JSON = json.dumps({
    "source_form": "nn_module",
    "category": "weighted",
    "target_ncnn_layer": "LayerNorm",
    "needs_weight": True,
    "torch_op": "nn.LayerNorm",
    "rank_coverage": [3, 4, 5],
    "files_to_write": ["pass_ncnn/nn_LayerNorm3d.cpp", "tests/ncnn/test_nn_LayerNorm3d.py"],
    "analog_ops": ["nn_LayerNorm"],
    "notes": "LayerNorm over last dims with affine weight+bias; ncnn LayerNorm "
             "param 0=affine_size(prod of normalized_shape), 1=eps, 2=affine.",
})

# --- canned coder output (path-first fenced blocks) ------------------------
CODER_RESPONSE = r'''
```cpp
pass_ncnn/nn_LayerNorm3d.cpp
// agent-authored pass for LayerNorm (3D normalized_shape, affine)
#include "pass_ncnn.h"

namespace pnnx {
namespace ncnn {

class nn_LayerNorm3d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.LayerNorm            op_0        1 1 input out normalized_shape=%normalized_shape eps=%eps elementwise_affine=%elementwise_affine @weight @bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const { return "LayerNorm"; }
    const char* name_str() const { return "ln"; }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<int>& normalized_shape = captured_params.at("normalized_shape").ai;
        int affine_size = normalized_shape[0];
        for (size_t i = 1; i < normalized_shape.size(); i++)
            affine_size *= normalized_shape[i];

        op->params["0"] = affine_size;
        op->params["1"] = captured_params.at("eps");
        op->params["2"] = captured_params.at("elementwise_affine").b ? 1 : 0;

        if (captured_params.at("elementwise_affine").b)
        {
            op->attrs["0"] = captured_attrs.at("op_0.weight");
            op->attrs["1"] = captured_attrs.at("op_0.bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_LayerNorm3d, 19)

} // namespace ncnn
} // namespace pnnx
```

```python
tests/ncnn/test_nn_LayerNorm3d.py
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=(3, 24, 64), eps=1e-3)

    def forward(self, w):
        return self.ln(w)

def test():
    net = Model().eval()
    torch.manual_seed(0)
    w = torch.rand(1, 2, 3, 24, 64)
    a = net(w)

    mod = torch.jit.trace(net, (w,))
    mod.save("test_nn_LayerNorm3d.pt")

    import os
    os.system("../../src/pnnx test_nn_LayerNorm3d.pt inputshape=[1,2,3,24,64]")

    import test_nn_LayerNorm3d_ncnn
    b = test_nn_LayerNorm3d_ncnn.test_inference()

    if not torch.allclose(a, b[0] if isinstance(b, (list, tuple)) else b, 1e-3, 1e-3):
        return False
    return True

if __name__ == "__main__":
    exit(0 if test() else 1)
```
'''


def stub_llm(prompt: str, model: str) -> str:
    if "Return ONLY a JSON" in prompt or "Return ONLY a JSON object" in prompt:
        return ANALYZER_JSON
    return CODER_RESPONSE


def main() -> None:
    cfg = GraphConfig(
        model="stub/canned",
        max_rounds=3,
        run_numeric=True,
        keep_changes_on_success=False,
        torch_install_dir=VENV_TORCH,
        build_jobs=8,
    )
    agent = GraphAgent(task_name="LayerNorm_3d", cfg=cfg, llm_query=stub_llm)
    summary = agent.run()
    print("\n================ SUMMARY ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
