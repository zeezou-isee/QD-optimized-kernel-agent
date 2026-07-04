"""Read-side API for the ncnn layer interface dictionary.

KernelAgent and GraphAgent both query this. The dictionary is the JSON file
produced by `extract_layer_interfaces.py` (Phase A). Loaded once via lru_cache
so prompt assembly stays cheap.

Public surface:
    load_dict()                                 -> {layer_name: record}
    get_interface(name)                         -> record | None
    render_for_prompt(name, *, role)            -> str
    guess_layer_from_task(task_name)            -> str | None
    layer_to_family(layer_name)                 -> str
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


# Path to the JSON dict produced by Phase A. Importing this resolves it
# relative to the project root, not the caller's CWD.
_THIS = Path(__file__).resolve()
_PROJ_ROOT = _THIS.parents[2]                 # .../QD-optimized-kernel-agent
DEFAULT_DICT_PATH = (_PROJ_ROOT / "experience_pool" / "backend_ncnn"
                     / "layer_interfaces.json")


# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_dict(path: str | None = None) -> dict[str, dict[str, Any]]:
    """Load the layer interface JSON once; returns {layer_name: record}.

    Returns an empty dict if the file doesn't exist (Phase A not yet run);
    callers should treat that as a clean fallback ("no dict info available")
    rather than an error.
    """
    p = Path(path) if path else DEFAULT_DICT_PATH
    if not p.exists():
        return {}
    records = json.loads(p.read_text(encoding="utf-8"))
    return {r["name"]: r for r in records if r.get("name")}


def get_interface(layer_name: str) -> dict[str, Any] | None:
    """Return the interface record for `layer_name`, or None if unknown.

    Lookup is case-INSENSITIVE — the dict is keyed by ncnn's exact PascalCase
    name (`Convolution`, `BinaryOp`, `LayerNorm`) but callers often pass file
    stems (`convolution`, `binaryop`) from KernelProfile.analog_layer.
    """
    if not layer_name:
        return None
    d = load_dict()
    if layer_name in d:
        return d[layer_name]
    lc = layer_name.lower()
    for k, v in d.items():
        if k.lower() == lc:
            return v
    return None


# ---------------------------------------------------------------------------
# task_name → ncnn layer name guesser. Used by KernelAgent to pre-resolve the
# analog layer BEFORE the analyzer LLM runs, so the reference block can be
# injected into the analyzer prompt itself (not just consumed afterward).
#
# Strategy: a small table of well-known task → ncnn mappings, plus a generic
# case-insensitive substring fallback. None when nothing matches.
# ---------------------------------------------------------------------------
_TASK_HINTS: dict[str, str] = {
    # element-wise
    "abs": "AbsVal",
    "absval": "AbsVal",
    "elu": "ELU",
    "relu": "ReLU",
    "prelu": "PReLU",
    "sigmoid": "Sigmoid",
    "tanh": "TanH",
    "swish": "Swish",
    "mish": "Mish",
    "gelu": "GELU",
    "hardsigmoid": "HardSigmoid",
    "hardswish": "HardSwish",
    "softmax": "Softmax",
    "softplus": "Softplus",
    "selu": "SELU",
    "clip": "Clip",
    # binary
    "add": "BinaryOp",
    "sub": "BinaryOp",
    "mul": "BinaryOp",
    "div": "BinaryOp",
    "max": "BinaryOp",
    "min": "BinaryOp",
    "pow": "BinaryOp",
    "mod": "BinaryOp",
    "and": "BinaryOp",
    "or": "BinaryOp",
    "xor": "BinaryOp",
    "greater": "BinaryOp",
    "less": "BinaryOp",
    "equal": "BinaryOp",
    # unary
    "exp": "UnaryOp",
    "log": "UnaryOp",
    "sin": "UnaryOp",
    "cos": "UnaryOp",
    "sqrt": "UnaryOp",
    "rsqrt": "UnaryOp",
    "neg": "UnaryOp",
    "asin": "UnaryOp",
    "acos": "UnaryOp",
    "atan": "UnaryOp",
    # conv family
    "conv": "Convolution",
    "convolution": "Convolution",
    "conv1d": "Convolution1D",
    "conv2d": "Convolution",
    "conv3d": "Convolution3D",
    "deconv": "Deconvolution",
    "deconvolution": "Deconvolution",
    "convtranspose": "Deconvolution",
    # matmul / linear
    "linear": "InnerProduct",
    "innerproduct": "InnerProduct",
    "gemm": "Gemm",
    "matmul": "MatMul",
    # norm
    "batchnorm": "BatchNorm",
    "layernorm": "LayerNorm",
    "groupnorm": "GroupNorm",
    "instancenorm": "InstanceNorm",
    "rmsnorm": "RMSNorm",
    # pooling
    "maxpool": "Pooling",
    "averagepool": "Pooling",
    "avgpool": "Pooling",
    "pooling": "Pooling",
    # reduction
    "reducemean": "Reduction",
    "reducesum": "Reduction",
    "reducemax": "Reduction",
    "reducemin": "Reduction",
    "reduceprod": "Reduction",
    "reducel1": "Reduction",
    "reducel2": "Reduction",
    "reduction": "Reduction",
    # tensor
    "concat": "Concat",
    "split": "Split",
    "reshape": "Reshape",
    "slice": "Slice",
    "padding": "Padding",
    "permute": "Permute",
    "flatten": "Flatten",
    "tile": "TileOnnx",
    # rnn
    "rnn": "RNN",
    "lstm": "LSTM",
    "gru": "GRU",
}


# When the task name is ambiguous, the PyTorch model's code is a better signal:
# Gemm.py may use `self.linear = nn.Linear(...)` (→ ncnn InnerProduct, simple
# Wx+b) or `torch.matmul(A, B)` (→ ncnn Gemm/MatMul, transA/transB matrix mul).
# These ncnn layers have INCOMPATIBLE param schemas; guessing wrong here makes
# the dictionary inject the wrong contract and the LLM fabricates an analog.
# Pattern → ncnn layer name (checked in order; first match wins per task).
# pnnx-driven layer choice (matters more than nn.* module choice in general):
#   - nn.Linear with 2D input  → pnnx emits ncnn `Gemm` (M=batch, N=out, K=in)
#   - nn.Linear with 1D input  → pnnx emits ncnn `InnerProduct` (the sample API)
#   - nn.Conv2d                → pnnx emits ncnn `Convolution`
#   - nn.Conv1d                → ncnn `Convolution1D`
#   - nn.Conv3d                → ncnn `Convolution3D`
# We do NOT override Gemm/MatMul/InnerProduct from nn.Linear here — empirical
# evidence (Mobilekernelbench Gemm.py 32x512 input) shows pnnx picks Gemm, not
# InnerProduct. Let the task name's natural mapping win, and rely on the
# baseline probe / GraphAgent to confirm the actual ncnn layer type.
_MODEL_CODE_OVERRIDES = [
    # (search pattern, candidate layers it applies to, override ncnn name)
    ("nn.Conv1d",         {"Convolution", "Convolution1D"},   "Convolution1D"),
    ("nn.Conv2d",         {"Convolution", "Convolution1D"},   "Convolution"),
    ("nn.Conv3d",         {"Convolution"},                    "Convolution3D"),
    ("nn.BatchNorm1d",    {"BatchNorm"},                      "BatchNorm"),
    ("nn.BatchNorm2d",    {"BatchNorm"},                      "BatchNorm"),
    ("nn.LayerNorm",      {"LayerNorm"},                      "LayerNorm"),
    ("nn.GroupNorm",      {"GroupNorm"},                      "GroupNorm"),
    ("nn.LSTM",           {"LSTM", "RNN"},                    "LSTM"),
    ("nn.GRU",            {"GRU", "RNN"},                     "GRU"),
    ("nn.RNN",            {"RNN"},                            "RNN"),
]


def guess_layer_from_task(task_name: str,
                          model_code: str | None = None) -> str | None:
    """Best-effort: ncnn layer name for a Mobilekernelbench-style task name.

    Match cascade:
      1. exact case-insensitive key in _TASK_HINTS
      2. longest hint key that is a substring of the lowercased task
      3. exact case-insensitive match against any dict layer name

    When `model_code` is provided AND the name-based guess hits a known
    ambiguity (Gemm/MatMul/Linear/Conv family), the code is scanned for
    nn.Linear / nn.Conv1d / etc. to pick the correct ncnn layer.

    Returns the canonical PascalCase ncnn name when found, else None.
    """
    if not task_name:
        return None
    t = task_name.lower().replace("_", "").replace("-", "")
    candidate: str | None = None
    if t in _TASK_HINTS:
        candidate = _TASK_HINTS[t]
    if candidate is None:
        matches = [(k, v) for k, v in _TASK_HINTS.items() if k in t]
        if matches:
            candidate = max(matches, key=lambda kv: len(kv[0]))[1]
    if candidate is None:
        d = load_dict()
        for layer in d:
            if layer.lower() == t:
                candidate = layer
                break
    # apply model-code-driven disambiguation
    if candidate is not None and model_code:
        for pat, applies_to, override in _MODEL_CODE_OVERRIDES:
            if candidate in applies_to and pat in model_code:
                if override != candidate:
                    return override
                break
    return candidate


# ---------------------------------------------------------------------------
# ncnn layer name -> op family. DEPRECATED for OptimizeAgent (wiki v1 dropped
# per-family playbook in favor of a generic primitives + bd_axes + regime
# knowledge structure — routing is now by roofline regime, not by op family).
# Kept for other agents that may still want family-level tagging.
# ---------------------------------------------------------------------------
_LAYER_TO_FAMILY: dict[str, str] = {
    # elementwise_binary
    "BinaryOp": "elementwise_binary",
    "Bias": "elementwise_binary",
    "Scale": "elementwise_binary",
    "Eltwise": "elementwise_binary",
    # elementwise_unary
    "UnaryOp": "elementwise_unary",
    "AbsVal": "elementwise_unary",
    # activation
    "ReLU": "activation",
    "PReLU": "activation",
    "Sigmoid": "activation",
    "TanH": "activation",
    "Swish": "activation",
    "Mish": "activation",
    "GELU": "activation",
    "ELU": "activation",
    "SELU": "activation",
    "HardSigmoid": "activation",
    "HardSwish": "activation",
    "Softplus": "activation",
    "Clip": "activation",
    "Dropout": "activation",
    # softmax
    "Softmax": "softmax",
    "LogSoftmax": "softmax",
    # normalization
    "BatchNorm": "normalization",
    "LayerNorm": "normalization",
    "GroupNorm": "normalization",
    "InstanceNorm": "normalization",
    "RMSNorm": "normalization",
    "Normalize": "normalization",
    # reduction
    "Reduction": "reduction",
    # conv
    "Convolution": "conv",
    "Convolution1D": "conv",
    "Convolution3D": "conv",
    "ConvolutionDepthWise": "conv",
    "ConvolutionDepthWise1D": "conv",
    "ConvolutionDepthWise3D": "conv",
    # deconv
    "Deconvolution": "deconv",
    "DeconvolutionDepthWise": "deconv",
    "Deconvolution1D": "deconv",
    "Deconvolution3D": "deconv",
    "DeconvolutionDepthWise1D": "deconv",
    # gemm
    "Gemm": "gemm",
    "MatMul": "gemm",
    "InnerProduct": "gemm",
    # pooling
    "Pooling": "pooling",
    "Pooling1D": "pooling",
    "Pooling3D": "pooling",
    # layout
    "Reshape": "layout",
    "Permute": "layout",
    "Flatten": "layout",
    "Concat": "layout",
    "Split": "layout",
    "Slice": "layout",
    "Padding": "layout",
    "Crop": "layout",
    "TileOnnx": "layout",
    "Squeeze": "layout",
    "Unsqueeze": "layout",
    "ExpandDims": "layout",
    # recurrent
    "RNN": "recurrent",
    "LSTM": "recurrent",
    "GRU": "recurrent",
}


def layer_to_family(layer_name: str | None) -> str:
    """Map an ncnn layer name to its op family (12-family taxonomy).

    Returns "unknown" for anything not in the table, including None/"". Callers
    (WikiLoader) treat "unknown" as "inject backend-level knowledge only, skip
    the per-family playbook page".
    """
    if not layer_name:
        return "unknown"
    if layer_name in _LAYER_TO_FAMILY:
        return _LAYER_TO_FAMILY[layer_name]
    # case-insensitive fallback for callers passing file-stems
    lc = layer_name.lower()
    for k, v in _LAYER_TO_FAMILY.items():
        if k.lower() == lc:
            return v
    return "unknown"


# ---------------------------------------------------------------------------
# Prompt rendering — two roles, two phrasings.
# ---------------------------------------------------------------------------
def _format_param(p: dict[str, Any]) -> str:
    default = p.get("default", "")
    derived = " (default derived from another param)" if p.get("default_is_var") else ""
    return f"    {p['id']:>3} → {p['name']} = {default}{derived}"


def _format_weight(w: dict[str, Any]) -> str:
    cond = f", only if {w['conditional']}" if w.get("conditional") else ""
    return f"    [{w['index']}] {w['var']}, size={w['size_expr']}, flag={w['flag']}{cond}"


def render_for_prompt(layer_name: str, *, role: str = "kernel") -> str:
    """Render an interface block to drop into an LLM prompt.

    role='kernel' — emphasis: "your load_param/load_model MUST match this"
    role='graph'  — emphasis: "your op->params[\"N\"]= MUST match these IDs"

    Returns "" when the layer is not in the dictionary (caller falls back
    to the legacy free-form prompt).
    """
    iface = get_interface(layer_name)
    if iface is None:
        return ""

    lines: list[str] = []
    if role == "kernel":
        lines.append(
            f"### REFERENCE: ncnn built-in `{iface['name']}` interface — "
            f"your kernel MUST match this contract"
        )
        lines.append(
            f"- forward overloads ({len(iface['forward_signatures'])}):"
        )
        for s in iface["forward_signatures"]:
            lines.append(f"    {s}")
        lines.append(
            f"- defaults: one_blob_only={iface['one_blob_only_default']}, "
            f"support_inplace={iface['support_inplace_default']}"
        )
        if iface["params"]:
            lines.append(
                "- params (use these EXACT IDs in your params dict):"
            )
            for p in iface["params"]:
                lines.append(_format_param(p))
        if iface["weights_load_order"]:
            lines.append(
                "- weights (your load_model must call mb.load in this order, "
                "and weight_keys must list them in the same order):"
            )
            for w in iface["weights_load_order"]:
                lines.append(_format_weight(w))
    elif role == "graph":
        lines.append(
            f"### TARGET LAYER `{iface['name']}` interface — "
            f"`op->params[\"N\"]=` in your pass_ncnn MUST use these IDs"
        )
        if iface["params"]:
            for p in iface["params"]:
                lines.append(_format_param(p))
        if iface["weights_load_order"]:
            lines.append(
                "- weight write order (op->weights / mb.load):"
            )
            for w in iface["weights_load_order"]:
                lines.append(_format_weight(w))
    else:
        return ""
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Param-value inference: dictionary-driven replacement for the hardcoded
# 5-elif in kernel_agent._infer_params. Maps ncnn param var names (from the
# dict) to actual values computable from the PyTorch state_dict.
#
# The mapping table is the source of truth for "given a param named X, where
# does its value come from?" — adding a new pattern here covers all ncnn
# layers whose load_param uses that var name, not just one op family.
# ---------------------------------------------------------------------------

def _resolve_param_value(var_name: str, state_dict: dict[str, Any]) -> Any | None:
    """Compute a concrete value for an ncnn param var from the model state_dict.

    Returns None when the value cannot be derived (no matching weight shape,
    or no pattern registered for this var name). None means "fall back to
    the LLM-provided value (if any) or the ncnn default".

    state_dict is the introspected PyTorch state_dict: {key: shape_list}.
    """
    if not state_dict:
        return None

    # Snapshot a "main weight" shape and a "has bias" flag once.
    weight_shape: list[int] | None = None
    for k, s in state_dict.items():
        if "weight" in k.lower() and isinstance(s, (list, tuple)) and len(s) >= 1:
            weight_shape = list(s)
            break
    has_bias = any("bias" in k.lower() for k in state_dict)

    n = var_name.lower()

    # --- output channels / output features --------------------------------
    if n in ("num_output", "out_features", "out_channels"):
        return int(weight_shape[0]) if weight_shape else None

    # --- input feature size (Linear only) ---------------------------------
    if n in ("input_dim",):
        # For nn.Embedding: weight is (num_embeddings, embedding_dim)
        if weight_shape and len(weight_shape) == 2:
            return int(weight_shape[0])
        return None

    # --- bias flag --------------------------------------------------------
    if n in ("bias_term",):
        return 1 if has_bias else 0

    # --- weight_data_size (= prod of weight shape) ------------------------
    if n in ("weight_data_size",):
        if not weight_shape:
            return None
        prod = 1
        for d in weight_shape:
            prod *= int(d)
        return prod

    # --- scale_data_size (Scale layer): same idea -------------------------
    if n in ("scale_data_size", "bias_data_size"):
        # use the matching named weight if present
        target = "scale" if n.startswith("scale") else "bias"
        for k, s in state_dict.items():
            if target in k.lower() and isinstance(s, (list, tuple)):
                prod = 1
                for d in s:
                    prod *= int(d)
                return prod
        return None

    # --- channels (BatchNorm/InstanceNorm) --------------------------------
    if n in ("channels",):
        if weight_shape and len(weight_shape) == 1:
            return int(weight_shape[0])
        return None

    # --- affine_size (LayerNorm/RMSNorm) — last-dim of normalized shape ---
    if n in ("affine_size",):
        if weight_shape and len(weight_shape) >= 1:
            return int(weight_shape[-1])
        return None

    # --- num_slope (PReLU) ------------------------------------------------
    if n in ("num_slope",):
        if weight_shape and len(weight_shape) == 1:
            return int(weight_shape[0])
        return 1     # PReLU collapses to a single shared slope

    # --- LSTM/GRU hidden_size --------------------------------------------
    if n in ("hidden_size", "num_output_hidden"):
        # weight_ih_l0 shape = (4*hidden_size, input_size) for LSTM,
        # (3*hidden_size, input_size) for GRU. Conservative: take any weight
        # ending in _hh_l0 (hidden→hidden) whose first dim is hidden*K.
        for k, s in state_dict.items():
            if "weight_hh" in k.lower() and isinstance(s, (list, tuple)) and len(s) == 2:
                # for LSTM K=4, GRU K=3; both divide evenly
                d0 = int(s[0])
                if d0 % 4 == 0:
                    return d0 // 4
                if d0 % 3 == 0:
                    return d0 // 3
        return None

    # nothing registered for this var name
    return None


def derive_params_from_dict(layer_name: str,
                            state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper: for each param in the layer's interface, try to
    resolve a value from state_dict. Returns {param_id_str: value}, only
    including params we could actually resolve.
    """
    iface = get_interface(layer_name)
    if not iface:
        return {}
    out: dict[str, Any] = {}
    for p in iface.get("params", []):
        v = _resolve_param_value(p["name"], state_dict)
        if v is not None:
            out[str(p["id"])] = v
    return out
