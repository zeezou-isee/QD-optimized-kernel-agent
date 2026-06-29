# opgen/ncnn_interface — ncnn built-in layer interface dictionary

Source-of-truth extractor for the param ID + weight order + forward signature
contract of every ncnn built-in layer. Output feeds the KernelAgent so a
generated kernel can be forced to use the same `pd.get(N, default)` IDs and
`mb.load(...)` order as the corresponding `ncnn` layer — otherwise the
LayerOracle / NetOracle will silently read wrong values at runtime.

## Run

```bash
.venv/bin/python -m opgen.ncnn_interface.extract_layer_interfaces \
    --ncnn-root /Users/xingze/Documents/project/kernelgen/ncnn
```

Outputs to `experience_pool/backend_ncnn/`:
- `layer_interfaces.json` — machine-readable (agent consumes this)
- `layer_interfaces.md`   — human-readable with `⚠ MISMATCH` ops bubbled to top

### Useful flags

- `--only Convolution,BinaryOp` — extract a few ops only (debug)
- `--diff old_layer_interfaces.json` — structural diff vs previous run
  (use when ncnn upstream is bumped, to see which interfaces moved)

## What's extracted (schema)

```json
{
  "name":              "Convolution",
  "header":            "convolution.h",
  "source":            "convolution.cpp",
  "base_class":        "Layer",
  "forward_signatures": ["int forward(...) const", "int forward(vec<Mat>&,...) const"],
  "one_blob_only_default": true,
  "support_inplace_default": false,
  "params": [
    {"id": 0, "name": "num_output",   "default": "0",         "default_is_var": false},
    {"id": 11, "name": "kernel_h",    "default": "kernel_w",  "default_is_var": true}
  ],
  "weights_load_order": [
    {"index": 0, "var": "weight_data", "size_expr": "weight_data_size", "flag": 0},
    {"index": 1, "var": "bias_data",   "size_expr": "num_output", "flag": 1,
     "conditional": "bias_term"}
  ],
  "doc_table_present": true,
  "mismatches": [
    {"type": "src_only", "id": 19, "name": "dynamic_weight"},
    {"type": "doc_only", "id": 17, "name": "impl_type"}
  ],
  "parse_warnings": ["..."]
}
```

### Field semantics

- `default_is_var: true` — the default expression is another variable
  (`pd.get(11, kernel_w)`); the agent must understand this is a **derived
  default**, not a literal. ncnn uses this heavily for Conv-family ops.
- `conditional` — the weight is loaded only inside an `if (<cond>) { ... }`
  block. Common case: `bias_data` loads only if `bias_term`.
- `mismatches.type`:
  - `doc_only` — doc table has this param ID but source doesn't (param removed
    or doc is stale)
  - `src_only` — source has it but doc doesn't (new param added since the table
    was last updated)
  - `name_diff` — same ID but the name differs between doc and source

## Scope (intentional)

- **Only the 110 layers under `ncnn/src/layer/*.h`** — NOT the per-arch
  subdirectories (`vulkan/`, `arm/`, `x86/`). Those are optimized backends
  of the same layer; the interface contract is inherited from base, not
  redeclared.
- **Regex-based, not a real C++ parser.** Targets the ~95% of common ncnn
  patterns. Anything exotic shows up under `parse_warnings` and ends up in
  the human-review pass.

## Phase B (not in this module yet)

This module only produces the dictionary. Wiring it into KernelAgent's prompt
(to force LLM-generated kernels to match the ncnn interface) is the next step,
deliberately kept separate so a human can review the dictionary first.
