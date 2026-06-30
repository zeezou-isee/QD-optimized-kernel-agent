# ncnn Layer–Net Contract (authoritative spec)

> Distilled **only** from ncnn source + official developer-guide docs (no guessing).
> Sources are cited as `[doc]` (docs/developer-guide/*.md) or `[src]` (ncnn/src/*).
> This is the knowledge base for the AdapterAgent: a layer that is *mathematically*
> correct but violates any rule below will pass a sandbox yet fail inside `ncnn::Net`.

A custom layer `class Cand_X : public ncnn::Layer` must satisfy **6 contracts** to run
correctly inside `ncnn::Net::Extractor`. Each is a hard interface, not a style choice.

---

## C1. Constructor flags  `[doc how-to-implement-custom-layer-step-by-step.md, layer-support-behavior.md]`

Set in the constructor (or toggled in `load_param`/`load_model` once params are known):

- `one_blob_only`  — `true` ⇒ exactly **1 input blob, 1 output blob**; ncnn calls the
  `(const Mat&, Mat&, ...)` overload. `false` ⇒ ncnn calls the `(vector<Mat>&, vector<Mat>&, ...)` overload.
- `support_inplace` — `true` ⇒ you implement `forward_inplace`; input/output share memory.

**forward overload truth table** `[doc step5]`:

| one_blob_only | support_inplace | implement |
|---|---|---|
| false | false | `forward(const vector<Mat>&, vector<Mat>&, opt)` |
| false | true  | `forward_inplace(vector<Mat>&, opt)` (+ optional non-inplace) |
| true  | false | `forward(const Mat&, Mat&, opt)` |
| true  | true  | `forward_inplace(Mat&, opt)` (+ optional non-inplace) |

If you set the flags but implement the wrong overload, ncnn calls a base no-op and the
output blob is empty ⇒ e2e shape mismatch / crash. **Match flags to overload exactly.**

A layer may **flip flags inside load_param** based on params (e.g. Gemm sets
`one_blob_only=true` only when exactly one of A/B is a runtime input). Replicate that logic.

---

## C2. load_param(const ParamDict& pd)  `[doc new-param-load-api.md]`

```cpp
int Cand_X::load_param(const ParamDict& pd) {
    foo = pd.get(<id>, <default>);   // scalar int/float
    arr = pd.get(<id>, Mat());       // array param
    return 0;
}
```

- **Param IDs are fixed by the layer's ncnn schema**, NOT chosen freely. They must match
  the IDs pnnx writes into `.ncnn.param`. Use the interface dict / the actual emitted
  `.ncnn.param` line as ground truth. (e.g. Convolution: `0=num_output 1=kernel_w 5=bias_term 6=weight_data_size 11=kernel_h ...`)
- IDs are **NOT ONNX/PyTorch attribute order** — they are ncnn's own numbering.
- scalar key index 0..19; array key = `-23300 - index`  `[doc param-and-model-file-structure.md]`.
- `pd.get(id, default)` returns the default if the key is absent — so unused params are safe.

---

## C3. load_model(const ModelBin& mb)  `[doc new-model-load-api.md, src/modelbin.cpp, tools/modelwriter.h]`

```cpp
int Cand_X::load_model(const ModelBin& mb) {
    weight_data = mb.load(weight_data_size, 0);  // PRIMARY weight: type 0
    if (bias_term) bias_data = mb.load(num_output, 1);  // SECONDARY: type 1
    return 0;
}
```

### The bin-type rule (THE most common e2e bug)

`mb.load(w, type)` semantics in `ModelBinFromDataReader::load` `[src/modelbin.cpp:82]`:

- **type == 0**: reads a **4-byte flag tag first**, then the data.
  - tag `0x00000000` ⇒ raw fp32 (w floats)
  - tag `0x01306B47` ⇒ fp16 storage (auto-converted to fp32)
  - tag `0x000D4B38` ⇒ int8; nonzero-otherwise ⇒ quantized table.
- **type == 1**: **no flag**, reads `w` raw fp32 directly.

### Which weight gets a tag is decided by modelwriter `[tools/modelwriter.h]`

pnnx writes the `.bin` with two helpers:
- `fwrite_weight_tag_data` → **writes a 4-byte tag** then data. Used for the layer's
  **PRIMARY weight** (e.g. Convolution/InnerProduct/Gemm `weight_data`/`A_data`/`B_data`).
  ⇒ you MUST read it with **type 0**.
- `fwrite_weight_data` → **NO tag**, raw data only. Used for **secondary** tensors
  (`bias_data`, BatchNorm `slope/mean/var/bias`, int8 scales).
  ⇒ you MUST read it with **type 1**.

**Per-layer exceptions (read modelwriter.h save case, do not assume):**
- BatchNorm writes slope, mean, var, bias **all via `fwrite_weight_data`** ⇒ **all type 1, no tag**.
- Convolution/InnerProduct/Gemm: primary weight = tag (type 0), bias = raw (type 1).

### Load order = modelwriter write order

`mb.load` calls must happen in the **exact order** modelwriter wrote them, gated by the
same conditionals (e.g. `if (bias_term)`). The interface dict `weights_load_order` field
encodes this: each entry has `var`, `size_expr`, `flag` (= the `type`), and `conditional`.

### Multi-dim loads

`mb.load(w,h,type)` / `mb.load(w,h,c,type)` just reshape after a flat `mb.load(w*h*..,type)`
`[src/modelbin.cpp:25]`. Gemm uses `mb.load(K, M, 0)` etc. — w is the inner (fastest) dim.

---

## C4. forward — Mat shape & memory conventions  `[src/mat.h, doc element-packing.md]`

### Mat axis convention
- `Mat.w` = innermost/fastest axis, then `Mat.h`, `Mat.d`, `Mat.c` = channels (outermost).
- `dims` ∈ {1,2,3,4}. 1D: `(w)`. 2D: `(w,h)`. 3D: `(w,h,c)`. 4D: `(w,h,d,c)`.
- Per-channel base pointer: `bottom_blob.channel(q)` (returns a Mat view of one channel).
- **`cstep`**: stride between channels in elements; channel q starts at `data + q*cstep`.
  `cstep` is padded for alignment ⇒ **iterate `w*h` per channel, not `w*h*c` flat.**
  Use `channel(q)` and the per-channel pointer; never assume channels are contiguous.

### Output allocation
- You MUST `top_blob.create(w, h, c, elemsize, opt.blob_allocator)` (matching dims) and
  check `if (top_blob.empty()) return -100;`  `[doc step6]`.
- Output dims/shape must match what the downstream layer + the graph expect.

### Input shape comes from the graph, not your imagination
- The `Mat` shape your forward receives is whatever the upstream blob produced, which is
  governed by the pnnx-emitted graph (`_ncnn.py` squeeze policy). A 2D `nn.Linear` input
  `[batch, in]` arrives as a Mat with `w=in, h=batch` (dims=2), **not** flattened to 1D.
  Handle the real dims; do not hardcode a single-sample 1D path.

### Packing
- NetOracle/our runner sets `opt.use_packing_layout=false` ⇒ **`elempack==1` guaranteed**;
  you do not need SIMD-packed paths for correctness. But still respect `cstep` (C4 above).
- If `support_packing=false` (default), ncnn guarantees elempack=1 input `[doc layer-support-behavior.md]`.

---

## C5. Numerical / storage options

- Our runner sets `use_fp16_packed/storage/arithmetic=false`, `use_bf16_storage=false`,
  `use_int8_inference=false`, `num_threads=1`. So fp32-only, single-thread is sufficient
  for correctness. Do not gate behavior on fp16 unless you also set the support flag.
- Avoid denormals: modelwriter calls `replace_denormals_with_zero` — don't reintroduce them.

---

## C6. Net integration / registration  `[doc step7, src/net.cpp]`

- Class registers via `DEFINE_LAYER_CREATOR(Cand_X)` (our installer strips/handles this).
- The layer line in `.ncnn.param`: `Cand_X  name  in_count out_count <ins> <outs> <params>`.
  Our retarget step rewrites the producing layer's **type** to `Cand_X`, keeping the SAME
  param IDs/values pnnx emitted. ⇒ your `load_param` must consume exactly those IDs (C2).
- `forward_layer` `[src/net.cpp:123]` dispatches on `one_blob_only`; with lightmode +
  `support_inplace`, ncnn may call the inplace path — implement whichever flags you set.

---

## Worked contract: Gemm (the nn.Linear case)  `[src/layer/gemm.cpp, tools/modelwriter.h]`

`nn.Linear(in, out)` → pnnx emits a **Gemm** layer (NOT InnerProduct) with:
- `constantA=0` (A is the runtime activation input), `constantB=1` (B = weight, from bin),
  `constantC=1` if bias else 0, `transB=1` (Linear weight is `[out,in]`),
  `constantN=out`, `constantK=in`.
- In `load_param`, Gemm sets `one_blob_only=true` when `constantA==0 && constantB==1 && constantC==1`
  (exactly one runtime input A). So forward gets ONE blob (A) and uses loaded B_data/C_data.

`load_model` (constantB=1, transB=1): `B_data = mb.load(constantK, constantN, 0)` → Mat w=K, h=N.
(For transB=0 it would be `mb.load(constantN, constantK, 0)`.)

`forward` computes `Y = alpha * op(A) @ op(B) + beta * C`, where op applies transA/transB.
A is `[M,K]` (Mat w=K,h=M), output Y is `[M,N]` (Mat w=N,h=M). For Linear: M=batch, N=out, K=in.

Param IDs (full): `0=alpha 1=beta 2=transA 3=transB 4=constantA 5=constantB 6=constantC
7=constantM 8=constantN 9=constantK 10=constant_broadcast_type_C 11=output_N1M ...`.

---

## How the AdapterAgent should use this

1. Look up the target ncnn layer's interface (dict) + read its built-in `src/layer/<x>.cpp`
   as the reference implementation of load_param/load_model/forward.
2. Read the **actual emitted `.ncnn.param` line** for this op to know the exact param
   IDs/values the layer will receive at runtime (especially constantA/B/C, transB, sizes).
3. Rewrite the candidate layer so C1–C6 hold, **preserving the algorithm** already proven.
4. On e2e failure, find the specific contract (C1–C6) or the specific `src` reference that
   was violated — cite it — then fix. Do not guess; go back to the source line.
