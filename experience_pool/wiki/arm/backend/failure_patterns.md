# ARM failure codes — preempt before the oracle rejects you

Oracle classifies failures via `opgen/layer_oracle/failure_taxonomy.py`.
Each code is a symptom the LLM can preempt.

- **`E1_COMPILE`** — build error. Usual roots: missing `#include`, wrong
  `Layer::` base (ncnn arm subclasses inherit the base op class, not `Layer`
  directly), NEON intrinsics used outside `#if __ARM_NEON`. Fix: keep guards
  around every intrinsic; base `.cpp` must not reference fp16/dotprod.
- **`E2_RUNTIME_CRASH`** — SIGSEGV/timeout. Usual roots: writing past
  `cstep` (see below), out-of-range channel indexing, uninitialized `top_blob`,
  infinite tail loop.
- **`E3_SHAPE_WRONG_COUNT`** — output element count doesn't match reference.
  Usual root: missing keepdim/axis-collapse, wrong reduce dimension, or an
  early `return` before writing the last channel.
- **`E4_LAYOUT_PERMUTED`** — values correct but axes transposed. Usual root:
  iterating in NCHW while reading like NHWC, or vice versa. ncnn is NCHW with
  `Mat.channel(c)` giving a 2D per-channel view; do not manually stride
  `c*w*h`.
- **`E5_VALUE_AFFINE`** — output is `a*ref + b` (sign flip, scale, offset).
  Usual roots: wrong bias load order (mb.load flag mismatch), unintended
  activation, forgotten division by kernel volume in avg pool, missing
  negation in a `sub`.
- **`E6_VALUE_NUMERICAL`** (with tag `NEON pattern: last-N-scalar-tail`) —
  scalar tail loop wrong or missing. Fix: after `for (i=0; i+4<=n; i+=4)`
  add `for (; i<n; ++i)` with the scalar op.
- **`E6_VALUE_NUMERICAL`** (with tag `NEON pattern: errors concentrate in
  lane(s) [k]`) — a lane-specific intrinsic bug (wrong `vgetq_lane_f32(v,i)`
  index, wrong `vextq_f32` offset).
- **`E6_VALUE_NUMERICAL`** (with tag `SUSPICION (channel gap)`) — you wrote
  to `w*h*c + y*w + x` instead of using `channel(c).data + y*w + x`. cstep
  padding gap corrupted subsequent channels.
- **`E6_VALUE_NUMERICAL`** (with tag `WEIGHT-MISALIGNMENT SUSPECT`) — sign
  flip everywhere + all-wrong values → `mb.load` was called with the wrong
  `flag` (0 = tagged serialization, 1 = raw). Consult the layer contract's
  `weights_load_order`.
- **`E6_NUMERICAL_INSTABILITY`** — NaN/Inf. Usual roots on ARM: fp16
  overflow (only if you wrote a `HAS_ASIMDHP` path — the base `.cpp` shouldn't
  produce this); reciprocal on zero (`vdivq_f32` when denom might be 0 —
  clamp with `vmaxq_f32` first).
