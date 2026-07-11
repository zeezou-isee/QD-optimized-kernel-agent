# QD-optimized-kernel-agent

> An LLM-agent system that **writes, verifies, and optimizes mobile-inference operators from scratch** — generating an ncnn kernel (and, for unsupported ops, the PyTorch→ncnn graph conversion), validating it numerically against PyTorch, then optimizing it with a two-layer **Quality-Diversity** search. (Research prototype, for AAAI 2027.)

Given a PyTorch operator definition, the system produces a correct, fast ncnn
operator end-to-end: **author kernel → (if the op is new) author the conversion
graph → install into ncnn → verify numerically → benchmark → optimize → register**.
It targets the **CPU/ARM** backend today (NEON + NC4HW4); Vulkan is designed but
deferred until a GPU device is available.

---

## What it solves

The classic mobile-kernel workflow (e.g. MoKA) can only rewrite the kernel of an
operator that *already exists* in the framework — it cannot introduce an operator
the framework doesn't have, because a new op first needs a PyTorch→ncnn **graph
conversion (PNNX pass)** before it can run at all. This project adds that missing
link and a real optimizer:

1. **KernelAgent** — writes an ncnn kernel from scratch (portable `base`, or a
   NEON/NC4HW4 `arm` subclass), verified by allclose vs PyTorch.
2. **GraphAgent** — for ops ncnn doesn't natively convert, authors the PNNX pass
   (PyTorch→ncnn), verified structurally + numerically.
3. **AdapterAgent** — when a kernel is *numerically* correct in the per-op sandbox
   but fails end-to-end inside `ncnn::Net`, it rewrites the kernel to satisfy the
   ncnn **Layer-Net contract** (weight `mb.load` type, forward overload vs flags,
   param IDs), armed with a contract spec distilled from ncnn source + the real
   `.ncnn.param`. See `AgentDesign/monologue/AdapterAgent.md`.
4. **OptimizeAgent** — a two-layer Quality-Diversity optimizer (LLM proposer +
   real on-machine measurement) that makes the verified kernel faster.
5. **OperatorAgent** — a decision-driven orchestrator that wires all of the above
   into one end-to-end flow.

---

## Architecture

```
OperatorAgent (orchestrator) — decision-driven 7-stage flow
  [1] KernelAgent (base)         write kernel, allclose vs PyTorch (LayerOracle)
  [1b] KernelAgent (arm)         NEON/NC4HW4 subclass of the base (optional)
  [2] Bridge                     install kernel(s) into ncnn/src/layer[/arm], rebuild libncnn
  [3] existence check            probe pnnx: does ncnn already convert this op?
        already supported  -> use the native conversion
        not supported      -> [3b] GraphAgent: author the PNNX pass (<=15 rounds)
  [4] end-to-end numeric         run the converted .ncnn model vs PyTorch (allclose)
        e2e fails  -> [4b] AdapterAgent: rewrite the (algorithm-correct) kernel to
                          satisfy the ncnn Layer-Net contract (mb.load weight type,
                          forward overload vs flags, param IDs), reinstall, re-check
  [5] production validation      compile + correctness [+ android benchmark]
  [6] OptimizeAgent              two-layer QD: make the (arm) kernel faster, re-validate
  [7] cleanup / --install        restore the source tree, or permanently register the op
```

All three sub-agents share one pattern: **agent loop (state machine) + functional
pipeline + 3 roles (analyzer / coder / debugger)**, where the loop repairs the
*first failing stage* each round and feeds the role only that stage's diagnostic.

### Verification backbone (`opgen/layer_oracle/`)
- **LayerOracle** — compile one candidate `.cpp` + `libncnn.a`, instantiate the
  class directly, run forward, allclose vs PyTorch. No ncnn-tree edits, no
  per-op C++ test. `arm` mode compiles the base `.cpp` in as well and runs the
  NC4HW4 packed path (`--packing 4`).
- **NetOracle** — install a verified kernel into ncnn, rebuild `libncnn.a`, run the
  *whole converted model* via a generic `ncnn::Net` runner vs PyTorch (catches
  semantic errors structural checks miss, e.g. `gt -> max`).

### The QD optimizer (`opgen/optimize/`)
Implements the design in the `算子优化-*.md` / `微观参数优化设计.md` documents as
**Proposer / Evaluator / Policy**:

- **Proposer** = LLM -> a *parameterized template* (kernel with `<KNOB>` placeholders)
  + discrete candidate values + LLM-derived physical-constraint equations.
- **Evaluator** (truth gate) = materialize -> compile -> **correctness oracle**
  (对拍 the baseline) -> **measure harness** (warmup + N runs + noise-floor σ).
  *Correct before fast.*
- **Policy** = two-layer heterogeneous search:
  - **outer** = **MAP-Elites** (Quality-Diversity): roofline picks one of two
    bottleneck-conditional behavior-descriptor coordinate systems; local cell
    competition keeps diverse kernels alive (anti-deception); the LLM is the
    variation operator; an **experience pool (兵器谱)** seeds & persists across ops.
  - **inner** = analytic pruning (免实测) + coarse grid + hill climb (exploits
    parameter-layer local smoothness; cheaper than TPE at small budgets).
  - a **best-first control arm** runs alongside so "use QD or not" is a
    data-driven verdict, not a belief.

Milestones: **M1** = inner loop, **M2** = outer MAP-Elites + roofline + experience
pool, **M3** = cross-op reuse + best-first comparison (all implemented & tested).

---

## Backends

| backend | status | notes |
|---|---|---|
| `base` | ✅ | portable C++ ncnn layer; runs anywhere |
| `arm`  | ✅ | NEON on an **arm64 host** (Apple Silicon / ARM Linux); subclasses the verified base. Validated at **elempack=1** (matching what NetOracle/production run); the packed NC4HW4 + fp16 paths are a future optimization/validation pass |
| `vulkan` | ⏳ deferred | gen+verify designed (BD axes, spec-constant params) — needs an `NCNN_VULKAN=ON` build + a GPU device |

---

## Repository layout

```
opgen/
  config.py            paths / runtime config (finds ../ncnn)
  llm_api.py           multi-provider LLM wrapper (DeepSeek / OpenRouter, routed by
                       model name; streaming; reasoning off by default)
  kernel/              KernelAgent: ncnn kernel writer (base + arm + vulkan)
  graph/               GraphAgent: PyTorch->ncnn PNNX conversion writer
  ncnn_interface/      110-layer interface dict + ncnn_contract.md (C1-C6 Layer-Net
                       contract); lookup.py injects param-id/weight/flag facts into prompts
  layer_oracle/        LayerOracle + NetOracle (compile/run/allclose verification)
  orchestrator/        OperatorAgent (flow) + AdapterAgent (e2e contract repair)
                       + production_validation
  optimize/            OptimizeAgent (QD optimizer)
    schemas.py         ParameterizedTemplate / MeasureSample / BasinValue / OptimizeResult
    evaluator/         cpu_runner, correctness_oracle, measure_harness, evaluator
    inner/             hardware_specs, constraint_engine, coarse_grid, hill_climb, inner_search
    policy/            roofline, bd, archive (MAP-Elites), experience_pool, map_elites, best_first
    test_m1/2/3.py     76 unit tests (fake evaluator/proposer; no LLM/ncnn needed)
  cli/                 run_kernel_agent / run_operator_agent / run_graph_agent /
                       run_perf_compare (our-vs-native speedup on device) / ...
  tools/               file/shell helpers
batch/                 batch_runner.py (one runner for all sets) + sets/{miniset,subset,all}.py
                       + results/*.json (resumable; ops with a result are skipped)
scripts/               device + analysis tools: bench_miniset_device / bench_vulkan_device /
                       bench_e2e_chain (real-phone perf) + rollup_stats (compile/functional/
                       speedup roll-up across ops)
dataset/
  Mobilekernelbench/            183 PyTorch reference operators (12 categories)
  Mobilekernelbench_miniset/    11-op fast smoke set
  Mobilekernelbench_subset/     ~26-op mid-tier coverage set
  Mobilekernelbench_unsupported/ 38 ops ncnn does NOT natively support (need the
                                 Cand-kernel pipeline) + _unsupported_index.json
  Mobilekernelbench_pnnx_native.json  the pnnx-native audit (142 supported / 38 unsupported)
AgentDesign/monologue/  per-agent design docs (KernelAgent / GraphAgent / AdapterAgent / ...)
算子优化-*.md / 微观参数优化设计.md   the optimizer design documents
opgen/docs/          background, ncnn graph/kernel notes, validation reports
```

---

## Setup

The repo is **not self-contained at runtime** — it needs an `ncnn/` checkout next
to it (the code walks up to find a directory containing `ncnn/`), a Python env, and
an OpenRouter key.

**1) Layout** (place this repo beside an ncnn checkout):
```
parent/
├── ncnn/                      # git clone https://github.com/Tencent/ncnn.git
└── QD-optimized-kernel-agent/ # this repo
```

**2) Python env + deps** (Python 3.12; deps pinned in `requirements.txt`)

Option A — **conda** (recommended when available):
```bash
cd QD-optimized-kernel-agent
conda create -n qdkernel python=3.12 -y
conda activate qdkernel
pip install -r requirements.txt          # numpy torch openai ncnn cmake
```

Option B — **venv** (fallback when conda can't be installed on the host, e.g.
this machine): identical deps, no conda:
```bash
cd QD-optimized-kernel-agent
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
> `cmake` and `ncnn` (pyncnn) are **runtime** deps, not just build tools: the
> oracles shell out to `cmake` to compile candidate kernels, and `pyncnn` runs
> the converted `.ncnn` model in-process for the end-to-end allclose. The
> pip-distributed `cmake` puts the CLI on PATH so no system install is needed.

**3) Build `libncnn.a`** (required for the kernel/optimize oracles):
```bash
cd ../ncnn
cmake -S . -B build_lib -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_TESTS=OFF -DNCNN_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build_lib -j8
```
> For the new-operator graph path (GraphAgent / OperatorAgent), also build **pnnx**
> under `ncnn/tools/pnnx` (needs PyTorch). Kernel writing + optimization alone do not.

**4) Configure** — the LLM provider is chosen by the `--model-name` you pass; set
the matching key (`opgen/llm_api.py` routes by model name):

| model name | provider | env var |
|---|---|---|
| `deepseek-v4-pro` / `deepseek-chat` / `deepseek-*` | DeepSeek | `DEEPSEEK_API_KEY` |
| anything else (`z-ai/...`, `anthropic/...`, `openai/...`) | OpenRouter | `OPENROUTER_API_KEY` |

```bash
export DEEPSEEK_API_KEY=...              # for deepseek-v4-pro
# or: export OPENROUTER_API_KEY=sk-or-v1-...
# ensure cmake is on PATH (conda/venv bin dir):
export PATH="$PWD/.venv/bin:$PATH"        # venv;  conda: `conda activate qdkernel` already does this
```

---

## Usage

```bash
cd QD-optimized-kernel-agent

# --- write + verify a kernel (allclose vs PyTorch) ---
python opgen/cli/run_kernel_agent.py --task Exp --backend base --model-name z-ai/glm-5.2
python opgen/cli/run_kernel_agent.py --task Exp --backend arm  --model-name z-ai/glm-5.2   # needs base first

# --- optimize a kernel (two-layer QD; on-machine measure) ---
python opgen/optimize/run_optimize.py --task Exp --backend arm --policy map_elites \
       --map-budget 20 --baseline-compare

# --- end-to-end "add a new ncnn operator" (kernel + graph + verify [+ optimize]) ---
python opgen/cli/run_operator_agent.py --task Greater --backends base,arm \
       --optimize --optimize-policy map_elites

# --- batch a whole set end-to-end (kernel + graph + e2e + production per op) ---
# sets: miniset (11) | subset (~26) | all (190). Resumable: ops already in the
# results json are skipped, so you can pre-seed it to skip already-tested ops.
DEEPSEEK_API_KEY=... python batch/batch_runner.py --set miniset --model deepseek-v4-pro
python batch/batch_runner.py --set subset --ops Gemm,LayerNorm    # debug a few ops
#   -> batch/results/<set>.json

# --- benchmark OUR kernel vs ncnn NATIVE on the REAL phone (speedup ratio) ---
# needs an adb-connected device; see "Measuring performance" below.
python opgen/cli/run_perf_compare.py --task Abs --backend arm --perf-comp-base --scale 8
#   -> batch/results/perf_compare.json  (speedup_shipped / speedup_fair per op)

# --- audit DECOMPOSED ops (pnnx -> native chain; QD winner never lands) ---
python scripts/audit_decomposed_ops.py
#   -> batch/results/decomposed_ops.json + a table of island-only ops
#      (e.g. LogSoftmax -> Softmax+UnaryOp): the monolithic Cand_<Op> is verified
#      & QD-optimized in isolation, but ncnn::Net runs the native chain at runtime,
#      so its speedup does NOT land. AUDIT these before quoting their QD gains.

# --- roll up compile / functional / speedup stats across many ops ---
python scripts/rollup_stats.py --source batch/results/all.json --backend arm
#   -> batch/results/rollup.csv + rollup_summary.json
#   rollup joins decomposed_ops.json: adds a `decomposed`/`optimization_lands`
#   column and reports speedup twice — overall AND `*_landed` (excluding
#   decomposed ops). Quote the landed-only numbers for QD-speedup claims.

# --- unit tests (no LLM / ncnn needed) ---
python opgen/optimize/test_m1.py && python opgen/optimize/test_m2.py && python opgen/optimize/test_m3.py
```

Key flags: `--policy {linear,map_elites}`, `--backends base[,arm]`,
`--experience-pool <json>` (cross-op warm-start + persist), `--baseline-compare`
(best-first control arm), `--install` (permanently register the verified op).

---

## Device-in-the-loop verification (+ inline speedup)

By default the authoring loop verifies on the **host** (Mac arm64 / MoltenVK). With
a phone attached you can add a **device gate**: after each round's host verify
passes, the kernel is compiled + run on the **real phone**, and device failures are
fed back to the LLM to repair. No device → falls back to host-only.

```bash
# kernel authoring with the on-phone gate (host stays the fast first filter)
python opgen/cli/run_kernel_agent.py --task Abs --backend arm  --device-verify auto \
       --dataset-root dataset/Mobilekernelbench
python opgen/cli/run_kernel_agent.py --task Abs --backend vulkan --device-verify auto ...
# full operator pipeline / batch with the gate
python opgen/cli/run_operator_agent.py --task Greater --backends base,arm --device-verify auto
python batch/batch_runner.py --set all --model claude-opus-4-8 --device-verify auto
```

Flags (on `run_kernel_agent` / `run_operator_agent` / `batch_runner`):
- **`--device-verify {off,auto,on}`** — `off` (default) = host-only; `auto` = use the
  phone if `adb` sees one, else host; `on` = same but warn if none.
- **`--device-simpleperf`** — also collect PMU (IPC/cache-miss) on device. **Default
  off** — correctness + latency need no simpleperf, and it adds ~2× overhead.
- **`--no-device-speedup`** — disable the inline speedup (default: measure it).

**Inline speedup — zero extra compile.** The candidate device runner already links
`libncnn` (arm) / `libncnn-vk` (vulkan), which contain `create_layer(<type>)` /
`create_layer_vulkan(<type>)`. So for a natively-supported op the gate instantiates
the **built-in ncnn op on the SAME already-compiled runner** (`--layer <type>`, using
ncnn's baked SPIR-V for vulkan) and times it — no separate native build. It records
into the kernel/op summary:
- `device_status` (`passed`/`failed`/`skipped`), `device_latency` (ours, ms),
- `device_native_latency` (native, ms), `device_speedup` (`native/ours`, **fair
  single-layer**, >1 = ours faster). Ops with no matching native variant → speedup skipped.

Failure policy: if the phone keeps rejecting a kernel to max-rounds but the host
passed, the op is still recorded `success` with `device_status=failed` (a flaky
device never hard-fails an otherwise-correct kernel). The device gate **never mutates
the ncnn source tree** (standalone runner linked against the prebuilt lib).

Validated: arm Abs ours 13.7ms vs native 18.9ms → **1.38×**; vulkan ReduceSum
0.95ms vs native 13.1ms → **13.7×** on Adreno; and the gate caught vulkan Greater/Mul
kernels that passed host MoltenVK but **failed on Adreno**, then repaired them in-loop.

---

## Measuring speedup vs ncnn native — two routes

There are **two** ways to get the our-vs-native speedup, with different tiers:

| | **inline** (device-in-the-loop) | **sweep** (`run_perf_compare.py`) |
|---|---|---|
| when | during authoring, after host verify | standalone, post-hoc |
| tier | **fair** single-layer (elempack=1 fp32) | **shipped** fp16+packing whole-net **and** fair |
| cost | ~free (reuses the same device runner via `create_layer`) | rebuilds benchncnn per op |
| native op | `create_layer[_vulkan](<type>)` on the same runner | benchncnn built-in / gpu=0 |
| output | `device_speedup` in the kernel/op summary | `batch/results/perf_compare.json` |

Use **inline** for a quick fair signal every time a kernel is authored; use the
**sweep** for the authoritative as-shipped (fp16+packing) numbers. See the two
sections below.

### Route A — inline speedup (device-in-the-loop)

See "Device-in-the-loop verification" above: pass `--device-verify auto`; the gate
also times the native ncnn op via `create_layer`/`create_layer_vulkan` on the SAME
device runner (zero extra compile) and records `device_speedup` (fair single-layer,
`native/ours`, >1 = ours faster). Disable with `--no-device-speedup`.

### Route B — sweep (`run_perf_compare.py`)

`opgen/cli/run_perf_compare.py` benchmarks **our generated kernel** against
**ncnn's native built-in op** on the same backend, on a **real phone** (never a
laptop for final perf), and computes the speedup ratio. It is standalone — it
does not touch the OperatorAgent flow.

```bash
# arm CPU: our kernel vs native, only when ncnn natively supports the op
python opgen/cli/run_perf_compare.py --task Abs --backend arm --perf-comp-base --scale 8
python opgen/cli/run_perf_compare.py --ops Abs,Conv,Gemm --backend arm --perf-comp-base
python opgen/cli/run_perf_compare.py --task Conv --backend vulkan --perf-comp-base
```

- **`--perf-comp-base`** — the gate. Without it, only OUR kernel is benchmarked.
  With it, if the op is natively supported (reuses the OperatorAgent existence
  check, `native_supported()`), the native op is also benchmarked and the ratio
  computed. `speedup = native_latency / ours_latency` (**>1 = our kernel wins**).
- **`--backend {base,arm,vulkan}`** — enforces a same-backend comparison (CPU for
  base/arm via benchncnn+simpleperf, GPU for vulkan). Vulkan is flagged
  `cross_runner` (ours = oracle-runner single dispatch, native = benchncnn gpu=0).
- **Precision fairness (two tiers).** benchncnn hardcodes fp16+packing=true, so a
  naive ratio pits ncnn's fp16 path against our fp32 kernel. Each op therefore
  records BOTH: `speedup_shipped` (native fp16+packing vs ours fp32, "as-shipped")
  and `speedup_fair` (both fp32). The fair tier needs the opt-in `fp16=`/`packing=`
  args added to `benchncnn.cpp` (already patched; defaults unchanged) — rebuild
  benchncnn after pulling.
- **`--no-simpleperf`** — latency-only: run benchncnn directly (no simpleperf).
  ~2× faster + no profiler perturbation (cleaner latency); loses PMU (IPC/cache-miss).
  Recommended for a pure speedup sweep. Default: simpleperf ON (collects PMU).
- **Adaptive loop**: the sweep probes per-iter latency and sizes `--loop` so each
  profile is ~10s regardless of op weight (a fixed 4000 loops stalls heavy ops — a
  186ms/iter deconv × 4000 = 744s). `--loop N` is the cap.
- Other flags: `--scale N` (grow the input's first dim so the kernel dominates net
  latency; only the first bracket is scaled so weight/bias inputs stay intact),
  `--record-timeout S`, `--dataset <root>` (default full Mobilekernelbench), `--out <json>`.
- Output: `batch/results/perf_compare.json`, keyed `"<op>:<backend>"`, merged
  incrementally (re-runnable).
- Full example (fair-tier arm sweep, latency-only):
  ```bash
  python opgen/cli/run_perf_compare.py --ops "$(cat batch/results/_success_ops.txt)" \
      --backend arm --perf-comp-base --scale 8 --loop 4000 --no-simpleperf
  ```

---

## Ablation experiments

Which agent modules can be turned on/off today, and how:

| module | togglable? | how |
|---|---|---|
| **Wiki / knowledge injection** | ✅ + A/B harness | env `KERNELGEN_WIKI={on,off}`; `opgen/optimize/ab_run.py` runs each task twice and writes `ab_report.json` |
| **QD (MAP-Elites) vs best-first** | ✅ | `--baseline-compare` runs a best-first control arm alongside QD; verdict in `summary.extra.baseline_comparison` |
| **outer policy** | ✅ | `--policy {linear,map_elites}` (`run_optimize.py`) / `--optimize-policy` (orchestrator) |
| **Experience pool warm-start** | ✅ | add/omit `--experience-pool <json>` (default = no warm-start) |
| **GraphAgent** | ⚠️ auto only | skipped automatically by the existence check when the op is native; no manual force flag |
| **ncnn interface-dict injection** | ❌ not yet | injected unconditionally (`ncnn_interface/`); needs a gate added to ablate |
| **Profiler feedback (`posthoc_bd`)** | ❌ not yet | self-disables only when no device PMU profile is attached; no independent switch |
| **fp16/packing prompt hints** | ❌ not yet | emitted unconditionally for the arm backend |

```bash
# wiki on/off ablation (dedicated harness): writes batch/results-style ab_report.json
python opgen/optimize/ab_run.py

# a single wiki-off run by hand
KERNELGEN_WIKI=off python opgen/optimize/run_optimize.py --task Conv --backend arm --policy map_elites

# QD vs best-first, on one op
python opgen/optimize/run_optimize.py --task Conv --backend arm --policy map_elites --baseline-compare

# experience-pool on vs off = with/without the flag (the pool json is created +
# persisted at the path you give; point later runs at it for cross-op warm-start)
python opgen/optimize/run_optimize.py --task Conv --backend arm --policy map_elites            # off
python opgen/optimize/run_optimize.py --task Conv --backend arm --policy map_elites \
       --experience-pool experience_pool/arm_sigma_pool.json                                    # on
```

> `ab_run.py` is **wiki-specific** (it hardcodes `KERNELGEN_WIKI` + `wiki_{on,off}`
> dirs). The QD/best-first and experience-pool ablations run today via the flags
> above but have no dedicated batch harness yet — pair the two runs manually or
> generalize `ab_run.py`'s env-var + mode list. The three ❌ modules need a
> toggle added at their injection sites before they can be ablated.

---

## Summarizing results (roll-up)

The three headline metrics live in different places: per-op **compile** and
**correctness** are in each `runs/<op>/operator/summary.json` (the batch `all.json`
**fuses** them into one `production` bool), and **speedup** is in
`perf_compare.json`. `scripts/rollup_stats.py` joins them into one table + rates:

```bash
# roll up a device-in-loop run; name outputs per backend so they don't clobber
python scripts/rollup_stats.py --source batch/results/all_devloop.json --backend arm \
    --out-csv batch/results/rollup_devloop_arm.csv \
    --out-json batch/results/rollup_devloop_arm_summary.json
# or an explicit op list / a full scan of runs/*/operator/summary.json
python scripts/rollup_stats.py --ops Abs,Conv,Gemm --backend arm
```

What it does:
- **Un-fuses compile vs correctness** — reads `phases.production.compile.ok` and
  `phases.production.correctness.passed` as separate columns (impossible from
  `all.json` alone, where they're a single `production` bool).
- Pulls all functional checkpoints: `phases.kernel.status` (numeric vs PyTorch),
  `end_to_end_numeric.passed` (whole `ncnn::Net`), `production.correctness.passed`.
- **Device columns** (from a `--device-verify` run): `device_status` +
  `device_latency` + inline `device_speedup` per op, and a device-gate pass-rate +
  inline-speedup summary (read from `phases.kernel[_arm]`, backend-appropriate).
- **Joins sweep speedup** from `perf_compare.json` by `op:backend`
  (`speedup_shipped` / `speedup_fair` + latencies).
- Outputs the CSV + `_summary.json` (pass through `--out-csv`/`--out-json` to keep
  backends separate, e.g. `rollup_devloop_arm.*`, `rollup_vulkan_device.csv`).

Typical flows:
- **inline** (one pass): `batch_runner.py --device-verify auto` → `rollup_stats.py`
  (correctness + device + fair speedup, all from the same run).
- **shipped sweep**: `batch_runner.py` → `run_perf_compare.py` (whole-net fp16
  speedup) → `rollup_stats.py` (join).

---

## Validated results (real LLM; on-machine compile + e2e; Apple M-series arm64)

- **miniset 11/11** end-to-end green on **both `base` and `arm`** backends
  (kernel + e2e numeric + production), every `kernel_arm` a real NEON override,
  **0 fallback**.
- **subset 26/26** end-to-end green (10 miniset-carried + 16 new, incl. Winograd /
  ConvTranspose / Group conv, MatMul variants, LayerNorm, Concat, Einsum-as-Permute).
- The hard failures found along the way were **harness/pipeline fidelity bugs**, not
  kernel-authoring failures — all fixed at source (deterministic weight seed; per-weight
  bin-tag packing; `mb.load` type guidance; symmetric production squeeze; arm elempack=1
  validation; decomposed-op retarget guard; multi-shape squeeze consistency).
- **Optimizer (real speedups, each candidate correctness-gated):** `Erf` base
  16.40→14.92 ms (-9.0%); `Exp` arm 19.67→18.95 ms (-3.6%); `Greater` arm full
  pipeline 19.61→17.44 ms (-11.1%), production re-validated. best-first control arm
  reports `tie` for trivial elementwise ops (the intended data-driven verdict).
- **76 unit tests** green for the optimizer (M1/M2/M3); taxonomy + retarget unit tests green.

---

## Design documents

- `算子优化-问题建模与体系设计.md` — problem framing: structured-space expensive
  black-box optimization; Proposer/Evaluator/Policy; roofline; QD/MAP-Elites.
- `算子优化-完整Workflow.md` — the consolidated, corrected end-to-end workflow
  (two BD coordinate systems, correctness oracle, measure harness, 50–150 budget).
- `微观参数优化设计.md` — inner-loop parameter tuning (LLM physical-constraint
  pruning + search).
- `算子端到端优化全流程.md` — a worked GEMM example of the three-stage flow.
- `opgen/docs/` — ncnn graph/kernel background + validation reports.

---

## Limitations / notes

- `arm` backend requires an **arm64 host**; on x86 only `base` is meaningful.
- Running the full orchestrator needs **cmake on PATH** (it rebuilds `libncnn.a`).
- Kernel authoring now covers the miniset/subset breadth (elementwise, logic,
  conv variants incl. Winograd/Group/ConvTranspose, matmul variants, norms, pooling,
  reduction, concat/reshape) at 11/11 and 26/26 e2e. Genuinely hard ops with no ncnn
  layer (einsum contractions, Det/LU, Unique/TopK, LSTM) remain the frontier and are
  what `dataset/Mobilekernelbench_unsupported/` collects for the new-op pipeline.
- Not yet done: Vulkan backend, arm fp16 + packed NC4HW4 validation (correctness is
  validated at elempack=1 fp32, matching NetOracle/production), real on-device benchmark.

---

## Acknowledgement

Built on **ncnn** (https://github.com/Tencent/ncnn) and the MobileKernelBench
operator dataset. Thanks to the ncnn authors.

