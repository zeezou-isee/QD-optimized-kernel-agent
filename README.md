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
  cli/                 run_kernel_agent / run_operator_agent / run_graph_agent / ...
  tools/               file/shell helpers
batch/                 batch_runner.py (one runner for all sets) + sets/{miniset,subset,all}.py
                       + results/*.json (resumable; ops with a result are skipped)
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
# sets: miniset (11) | subset (~26) | all (183). Resumable: ops already in the
# results json are skipped, so you can pre-seed it to skip already-tested ops.
DEEPSEEK_API_KEY=... python batch/batch_runner.py --set miniset --model deepseek-v4-pro
python batch/batch_runner.py --set subset --ops Gemm,LayerNorm    # debug a few ops
#   -> batch/results/<set>.json

# --- unit tests (no LLM / ncnn needed) ---
python opgen/optimize/test_m1.py && python opgen/optimize/test_m2.py && python opgen/optimize/test_m3.py
```

Key flags: `--policy {linear,map_elites}`, `--backends base[,arm]`,
`--experience-pool <json>` (cross-op warm-start + persist), `--baseline-compare`
(best-first control arm), `--install` (permanently register the verified op).

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
