# Plan: OptimizeAgent must measure candidate latency on the REAL phone

## Problem (verified in code)
- `opgen/optimize/evaluator/cpu_runner.py::run_once` times a candidate as **macOS
  subprocess wall-clock** (`time.perf_counter()` around `subprocess.run`), incl.
  fork/exec overhead. The code itself flags this as "accepted at M1".
- `optimize_agent.py`: `base_lat = ev.evaluate(base_t).latency_ms` and every candidate
  go through this SAME host harness — so the search's within-comparison is *consistent*
  (the "device−host subtraction → noise" critique is NOT what happens), BUT it optimizes
  **host** latency, not phone latency.
- The real on-device `baseline_perf` (from `[5] profile_op`) is passed to OptimizeAgent
  but used ONLY at `optimize_agent.py:387 res.best_perf = dict(self.baseline_perf)` —
  i.e. reporting the starting point. So the REPORTED best_perf is device-tier while the
  search actually optimized host-tier → misleading; and fast kernels' host wall-clock is
  dominated by subprocess overhead → noisy optimization signal.

Requirement (user): optimization that uses latency as its objective MUST use real
on-phone runtime, for BOTH arm and vulkan.

## Fix
Make the optimize evaluator measure each candidate's latency **on the device**, reusing
the already-built device runners' `--bench` clean timing:
- **arm/base** → `DeviceOracle` (`layer_oracle_runner --bench`, on Android arm64).
- **vulkan** → `VulkanDeviceOracle` (`vulkan_oracle_runner --bench`, on Adreno).

Both already return `latency` (min single-forward ms) from the same runner used for the
device gate. Baseline and every candidate then go through the SAME device path →
consistent + real-phone → the search optimizes true device latency.

## Changes (all in `opgen/optimize/` + a new module; do NOT edit device_oracle.py etc.)
1. **New `opgen/optimize/evaluator/device_measure.py`** — thin wrapper that, given a
   candidate's compiled kernel files + params + inputs, cross-compiles the device runner
   (import & call `DeviceOracle`/`VulkanDeviceOracle` — import only, no edits) and returns
   min latency (ms) via `--bench`. Correctness stays on the host LayerOracle (fast); the
   device call is latency-only (`measure_speedup=False`, reuse `.verify` or add a `.measure`).
2. **`evaluator/evaluator.py::evaluate`** — add `device_measure: bool`/backend. When on and a
   device is present: keep host compile+correctness, but set `latency_ms` from the device
   measurement (fallback to host wall-clock when no device → current behavior).
3. **`optimize_agent.py`** — thread a `device_measure` flag into `Evaluator`; `base_lat` and
   candidates then both come from the device path (already same code path, so consistent).
   `res.best_perf` becomes genuinely device-tier and matches the search objective.
4. **`orchestrator/operator_agent.py::_run_optimization`** — pass `device_measure=self.device_verify
   in (auto,on)` so phase [6] optimizes on-phone; drop the report-only baseline_perf mismatch.
5. **CLI** — reuse `--device-verify {off,auto,on}` (already exists) to also gate the optimize
   evaluator; no new flag needed (or add `--optimize-on-device` if we want them independent).

## Efficiency note
Device measure per candidate = NDK compile + push + `--bench` (~15-20s). With a map_elites
budget of N candidates that's N×. Mitigation: host-prune first (compile+correctness+quick
host latency), then device-measure only survivors / promoted cells. Start simple
(device-measure every evaluated candidate), add pruning if too slow.

## Verification (after the vulkan run frees the device)
- `run_optimize.py --task Abs --backend arm --device-verify auto` → base_lat and candidate
  latencies are on-device ms (cross-check vs op_profiler); improvement is real-phone.
- Same for `--backend vulkan` (Adreno).
- Confirm `res.best_perf` tier == search objective tier (both device).

## Sequencing
Implement AFTER the current vulkan full-dataset run finishes (editing the shared
device/kernel files mid-run would break the batch's per-op subprocess re-imports; and
on-device testing would contend with / pollute the running batch on the same phone).
