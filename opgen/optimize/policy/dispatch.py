"""Dispatch prior — ncnn's algo-family selection as a QD niche prior (design §8.4).

Consumes `experience_pool/backend_ncnn/dispatch_heuristics.json` (distilled from
ncnn source by `ncnn_interface/distill_dispatch.py`). Given an op's (family,
shape), it reports which `algo_family` niches ncnn's experts PREFER — used to
(1) order phase-1 illumination targets, (2) rank phase-2 top-k, (3) inject a
precise per-op hint into the proposer prompt.

SOFT prior: it *orders/biases*, it never hard-bans — QD still explores every
niche for coverage (design A.5). Only active for **compute_bound conv/deconv**
ops (the algo_family axis); otherwise `DispatchPrior.active is False` (no-op).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .roofline import COMPUTE_BOUND

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_PATH = _REPO / "experience_pool" / "backend_ncnn" / "dispatch_heuristics.json"

_CACHE: dict[str, dict] = {}


def load_dispatch(path: str | Path | None = None) -> dict:
    """Load the distilled dispatch doc (cached). Returns {} when absent — callers
    then degrade to an inactive (no-op) prior."""
    p = Path(path) if path else _DEFAULT_PATH
    key = str(p)
    if key not in _CACHE:
        try:
            _CACHE[key] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _CACHE[key] = {}
    return _CACHE[key]


# --- op → dispatch family key -------------------------------------------------
def _dispatch_key_from_layer(layer: str) -> str | None:
    lo = (layer or "").lower()
    if "convolutiondepthwise" in lo or "depthwise" in lo:
        return "conv_dw"
    if "deconvolution" in lo:
        return "deconv"
    if "convolution" in lo:
        return "conv"
    return None


def _pget(params: dict, pid: int, default: Any = None) -> Any:
    """Read ncnn param id (int or str key), coerced to int; default if absent/bad."""
    for k in (pid, str(pid)):
        if k in params:
            try:
                return int(params[k])
            except (TypeError, ValueError):
                return default
    return default


def shape_from_ncnn_params(params: dict[int, Any] | None) -> dict[str, int | None]:
    """Best-effort (kernel, stride, dilation, cin, cout) from ncnn conv params.

    IDs: 0=num_output, 1=kernel_w, 2=dilation_w, 3=stride_w, 6=weight_data_size,
    11=kernel_h. num_input is inferred from weight_data_size (approximate for
    grouped/depthwise — the prior is soft, so approximation is fine). Missing
    fields stay None → the predicate matcher treats unknown channels permissively.
    """
    params = params or {}
    kw = _pget(params, 1)
    kh = _pget(params, 11, kw)
    cout = _pget(params, 0)
    wds = _pget(params, 6)
    cin = None
    if wds and cout and kw and kh and cout * kw * kh > 0:
        cin = max(1, int(wds / (cout * kw * kh)))
    return {"kernel": kw, "stride": _pget(params, 3), "dilation": _pget(params, 2),
            "cin": cin, "cout": cout}


# --- predicate evaluation -----------------------------------------------------
def _match(pred: dict, shape: dict) -> bool:
    if pred.get("default"):
        return True
    for field_, key in (("kernel", "kernel"), ("stride", "stride"), ("dilation", "dilation")):
        if field_ in pred:
            v = shape.get(key)
            if v is None or v != pred[field_]:
                return False
    for pk, op in (("channel_any_ge", ">="), ("channel_any_gt", ">")):
        if pk in pred:
            thr = pred[pk]
            vals = [v for v in (shape.get("cin"), shape.get("cout")) if v is not None]
            if not vals:
                continue                       # unknown channels → permissive
            ok = any((v >= thr) if op == ">=" else (v > thr) for v in vals)
            if not ok:
                return False
    return True


def rank_niches(doc: dict, key: str, shape: dict) -> list[dict]:
    """Ordered algo_family prior: preferred (ncnn-selected here) first by source
    priority, then the remaining axis values as 'explore' (allowed, low rank)."""
    fam = (doc.get("families") or {}).get(key) or {}
    axis = doc.get("algo_family_axis") or ["direct", "gemm", "winograd", "fft", "dw"]
    best: dict[str, tuple[int, str, str]] = {}
    for r in fam.get("rules", []):
        if _match(r.get("predicate", {}), shape):
            af = r["algo_family"]
            prio = r.get("priority", 9)
            if af not in best or prio < best[af][0]:
                best[af] = (prio, r.get("when", ""), r.get("cite", ""))
    out: list[dict] = []
    for af, (prio, reason, cite) in sorted(best.items(), key=lambda kv: kv[1][0]):
        out.append({"algo_family": af, "preferred": True, "priority": prio,
                    "reason": reason, "cite": cite})
    seen = {r["algo_family"] for r in out}
    for af in axis:
        if af not in seen:
            out.append({"algo_family": af, "preferred": False, "priority": 99,
                        "reason": "explore (not ncnn-preferred for this shape)", "cite": ""})
    return out


# --- public prior object ------------------------------------------------------
@dataclass
class DispatchPrior:
    active: bool = False
    key: str | None = None
    shape: dict = field(default_factory=dict)
    ranked: list[dict] = field(default_factory=list)

    def preferred_order(self) -> list[str]:
        return [r["algo_family"] for r in self.ranked if r["preferred"]]

    def all_order(self) -> list[str]:
        return [r["algo_family"] for r in self.ranked]

    def rank_of(self, algo_family: str) -> int:
        """0-based rank of an algo_family in the prior (lower = more preferred).
        Unknown → large number, so unranked cells sort last in phase-2 top-k."""
        for i, r in enumerate(self.ranked):
            if r["algo_family"] == algo_family:
                return i
        return 99

    def render(self) -> str:
        if not self.active or not self.ranked:
            return ""
        sh = ", ".join(f"{k}={v}" for k, v in self.shape.items() if v is not None)
        lines = [f"ncnn dispatch prior for `{self.key}` ({sh or 'shape unknown'}) — "
                 f"which algorithm family ncnn's experts pick here:"]
        for r in self.ranked:
            tag = "PREFERRED" if r["preferred"] else "explore"
            cite = f"  [{r['cite']}]" if r.get("cite") else ""
            lines.append(f"- **{r['algo_family']}**: {tag} — {r['reason']}{cite}")
        lines.append("Bias bd_labels toward the PREFERRED algo_family cells first; "
                     "you MAY still target others for coverage.")
        return "\n".join(lines)


def build_prior(task_name: str, params: dict | None, regime: str, *,
                model_code: str = "", backend: str = "arm",
                path: str | Path | None = None) -> DispatchPrior:
    """Construct the dispatch prior for an op. Inactive (no-op) unless the op is a
    compute_bound conv/deconv with a distilled rule set."""
    if regime != COMPUTE_BOUND:
        return DispatchPrior()
    try:
        from ncnn_interface import guess_layer_from_task
        layer = guess_layer_from_task(task_name, model_code)
    except Exception:  # noqa: BLE001
        layer = task_name
    key = _dispatch_key_from_layer(layer or "")
    if not key:
        return DispatchPrior()
    doc = load_dispatch(path)
    if key not in (doc.get("families") or {}):
        return DispatchPrior(active=False, key=key)
    shape = shape_from_ncnn_params(params)
    return DispatchPrior(active=True, key=key, shape=shape,
                         ranked=rank_niches(doc, key, shape))
