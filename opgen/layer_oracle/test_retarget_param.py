"""Minimal unit verification for .ncnn.param re-targeting (retarget_param_layer).

Pure text-level checks (no torch / ncnn needed). Confirms that:
  - only the layer TYPE token of matching lines is rewritten,
  - the two header lines, blob names, layer names, params and whitespace are kept,
  - matching is exact by type (HardSigmoid != Sigmoid) and never hits a blob/layer
    NAME that coincidentally equals the type,
  - layer_name filter retargets exactly one line,
  - the result still parses (parse_ncnn_io) with identical IO blobs.

Run:  python opgen/layer_oracle/test_retarget_param.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # opgen/ on path

from layer_oracle.net_oracle import (  # noqa: E402
    retarget_param_layer,
    retarget_param_output_layer,
    parse_ncnn_io,
)

# A realistic tiny pnnx-style .ncnn.param:
#   - an Input layer (must NOT be retargeted)
#   - two Sigmoid layers (TYPE = Sigmoid)
#   - one HardSigmoid layer (must NOT match "Sigmoid")
#   - a layer whose NAME is "Sigmoid" but TYPE is ReLU (must NOT be retargeted)
#   - a blob coincidentally named "Sigmoid" (must NOT be touched)
PARAM = (
    "7767517\n"
    "5 5\n"
    "Input            in0          0 1 in0 0=4 1=4 2=3\n"
    "Sigmoid          sig1         1 1 in0 Sigmoid\n"        # blob named 'Sigmoid'
    "Sigmoid          act2         1 1 Sigmoid out1\n"
    "HardSigmoid      hs3          1 1 out1 out2 0=0.2 1=0.5\n"
    "ReLU             Sigmoid      1 1 out2 out3\n"          # layer NAME == 'Sigmoid'
)

_FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        _FAILS.append(msg)


def main() -> int:
    # --- 1) retarget all Sigmoid-typed layers ---
    new, n = retarget_param_layer(PARAM, "Sigmoid", "Cand_Sigmoid")
    lines = new.splitlines()

    check(n == 2, f"replaced exactly 2 Sigmoid-typed layers (got {n})")
    check(lines[0] == "7767517", "magic header preserved")
    check(lines[1] == "5 5", "count header preserved")

    # the two Sigmoid layers became Cand_Sigmoid; everything else on the line intact
    check(lines[2].startswith("Input "), "Input layer untouched")
    check(lines[3].split()[0] == "Cand_Sigmoid", "layer 1 type -> Cand_Sigmoid")
    check(lines[3].split()[1] == "sig1", "layer 1 name 'sig1' preserved")
    check("in0 Sigmoid" in lines[3], "blob named 'Sigmoid' on layer 1 NOT touched")
    check(lines[4].split()[0] == "Cand_Sigmoid", "layer 2 type -> Cand_Sigmoid")
    check("Sigmoid out1" in lines[4], "input blob 'Sigmoid' on layer 2 NOT touched")
    check(lines[5].split()[0] == "HardSigmoid", "HardSigmoid NOT matched by 'Sigmoid'")
    check(lines[6].split()[0] == "ReLU" and lines[6].split()[1] == "Sigmoid",
          "layer NAMED 'Sigmoid' (type ReLU) NOT retargeted")
    check("0=0.2 1=0.5" in lines[4 + 0] or "0=0.2 1=0.5" in new, "params preserved")

    # IO blobs unchanged after retarget
    in_a, out_a = parse_ncnn_io(PARAM)
    in_b, out_b = parse_ncnn_io(new)
    check(in_a == in_b and out_a == out_b, f"IO blobs unchanged ({in_a}->{out_a})")

    # --- 2) retarget a single layer by name ---
    new1, n1 = retarget_param_layer(PARAM, "Sigmoid", "Cand_Sigmoid", layer_name="act2")
    l1 = new1.splitlines()
    check(n1 == 1, f"layer_name filter retargets exactly 1 (got {n1})")
    check(l1[3].split()[0] == "Sigmoid", "non-matching name 'sig1' left as Sigmoid")
    check(l1[4].split()[0] == "Cand_Sigmoid", "named 'act2' -> Cand_Sigmoid")

    # --- 3) no match -> 0 replacements, text unchanged ---
    new0, n0 = retarget_param_layer(PARAM, "Tanh", "Cand_Tanh")
    check(n0 == 0 and new0 == PARAM, "no-match leaves param byte-identical")

    # --- 4) output-layer retarget: robust where native type != task name ---
    # torch.exp converts to ncnn 'UnaryOp' (op_type 0=7), not 'Exp'. We can't guess
    # the type from the name, but we CAN retarget the layer producing the output.
    UNARY = (
        "7767517\n"
        "2 2\n"
        "Input            in0    0 1 in0 0=8 1=8 2=4\n"
        "UnaryOp          exp0   1 1 in0 out0 0=7\n"
    )
    ru, nu = retarget_param_output_layer(UNARY, "Cand_Exp")
    lu = ru.splitlines()
    check(nu == 1, f"output-layer retarget hit exactly 1 (got {nu})")
    check(lu[2].startswith("Input "), "Input not retargeted")
    check(lu[3].split()[0] == "Cand_Exp", "UnaryOp output layer -> Cand_Exp")
    check("in0 out0 0=7" in lu[3], "blobs + op_type param preserved")
    _, out_u = parse_ncnn_io(UNARY)
    _, out_u2 = parse_ncnn_io(ru)
    check(out_u == out_u2 == "out0", "output blob unchanged after retarget")

    # idempotent: retargeting an already-Cand_ output layer is a no-op rewrite
    ri, ni = retarget_param_output_layer(ru, "Cand_Exp")
    check(ni == 1 and ri == ru, "idempotent: cls->cls leaves text identical")

    print()
    if _FAILS:
        print(f"FAILED ({len(_FAILS)})")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
