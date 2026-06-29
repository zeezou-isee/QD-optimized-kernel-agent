"""Load ncnn/docs/.../operation-param-weight-table.md into a dict for cross-check.

The table is markdown-formatted:

    |operation|param id|param phase|default value|weight order|
    |:---:|:---:|:---:|:---:|:---:|
    |AbsVal|||
    |ArgMax|0|out_max_val|0|
    ||1|topk|1|
    |BatchNorm|0|channels|0|slope mean variance bias|

A new op starts on a row where col-1 is non-empty. Continuation rows have col-1
empty and add another (id, name, default) to the previous op.

Returned shape:
    {
      "AbsVal":     {"params": [], "weight_order": []},
      "BatchNorm":  {"params": [{"id":0,"name":"channels","default":"0"},
                                {"id":1,"name":"eps","default":"0.f"}],
                     "weight_order": ["slope","mean","variance","bias"]},
      ...
    }
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# Parse rows by splitting on `|` instead of regex-matching exactly 5 columns —
# the table is hand-maintained and uses both `|AbsVal|||` (3 pipes, 0-param ops)
# and `|BinaryOp|0|op_type|0|` (5 pipes, full row). A split-and-pad pass copes
# with both shapes.

def load_doc_table(md_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not md_path.exists():
        return out
    cur_op: str | None = None
    for raw in md_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("|") or not line.endswith("|"):
            cur_op = None
            continue
        # strip the leading and trailing `|`, then split
        cols_raw = [c.strip() for c in line[1:-1].split("|")]
        # pad up to 5 cols so unpacking is safe; truncate longer rows
        cols = (cols_raw + ["", "", "", "", ""])[:5]
        op_col, id_col, name_col, default_col, weight_col = cols
        # skip header / alignment rows
        if op_col == "operation" or set(op_col) == {":", "-"} or all(set(c) <= {":", "-"} for c in cols if c):
            cur_op = None
            continue
        # continuation row (op col blank, but we have a current op)
        if not op_col:
            if cur_op and id_col.isdigit():
                out[cur_op]["params"].append({
                    "id": int(id_col),
                    "name": name_col,
                    "default": default_col,
                })
            continue
        # new op row
        cur_op = op_col
        entry = out.setdefault(cur_op, {"params": [], "weight_order": []})
        if id_col.isdigit():
            entry["params"].append({
                "id": int(id_col),
                "name": name_col,
                "default": default_col,
            })
        if weight_col:
            entry["weight_order"] = weight_col.split()
    return out
