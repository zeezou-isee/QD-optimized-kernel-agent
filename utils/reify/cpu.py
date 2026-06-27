"""CPU/ARM 重具体化引擎(§6.4 重编译路径)。

CPU 无运行时常量注入,旋钮是编译期的:
  - MACRO          → 在源码头部注入 #define NAME value。
  - TEMPLATE_PARAM → 文本替换占位符 token «NAME»(C++ 模板参数实例化)。

reify 只产出「无占位符的具体化源码 + 构建 manifest」;真正的隔离增量重编译
(只重编被替换的单个 layer,避免每点全量编译炸预算)由 #1 的 build 流程执行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.param_contract import BindingKind, KernelTemplate, ParamType


# 占位符约定:TEMPLATE_PARAM 在源码中以 «NAME» 标记(与 C/C++ 标识符不冲突)。
def _placeholder(name: str) -> str:
    return f"«{name}»"  # «NAME»


@dataclass
class ReifiedCpu:
    """CPU/ARM 具体化产物 —— build 流程的输入 manifest。"""

    op_id: str
    source: str                       # 无占位符的具体化 C++ 源码
    defines: dict[str, Any] = field(default_factory=dict)  # 也可由 build 转成 -D 编译参数
    values: dict[str, Any] = field(default_factory=dict)


def reify_cpu(tpl: KernelTemplate, values: dict[str, Any]) -> ReifiedCpu:
    src = tpl.source
    defines: dict[str, Any] = {}

    for p in tpl.params:
        v = _format(p.type, values[p.name])
        if p.binding is BindingKind.MACRO:
            defines[p.name] = v
        elif p.binding is BindingKind.TEMPLATE_PARAM:
            token = _placeholder(p.name)
            if token not in src:
                raise ValueError(
                    f"param {p.name!r}: 源码中未找到占位符 {token}"
                )
            src = src.replace(token, str(v))
        else:
            raise ValueError(f"CPU/ARM 不支持 binding={p.binding.value}")

    if defines:
        header = "".join(f"#define {k} {v}\n" for k, v in defines.items())
        src = header + src

    return ReifiedCpu(
        op_id=tpl.op_id,
        source=src,
        defines=defines,
        values=dict(values),
    )


def _format(ptype: ParamType, v: Any) -> Any:
    """格式化为可直接写进 C++ 源码的字面量。"""
    if ptype is ParamType.BOOL:
        return 1 if v else 0
    if ptype is ParamType.FLOAT:
        # 带 f 后缀避免被当作 double,影响 NEON 路径
        return f"{float(v)}f"
    if ptype is ParamType.ENUM:
        return v  # 枚举标签直接作为宏值(须是合法 C++ token,如 NCHW)
    return int(v)
