"""Vulkan 重具体化引擎(§6.4「近乎免费」路径)。

ncnn Vulkan 旋钮分两类(对应 BindingKind):
  - SPEC_CONSTANT → 灌进 specializations[](vk_specialization_type union: .i/.f/.u32),
        运行时 pipeline->create(spv, size, specializations) 时驱动重优化 SPIR-V。
  - LOCAL_SIZE    → workgroup 尺寸。实测 ncnn shader 全仓未用 local_size_x_id,
        故走 C++ 侧 set_optimal_local_size_xyz(w,h,c),不进 specializations[]。

reify 产出 ReifiedVulkan = (具体化 shader 源码 + spec 数组 + local_size),
交给 #1 的 C++ harness 去 compile_spirv_module → create。本步不碰 GPU。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.param_contract import (
    BindingKind,
    KernelTemplate,
    ParamSpec,
    ParamType,
)


@dataclass
class SpecEntry:
    """specializations[] 的一个槽位。field 决定 harness 用 union 的哪个成员。"""

    spec_id: int
    value: Any
    field: str  # "i" | "f" | "u32"


@dataclass
class ReifiedVulkan:
    """Vulkan 具体化产物 —— harness 的输入 manifest。"""

    op_id: str
    shader_source: str               # 无占位符的 .comp 源码(spec constant 仍由驱动注入)
    specializations: list[SpecEntry] = field(default_factory=list)
    local_size: tuple[int, int, int] | None = None  # 给 set_optimal_local_size_xyz
    values: dict[str, Any] = field(default_factory=dict)  # 本次取值快照(落格/复现用)


_TYPE_TO_FIELD = {
    ParamType.INT: "i",
    ParamType.BOOL: "i",      # ncnn 以 int 表 bool
    ParamType.ENUM: "i",      # 枚举须映射为 int 档位(见下)
    ParamType.FLOAT: "f",
}


def reify_vulkan(tpl: KernelTemplate, values: dict[str, Any]) -> ReifiedVulkan:
    specs: list[SpecEntry] = []
    local_xyz: dict[str, int] = {}

    for p in tpl.params:
        v = values[p.name]
        if p.binding is BindingKind.SPEC_CONSTANT:
            specs.append(
                SpecEntry(spec_id=p.spec_id, value=_coerce(p, v), field=_field_of(p))
            )
        elif p.binding is BindingKind.LOCAL_SIZE:
            # 约定 LOCAL_SIZE 参数名以 _x/_y/_z 结尾,分别填 workgroup 三维
            axis = p.name.rsplit("_", 1)[-1]
            if axis not in ("x", "y", "z"):
                raise ValueError(
                    f"LOCAL_SIZE 参数 {p.name!r} 名须以 _x/_y/_z 结尾以标明轴"
                )
            local_xyz[axis] = int(v)
        else:
            raise ValueError(f"Vulkan 不支持 binding={p.binding.value}")

    local_size = None
    if local_xyz:
        local_size = (
            local_xyz.get("x", 4),
            local_xyz.get("y", 4),
            local_xyz.get("z", 4),
        )

    return ReifiedVulkan(
        op_id=tpl.op_id,
        shader_source=tpl.source,  # spec constant 无需文本替换;驱动按 spec_id 注入
        specializations=specs,
        local_size=local_size,
        values=dict(values),
    )


def _field_of(p: ParamSpec) -> str:
    f = _TYPE_TO_FIELD.get(p.type)
    if f is None:
        raise ValueError(f"param {p.name!r}: 类型 {p.type} 无法映射到 spec union 字段")
    return f


def _coerce(p: ParamSpec, v: Any) -> Any:
    """把候选值规整为 union 字段所需的 Python 类型。

    ENUM:候选是标签字符串,spec constant 只能吃数值 → 映射为「在 candidates
    中的下标」。harness 侧与 shader 侧须共享同一档位约定。
    """
    if p.type is ParamType.ENUM:
        return p.candidates.index(v)
    if p.type is ParamType.FLOAT:
        return float(v)
    if p.type is ParamType.BOOL:
        return int(bool(v))
    return int(v)
