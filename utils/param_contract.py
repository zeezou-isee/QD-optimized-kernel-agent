"""
核心数据流(§3):
    Proposer(LLM) ──► KernelTemplate(带占位符的 source + params[] + 理由)
                          │
    外层 → 内层: (parameterized_template, candidate_values, hardware_specs)
    内层 → 外层: (best_params, best_latency, correct, profile)

本模块只定义「契约 + 校验」,不含重具体化引擎本身(见后续 reify/ 模块)。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# 后端枚举
# ---------------------------------------------------------------------------
class Backend(str, Enum):
    """目标后端。参数化机制按后端分,不按算子分。"""

    VULKAN = "vulkan"        # ncnn Vulkan:运行时 compile_spirv_module + spec constant,近乎免重编
    CPU_ARM = "cpu_arm"      # ncnn CPU/ARM:宏/模板参数 + 隔离增量重编译


# ---------------------------------------------------------------------------
# ncnn Vulkan spec_id 保留号段(实测自 frameworks/ncnn/src/layer/vulkan/shader/*.comp)
# ---------------------------------------------------------------------------
# 框架自身占用:constant_id 0~3 = bias_term/activation_type/activation_param_0/1;
# 其后 shape_constant_id_offset(实测最高到 12)再加最多 10 个 shape 槽 → 约用到 0~22。
# 可调旋钮的 spec_id 一律从 TUNABLE_SPEC_ID_BASE 起,彻底避开框架号段(防撞车)。
NCNN_RESERVED_SPEC_ID_MAX = 63
TUNABLE_SPEC_ID_BASE = 64


# ---------------------------------------------------------------------------
# 参数绑定方式:旋钮在源码里「如何被替换」,决定 reify 引擎怎么灌值
# ---------------------------------------------------------------------------
class BindingKind(str, Enum):
    """旋钮与源码的绑定方式。同一后端可混用多种(如 Vulkan 的 tile 走 spec
    constant,但 wg_size 走 local_size——见下方说明)。"""

    # Vulkan:layout(constant_id=N) const ...。运行时灌进 specializations[],
    #   create_pipeline 时驱动重优化 SPIR-V,无需重生成源码(§6.4「近乎免费」)。
    SPEC_CONSTANT = "spec_constant"

    # Vulkan:workgroup 尺寸。实测 ncnn shader 全仓 *未* 用 local_size_x_id,
    #   而由 C++ 侧 set_optimal_local_size_xyz 设定 → 单列一种 binding,
    #   reify 时走 pipeline 的 local_size 接口而非 specializations[]。
    LOCAL_SIZE = "local_size"

    # CPU/ARM:#define NAME value,编译期生效 → 需重编译(§6.4)。
    MACRO = "macro"

    # CPU/ARM:C++ 模板参数 template<int NAME> → 实例化 + 重编译。
    TEMPLATE_PARAM = "template_param"


class ParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"          # 取值来自一组离散标签(如 layout 族),候选即全部合法值


@dataclass
class ParamSpec:
    """单个可调旋钮的声明。由 LLM 在生成 kernel 时按契约吐出。

    注意 candidates 是 *该模板* 暴露的候选档位(per-template,不是 per-operator
    的全局清单)——同一算子的不同算法族模板会暴露完全不同的旋钮。
    """

    name: str                         # 源码中占位符标识(宏名 / 模板参数名 / 常量名)
    type: ParamType
    binding: BindingKind
    candidates: list[Any]             # 离散候选值(内层粗网格/爬山在此之上搜)
    spec_id: int | None = None        # 仅 SPEC_CONSTANT 必填;须 >= TUNABLE_SPEC_ID_BASE
    default: Any | None = None        # 缺省值(冷启动/对照基线用)
    note: str = ""                    # LLM 给的「为何可调 / 物理含义」,便于解析剪枝

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError(f"param {self.name!r}: candidates 不可为空")
        if self.default is None:
            self.default = self.candidates[0]
        elif self.default not in self.candidates:
            raise ValueError(
                f"param {self.name!r}: default {self.default!r} 不在 candidates 中"
            )


@dataclass
class KernelTemplate:
    """一份参数化模板 —— Proposer 的产出、内层搜索的输入(§3 接口契约)。

    source 是带占位符的 kernel 源码;params 声明这些占位符如何被替换。
    reify 引擎吃 (template, 一组具体 param 取值) → 产出可测的具体化 kernel。
    """

    op_id: str                        # 算子标识(数据集相对路径,如 "Matrix/GEMM")
    backend: Backend
    source: str                       # 带占位符的源码(Vulkan .comp / C++)
    params: list[ParamSpec] = field(default_factory=list)
    structural_bd: dict[str, str] = field(default_factory=dict)  # §4.3 结构标签轴(算法族/布局/计算映射…),生成时即知
    rationale: str = ""               # §3「为何应有增益」的理由

    def grid_size(self) -> int:
        """候选网格规模 = ∏ 各旋钮候选数(解析剪枝前的上界)。"""
        n = 1
        for p in self.params:
            n *= len(p.candidates)
        return n

    def default_values(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


# ---------------------------------------------------------------------------
# 契约校验:在模板进入内层搜索前调用,挡掉 LLM 产出的不合规声明
# ---------------------------------------------------------------------------
def validate_template(tpl: KernelTemplate) -> None:
    """校验模板符合契约。不合规直接抛错,避免脏模板污染搜索/落格。"""
    names: set[str] = set()
    spec_ids: set[int] = set()

    for p in tpl.params:
        if p.name in names:
            raise ValueError(f"重复的参数名: {p.name!r}")
        names.add(p.name)

        _check_binding_backend(tpl.backend, p)

        if p.binding is BindingKind.SPEC_CONSTANT:
            if p.spec_id is None:
                raise ValueError(f"param {p.name!r}: SPEC_CONSTANT 必须指定 spec_id")
            if p.spec_id < TUNABLE_SPEC_ID_BASE:
                raise ValueError(
                    f"param {p.name!r}: spec_id={p.spec_id} 落入框架保留段 "
                    f"(< {TUNABLE_SPEC_ID_BASE}),会与 ncnn 内置常量撞车"
                )
            if p.spec_id in spec_ids:
                raise ValueError(f"param {p.name!r}: spec_id={p.spec_id} 重复")
            spec_ids.add(p.spec_id)


def _check_binding_backend(backend: Backend, p: ParamSpec) -> None:
    """binding 与后端必须匹配(spec/local_size↔Vulkan,macro/template↔CPU)。"""
    vulkan_bindings = {BindingKind.SPEC_CONSTANT, BindingKind.LOCAL_SIZE}
    cpu_bindings = {BindingKind.MACRO, BindingKind.TEMPLATE_PARAM}
    if backend is Backend.VULKAN and p.binding not in vulkan_bindings:
        raise ValueError(
            f"param {p.name!r}: binding={p.binding.value} 不适用于 Vulkan 后端"
        )
    if backend is Backend.CPU_ARM and p.binding not in cpu_bindings:
        raise ValueError(
            f"param {p.name!r}: binding={p.binding.value} 不适用于 CPU/ARM 后端"
        )


# ---------------------------------------------------------------------------
# 序列化:模板在 LLM ↔ 搜索器 ↔ 落盘之间以 JSON 流转
# ---------------------------------------------------------------------------
def template_to_json(tpl: KernelTemplate) -> str:
    return json.dumps(asdict(tpl), ensure_ascii=False, indent=2)


def template_from_json(data: str | dict) -> KernelTemplate:
    obj = json.loads(data) if isinstance(data, str) else data
    params = [
        ParamSpec(
            name=p["name"],
            type=ParamType(p["type"]),
            binding=BindingKind(p["binding"]),
            candidates=p["candidates"],
            spec_id=p.get("spec_id"),
            default=p.get("default"),
            note=p.get("note", ""),
        )
        for p in obj.get("params", [])
    ]
    return KernelTemplate(
        op_id=obj["op_id"],
        backend=Backend(obj["backend"]),
        source=obj["source"],
        params=params,
        structural_bd=obj.get("structural_bd", {}),
        rationale=obj.get("rationale", ""),
    )
