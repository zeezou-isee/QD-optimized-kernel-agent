"""重具体化引擎(reify)—— 共享类型与分派入口。

对应《算子优化-完整Workflow.md》§6.4。本包把「参数化模板 + 一组具体参数值」
变成「可被 Evaluator 测量的具体化 kernel」。

设计原则:**reify 是纯函数**——只做「填值 → 产出无占位符的具体化产物 + 运行时
旋钮」,*不* 调用 GPU / 编译器。真正的 compile_spirv_module / pipeline->create /
增量编译由 #1 Evaluator 的 C++ harness 与 build 流程执行;reify 只产出它们消费的
manifest。这样 reify 在 #1(测量闭环)就绪前即可独立单测。

按后端分派:
    Vulkan  → reify.vulkan(运行时 spec constant + local_size,免重编)
    CPU/ARM → reify.cpu(宏/模板实例化 + 隔离增量重编译)
"""

from __future__ import annotations

from typing import Any

from utils.param_contract import Backend, KernelTemplate, validate_template


def reify(tpl: KernelTemplate, values: dict[str, Any]) -> Any:
    """统一入口:按 backend 分派。返回后端特定的 Reified* 对象。

    这是内层参数搜索(§6)调用的唯一接口。
    """
    validate_template(tpl)
    _check_values_cover_params(tpl, values)

    if tpl.backend is Backend.VULKAN:
        from utils.reify.vulkan import reify_vulkan

        return reify_vulkan(tpl, values)
    if tpl.backend is Backend.CPU_ARM:
        from utils.reify.cpu import reify_cpu

        return reify_cpu(tpl, values)
    raise ValueError(f"未知后端: {tpl.backend}")


def _check_values_cover_params(tpl: KernelTemplate, values: dict[str, Any]) -> None:
    """每个参数都必须有取值,且取值须在该参数的候选集内(挡掉越界点)。"""
    for p in tpl.params:
        if p.name not in values:
            raise ValueError(f"reify: 缺少参数取值 {p.name!r}")
        if values[p.name] not in p.candidates:
            raise ValueError(
                f"reify: 参数 {p.name!r} 取值 {values[p.name]!r} 不在候选 {p.candidates}"
            )
    extra = set(values) - {p.name for p in tpl.params}
    if extra:
        raise ValueError(f"reify: 多余的参数取值 {sorted(extra)}")
