# eval — 驱动 `agents/` 的实验 / 验证脚本

这里是一组**一次性实验 / 批量验证脚本**,用来在 `datasets/MobileKernelBench` 上驱动
`agents/`(KernelAgent / GraphAgent / OperatorAgent),并把结果写成报告。它们不是
agent 系统本身,只是跑实验的外围工具。

> 运行前提与 `agents/` 相同:需要 `torch numpy openai`、构建好的 `frameworks/ncnn`
> `libncnn`(`build/` 或 `build_lib/`)与 `tools/pnnx`,以及 `OPENROUTER_API_KEY`。
> 每个脚本都会把仓库根与 `agents/` 加到 `sys.path`(`import agents; bootstrap_paths()`),
> torch 目录通过 `_torch_dir()` 自动探测(探测不到则交给 pnnx 自己找)。

## 脚本一览

| 脚本 | 作用 | 是否用 LLM | 产物 |
|---|---|---|---|
| `batch_probe.py` | 廉价分类:数据集里“未支持”算子里哪些其实已被原生 pnnx 支持 | 否 | `probe_classify.json` |
| `verify_ops.py` | 已注册算子(Greater/LessEqual)的原生 pnnx+ncnn 端到端数值验证 | 否 | `REPORT.md` / `results.json` |
| `verify_imperative_path.py` | 手写(无 LLM)命令式 pass 注入路径的可行性验证(Trilu_lower) | 否 | — |
| `batch_test.py` | 对 10 个多样的真未支持算子跑 OperatorAgent 全流程 | 是 | `BATCH_REPORT.md` / `batch_results.json` |
| `batch_test_hard.py` | 对 10 个“难”算子只跑 kernel+graph,捕获 graph 失败原因 | 是 | `batch_hard_results.json` |
| `rerun_failed.py` | 用修好的 harness 重跑此前失败的算子 | 是 | `rerun_results.json` |
| `rerun_remaining.py` | 续跑上轮未完成的算子(OneHot/Det/Unique) | 是 | `rerun_results.json` |
| `rerun_multinode.py` | 命令式 pass 支持后重跑 3 个多节点算子 | 是 | `rerun_multinode_results.json` |
| `v03_smoke.py` | 4 个代表算子的 7 阶段流程冒烟 | 是 | `v03_smoke_results.json` |
| `install_ops.py` | 把验证过的算子**永久注册**进 ncnn+pnnx(不回滚) | 否 | 改写 ncnn 源码树 |

仓库里保留的报告 `REPORT.md` / `BATCH_REPORT.md` / `pnnx_shortcut_report.md` 是历史运行的
可读结论(README 中引用的验证结果);各 `*_results.json` 原始 dump 为运行时产物,已不入库。

## 运行

```bash
cd /home/pc/projects/KernelAgent
export OPENROUTER_API_KEY=sk-or-v1-...
python eval/batch_probe.py          # 无 LLM,先看哪些算子真未支持
python eval/verify_ops.py           # 无 LLM,验证已注册算子
python eval/batch_test.py           # 用 LLM 跑端到端批量测试
```

> ⚠️ `install_ops.py` 会改写 `frameworks/ncnn` 源码树且**不回滚**,确认后再跑。
