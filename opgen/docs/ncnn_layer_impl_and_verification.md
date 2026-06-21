# 从零在 ncnn 中实现一个 layer(算子)并验证其正确性

> 背景:MoKA 的算子验证建立在"该 layer 已存在"的前提上(它改的是已有 layer + 已有计算图转换)。本文回答更底层的问题:**从无到有**怎么新增一个 ncnn layer、需要哪些文件、以及——关键——**在没有 baseline 的情况下怎么验证这个 layer 的输出是对的**。

基于 `ncnn/` 源码与 `docs/developer-guide/how-to-implement-custom-layer-step-by-step.md` 实测整理。

---

## 0. 先厘清一个依赖关系(回答之前的"先有鸡还是先有蛋")

要在 ncnn 里跑**任何**推理(无论是 layer 单测,还是 pnnx 端到端),**该 layer 必须先被注册存在**——哪怕只是一个最朴素的 forward。所以从零的正确顺序是:

```
① 实现 layer(src/layer + 注册)  ──②外部 oracle 验证(vs PyTorch,layer 级,不需要计算图)
                                          │
                                          ▼ layer 数学正确后
③ 实现 pnnx 计算图转换(PyTorch op → 这个 layer) ──④ 端到端验证(pnnx→ncnn vs PyTorch)
```

**layer 在前,转换在后。** layer 的正确性靠"外部 oracle(PyTorch)"独立验证,不依赖计算图转换、也不依赖任何 baseline。

---

## 1. 需要实现哪些文件

### 1.1 layer 本体(必需)
```
ncnn/src/layer/mylayer.h     # 类声明:继承 ncnn::Layer
ncnn/src/layer/mylayer.cpp   # 实现 + DEFINE_LAYER_CREATOR(MyLayer)
```
最小骨架(参考文档 step1–step6 与 `src/layer/absval.*`):

```cpp
// mylayer.h
#include "layer.h"
namespace ncnn {
class MyLayer : public Layer {
public:
    MyLayer();                                   // 设定 one_blob_only / support_inplace
    virtual int load_param(const ParamDict& pd); // 读 0=.. 1=.. 参数
    virtual int load_model(const ModelBin& mb);  // 读权重(无权重可省)
    virtual int forward(const Mat& bottom, Mat& top, const Option& opt) const;
private:
    float eps;
};
} // namespace ncnn

// mylayer.cpp
#include "mylayer.h"
namespace ncnn {
MyLayer::MyLayer() { one_blob_only = true; support_inplace = false; }
int MyLayer::load_param(const ParamDict& pd) { eps = pd.get(0, 1e-5f); return 0; }
int MyLayer::load_model(const ModelBin&) { return 0; }
int MyLayer::forward(const Mat& bottom, Mat& top, const Option& opt) const {
    // ... 计算,写 top ...
    return 0;
}
DEFINE_LAYER_CREATOR(MyLayer)   // 生成 MyLayer_layer_creator
} // namespace ncnn
```

**forward 接口按行为四选一**(文档 step5 表):

| one_blob_only | support_inplace | 实现哪个 |
|---|---|---|
| true | false | `forward(const Mat&, Mat&, opt)` |
| true | true | `forward_inplace(Mat&, opt)`(可选再写非 inplace) |
| false | false | `forward(vector<Mat>&in, vector<Mat>&out, opt)` |
| false | true | `forward_inplace(vector<Mat>&, opt)` |

### 1.2 注册(必需,二选一)

**A. 编译进内置注册表(MoKA / 端到端走这条)**
在 `ncnn/src/CMakeLists.txt` 加一行:
```cmake
ncnn_add_layer(MyLayer)
```
`cmake/ncnn_add_layer.cmake` 会**自动生成** `layer_registry.h` 条目、layer 类型枚举、以及 naive/arch-opt 变体声明。之后 `ncnn::create_layer("MyLayer")` 和按 typeindex 创建都可用。**改了要重编 ncnn。**

> 注意:`ncnn_add_layer(X)` 默认 ON;`ncnn_add_layer(X OFF)` 表示不编进库(如 ArgMax)。

**B. 运行时注册(不改 ncnn 源码、不重编库)**
```cpp
net.register_custom_layer("MyLayer", MyLayer_layer_creator);
net.load_param(...); net.load_model(...);
```
适合插件式;但 layer 单测框架 `test_layer` 走的是内置注册表(A),所以做系统化验证通常用 A。

### 1.3 (可选)性能优化变体
`src/layer/x86/mylayer_x86.cpp`、`src/layer/arm/mylayer_arm.cpp` 等——arch 特定加速。**从零阶段不需要**,先把朴素 `src/layer/mylayer.cpp` 写对。

### 1.4 (验证用)测试文件
- 内置一致性测试:`ncnn/tests/test_mylayer.cpp` + 在 `tests/CMakeLists.txt` 注册。
- 外部 oracle 测试:见第 3 节(MoKA 的 `tests_naive` 风格)。

---

## 2. ncnn 自带的 `test_layer` 验证的是什么(关键认知)

`tests/test_*.cpp` 调 `test_layer("MyLayer", pd, weights, a)`,其内部(`tests/testutil.cpp`)逻辑是:

```cpp
b = test_layer_naive(...)  // 用 create_layer_naive:关闭 packing/fp16/bf16/vulkan 的"朴素"实现
c = test_layer_cpu(...)    // 开启 packing/fp16/bf16 等优化路径
CompareMat(b, c, epsilon)  // 比较:优化路径 == 朴素实现?
```

→ **它只验"内部一致性"**:你的优化变体(以及不同 packing/数据类型/形状布局下)和你自己的朴素实现结果一致。
→ **它默认朴素实现就是对的,不和 PyTorch 等外部真值比较。**

含义:
- 对一个**从零写的算子**,你最先写的 forward 就是"朴素实现"。`test_layer` **无法告诉你这个朴素实现的数学对不对**(它没有外部参照),只能在你加了优化变体后保证它们与朴素一致,以及朴素实现在各种形状/布局下不崩。
- 所以 **"从零算子是否正确"这件事,`test_layer` 给不了答案,必须靠外部 oracle。**

---

## 3. 从零验证正确性:用 PyTorch 当 oracle(核心)

**没有 baseline 时,正确性的"标准答案"是 PyTorch。** layer 级验证(不需要计算图转换):

```
PyTorch 算子 ──给定输入 x──► 参考输出 y_ref   (ground truth)
                                  │
直接构造 ncnn MyLayer ──同一个 x──► y_ncnn
                                  │
                  allclose(y_ref, y_ncnn) ◄──┘
```

这正是 MoKA `operator_verification.py` 做的事(它跑的是编译好的 `ncnn/build/tests_naive/<test>` 可执行):
1. Python 跑 PyTorch:`model(x)` 得 `y_ref`;把 `x`(及权重)用自定义二进制协议 `[ndim][dims...][float data]` 写成 `.bin`。
2. C++ 测试程序:`ncnn::create_layer("MyLayer")` → `load_param`/`load_model` → 把 `x.bin` 读成 `ncnn::Mat` → `forward` → 把 `y_ncnn` 写回 `.bin`。
3. Python 读回 `y_ncnn`,`mean_abs_diff <= 1e-3` 即通过。

**这条链路完全不需要 pnnx、不需要计算图转换、不需要任何 baseline** —— 只要 layer 能被创建并 forward。它是验证"从零 layer 数学正确"的唯一可靠手段。

### 最小外部验证 harness(C++ 侧示意)
```cpp
// test_mylayer_naive.cpp  —— 编成可执行,argv: in.bin [weights...] out.bin
#include "layer.h"
#include "net.h"
int main(int argc, char** argv) {
    ncnn::Mat in = read_bin(argv[1]);                 // 自定义 [ndim][dims][data] 读入
    ncnn::ParamDict pd; /* pd.set(0, ...); 按需 */
    std::vector<ncnn::Mat> weights = read_weights(argv, ...); // 若有权重
    ncnn::Layer* op = ncnn::create_layer("MyLayer");  // 需 layer 已注册(1.2 A)
    op->load_param(pd);
    ncnn::ModelBinFromMatArray mb(weights.data()); op->load_model(mb);
    ncnn::Option opt; op->create_pipeline(opt);
    ncnn::Mat out; op->forward(in, out, opt);          // 或 forward_inplace
    write_bin(out, argv[argc-1]);                      // 写回供 Python 比对
    op->destroy_pipeline(opt); delete op; return 0;
}
```
Python 侧:跑 PyTorch 得 `y_ref` + 写 `in.bin` → 调上面可执行 → 读 `out.bin` → `np.allclose`。

> shape 映射要注意 ncnn 约定:PyTorch `(N,C,H,W)` → ncnn `Mat(w,h,c)`;`(N,C)` → `Mat(c)` 等(见 MoKA `torch_shape_to_ncnn`)。

---

## 4. 两种验证的分工(一句话)

| 验证 | 工具 | 比较对象 | 回答的问题 | 从零是否够用 |
|---|---|---|---|---|
| **外部 oracle** | 自建 harness / MoKA tests_naive | ncnn 输出 vs **PyTorch** | 我的算子**数学**对不对 | ✅ 核心,必需 |
| **内部一致性** | ncnn `tests/test_*.cpp`(`test_layer`) | 优化路径 vs **自己的朴素实现** | 优化/打包/形状有没有引入偏差、会不会崩 | ❌ 不验数学正确,加优化后才有意义 |

**从零的正确做法:先用外部 oracle 确认朴素 forward 数学正确 → 再(可选)加优化变体并用 `test_layer` 保证一致。**

---

## 5. 这对 graph_agent 的意义

- 之前 graph_agent 的 `verify_numeric` 是**端到端**(pnnx→ncnn vs PyTorch),它隐含要求 layer + 计算图转换都就绪。
- 真正"从零"应拆成**两级 agent / 两步**:
  1. **layer agent**:写 `src/layer/mylayer.{h,cpp}` + `ncnn_add_layer` → **layer 级外部 oracle 验证**(本文第 3 节,不需要计算图)。这一步先把 kernel 数学验对。
  2. **graph agent**(已有):写 pnnx 转换 → 端到端验证。
- 关键依赖:**graph agent 的端到端数值验证,前提是 layer 已存在且已通过 layer 级验证**;否则只能做结构验证。这正是你指出的"pnnx 验证必须 layer 中有算子文件存在"。

---

## 6. 从零新增一个算子的完整文件清单

| 步骤 | 文件 | 必需性 |
|---|---|---|
| layer 声明 | `ncnn/src/layer/mylayer.h` | ✅ |
| layer 实现 + `DEFINE_LAYER_CREATOR` | `ncnn/src/layer/mylayer.cpp` | ✅ |
| 内置注册 | `ncnn/src/CMakeLists.txt` 加 `ncnn_add_layer(MyLayer)` | ✅(端到端/单测需要) |
| layer 级外部验证 | `tests_naive/test_mylayer.cpp`(C++)+ Python oracle 脚本 | ✅(验数学正确) |
| 内部一致性测试 | `ncnn/tests/test_mylayer.cpp` + `tests/CMakeLists.txt` | 建议(加优化后) |
| arch 优化 | `src/layer/{x86,arm}/mylayer_*.cpp` | 可选(性能) |
| 计算图转换 | `tools/pnnx/src/pass_ncnn/...`(+ level1/2) | 后续(让模型能转到此 layer) |

---

## 7. 结论

> **从零验证一个 ncnn layer 是否正确,靠的是"PyTorch 作 oracle 的 layer 级数值对比"**(给定输入 → ncnn forward 输出 vs PyTorch 输出 allclose),**不需要 baseline、不需要计算图转换**——只需 layer 能被创建并 forward。
> ncnn 自带的 `test_layer` 只验"优化路径 == 朴素实现"的内部一致性,**不验数学正确性**,因此不能替代外部 oracle。
> 完整从零顺序:实现 layer → 外部 oracle 验数学 →(可选)优化+内部一致性 → 写 pnnx 转换 → 端到端验证。
