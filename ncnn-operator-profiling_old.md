# ncnn 算子硬件指标测试流程

适用场景:优化或新增 ncnn 移动端算子(CPU 后端 / Vulkan 后端)时,采集硬件相关指标以定位性能瓶颈。文档以**命令行工具优先**。

---

## 0. simpleperf 对 CPU 后端是否足够?

simpleperf 是 CPU 微架构分析的核心,覆盖 PMU 硬件计数器与采样/火焰图,但**不足以单独完成** CPU 算子分析,需补充:

- **频率 / 温度**:simpleperf 不负责确认 DVFS 与热降频,需 sysfs / perfetto。降频时段的数据不可比。
- **线程调度 / 大小核**:ncnn CPU 后端用 OpenMP 多线程,线程落大核还是小核、是否被抢占、负载是否均衡,需 perfetto/atrace 时间线。
- **DDR 带宽**:memory-bound 算子(elementwise、depthwise conv、pooling)的真实瓶颈;simpleperf 的 PMU 只能用 cache miss 间接反映,总线带宽需 perfetto memory counter 或厂商工具。

**结论**:simpleperf 提供"指令 / cache / 分支级"核心数据,但完整判定瓶颈仍需 `simpleperf + perfetto + sysfs` 组合。

---

## 1. 通用前置(两种后端都适用)

测试纪律决定数据是否可比:

- 锁频或固定 governor(`performance`),记录起始频率
- 监控温度,排除 thermal throttling 时段的数据
- warmup 若干次后再采样,丢弃首次(含初始化 / pipeline 编译)
- 绑核:`taskset` 固定到目标核(大核或小核),避免迁移
- 多次测量取中位数,记录方差
- 单算子隔离:用最小网络或 `benchncnn` 单层 benchmark,避免被其他层污染

基础工具:
- `benchncnn` —— 端到端 / layer-wise 基线
- sysfs 采样脚本 —— 频率 / 温度
- `perfetto` —— 系统时间线

### sysfs 采样脚本(频率 / 温度,后台运行)

```bash
#!/system/bin/sh
# sample_sys.sh — 跑算子时后台采样,事后对齐时间戳
while true; do
  ts=$(date +%s.%N)
  cpufreq=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq 2>/dev/null)
  temp=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
  # Adreno: /sys/class/kgsl/kgsl-3d0/gpuclk 与 /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage
  # Mali:   /sys/class/devfreq/*.mali/cur_freq 与 load
  echo "$ts cpu7=$cpufreq temp=$temp"
  sleep 0.1
done
```

---

## 2. CPU 后端算子测试流程

### 2.1 端到端基线
```bash
# 固定线程数与绑核,powersave 关闭
./benchncnn 20 4 0 -1 1   # loop=20 threads=4 powersave=0 gpu=-1(CPU) cooldown=1
```
记录目标算子所在网络的 layer-wise 耗时,确定该算子占比。

### 2.2 PMU 硬件计数器(simpleperf stat)
```bash
# 关键计数器:周期/指令/IPC、cache、分支、TLB
simpleperf stat -e cpu-cycles,instructions,\
cache-references,cache-misses,\
branch-instructions,branch-misses,\
raw-l1d-cache,raw-l1d-cache-refill \
--app <pkg> --duration 10
```
关注:
- **IPC**(instructions / cpu-cycles):低 IPC → stall,看后续 cache / 内存
- **cache-misses 率**:高 → memory-bound,考虑 blocking / 数据排布
- **branch-misses**:高 → 分支预测差,检查内层循环分支

### 2.3 采样火焰图(定位热点函数 / 指令)
```bash
simpleperf record -e cpu-cycles --app <pkg> -g --duration 10 -o perf.data
simpleperf report -g --sort symbol
# 或导出火焰图 / 反汇编级热点
simpleperf report --symbols <op_func> -i perf.data
```
确认热点是否落在你的算子 kernel,以及是否用上 SIMD(NEON / fp16)。

### 2.4 线程与调度(perfetto)
```bash
perfetto -c config.pbtxt -o trace.pftrace   # data source: sched, cpufreq
```
检查:OpenMP 线程是否均衡、是否落在大核、是否被抢占 / 迁移。

### 2.5 内存带宽
- 优先 perfetto memory counter / SoC 总线 counter
- 否则用 simpleperf 的 last-level cache miss × cache line 估算带宽下界

### 2.6 CPU 侧判定逻辑
1. IPC 高且热点在 kernel → compute-bound,优化 SIMD / 指令级
2. IPC 低 + cache miss 高 → memory-bound,优化数据排布 / blocking / pack
3. 热点不在 kernel(在 reorder / pack / 同步) → 优化访存预处理或线程切分
4. 全程先用 sysfs 确认无降频,否则数据作废

---

## 3. Vulkan 后端算子测试流程

GPU 算子是 CPU+GPU 协同,需同时看两侧。

### 3.1 端到端基线
```bash
./benchncnn 20 1 0 0 1     # gpu=0 走 Vulkan
```

### 3.2 GPU 逐 dispatch 计时(主指标,自埋点)
在 ncnn `VkCompute` 的每个 `vkCmdDispatch` 前后插入 `vkCmdWriteTimestamp`,
读回后乘 `VkPhysicalDeviceLimits::timestampPeriod` 得每算子 GPU 执行时间。
纯文本输出、可脚本化、可进 CI,是优化前后对比的核心数据源。

### 3.3 判定瓶颈在 CPU 提交还是 GPU 执行
- 对比 **GPU dispatch 总时间** vs **端到端时间**
- 差值大 → CPU-bound(命令录制 / submit / 同步 / CPU 回退算子)
  - 用 simpleperf 看录制线程热点,perfetto 看 submit 间隔与 CPU/GPU 重叠
- 差值小、GPU 时间占主 → GPU-bound,进 3.4

### 3.4 GPU 硬件计数器(带宽 / ALU / occupancy / cache)
跨厂商纯 CLI 无银弹:
- **Mali**:`gatord`(Arm Streamline 采集端,设备侧 CLI),拿带宽 / ALU 利用率 / cache
- **Adreno**:Snapdragon Profiler(GUI 为主)抓一次定性;逐 dispatch 计时仍靠 timestamp query
- **跨厂商 trace**:`agi`(Android GPU Inspector)CLI 采集,离线分析

关注:内存带宽利用率(低算术强度算子常 memory-bound)、occupancy、寄存器压力 / spilling、workgroup size。

### 3.5 系统侧(CLI 充分)
- `perfetto`:GPU 频率(gpufreq)、调度、内存 counter
- sysfs:GPU 频率 / busy% / 温度(见 §1 脚本)
- 确认无热降频与 DDR 带宽争用

### 3.6 数据上传 / 下载开销
检查 host↔device 的 packing 转换(pack1/4/8、fp32↔fp16)与 CPU 回退算子带来的 download/upload;移动端 UMA 省拷贝但转换仍在 CPU。

---

## 4. 工具速查

| 维度 | CPU 后端 | Vulkan 后端 | CLI |
|------|----------|-------------|-----|
| 端到端 / layer-wise | benchncnn | benchncnn | ✅ |
| 算子执行时间 | simpleperf record | timestamp query 埋点 | ✅ |
| 硬件计数器 | simpleperf stat | gatord(Mali)/ SDP(Adreno) | 部分 |
| 热点函数 | simpleperf report | simpleperf(录制线程) | ✅ |
| 线程 / 调度 | perfetto | perfetto | ✅ |
| 频率 / 温度 | sysfs | sysfs | ✅ |
| 内存带宽 | perfetto / cache 估算 | gatord / perfetto | 部分 |

---

## 6. 并发测量约束(同一趟能否多工具齐开)

硬件 PMU 计数器有限(ARM 核通常 ~6 个可编程 + 1 个 cycle),关键看工具是否抢这套寄存器。

**可并行(零冲突)**
- sysfs 采样脚本:只读文件,不碰 PMU,开销可忽略,**每一趟都应开着**。
- perfetto 的 ftrace 数据源(sched_switch / cpu_frequency / gpufreq):走 ftrace,与 PMU 不冲突。

**互斥(抢同一组 PMU,不可同趟)**
- simpleperf stat 与 perfetto 的 `linux.perf` counter:都用 `perf_event_open` 抢硬件 PMU。perfetto 配置**不要开 PMU counter**,把计数器留给 simpleperf。
- simpleperf stat 与 simpleperf record:一趟只跑一个。
- 单条 stat 列超过 ~6 个 event 会触发内核 multiplexing(时间复用 + 外推),精度下降 → **分批跑**。

**观察者效应**
- record 采样 / 多 profiler 叠加都会扰动那 100 次循环的耗时 → headline latency 从"干净趟"取,不要从被 profiling 的趟取。

**结论:多趟分离,每趟只隔离一个关注点;sysfs 全程常开。**

---

<!-- APPEND2 -->

## 7. 实例:LogSoftmax @ 小米13 CPU 后端

设备:小米 13(Snapdragon 8 Gen 2,Adreno 740,1×X3 大核 + 2×A715 + 2×A710 + 3×A510)。
命令:`./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32]`
解析:loop=100,threads=4,powersave=2(只用大核簇),gpu=-1(CPU),cooldown=1。

> 注意 benchncnn 是原生二进制非 app,simpleperf 用 `-p <pid>` 附加或直接拉起进程,**不要用 `--app`**。
> shape=[512,32] 数据量小,LogSoftmax 是 reduce(max/sum-exp)+ elementwise,典型 **memory-bound + exp/log 计算**,需重点验证多线程是否真有收益(规模小,4 线程可能因 fork/同步开销反而变慢)。

**趟 1 — 干净基线(取 headline latency)**
```bash
adb shell ./sample_sys.sh > sys_cpu.log &   # sysfs 常开
adb shell ./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32]
# 记录中位 latency;对照 sys_cpu.log 确认全程无降频
```

**趟 2 — PMU 计数器(分批避免 multiplexing)**
```bash
# batch A: 指令级 / IPC
simpleperf stat -e cpu-cycles,instructions,branch-instructions,branch-misses \
  -p $(pidof benchncnn) --duration 15
# batch B: 访存
simpleperf stat -e cache-references,cache-misses,raw-l1d-cache,raw-l1d-cache-refill \
  -p $(pidof benchncnn) --duration 15
```
看 IPC(低→stall)、cache-miss 率(高→memory-bound,符合 LogSoftmax 预期)、branch-miss(reduce 循环边界)。

**趟 3 — 采样火焰图**
```bash
simpleperf record -e cpu-cycles -g -p $(pidof benchncnn) --duration 15 -o perf.data
simpleperf report -g --sort symbol
```
确认热点是否在 LogSoftmax kernel 的 exp/log;检查是否走了 NEON/fp16 向量化路径,还是标量 expf。

**趟 4 — 线程调度(perfetto,仅 ftrace,不开 PMU)**
```bash
perfetto -c cpu_sched.pbtxt -o trace.pftrace   # data source: sched, cpufreq
```
看 4 个 OpenMP 线程是否都落大核、负载是否均衡、是否被抢占;对比 threads=1 验证多线程净收益。

**判定**:IPC 低 + cache-miss 高 → 优化数据排布/blocking;热点在 expf 标量 → 上向量化 exp;多线程无收益 → 该规模降线程数。

---

## 8. 实例:LogSoftmax @ 小米13 Vulkan 后端

命令:`./benchncnn 100 1 0 0 1 param=model.param shape=[512,32]`
解析:loop=100,threads=1,powersave=0,gpu=0(Vulkan / Adreno 740),cooldown=1。

> shape=[512,32] 很小,GPU 计算量极低,**大概率不是 GPU 执行 bound,而是命令录制/提交/同步 + host↔device 传输 bound**。验证这一点是分析重心。

**趟 1 — 干净基线**
```bash
adb shell ./sample_sys.sh > sys_gpu.log &   # 含 Adreno: /sys/class/kgsl/kgsl-3d0/gpuclk、gpu_busy_percentage
adb shell ./benchncnn 100 1 0 0 1 param=model.param shape=[512,32]
```

**趟 2 — GPU 逐 dispatch 计时(主指标,自埋点)**
在 ncnn `VkCompute` 的 LogSoftmax dispatch 前后插 `vkCmdWriteTimestamp`,
读回 × `timestampPeriod` → 该算子 GPU 纯执行时间(注意 LogSoftmax 可能拆成 reduce_max / exp_sum / div 多个 dispatch,逐个计时)。

**趟 3 — 判定 CPU提交 bound vs GPU执行 bound(关键)**
```bash
# 端到端 latency(趟1) 对比 GPU dispatch 总时间(趟2)
simpleperf record -e cpu-cycles -g -p $(pidof benchncnn) --duration 15 -o perf_gpu.data
simpleperf report -g --sort symbol   # 看 CPU 是否耗在 vkCmd 录制/submit/同步
```
- 端到端 ≫ GPU 时间 → **CPU 提交 bound**(小 tensor 常见):热点在 vkCmdDispatch/descriptor 更新/queueSubmit/fence 等待 → 减少 dispatch 数、batch 化、复用 descriptor。
- 端到端 ≈ GPU 时间 → GPU bound,进趟 4。

**趟 4 — GPU 硬件计数器(按需,定性)**
Adreno 无强 CLI 计数器路径:用 **Snapdragon Profiler(GUI)** 抓一次,看带宽利用率 / ALU 占用 / occupancy;逐 dispatch 计时仍以趟 2 timestamp query 为准。

**趟 5 — 传输 / packing 开销**
检查 host→device 上传与 device→host 下载、fp32↔fp16 与 pack1/4/8 转换是否在 CPU 占大头(在趟 3 的火焰图里一并看)。

**判定**:小 shape 下若确认 CPU 提交 bound,则该算子在 GPU 上意义有限,结论可能是"该规模走 CPU 后端更优",或需算子融合减少 dispatch。

---

## 9. 一句话总结

- **CPU 后端**:`simpleperf`(PMU 分批 + 火焰图,核心)+ `perfetto`(调度)+ `sysfs`(频率/温度);LogSoftmax 重点查 memory-bound 与 exp/log 向量化,并验证 threads=4 是否真有收益。
- **Vulkan 后端**:`timestamp query`(GPU 计时,核心)+ `simpleperf`(判 CPU 提交 bound)+ `perfetto`/`sysfs`;小 tensor 多为提交 bound,GPU 计数器(Adreno 用 SDP)按需深挖。
- 多趟分离测量,PMU 不与 perfetto counter 同趟;sysfs 全程常开排除热降频,否则数据不可比。

