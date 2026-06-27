# ncnn 算子 profiling 流程(面向 LLM 代码优化)

适用场景:为 ncnn 移动端算子(CPU 后端 / Vulkan 后端)采集硬件 profiling 数据,**喂给 LLM 用于判断性能瓶颈并提出更优的算子代码方案**。文档以命令行工具优先。

---

## 0. 目标与取舍原则(本次重构的核心)

数据是给 LLM 做**代码层面**优化用的,不是给人做系统调优用的。因此每个指标只保留两类,其余不测:

- **A 类 — 代码可行动信号**:能直接映射到一处算子代码改动(向量化、数据排布、dispatch 融合、workgroup 调整等)。这些是喂给 LLM 的主体。
- **B 类 — 数据有效性闸门**:本身不指导代码,只用来确认这次测量可信(没降频、没被抢占污染)。最小化测量,**不作为优化信号喂给 LLM**。

明确取舍:

- **降级为 B 类(只做闸门,不喂 LLM)**:CPU/GPU 频率、温度、DVFS/热降频。只需确认"全程没降频",不再作为分析维度展开。
- **剔除(本阶段不测)**:SoC 总线 DDR 系统级带宽争用、跨组件功耗、大小核具体落核——单算子隔离基准下,这些与算子代码无关,既非 A 也非 B。
- **保留的边界例外**:线程负载是否均衡——它揭示算子**并行切分策略**(代码里如何沿哪个维度切 OpenMP)是否合理,属于 A 类;但"线程落在哪个物理核"是配置问题,归 B 类闸门即可。

> 一句话:凡是不能转成"改算子哪一行"的指标,要么降级成有效性检查,要么不测。

---

## 1. 通用前置(精简到代码相关 + 有效性)

- 单算子隔离:用最小网络 / `benchncnn` 单层,避免被其他层污染(否则 LLM 拿到的是混合信号)。
- warmup 后采样,丢弃首次(含初始化 / pipeline 编译)。
- 多次取中位数,记录方差。
- **有效性闸门(B 类,轻量)**:跑测时后台 `cat` 一次频率/温度,确认无降频即可。降频时段数据作废重测。

```bash
#!/system/bin/sh
# gate.sh — 仅作有效性闸门:确认无热降频,不作为分析维度
while true; do
  echo "$(date +%s.%N) cpu=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq 2>/dev/null) temp=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)"
  sleep 0.2
done
```

## 2. CPU 后端:采集面向代码的指标

每项后标注它**驱动哪类代码改动**,这是喂给 LLM 时要一并提供的"指标→可选优化"映射。

### 2.1 算子耗时占比(定位是否值得优化)
```bash
./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32]   # layer-wise 耗时
```
→ 确认目标算子在网络中的占比,圈定优化对象。

### 2.2 PMU 计数器(A 类核心,分批避免 multiplexing)
```bash
# batch A: 指令级 / IPC
simpleperf stat -e cpu-cycles,instructions,branch-instructions,branch-misses \
  -p $(pidof benchncnn) --duration 15
  
# batch B: 访存
simpleperf stat -e cache-references,cache-misses,raw-l1d-cache,raw-l1d-cache-refill \
  -p $(pidof benchncnn) --duration 15
```
指标 → 代码改动映射(喂 LLM 用):
- **IPC 低 + cache-miss 高** → memory-bound → 改数据排布 / blocking / pack 布局
- **IPC 高、热点在 kernel** → compute-bound → 上 NEON / fp16 向量化、减少标量运算
- **branch-misses 高** → 内层循环分支多 → 循环展开 / 去分支 / 边界预处理

### 2.3 采样火焰图 + 反汇编(A 类,定位到具体代码)
```bash
simpleperf record -e cpu-cycles -g -p $(pidof benchncnn) --duration 15 -o perf.data
simpleperf report -g --sort symbol
simpleperf report --symbols <op_func> -i perf.data   # 反汇编级:看是否真走了 SIMD
```
→ 热点落在算子的哪个函数/循环、是否标量 expf 而非向量化——直接告诉 LLM 改哪段。

### 2.4 线程切分是否合理(A 类,但只看负载均衡)
```bash
perfetto -c sched.pbtxt -o trace.pftrace   # 仅 ftrace sched,不开 PMU counter
```
→ 只关心**各线程负载是否均衡**:不均衡 → 算子的并行维度切错了(代码里 OpenMP 沿哪维切、粒度多大)。
不关心线程落在哪个物理核(那是配置/B 类)。

### 2.5 不再单独测的项(本阶段)
- 系统级 DDR 带宽争用:cache-miss 已能给出 memory-bound 的代码信号,系统带宽与算子代码无关 → 不测。
- 频率/温度:仅 §1 闸门确认无降频,不作为分析维度。

---

## 3. Vulkan 后端:采集面向代码的指标

### 3.1 GPU 逐 dispatch 计时(A 类主指标,自埋点)
在 ncnn `VkCompute` 的算子 dispatch 前后插 `vkCmdWriteTimestamp`,
读回 × `timestampPeriod` → 每个 dispatch 的 GPU 纯执行时间
(LogSoftmax 常拆成 reduce_max / exp_sum / div 多个 dispatch,逐个计时)。
→ 直接告诉 LLM:哪个 dispatch 慢、是否值得**算子融合**减少 dispatch 数。

### 3.2 判定 CPU提交 bound vs GPU执行 bound(A 类,决定优化方向)
```bash
simpleperf record -e cpu-cycles -g -p $(pidof benchncnn) --duration 15 -o perf_gpu.data
simpleperf report -g --sort symbol   # CPU 是否耗在 vkCmd 录制/submit/同步
```
对比端到端 latency 与 §3.1 GPU 总时间:
- **端到端 ≫ GPU 时间 → CPU 提交 bound**(小 tensor 常见):热点在 vkCmdDispatch / descriptor 更新 / queueSubmit / fence
  → 代码改动:**融合 dispatch、复用 descriptor set、减少 barrier**。
- **端到端 ≈ GPU 时间 → GPU 执行 bound** → 进 §3.3。

### 3.3 GPU shader 微架构(A 类,GPU bound 时才挖)
Adreno(小米13)无强 CLI 计数器路径:用 **Snapdragon Profiler** 抓一次,看
带宽利用率 / ALU 占用 / occupancy / 寄存器压力 / workgroup size。
→ 代码改动:调 local workgroup size、降寄存器压力、改 shader 访存模式 / 数据 packing。

### 3.4 host↔device 传输与 packing(A 类)
在 §3.2 火焰图里一并看 fp32↔fp16、pack1/4/8 转换、CPU 回退算子的 download/upload。
→ 代码改动:减少 layout 转换、让算子吃 packed 布局、补齐缺失的 GPU shader 实现。

### 3.5 不再单独测的项(本阶段)
GPU 频率/温度仅作降频闸门;系统功耗、跨组件带宽争用与算子 shader 代码无关 → 不测。

---

## 4. 工具速查(标注类别)

| 指标 | 工具 | 类别 | 驱动的代码改动 |
|------|------|------|----------------|
| 算子耗时占比 | benchncnn | A | 是否值得优化 |
| IPC / cache / 分支 | simpleperf stat | A | 向量化 / 数据排布 / 去分支 |
| 热点函数 + 反汇编 | simpleperf record/report | A | 改具体循环、补 SIMD |
| 线程负载均衡 | perfetto(ftrace) | A | 并行切分维度 |
| GPU dispatch 耗时 | timestamp query 埋点 | A | 算子融合 / dispatch 数 |
| CPU提交 vs GPU执行 | simpleperf + §3.1 对比 | A | 提交侧 vs shader 侧方向 |
| GPU shader 计数器 | Snapdragon Profiler | A | workgroup / 寄存器 / 访存 |
| 频率 / 温度 | sysfs `gate.sh` | B | (仅有效性闸门) |
| 系统带宽 / 功耗 | — | 剔除 | 与算子代码无关,不测 |

---

## 5. 并发测量约束

- **可并行**:`gate.sh`(只读 sysfs,常开)、perfetto ftrace(与 PMU 不冲突)。
- **互斥**:simpleperf stat 与 perfetto 的 `linux.perf` counter 抢同一组 PMU → perfetto 不开 PMU counter;stat 与 record 不同趟;单条 stat 超 ~6 event 触发 multiplexing → 分批。
- **观察者效应**:headline latency 从"仅开 gate.sh 的干净趟"取,不从被 profiling 的趟取。
- benchncnn 是原生二进制,simpleperf 用 `-p $(pidof benchncnn)` 附加,**不要用 `--app`**。

---

## 6. 实例:LogSoftmax @ 小米13

设备:小米 13(Snapdragon 8 Gen 2,Adreno 740)。shape=[512,32] 数据量小:
- CPU 侧:LogSoftmax = reduce(max / sum-exp)+ elementwise,典型 **memory-bound + exp/log 计算**;threads=4 在此规模可能因 fork/同步开销无净收益。
- Vulkan 侧:GPU 计算量极低,**大概率 CPU 提交 bound**——这是给 LLM 的首要判断。

### CPU 后端 `./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32]`
1. **趟1 干净基线**:`gate.sh` 常开 → 取中位 latency,确认无降频。
2. **趟2 PMU 分批**(§2.2):预期 cache-miss 偏高(memory-bound)、查 expf 是否标量。
3. **趟3 火焰图 + 反汇编**(§2.3):确认热点在 exp/log,是否走 NEON。
4. **趟4 perfetto 负载均衡**(§2.4):对比 threads=1,判断多线程是否该保留。
→ 交给 LLM 的结论形如:"memory-bound,exp 为标量热点,4 线程负载不均"→ LLM 提向量化 exp + 调整切分 + 该规模降线程。

### Vulkan 后端 `./benchncnn 100 1 0 0 1 param=model.param shape=[512,32]`
1. **趟1 干净基线**:`gate.sh` 常开取 latency。
2. **趟2 timestamp query**(§3.1):各 dispatch(reduce_max/exp_sum/div)GPU 时间。
3. **趟3 提交 vs 执行判定**(§3.2):端到端 ≫ GPU 时间则确认提交 bound。
4. **趟4(仅 GPU bound 时)** SDP 看 shader 计数器。
→ 交给 LLM 的结论形如:"提交 bound,3 个 dispatch 串行 + descriptor 反复绑定"→ LLM 提融合为单 shader / 复用 descriptor;若该规模始终提交 bound,结论可能是"小 shape 走 CPU 后端更优"。

---

## 8. 可执行脚本(LogSoftmax,按执行顺序)

前提:`benchncnn`、`model.param`/`model.bin`、`simpleperf` 已 push 到 `/data/local/tmp/`,在 `adb shell` 内执行。首次跑前 `ls` 确认:cpu7 是否大核、thermal_zone0 是否 CPU 温区、LogSoftmax 符号名。

### 8.0 共用:有效性闸门(每趟后台常开)
```bash
# cat > /data/local/tmp/gate.sh <<'EOF'
# #!/system/bin/sh
# # 仅判断有没有降频,不作为分析维度
# while true; do
#   echo "$(date +%s.%N) cpu7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq 2>/dev/null) temp=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)"
#   sleep 0.2
# done
# EOF
# chmod +x /data/local/tmp/gate.sh
```

<!-- APPEND8 -->

### 8.A CPU 后端 `./benchncnn 100 4 2 -1 1`
```bash
cd /data/local/tmp

# # ---- 趟1:干净基线(取 headline latency,不挂 profiler)----
# sh gate.sh > gate_cpu1.log 2>&1 & GATE=$!
# ./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32]   # 记录中位 latency
# kill $GATE
# # 查 gate_cpu1.log:cpu7 频率稳定在峰值、temp 不持续上升,否则作废重测

# ---- 趟2:PMU 计数器,分两批避免 multiplexing ----
# batch A:指令级 / IPC
sh gate.sh > gate_cpu2a.log 2>&1 & GATE=$!
./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32] & BENCH=$!
./simpleperf stat -e cpu-cycles,instructions,branch-instructions,branch-misses \
  -p $BENCH --duration 15
wait $BENCH; kill $GATE
# batch B:访存
sh gate.sh > gate_cpu2b.log 2>&1 & GATE=$!
./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32] & BENCH=$!
./simpleperf stat -e cache-references,cache-misses,raw-l1d-cache,raw-l1d-cache-refill \
  -p $BENCH --duration 15
wait $BENCH; kill $GATE

# ---- 趟3:采样火焰图 + 反汇编(定位到 LogSoftmax 具体循环)----
sh gate.sh > gate_cpu3.log 2>&1 & GATE=$!
./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32] & BENCH=$!
./simpleperf record -e cpu-cycles -g -p $BENCH --duration 15 -o perf_cpu.data
wait $BENCH; kill $GATE
./simpleperf report -g --sort symbol -i perf_cpu.data
# 反汇编级:确认 exp 是标量 expf 还是向量化(填入实际符号名)
./simpleperf report --symbols "ncnn::LogSoftmax::forward_inplace" -i perf_cpu.data

# ---- 趟4:线程负载均衡(perfetto,仅 ftrace,不开 PMU)----
cat > sched.pbtxt <<'EOF'
buffers { size_kb: 65536 }
data_sources { config {
  name: "linux.ftrace"
  ftrace_config { ftrace_events: "sched/sched_switch" ftrace_events: "sched/sched_wakeup" }
}}
duration_ms: 8000
EOF
sh gate.sh > gate_cpu4.log 2>&1 & GATE=$!
perfetto -c sched.pbtxt --txt -o trace_cpu.pftrace & PF=$!
./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32]
wait $PF; kill $GATE
# 对照:单独跑 threads=1,判断多线程是否真有净收益
./benchncnn 100 1 2 -1 1 param=model.param shape=[512,32]
```
取回分析:`adb pull /data/local/tmp/perf_cpu.data /data/local/tmp/trace_cpu.pftrace ./`

### 8.B Vulkan 后端 `./benchncnn 100 1 0 0 1`
```bash
cd /data/local/tmp

# ---- 趟1:干净基线 ----
sh gate.sh > gate_gpu1.log 2>&1 & GATE=$!
./benchncnn 100 1 0 0 1 param=model.param shape=[512,32]   # 记录端到端 latency
kill $GATE

# ---- 趟2:GPU 逐 dispatch 计时(需埋点构建,见 §3.1)----
# 前提:用插了 vkCmdWriteTimestamp 的 benchncnn 构建(此处记为 benchncnn_ts)。
# 从其日志读各 dispatch(reduce_max / exp_sum / div)GPU ns,累加得 GPU 执行总时间。
./benchncnn_ts 100 1 0 0 1 param=model.param shape=[512,32]

# ---- 趟3:判定 CPU提交 bound vs GPU执行 bound(关键)----
sh gate.sh > gate_gpu3.log 2>&1 & GATE=$!
./benchncnn 100 1 0 0 1 param=model.param shape=[512,32] & BENCH=$!
./simpleperf record -e cpu-cycles -g -p $BENCH --duration 15 -o perf_gpu.data
wait $BENCH; kill $GATE
./simpleperf report -g --sort symbol -i perf_gpu.data
# 判定:
#  端到端(趟1) ≫ GPU总时间(趟2) → 提交 bound:热点应在
#       vkCmdDispatch / vkUpdateDescriptorSets / vkQueueSubmit / fence 等待
#  端到端 ≈ GPU总时间              → GPU 执行 bound,进趟4
# 同一火焰图里一并看 fp32↔fp16 / pack1-4-8 转换、CPU 回退算子的 upload/download 占比

# ---- 趟4(仅当 GPU 执行 bound 时):GPU shader 计数器 ----
# Adreno 740 无好用 CLI 计数器路径,用 Snapdragon Profiler(GUI)抓一次:
#   带宽利用率 / ALU 占用 / occupancy / 寄存器压力 / workgroup size
# 逐 dispatch 时间仍以趟2 timestamp query 为准。
```
取回分析:`adb pull /data/local/tmp/perf_gpu.data ./`

### 8.C 执行顺序小结
- **CPU**:趟1 基线 → 趟2 PMU(两批)→ 趟3 火焰图+反汇编 → 趟4 perfetto 负载均衡(+threads=1 对照)
- **Vulkan**:趟1 基线 → 趟2 timestamp 计时(需埋点)→ 趟3 simpleperf 判提交/执行 bound → 趟4 SDP(仅 GPU bound 时)
- 通用:每趟 gate.sh 常开;PMU 不与 perfetto counter 同趟;headline latency 只从趟1 干净趟取;benchncnn 是原生二进制,simpleperf 用 `-p $!` 附加而非 `--app`。

---

## 9. 一句话总结

- 只采两类数据:**A 类(能映射到算子代码改动)** 喂 LLM;**B 类(频率/温度)** 仅作降频闸门。系统带宽/功耗与算子代码无关,本阶段不测。
- **CPU 后端**:simpleperf(PMU 分批 + 火焰图/反汇编,A 类核心)+ perfetto 负载均衡;每项都带"指标→代码改动"映射再喂 LLM。
- **Vulkan 后端**:timestamp query(A 类核心)+ simpleperf 判提交/执行 bound + SDP(GPU bound 时);核心产出是"瓶颈在提交侧还是 shader 侧"。
- 多趟分离,PMU 不与 perfetto counter 同趟;gate.sh 全程常开排除降频。
