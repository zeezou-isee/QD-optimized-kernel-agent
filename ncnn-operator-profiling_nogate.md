# ncnn 算子 profiling 可执行脚本(LogSoftmax,不使用 gate.sh)

本文件是在 `ncnn-operator-profiling.md` 基础上派生的副本,去掉了有效性闸门 gate.sh。

前提:`benchncnn`、`model.param`/`model.bin`、`simpleperf` 已 push 到 `/data/local/tmp/`,在 `adb shell` 内执行。首次跑前 `ls` 确认:LogSoftmax 符号名是否匹配。

## 0. 放弃 gate.sh 的影响与替代

不再单独采频率/温度,失去了"这趟有没有热降频"的显式判据。替代办法:
- 命令自带 `cooldown=1`,每趟间隔散热;
- warmup 后取中位数,多趟复测;
- 观察 benchncnn 输出的 **min/avg/max 方差**——方差骤增通常就是降频信号,据此判断是否作废重测。

每趟仍把 benchncnn 后台拉起,以便 simpleperf 用 `-p $!` 附加(原生二进制不能用 `--app`)。

## 0.5 本机约束:perfetto 系统级 trace 不可用(小米13 + 无 root,实测)

设备:小米13(Snapdragon 8 Gen 2),adb shell 普通用户(uid=2000 shell,`su` not found,无 root)。实测 perfetto 走不通,**全文不依赖 perfetto / 任何系统级 trace 服务**:

- `traced`/`traced_probes` 默认未启用;`setprop persist.traced.enable 1` 能被 shell 接受并按需拉起服务,ftrace 也能 setup。
- 但 **SELinux 策略**禁止 perfetto 域 `u:r:perfetto:s0` 读写 shell 数据目录 `shell_data_file`:logcat 报 `avc: denied { search/read/write } /data/local/tmp/*`。perfetto 内部仍要在该目录建临时文件中转,`-c -`/`-o -` 也绕不开;`/data/misc/perfetto-traces/` 不存在且 shell 无权 mkdir(errno 2)。非 root 改不了策略。
- **误导坑**:Android perfetto 的错误只进 logcat,不进 stderr。`2> perfetto.log` 永远是空,排查必须 `logcat -d -s perfetto`。

**因此采集一律用 `simpleperf`**(直接 `perf_event_open`,零服务、零受限目录,stat/record/report/`--per-thread` 均实测可用)。需要"线程调度时间线"这类只有 system trace 能给的视图时,在本机不可得 → 用 simpleperf `--per-thread` 的计数近似(看负载均衡),或换 root 设备。

## 1. CPU 后端 `./benchncnn 100 4 2 -1 1`
```bash
cd /data/local/tmp

# ---- 趟1:干净基线(取 headline latency,不挂 profiler)----
./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32]
# 记录中位 latency;若 min/avg/max 方差异常大,散热后重测

# ---- 趟2:PMU 计数器(6 event,一次跑完不触发 multiplexing)----
# cpu-cycles 走专用 cycle 计数器,不占可编程的 6 个
# ./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32] & BENCH=$!
# ./simpleperf stat \
#   -e cpu-cycles,instructions,cache-references,cache-misses,branch-misses \
#   -p $BENCH --duration 15
# wait $BENCH

cd /data/local/tmp/ncnn && \
taskset f0 simpleperf stat \
-e cpu-cycles,instructions,cache-references,cache-misses,branch-misses \
./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32]

# IPC=instructions/cpu-cycles;cache-miss率=cache-misses/cache-references(memory-bound 判据);
# branch-miss率=branch-misses/branch-instructions(去分支判据)
# 如已确认 memory-bound、需细分 miss 到 L2 还是 DRAM,再补一趟 raw-l1d-cache,raw-l1d-cache-refill

# # ---- 趟3:采样火焰图 + 反汇编(定位到 LogSoftmax 具体循环)----
# simpleperf record -e cpu-cycles -g -o perf_cpu.data \
#   ./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32]
# simpleperf report -g --sort symbol -i perf_cpu.data
# # 反汇编级:确认 exp 是标量 expf 还是向量化(填入实际符号名)
# simpleperf report --symbols "ncnn::LogSoftmax::forward_inplace" -i perf_cpu.data

# ---- 趟4:线程负载均衡(simpleperf --per-thread,子进程形式,不依赖 perfetto)----
# 为什么不用 perfetto:见文档顶部「本机约束」。改用 simpleperf:直接 perf_event_open,
#   零服务、零受限目录。注意子进程形式默认进程聚合,必须带 --per-thread 才拆线程。
cd /data/local/tmp/ncnn && \
taskset f0 simpleperf stat --per-thread \
-e cpu-cycles,instructions \
./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32]
# 读法:按 tid 看各 OpenMP 工作线程的 cpu-cycles——数值接近=负载均衡;差异大=切分维度/粒度有问题。
# 小负载提醒(实测):shape 很小时工作线程生命周期极短(avg~0.2ms),stat 常只采到顶层进程,
#   这本身就说明"该规模多线程收益有限",与下面 threads=1 对照互相印证。

# 对照:单独跑 threads=1,判断多线程是否真有净收益
cd /data/local/tmp/ncnn && \
taskset f0 ./benchncnn 100 1 2 -1 1 param=model.param shape=[512,32]
```

## 2. Vulkan 后端 `./benchncnn 100 1 0 0 1`

> 受本机约束(§0.5)影响:Vulkan 段同样不依赖 perfetto;CPU 侧分析统一用 simpleperf 子进程形式。
> GPU 侧硬件计数器(趟4)在无 root 量产机上多半也不可得,见该趟降级说明。

```bash
cd /data/local/tmp/ncnn_vk

# ---- 趟1:干净基线 ----
./benchncnn 100 1 0 0 1 param=model.param shape=[512,32]   # 记录端到端 latency

# ---- 趟2:GPU 逐 dispatch 计时(需埋点构建)----
# 前提:用插了 vkCmdWriteTimestamp 的 benchncnn 构建(此处记为 benchncnn_ts)。
# 从其日志读各 dispatch(reduce_max / exp_sum / div)GPU ns,累加得 GPU 执行总时间。
# 说明:timestamp query 是 Vulkan API 内的计时,纯应用层、不碰 SELinux/服务,
#   是本机唯一能拿到的 GPU 侧定量数据,务必作为主指标。
./benchncnn_ts 100 1 0 0 1 param=model.param shape=[512,32]

# ---- 趟3:判定 CPU提交 bound vs GPU执行 bound(关键,simpleperf 子进程形式)----
cd /data/local/tmp/ncnn && \
taskset f0 simpleperf record -e cpu-cycles -g -o perf_gpu.data \
  ./benchncnn 100 1 0 0 1 param=model.param shape=[512,32]
simpleperf report -g --sort symbol -i perf_gpu.data
# 判定:
#  端到端(趟1) ≫ GPU总时间(趟2) → 提交 bound:热点应在
#       vkCmdDispatch / vkUpdateDescriptorSets / vkQueueSubmit / fence 等待
#  端到端 ≈ GPU总时间              → GPU 执行 bound,进趟4
# 同一火焰图里一并看 fp32↔fp16 / pack1-4-8 转换、CPU 回退算子的 upload/download 占比
# 本机现实:shape 很小(avg~0.18ms),几乎必为提交 bound——这正是 simpleperf 火焰图能确证、
#   且不依赖任何 GPU 计数器就能得出的结论,对小算子已足够指导优化(融合 dispatch/复用 descriptor)。

# ---- 趟4(仅当 GPU 执行 bound 时):GPU shader 计数器 ----
# 本机约束:Adreno 无 CLI 计数器路径;Snapdragon Profiler 走 GUI,且其 GPU Metrics
#   依赖设备侧采集服务,在无 root 量产机上常被 SELinux/权限挡(同 perfetto 处境)。
# 降级路径(无 root 时):
#   a) 主指标退回趟2 的 timestamp query 逐 dispatch 时间,据此推断 GPU 侧热点;
#   b) 通过改 local workgroup size / 数据 packing 做 A/B,用趟1 端到端 latency 对比验证,
#      以"实验对比"替代"计数器直读";
#   c) 需要硬件计数器(带宽/occupancy/寄存器)时,换一台 root/userdebug 设备用 SDP。
```
取回分析:`adb pull /data/local/tmp/ncnn/perf_gpu.data ./`

## 3. 执行顺序小结
- **CPU**:趟1 基线 → 趟2 PMU(6 event 一次跑完)→ 趟3 火焰图+反汇编 → 趟4 simpleperf `--per-thread` 负载均衡(+threads=1 对照)
- **Vulkan**:趟1 基线 → 趟2 timestamp 计时(需埋点,本机唯一 GPU 定量数据)→ 趟3 simpleperf 判提交/执行 bound → 趟4 GPU 计数器(本机不可得,走降级:timestamp + A/B 对比,或换 root 设备 SDP)
- 通用:headline latency 只从趟1 干净趟取;simpleperf 用子进程形式(`taskset f0 simpleperf ... ./benchncnn`)或 `-p $!` 附加,勿用 `--app`;放弃 gate.sh 后靠 cooldown + 方差观察代替降频闸门。
- **本机约束(§0.5)**:perfetto / 系统级 trace / SDP GPU 计数器在小米13 无 root 下均被 SELinux 挡,全流程一律用 simpleperf + timestamp query。
