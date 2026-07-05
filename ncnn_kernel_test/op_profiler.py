"""算子 profiling 工具: 单算子 + 批量, 单文件供项目代码 import 调用。

对外主要接口:
    profile_operator(op, param, shape, ...) -> dict        单算子, 返回算子级指标 dict
    profile_many(manifest, ...) -> list[dict]              批量遍历 manifest
    gen_manifest(dataset_root) -> list[dict]               从数据集枚举算子清单骨架

它在 Android 设备上用 simpleperf record 采集指定 ncnn 算子, 自动发现热点符号,
返回算子级 PMU 派生指标(IPC / cache-miss率 / branch-miss率)及可信度标注。
返回值是普通 dict(可直接 json.dumps 或塞进 pipeline 的 profile 字段)。

依赖: 主机装有 adb 且设备已连接; 设备 /data/local/tmp/ncnn 下有 benchncnn、
      simpleperf, 以及对应算子的 .param 文件(由调用方预先 push)。

实测定下的默认配置(见 ncnn-operator-profiling_nogate.md §0.6 / §1.5):
  - 事件全加 :u(user 版机型 SELinux 拒 {kernel} 权限)
  - taskset c0(同档超大核, 避免异构核混频)
  - cooldown=1 + loop=10000(散热保护 + 干净信号, 二者正交)
  - threads=1 出最干净算子画像
  - benchncnn 结果在 stderr → 命令一律 2>&1
"""

from __future__ import annotations

import json
import re
import subprocess

# 6 个 PMU 事件(全部 :u, 只统计用户态)。
_EVENTS = (
    "cpu-cycles:u", "instructions:u", "cache-references:u",
    "cache-misses:u", "raw-br-retired:u", "raw-br-mis-pred-retired:u",
)

# 框架/启动/fp16辅助 符号 denylist: 剔除后 cpu-cycles 段的 top 即算子热点。
_FRAMEWORK_RE = re.compile(
    r"\[linker\]|kmp_|__kmp|GOMP|omp_|scudo|__mem|__str|ld-android|"
    r"libc|libm|pthread|malloc|free\b|operator (new|delete)|@plt|__emutls|"
    r"ncnn::Net|load_model|load_param|ParamDict|Extractor|create_layer|"
    r"create_pipeline|ncnn::Mat|do_forward_layer|get_physical|cpu_info|"
    r"cpu_count|try_initialize|Layer_final|cast_fp(16|32)|Packing_arm|"
    r"xml::|sha256|std::__ndk1::vector"
)


def _adb(cmd: str, timeout: int) -> tuple[str, int]:
    """在设备上跑 shell 命令, 返回 (合并输出, 返回码)。"""
    r = subprocess.run(["adb", "shell", cmd], capture_output=True,
                       text=True, timeout=timeout)
    return (r.stdout + r.stderr, r.returncode)


def _parse_report(txt: str) -> dict:
    """把 simpleperf report 文本解析成 {event: {total, samples, rows:[(pct, sym)]}}。"""
    ev: dict = {}
    cur = None
    for ln in txt.splitlines():
        if m := re.match(r"Event: (\S+)", ln):
            cur = m.group(1)
            ev[cur] = {"total": 0, "samples": 0, "rows": []}
        elif cur and (m := re.match(r"Event count: (\d+)", ln)):
            ev[cur]["total"] = int(m.group(1))
        elif cur and (m := re.match(r"Samples: (\d+)", ln)):
            ev[cur]["samples"] = int(m.group(1))
        elif cur and (m := re.match(r"\s*([\d.]+)%\s+(.+)", ln)):
            ev[cur]["rows"].append((float(m.group(1)), m.group(2).strip()))
    return ev


def _discover_symbol(ev: dict, denylist_extra: tuple) -> tuple[str, float]:
    """剔除框架符号后取 cpu-cycles 段 top; 返回 (symbol, 断崖比 top/次名)。"""
    extra = re.compile("|".join(denylist_extra)) if denylist_extra else None
    cand = [(p, s) for p, s in ev.get("cpu-cycles:u", {}).get("rows", [])
            if not _FRAMEWORK_RE.search(s) and not (extra and extra.search(s))]
    if not cand:
        return ("", 0.0)
    top = cand[0]
    cliff = top[0] / cand[1][0] if len(cand) > 1 and cand[1][0] > 0 else 999.0
    return (top[1], round(cliff, 2))


def _pct_of(ev: dict, event: str, symbol: str) -> float:
    """该 event 下目标符号占比(0~1), 用函数名前缀匹配以容忍签名差异。"""
    key = symbol.split("(")[0].strip()
    if not key:
        return 0.0
    for p, s in ev.get(event, {}).get("rows", []):
        if key in s:
            return p / 100.0
    return 0.0


_LAT_RE = re.compile(
    r"min\s*=\s*(?P<mn>[\d.]+)\s+max\s*=\s*(?P<mx>[\d.]+)\s+avg\s*=\s*(?P<av>[\d.]+)")


def _parse_latency(txt: str) -> dict:
    """Extract benchncnn's 'min= max= avg=' line (ms). The record command already
    runs benchncnn, so its latency is in the same output — no extra run needed."""
    m = _LAT_RE.search(txt)
    if not m:
        return {"latency_avg": None, "latency_min": None, "latency_max": None}
    return {"latency_avg": float(m.group("av")), "latency_min": float(m.group("mn")),
            "latency_max": float(m.group("mx"))}


def profile_operator(
    op: str,
    param: str,
    shape: str,
    *,
    threads: int = 1,
    loop: int = 10000,
    cooldown: int = 1,
    taskset_mask: str = "c0",
    sample_freq: int = 50000,
    device_dir: str = "/data/local/tmp/ncnn",
    symbol_hint: str | None = None,
    denylist_extra: tuple = (),
    min_samples: int = 100,
    trust_fraction: float = 0.15,
    timeout: int = 180,
    simpleperf_cmd: str = "simpleperf",
) -> dict:
    """采集单个 ncnn 算子的算子级 PMU profile, 返回 dict。

    参数
    ----
    op           : 算子名(仅用于回填到结果, 不影响采集)。
    param        : 设备 device_dir 下的 .param 文件名(需调用方预先 push)。
    shape        : 输入 shape 字符串, 如 "[512,32]"。
    threads      : benchncnn 线程数(1=最干净算子画像; 2=双超大核并行 baseline)。
    loop         : benchncnn 迭代次数(信号杠杆, 越大算子占比越高; 与 cooldown 正交)。
    cooldown     : benchncnn 趟间散热 sleep 开关(1=保留散热保护)。
    taskset_mask : CPU 亲和性掩码(c0=同档超大核; 异构 PMU 机型先查布局再定)。
    sample_freq  : simpleperf -f 采样率(不加 loop 也能提样本密度)。
    symbol_hint  : 指定热点符号则跳过自动发现; None=每次自动发现(不缓存)。
    denylist_extra: 该算子额外要排除的符号正则(自动发现误判时用)。
    min_samples  : 事件样本数低于此值则该指标标记不可信。
    trust_fraction: 算子占比低于此值则整体 trustworthy=False。
    simpleperf_cmd: 设备上调用 simpleperf 的方式(默认 "simpleperf" 走 PATH;
                    若 simpleperf 是 push 到 device_dir 的副本则传 "./simpleperf")。

    返回
    ----
    dict, 字段见下。采集/解析失败时返回 {"op", "error", ...} 且其余字段为 None。
    """
    # ---- 1. record(设备上采集)----
    rec_cmd = (
        f"cd {device_dir} && KMP_BLOCKTIME=0 OMP_WAIT_POLICY=passive "
        f"taskset {taskset_mask} {simpleperf_cmd} record -e {','.join(_EVENTS)} "
        f"-f {sample_freq} -o /data/local/tmp/perf_op.data "
        f"./benchncnn {loop} {threads} 2 -1 {cooldown} "
        f"param={param} shape='{shape}' 2>&1"
    )
    base = {"op": op, "shape": shape, "threads": threads}
    try:
        out, rc = _adb(rec_cmd, timeout)
    except subprocess.TimeoutExpired:
        return {**base, "error": "record 超时(设备无响应或 loop 过大)"}
    if "Samples recorded" not in out:
        return {**base, "error": f"record 失败: {out.strip()[-200:]}"}
    # benchncnn ran as part of the record command above; its latency line is in
    # `out` — parse it here so we don't run benchncnn a second time.
    lat = _parse_latency(out)

    # ---- 2. report + 解析 ----
    rep_cmd = (f"cd {device_dir} && {simpleperf_cmd} report -i /data/local/tmp/perf_op.data "
               f"--sort symbol --percent-limit 0 2>&1")
    try:
        rep, _ = _adb(rep_cmd, timeout)
    except subprocess.TimeoutExpired:
        return {**base, **lat, "error": "report 超时"}
    ev = _parse_report(rep)
    if "cpu-cycles:u" not in ev:
        return {**base, **lat, "error": "report 无 cpu-cycles 事件(符号化失败?)"}

    # ---- 3. 符号发现 + 指标计算 ----
    if symbol_hint:
        sym, cliff = symbol_hint, None
    else:
        sym, cliff = _discover_symbol(ev, denylist_extra)

    def cnt(e):   return _pct_of(ev, e, sym) * ev.get(e, {}).get("total", 0)
    def nsamp(e): return _pct_of(ev, e, sym) * ev.get(e, {}).get("samples", 0)

    cyc, ins = cnt("cpu-cycles:u"), cnt("instructions:u")
    cref, cmiss = cnt("cache-references:u"), cnt("cache-misses:u")
    br, brm = cnt("raw-br-retired:u"), cnt("raw-br-mis-pred-retired:u")
    frac = _pct_of(ev, "cpu-cycles:u", sym)
    low = [e for e in _EVENTS if nsamp(e) < min_samples]

    note = ""
    if not sym:
        note = "未发现算子符号(可能全是框架开销, 或符号被 strip)"
    elif cliff is not None and cliff < 3:
        note = f"自动发现断崖比={cliff} 偏低, 建议人工确认符号或加大 loop"
    elif frac < trust_fraction:
        note = f"算子占比仅 {frac:.1%}, 信噪比低, 指标供方向参考"
    elif low:
        note = (f"低频事件样本不足({','.join(e.replace(':u','') for e in low)}), "
                f"其派生指标供参考")

    return {
        **base,
        "operator_symbol": sym,
        "discovery_cliff": cliff if cliff is not None else -1.0,
        "operator_fraction": round(frac, 4),
        "ipc": round(ins / cyc, 3) if cyc else None,
        "cache_miss_rate": round(cmiss / cref, 5) if cref else None,
        "branch_miss_rate": round(brm / br, 5) if br else None,
        **lat,
        "low_sample_events": low,
        "trustworthy": bool(sym) and frac >= trust_fraction and not low,
        "note": note,
        "error": None,
    }


def profile_many(
    manifest: list[dict],
    *,
    thread_configs: tuple = (1, 2),
    push_dir: str | None = None,
    device_dir: str = "/data/local/tmp/ncnn",
    on_result=None,
    **profile_kwargs,
) -> list[dict]:
    """批量遍历 manifest, 每算子跑所有 thread 配置, 返回结果列表。

    参数
    ----
    manifest      : 算子清单, 每条目至少含 {op, param, shape};
                    可选 {symbol_hint, denylist_extra, loop}。
    thread_configs: 每算子要采的线程配置(默认 1=画像 + 2=并行 baseline)。
    push_dir      : 若给出, 逐算子把 push_dir/<param> adb push 到 device_dir;
                    None 则假设 param 已在设备上。
    on_result     : 可选回调 on_result(index, total, result), 用于打印进度。
    profile_kwargs: 透传给 profile_operator 的其余参数(taskset_mask 等)。

    返回
    ----
    list[dict], 每元素 {op, category, configs:[profile, ...]}。
    """
    results = []
    total = len(manifest)
    for i, entry in enumerate(manifest):
        op = entry["op"]
        param = entry["param"]
        shape = entry.get("shape", "[512,32]")

        if push_dir is not None:
            r = subprocess.run(
                ["adb", "push", f"{push_dir}/{param}", f"{device_dir}/{param}"],
                capture_output=True, text=True)
            if r.returncode != 0:
                res = {"op": op, "error": f"param push 失败: {param}"}
                results.append(res)
                if on_result:
                    on_result(i, total, res)
                continue

        configs = []
        for t in thread_configs:
            configs.append(profile_operator(
                op, param, shape, threads=t,
                loop=entry.get("loop", 10000),
                symbol_hint=entry.get("symbol_hint"),
                denylist_extra=tuple(entry.get("denylist_extra", [])),
                device_dir=device_dir,
                **profile_kwargs,
            ))
        res = {"op": op, "category": entry.get("category"), "configs": configs}
        results.append(res)
        if on_result:
            on_result(i, total, res)
    return results


def gen_manifest(dataset_root: str, default_shape: str = "[512,32]") -> list[dict]:
    """从 dataset_root 下的 **/*.py 生成算子清单骨架(param 约定为同名 .ncnn.param)。"""
    from pathlib import Path
    rows = []
    for py in sorted(Path(dataset_root).rglob("*.py")):
        rows.append({
            "op": py.stem,
            "category": py.parent.name,
            "param": f"{py.stem}.ncnn.param",
            "shape": default_shape,
            "symbol_hint": None,
            "denylist_extra": [],
        })
    return rows


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="算子 profiling(单个自测 / 批量)")
    sub = ap.add_subparsers(dest="mode")

    one = sub.add_parser("one", help="单算子自测")
    one.add_argument("--op", default="Softmax")
    one.add_argument("--param", default="model.param")
    one.add_argument("--shape", default="[512,32]")
    one.add_argument("--threads", type=int, default=1)
    one.add_argument("--loop", type=int, default=10000)

    gm = sub.add_parser("gen-manifest", help="生成 manifest 骨架到 stdout")
    gm.add_argument("--dataset-root", required=True)

    many = sub.add_parser("many", help="批量采集")
    many.add_argument("--manifest", required=True)
    many.add_argument("--out", default="op_profiles.json")
    many.add_argument("--push-dir", default=None)
    many.add_argument("--limit", type=int, default=0)

    a = ap.parse_args()
    if a.mode == "one":
        print(json.dumps(
            profile_operator(a.op, a.param, a.shape, threads=a.threads, loop=a.loop),
            ensure_ascii=False, indent=2))
    elif a.mode == "gen-manifest":
        print(json.dumps(gen_manifest(a.dataset_root), ensure_ascii=False, indent=2))
    elif a.mode == "many":
        entries = json.loads(Path(a.manifest).read_text())
        if a.limit:
            entries = entries[:a.limit]
        def _prog(i, n, res):
            c0 = (res.get("configs") or [{}])[0]
            print(f"[{i+1}/{n}] {res['op']}: sym={c0.get('operator_symbol', res.get('error'))} "
                  f"ipc={c0.get('ipc')} frac={c0.get('operator_fraction')} "
                  f"trust={c0.get('trustworthy')}", flush=True)
        out = profile_many(entries, push_dir=a.push_dir, on_result=_prog)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\n写出 {len(out)} 个算子 → {a.out}")
    else:
        ap.print_help()


