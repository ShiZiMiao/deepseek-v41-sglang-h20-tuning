#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 @ SGLang · 可复现基线压测套件

用途: 优化前记录基线, 优化后用**同一条命令**复跑, 逐场景对比。

保证可比性的四条纪律:
  1. 每个请求在 prompt 最前面注入唯一 salt -> radix 前缀缓存必然失效, 全量重新预填充;
  2. ignore_eos=True + 固定 max_tokens -> 输出长度确定, TPOT 不受生成内容影响;
  3. 高并发场景用独立语料文件 -> 请求之间不共享前缀;
  4. 指标一律由服务端 usage 回传的 token 数计算, 不用客户端估算。

采集指标:
  客户端   TTFT / TPOT(p50,p95) / 解码 tok/s / 输入吞吐 / 输出吞吐 / 缓存命中 token
  服务端   日志采样的投机接受长度 accept len、gen throughput、prefill 输入吞吐
  运行环境 GPU 显存峰值 / 利用率均值 / server_info 配置快照

用法:
  python bench_suite.py --tag baseline           # 全量
  python bench_suite.py --tag baseline --only D1_30K,P_5x300K
  python bench_suite.py --tag baseline --list
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import math

import aiohttp

BENCH_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BENCH_DIR / "corpus"
RESULTS_DIR = BENCH_DIR / "results"
DEFAULT_LOG = BENCH_DIR.parent / "logs" / "server-latest.log"

BASE_URL = os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:30010")
MODEL = os.environ.get("BENCH_MODEL", "DeepSeek-V4.1-Flash-1017")

# ------------------------------------------------------------------ 语料
# (文件名模板, 截断字节数): None 表示用整份文件
CTX_SOURCE = {
    "1k": ("ctx-30k-1.txt", 4_000),
    "8k": ("ctx-30k-1.txt", 30_000),
    "30k": ("ctx-30k-{i}.txt", None),
    "128k": ("ctx-500k.txt", 500_000),
    "300k": ("ctx-300k-{i}.txt", None),
    "500k": ("ctx-500k.txt", None),
}

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "读取文件内容",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "写入文件",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "执行 shell 命令",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}}},
]

# ------------------------------------------------------------------ 场景表
# (id, 说明, 模式, ctx, 并发, 输出 token, 重复次数)
SCENARIOS = [
    ("D1_1K",    "单请求·1K 上下文纯解码",            "completion", "1k",   1,  256, 3),
    ("D1_30K",   "单请求·30K 上下文解码",             "completion", "30k",  1,  256, 3),
    ("D1_128K",  "单请求·128K 上下文解码",            "completion", "128k", 1,  128, 2),
    ("T8_1K",    "8 并发·1K 上下文吞吐",              "completion", "1k",   8,  256, 2),
    ("T32_1K",   "32 并发·1K 上下文吞吐上限",         "completion", "1k",  32,  192, 1),
    ("P_128K",   "单请求·128K 冷预填充",              "completion", "128k", 1,   32, 2),
    ("D1_300K",  "单请求·300K 上下文预填充与解码",     "completion", "300k", 1,  128, 2),
    ("P_5x30K",  "5 并发·30K 冷预填充（开场）",       "completion", "30k",  5,  128, 1),
    ("P_5x300K", "5 并发·300K 冷预填充（重上下文）",  "completion", "300k", 5,  128, 1),
    ("HOL",      "队头阻塞探针 1x128K + 31x1K",       "mixed",      "128k", 32, 128, 1),
    ("CACHE",    "前缀缓存复用（同一 30K 提示连发 2 次）", "completion", "30k", 1, 128, 1),
    ("CHAT_30K", "真实 agent 画像 chat+tools+thinking", "chat",     "30k",  1,  512, 2),
]

SCEN_MAP = {s[0]: s for s in SCENARIOS}


# ------------------------------------------------------------------ 工具
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_ctx(label: str, idx: int, salt: str) -> str:
    """按场景取语料, 并注入唯一 salt 使前缀缓存失效。"""
    tmpl, cut = CTX_SOURCE[label]
    name = tmpl.format(i=idx)
    text = (CORPUS_DIR / name).read_text(encoding="utf-8", errors="ignore")
    if cut:
        text = text[:cut]
    # salt 放在最前面: 整条 prompt 的公共前缀为空 -> 必然全量预填充
    return f"## run={salt} shard={idx}\n{text}"


def build_payload(mode: str, prompt: str, out_len: int) -> dict:
    if mode == "chat":
        return {
            "model": MODEL,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": "请定位 Handler 中 strict 模式的批量边界问题并给出修复代码。"}],
            "tools": TOOLS,
            "reasoning_effort": "low",
            "max_tokens": out_len,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    return {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": out_len,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def one_request(session: aiohttp.ClientSession, mode: str, prompt: str,
                      out_len: int, idx: int) -> dict:
    path = "/v1/chat/completions" if mode == "chat" else "/v1/completions"
    payload = build_payload(mode, prompt, out_len)
    rec = {"idx": idx, "mode": mode, "ok": False}
    t0 = time.perf_counter()
    first = last = None
    pieces = 0
    usage = {}
    finish = ""
    try:
        async with session.post(BASE_URL + path, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                rec["error"] = f"HTTP {resp.status}: {body[:200]}"
                rec["total_s"] = round(time.perf_counter() - t0, 3)
                return rec
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                ch = (obj.get("choices") or [{}])[0]
                delta = ch.get("delta") or {}
                text = ch.get("text") if mode != "chat" else (
                    delta.get("content") or delta.get("reasoning_content"))
                if text:
                    now = time.perf_counter() - t0
                    if first is None:
                        first = now
                    last = now
                    pieces += 1
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rec["total_s"] = round(time.perf_counter() - t0, 3)
        return rec

    total = time.perf_counter() - t0
    pt = usage.get("prompt_tokens") or 0
    ct = usage.get("completion_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    n_out = ct or pieces
    ttft = (first or 0.0)
    tpot = ((last - first) / (n_out - 1)) if (first and last and n_out > 1) else float("nan")
    rec.update({
        "ok": True,
        "prompt_tokens": pt,
        "cached_tokens": cached,
        "completion_tokens": n_out,
        "ttft_s": round(ttft, 3),
        "tpot_ms": round(tpot * 1000, 2) if tpot == tpot else None,
        "decode_tps": round(1.0 / tpot, 1) if tpot == tpot and tpot > 0 else None,
        "prefill_tps": round(pt / ttft, 0) if ttft > 0 else None,
        "total_s": round(total, 3),
        "finish": finish,
        "pieces": pieces,
    })
    return rec


# ------------------------------------------------------------------ 日志探针
class LogProbe:
    """从服务日志采样投机接受长度 / 解码吞吐 / 预填充吞吐(仅 rank0 行)。"""

    RE_ACCEPT = re.compile(r"accept len: ([\d.]+)")
    RE_GEN = re.compile(r"gen throughput \(token/s\): ([\d.]+)")
    RE_IN = re.compile(r"input throughput \(token/s\): ([\d.]+)")
    RE_TP = re.compile(r"TP(\d+) EP\d+\]")

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.pos = 0
        if self.path and self.path.exists():
            self.pos = self.path.stat().st_size

    def collect(self) -> dict:
        if not self.path or not self.path.exists():
            return {}
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.pos)
                raw = fh.read()
                self.pos = fh.tell()
        except OSError:
            return {}
        acc, gen, inp = [], [], []
        for line in raw.decode("utf-8", "ignore").splitlines():
            mt = self.RE_TP.search(line)
            if mt and mt.group(1) != "0":   # 多 rank 会重复打印, 只取 rank0
                continue
            if m := self.RE_ACCEPT.search(line):
                acc.append(float(m.group(1)))
            if "Decode batch" in line and (m := self.RE_GEN.search(line)):
                gen.append(float(m.group(1)))
            if "Prefill batch" in line and (m := self.RE_IN.search(line)):
                inp.append(float(m.group(1)))
        out = {}
        if acc:
            out["accept_len_mean"] = round(statistics.fmean(acc), 3)
            out["accept_len_n"] = len(acc)
        if gen:
            out["srv_gen_tps_p50"] = round(statistics.median(gen), 1)
            out["srv_gen_tps_max"] = round(max(gen), 1)
        if inp:
            out["srv_prefill_tps_max"] = round(max(inp), 0)
        return out


# ------------------------------------------------------------------ GPU 采样
class GpuSampler(threading.Thread):
    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_evt = threading.Event()
        self.mem_peak = 0
        self.util_samples: list[float] = []

    def run(self) -> None:
        while not self.stop_evt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                mems, utils = [], []
                for line in out.strip().splitlines():
                    a, b = (x.strip() for x in line.split(",")[:2])
                    mems.append(int(float(a)))
                    utils.append(float(b))
                if mems:
                    self.mem_peak = max(self.mem_peak, max(mems))
                    self.util_samples.append(statistics.fmean(utils))
            except Exception:  # noqa: BLE001
                pass
            self.stop_evt.wait(self.interval)

    def stop(self) -> dict:
        self.stop_evt.set()
        self.join(timeout=10)
        return {
            "gpu_mem_peak_mib": self.mem_peak,
            "gpu_util_avg_pct": round(statistics.fmean(self.util_samples), 1)
            if self.util_samples else None,
        }


# ------------------------------------------------------------------ 场景执行
async def run_batch(mode: str, ctx: str, conc: int, out_len: int, salt: str,
                    prompts: list[str] | None = None) -> list[dict]:
    if prompts is None:
        prompts = [read_ctx(ctx, i + 1, salt) for i in range(conc)]
    timeout = aiohttp.ClientTimeout(total=3600, sock_read=3600)
    conn = aiohttp.TCPConnector(limit=0, force_close=False)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as sess:
        return await asyncio.gather(
            *[one_request(sess, mode, prompts[i], out_len, i) for i in range(len(prompts))])


LOG_PATH: Path | None = None


async def measure(mode: str, ctx: str, conc: int, out_len: int, salt: str,
                  log_path: Path | None) -> list[dict]:
    """单请求场景额外采样该请求自身的投机接受长度, 用于折算"每步验证耗时"。

    并发场景无法从共享日志里区分每个请求的解码行, 只保留窗口均值。
    """
    if conc != 1 or log_path is None:
        return await run_batch(mode, ctx, conc, out_len, salt)
    probe = LogProbe(log_path)
    res = await run_batch(mode, ctx, 1, out_len, salt)
    for _ in range(3):
        await asyncio.sleep(0.6)
        st = probe.collect()
        if st.get("accept_len_mean"):
            acc = st["accept_len_mean"]
            res[0]["accept_len"] = acc
            if res[0].get("tpot_ms"):
                res[0]["per_step_ms"] = round(res[0]["tpot_ms"] * acc, 2)
            break
    return res


async def run_hol(conc: int, out_len: int, salt: str) -> list[dict]:
    """队头阻塞探针: 1 条 128K 冷预填充先发 0.5s, 随后 conc-1 条 1K 短请求同时到达。"""
    long_prompt = read_ctx("128k", 9, f"{salt}-hol")
    short_prompts = [read_ctx("1k", 1, f"{salt}-hol{i}") for i in range(conc - 1)]
    timeout = aiohttp.ClientTimeout(total=3600, sock_read=3600)
    async with aiohttp.ClientSession(timeout=timeout,
                                     connector=aiohttp.TCPConnector(limit=0)) as sess:

        async def delayed(prompt: str, idx: int) -> dict:
            await asyncio.sleep(0.5)
            return await one_request(sess, "completion", prompt, out_len, idx)

        tasks = [one_request(sess, "completion", long_prompt, out_len, 0)]
        tasks += [delayed(p, i + 1) for i, p in enumerate(short_prompts)]
        res = await asyncio.gather(*tasks)
    return list(res)


async def flush_cache(session: aiohttp.ClientSession) -> bool:
    for ep in ("/flush_cache",):
        try:
            async with session.post(BASE_URL + ep) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
    return False


def summarize(sid: str, desc: str, results: list[dict], wall: float, extra: dict,
              n_batches: int = 1) -> dict:
    ok = [r for r in results if r.get("ok")]
    tt = [r["ttft_s"] for r in ok]
    tp = [r["tpot_ms"] for r in ok if r.get("tpot_ms")]
    dec = [r["decode_tps"] for r in ok if r.get("decode_tps")]
    out_tok = sum(r["completion_tokens"] for r in ok)
    in_tok = sum(r["prompt_tokens"] for r in ok)
    cached = sum(r["cached_tokens"] for r in ok)
    nb = max(1, n_batches)   # 吞吐按"单轮"折算, 避免多轮重复累加导致虚高
    acc = [r["accept_len"] for r in ok if r.get("accept_len")]
    step = [r["per_step_ms"] for r in ok if r.get("per_step_ms")]
    pref = [r["prefill_tps"] for r in ok if r.get("prefill_tps")]

    def pct(vals: list[float], q: float):
        """线性插值分位数; 直接用 round() 取索引在 n=2 时会退化成 min。"""
        if not vals:
            return None
        s = sorted(vals)
        if len(s) == 1:
            return round(s[0], 3)
        pos = q * (len(s) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return round(s[lo] + (s[hi] - s[lo]) * (pos - lo), 3)

    return {
        "scenario": sid,
        "desc": desc,
        "concurrency": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "wall_s": round(wall, 2),
        "reps": nb,
        "prompt_tokens_total": in_tok,
        "cached_tokens_total": cached,
        "out_tokens_total": out_tok,
        "ttft_p50_s": pct(tt, 0.5),
        "ttft_p95_s": pct(tt, 0.95),
        "ttft_max_s": round(max(tt), 3) if tt else None,
        "tpot_p50_ms": pct(tp, 0.5),
        "tpot_p95_ms": pct(tp, 0.95),
        "tpot_max_ms": round(max(tp), 2) if tp else None,
        "decode_tps_p50": round(statistics.median(dec), 1) if dec else None,
        "decode_tps_max": round(max(dec), 1) if dec else None,
        "decode_tps_min": round(min(dec), 1) if dec else None,
        # 每步验证耗时 = TPOT x 接受长度: 剔除"内容导致的接受长度差异", 反映引擎真实解码开销
        "per_step_ms_p50": round(statistics.median(step), 2) if step else None,
        "accept_len_p50": round(statistics.median(acc), 2) if acc else None,
        "prefill_tps_p50": round(statistics.median(pref), 0) if pref else None,
        "out_tput_tps": round(out_tok / nb / wall, 1) if wall > 0 else None,
        "total_tput_tps": round((in_tok + out_tok) / nb / wall, 1) if wall > 0 else None,
        "req_tput_rps": round(len(ok) / nb / wall, 2) if wall > 0 else None,
        **extra,
    }


async def main_async(args: argparse.Namespace) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = [s.strip() for s in args.only.split(",")] if args.only else [s[0] for s in SCENARIOS]
    unknown = [i for i in ids if i not in SCEN_MAP]
    if unknown:
        log(f"未知场景: {unknown}")
        return 2

    # salt 只与 (场景, 轮次) 有关, 与 --tag 无关:
    # 保证基线 run 与优化后 run 的 prompt 逐字节相同, 又能在同一次 run 内避免跨轮命中缓存。
    salt = args.salt
    log(f"run tag = {args.tag} | salt = {salt} | 场景数 = {len(ids)} | 服务 = {BASE_URL}")

    # ---- 健康检查 + 配置快照
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
        async with s.get(BASE_URL + "/health") as r:
            if r.status != 200:
                log("服务不健康, 退出")
                return 1
        async with s.get(BASE_URL + "/server_info") as r:
            info = await r.json()
        flushed = await flush_cache(s)
    log(f"前缀缓存已清空: {flushed}")

    launch_args = None
    snap = {}
    for key in ("tp_size", "ep_size", "mem_fraction_static", "max_total_tokens",
                "max_running_requests", "chunked_prefill_size", "speculative_algorithm",
                "speculative_num_draft_tokens", "speculative_dspark_block_size",
                "attention_backend", "kv_cache_dtype", "page_size", "context_length",
                "schedule_policy", "disable_overlap_schedule", "enable_cache_report",
                "launch_command"):
        snap[key] = info.get(key)
    launch_args = info.get("launch_command")
    startup = info.get("startup_time")
    if isinstance(startup, dict) and isinstance(info.get("internal_states"), list):
        st = info["internal_states"][0]
        if isinstance(st, dict):
            startup = st.get("startup_time", startup)

    meta = {
        "tag": salt,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": BASE_URL,
        "model": MODEL,
        "server_args": snap,
        "startup_time": startup,
        "scenarios": ids,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    gpu = GpuSampler()
    gpu.start()

    rows, raw_all = [], {}
    t_suite = time.perf_counter()
    log_path = Path(args.log) if args.log else DEFAULT_LOG
    for sid in ids:
        _, desc, mode, ctx, conc, out_len, reps = SCEN_MAP[sid]
        log(f"=== {sid} {desc} | conc={conc} out={out_len} reps={reps}")
        probe = LogProbe(log_path)
        batch_results = []
        wall_best = None
        # 预热 1 次(不计分), 让 CUDA graph / kernel 热起来
        if args.warmup and sid != "CACHE":
            warm_ctx = "1k" if sid == "HOL" else ctx
            await run_batch(mode, warm_ctx, 1, 32, f"{salt}-{sid}-warm")
        rep_walls = []
        for rep in range(reps):
            t0 = time.perf_counter()
            if sid == "HOL":
                # 1 条 128K 长请求先发 0.5s, 随后 31 条 1K 短请求涌入, 观测短请求队头阻塞
                results = await run_hol(conc, out_len, salt)
            elif sid == "CACHE":
                # 两次同一 prompt(同 salt): 第一次冷预填充, 第二次命中前缀缓存
                csalt = f"{salt}-cache"
                r1 = (await measure("completion", ctx, 1, out_len, csalt, log_path))[0]
                r2 = (await measure("completion", ctx, 1, out_len, csalt, log_path))[0]
                r1["idx"], r2["idx"] = 0, 1
                results = [r1, r2]
            else:
                results = await measure(mode, ctx, conc, out_len, f"{salt}-{sid}-r{rep}", log_path)
            # wall 用"该轮最慢请求的端到端耗时", 避免把客户端侧的日志采样等待算进去
            ok_totals = [r["total_s"] for r in results if r.get("ok")]
            wall = max(ok_totals) if ok_totals else time.perf_counter() - t0
            for r in results:
                r["rep"] = rep
            batch_results.extend(results)
            rep_walls.append(wall)
            n_ok = sum(1 for r in results if r.get("ok"))
            log(f"    rep{rep + 1}: ok={n_ok}/{len(results)} wall={wall:.2f}s")
        # 取中位一轮的口径作为该场景 wall(并发场景即该轮总耗时)
        wall_best = statistics.median(rep_walls)
        lstat = probe.collect()
        # CACHE 两次请求语义不同(冷/热), 聚合只用命中缓存的那次, 冷/热分别单列
        summary_input = batch_results[1:] if (sid == "CACHE" and len(batch_results) >= 2) \
            else batch_results
        row = summarize(sid, desc, summary_input, wall_best, lstat,
                        n_batches=len(rep_walls))
        if sid == "CACHE" and len(batch_results) >= 2:
            row["cache_hit_tokens"] = [r.get("cached_tokens") for r in batch_results[:2]]
            row["cache_ttft_cold_s"] = batch_results[0]["ttft_s"]
            row["cache_ttft_hit_s"] = batch_results[1]["ttft_s"]
            row["cache_decode_cold_tps"] = batch_results[0]["decode_tps"]
            row["cache_decode_hit_tps"] = batch_results[1]["decode_tps"]
        if sid == "HOL":
            shorts = [r["ttft_s"] for r in batch_results if r["idx"] > 0 and r.get("ok")]
            if shorts:
                row["hol_short_ttft_p50_s"] = round(statistics.median(shorts), 3)
                row["hol_short_ttft_max_s"] = round(max(shorts), 3)
        rows.append(row)
        raw_all[sid] = batch_results
        log(f"    -> TTFT p50={row['ttft_p50_s']}s  TPOT p50={row['tpot_p50_ms']}ms  "
            f"decode={row['decode_tps_p50']}tok/s  out_tput={row['out_tput_tps']}tok/s  "
            f"accept={row.get('accept_len_mean')}")
        # 场景之间清缓存, 防止后续场景命中上一场景的前缀
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
            await flush_cache(s)
        await asyncio.sleep(2)

    suite_wall = time.perf_counter() - t_suite
    gpu_stat = gpu.stop()
    meta["suite_wall_s"] = round(suite_wall, 1)
    meta["gpu"] = gpu_stat
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    (out_dir / "raw.json").write_text(
        json.dumps(raw_all, ensure_ascii=False, indent=1))
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2))

    cols = ["scenario", "desc", "concurrency", "reps", "ok", "failed", "wall_s",
            "prompt_tokens_total", "cached_tokens_total", "out_tokens_total",
            "ttft_p50_s", "ttft_p95_s", "ttft_max_s",
            "tpot_p50_ms", "tpot_p95_ms", "tpot_max_ms",
            "decode_tps_p50", "decode_tps_min", "decode_tps_max",
            "per_step_ms_p50", "prefill_tps_p50",
            "out_tput_tps", "total_tput_tps", "req_tput_rps",
            "accept_len_p50", "accept_len_mean", "srv_gen_tps_p50", "srv_prefill_tps_max"]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    log(f"完成: {out_dir}  (总耗时 {suite_wall:.0f}s, GPU 显存峰值 {gpu_stat['gpu_mem_peak_mib']}MiB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="SGLang 基线压测套件")
    ap.add_argument("--tag", default="baseline", help="本次运行标签, 也是 results 子目录名")
    ap.add_argument("--salt", default="s1",
                    help="prompt 前缀盐值; 基线/优化后必须一致(改变它=换了一份输入)")
    ap.add_argument("--only", default="", help="只跑指定场景, 逗号分隔")
    ap.add_argument("--log", default="", help="服务日志路径(采集 accept len), 默认 ../logs/server-latest.log")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false", help="跳过每场景预热请求")
    ap.add_argument("--list", action="store_true", help="列出场景后退出")
    args = ap.parse_args()
    if args.list:
        for sid, desc, mode, ctx, conc, out, reps in SCENARIOS:
            print(f"{sid:10s} conc={conc:<3d} ctx={ctx:<5s} out={out:<4d} reps={reps}  {desc}")
        return 0
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
