#!/usr/bin/env python3
"""启动后暖机：触发预填充 CUDA graph 的懒捕获，避免第一个真实用户吃到冷捕获延迟。

背景：分支的预填充图按"本卡精确行数"懒捕获，进程刚起来时每个新的行数组合都要现场
捕获（每段 ~93ms × 24 段，且伴随 eager 回退）。实测冷启动后第一个 5×300K 请求
用了 127s，复跑只需 20.6s。

用法:
  python warmup.py                 # 级别 1（默认）：128K 单发 + 300K 单发 + 5×300K 并发
  python warmup.py --level 1       # 只做 128K + 300K 单发（快）
  python warmup.py --level 0       # 不做
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import aiohttp

BASE = "http://127.0.0.1:30010"
MODEL = "DeepSeek-V4.1-Flash-1017"
CORPUS = Path(__file__).resolve().parent / "corpus"


def build_prompt(target_tokens: int) -> str:
    """按 ~0.25 token/字符 的经验比例拼一段足够长的语料。"""
    need = int(target_tokens / 0.25)
    buf: list[str] = []
    files = sorted(CORPUS.glob("*.txt"))
    while sum(len(x) for x in buf) < need and files:
        for p in files:
            buf.append(p.read_text(errors="ignore"))
            if sum(len(x) for x in buf) >= need:
                break
    return "\n".join(buf)[:need]


async def one(sess: aiohttp.ClientSession, prompt: str, out: int) -> float:
    t0 = time.perf_counter()
    async with sess.post(f"{BASE}/v1/completions", json={
        "model": MODEL, "prompt": prompt, "max_tokens": out,
        "temperature": 0.0, "ignore_eos": True,
    }) as r:
        await r.json()
    return time.perf_counter() - t0


def salted(prompt: str, k: int) -> str:
    """每个并发请求加不同前缀。

    踩过的坑：5 个请求喂同一个 prompt 时，后 4 个命中前缀缓存，批形状变成
    "1 个真预填 + 4 个命中"，完全触发不到多请求同时预填充的行数组合
    —— 实测同一 prompt 暖机后，首个真实 5×300K 仍需 135s。
    """
    return f"## warmup shard={k}\n" + prompt


async def run(level: int) -> int:
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        if level >= 1:
            for tok, tag in ((131072, "128K"), (299000, "300K")):
                p = build_prompt(tok)
                dt = await one(sess, p, 8)
                print(f"  [{tag} 单发] {dt:.1f}s")
        if level >= 2:
            p = build_prompt(299000)
            t0 = time.perf_counter()
            await asyncio.gather(*[one(sess, salted(p, k), 8) for k in range(5)])
            print(f"  [5×300K 并发(不同前缀)] {time.perf_counter() - t0:.1f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1, choices=[0, 1, 2])
    args = ap.parse_args()
    if args.level == 0:
        print("暖机: 跳过")
        return 0
    print(f"暖机开始 (level={args.level}) ...")
    t0 = time.perf_counter()
    asyncio.run(run(args.level))
    print(f"暖机完成，耗时 {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
