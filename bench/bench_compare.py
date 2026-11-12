#!/usr/bin/env python3
"""对比两次 bench_suite 运行的结果, 输出优化前后的逐场景 delta 表。

用法:
  python bench_compare.py baseline-20261110 opt-20261112
  python bench_compare.py baseline-20261110 opt-20261112 --md   # 输出 markdown

方向约定: TTFT/TPOT 越低越好, 吞吐/解码速度越高越好, 表中自动判定涨跌好坏。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

# 并发预填充挤压场景里, 接受长度本身不可复现 -> TPOT/解码 的跨轮偏差实测可达 ±40%,
# 阈值放大 5 倍, 避免把噪声误判成收益。
NOISY_DECODE = {"P_5x30K", "P_5x300K", "HOL"}
NOISY_KEYS = {"tpot_p50_ms", "tpot_p95_ms", "decode_tps_p50", "decode_tps_min"}

# 指标 -> (显示名, 方向, 噪声阈值%)  direction: +1 越大越好, -1 越小越好, 0 中性
# 阈值取自同配置连跑两轮的实测偏差(见 基线性能报告.md §3.5), 变化小于阈值一律判为噪声。
METRICS = [
    ("ttft_p50_s",      "TTFT p50 (s)",      -1, 5.0),
    ("ttft_p95_s",      "TTFT p95 (s)",      -1, 5.0),
    ("tpot_p50_ms",     "TPOT p50 (ms)",     -1, 15.0),
    ("tpot_p95_ms",     "TPOT p95 (ms)",     -1, 15.0),
    ("decode_tps_p50",  "解码 p50 (tok/s)",  +1, 8.0),
    ("decode_tps_min",  "解码 min (tok/s)",  +1, 8.0),
    ("per_step_ms_p50", "每步耗时 (ms)",     -1, 12.0),
    ("prefill_tps_p50", "预填充 (tok/s)",    +1, 5.0),
    ("out_tput_tps",    "输出吞吐 (tok/s)",  +1, 8.0),
    ("total_tput_tps",  "总吞吐 (tok/s)",    +1, 8.0),
    ("wall_s",          "总耗时 (s)",        -1, 5.0),
    ("accept_len_mean", "接受长度",           0, 0.0),   # 内容驱动, 不作判定
]


def load(tag: str) -> dict:
    path = RESULTS / tag / "summary.json"
    if not path.exists():
        sys.exit(f"缺少结果文件: {path}")
    data = json.loads(path.read_text())
    return {r["scenario"]: r for r in data["rows"]}, data["meta"]


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--md", action="store_true", help="输出 markdown 表")
    args = ap.parse_args()

    base, bmeta = load(args.base)
    new, nmeta = load(args.new)

    print(f"基线: {args.base}  ({bmeta['ts']})")
    print(f"对比: {args.new}  ({nmeta['ts']})")
    bargs, nargs = bmeta.get("server_args") or {}, nmeta.get("server_args") or {}
    diff = {k: (bargs.get(k), nargs.get(k)) for k in set(bargs) | set(nargs)
            if bargs.get(k) != nargs.get(k)}
    if diff:
        print("\n配置差异:")
        for k, (b, n) in sorted(diff.items()):
            print(f"  {k}: {fmt(b)} -> {fmt(n)}")
    else:
        print("\n配置无差异(仅参数外因素变化)")

    for sid in [s for s in base if s in new]:
        b, n = base[sid], new[sid]
        print(f"\n### {sid} · {b['desc']}")
        if sid in NOISY_DECODE:
            print("（本场景并发预填充挤压, 接受长度跨轮不可复现: TPOT/解码 仅作参考, 不给判定）")
        if args.md:
            print("| 指标 | 基线 | 优化后 | 变化 | 判定 |")
            print("|---|---|---|---|---|")
        else:
            print(f"{'指标':<20}{'基线':>12}{'优化后':>12}{'变化':>12}  判定")
        for key, label, direction, noise in METRICS:
            bv, nv = b.get(key), n.get(key)
            if bv in (None, 0) or nv is None:
                continue
            pct = (nv - bv) / bv * 100.0
            verdict = ""
            if sid in NOISY_DECODE and key in NOISY_KEYS:
                verdict = "— 参考值(该场景噪声大)"
            elif not direction:
                verdict = "— 内容驱动, 不作为依据"
            elif abs(pct) < noise:
                verdict = f"≈ 噪声内 (±{noise:g}%)"
            else:
                good = (pct > 0) == (direction > 0)
                verdict = "✅ 改善" if good else "❌ 劣化"
            if args.md:
                print(f"| {label} | {fmt(bv)} | {fmt(nv)} | {pct:+.1f}% | {verdict} |")
            else:
                print(f"{label:<20}{fmt(bv):>12}{fmt(nv):>12}{pct:>+11.1f}%  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
