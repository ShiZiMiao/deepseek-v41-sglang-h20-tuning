#!/usr/bin/env python3
"""无 CP + 保留视觉 的配置扫参驱动。

每个 trial = 停服 -> 用指定 EXTRA_ARGS 拉起 -> 等健康 -> 跑定向场景 -> 记录。
失败(OOM/启动崩溃)的 trial 自动跳过，最后恢复到最优/默认配置。

用法:
  python sweep.py --list                    # 只看计划
  python sweep.py                           # 跑全部 trial
  python sweep.py --only t1,t3              # 只跑指定 trial
  python sweep.py --resume                  # 已有结果的 trial 跳过
  python sweep.py --restore                 # 只恢复默认配置
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path("scripts")
BENCH_DIR = SCRIPT_DIR / "bench"
RESULTS = BENCH_DIR / "results"
PY = "/opt/venvs/sglang/bin/python"
PORT = 30010

# 定向场景: 覆盖预填充(30K/128K/300K)、重并发预填充、真实画像、吞吐上限
SCEN = "D1_30K,D1_128K,D1_300K,P_5x300K,T32_1K,CHAT_30K"

# trial: (名称, 说明, 环境变量覆盖)
# 注意: prefill CUDA graph 与 bounded replay 互斥（分支里直接 raise），
#       所以 graph 路线必须同时 BOUNDED_REPLAY=0，属于"用一个换另一个"。
TRIALS = [
    ("t0-current", "当前生产：replay 开 + chunk 16384 + graph 关", {}),
    ("t3-chunk32k", "replay 开 + chunk/max-prefill 32768", {
        "EXTRA_ARGS": "--chunked-prefill-size 32768 --max-prefill-tokens 32768",
    }),
    ("t4-chunk64k", "replay 开 + chunk/max-prefill 65536", {
        "EXTRA_ARGS": "--chunked-prefill-size 65536 --max-prefill-tokens 65536",
    }),
    ("t1b-graph-noreplay", "graph full + replay 关（chunk 16384）", {
        "EXTRA_ARGS": "--cuda-graph-backend-prefill full",
        "BOUNDED_REPLAY": "0",
    }),
    ("t2b-graph32k-noreplay", "graph full + replay 关 + chunk 32768", {
        "EXTRA_ARGS": "--cuda-graph-backend-prefill full "
                      "--chunked-prefill-size 32768 --max-prefill-tokens 32768",
        "BOUNDED_REPLAY": "0",
    }),
    ("t5-chunk8k", "replay 开 + chunk/max-prefill 8192（反向验证）", {
        "EXTRA_ARGS": "--chunked-prefill-size 8192 --max-prefill-tokens 8192",
    }),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: list[str], env: dict | None = None, timeout: int = 300) -> tuple[int, str]:
    e = os.environ.copy()
    if env:
        e.update(env)
    p = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=e, timeout=timeout,
                       capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def health() -> bool:
    try:
        p = subprocess.run(["curl", "-sf", "-m", "3", f"http://127.0.0.1:{PORT}/health"],
                           capture_output=True, timeout=8)
        return p.returncode == 0
    except Exception:
        return False


def stop() -> None:
    sh(["./stop.sh"], timeout=180)


def start(env_over: dict, wait_s: int = 900) -> tuple[bool, float]:
    """拉起服务并等到健康。返回 (成功, 耗时秒)。"""
    t0 = time.time()
    e = os.environ.copy()
    e.update(env_over)
    launch_log = SCRIPT_DIR / "logs" / f"sweep-{int(t0)}.log"
    with open(launch_log, "w") as fh:
        subprocess.Popen(["./start-opt.sh"], cwd=str(SCRIPT_DIR), env=e,
                         stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    while time.time() - t0 < wait_s:
        if health():
            return True, time.time() - t0
        # 进程已死则提前失败
        p = subprocess.run(["pgrep", "-f", "sglang.launch_server"], capture_output=True)
        if p.returncode != 0 and time.time() - t0 > 60:
            return False, time.time() - t0
        time.sleep(5)
    return False, time.time() - t0


def run_bench(tag: str) -> dict | None:
    path = RESULTS / tag / "summary.json"
    rc, out = sh([PY, str(BENCH_DIR / "bench_suite.py"), "--tag", tag, "--only", SCEN,
                  "--log", str(SCRIPT_DIR / "logs" / "opt-latest.log")],
                 env={"PYTHONPATH": ""}, timeout=900)
    if rc != 0 or not path.exists():
        log(f"  压测失败 rc={rc}: {out.strip().splitlines()[-1] if out.strip() else ''}")
        return None
    return json.loads(path.read_text())


def brief(summ: dict) -> dict:
    rows = {r["scenario"]: r for r in summ["rows"]}
    g = lambda s, k: rows.get(s, {}).get(k)
    return {
        "ttft_30k": g("D1_30K", "ttft_p50_s"),
        "ttft_128k": g("D1_128K", "ttft_p50_s"),
        "ttft_300k": g("D1_300K", "ttft_p50_s"),
        "p5x300k": g("P_5x300K", "ttft_p50_s"),
        "chat_ttft": g("CHAT_30K", "ttft_p50_s"),
        "chat_dec": g("CHAT_30K", "decode_tps_p50"),
        "t32": g("T32_1K", "out_tput_tps"),
        "pre_128k": g("D1_128K", "prefill_tps_p50"),
    }


def table(records: list[tuple[str, str, dict | None, float]]) -> None:
    hdr = (f"{'trial':<20}{'状态':<8}{'30K首字':>9}{'128K首字':>10}"
           f"{'300K首字':>10}{'5x300K':>9}{'CHAT首字':>10}{'T32吞吐':>9}{'启动':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, _desc, b, dt in records:
        if not b:
            print(f"{name:<20}{'失败':<8}{'—':>9}{'—':>10}{'—':>10}{'—':>9}{'—':>10}{'—':>9}{dt:>6.0f}s")
            continue
        f = lambda v: "—" if v is None else f"{v:g}"
        print(f"{name:<20}{'OK':<8}{f(b['ttft_30k']):>9}{f(b['ttft_128k']):>10}"
              f"{f(b['ttft_300k']):>10}{f(b['p5x300k']):>9}{f(b['chat_ttft']):>10}"
              f"{f(b['t32']):>9}{dt:>6.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, d, e in TRIALS:
            print(f"{n:<20} {d}")
            for k, v in e.items():
                print(f"    {k}={v}")
        return 0

    if args.restore:
        stop()
        ok, dt = start({})
        log(f"恢复默认配置: {'成功' if ok else '失败'}, {dt:.0f}s")
        return 0 if ok else 1

    todo = TRIALS
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        todo = [t for t in TRIALS if t[0] in want]

    records = []
    for name, desc, env in todo:
        tag = f"sweep-{name}"
        if args.resume and (RESULTS / tag / "summary.json").exists():
            summ = json.loads((RESULTS / tag / "summary.json").read_text())
            log(f"{name}: 复用已有结果")
            records.append((name, desc, brief(summ), 0.0))
            continue

        log(f"=== {name}: {desc}")
        stop()
        ok, dt = start(env)
        if not ok:
            log(f"  启动失败（{dt:.0f}s），跳过。可能 OOM 或参数不被接受")
            log_tail = ""
            latest = sorted((SCRIPT_DIR / "logs").glob("opt-2*.log"))
            if latest:
                lines = latest[-1].read_text(errors="ignore").splitlines()[-12:]
                log_tail = "\n".join(lines)
            if log_tail:
                log("  日志尾部:\n" + log_tail)
            records.append((name, desc, None, dt))
            continue

        log(f"  就绪 {dt:.0f}s，开始压测")
        summ = run_bench(tag)
        b = brief(summ) if summ else None
        if b:
            log(f"  结果: 128K={b['ttft_128k']}s  5x300K={b['p5x300k']}s  "
                f"T32={b['t32']}tok/s  CHAT={b['chat_ttft']}s")
        records.append((name, desc, b, dt))

    table(records)
    out = RESULTS / "sweep-summary.json"
    out.write_text(json.dumps(
        [{"trial": n, "desc": d, "metrics": b, "startup_s": round(dt, 0)}
         for n, d, b, dt in records], ensure_ascii=False, indent=2))
    log(f"汇总已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
