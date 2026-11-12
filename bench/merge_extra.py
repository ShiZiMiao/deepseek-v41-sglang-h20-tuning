#!/usr/bin/env python3
"""把补充场景(--only 单跑)的结果行并入指定的基线结果目录。

用法: python merge_extra.py <目标tag> <来源tag> <场景ID>
例:   python merge_extra.py baseline-20261110 extra-x1 D1_300K
"""

import csv
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
COLS = ["scenario", "desc", "concurrency", "reps", "ok", "failed", "wall_s",
        "prompt_tokens_total", "cached_tokens_total", "out_tokens_total",
        "ttft_p50_s", "ttft_p95_s", "ttft_max_s", "tpot_p50_ms", "tpot_p95_ms", "tpot_max_ms",
        "decode_tps_p50", "decode_tps_min", "decode_tps_max", "per_step_ms_p50",
        "prefill_tps_p50", "out_tput_tps", "total_tput_tps", "req_tput_rps",
        "accept_len_p50", "accept_len_mean", "srv_gen_tps_p50", "srv_prefill_tps_max"]


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    tag, src, sid = sys.argv[1:4]
    dst_dir, src_dir = RESULTS / tag, RESULTS / src
    summ = json.loads((dst_dir / "summary.json").read_text())
    raw = json.loads((dst_dir / "raw.json").read_text())
    src_summ = json.loads((src_dir / "summary.json").read_text())
    src_raw = json.loads((src_dir / "raw.json").read_text())

    row = next((r for r in src_summ["rows"] if r["scenario"] == sid), None)
    if row is None:
        print(f"来源结果里没有场景 {sid}")
        return 1
    summ["rows"] = [r for r in summ["rows"] if r["scenario"] != sid] + [row]
    order = [r["scenario"] for r in json.loads((src_dir / "summary.json").read_text())["rows"]]
    pos = {s: i for i, s in enumerate(order)}
    summ["rows"].sort(key=lambda r: pos.get(r["scenario"], 999))

    meta = json.loads((dst_dir / "meta.json").read_text())
    meta["scenarios"] = [r["scenario"] for r in summ["rows"]]
    meta.setdefault("merged_from", []).append({"scenario": sid, "src": src})

    (dst_dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    (dst_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    if sid in src_raw:
        raw[sid] = src_raw[sid]
        (dst_dir / "raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1))
    with open(dst_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(summ["rows"])
    print(f"{tag} <- {sid} from {src}: TTFT={row['ttft_p50_s']}s "
          f"prefill={row.get('prefill_tps_p50')}tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
