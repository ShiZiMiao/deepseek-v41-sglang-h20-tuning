#!/usr/bin/env python3
"""识图能力回归测试（配置变更后必跑）。

设计原则：
  1. ground truth 由脚本生成，判定程序化，不靠人眼；
  2. 覆盖"静默答错"高发处：长序列中的图像位置、多图、旋转/低对比、
     密集文字、UI 截图、大图小字、重叠计数；
  3. real 组需要自备真实图片（放 bench/assets/real/，或用 VISION_REAL_DIR 指定），
     且断言与具体图片内容绑定，换图需同步改断言；目录不存在则自动跳过该组。

用法: python vision_check.py [--only basic|real|ocr|count|rel|ui|big|long|all]
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

BASE = "http://127.0.0.1:30010"
MODEL = "DeepSeek-V4.1-Flash-1017"
CORPUS = Path(__file__).resolve().parent / "corpus"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
REAL_DIR = Path(
    os.environ.get("VISION_REAL_DIR", Path(__file__).resolve().parent / "assets" / "real")
)
REAL = (
    sorted(p for p in REAL_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if REAL_DIR.is_dir()
    else []
)
RESULTS: list[tuple[str, str, bool, str]] = []


def font(sz: int, bold: bool = True):
    return ImageFont.truetype(FONT_B if bold else FONT_R, sz)


def enc(img, fmt: str = "PNG") -> str:
    if isinstance(img, Path):
        m = "jpeg" if img.suffix.lower() in (".jpg", ".jpeg") else "png"
        return f"data:image/{m};base64," + base64.b64encode(img.read_bytes()).decode()
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    m = "png" if fmt == "PNG" else "jpeg"
    return f"data:image/{m};base64," + base64.b64encode(buf.getvalue()).decode()


def ask(images, question: str, max_tokens: int = 200, fmt: str = "PNG"):
    parts = [{"type": "image_url", "image_url": {"url": enc(im, fmt)}} for im in images]
    parts.append({"type": "text", "text": question})
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": max_tokens, "temperature": 0.0, "reasoning_effort": "none",
    }, timeout=900)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    d = r.json()
    return (d["choices"][0]["message"].get("content") or "").strip()


def case(group: str, name: str, images, q: str, check,
         max_tokens: int = 200, fmt: str = "PNG") -> None:
    try:
        got = ask(images, q, max_tokens, fmt)
    except Exception as e:  # noqa: BLE001
        RESULTS.append((group, name, False, f"<异常> {e}"))
        print(f"[ERR ] {group}/{name}: {e}")
        return
    ok = bool(check(got))
    RESULTS.append((group, name, ok, got))
    print(f"[{'OK  ' if ok else 'FAIL'}] {group}/{name}")
    if not ok:
        print(f"       -> {got[:200]!r}")


def has(t: str, *subs: str) -> bool:
    x = t.lower()
    return all(s.lower() in x for s in subs)


def num(t: str, n: int, cn: str = "") -> bool:
    return re.search(rf"\b{n}\b", t) is not None or (bool(cn) and cn in t)


# ---------------------------------------------------------------- 图像生成
def g_solid(color, size=128):
    return Image.new("RGB", (size, size), color)


def g_text(text, w=360, h=120, sz=None):
    img = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(img).text((12, 30), text, fill="black",
                             font=font(sz or max(18, h // 4)))
    return img


def g_shapes():
    img = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((20, 20, 120, 120), fill="red")
    d.rectangle((150, 150, 240, 240), fill="blue")
    return img


def g_table():
    rows = [["A", "17", "3"], ["B", "48", "91"], ["C", "6", "725"],
            ["D", "33", "8"], ["E", "102", "54"]]
    img = Image.new("RGB", (420, 300), "white")
    d = ImageDraw.Draw(img)
    cw, ch, x0, y0 = 140, 52, 10, 12
    for r in range(6):
        d.line([(x0, y0 + r * ch), (x0 + 3 * cw, y0 + r * ch)], fill="black", width=2)
    for c in range(4):
        d.line([(x0 + c * cw, y0), (x0 + c * cw, y0 + 5 * ch)], fill="black", width=2)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            d.text((x0 + c * cw + 28, y0 + r * ch + 14), val, fill="black", font=font(24))
    return img


def g_count(n: int, cols: int = 5):
    rows = (n + cols - 1) // cols
    cell = 100
    img = Image.new("RGB", (cols * cell + 40, rows * cell + 40), "white")
    d = ImageDraw.Draw(img)
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= n:
                break
            cx, cy = 40 + c * cell + 30, 40 + r * cell + 30
            d.ellipse((cx - 26, cy - 26, cx + 26, cy + 26), fill="red")
            k += 1
    return img


def g_small():
    img = Image.new("RGB", (760, 420), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 40), "Quarterly Throughput Report", fill="black", font=font(30))
    for i, t in enumerate([
        "Cluster: h20-node-07   Region: cn-hangzhou",
        "Q3 tokens served: 18,420,913,775",
        "p99 TTFT: 1.83 s    p50 decode: 519 tok/s",
        "Incident ref: TCK-4821 (resolved 2026-11-09)",
    ]):
        d.text((40, 110 + i * 52), t, fill="#222222", font=font(22, False))
    d.text((600, 396), "SN:ZX-9137", fill="#666666", font=font(13, False))
    return img


def g_arrows():
    img = Image.new("RGB", (720, 420), "white")
    d = ImageDraw.Draw(img)
    for x, y, name in [(40, 40, "ALPHA"), (500, 40, "BETA"), (260, 300, "GAMMA")]:
        d.rectangle((x, y, x + 170, y + 70), outline="black", width=3)
        d.text((x + 30, y + 22), name, fill="black", font=font(24))
    d.line([(210, 75), (500, 75)], fill="black", width=3)
    d.polygon([(500, 75), (486, 68), (486, 82)], fill="black")
    d.line([(585, 110), (350, 300)], fill="black", width=3)
    d.polygon([(350, 300), (368, 292), (360, 306)], fill="black")
    d.line([(300, 300), (120, 120)], fill="black", width=3)
    d.polygon([(120, 120), (138, 126), (128, 112)], fill="black")
    return img


def g_dense():
    img = Image.new("RGB", (900, 520), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 24), "Deployment Checklist", fill="black", font=font(26))
    for i, t in enumerate([
        "01  checkpoint integrity  CHECKSUM_OK      475 GB",
        "02  kv cache dtype        fp8_e4m3         page 256",
        "03  prefill cp size       8                interleave",
        "04  dspark block size     5                draft 6",
        "05  max running requests  64               validated",
        "06  rollout window        2026-11-11 09:00 CST",
    ]):
        d.text((30, 90 + i * 62), t, fill="black", font=font(21, False))
    return img


def g_ui():
    ui = Image.new("RGB", (1280, 800), "#f8fafc")
    d = ImageDraw.Draw(ui)
    d.rectangle((0, 0, 1280, 64), fill="#1e293b")
    d.text((24, 18), "SGLang Console", fill="white", font=font(26))
    d.rectangle((40, 100, 420, 300), fill="white", outline="#e2e8f0", width=2)
    d.text((64, 124), "Status", fill="#64748b", font=font(20, False))
    d.ellipse((64, 168, 88, 192), fill="#22c55e")
    d.text((100, 166), "Healthy", fill="#0f172a", font=font(24))
    d.text((64, 220), "Uptime 3h 12m    Requests 48,213", fill="#475569", font=font(19, False))
    d.text((64, 258), "Error rate 0.42%", fill="#475569", font=font(19, False))
    d.rectangle((460, 100, 1240, 210), fill="#fef2f2", outline="#fca5a5", width=2)
    d.text((486, 122), "Prefill timeout on request req-9f3a", fill="#b91c1c", font=font(23))
    d.text((486, 162), "context=312,884 tokens  elapsed=61.2s  retries=1",
           fill="#7f1d1d", font=font(19, False))
    d.rectangle((460, 250, 1240, 296), fill="#e2e8f0")
    for name, x in [("Request", 480), ("Tokens", 800), ("TTFT", 1000)]:
        d.text((x, 262), name, fill="#334155", font=font(20, False))
    for i, (a, b, c, col) in enumerate([("req-9f3a", "312,884", "61.2s", "#dc2626"),
                                        ("req-a71c", "18,204", "0.48s", "#0f172a"),
                                        ("req-b55d", "4,912", "0.09s", "#0f172a")]):
        y = 310 + i * 56
        d.text((480, y), a, fill=col, font=font(20, False))
        d.text((800, y), b, fill=col, font=font(20, False))
        d.text((1000, y), c, fill=col, font=font(20, False))
    d.text((40, 720), "prefill CP8 enabled", fill="#64748b", font=font(20, False))
    return ui


def g_big():
    big = Image.new("RGB", (3000, 2000), "white")
    d = ImageDraw.Draw(big)
    d.text((100, 80), "SYSTEM DASHBOARD", fill="black", font=font(64))
    for i in range(18):
        st = "OK" if i % 5 else "DEGRADED"
        d.text((100, 240 + i * 92), f"node-{i:02d}  status={st}  tokens={1000 + i * 137}",
               fill="#222222", font=font(34, False))
    d.text((2200, 1950), "BUILD-88213", fill="#444444", font=font(28, False))
    return big


def long_ctx(tokens: int) -> str:
    buf = []
    for p in sorted(CORPUS.glob("*.txt")):
        buf.append(p.read_text(errors="ignore"))
        if sum(len(x) for x in buf) > tokens * 2:
            break
    return "\n".join(buf)[: tokens * 2]


def run(group: str) -> None:
    if group in ("basic", "all"):
        case("basic", "纯红块", [g_solid((255, 0, 0))],
             "这张图是什么颜色？只回答颜色名。", lambda t: "红" in t, 16)
        case("basic", "OCR HELLO 42", [g_text("HELLO 42")],
             "图片里写了什么？原样输出。", lambda t: has(t, "HELLO", "42"), 32)
        case("basic", "图形+颜色", [g_shapes()],
             "图中有哪些形状和颜色？简短列出。",
             lambda t: "红" in t and "蓝" in t, 64)
        case("basic", "双图区分", [g_solid((0, 200, 0)), g_solid((255, 255, 0))],
             "两张图依次是什么颜色？用逗号分隔。", lambda t: "绿" in t and "黄" in t, 32)

    if group in ("real", "all"):
        # 示例组：断言是针对一对特定架构图写死的。换成自己的图时，
        # 请同步修改下面的问题与 lambda 判定条件。
        p1, p2 = (REAL[0], REAL[1]) if len(REAL) >= 2 else (None, None)
        if p1 and p1.exists():
            case("real", "图1 红圈数=2", [p1],
                 "图中一共有几个红色圆形？只回答数字。", lambda t: num(t, 2, "两"), 16)
            case("real", "图1 蓝色标签", [p1],
                 "图中左侧蓝色小字写的是什么？原样输出。",
                 lambda t: has(t, "malicious commit"), 64)
            case("real", "图1 虚线框标题", [p1],
                 "红色虚线框上方的标题文字是什么？",
                 lambda t: has(t, "external repository"), 64)
        if p2 and p2.exists():
            case("real", "图2 红圈数=4", [p2],
                 "图中一共有几个红色圆形？只回答数字。", lambda t: num(t, 4, "四"), 16)
            case("real", "图2 指向ID3的标签", [p2],
                 "有一条蓝色标签的箭头是【指向】红色圆圈 Threat ID 3 的，它写的是什么？只输出该标签。",
                 lambda t: has(t, "intercept"), 64)
            case("real", "图2 嵌套容器", [p2],
                 "最外层和最内层的红色虚线框分别叫什么？简短回答。",
                 lambda t: has(t, "local area network") and has(t, "computer"), 128)

    if group in ("ocr", "all"):
        case("ocr", "表格 3行2列=6", [g_table()],
             "表格第 3 行第 2 列的数字是多少？只回答数字。", lambda t: num(t, 6), 16)
        case("ocr", "小字 事故编号", [g_small()],
             "图中 Incident ref 后面的编号是什么？只回答编号。",
             lambda t: has(t, "TCK-4821"), 32)
        case("ocr", "小字 右下角水印", [g_small()],
             "图片右下角很小的灰色文字 SN: 后面是什么？只回答编号。",
             lambda t: has(t, "ZX-9137"), 32)
        case("ocr", "密集文本 第03行", [g_dense()],
             "第 03 行里 prefill cp size 的值是多少？只回答数字。", lambda t: num(t, 8), 16)
        case("ocr", "密集文本 日期", [g_dense()],
             "第 06 行 rollout window 的日期是哪天？只回答日期。",
             lambda t: has(t, "2026-11-11"), 32)
        case("ocr", "旋转90°",
             [g_text("ROTATED-7391", 400, 200, 30).rotate(90, expand=True, fillcolor="white")],
             "把图中的文字原样读出来。", lambda t: has(t, "ROTATED", "7391"), 64)
        case("ocr", "中文+编号", [g_text("校验码 AB-7391", 520, 120, 26)],
             "把图中的文字读出来。", lambda t: has(t, "7391"), 64)
        faint = Image.blend(Image.new("RGB", (520, 120), "white"),
                            g_text("FAINT-2048", 520, 120, 30), 0.25)
        case("ocr", "低对比度", [faint], "图中写了什么？尽量读出。",
             lambda t: has(t, "2048"), 32)

    if group in ("count", "all"):
        for n in (3, 7, 12, 20):
            case("count", f"{n} 个圆", [g_count(n)],
                 "图中有几个红色圆点？只回答数字。", lambda t, n=n: num(t, n), 16)
        for n in (4, 10):
            img = Image.new("RGB", (720, 60 * n + 40), "white")
            d = ImageDraw.Draw(img)
            for i in range(n):
                d.text((30, 24 + i * 60), f"ITEM-{i+1:02d} value={i*7+3}",
                       fill="black", font=font(26))
            case("count", f"文本 {n} 行", [img],
                 "图中一共有几行文字？只回答数字。", lambda t, n=n: num(t, n), 16)

    if group in ("rel", "all"):
        a = g_arrows()
        case("rel", "ALPHA->BETA", [a], "ALPHA 这个框的箭头指向哪个框？只回答名称。",
             lambda t: has(t, "beta") and not has(t, "gamma"), 32)
        case("rel", "谁指向 ALPHA", [a], "哪个框的箭头指向 ALPHA？只回答名称。",
             lambda t: has(t, "gamma"), 32)

    if group in ("ui", "all"):
        u = g_ui()
        case("ui", "报错请求信息", [u],
             "报错的那条请求 ID、上下文长度、耗时分别是多少？简短回答。",
             lambda t: has(t, "req-9f3a") and "312" in t and "61.2" in t, 128)
        case("ui", "状态卡", [u], "左侧状态卡的 uptime 和错误率是多少？",
             lambda t: "3h" in t.replace(" ", "") and "0.42" in t, 64)
        case("ui", "JPEG 压缩后", [u], "报错提示里的 request ID 是什么？只回答 ID。",
             lambda t: has(t, "req-9f3a"), 32, fmt="JPEG")

    if group in ("big", "all"):
        b = g_big()
        case("big", "3000x2000 定位行", [b],
             "node-07 这一行的 status 和 tokens 分别是什么？只回答两个值。",
             lambda t: has(t, "ok") and "1959" in t, 64)
        case("big", "角落小字", [b], "图片右下角的编号是什么？只回答编号。",
             lambda t: "88213" in t, 32)

    if group in ("long", "all"):
        ctx = long_ctx(40000)
        parts = [
            {"type": "image_url", "image_url": {"url": enc(g_solid((0, 0, 255)))}},
            {"type": "text",
             "text": ctx + "\n\n上面这段代码之前有一张图片，它是什么颜色？只回答颜色名。"},
        ]
        try:
            r = requests.post(f"{BASE}/v1/chat/completions", json={
                "model": MODEL, "messages": [{"role": "user", "content": parts}],
                "max_tokens": 32, "temperature": 0.0, "reasoning_effort": "none",
            }, timeout=900)
            got = (r.json()["choices"][0]["message"].get("content") or "").strip()
            ok = "蓝" in got
            RESULTS.append(("long", "40K 文本 + 图在前", ok, got))
            print(f"[{'OK  ' if ok else 'FAIL'}] long/40K 文本 + 图在前")
        except Exception as e:  # noqa: BLE001
            RESULTS.append(("long", "40K 文本 + 图在前", False, str(e)))
            print(f"[ERR ] long: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "basic", "real", "ocr", "count", "rel", "ui", "big", "long"])
    args = ap.parse_args()
    print(f"目标: {BASE}  模型: {MODEL}")
    run(args.only)
    total, passed = len(RESULTS), sum(1 for *_, ok, _ in
                                      [(g, n, o, t) for g, n, o, t in RESULTS] if ok)
    print("\n" + "=" * 70)
    print(f"总计: {passed}/{total} 通过")
    bad = [(g, n, t) for g, n, ok, t in RESULTS if not ok]
    for g, n, t in bad:
        print(f"  失败 {g}/{n} -> {t[:160]!r}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
