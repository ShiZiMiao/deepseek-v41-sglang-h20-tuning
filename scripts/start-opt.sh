#!/usr/bin/env bash
#
# DeepSeek-V4.1-Flash-1017 · 优化验证启动脚本（与 start.sh 并存，互不影响）
#
# 用途：按docs/baseline-benchmark-report.md§6 的分阶段计划，逐个变量验证优化项。
#       基线配置永远由 start.sh 负责，本脚本只做实验；回滚 = ./stop.sh && ./start.sh
#
# 用法:
#   ./start-opt.sh                          # 默认 = branch + CP8（文章的完整配方）
#   PROFILE=stock ./start-opt.sh            # 只开 CP8，不换代码（验证 CP 本身值多少）
#   PARALLEL=ours ./start-opt.sh            # 保留我们的 EP8/attn-dp1，只加 CP
#   CHUNKED_PREFILL_SIZE=8192 ./start-opt.sh
#
# 关闭任一项做消融（都默认 1/开）:
#   CP=0 BOUNDED_REPLAY=0 MOE_BACKEND=auto ./start-opt.sh
#
# 代码来源（PROFILE=branch 时）:
#   /opt/sglang-dsv41 = antgroup/sglang 分支 dsv41-flash-h20 @6e6d5d9e13
#   仅通过 PYTHONPATH 覆盖，不改动 /opt/venvs/sglang 里的 0.5.23
#
set -euo pipefail

# ========================= 可配置项 =========================
MODEL_PATH="${MODEL_PATH:-/models/weights/DeepSeek-V4.1-Flash-1017}"
VENV_PY="${VENV_PY:-/opt/venvs/sglang/bin/python}"

PROFILE="${PROFILE:-branch}"        # branch | stock
PARALLEL="${PARALLEL:-article}"     # article = TP8 无 EP（文章实测配方） | ours = EP8 + attn-dp1
# prefill CP8 + CP decode attn TP。
# 上游版本里视觉塔与 CP 互斥；本地分支副本已打补丁放开（见下方自动探测），
# 因此默认两者都开。若补丁被回滚，脚本会自动把 CP 降为 0。
CP="${CP:-1}"
MOE_BACKEND="${MOE_BACKEND:-humming}"
BOUNDED_REPLAY="${BOUNDED_REPLAY:-1}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.05}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
MEM_FRACTION="${MEM_FRACTION:-0.84}"
# 0 = 加载完整模型（含视觉塔，支持图片输入）；1 = 只加载语言模型（文章原配方）。
# 2026-11-10 改为默认 0：业务需要识图。原先设 1 是因为视觉处理器初始化会触发
# 分支 rust/ 目录的 rust-toolchain.toml(channel=1.92)，rustup 去线上下载工具链，
# 把 8 个 rank 全堵在 barrier 上（实测 10min+ 无进展）。
# 该根因现已消除：工具链 1.92 已装好，Rust 图像处理器扩展已预编译进
# ~/.cache/sglang/rust_extensions/，启动时直接命中缓存，不再联网。
LANGUAGE_ONLY="${LANGUAGE_ONLY:-0}"

BRANCH_ROOT="${BRANCH_ROOT:-/opt/sglang-dsv41}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30010}"
SERVED_NAME="${SERVED_NAME:-DeepSeek-V4.1-Flash-1017}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
# KV 池上限沿用基线钉死值，隔离"容量"变量，只让速度变量变化
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-3145728}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
ENGRAM_HOST_TABLE="${ENGRAM_HOST_TABLE:-1}"
ENGRAM_LAYOUT="${ENGRAM_LAYOUT:-per_rank}"
ENGRAM_STAGING="${ENGRAM_STAGING:-1}"
NUMA_BIND="${NUMA_BIND:-1}"
START_TIMEOUT="${START_TIMEOUT:-1800}"
# 启动后暖机（触发预填充 CUDA graph 懒捕获，避免第一个真实用户吃到冷捕获延迟）：
#   0=跳过  1=128K+300K 各一发  2=再加 5×300K 并发（默认）
WARMUP="${WARMUP:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/opt-$(date +%Y%m%d-%H%M%S).log"
PID_FILE="${SCRIPT_DIR}/server.pid"
LATEST_LINK="${LOG_DIR}/opt-latest.log"

log() { echo "[start-opt.sh] $*"; }

# ---------------------------- 预检 ----------------------------
[[ -x "${VENV_PY}" ]] || { log "错误: 找不到 Python 环境 ${VENV_PY}"; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { log "错误: 权重目录无效: ${MODEL_PATH}"; exit 1; }
if [[ "${PROFILE}" == "branch" ]]; then
    [[ -f "${BRANCH_ROOT}/python/sglang/__init__.py" ]] || {
        log "错误: 分支代码不存在: ${BRANCH_ROOT}/python/sglang"; exit 1; }
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    if [[ -f "${PID_FILE}" ]]; then
        OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
            log "服务已在运行 (PID ${OLD_PID})，请先执行 ./stop.sh"; exit 1
        fi
        rm -f "${PID_FILE}"
    fi
    if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
        log "错误: 端口 ${PORT} 已被占用，请先释放"; exit 1
    fi
fi

mkdir -p "${LOG_DIR}"
ln -sfn "${LOG_FILE}" "${LATEST_LINK}"

# ---------------------------- 环境变量 ----------------------------
# start.sh 会 unset PYTHONPATH（防 IDE 注入），这里按需要重新指定
unset PYTHONPATH || true
export CODEBUDDY_SAFE_DELETE_ENABLED=0
export PYTHONUNBUFFERED=1
export SGLANG_DSV41_REASONING_EFFORT="${REASONING_EFFORT}"
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${ENGRAM_HOST_TABLE}" == "1" ]]; then
    export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
    export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT="${ENGRAM_LAYOUT}"
    export SGLANG_OPT_DSV41_ENGRAM_HOST_STAGING="${ENGRAM_STAGING}"
    log "engram 表:   主机内存, layout=${ENGRAM_LAYOUT}, staging=${ENGRAM_STAGING}"
fi

if [[ "${PROFILE}" == "branch" ]]; then
    export PYTHONPATH="${BRANCH_ROOT}/python"
    export SGLANG_OPT_DSV41_SM90=1
    export SGLANG_OPT_FP8_WO_A_GEMM=1
    export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}'
    # 工具链 1.92 已按分支 rust/rust-toolchain.toml 装好（含 clippy/rustfmt），
    # 不再覆盖 RUSTUP_TOOLCHAIN —— 保持与预编译缓存一致的 1.92，避免指纹不匹配触发重编译。
    # 有损项默认不开（文章只在 PD 的 D 角色开 fp8_lm_head_target）
    log "代码:        ${BRANCH_ROOT} (dsv41-flash-h20) + SGLANG_OPT_DSV41_SM90=1"
else
    log "代码:        环境内置 sglang 0.5.23（仅开 CP，不换代码）"
fi

# ---------------------------- 组装参数 ----------------------------
ARGS=(
    --model-path "${MODEL_PATH}"
    --served-model-name "${SERVED_NAME}"
    --trust-remote-code
    --host "${HOST}"
    --port "${PORT}"
    --tp 8
    --context-length "${CONTEXT_LENGTH}"
    --mem-fraction-static "${MEM_FRACTION}"
    --max-total-tokens "${MAX_TOTAL_TOKENS}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size 5
    --reasoning-parser auto
    --tool-call-parser auto
    --default-chat-template-kwargs "{\"thinking\": true, \"reasoning_effort\": \"${REASONING_EFFORT}\"}"
    --cuda-graph-max-bs-decode 64
    --enable-cache-report
)

if [[ "${LANGUAGE_ONLY}" == "1" ]]; then
    # 注意: 不可与 --language-only 同时给, 分支里会直接 raise ValueError
    ARGS+=(--language-model-only)
    log "模型范围:    仅语言模型（无识图）"
else
    log "模型范围:    完整模型（含视觉塔，支持图片输入）"
fi

if [[ "${PARALLEL}" == "ours" ]]; then
    ARGS+=(--attn-dp-size 1 --ep-size 8)
    log "并行:        TP=8 EP=8 attn-dp=1（保留基线并行方式）"
else
    log "并行:        TP=8（文章的实测配方，无 EP）"
fi

# 上游禁止「视觉塔 + prefill CP」同时启用（models/deepseek_v4.py 里 attn_cp_size != 1 直接 raise）。
# 我们的本地分支副本已打补丁放开（标记 `LOCAL EXPERIMENT 2026-11-11`），此处探测补丁是否存在：
# 补丁在 -> 两者都开；补丁被 git checkout 回滚 -> 自动降 CP，避免启动即崩。
if [[ "${LANGUAGE_ONLY}" != "1" && "${CP}" == "1" && "${ALLOW_VISION_CP:-auto}" != "1" ]]; then
    if ! grep -q "LOCAL EXPERIMENT 2026-11-11" \
        "${BRANCH_ROOT}/python/sglang/srt/models/deepseek_v4.py" 2>/dev/null; then
        log "[告警] 分支代码未包含 vision+CP 补丁，已自动把 CP 降为 0（无识图损失，仅预填充变慢）。"
        log "       要 CP8：重新打补丁，或设 LANGUAGE_ONLY=1。"
        CP=0
    fi
fi

if [[ "${CP}" == "1" ]]; then
    ARGS+=(--enable-prefill-cp --cp-strategy interleave --enable-cp-decode-attn-tp)
    log "CP 预填充:   开启 (interleave, decode attn TP)"
else
    log "CP 预填充:   关闭（预填充吞吐约为开启 CP 的 40%）"
fi

if [[ "${BOUNDED_REPLAY}" == "1" ]]; then
    ARGS+=(--enable-decoder-swa-bounded-replay --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}")
    log "晚层跳过:    开启 bounded replay (近似, ratio=${SWA_FULL_TOKENS_RATIO})"
else
    # 必须显式关: 该选项在"支持的配置"下默认开启(例如关掉 CP 时),
    # 仅靠"不传参数"并不能保证关闭。
    ARGS+=(--no-enable-decoder-swa-bounded-replay)
    log "晚层跳过:    关闭 bounded replay（完整 prefill，数值精确）"
fi

ARGS+=(--moe-runner-backend "${MOE_BACKEND}")
log "MoE 后端:    ${MOE_BACKEND}"

# 基线专用的两个参数在分支上不存在/冲突，按 PROFILE 决定是否带上
if [[ "${PROFILE}" == "stock" ]]; then
    ARGS+=(--cuda-graph-backend-prefill breakable
           --cuda-graph-max-bs-prefill 8192
           --cuda-graph-max-seq-len-prefill 16384
           --disable-flashinfer-autotune)
fi

EXTRA_ARGS_ARR=()
[[ -n "${EXTRA_ARGS}" ]] && read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS}"
ARGS+=("${EXTRA_ARGS_ARR[@]}")

LAUNCH_PREFIX=()
if [[ "${NUMA_INTERLEAVE:-1}" == "1" ]] && command -v numactl >/dev/null 2>&1; then
    LAUNCH_PREFIX=(numactl --interleave=all)
    # 关键：sglang 自己会把每个 worker 的内存拉回本地，父进程的 interleave 永远不生效——
    #   V2 路径：spawn worker 时套 numactl --membind=<本地节点>（numa_maps 显示 bind:N）
    #   AUTO 路径：worker 内调用 numa_set_preferred(本地节点)（numa_maps 显示 prefer:N）
    # 要真正交错，这两层都得关掉。关掉后 CPU 亲和仍由下方 NUMA_BIND 的 taskset 保证。
    # 2026-11-12 A/B 实测（各 2 轮 × 12 次，全新启动）：1K 单流 623 → 648 tok/s（+4%），
    # 30K 无差异；见 docs/optimization-report.md §「NUMA 交错 A/B」。
    export SGLANG_NUMA_BIND_V2=0
    export SGLANG_AUTO_NUMA_BIND=0
    log "NUMA 内存:  交错放置（已关框架 per-worker 绑定；实测 1K 单流 +4%）"
fi

log "监听:        http://${HOST}:${PORT}"
log "日志:        ${LOG_FILE}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    log "DRY_RUN: 仅打印启动命令，不实际启动"
    printf 'PYTHONPATH=%s \\\n' "${PYTHONPATH:-<未设置>}"
    printf '%s\n' "${LAUNCH_PREFIX[*]:-}" "${VENV_PY} -m sglang.launch_server ${ARGS[*]}" | sed '/^$/d'
    exit 0
fi

setsid "${LAUNCH_PREFIX[@]}" "${VENV_PY}" -m sglang.launch_server "${ARGS[@]}" \
    >>"${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
log "服务进程已拉起 (PID ${SERVER_PID})，等待模型加载..."

# ---------------------------- NUMA 绑核 ----------------------------
if [[ "${NUMA_BIND}" == "1" ]]; then
    bind_rank() {
        local rank="$1" cpus="$2" pids p
        # 进程名随并行方式变化：普通 TP/EP 是 scheduler_TP{N}_EP{N}，
        # 开 prefill CP 后变成 scheduler_ATTN_CP{N}_TP{N}。两种都要匹配，
        # 否则绑核会静默失效（踩过：CP 模式下旧的单模式匹配数为 0）。
        for _ in $(seq 1 120); do
            pids="$(pgrep -g "${SERVER_PID}" -f "scheduler_(ATTN_CP${rank}_TP${rank}|TP${rank}_EP)" || true)"
            [[ -n "${pids}" ]] && break
            sleep 2
        done
        for p in ${pids}; do
            taskset -pc "${cpus}" "${p}" >/dev/null 2>&1 && log "  [OK] TP${rank} (${p}) -> CPU ${cpus}"
        done
    }
    BIND_PIDS=()
    for i in 0 1 2 3; do bind_rank "${i}" "0-119" & BIND_PIDS+=($!); done
    for i in 4 5 6 7; do bind_rank "${i}" "120-239" & BIND_PIDS+=($!); done
    wait "${BIND_PIDS[@]}"
fi

# ---------------------------- 健康检查 ----------------------------
HEALTH_URL="http://127.0.0.1:${PORT}/health"
START_TS="$(date +%s)"
while true; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "错误: 服务进程已退出，最近日志："
        tail -n 60 "${LOG_FILE}" || true
        rm -f "${PID_FILE}"
        exit 1
    fi
    curl -sf --max-time 3 "${HEALTH_URL}" >/dev/null 2>&1 && break
    NOW="$(date +%s)"
    if (( NOW - START_TS > START_TIMEOUT )); then
        log "错误: 等待超时 (${START_TIMEOUT}s)，日志 ${LOG_FILE}"; exit 1
    fi
    sleep 5
done
ELAPSED=$(( $(date +%s) - START_TS ))
log "健康检查通过，耗时 ${ELAPSED}s"

# ---------------------------- 就绪信息（只报告，不阻断） ----------------------------
INFO_JSON="$(curl -s --max-time 15 "http://127.0.0.1:${PORT}/get_server_info")"
log "server_info 关键字段（与基线不同属正常，需人工确认）:"
printf '%s' "${INFO_JSON}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for k in ('max_total_num_tokens','speculative_algorithm','attention_backend',
          'kv_cache_dtype','page_size','max_running_requests','enable_prefill_cp',
          'cp_strategy','moe_runner_backend'):
    print(f'  {k:28s} = {d.get(k, \"<无此字段>\")}')
" 2>/dev/null || log "  (server_info 解析失败)"

# ---------------------------- 冒烟请求 ----------------------------
log "冒烟请求: max_tokens=8 ..."
SMOKE_RAW="$(curl -s --max-time 120 -w $'\n%{http_code}' "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"请说一句你好\"}],\"max_tokens\":8,\"reasoning_effort\":\"none\"}")"
SMOKE_VERDICT="$(printf '%s' "${SMOKE_RAW}" | python3 -c "
import json, sys
raw = sys.stdin.read()
body, _, code = raw.rpartition('\n')
try:
    d = json.loads(body)
    ch = d['choices'][0]
    text = (ch['message'].get('content') or '').strip()
    print(('OK|' if code == '200' and text else 'FAIL|') + (text[:40] or 'HTTP=%s 空文本' % code))
except Exception as e:
    print('FAIL|解析失败: %s' % e)
" 2>/dev/null || echo "FAIL|curl/python 异常")"

if [[ "${SMOKE_VERDICT}" == OK\|* ]]; then
    log "  [OK] 冒烟通过: ${SMOKE_VERDICT#OK|}"
else
    log "  [告警] 冒烟失败: ${SMOKE_VERDICT#FAIL|}（服务保持运行，请核查日志后决定是否回滚）"
    exit 1
fi

# ---------------------------- 暖机（预填充图懒捕获） ----------------------------
if [[ "${WARMUP}" != "0" ]]; then
    log "暖机: 触发预填充图捕获 (level=${WARMUP}) ..."
    if "${VENV_PY}" "${SCRIPT_DIR}/bench/warmup.py" --level "${WARMUP}" 2>&1 | sed 's/^/    /'; then
        :
    else
        log "  [告警] 暖机失败，不影响服务，首个重上下文请求可能偏慢"
    fi
fi

log "启动完成，耗时 ${ELAPSED}s（健康检查）+ 暖机。回滚: ./stop.sh && ./start.sh"
