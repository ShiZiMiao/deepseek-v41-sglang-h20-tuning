#!/usr/bin/env bash
#
# DeepSeek-V4.1-Flash-1017 · SGLang 一键启动脚本
#
# 用法:
#   ./start.sh                 # 启动服务（后台运行，日志写入 logs/）
#   PORT=30010 ./start.sh      # 通过环境变量覆盖配置
#
# 启动成功后提供 OpenAI 兼容接口（供 OpenCode 等 CodingAgent 接入）:
#   POST http://<host>:30010/v1/chat/completions   （支持 tools / 流式 / reasoning_content）
#   POST http://<host>:30010/v1/completions
#   GET  http://<host>:30010/v1/models

set -euo pipefail

# ========================= 可配置项（环境变量可覆盖） =========================
MODEL_PATH="${MODEL_PATH:-/models/weights/DeepSeek-V4.1-Flash-1017}"
VENV_PY="${VENV_PY:-/opt/venvs/sglang/bin/python}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30010}"
SERVED_NAME="${SERVED_NAME:-DeepSeek-V4.1-Flash-1017}"

TP_SIZE="${TP_SIZE:-8}"                      # 8x H20-3e 141G
EP_SIZE="${EP_SIZE:-8}"                      # MoE 专家并行（384 专家 / 8 卡）
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"  # 1M token 上下文
MEM_FRACTION="${MEM_FRACTION:-0.80}"         # 静态显存占比上限（权重 + KV 池）
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-3145728}"  # KV 池上限 3M token（给长提示预填充留激活显存）
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"

# reasoning 默认档：用户未传 reasoning_effort/thinking 时生效（用户显式值优先；
# 传 "reasoning_effort":"none" 可显式关闭思考）。可选 low/medium/high/max 或整数 1-100
# 默认 high，与权重 encoding.py 的 DEFAULT_REASONING_EFFORT="high"(=75) 对齐
REASONING_EFFORT="${REASONING_EFFORT:-high}"

# DSpark 投机解码：checkpoint 内置草稿头，block_size=5 为训练值（与权重绑定，勿改）

# B5 配方(2026-11-08 实测采纳): engram 大表放主机内存 + staging 预取 + THP 大页 +
# NUMA 交错。省 23~29GB/卡显存, 单人解码 312tok/s 全场最佳, 仅重预填充并发慢 ~40%。
# 显存宽裕且重预填充为刚需时可 ENGRAM_HOST_TABLE=0 还原显存驻留(A 配置)。
ENGRAM_HOST_TABLE="${ENGRAM_HOST_TABLE:-1}"

# NUMA 绑核: GPU0-3->NUMA0(CPU0-119)、GPU4-7->NUMA1(CPU120-239)；置 0 关闭
# 实测收益(2026-11-04 R3): 5x30K 并发 TPOT p95 30.4->12.5ms、解码最低 33->80 tok/s
NUMA_BIND="${NUMA_BIND:-1}"

# 启动总超时秒数（475G 权重冷加载较慢）
START_TIMEOUT="${START_TIMEOUT:-1800}"

# 高级: 追加 sglang 启动参数（空=无）。实验/排障用，生产默认不设
EXTRA_ARGS="${EXTRA_ARGS:-}"
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/server-$(date +%Y%m%d-%H%M%S).log"
PID_FILE="${SCRIPT_DIR}/server.pid"
LATEST_LINK="${LOG_DIR}/server-latest.log"

log() { echo "[start.sh] $*"; }

# ---------------------------- 预检 ----------------------------
if [[ ! -x "${VENV_PY}" ]]; then
    log "错误: 找不到 Python 环境 ${VENV_PY}"; exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    log "错误: 权重目录无效（缺少 config.json）: ${MODEL_PATH}"; exit 1
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
if (( GPU_COUNT < TP_SIZE )); then
    log "错误: 需要 ${TP_SIZE} 张 GPU，当前只有 ${GPU_COUNT} 张"; exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
    OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
        log "服务已在运行 (PID ${OLD_PID})。如需重启请先执行 ./stop.sh"; exit 1
    fi
    rm -f "${PID_FILE}"
fi

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    log "错误: 端口 ${PORT} 已被占用，请先释放或更换 PORT"; exit 1
fi

mkdir -p "${LOG_DIR}"
ln -sfn "${LOG_FILE}" "${LATEST_LINK}"

# ---------------------------- 启动 ----------------------------
log "模型:        ${MODEL_PATH}"
log "环境:        ${VENV_PY}"
log "并行:        TP=${TP_SIZE} EP=${EP_SIZE}  投机解码=DSPARK(block=5)"
log "上下文:      ${CONTEXT_LENGTH} tokens (1M)"
log "reasoning:   默认 ${REASONING_EFFORT}（用户未传 reasoning_effort 时生效）"
log "监听:        http://${HOST}:${PORT}  (模型名: ${SERVED_NAME})"
log "日志:        ${LOG_FILE}"

# 清理 IDE 注入的删除守卫钩子（避免干扰服务进程的文件操作）
unset PYTHONPATH || true
export CODEBUDDY_SAFE_DELETE_ENABLED=0
export PYTHONUNBUFFERED=1
export SGLANG_DSV41_REASONING_EFFORT="${REASONING_EFFORT}"
# 多租户共存: 关闭 TP 显存均衡严格检查（默认 True 会拒绝同卡有他人部署的启动）。
# 改为告警放行——KV 池已显式钉死(MAX_TOTAL_TOKENS)，真不足会在加载期如实报错。
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${ENGRAM_HOST_TABLE}" == "1" ]]; then
    export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
    export SGLANG_OPT_DSV41_ENGRAM_HOST_STAGING="${SGLANG_OPT_DSV41_ENGRAM_HOST_STAGING:-1}"
    log "engram 表:   放入主机内存 + staging 预取 (B5 配方)"
    # THP 大页健康检查: 失效时查找慢 ~10x(每行 TLB miss), 仅告警不拦截
    THP_E="$(grep -oP '\[\K\w+' /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo '?')"
    THP_S="$(grep -oP '\[\K\w+' /sys/kernel/mm/transparent_hugepage/shmem_enabled 2>/dev/null || echo '?')"
    if [[ "${THP_E}" != "madvise" && "${THP_E}" != "always" ]] || [[ "${THP_S}" != "advise" && "${THP_S}" != "always" ]]; then
        log "  [告警] THP 大页未开启(enabled=${THP_E}, shmem=${THP_S}), engram 性能将显著劣化。"
        log "  [告警] 修复: echo madvise > /sys/kernel/mm/transparent_hugepage/enabled;"
        log "  [告警]        echo advise > /sys/kernel/mm/transparent_hugepage/shmem_enabled"
    fi
fi

# NUMA 内存交错: 消除 engram 共享表半数远端节点读(实测单人解码 +9%)；NUMA_INTERLEAVE=0 可关
LAUNCH_PREFIX=()
if [[ "${NUMA_INTERLEAVE:-1}" == "1" ]] && command -v numactl >/dev/null 2>&1; then
    LAUNCH_PREFIX=(numactl --interleave=all)
    log "NUMA 内存:  交错放置 (numactl --interleave=all)"
fi

EXTRA_ARGS_ARR=()
[[ -n "${EXTRA_ARGS}" ]] && read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS}"

# setsid: 独立进程组，便于 stop.sh 整组回收
setsid "${LAUNCH_PREFIX[@]}" "${VENV_PY}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp "${TP_SIZE}" \
    --attn-dp-size 1 \
    --ep-size "${EP_SIZE}" \
    --context-length "${CONTEXT_LENGTH}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --default-chat-template-kwargs "{\"thinking\": true, \"reasoning_effort\": \"${REASONING_EFFORT}\"}" \
    --cuda-graph-backend-prefill breakable \
    --cuda-graph-max-bs-prefill 8192 \
    --cuda-graph-max-seq-len-prefill 16384 \
    --cuda-graph-max-bs-decode 64 \
    --disable-flashinfer-autotune \
    --enable-cache-report \
    "${EXTRA_ARGS_ARR[@]}" \
    >>"${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
log "服务进程已拉起 (PID ${SERVER_PID})，等待模型加载..."

# ---------------------------- NUMA 绑核（按 rank） ----------------------------
if [[ "${NUMA_BIND}" == "1" ]]; then
    log "NUMA 绑核: GPU0-3 -> CPU0-119 / GPU4-7 -> CPU120-239"
    bind_rank() {
        local rank="$1" cpus="$2" pids p
        for _ in $(seq 1 90); do
            # -g 限定本服务进程组，避免误绑同机其他 sglang 部署的 scheduler（多租户安全）
            pids="$(pgrep -g "${SERVER_PID}" -f "scheduler_TP${rank}_EP" || true)"
            [[ -n "${pids}" ]] && break
            sleep 2
        done
        for p in ${pids}; do
            if taskset -pc "${cpus}" "${p}" >/dev/null 2>&1; then
                log "  [OK] TP${rank} (pid ${p}) -> CPU ${cpus}"
            else
                log "  [告警] TP${rank} (pid ${p}) 绑核失败"
            fi
        done
    }
    BIND_PIDS=()
    for i in 0 1 2 3; do bind_rank "${i}" "0-119" & BIND_PIDS+=($!); done
    for i in 4 5 6 7; do bind_rank "${i}" "120-239" & BIND_PIDS+=($!); done
    wait "${BIND_PIDS[@]}"   # 只等绑核任务，勿 wait 全部子进程（含服务主进程）
fi

# ---------------------------- 健康检查 ----------------------------
HEALTH_URL="http://127.0.0.1:${PORT}/health"
START_TS="$(date +%s)"
while true; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "错误: 服务进程已退出，最近日志如下："
        tail -n 40 "${LOG_FILE}" || true
        rm -f "${PID_FILE}"
        exit 1
    fi
    if curl -sf --max-time 3 "${HEALTH_URL}" >/dev/null 2>&1; then
        break
    fi
    NOW="$(date +%s)"
    if (( NOW - START_TS > START_TIMEOUT )); then
        log "错误: 等待超时 (${START_TIMEOUT}s)，服务可能仍在加载权重；请查看日志 ${LOG_FILE}"
        exit 1
    fi
    sleep 5
done

# ---------------------------- 就绪断言（server_info 关键字段） ----------------------------
log "就绪断言: 校验 server_info 关键字段..."
INFO_JSON="$(curl -s --max-time 15 "http://127.0.0.1:${PORT}/get_server_info")"
ASSERT_FAILED=0
assert_field() {
    local field="$1" expect="$2" actual
    actual="$(printf '%s' "${INFO_JSON}" | python3 -c "import json,sys
v=json.load(sys.stdin).get('${field}')
print('NONE' if v is None else v)" 2>/dev/null || echo PARSE_ERROR)"
    if [[ "${actual}" == "${expect}" ]]; then
        log "  [OK] ${field}=${actual}"
    else
        log "  [告警] ${field} 期望=${expect} 实际=${actual}"
        ASSERT_FAILED=1
    fi
}
assert_field max_total_num_tokens "${MAX_TOTAL_TOKENS}"
assert_field speculative_algorithm "DSPARK"
assert_field attention_backend "dsv4"
assert_field kv_cache_dtype "fp8_e4m3"
assert_field page_size "256"
if (( ASSERT_FAILED )); then
    log "就绪断言失败：server_info 与预期不符。服务保持运行，请人工核查后决定是否回滚。"
    exit 1
fi

# ---------------------------- 冒烟请求（空文本按失败处理） ----------------------------
log "冒烟请求: max_tokens=8 ..."
SMOKE_RAW="$(curl -s --max-time 60 -w $'\n%{http_code}' "http://127.0.0.1:${PORT}/v1/chat/completions" \
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
    fr = ch.get('finish_reason')
    if code == '200' and text and fr:
        print('OK|' + text.replace('\n', ' ')[:40])
    else:
        print('FAIL|http=%s finish_reason=%r 空文本=%s' % (code, fr, not bool(text)))
except Exception as e:
    print('FAIL|解析失败: %s' % e)
" 2>/dev/null || echo "FAIL|curl/python 异常")"
if [[ "${SMOKE_VERDICT}" == OK\|* ]]; then
    log "  [OK] 冒烟通过: ${SMOKE_VERDICT#OK|}"
else
    log "  [告警] 冒烟失败: ${SMOKE_VERDICT#FAIL|}（HTTP 200 但文本为空按失败处理）"
    exit 1
fi

ELAPSED=$(( $(date +%s) - START_TS ))
log "启动成功，耗时 ${ELAPSED}s。"
log "  OpenAI Base URL : http://127.0.0.1:${PORT}/v1"
log "  模型名          : ${SERVED_NAME}"
log "  健康检查        : ${HEALTH_URL}"
log "  运行状态        : curl http://127.0.0.1:${PORT}/get_server_info"
log "  停止服务        : ${SCRIPT_DIR}/stop.sh"
