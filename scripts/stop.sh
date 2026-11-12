#!/usr/bin/env bash
#
# DeepSeek-V4.1-Flash-1017 · SGLang 一键停止脚本
#
# 用法:
#   ./stop.sh          # 优雅停止服务（超时后强制终止）
#   FORCE=1 ./stop.sh  # 直接强制终止

set -euo pipefail

PORT="${PORT:-30010}"
FORCE="${FORCE:-0}"
STOP_TIMEOUT="${STOP_TIMEOUT:-60}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/server.pid"

log() { echo "[stop.sh] $*"; }

kill_group() {
    local pid="$1" sig="$2"
    # start.sh 使用 setsid 启动，进程组 ID == 主进程 PID
    kill "-${sig}" -- "-${pid}" 2>/dev/null || kill "-${sig}" "${pid}" 2>/dev/null || true
}

# ---------------------------- 找到目标进程 ----------------------------
TARGET_PID=""
if [[ -f "${PID_FILE}" ]]; then
    CAND="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${CAND}" ]] && kill -0 "${CAND}" 2>/dev/null; then
        TARGET_PID="${CAND}"
    else
        rm -f "${PID_FILE}"
    fi
fi

# 兜底：pid 文件缺失时，按端口/进程名查找
if [[ -z "${TARGET_PID}" ]]; then
    PORT_PID="$(ss -ltnpH "sport = :${PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
    if [[ -n "${PORT_PID}" ]]; then
        TARGET_PID="${PORT_PID}"
    else
        PORT_PID="$(pgrep -f "sglang.launch_server.*--port ${PORT}" | head -1 || true)"
        [[ -n "${PORT_PID}" ]] && TARGET_PID="${PORT_PID}"
    fi
fi

if [[ -z "${TARGET_PID}" ]]; then
    log "服务未在运行（PID 文件失效且端口 ${PORT} 无监听进程），无需停止。"
    rm -f "${PID_FILE}"
    exit 0
fi

log "停止服务进程组 (PID ${TARGET_PID}) ..."
MEM_BEFORE="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')"

if [[ "${FORCE}" == "1" ]]; then
    kill_group "${TARGET_PID}" KILL
else
    kill_group "${TARGET_PID}" TERM
    # 等待优雅退出（含 8 卡 NCCL/子进程回收）
    DEADLINE=$(( $(date +%s) + STOP_TIMEOUT ))
    while kill -0 "${TARGET_PID}" 2>/dev/null; do
        if (( $(date +%s) >= DEADLINE )); then
            log "优雅停止超时 (${STOP_TIMEOUT}s)，强制终止..."
            kill_group "${TARGET_PID}" KILL
            break
        fi
        sleep 2
    done
fi

# 清理残留的 sglang 工作进程（防止孤儿进程占卡）
sleep 2
for _ in 1 2 3; do
    LEFTOVER="$(pgrep -f "sglang.launch_server.*--port ${PORT}" || true)"
    [[ -z "${LEFTOVER}" ]] && break
    kill -KILL ${LEFTOVER} 2>/dev/null || true
    sleep 1
done

rm -f "${PID_FILE}"

# 残留断言（多租户安全）：以"本进程组无存活进程"为判据。
# 不再要求显存归零——本机为多租户（如 GPU4-7 驻有 Qwen 等他人部署），
# 全卡显存检查会把他人负载误报为残留；显存仅作信息性对比输出。
for _ in 1 2 3 4 5; do
    LEFTOVER="$(pgrep -g "${TARGET_PID}" 2>/dev/null || true)"
    [[ -z "${LEFTOVER}" ]] && break
    sleep 2
done
if [[ -n "${LEFTOVER}" ]]; then
    log "告警: 进程组 (PGID ${TARGET_PID}) 仍有残留: ${LEFTOVER}，强制清除..."
    kill -KILL ${LEFTOVER} 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        sleep 2
        LEFTOVER="$(pgrep -g "${TARGET_PID}" 2>/dev/null || true)"
        [[ -z "${LEFTOVER}" ]] && break
    done
fi
if [[ -z "${LEFTOVER}" ]]; then
    log "进程组残留检查: 通过（PGID ${TARGET_PID} 无存活进程）。"
else
    log "告警: 强制清除后仍有残留: ${LEFTOVER}，请人工检查。"
    exit 1
fi
MEM_AFTER="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')"
log "显存变化(信息性, 含同机他人部署): 停止前 [${MEM_BEFORE}] -> 停止后 [${MEM_AFTER}] MiB"

log "已完成。端口 ${PORT} 已释放。"
