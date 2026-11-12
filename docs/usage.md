# DeepSeek-V4.1-Flash-1017 使用文档（客户端接入）

> 适用对象：OpenCode 等 CodingAgent / 任意 OpenAI 兼容客户端
> 服务地址：`http://<本机IP>:30010/v1`　模型名：`DeepSeek-V4.1-Flash-1017`

---

## 1. 思考模式与 `reasoning_effort`

服务端已设置部署级默认（`--default-chat-template-kwargs '{"thinking": true, "reasoning_effort": "high"}'`）：
**不传任何参数也会走 thinking 通道**，思维链与正文自动分离：

- `message.reasoning_content` → 思维链（可单独渲染/折叠）
- `message.content` → 最终回答
- `message.tool_calls` → 工具调用（OpenAI 标准格式）

| 请求 | reasoning_content | content |
|---|---|---|
| **不传（2026-11-04 起的默认）** | ✅ 有思维链（2026-11-12 实测 582 字符） | 纯最终答案 |
| `{"thinking": false}` 或 `"reasoning_effort": "none"` | 空 | 直接回答（实测 382 字符） |
| `"reasoning_effort": "high"` | 467 字符（2026-11-03 样本） | 纯最终答案（reasoning_tokens: 172） |

> ⚠️ **历史坑（已修复）**：2026-11-04 之前服务端没有设默认值，请求不带 `reasoning_effort` 时
> 模型不走 thinking 通道，思维链会直接写进 `content`，污染回答与工具参数解析。
> 现在默认已补齐；仅为兼容旧客户端，仍建议显式传 `reasoning_effort`。

### 1.1 思考强度档位与默认行为（权威口径，2026-11-03 实验确认）

模型编码器（权重目录 `encoding/encoding.py`）通过提示词行
`Reasoning Effort: N (range 1-100, ...)` 控制思考深度，**原生档位仅 3 个**：

| 原生档 | 预算 | | API 传法（协议层 8 档，上游折叠） | 落到预算 |
|---|---|---|---|---|
| `low` | **50** | | `minimal` / `low` | 50 |
| `high` | **75**（`DEFAULT_REASONING_EFFORT`） | | `medium` / `high` / `xhigh` | 75 |
| `max` | **100** | | `ultra` / `max` | 100 |
| 整数 **1~100** | **N**（唯一精确控制路径） | | 浮点 `0.0~0.99`（sglang 扩展） | ×100 取整 |

**默认行为（2026-11-04 起本部署配置）**：

服务端通过 `--default-chat-template-kwargs '{"thinking": true, "reasoning_effort": "high"}'`
设置了部署级默认值——**用户未传时自动补 `thinking=on + effort=high`，显式值始终优先**；
默认档与权重 `encoding.py` 的 `DEFAULT_REASONING_EFFORT="high"`（=75）一致：

| 传法 | 是否思考 | 生效强度 |
|---|---|---|
| 什么都不传 | ✅（部署默认开） | **75（默认 high，与模型缺省一致）** |
| `"reasoning_effort": "low"` | ✅ | 50 |
| `"reasoning_effort": 30` | ✅ | 30 |
| `"thinking": true`（不传 effort） | ✅ | 75（模型编码器缺省 high） |
| `"reasoning_effort": "none"` | ❌ 显式关闭 | 无 |
| `"thinking": false` | ❌ 显式关闭 | 无 |

### 1.2 2026-11-12 复验（同一道多步推理题，temperature=0）

| 传法 | 思维链字符数 | 说明 |
|---|---|---|
| `{"thinking": false}` | 0 | 直接回答 |
| `"reasoning_effort": "low"` | 345 | 预算 50 |
| `"reasoning_effort": "high"`（默认） | 796 | 预算 75 |
| `"reasoning_effort": "max"` | 816 | 预算 100 |
| `"reasoning_effort": 100`（整数） | 1192 | 唯一精确控制路径 |
| `"reasoning_effort": "medium"` / `"xhigh"` | 609 / 609 | 协议层折叠到 75（与 high 同档，见 §1.1 映射表） |

> 复验结论：§1.1 的档位映射表与线上行为一致——`medium`/`xhigh` 不是"非法值"，而是
> 被协议层折叠到 `high(75)`，因此与 `high` 输出完全相同；需要更细的控制请传整数。

### 1.3 与另外两个模型的差异（速查）

| 模型 | 默认 | 强度可选值 | 默认强度 | 思维链字段 |
|---|---|---|---|---|
| **DeepSeek-V4.1-Flash（本模型）** | 思考开 | `low/high/max` 或 **整数 1–100**（协议层 8 档折叠） | **high(75)** | `reasoning_content` |
| Qwen3.8-Flash-Next | 思考开 | `xhigh/medium/low`（传错值 HTTP 400） | xhigh | `reasoning_content` |
| Qwen3.8-27B | 思考开 | `xhigh/medium/low`（传错值 HTTP 400） | xhigh | **`reasoning`**（vLLM 字段名） |

> 更换默认档：启动时 `REASONING_EFFORT=low ./start.sh`（支持 low/medium/high/max
> 或整数 1~100）。实验记录（部署同款 deepseek-recipe 编码链路离线渲染）：

```
A) thinking=True, 不传 reasoning_effort  -> Reasoning Effort: 75
B) thinking=True, reasoning_effort=null  -> Reasoning Effort: 75
C) thinking=False, 不传                  -> (无 Reasoning Effort 行)
线上复证: 完全不传的请求 reasoning_content 为空、reasoning_tokens=0（不思考）
```

附带效应：`max_tokens` 未指定时的默认上限为——不思考 8192 / 思考+`max` 131072 /
思考+其他档 65536。另注意 `/v1/responses` 流式回显档位限定 `{minimal,low,medium,high}`，
用 `ultra/xhigh/max` 时该接口回显的档位字段退化（`/v1/chat/completions` 不受影响）。

---

## 2. 可复制示例

### 2.1 带工具定义 + reasoning_effort 的请求（CodingAgent 典型调用）

```bash
curl -s http://<本机IP>:30010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "DeepSeek-V4.1-Flash-1017",
  "reasoning_effort": "low",
  "max_tokens": 2048,
  "messages": [
    {"role": "user", "content": "杭州今天多少度？帮我查一下"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询指定城市天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "城市名"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}'
```

预期响应（要点）：

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "content": "",
      "reasoning_content": "...",
      "tool_calls": [{
        "id": "call_xxx",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"杭州\"}"}
      }]
    }
  }]
}
```

### 2.2 工具结果回灌（多轮）

```json
{
  "model": "DeepSeek-V4.1-Flash-1017",
  "reasoning_effort": "low",
  "messages": [
    {"role": "user", "content": "杭州今天多少度？帮我查一下"},
    {"role": "assistant", "tool_calls": [
      {"id": "call_xxx", "type": "function",
       "function": {"name": "get_weather", "arguments": "{\"city\": \"杭州\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_xxx", "content": "{\"temp\": \"26°C\"}"}
  ],
  "tools": [ /* 同 2.1 的 tools 定义 */ ]
}
```

### 2.3 纯文本 + reasoning（无需工具）

```json
{
  "model": "DeepSeek-V4.1-Flash-1017",
  "reasoning_effort": "high",
  "max_tokens": 4096,
  "messages": [{"role": "user", "content": "你的问题"}]
}
```

---

## 3. 能力与限制速查

| 项 | 说明 |
|---|---|
| 上下文 | 1,048,576 token（`max_model_len`，输入+输出共享）；1M 实测通过 |
| 最大输出 | API 上限 1,048,576（实际 = 窗口 − prompt_tokens）；不传 `max_tokens` 默认：不思考 8192 / 思考+max 131072 / 思考其他 65536。注：官方托管 API 的 384K 输出上限是其服务策略，本地无此限制（实测 384K 请求可接受） |
| TTFT 量级（实测） | 30K≈2.1s、128K≈12s、500K≈86s、1M≈6 分钟（单请求冷）；同前缀续轮 <1s；并发冷轮按自身上下文排队（详见 `docs/validation.md`） |
| 流式 | `"stream": true`（SSE，`chat.completion.chunk`） |
| 工具调用 | OpenAI `tools` 协议；`finish_reason: tool_calls` |
| reasoning 分离 | 建议显式传 `reasoning_effort`；未传时服务默认 `high`+思考开（§1.1） |
| 思考强度 | 原生 `low`=50 / `high`=75 / `max`=100 + 整数 1~100；部署默认 `high`（与模型缺省一致），`none` 显式关闭 |
| 缓存命中统计 | `usage.prompt_tokens_details.cached_tokens`（OpenAI 标准字段，已开启 `--enable-cache-report`）；单请求命中率 = cached/prompt，聚合 = Σcached/Σprompt；流式在末尾 usage chunk 中同样携带 |
| 并发 | `max_running_requests=32`，超出排队 |
| 图片输入 | ✅ 支持（含多图/交错图块）；带图请求不复用前缀缓存（见验证文档） |
| 视频 / PDF / 音频输入 | ❌ 不支持（视频音频协议层存在但 dsv41 编码层拒绝；PDF 无 API 入口）——客户端预拆为文本/图片 |
| 结构化输出 | ✅ `response_format: json_schema`（strict，outlines 语法约束），实测输出符合 schema |
| 对话中系统消息 | ✅ 支持（中途 system 指令立即生效） |
| 原生联网搜索 | ❌ 不支持（web_search 需 EXA_API_KEY/浏览器 MCP，未配置）——用 tools 自定义搜索函数替代 |
| API Key | 服务未启用鉴权，任意非空值；如对非本机开放请前置网关鉴权 |

## 4. OpenCode 配置

```
Base URL : http://<本机IP>:30010/v1
Model    : DeepSeek-V4.1-Flash-1017
API Key  : 任意非空字符串
```

并在模型参数/请求配置中加入 `reasoning_effort`（如 `low`），否则思维链混入正文。
