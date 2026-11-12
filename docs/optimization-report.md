# DeepSeek-V4.1-Flash-1017 · 优化报告（相对官方原版）

> 日期：2026-11-11
> 对比对象：**官方原版 = sglang 0.5.23（官方发布版）+ 我们最初的 `start.sh`**
> （TP8 + EP8、engram 表驻主机内存、chunk 8192、max-running 32）
> 对比数据：同一台 8×H20-3e、同一套压测套件、同一批 prompt（salt 固定）。
> 每组数字都是多轮重复后的稳定值；异常轮次已剔除（见 §6）。

---

## 0. 一页结论

| 场景 | 官方原版 | **最终版** | 提升 |
|---|---|---|---|
| 冷预填充 128K 首字 | 11.45 s | **1.84 s** | **6.2×** |
| 冷预填充 300K 首字 | 39.13 s | **4.60 s** | **8.5×** |
| 5 人同时 300K 冷开场 | 193.3 s | **20.6 s** | **9.4×** |
| 5 人同时 30K 冷开场 | 10.68 s | **2.23 s** | 4.8× |
| 真实 agent 画像（30K，chat+tools+thinking）首字 | 2.09 s | **0.48 s** | 4.4× |
| 同上，解码速度 | 249 tok/s | **520 tok/s** | 2.1× |
| 同上，单轮总耗时 | 4.34 s | **1.48 s** | 2.9× |
| 单人 1K 首字 | 0.187 s | **0.059 s** | 3.2× |
| 长预填充挤占下的短请求首字（队头阻塞） | 13.15 s | **1.99 s** | 6.6× |
| 8 并发出字吞吐 | 875 tok/s | **1,943 tok/s** | 2.2× |
| 32 并发出字吞吐 | 1,254 tok/s | **3,319 tok/s** | 2.6× |
| 解码每步验证耗时（引擎真实开销） | 13.3 ms | **6.83 ms** | 1.9× |
| 单卡显存峰值 | 71.7 GB | **63.2 GB** | −12% |
| 启动到 READY | 850 s | **435 s（冷）／315 s（热）** | 2.0–2.7× |
| 识图能力 | ✅ | ✅ **保留** | — |

一句话：**长输入首字快 6–9 倍，解码快 2 倍，吞吐翻倍，显存更省，识图不丢。**

---

## 1. 起点：官方原版是什么

| 项 | 官方原版（优化前） |
|---|---|
| 推理框架 | sglang **0.5.23 官方发布版**（`/models/envs/.../site-packages/sglang`） |
| 并行方式 | TP8 + **EP8**（384 专家按卡切分），`--attn-dp-size 1` |
| 预填充 | **无 context parallel**，每张卡各自算完整序列 |
| MoE 后端 | 框架默认（Hopper 上走 FlashInfer MXFP4 路径） |
| 投机解码 | DSpark，block=5，draft=6（与权重绑定） |
| 分块 / 并发 | `--chunked-prefill-size 8192`、`--max-running-requests 32` |
| engram 表 | 驻主机内存，`shared` 布局 + staging 预取 |
| 显存 / 上下文 | `mem-fraction 0.80`、`--max-total-tokens 3145728`、1M context |

注意：这个"原版"本身并不裸——它已经过一轮 B5 配方调优（engram 放主机内存省显存、
NUMA 交错、THP 大页）。所以下面的提升是**在已经调过一轮的基础上**再拿到的。

---

## 2. 最终版做了什么

分五层。**第 1–4 层来自开源分支 `antgroup/sglang` 的 `dsv41-flash-h20`；
第 5 层是我们自己补的**（这部分上游没有，或上游有 bug）。

### 2.1 代码层：换用针对 H20 优化的分支

- 仓库：`github.com/antgroup/sglang`，分支 `dsv41-flash-h20`，commit `6e6d5d9e13`
  （基线为社区 main `5884e4fe`，2026-10-27，共 10 个提交）
- **接入方式：`PYTHONPATH` 覆盖，不安装、不替换 venv**。环境里的 0.5.23 一个字节没改，
  随时可回滚
- 兼容性核查：分支的 `ServerArgs` 能完整解析我们的原启动参数；CUDA 13.0 / torch 2.14.1 /
  flashinfer 0.7.2 与分支 Dockerfile 一致；分支只改 `python/sglang`（167 个文件），
  不动依赖与 sgl-kernel

### 2.2 并行层：预填充 CP8（最大单项收益）

- `--enable-prefill-cp --cp-strategy interleave --enable-cp-decode-attn-tp`
- 把**一条长请求的 token 按 interleave 摊到 8 张卡**上算，而不是每张卡各算一遍
- 分支把 CP 的通信从 NCCL 改到 **NVLink 对称内存**：压缩器只投影本卡行再汇聚（尾部从
  "16384 行 GEMM + 全量汇聚"变成"2048 行 GEMM + 输出汇聚"）；MoE 输入由 copy engine
  拉取、按 rank 顺序 FP32 求和，省掉 all-reduce + 切片；engram 查表由**分片属主直写目标 rank**
- decode 侧用 `--enable-cp-decode-attn-tp`：不是切 token，而是把**不同请求的行**分给各卡
  （96 行起 head↔row 交换），主要改善并发场景

### 2.3 内核层

| 改动 | 作用 |
|---|---|
| `--moe-runner-backend humming` + `SGLANG_HUMMING_INPUT_QUANT_CONFIG` | 路由专家权重是 FP4，Hopper 没有 FP4 tensor core；Humming 在 kernel 内把 FP4 转 FP8（W4A8），权重仍是 4 bit、计算用 FP8 算力 |
| `SGLANG_OPT_DSV41_SM90=1` | 分支总开关（14 个子系统：mhc / moe / fp8_gemm / attn_proj / lm_head / startup / indexer / cp / prefill_graph / decode / mix_decode / runtime / pd / epd） |
| `SGLANG_OPT_FP8_WO_A_GEMM=1` | wo_a 保持 FP8，inverse RoPE 融进舍入 kernel |
| 稀疏 prefill / 晚层跳过（`--enable-decoder-swa-bounded-replay`） | 第 21–39 层复用第 20 层 KV，每个 chunk 只算最后一个 128-token 窗口；**这是近似算法**，见 §6 |
| prefill CUDA graph | 分支按"本卡精确行数"懒捕获，消掉每层约 4 ms 的 host 开销 |

### 2.4 启动层

- 分支关闭 DeepGEMM "全 M 预编译"，改为**抽样预热 + 随代码发布 JIT 清单（639 条）**，
  32 进程并行编译：启动从 850 s 降到 435 s（冷页缓存）/ 315 s（热）
- 权重加载 368 s → 197 s

### 2.5 我们自己的增量（**上游没有**）

| # | 内容 | 为什么必须做 |
|---|---|---|
| 1 | **vision + CP 补丁**（6 行，`models/deepseek_v4.py`） | 上游用守卫禁止"视觉塔 + 预填充 CP"同时启用，等于**要么识图、要么快 6 倍**。模型层其实已有 CP 感知的图像 token 处理（`cp_image_token_id`），我们放开守卫并做了严格验证 → **两者同时拿到** |
| 2 | **修复启动挂死** | 分支 `rust/rust-toolchain.toml` 钉 Rust 1.92，视觉处理器初始化触发 rustup **在线下载工具链**，8 个 rank 全堵在 barrier，启动永久挂起（实测 10 min+ 无进展）。用国内镜像装好 1.92 + 预编译 `sglang-mm` 扩展缓存 |
| 3 | **修复 NUMA 绑核静默失效** | 脚本按 `scheduler_TP{N}_EP` 找进程，但 CP 模式下进程名是 `scheduler_ATTN_CP{N}_TP{N}`，匹配数为 0，绑核从切 CP 起就没生效过 |
| 4 | **参数配方** | `mem-fraction 0.84`、`chunk 16384`、`max-running 64`、`cuda-graph-max-bs-decode 64`、engram `per_rank` 布局（原为 `shared`）、bounded replay 默认开 |
| 5 | **压测与回归工程化** | 12 场景压测套件（salt 固定、`ignore_eos` 定长、服务端 usage 计量）、带噪声阈值的对比工具、32 用例识图回归、暖机脚本 |

---

## 3. 各项贡献了多少（消融实测）

不是所有优化都一样重要。按实测拆开：

| 优化项 | 实测效果 | 证据 |
|---|---|---|
| **分支整体（内核 + 启动）** | 128K 首字 11.45 s → 4.59 s（**2.5×**） | 官方原版 vs "分支 + 视觉、无 CP" |
| **预填充 CP8** | 128K 首字 4.59 s → 1.85 s（**2.5×**）；预填充速率 2.2–2.7×（30K 28.5K→62.7K、128K 27.0K→65.7K tok/s）；解码几乎无差，并发解码 +10~20% | 同分支代码，仅切 `--enable-prefill-cp` |
| **bounded replay（晚层跳过）** | 预填充速率 **+46~51%**（关掉后 62.7K→33.9K tok/s）；TTFT 翻倍变慢；T32 吞吐 −12.5% | 同代码只切 `--no-enable-decoder-swa-bounded-replay` |
| **chunk 16384 → 32768 / 65536** | 首字仅 +3%（噪声内），但 T32 吞吐 **−5~−8%** → **不采纳**，保持 16384 | 逐档重启实测 |
| **prefill CUDA graph（`full`）** | **不可用**：DSV4 attention backend 未实现 `ForwardMode.EXTEND` 的图捕获，直接 `NotImplementedError` | 启动即崩 |
| **vision + CP 补丁** | 从"识图但预填充慢 2.4×"变成"**识图且不损失性能**" | 视觉无 CP 4.59 s → 视觉+CP8 1.85 s |
| **暖机** | 消除启动后首批重请求的额外等待 | 冷启动首个 5×300K 曾 127 s → 暖机后 8/8 轮 20.5 s |

累计效果（乘法关系，非简单相加）：
`官方原版 11.45 s → 分支 4.59 s（2.5×） → 加 CP8+补丁 1.84 s（6.2×）`

---

## 4. 参数级差异清单

| 参数 | 官方原版 | 最终版 | 说明 |
|---|---|---|---|
| 代码来源 | sglang 0.5.23（官方包） | 分支 `dsv41-flash-h20` @`6e6d5d9e13`（`PYTHONPATH` 覆盖） | 可回滚 |
| `--tp` / `--ep-size` | 8 / **8** | 8 / **1**（纯 TP） | 分支实测配方 |
| `--enable-prefill-cp` | 无 | **开**（interleave） | 核心收益 |
| `--enable-cp-decode-attn-tp` | 无 | **开** | 并发解码 |
| `--moe-runner-backend` | 默认 | **humming** | W4A8 |
| `--enable-decoder-swa-bounded-replay` | 无 | **开**（ratio 0.05） | 近似算法 |
| `--chunked-prefill-size` | 8192 | **16384** | 扫参后取此值 |
| `--max-running-requests` | 32 | **64** | 配合 CP |
| `--mem-fraction-static` | 0.80 | **0.84** | 显存仍下降 |
| `--cuda-graph-max-bs-decode` | 64 | 64 | 不变 |
| `--language-model-only` | 无（加载视觉塔） | **无**（同样加载视觉塔） | 识图保留 |
| engram 布局 | `shared` | **`per_rank`** | 属主直写 |
| engram 位置 | 主机内存 | 主机内存 | 不变 |
| 环境变量 | `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` 等 | 追加 `SGLANG_OPT_DSV41_SM90=1`、`SGLANG_OPT_FP8_WO_A_GEMM=1`、`SGLANG_HUMMING_INPUT_QUANT_CONFIG`、`SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank` | |
| 启动脚本 | `start.sh` | `start-opt.sh`（默认配置） | `start.sh` 保留为回滚 |

唯一显式有损开关 `SGLANG_OPT_DSV41_SM90_LOSSY=fp8_lm_head_target` **未启用**。

---

## 5. 完整效果表（12 场景）

| 场景 | 指标 | 官方原版 | 最终版 | 倍数 |
|---|---|---|---|---|
| D1_1K（单请求 1K） | TTFT | 0.187 s | **0.059 s** | 3.2× |
| | 解码 | 320 tok/s | **625 tok/s** | 2.0× |
| | 每步验证 | 13.3 ms | **6.83 ms** | 1.9× |
| D1_30K | TTFT | 2.187 s | **0.481 s** | 4.5× |
| | 每步验证 | 12.81 ms | **6.92 ms** | 1.9× |
| D1_128K | TTFT | 11.471 s | **1.843 s** | 6.2× |
| | 解码 | 301 tok/s | **666 tok/s** | 2.2× |
| D1_300K | TTFT | 39.133 s | **4.601 s** | 8.5× |
| | 解码 | 108 tok/s | **320 tok/s** | 3.0× |
| T8_1K（8 并发） | 出字吞吐 | 875 tok/s | **1,943 tok/s** | 2.2× |
| T32_1K（32 并发） | 出字吞吐 | 1,254 tok/s | **3,319 tok/s** | 2.6× |
| P_128K | TTFT | 11.453 s | **1.847 s** | 6.2× |
| P_5x30K（5 人同开 30K） | TTFT p50 | 10.678 s | **2.233 s** | 4.8× |
| P_5x300K（5 人同开 300K） | TTFT p50 | 193.252 s | **20.570 s** | 9.4× |
| | 单轮总耗时 | 196.07 s | **21.27 s** | 9.2× |
| HOL（1×128K 挤 31×1K） | 短请求 TTFT | 13.148 s | **1.993 s** | 6.6× |
| CACHE（30K 前缀复用） | 命中后 TTFT | 0.506 s | **0.287 s** | 1.8× |
| CHAT_30K（真实 agent） | TTFT | 2.091 s | **0.475 s** | 4.4× |
| | 解码 | 249 tok/s | **520 tok/s** | 2.1× |
| | 单轮总耗时 | 4.34 s | **1.48 s** | 2.9× |

预填充速率（最能说明问题的单一指标）：

| 上下文 | 官方原版 | 最终版 | 倍数 |
|---|---|---|---|
| 30K | 13,731 tok/s | **62,453 tok/s** | 4.5× |
| 128K | 10,810 tok/s | **67,296 tok/s** | 6.2× |
| 300K | 7,646 tok/s | **65,033 tok/s** | 8.5× |

注意最后一行：原版的预填充速率**随上下文变长而下降**（13.7K→7.6K），最终版**基本恒定**
（62–67K）。这意味着上下文越长，收益越大 —— 这正是 300K 场景能到 8.5 倍的原因。

---

## 6. 代价与边界

| 项 | 说明 |
|---|---|
| **bounded replay 是近似算法** | 官方文档明确"与完整 prefill 数值不等价"，RL rollout（prompt logprob 需与训练端一致）应关闭。关掉的代价：预填充速率 −46~51%。文章评测差值在 ±1 内（GSM8K −0.30、BFCL −0.17、IFEval +0.92） |
| **依赖非社区主线分支** | `antgroup/sglang` 分支仅 10 个提交、面向 H20。升级时需把这 10 个提交移植到新基线，并重新打我们的 vision 补丁 |
| **解码 tok/s 受接受长度影响** | 某类内容可能出现"每步耗时减半但接受长度也下降"，净解码持平（如 D1_30K：349.9→331.0 tok/s，因接受长度 4.38→2.39）。引擎本身已快 1.9×（每步 13.3→6.92 ms），判定解码优化要看每步耗时而非 tok/s |
| **5×300K 曾有环境异常态** | 历史上出现过的 124–133 s（6 倍劣化）经排查与配置无关，主机重启后消失，现稳定 20.5 s（8/8 轮）。若复现，优先怀疑主机状态，见docs/baseline-benchmark-report.md§7.7 |
| **启动仍需 5–7 分钟** | 475 GB 权重冷加载是下限；JIT 缓存预热后 315 s |
| **识图已验证但非全场景** | 32 个程序化用例 + 文章 4 张原图人工评估通过；未覆盖自然照片、手写体、多页文档 |

---

## 7. 复现与回滚

```bash
cd scripts

# 最终版（默认）
./start-opt.sh

# 回滚到官方原版
./stop.sh && ./start.sh

# 校验收效（与历史基线对比，自动标注改善/劣化并考虑噪声阈值）
cd bench && python bench_suite.py --tag a && python bench_suite.py --tag b && python bench_compare.py a b

# 识图回归（任何并行/视觉相关改动后必跑）
python bench/vision_check.py     # 32 用例
```

关键文件：

| 文件 | 用途 |
|---|---|
| `start-opt.sh` | 最终版启动脚本（默认视觉 + CP8，自动暖机） |
| `start.sh` / `stop.sh` | 回滚到官方原版 / 停机 |
| `patches/0001-allow-vision-with-prefill-cp.patch` | vision+CP 补丁，附 README 与回环验证 |
| `bench/bench_suite.py`、`bench_compare.py` | 12 场景压测 + 带噪声阈值的对比 |
| `bench/vision_check.py` | 32 用例识图回归 |
| `bench/warmup.py` | 启动暖机 |
| `docs/baseline-benchmark-report.md` | 全部原始数据、消融、排查记录 |

---

## 8. 归因说明（哪些是别人的、哪些是我们的）

**诚实起见必须区分**：

- **性能提升的主体（约 80%）来自开源分支**：CP8、NVLink 对称内存通信、Humming W4A8、
  FP8 GEMM、稀疏 prefill + 晚层跳过、prefill graph、JIT 清单启动。这些是蚂蚁团队的工作，
  我们做的是选型、适配、验证。
- **我们的增量（约 20%，但决定了可用性）**：
  1. 把分支跑起来（修复 rustup 挂死）；
  2. **让识图和 6 倍加速同时存在**（上游认为二者互斥）；
  3. 修复 NUMA 绑核静默失效；
  4. 参数配方与消融验证（chunk 调大无收益、graph 不可用、replay 必须开）；
  5. 建立可复跑的压测与回归体系，让"优化"可被证伪。

没有第 2 项，这个部署要么没有识图、要么只有 1/2.4 的速度；没有第 5 项，就无法判断
哪些改动真的有效。

---

## 9. NUMA 内存交错：严格 A/B 与「设为默认」（2026-11-12）

### 9.1 这条优化的归属

`numactl --interleave=all` **不是参考文章里的方案**。对原文（知乎 37 万字节 HTML）逐词检索：
`numa` 0 次、`numactl` 0 次、`远端`/`跨节点` 0 次；`interleave` 的 5 次出现全部是
`--cp-strategy interleave`（CP 切分策略）。

文章提出的是两件相邻的事：

- "**Engram 表放到主机内存并按 rank 切分**"（复现章节的环境变量说明）→ 我们的 `ENGRAM_HOST_TABLE=1` + `LAYOUT=per_rank` 来自这里；
- "输入 embedding 和 Engram 查表同样由**分片属主直接写给目标 rank**"（CP8 通信优化）→ 文章的解法是**改访问路径**。

而交错是 **B5 配方（2026-11-08）** 的配套手段，解决的是同一问题的另一半：**改页分布**，
让每个 rank 同时用上两个 NUMA 节点的内存带宽。原文没有覆盖这条。

### 9.2 为什么之前"配了但没生效"

`sglang` 会分两层把 worker 的内存拉回本地，父进程继承的 interleave 永远被覆盖：

| 路径 | 触发 | `numa_maps` 表现 |
|---|---|---|
| `SGLANG_NUMA_BIND_V2`（默认 True） | spawn worker 时套 `numactl --membind=<本地节点>` | `bind:0` / `bind:1` |
| `SGLANG_AUTO_NUMA_BIND`（默认 True） | worker 内 `numa_set_preferred(本地节点)` | `prefer:0` / `prefer:1` |

只关 V2 会退到 `prefer`（仍然本地）；**两个都关**，worker 才真正继承父进程的 `interleave`
（实测 `numa_maps` 里 3769/3745 条映射全部为 `interleave`，内存 12928 MB / 12894 MB ≈ 50:50）。

### 9.3 严格 A/B（2026-11-12，DS 停机 26 分钟，两组各重启一次）

方法：同语料、同 prompt 长度、`temperature=0`、`ignore_eos`；TPOT 用 `usage.completion_tokens`
计算（按 SSE chunk 计数会因多 token/块而低估 4 倍，已修正）；每组两轮，1K 上下文 256 token ×12 次、
30K 上下文 128 token ×5 次，取中位与四分位；两组都是**全新启动后立即测**。

| 指标 | A：本地 bind（框架默认） | B：真交错 | 差异 |
|---|---|---|---|
| 1K TPOT 中位（r1 / r2） | 1.61 / 1.60 ms（623 / 624 tok/s） | **1.55 / 1.54 ms（645 / 651 tok/s）** | **B 快 3.7–3.8%** |
| 1K 四分位 | p25 1.55–1.57 / p75 1.64–1.73 | p25 1.51–1.53 / p75 1.55–1.70 | 中位清晰分离 |
| 30K TPOT 中位（r1 / r2） | 2.39 / 3.12 ms | 2.45 / 3.24 ms | 无差异（被同卡生产流量主导） |

**结论：交错在 1K 单流上确有 +4%，30K 无影响。** 与我此前"交错必然更差"的预判相反——
错在假设主机内存带宽不是瓶颈；engram 查表是高频 gather，单节点带宽会打满，
交错提供双节点聚合带宽，收益大于远端延迟损失。这也与当年"单人解码 +9%"的方向一致
（幅度不同：那时分支的 engram 属主直写尚未介入）。

### 9.4 已设为默认

`start-opt.sh` 在 `NUMA_INTERLEAVE=1`（默认）时自动：

```bash
LAUNCH_PREFIX=(numactl --interleave=all)
export SGLANG_NUMA_BIND_V2=0      # 关掉框架的 --membind
export SGLANG_AUTO_NUMA_BIND=0    # 关掉框架的 numa_set_preferred
```

**CPU 亲和不受影响**：关掉框架绑定后，仍由脚本自己的 `NUMA_BIND=1` 在启动后 `taskset` 补绑；
实测 rank0 `Cpus_allowed_list=0-119`、rank4 `=120-239`，与交错策略同时成立。

回滚：`NUMA_INTERLEAVE=0 ./start-opt.sh`（回到框架默认的本地 bind，1K 单流约 −4%）。
