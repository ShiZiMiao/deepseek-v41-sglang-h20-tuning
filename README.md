# DeepSeek-V4.1-Flash on 8×H20: SGLang 部署调优

在 8 张 H20-3e 上用 SGLang 部署 DeepSeek-V4.1-Flash 的一套**可复现**调优方案：长输入首字延迟降低 6–9 倍、解码吞吐翻倍、显存更省，同时**保留识图能力**。

> 非官方项目。性能提升的主体（约 80%）来自开源分支 [`antgroup/sglang`](https://github.com/antgroup/sglang) 的 `dsv41-flash-h20`；本仓库提供的是适配、补丁、参数配方和验证体系。详见 [归因](#归因哪些是别人的哪些是本仓库的)。

```
官方 0.5.23 + 朴素启动参数  →  本方案
128K 冷预填充首字  11.45s  →  1.84s    (6.2×)
300K 冷预填充首字  39.13s  →  4.60s    (8.5×)
5 并发 300K 冷开场 193.3s  →  20.6s    (9.4×)
8 并发出字吞吐     875     →  1,943 tok/s
单卡显存峰值       71.7GB  →  63.2GB
启动到 READY       850s    →  435s (冷) / 315s (热)
识图               ✅      →  ✅
```

## 实测对比

同一台机器、同一套压测套件、同一批 prompt（salt 固定），多轮重复取稳定值。

| 场景 | 官方原版 | 本方案 | 提升 |
|---|---|---|---|
| 冷预填充 128K 首字 | 11.45 s | **1.84 s** | 6.2× |
| 冷预填充 300K 首字 | 39.13 s | **4.60 s** | 8.5× |
| 5 人同时 300K 冷开场 | 193.3 s | **20.6 s** | 9.4× |
| 5 人同时 30K 冷开场 | 10.68 s | **2.23 s** | 4.8× |
| 真实 agent 画像（30K，chat+tools+thinking）首字 | 2.09 s | **0.48 s** | 4.4× |
| 同上，解码速度 | 249 tok/s | **520 tok/s** | 2.1× |
| 同上，单轮总耗时 | 4.34 s | **1.48 s** | 2.9× |
| 单人 1K 首字 | 0.187 s | **0.059 s** | 3.2× |
| 长预填充挤占下的短请求首字（队头阻塞） | 13.15 s | **1.99 s** | 6.6× |
| 8 并发出字吞吐 | 875 tok/s | **1,943 tok/s** | 2.2× |
| 32 并发出字吞吐 | 1,254 tok/s | **3,319 tok/s** | 2.6× |
| 解码每步验证耗时 | 13.3 ms | **6.83 ms** | 1.9× |
| 单卡显存峰值 | 71.7 GB | **63.2 GB** | −12% |
| 启动到 READY | 850 s | **435 s（冷）/ 315 s（热）** | 2.0–2.7× |

全部原始数据、消融实验和排查记录见 [`docs/optimization-report.md`](docs/optimization-report.md) 与 [`docs/baseline-benchmark-report.md`](docs/baseline-benchmark-report.md)。

## 方案构成

| 层 | 内容 |
|---|---|
| 代码 | 用 `PYTHONPATH` 覆盖加载分支，**不安装、不替换 venv**，环境里的 0.5.23 一个字节没改，随时可回滚 |
| 并行 | 预填充 CP8（`--enable-prefill-cp --cp-strategy interleave`）+ 解码侧 `--enable-cp-decode-attn-tp` |
| 内核 | `--moe-runner-backend humming`（FP4 权重 → FP8 计算，W4A8）、`SGLANG_OPT_DSV41_SM90=1`、bounded replay 晚层跳过、prefill CUDA graph |
| 启动 | JIT 清单 + 抽样预热替代 DeepGEMM 全量预编译，32 进程并行编译 |
| **本仓库增量** | vision + CP 补丁、rustup 在线下载导致的启动挂死修复、NUMA 绑核静默失效修复、参数配方、压测/回归工程化 |

## 目录

```
patches/0001-allow-vision-with-prefill-cp.patch   放开「视觉塔 + 预填充 CP」互斥的 6 行补丁
scripts/start-opt.sh      优化版启动（默认视觉 + CP8，自动暖机、NUMA 绑核）
scripts/start.sh          官方原版基线启动，用于回滚对照
scripts/stop.sh           停机
bench/bench_suite.py      12 场景压测（salt 固定、ignore_eos 定长、服务端 usage 计量）
bench/bench_compare.py    两次结果对比，带噪声阈值与改善/劣化标注
bench/vision_check.py     识图回归（程序化生成 ground truth）
bench/sweep.py            参数扫描
bench/warmup.py           启动暖机
docs/                     优化报告 / 基线性能报告 / 部署 / 使用 / 验证
```

## 环境要求

| 项 | 值 |
|---|---|
| GPU | 8× NVIDIA H20-3e（141 GB），NVLink |
| 驱动 / CUDA | 595.91.07 / CUDA 13.2 |
| Python | 3.13 |
| sglang | 0.5.23（venv 内，作为回滚基线） |
| 优化分支 | `antgroup/sglang` @ `dsv41-flash-h20` (`6e6d5d9`) |
| 模型 | DeepSeek-V4.1-Flash-1017（权重不入库） |

## 快速开始

```bash
# 1. 取优化分支（只读引用，不装进 venv）
git clone -b dsv41-flash-h20 https://github.com/antgroup/sglang.git /opt/sglang-dsv41

# 2. 打补丁：允许视觉塔与 prefill CP 同时启用
cd /opt/sglang-dsv41 && git apply /path/to/repo/patches/0001-allow-vision-with-prefill-cp.patch

# 3. 按实际路径覆盖变量（脚本全部支持环境变量）
export MODEL_PATH=/models/weights/DeepSeek-V4.1-Flash-1017
export VENV_PY=/opt/venvs/sglang/bin/python
export BRANCH_ROOT=/opt/sglang-dsv41

# 4. 先干跑确认命令，再真启动
DRY_RUN=1 ./scripts/start-opt.sh
./scripts/start-opt.sh

# 5. 回滚到官方原版
./scripts/stop.sh && ./scripts/start.sh
```

常用开关：

```bash
PROFILE=stock ./scripts/start-opt.sh     # 只开 CP，不换代码
PARALLEL=ours ./scripts/start-opt.sh     # 保留 EP8 + attn-dp1，只加 CP
LANGUAGE_ONLY=1 ./scripts/start-opt.sh   # 只加载语言模型（无识图）
CP=0 BOUNDED_REPLAY=0 MOE_BACKEND=auto ./scripts/start-opt.sh   # 消融
```

## 压测与回归

压测端只需要三个第三方包：`pip install aiohttp requests pillow`。

```bash
cd bench
python bench_suite.py --tag mine --list          # 看全部场景
python bench_suite.py --tag mine                 # 跑
python bench_compare.py baseline mine --md       # 与另一次结果对比（含噪声阈值）
python vision_check.py                           # 识图回归
```

`bench/corpus/` 内是**合成语料**（程序生成的代码风格文本），用于固定 30K/300K/500K 输入长度。

## 已知代价与边界

- **bounded replay 是近似算法**：与完整 prefill 数值不等价，RL rollout（要求 prompt logprob 与训练端一致）应关闭；关闭后预填充速率下降 46–51%。
- **依赖非社区主线分支**：`dsv41-flash-h20` 只有 10 个提交且面向 H20，升级基线需要移植提交并重打 vision 补丁。
- **解码 tok/s 受投机接受长度影响**：某些内容会「每步耗时减半但接受长度也降」，判断解码优化应看每步耗时而非 tok/s。
- **启动仍需 5–7 分钟**：475 GB 权重冷加载是下限。

## 归因：哪些是别人的，哪些是本仓库的

- **来自开源分支（约 80%）**：CP8、NVLink 对称内存上的 CP 通信、Humming W4A8、FP8 GEMM、稀疏 prefill + 晚层跳过、prefill graph、JIT 清单启动。这些是蚂蚁团队的工作，本仓库做的是选型、适配和验证。
- **本仓库增量（约 20%，但决定可用性）**：
  1. vision + CP 补丁——上游认为二者互斥，实际可以同时拿到识图和 6 倍加速；
  2. 修复 rustup 在线下载工具链导致的启动永久挂起；
  3. 修复 CP 模式下 NUMA 绑核静默失效（进程名变了，匹配数为 0）；
  4. 参数配方与消融验证（chunk 调大无收益、graph 不可用、replay 必须开）；
  5. 可复跑的压测与识图回归体系，让「优化」可被证伪。

## License

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。补丁修改的是 Apache-2.0 许可的 SGLang 源码，已在补丁内标注修改点。
