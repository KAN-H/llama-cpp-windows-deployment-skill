---
name: llama-cpp-windows-deployment
description: 'llama.cpp Windows 平台多模型部署与优化。涵盖：Router Mode 多模型管理、Gemma 4/Qwen/Phi-4 模型部署、QAT–MTP 推理优化、RTX 5060 Ti 16GB 显存调优、CPU 内存受限场景适配、预编译包快速部署、Preset 差异化配置、WSL2 通路配置、Agent/API 接入。Use when: 部署 llama.cpp 服务、配置多模型路由、优化推理性能、排查显存/内存溢出、配置 MTP 投机解码、迁移 Qwen/Gemma 模型。English: deploying llama.cpp on Windows, configuring Router Mode, optimizing GPU/CPU inference, troubleshooting OOM, setting up MTP speculative decoding, migrating between Gemma 4 and Qwen models.'
argument-hint: '描述你的部署场景：硬件配置、目标模型、显存大小、是否使用 Router Mode / MTP'
user-invocable: true
---

# llama.cpp Windows 多模型部署与优化集成技能

> **版本**: v2.0 | **基准硬件**: RTX 5060 Ti 16GB + Intel U7 270K / CPU-only | **平台**: Windows 10/11 + WSL2 | **llama.cpp 版本**: b10056 – b10066+

## 一、When to Use（触发词）

| 场景 | 触发词 |
|------|--------|
| 多模型路由部署 | "Router Mode", "多模型管理", "单端口多模型" |
| Gemma 4 部署 | "Gemma-4", "QAT", "12B", "26B-A4B", "mmproj" |
| Qwen 系列迁移 | "Qwen3", "Qwen3.6", "30B-A3B", "35B-A3B-MTP" |
| Phi-4 CPU 推理 | "Phi-4-mini", "CPU推理", "9GB内存", "纯CPU" |
| MTP 投机解码 | "MTP", "draft-mtp", "spec-draft", "投机解码", "draft acceptance" |
| 显存优化 | "16GB显存", "显存溢出", "OOM", "KV Cache量化" |
| 环境搭建 | "预编译包", "Blackwell", "sm_120", "首次部署" |
| WSL2 连接 | "WSL2", "宿主机访问", "NAT" |
| Agent 对接 | "Hermes", "Claude Code", "Codex CLI", "LangChain" |

## 二、Prerequisites（前置条件）

### 2.1 硬件要求
| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | RTX 5060 Ti 8GB | RTX 5060 Ti 16GB |
| 驱动 | NVIDIA ≥ 610.47 | ≥ 610.62 |
| 内存 | 9GB（CPU-only） | 32GB+ |
| CPU | 8 核 | Intel U7 270K（20核） |
| OS | Windows 10 64-bit | Windows 11 24H2 |

### 2.2 软件要求
- `llama.cpp` 预编译包（`llama-b9811+-bin-win-cuda-13.3-x64.zip`，Blackwell sm_120 原生支持）
- 模型文件：GGUF 格式（标准 / QAT-UD / MTP）
- NVIDIA 驱动正常，`nvidia-smi` 可运行（GPU 场景）
- PowerShell + CMD（BAT 脚本兼容）

### 2.3 目录规范
```
<models-dir>\
├─ chat\                          # Router Mode 扫描的对话模型目录
│  ├─ gemma-4-12b-it-Q5_K_M\     # 无 preset.json → 继承全局参数
│  │  └── model.gguf
│  ├─ gemma-4-12B-it-qat-UD-Q4_K_XL\
│  │  ├── model.gguf
│  │  └── mmproj-F16.gguf        # 多模态投影文件
│  ├─ Qwen3.5-2B-Q4_K_M\
│  │  ├── model.gguf
│  │  └── preset.json            # ctx_size: 32768
│  └── Qwen3.6-35B-A3B-MTP-GGUF\
│     ├── model.gguf
│     └── preset.json
├─ embedding\                     # Embedding 模型独立目录
└─ gemma4_mtp\                    # MTP Draft 模型集中存放
   ├── mtp-gemma-4-12b-it-Q8_0.gguf
   └── mtp-gemma-4-26B-A4B-it-Q8_0.gguf
```

## 三、Core Concepts（核心概念）

| 概念 | 说明 |
|------|------|
| **Router Mode** | 单端口多模型动态加载，请求时指定 `model` 参数自动加载对应 GGUF |
| **Preset (preset.json)** | 模型目录下的独立配置，覆盖 Router 全局参数。**优先级：preset.json > 启动脚本全局参数** |
| **MTP (Multi-Token Prediction)** | 投机解码技术，每步预测多个 token。Gemma 4 用外挂 draft 模型；Qwen3.6 用内置 MTP heads |
| **QAT (Quantization-Aware Training)** | 训练时感知量化的模型，对 KV Cache 精度敏感，推荐 `q8_0/q8_0` |
| **UD (Uniform Decomposition)** | 均匀分解量化，在 XL 级别保留更高比特给敏感层 |
| **KV Cache** | 上下文 Key-Value 缓存，是显存/内存占用的主要变量。公式：`≈ 2 × ctx × layers × hidden_dim × precision_bytes` |
| **Blackwell (sm_120)** | RTX 50 系架构，需 CUDA 13.3+ 驱动 ≥ 610.47 |
| **子命令架构** | 新版 llama.cpp 用 `llama <子命令>` 结构，`llama-server` / `llama-cli` / `llama-embed` |

## 四、Deployment Workflow（部署工作流）

### Step 1：环境校验与版本指纹

**每次换 build 或换模型族前必做**，防止参数 silent fail：

```powershell
# PowerShell（llama.cpp 目录下）
.\llama-server.exe --help | Select-String "spec-draft-n-max"
.\llama-server.exe --help | Select-String "spec-type"
.\llama-server.exe --help | Select-String "draft-"
```

| 指纹特征 | 含义 | 后续操作 |
|---------|------|---------|
| `--spec-draft-n-max` + `draft-mtp` 存在 | b10056 标准版，用 `--spec-*` 命名 | 按本技能 MTP 参数 |
| `--draft-model` / `--draft-mtp-n` 出现 | build 已迁移到 `--draft-*` 命名 | 切换 MTP 参数命名 |
| `--spec-draft-buffer` 不存在 | b10056 已摘除，正常 | 不要写此参数 |
| 模型名含 `MTP` 后缀 | 内置 MTP heads（Qwen3.6 系） | **不需要** `--model-draft` |

### Step 2：选择部署模式

#### 模式 A：Router Mode（多模型热切换）⭐ 推荐

```bat
@echo off
chcp 65001 >nul
title llama.cpp Router - RTX 5060 Ti
cd /d <llama.cpp-dir>

llama-server.exe ^
  --models-dir <models-dir>\chat ^
  --host 0.0.0.0 ^
  --port 8080 ^
  -fa on ^
  -c 8192 ^           :: 全局默认，保护 12B+ 模型
  -np 1 ^
  -t 16 ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  --metrics ^
  --timeout 600

pause
```

**关键规则**：
- 不要在 Router 全局参数中设置 `-ngl` / `-t` 等硬件参数→在单个模型 `preset.json` 中定义
- 小模型（≤3B）通过 `preset.json` 覆盖 `ctx_size` 释放长文本潜力
- 大模型（≥12B）不设 `preset.json`→自动继承全局 `-c 8192` 保显存

#### Preset 模板（放入模型目录）

**小模型（0.8B–3B）长文本配置**：`<models-dir>\chat\<model-dir>\preset.json`
```json
{
    "ctx_size": 32768,
    "n_gpu_layers": 99,
    "n_threads": 16,
    "flash_attn": true
}
```

**大模型（≥12B）无需创建 preset.json**，继承全局参数。

#### 模式 B：单模型实例（专用端口）

适用于需要独占 GPU 资源的高负载场景，每个模型开独立端口。

### Step 3：模型族专项配置

#### 3A：Gemma 4 系列（含 QAT + MTP）

**硬件匹配矩阵（RTX 5060 Ti 16GB）**

| 模型 | GGUF 类型 | 权重 VRAM | 推荐 ctx | MTP 可行性 | 备注 |
|------|----------|----------|---------|-----------|------|
| 12B Q5_K_M | 标准 | ~8.5 GB | 64K ✅ | 32K ✅ | 日常首选 |
| 12B UD-Q8_K_XL | UD imatrix | ~13 GB | 64K ✅ | ❌ 16GB 扛不住 | 高质量基线 |
| 12B qat-UD-Q4_K_XL | QAT+UD | ~6.7 GB | 128K ✅ | ❌ arch 不对齐 | 多模态 + mmproj |
| 26B-A4B qat-UD-Q4_K_XL | MoE A4B | ~10.5 GB | 32K | 需专用 A4B draft | ModelScope 专页 |

**核心参数（所有 Gemma 4 共用）**：
```bat
-ngl 99 -fa on -np 1 -t 10 --batch-size 1024
--cache-type-k q8_0 --cache-type-v q8_0  :: QAT 对 KV 精度敏感，不要降 q4
--no-mmap --host 0.0.0.0
```

**MTP 专有参数（Gemma 4 外挂 draft）**：
```bat
--model-draft <path_to_mtp_draft.gguf>
--spec-type draft-mtp
--spec-draft-n-max 2-3    :: 5060Ti 保守给 2-3，Unsloth 官方给 4
--gpu-layers-draft 60     :: Q5 主模型时给 60，Q8 时给 50
```

> ⚠️ **QAT-UD 与 MTP 不推荐同开**：QAT-UD 是多模态主（带 mmproj），MTP draft 是纯文本 arch，b10056 会因 arch 不对齐 silent skip。QAT-UD 走裸跑 + `--mmproj`。

**26B-A4B MoE 特殊处理**：
- `-ngld` 必须保守（40-50），防止主模型 OOM
- MTP draft 必须用 **A4B 专用** GGUF（不可复用 12B draft）
- Context 限制在 32K（拉 64K 需降 KV 到 q4_0）

#### 3B：Qwen 系列迁移

**三档路线速查**：

| 模型 | MTP 方式 | 参数形态 |
|------|---------|---------|
| Qwen3-30B-A3B | 无（可外挂小 draft） | `--spec-type draft` + `--model-draft` |
| Qwen3.6-35B-A3B-MTP | 内置 heads | `--spec-type draft-mtp`，**无** `--model-draft` |
| Qwen3.5-2B/0.8B | 无 | 裸跑，通过 preset.json 扩 ctx |

> ⚠️ 拿 30B-A3B 硬开 `draft-mtp` 会报 `model does not support MTP`；拿 35B-A3B-MTP 还写 `--model-draft` 会多占显存。

**采样参数（区别于 Gemma 4 的默认值）**：
```bat
--temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05
--chat-template-kwargs "{\"enable_thinking\":false}"
```

**Qwen3.6-35B-A3B-MTP 脚本要点**：
- **删除** `--model-draft`、`--gpu-layers-draft`
- `--spec-draft-n-max 3`（5060Ti 16GB 推荐，可试 4）
- context 上限 32K（拉 64K 需降 KV 到 q4_0）

#### 3C：Phi-4-mini CPU 推理（9GB 内存受限场景）

```bat
@echo off
chcp 65001 >nul
title Phi-4-mini (CPU - 9GB Safe Mode)
cd /d <llama.cpp-dir>

set "MODEL_DIR=<models-dir>\chat\Phi-4-mini-instruct-Q4_K_M"
set "CTX_SIZE=32768"       :: 从 32K 起测，稳定后试 48K/64K
set "BATCH_SIZE=256"
set "THREADS=16"

llama-server.exe ^
  --model "%MODEL_DIR%\Phi-4-mini-instruct-Q4_K_M.gguf" ^
  --ctx-size %CTX_SIZE% ^
  --batch-size %BATCH_SIZE% ^
  --threads %THREADS% ^
  --cache-type-k q4_0 ^     :: CPU 场景降 KV 精度保内存
  --host 0.0.0.0 ^
  --port 8083 ^
  --cors-origins localhost ^
  --no-mmap ^
  --no-metrics

pause
```

**CPU 优化黄金法则**：
- KV Cache 必须量化（`q4_0` 或 `q2_k`），否则 64K = 12.8GB 超 9GB 内存
- `--batch-size 256`，降低瞬时内存峰值
- 支持 `--no-metrics` 节省少量内存
- 不支持参数：`--memory-limit`、`--max-batch-size`、`--numa`

> ⚠️ **BAT 注释提示**：行末 `::` 注释在 `if() (...)` 括号块内可能导致解析错误，如需在括号块内注释请改用 `REM` 语句。

> 📎 **完整脚本参考**：[`./references/gemma4-menu-scripts.bat`](./references/gemma4-menu-scripts.bat)（Gemma 4 10 选项菜单）、[`./references/qwen-scripts.bat`](./references/qwen-scripts.bat)（Qwen 三档部署脚本）、[`./references/preset-templates.json`](./references/preset-templates.json)（各场景 Preset 模板集合）

### Step 4：安全配置（WSL2 + Agent 对接）

```bat
set "API_KEY=sk-local-001"
set "CORS_ORIGINS=http://localhost:* https://localhost:*"
```

| `--host` 值 | WSL2 连通性 | 说明 |
|-------------|------------|------|
| `127.0.0.1` | ❌ 不通 | WSL2 NAT 下 localhost 隔离 |
| `0.0.0.0` + `--api-key` | ✅ 通 | WSL2 用宿主机 vEthernet IP 访问 |
| `.wslconfig` 开 `networkingMode=mirrored` | ✅ 通 | Win11 22H2+ 方案 |

**Agent 接入示例**：
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://<宿主机IP>:8080/v1",
    api_key="sk-local-001",
    model="Qwen3.5-2B-Q4_K_M",
)
```

## 五、Parameter Dictionary（参数词典）

| 参数 | 推荐值 | 适用场景 | 理由 |
|------|--------|---------|------|
| `-ngl` | 99 | GPU 全卸载 | 16GB 显存全量利用 |
| `-fa` | on | 所有模型 | FlashAttention 降显存 |
| `-np` | 1 | 单卡单用户 | 防多批次显存叠加 |
| `-t` | 10-16 | 全场景 | 物理核数，防 Windows oversubscribe |
| `--batch-size` | 1024 (GPU) / 256 (CPU) | 分场景 | GPU 用大 batch 提吞吐，CPU 用小 batch 保内存 |
| `--cache-type-k/v` | q8_0 (GPU) / q4_0 (CPU) | 分场景 | QAT 模型必须 q8_0；CPU 场景可降 |
| `--no-mmap` | 置尾 | Windows | 大 GGUF 长时运行防偶发卡顿 |
| `--host 0.0.0.0` | 必设 | 服务部署 | WSL2 + 局域网访问 |

## 六、MTP Health Diagnostics（健康诊断）

服务启动后，从 `slot print_timing` 日志提取三维度：

| 指标 | 健康阈值 | 处置 |
|------|---------|------|
| `draft acceptance` (A) | > 0.5 ✅ | 0.3-0.5 ⚠️ 检查 draft 对齐；< 0.3 ❌ 换 draft |
| `mean len` (L) | 接近 `spec-draft-n-max` | 显著低于 n-max → 可试提 n-max |
| `graphs reused` (R) | > 0 ✅ | = 0 → draft 被跳过，查 arch/显存 |

**实测参考**（Q5 + Q8 draft）：`A=0.549 / L=2.10 / n-max=2 / R=283` → ✅ 全线绿。

> 🔧 **自动化诊断**：运行 [`./scripts/detect.ps1`](./scripts/detect.ps1) 一键检测 CUDA 架构、MTP 指纹、驱动版本、模型文件和显存状态。

<details>
<summary>🔀 跨模型族迁移检查清单（点击展开）</summary>

| 检查项 | Gemma 4 (外挂 draft) | Qwen3-30B-A3B (无 MTP) | Qwen3.6-35B-A3B-MTP (内置 heads) |
|--------|---------------------|------------------------|----------------------------------|
| `--model-draft` | ✅ 必写 | 外挂时才写 | ❌ 删除 |
| `--gpu-layers-draft` | ✅ 写（控显存） | 外挂时才写 | ❌ 删除 |
| `--spec-type` | `draft-mtp` | `draft` | `draft-mtp` |
| 采样参数 | temp=1.0 / top_p=0.9 | temp=0.7 / top_p=0.8 / top_k=20 | temp=0.7 / top_p=0.8 / top_k=20 |
| `--chat-template-kwargs` | 不需要 | `{"enable_thinking":false}` | `{"enable_thinking":false}` |
| KV Cache | q8_0/q8_0（QAT 敏感） | q8_0/q8_0 | q8_0/q8_0（拉 64K 可降 q4_0） |
| ctx 上限 (16GB) | 12B=64K / 26B-A4B=32K | 48K–64K | 32K |
| MTP 诊断解读 | 读外挂 draft 的 A/L/R | 读外挂 draft 的 A/L/R | 读内置 heads 产出（R 含义不同） |

</details>

## 七、Troubleshooting（常见故障排查）

| 现象 | 根因 | 解决方案 |
|------|------|----------|
| 脚本闪退无输出 | 不支持参数 / 中文注释 | 移除非支持参数，删除中文符号 |
| `invalid argument: --verbose-prefill` | b10066 不支持 | 删除参数，升级 b10070+ 后可加回 |
| `CORS is set to allow all origins ('*')` | 未设 `--api-key` | 添加 `--api-key <自定义值>` |
| WSL2 无法连接 | `--host 127.0.0.1` | 改为 `--host 0.0.0.0` |
| 小模型 ctx 被限制在 8K | 全局 `-c` 一刀切 | 在小模型目录创建 `preset.json` 覆盖 |
| `unknown command '-m'` | 新版子命令架构 | 改用 `llama-cli -m` 或 `llama-server --model` |
| `sm_89` 而非 `sm_120` | 下载了 CUDA 12.4 版 | 下载 CUDA 13.3 版预编译包 |
| MoE 模型 `draft-mtp` 报错 | 模型未训练 MTP heads | 确认模型是否支持（30B-A3B 不支持）|
| 26B 加载 OOM | `-ngld` 过高 / ctx 过大 | 降 `-ngld` 到 40-50，ctx 限制 32K |
| 26B MTP `acceptance` 极低 | 用了 12B 的 draft | 换 A4B 专用 MTP GGUF |
| Phi-4 64K OOM | KV Cache 超 9GB | 降 KV 到 q4_0，从 32K 逐步测试 |

## 八、Version Upgrade Notes（版本升级说明）

| 目标版本 | 恢复/调整的参数 | 说明 |
|---------|---------------|------|
| b10070+ | `--verbose-prefill` | 验证 MTP Prefill 生效 |
| b10070+ | `--defrag-thresh 0.1` | 优化长对话 KV Cache 碎片 |
| 迁移到 `--draft-*` 命名 | `--draft-model`, `--draft-mtp-n` | Qwen3.6 线已迁，Gemma 4 线未迁 |

## 九、Verification Checklist（验证清单）

部署完成后，Agent 自动校验以下五项：

- [ ] **ctx 数值验证**：日志 `n_ctx_slot = <目标值>`（128K→`131072` / 64K→`65536` / 32K→`32768`）
- [ ] **推理速度**：`tg ≥ 60 t/s`（12B 标准）/ `tg ≥ 85 t/s`（12B QAT 128K）/ `tg ≥ 1500 t/s prompt`（26B）
- [ ] **MTP 健康**：`draft acceptance > 0.5`（如适用）
- [ ] **WSL2 连通**：`curl http://<宿主机IP>:<port>/health` → `{"status":"ok"}`
- [ ] **安全配置**：无 API Key → `401`；跨域仅允许配置来源

---

**Skill End**
Agent 执行完成后应输出《部署验证报告》，包含以上五项指标及模型加载状态。
