# 2026-08-29 Session Experience — Qwen3.8 极限优化 + 更新器工程修复 + WSL/VS Code 对接

> 沉淀自 2026-08-28/29 两次会话：Qwen3.8-27B 部署与极限优化、`update-launchers` 自动同步器工程缺陷修复、Router fit 实测、VS Code customendpoint 接入排障。
> 基准：RTX 5060 Ti 16GB / 48GB RAM / 24 线程 / llama.cpp b10448+（2026-08-16 构建）。

## 一、Qwen3.8-27B 模型档案（GGUF 元数据实测）

| 项 | 值 |
|----|----|
| arch | `qwen35` |
| 层数 | 65（**blk.64 为 MTP/nextn 专用**，其余 64 层为注意力块） |
| KV head / key / value | 4 / 256 / 256 |
| ctx_train | 262144 |
| 内置 MTP | `nextn_predict_layers=1`（**勿传 `--model-draft`**） |
| 注意力 | `full_attention_interval=4`：每 4 层 1 层全注意力，其余滑动窗口（默认 4096） |
| KV 估算 | 64K q8_0≈2.85GB / 128K≈5.28GB（audit 需支持 interval 才能算准） |
| 量化体积 | Q3_K_XL=13.15GB / IQ3_S=12.04GB |

官方采样（Unsloth）：thinking `temp 1.0/top-p 0.95/top-k 20/min-p 0.0/presence 0.0/repeat 1.0`；non-thinking `temp 0.7/top-p 0.80/top-k 20/presence 1.5`。`reasoning_effort` 经 `--chat-template-kwargs` 传递。

## 二、16GB 部署速度矩阵（实测）

| 配置 | 上下文 | tg t/s | 备注 |
|------|--------|--------|------|
| Q3_K_XL 64K fit | 64K | 22.0 | 全速稳定 |
| IQ3_S 64K fit | 64K | 27.5–28.7 | 更小权重更快 |
| Q3_K_XL 128K fit | 128K | 12.6 | -43% |
| IQ3_S 128K fit | 128K | 16.0 | -41% |
| IQ3_S 128K `-ngl 56` | 128K | 16.9 | |
| IQ3_S 128K `-ngl 58` + `--batch-size 512` | 128K | **18.8** | 最优 |

**128K 提速方法论**：fit 在 128K 下偏保守（留余量导致过多层在 CPU）。手工 `-ngl` 扫描可显著提升：IQ3_S 128K 从 fit 16.0 → ngl58+batch512 **18.8（+17.5%）**。但 ngl 过高使显存打满（<200MB 余量）触发 CUDA graph 回退反而降速（ngl60=15.7）。**流程**：fit 基线 → `-ngl` 递减试探（如 60→58→56）→ 记录每档 tg + `nvidia-smi` 余量 → 取**余量 ≥400MB 的最快档**。

## 三、长会话稳定性验证方法（沉淀为 SOP）

1. **多轮累积会话**：10 轮累积 messages，逐轮记录 `predicted_per_second` / `cached_tokens` / `finish_reason`。验收：tg 波动 <5%、缓存递增（前缀复用生效）。
2. **上下文跑满 + 埋针检索（needle-in-haystack）**：自编长文分档（8K/16K/32K/56K），在 0.25/0.5/0.75 位置埋 3 条独特事实，每档末尾提问。验收：各档 ≥2/3 检索正确。实测两模型 16K 以上全部 3/3，56K 时仍正确且 tg 仅略降。
3. **跑满后连续 5 轮**：45K 上下文后继续 5 轮混合提问。实测 Q3_K_XL 17.7→18.4、IQ3_S 恒 21.4-21.5（零降速），前缀缓存命中 ~45K。
4. **注意**：思考型模型抽验时 `max_tokens` 必须 ≥300，否则思考链先耗尽 token 导致 `content` 为空/截断（曾误判为模型问题）。

## 四、update-launchers 自动同步器工程缺陷修复（3 连）

1. **`KeyError: 't'`（第二次同步必崩）**：`make_auto` 生成的 group 缺 `"t":"group"`；`build_varmap`/`render_launcher`/`main` 三处 `it["t"]` 直接取值 → 第二次同步（region 已含无 t 的组）必然崩溃，`--audit` 也在对应启动器前中断。**修复**：生成组补 `"t":"group"` + 三处改 `.get("t")` 防御。
2. **孤立 LF（混合换行）**：`layer_seg`/`extra` 等多行参数串内嵌 `\n`，与 CRLF join 混排 → body 含孤立 LF（bat 字节级污染，cmd 解析隐患）。**修复**：构造 body 时 `parts` 内每个元素 `s.split("\n")` 扁平化后再 `"\r\n".join`。
3. **字符串 replace 注入静默失败**：按"行顺序假设"的 `replace` 注入参数，当实际行序不同时静默不生效（曾只注入 `--metrics`，`--reasoning-budget`/`--reasoning-format`/`--ubatch-size` 全部漏掉，且无报错）。**修复**：用完整相邻行块做 oldString，且注入后**断言统计**（如参数出现次数）再落盘。

其他工程要点：
- `render_launcher` 增加**重复 goto 断言**（重复标签会让 `goto` 命中第一个旧块，新块永不执行）。
- `QUANT_RE` 需覆盖新量化名（补 `IQ\d_S` 后 IQ3_S 才显示量化名而非 `?`）。
- mmproj dtype（F16/F32/BF16）编码在文件名：新增 `--fix-mmproj` 自动把静态引用同步到磁盘最新文件；自动条目按 mtime 最新优先。
- audit 的 KV 估算需支持 `full_attention_interval`（否则 qwen35 按全注意力高估 3 倍：9.27G vs 实测 2.85G）。
- 新模型一键加入：放入模型目录 → `--check`（自动识别 [NEW] + 套 profile）→ `--yes`。**无需手改注册表**，只有自定义参数才改 `launcher-models.json`。

## 五、Router fit 多模型限制（实测）

- `--models-preset` + `--fit on`：单模型加载正常（Q3_K_XL 经 Router 8s、tg=26.8、fit 满载 15.9GB）。
- **多 27B 并存**：先加载者 fit 独占显存（15.9GB），后加载者只剩 ~117MB → fit 被迫全 CPU → **4.5 t/s**。属 16GB 物理限制，非配置 bug。
- **策略**：Router 同时只驻留一个 27B + 小模型；不同 27B 分端口跑；`--models-max 3` 仅作上限。fit 化（删全局 `-ngl 99`）比硬编码安全（不 OOM 崩溃，只是后加载者降速）。

## 六、WSL / VS Code 对接 — ECONNREFUSED 排查 SOP

**现象**：VS Code customendpoint 调用报 `connect ECONNREFUSED 127.0.0.1:8084`。

**双根因**：
1. **端口无服务**：测试/清理后 llama-server 未启动。`netstat -ano | findstr :8084` 应为空则先启动启动器。
2. **跨主机 localhost 隔离**：错误堆栈含 `/home/<user>/.vscode-server/` → VS Code 跑在 **WSL 远程**，`localhost` 解析到 WSL 内部；WSL2 NAT 与 Windows 宿主隔离。

**修复**：
- URL 改宿主机 IP：WSL 内 `cat /etc/resolv.conf | grep nameserver` 或 `ip route | grep default`，如 `http://<nameserver-ip>:8084/v1`；
- 或 `.wslconfig` 开 `networkingMode=mirrored`（Win11 22H2+）后 `localhost` 共享；
- 服务端必须 `--host 0.0.0.0` + `--api-key`。

**VS Code customendpoint 配置要点**：
- `maxInputTokens` ≤ 服务端 `-c`（64K→约 60000；128K→128000）；
- `settings` 键名须与模型 `id`/`name` 一致；
- `supportsReasoningEffort` 勿含 llama-server 不支持的 `"none"`（建议 low/medium/high）；
- `apiKey` 与启动器 `--api-key` 一致（否则连接通后遇 401）。

## 七、参数名与转义坑速查

| 项 | 坑 | 正确 |
|----|----|------|
| 重复惩罚 | `--repetition-penalty`（报 invalid argument） | `--repeat-penalty` |
| mmap | `--no-mmap`（DEPRECATED 告警） | `--load-mode mmap` |
| KV 缓存复用 | `--cache-reuse`（`cache_reuse is not supported by this context` 自动禁用） | 不加 |
| 碎片整理 | `--defrag-thresh`/`--defrag-thold`（DEPRECATED） | 不加（内建管理） |
| reasoning_effort（cmd/bat） | `"{\"reasoning_effort\":\"medium\"}"` 在 cmd 下正确 | 保持（CommandLineToArgvW 解析） |
| reasoning_effort（PowerShell） | 双引号+反斜杠被拆散传成 `{\` 报 JSON parse | 单引号 `'{"reasoning_effort":"medium"}'` |
| 思考型抽验 | `max_tokens` 过小 → content 空 | ≥300 |
